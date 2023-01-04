#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Alias.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {

void MembarAnalysis::run() {
  auto *operation = allocation->getOperation();
  RegionInfo regionInfo;
  OpBuilder builder(operation);
  dfsOperation(operation, &regionInfo, &builder);
}

void MembarAnalysis::dfsOperation(Operation *operation,
                                  RegionInfo *parentRegionInfo,
                                  OpBuilder *builder) {
  transfer(operation, parentRegionInfo, builder);
  if (operation->getNumRegions()) {
    // If there's any nested regions, we need to visit them.
    // scf.if and scf.else: two regions
    // scf.if only: two regions
    // scf.for: one region
    RegionInfo curRegionInfo;
    auto traverseRegions = [&]() -> auto{
      for (auto &region : operation->getRegions()) {
        // Copy the parent info as the current info.
        RegionInfo regionInfo = *parentRegionInfo;
        for (auto &block : region.getBlocks()) {
          assert(region.getBlocks().size() == 1 &&
                 "Multiple blocks in a region is not supported");
          for (auto &op : block.getOperations()) {
            // Traverse the nested operation.
            dfsOperation(&op, &regionInfo, builder);
          }
        }
        curRegionInfo.join(regionInfo);
      }
      // Set the parent region info as the union of the nested region info.
      *parentRegionInfo = curRegionInfo;
    };

    traverseRegions();
    if (isa<scf::ForOp>(operation)) {
      // scf.for can have two possible inputs: the init value and the
      // previous iteration's result. Although we've applied alias analysis,
      // there could be unsynced memory accesses on reused memories.
      // For example, consider the following code:
      // %1 = convert_layout %0: blocked -> shared
      // ...
      // gpu.barrier
      // ...
      // %5 = convert_layout %4 : shared -> dot
      // %6 = tt.dot %2, %5
      // scf.yield
      //
      // Though %5 could be released before scf.yield, it may shared the same
      // memory with %1. So we actually have to insert a barrier before %1 to
      // make sure the memory is synced.
      traverseRegions();
    }
  }
}

void MembarAnalysis::transfer(Operation *op, RegionInfo *regionInfo,
                              OpBuilder *builder) {
  if (isa<scf::ForOp>(op) || isa<scf::IfOp>(op) || isa<scf::YieldOp>(op) ||
      isa<tensor::ExtractSliceOp>(op) || isa<triton::gpu::AllocTensorOp>(op)) {
    // Do not insert barriers before control flow operations and
    // alloc/extract/insert
    // alloc is an allocation op without memory write.
    // FIXME(Keren): extract_slice is always alias for now
    return;
  }

  if (isa<gpu::BarrierOp>(op)) {
    // If the current op is a barrier, we sync previous reads and writes
    regionInfo->sync();
    return;
  }

  if (isa<triton::gpu::AsyncWaitOp>(op) &&
      !isa<gpu::BarrierOp>(op->getNextNode())) {
    // If the current op is an async wait and the next op is not a barrier we
    // insert a barrier op and sync
    regionInfo->sync();
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPointAfter(op);
    builder->create<gpu::BarrierOp>(op->getLoc());
    regionInfo->sync();
    return;
  }

  RegionInfo curRegionInfo;
  for (Value value : op->getOperands()) {
    for (auto bufferId : allocation->getBufferIds(value)) {
      if (bufferId != Allocation::InvalidBufferId) {
        if (isa<triton::gpu::InsertSliceAsyncOp>(op) ||
            isa<tensor::InsertSliceOp>(op)) {
          // FIXME(Keren): insert_slice and insert_slice_async are always alias
          // for now
          curRegionInfo.syncWriteBuffers.insert(bufferId);
        } else {
          // ConvertLayoutOp: shared memory -> registers
          curRegionInfo.syncReadBuffers.insert(bufferId);
        }
      }
    }
  }
  for (Value value : op->getResults()) {
    // ConvertLayoutOp: registers -> shared memory
    auto bufferId = allocation->getBufferId(value);
    if (bufferId != Allocation::InvalidBufferId) {
      curRegionInfo.syncWriteBuffers.insert(bufferId);
    }
  }
  // Scratch buffer is considered as both shared memory write & read
  auto bufferId = allocation->getBufferId(op);
  if (bufferId != Allocation::InvalidBufferId) {
    curRegionInfo.syncWriteBuffers.insert(bufferId);
    curRegionInfo.syncReadBuffers.insert(bufferId);
  }

  if (regionInfo->isIntersected(curRegionInfo, allocation)) {
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPoint(op);
    builder->create<gpu::BarrierOp>(op->getLoc());
    regionInfo->sync();
  }
  // Update the region info, even if barrier is inserted, we have to maintain
  // the current op's read/write buffers.
  regionInfo->join(curRegionInfo);
}

} // namespace mlir
