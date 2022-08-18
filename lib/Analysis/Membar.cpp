#include "triton/Analysis/Membar.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Dialect/GPU/GPUDialect.h"

namespace mlir {

void MembarAnalysis::run() {
  auto *operation = allocation->getOperation();
  operation->getContext()->getOrLoadDialect<mlir::gpu::GPUDialect>();
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
  }
}

void MembarAnalysis::transfer(Operation *op, RegionInfo *regionInfo,
                              OpBuilder *builder) {
  if (op->getNumResults() < 1)
    return;

  if (dyn_cast<gpu::BarrierOp>(op)) {
    // If the current op is a barrier, we sync previous reads and writes
    regionInfo->sync();
    return;
  }

  if (dyn_cast<triton::gpu::AsyncWaitOp>(op)) {
    // If the current op is an async wait, we insert a barrier op and sync
    // previous reads and writes.
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPointAfter(op);
    builder->create<gpu::BarrierOp>(op->getLoc());
    regionInfo->sync();
    return;
  }

  auto addBuffer = [&](RegionInfo::BufferIdSetT &bufferSet,
                       Allocation::BufferId bufferId) {
    if (bufferId != Allocation::InvalidBufferId) {
      bufferSet.insert(bufferId);
    }
  };

  RegionInfo curRegionInfo;
  for (Value value : op->getOperands()) {
    // ConvertLayoutOp: shared memory -> registers
    addBuffer(curRegionInfo.syncReadBuffers, allocation->getBufferId(value));
  }
  for (Value value : op->getResults()) {
    // ConvertLayoutOp: registers -> shared memory
    addBuffer(curRegionInfo.syncWriteBuffers, allocation->getBufferId(value));
  }
  // Scratch buffer is considered as a shared memory read
  addBuffer(curRegionInfo.syncReadBuffers, allocation->getBufferId(op));

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
