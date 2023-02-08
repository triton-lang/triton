#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Alias.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <deque>

namespace mlir {

void MembarAnalysis::run() {
  auto *operation = allocation->getOperation();
  OpBuilder builder(operation);
  resolve(operation, &builder);
}

void MembarAnalysis::resolve(Operation *operation, OpBuilder *builder) {
  // Initialize the blockList and push the entry block
  std::deque<Block *> blockList;
  blockList.emplace_back(&operation->getRegion(0).front());

  // A fixed point algorithm
  while (!blockList.empty()) {
    auto *block = blockList.front();
    blockList.pop_front();
    // Make a copy of the input blockInfo but not update
    auto inputBlockInfo = inputBlockInfoMap.lookup(block);
    SmallVector<Block *> successors;
    for (auto &op : block->getOperations()) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        collect(&op, successors);
      else
        update(&op, &inputBlockInfo, builder);
    }
    // Get the reference because we want to update if it changed
    if (inputBlockInfo != outputBlockInfoMap[block].join(inputBlockInfo)) {
      // If the inputBlockInfo is different from the outputBlockInfo, we
      // need to process the successors
      blockList.emplace_back(block);
    } else {
      // Otherwise, we can skip the successors
      continue;
    }
    // Update the successors
    for (auto *successor : successors) {
      inputBlockInfoMap[successor].join(inputBlockInfo);
      blockList.emplace_back(successor);
    }
  }
}

void MembarAnalysis::collect(Operation *op, SmallVector<Block *> &successors) {
  // If this operation has no successors, we treat it as an exiting terminator.
  if (op->getNumSuccessors() == 0) {
    Region *parentRegion = op->getParentRegion();
    Operation *parentOp = parentRegion->getParentOp();

    // Check to see if the parent tracks region control flow.
    auto regionInterface = dyn_cast<RegionBranchOpInterface>(parentOp);
    if (!regionInterface)
      return;

    // Query the set of successors of the current region
    SmallVector<RegionSuccessor, 1> regionSuccessors;
    regionInterface.getSuccessorRegions(parentRegion->getRegionNumber(),
                                        regionSuccessors);
    if (regionSuccessors.empty())
      return;

    for (auto &it : regionSuccessors) {
      // If the successor is a region, add its entry block to the queue.
      // If the successor is the parent operation, add its entry block to the
      // queue if it is a return-like operation.
      auto *region = it.getSuccessor();
      if (region) {
        successors.emplace_back(&region->front());
      } else if (!region && mlir::isRegionReturnLike(op)) {
        successors.emplace_back(&parentOp->getRegion(0).front());
      }
    }
  }

  // If this operation has successors, add them to the queue.
  Block *block = op->getBlock();
  if (auto branchInterface = dyn_cast<BranchOpInterface>(op)) {
    Block *parentBlock = branchInterface->getBlock();
    for (Block *successor : parentBlock->getSuccessors()) {
      successors.push_back(successor);
    }
  }

  llvm_unreachable("unhandled terminator");
}

void MembarAnalysis::update(Operation *op, BlockInfo *blockInfo,
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
    blockInfo->sync();
    return;
  }

  if (isa<triton::gpu::AsyncWaitOp>(op) &&
      !isa<gpu::BarrierOp>(op->getNextNode())) {
    // If the current op is an async wait and the next op is not a barrier we
    // insert a barrier op and sync
    blockInfo->sync();
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPointAfter(op);
    builder->create<gpu::BarrierOp>(op->getLoc());
    blockInfo->sync();
    return;
  }

  BlockInfo curBlockInfo;
  for (Value value : op->getOperands()) {
    for (auto bufferId : allocation->getBufferIds(value)) {
      if (bufferId != Allocation::InvalidBufferId) {
        if (isa<triton::gpu::InsertSliceAsyncOp>(op) ||
            isa<tensor::InsertSliceOp>(op)) {
          // FIXME(Keren): insert_slice and insert_slice_async are always
          // alias for now
          curBlockInfo.syncWriteBuffers.insert(bufferId);
        } else {
          // ConvertLayoutOp: shared memory -> registers
          curBlockInfo.syncReadBuffers.insert(bufferId);
        }
      }
    }
  }
  for (Value value : op->getResults()) {
    // ConvertLayoutOp: registers -> shared memory
    auto bufferId = allocation->getBufferId(value);
    if (bufferId != Allocation::InvalidBufferId) {
      curBlockInfo.syncWriteBuffers.insert(bufferId);
    }
  }
  // Scratch buffer is considered as both shared memory write & read
  auto bufferId = allocation->getBufferId(op);
  if (bufferId != Allocation::InvalidBufferId) {
    curBlockInfo.syncWriteBuffers.insert(bufferId);
    curBlockInfo.syncReadBuffers.insert(bufferId);
  }

  if (blockInfo->isIntersected(curBlockInfo, allocation)) {
    OpBuilder::InsertionGuard g(*builder);
    builder->setInsertionPoint(op);
    builder->create<gpu::BarrierOp>(op->getLoc());
    blockInfo->sync();
  }
  // Update the region info, even if barrier is inserted, we have to maintain
  // the current op's read/write buffers.
  blockInfo->join(curBlockInfo);
}

} // namespace mlir
