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
  // Initialize the blockList
  std::deque<Block *> blockList;
  operation->walk([&](Block *block) { blockList.push_back(block); });

  // A fixed point algorithm
  while (!blockList.empty()) {
    auto *block = blockList.front();
    blockList.pop_front();
    // Make a copy of the input blockInfo but not update
    auto inputBlockInfo = inputBlockInfoMap.lookup(block);
    SmallVector<Block *, 4> curSucessors;
    for (auto &op : block->getOperations()) {
      if (auto regionBranch = dyn_cast<RegionBranchOpInterface>(op)) {
        SmallVector<RegionSuccessor, 4> successors;
        regionBranch.getSuccessorRegions(llvm::None, successors);
        for (auto &successor : successors) {
          curSucessors.push_back(&successor.getSuccessor()->front());
        }
      } else if (auto branch = dyn_cast<BranchOpInterface>(op)) {
        for (auto *successor : block->getSuccessors()) {
          curSucessors.push_back(successor);
        }
      } else {
        transfer(&op, &inputBlockInfo, builder);
      }
    }
    // Get the reference because we want to update if it changed
    auto &outputBlockInfo = outputBlockInfoMap[block];
    if (inputBlockInfo == outputBlockInfo) {
      // If the inputBlockInfo is the same as the outputBlockInfo, we do not
      // need to update the successors
      continue;
    }
    // Update the successors
    outputBlockInfo.join(inputBlockInfo);
    for (auto *successor : curSucessors) {
      inputBlockInfoMap[successor].join(outputBlockInfo);
      blockList.push_back(successor);
    }
    blockList.push_back(block);
  }
}

void MembarAnalysis::transfer(Operation *op, BlockInfo *blockInfo,
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
          // FIXME(Keren): insert_slice and insert_slice_async are always alias
          // for now
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
