#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Alias.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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
  operation->walk<WalkOrder::PreOrder>([&](Block *block) {
    for (auto &op : block->getOperations()) {
      // Check if the operation belongs to scf dialect, if so, we need to
      // throw an error
      if (op.getDialect()->getNamespace() == "scf") {
        llvm::report_fatal_error(
            "scf dialect is not supported in membar. Please lower it "
            "to cf dialect first.");
        return;
      }
    }
    if (block->isEntryBlock())
      blockList.emplace_back(block);
  });

  // A fixed point algorithm
  while (!blockList.empty()) {
    auto *block = blockList.front();
    blockList.pop_front();
    // Make a copy of the inputblockInfo but not update
    auto inputBlockInfo = inputBlockInfoMap.lookup(block);
    SmallVector<Block *> successors;
    for (auto &op : block->getOperations()) {
      if (op.hasTrait<OpTrait::IsTerminator>()) {
        visitTerminator(&op, successors);
      } else {
        update(&op, &inputBlockInfo, builder);
      }
    }
    // Get the reference because we want to update if it changed
    if (outputBlockInfoMap.count(block) &&
        inputBlockInfo == outputBlockInfoMap[block]) {
      // If we have seen the block before and the inputBlockInfo is the same as
      // the outputBlockInfo, we skip the successors
      continue;
    }
    // Update the current block
    outputBlockInfoMap[block].join(inputBlockInfo);
    // Update the successors
    for (auto *successor : successors) {
      inputBlockInfoMap[successor].join(outputBlockInfoMap[block]);
      blockList.emplace_back(successor);
    }
  }
}

void MembarAnalysis::visitTerminator(Operation *op,
                                     SmallVector<Block *> &successors) {
  if (auto branchInterface = dyn_cast<BranchOpInterface>(op)) {
    Block *parentBlock = branchInterface->getBlock();
    for (Block *successor : parentBlock->getSuccessors()) {
      successors.push_back(successor);
    }
    return;
  }
  // Otherwise, it could be a return op
  assert(isa<func::ReturnOp>(op) && "Unknown terminator");
}

void MembarAnalysis::update(Operation *op, BlockInfo *blockInfo,
                            OpBuilder *builder) {
  if (isa<triton::gpu::ExtractSliceOp>(op) ||
      isa<triton::gpu::AllocTensorOp>(op) || isa<triton::TransOp>(op)) {
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
