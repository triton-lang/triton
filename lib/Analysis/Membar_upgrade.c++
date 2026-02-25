
/**
 * @file Membar_upgrade.c++
 * @brief Memory barrier analysis and synchronization for Triton GPU backend.
 * @author Upgraded
 * @date 2026
 */

#include "../lib/Conversion/TritonGPUToLLVM/Utility.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton/Analysis/Alias.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PTXAsmFormat.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"
#include <deque>
#include <unordered_map>

namespace mlir {

/**
 * @brief Run the memory barrier analysis for a function.
 * @param funcBlockInfoMap Map of function block info.
 */
void MembarAnalysis::run(FuncBlockInfoMapT &funcBlockInfoMap) {
  auto funcOp = dyn_cast<FunctionOpInterface>(allocation->getOperation());
  if (!funcOp)
    return; // Handle cases where the cast fails
  OpBuilder builder(funcOp.getContext());
  resolve(funcOp, &funcBlockInfoMap, &builder);
}

/**
 * @brief Resolve memory barriers and propagate block info through the function
 * CFG.
 * @param funcOp The function operation interface.
 * @param funcBlockInfoMap Pointer to the function block info map.
 * @param builder Pointer to the OpBuilder.
 */
void MembarAnalysis::resolve(FunctionOpInterface funcOp,
                             FuncBlockInfoMapT *funcBlockInfoMap,
                             OpBuilder *builder) {
  std::unordered_map<Block *, BlockInfo> inputBlockInfoMap, outputBlockInfoMap;
  std::deque<Block *> blockList;

  funcOp.walk<WalkOrder::PreOrder>([&](Block *block) {
    if (std::any_of(block->begin(), block->end(), [](Operation &op) {
          return op.getDialect()->getNamespace() == "scf";
        })) {
      llvm::report_fatal_error(
          "SCF dialect is not supported in Membar. Please lower it "
          "to CF dialect first.");
    }
    if (block->isEntryBlock())
      blockList.push_back(block);
  });

  while (!blockList.empty()) {
    auto *block = blockList.front();
    blockList.pop_front();

    auto inputBlockInfo = inputBlockInfoMap[block];
    SmallVector<Block *> successors;

    for (auto &op : block->getOperations()) {
      if (op.hasTrait<OpTrait::IsTerminator>()) {
        visitTerminator(&op, successors);
      } else {
        update(&op, &inputBlockInfo, funcBlockInfoMap, builder);
      }
    }

    auto &outputBlockInfo = outputBlockInfoMap[block];
    if (outputBlockInfo == inputBlockInfo) {
      continue; // Skip if input and output info haven't changed
    }

    outputBlockInfo.join(inputBlockInfo);

    for (auto *successor : successors) {
      inputBlockInfoMap[successor].join(outputBlockInfo);
      blockList.push_back(successor);
    }
  }

  auto &funcBlockInfo = (*funcBlockInfoMap)[funcOp];
  funcOp.walk<WalkOrder::PreOrder>([&](Block *block) {
    block->walk([&](triton::ReturnOp returnOp) {
      funcBlockInfo.join(outputBlockInfoMap[block]);
    });
  });
}

/**
 * @brief Visit a terminator operation and collect successor blocks.
 * @param op The operation pointer.
 * @param successors Reference to a vector of successor blocks.
 */
void MembarAnalysis::visitTerminator(Operation *op,
                                     SmallVector<Block *> &successors) {
  if (auto branchInterface = dyn_cast<BranchOpInterface>(op)) {
    auto &parentBlock = branchInterface->getBlock();
    successors.append(parentBlock->succ_begin(), parentBlock->succ_end());
    return;
  }

  if (isa<triton::ReduceReturnOp, triton::ScanReturnOp, triton::ReturnOp>(op)) {
    return;
  }

  llvm_unreachable("Unknown terminator encountered in Membar analysis");
}

/**
 * @brief Update block info for a given operation, handling barriers and
 * dependencies.
 * @param op The operation pointer.
 * @param blockInfo Pointer to the current block info.
 * @param funcBlockInfoMap Pointer to the function block info map.
 * @param builder Pointer to the OpBuilder.
 */
void MembarAnalysis::update(Operation *op, BlockInfo *blockInfo,
                            FuncBlockInfoMapT *funcBlockInfoMap,
                            OpBuilder *builder) {
  if (isa<triton::gpu::ExtractSliceOp, triton::gpu::AllocTensorOp,
          triton::TransOp>(op)) {
    return; // No synchronization required
  }

  if (isBarrierOp(op)) {
    blockInfo->sync();
    return;
  }

  if (auto asyncWaitOp = dyn_cast<triton::gpu::AsyncWaitOp>(op)) {
    handleAsyncWaitOp(asyncWaitOp, blockInfo, builder);
    return;
  }

  BlockInfo curBlockInfo = collectDependencies(op);

  if (blockInfo->isIntersected(curBlockInfo)) {
    insertSyncBarrier(op, builder);
    blockInfo->sync();
  }

  blockInfo->join(curBlockInfo);
}

/**
 * @brief Check if an operation is a barrier operation.
 * @param op The operation pointer.
 * @return True if the operation is a barrier, false otherwise.
 */
bool MembarAnalysis::isBarrierOp(Operation *op) const {
  return isa<gpu::BarrierOp>(op) ||
         (isa<LLVM::InlineAsmOp>(op) &&
          dyn_cast<LLVM::InlineAsmOp>(op).getAsmString().find("bar.sync") !=
              std::string::npos);
}

/**
 * @brief Handle an async wait operation, inserting barriers as needed.
 * @param op The async wait operation.
 * @param blockInfo Pointer to the current block info.
 * @param builder Pointer to the OpBuilder.
 */
void MembarAnalysis::handleAsyncWaitOp(triton::gpu::AsyncWaitOp op,
                                       BlockInfo *blockInfo,
                                       OpBuilder *builder) {
  blockInfo->sync();
  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointAfter(op);

  if (auto optionalAgentId = getWSAgentId(op)) {
    int agentId = *optionalAgentId, roleId = 0;
    if (auto optionalRoleId = getWSRoleId(op)) {
      roleId = *optionalRoleId;
    }
    int barId = agentId + roleId + nameBarrierIdBegin;
    assert(barId < nameBarrierIdEnd);
    barSync(*builder, op, barId, 128);
  } else {
    builder->create<gpu::BarrierOp>(op->getLoc());
  }

  blockInfo->sync();
}

/**
 * @brief Collect dependencies for a given operation.
 * @param op The operation pointer.
 * @return BlockInfo containing dependency information.
 */
BlockInfo MembarAnalysis::collectDependencies(Operation *op) const {
  BlockInfo curBlockInfo;

  if (auto callOp = dyn_cast<triton::CallOp>(op)) {
    if (auto callee = dyn_cast<FunctionOpInterface>(callOp.resolveCallable())) {
      curBlockInfo = funcBlockInfoMap->lookup(callee);
    }
  } else {
    for (auto value : op->getOperands()) {
      updateBlockInfoForOperand(value, op, curBlockInfo);
    }
    for (auto value : op->getResults()) {
      updateBlockInfoForResult(value, op, curBlockInfo);
    }
    if (auto bufferId = allocation->getBufferId(op);
        bufferId != Allocation::InvalidBufferId) {
      curBlockInfo.syncWriteIntervals.insert(
          allocation->getAllocatedInterval(bufferId));
      curBlockInfo.syncReadIntervals.insert(
          allocation->getAllocatedInterval(bufferId));
    }
  }

  return curBlockInfo;
}

/**
 * @brief Update block info for an operand value.
 * @param value The operand value.
 * @param op The operation pointer.
 * @param curBlockInfo Reference to the current block info.
 */
void MembarAnalysis::updateBlockInfoForOperand(Value value, Operation *op,
                                               BlockInfo &curBlockInfo) const {
  for (auto bufferId : allocation->getBufferIds(value)) {
    if (bufferId != Allocation::InvalidBufferId) {
      if (isa<triton::gpu::InsertSliceAsyncOp, tensor::InsertSliceOp>(op)) {
        curBlockInfo.syncWriteIntervals.insert(
            allocation->getAllocatedInterval(bufferId));
      } else {
        curBlockInfo.syncReadIntervals.insert(
            allocation->getAllocatedInterval(bufferId));
      }
    }
  }
}

/**
 * @brief Update block info for a result value.
 * @param value The result value.
 * @param op The operation pointer.
 * @param curBlockInfo Reference to the current block info.
 */
void MembarAnalysis::updateBlockInfoForResult(Value value, Operation *op,
                                              BlockInfo &curBlockInfo) const {
  if (auto bufferId = allocation->getBufferId(value);
      bufferId != Allocation::InvalidBufferId) {
    curBlockInfo.syncWriteIntervals.insert(
        allocation->getAllocatedInterval(bufferId));
  }
}

/**
 * @brief Insert a synchronization barrier at the given operation.
 * @param op The operation pointer.
 * @param builder Pointer to the OpBuilder.
 */
void MembarAnalysis::insertSyncBarrier(Operation *op,
                                       OpBuilder *builder) const {
  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPoint(op);

  if (auto optionalAgentId = getWSAgentId(op)) {
    int agentId = *optionalAgentId, roleId = 0;
    if (auto optionalRoleId = getWSRoleId(op)) {
      roleId = *optionalRoleId;
    }
    int barId = agentId + roleId + nameBarrierIdBegin;
    assert(barId < nameBarrierIdEnd);
    barSync(*builder, op, barId, 128);
  } else {
    builder->create<gpu::BarrierOp>(op->getLoc());
  }
}

} // namespace mlir
