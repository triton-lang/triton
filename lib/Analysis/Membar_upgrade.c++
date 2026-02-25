#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Alias.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "../lib/Conversion/TritonGPUToLLVM/Utility.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "triton/Conversion/TritonGPUToLLVM/PTXAsmFormat.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <deque>
#include <unordered_map>

namespace mlir {

void MembarAnalysis::run(FuncBlockInfoMapT &funcBlockInfoMap) {
    auto funcOp = dyn_cast<FunctionOpInterface>(allocation->getOperation());
    if (!funcOp) return; // Handle cases where the cast fails
    OpBuilder builder(funcOp.getContext());
    resolve(funcOp, &funcBlockInfoMap, &builder);
}

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

void MembarAnalysis::update(Operation *op, BlockInfo *blockInfo,
                            FuncBlockInfoMapT *funcBlockInfoMap,
                            OpBuilder *builder) {
    if (isa<triton::gpu::ExtractSliceOp, triton::gpu::AllocTensorOp, triton::TransOp>(op)) {
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

bool MembarAnalysis::isBarrierOp(Operation *op) const {
    return isa<gpu::BarrierOp>(op) ||
           (isa<LLVM::InlineAsmOp>(op) &&
            dyn_cast<LLVM::InlineAsmOp>(op).getAsmString().find("bar.sync") != std::string::npos);
}

void MembarAnalysis::handleAsyncWaitOp(triton::gpu::AsyncWaitOp op, BlockInfo *blockInfo, OpBuilder *builder) {
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

BlockInfo MembarAnalysis::collectDependencies(Operation *op) const {
    BlockInfo curBlockInfo;

    if (auto callOp = dyn_cast<triton::CallOp>(op)) {
        if (auto callee = dyn_cast<FunctionOpInterface>(callOp.resolveCallable())) {
            curBlockInfo = funcBlockInfoMap->lookup(callee);
        }
    } else {
        for (Value value : op->getOperands()) {
            updateBlockInfoForOperand(value, op, curBlockInfo);
        }
        for (Value value : op->getResults()) {
            updateBlockInfoForResult(value, op, curBlockInfo);
        }
        if (auto bufferId = allocation->getBufferId(op); bufferId != Allocation::InvalidBufferId) {
            curBlockInfo.syncWriteIntervals.insert(allocation->getAllocatedInterval(bufferId));
            curBlockInfo.syncReadIntervals.insert(allocation->getAllocatedInterval(bufferId));
        }
    }

    return curBlockInfo;
}

void MembarAnalysis::updateBlockInfoForOperand(Value value, Operation *op, BlockInfo &curBlockInfo) const {
    for (auto bufferId : allocation->getBufferIds(value)) {
        if (bufferId != Allocation::InvalidBufferId) {
            if (isa<triton::gpu::InsertSliceAsyncOp, tensor::InsertSliceOp>(op)) {
                curBlockInfo.syncWriteIntervals.insert(allocation->getAllocatedInterval(bufferId));
            } else {
                curBlockInfo.syncReadIntervals.insert(allocation->getAllocatedInterval(bufferId));
            }
        }
    }
}

void MembarAnalysis::updateBlockInfoForResult(Value value, Operation *op, BlockInfo &curBlockInfo) const {
    if (auto bufferId = allocation->getBufferId(value); bufferId != Allocation::InvalidBufferId) {
        curBlockInfo.syncWriteIntervals.insert(allocation->getAllocatedInterval(bufferId));
    }
}

void MembarAnalysis::insertSyncBarrier(Operation *op, OpBuilder *builder) const {
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
