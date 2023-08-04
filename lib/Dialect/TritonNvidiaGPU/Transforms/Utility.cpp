/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include <fstream>

namespace mlir {

namespace {

// Suppose the kernel has following structure:
// ```
// scf.for(...) {
//   compute_0(i)
//   barrier(...)
//   compute_1(i)
// }
// ```
// Due to the barrier between compute_0(i) and compute_1(i), we
// can not pre-compute compute_0(i+1) before compute_1(i) semantically.
// In some case, it may be still functionally correct to pre-compute
// compute_0(i+1) while it's very hard to prove it at compile time.
//
// Here we use a simple strategy: skip auto wrap specialize those kernels that
// use global barriers.
//
// Another remaining question is how to detect barrier in a triton program.
// There is not a barrier op in triton yet. It's usually implemented using
// atomic_* ops. Hence we simply detect if there are some atomc_* ops. It may
// miss some auto-WS opportunities and we leave it for the future to improve it.
bool hasUnsafeBarrier(triton::FuncOp funcOp) {
  return funcOp
      ->walk([](Operation *op) {
        if (isa<triton::AtomicRMWOp, triton::AtomicCASOp>(op))
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

// Assigns `dependentSet` and returns ok if the analysis is successful.
// We do not support dependency analysis across load/store, thus a failure will
// be returned if encountering such cases.
LogicalResult getDependentPointers(Value ptr, DenseSet<Value> &dependentSet,
                                   DenseSet<Value> &processedSet) {
  // early return if processed
  if (!processedSet.insert(ptr).second)
    return success();

  if (auto blockArg = ptr.dyn_cast<BlockArgument>()) {
    if (!blockArg.getOwner()->isEntryBlock())
      return failure();
    auto parentOp = blockArg.getOwner()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      if (blockArg.getArgNumber() >= forOp.getNumInductionVars()) {
        if (failed(getDependentPointers(
                forOp.getOpOperandForRegionIterArg(blockArg).get(),
                dependentSet, processedSet)))
          return failure();

        unsigned operandIdx =
            blockArg.getArgNumber() - forOp.getNumInductionVars();
        return getDependentPointers(
            forOp.getBody()->getTerminator()->getOperand(operandIdx),
            dependentSet, processedSet);
      }
    } else if (auto funcOp = dyn_cast<triton::FuncOp>(parentOp)) {
      dependentSet.insert(ptr);
      return success();
    }
    // unknown ops, return failure for correctness.
    return failure();
  }

  auto definingOp = ptr.getDefiningOp();
  assert(definingOp);
  if (auto makeTensorPtrOp = ptr.getDefiningOp<triton::MakeTensorPtrOp>()) {
    return getDependentPointers(makeTensorPtrOp.getBase(), dependentSet,
                                processedSet);
  } else if (auto advanceOp = ptr.getDefiningOp<triton::AdvanceOp>()) {
    return getDependentPointers(advanceOp.getPtr(), dependentSet, processedSet);
  } else if (auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>()) {
    return getDependentPointers(addPtrOp.getPtr(), dependentSet, processedSet);
  } else if (auto loadOp = ptr.getDefiningOp<triton::AddPtrOp>()) {
    // not support load dependent ptr
    return failure();
  } else if (auto forOp = ptr.getDefiningOp<scf::ForOp>()) {
    unsigned idx = ptr.cast<OpResult>().getResultNumber();
    return getDependentPointers(
        forOp.getBody()->getTerminator()->getOperand(idx), dependentSet,
        processedSet);
  } else if (auto ifOp = ptr.getDefiningOp<scf::IfOp>()) {
    unsigned idx = ptr.cast<OpResult>().getResultNumber();
    if (ifOp.elseBlock() &&
        failed(getDependentPointers(ifOp.elseYield()->getOperand(idx),
                                    dependentSet, processedSet)))
      return failure();
    return getDependentPointers(ifOp.thenYield()->getOperand(idx), dependentSet,
                                processedSet);
  } else if (!definingOp->getNumRegions()) {
    for (Value operand : definingOp->getOperands())
      if (failed(getDependentPointers(operand, dependentSet, processedSet)))
        return failure();
    return success();
  }
  // unknown ops, return failure for correctness.
  return failure();
}

// Suppose the kernel has following structure:
// ```
// scf.for(...) {
//   v(i) = load(ptr)
//   new_v(i) = some_compute(v(i), ...)
//   store(new_v(i), ptr)
// }
// ```
//
// There is an implicit dependency between load(i+1) and store(i), which means
// we can not pre-compute load(i+1) before store(i).
//
// To avoid such load after store conflict, we simply disallow mixed load and
// store for the same buffer. It's a conservative strategy and can be relaxed in
// case necessary.
bool hasUnsafeLoadAfterStore(triton::FuncOp funcOp) {
  // TODO: support CFG
  if (funcOp.getBody().getBlocks().size() > 1)
    return true;

  DenseMap<Value, bool> ptrStoreMap;
  DenseMap<Value, bool> ptrLoadMap;
  if (funcOp
          ->walk([&](triton::LoadOp loadOp) {
            DenseSet<Value> dependentSet, processedSet;
            if (failed(getDependentPointers(loadOp.getPtr(), dependentSet,
                                            processedSet)))
              return WalkResult::interrupt();
            for (Value v : dependentSet)
              ptrLoadMap[v] = true;
            return WalkResult::advance();
          })
          .wasInterrupted())
    return false;
  auto result = funcOp->walk([&](Operation *op) {
    if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
      DenseSet<Value> dependentSet, processedSet;
      if (failed(getDependentPointers(storeOp.getPtr(), dependentSet,
                                      processedSet)))
        return WalkResult::interrupt();

      for (Value v : dependentSet)
        ptrStoreMap[v] = true;

      // TODO: relax the restriction in case necessary.
      // If a store is inside a region, e.g. scf.while/for/if, its
      // dependent ptrs are not allowed to be loaded.
      if (op->getParentOp() != funcOp) {
        for (Value v : dependentSet)
          if (ptrLoadMap.find(v) != ptrLoadMap.end())
            return WalkResult::interrupt();
      }
    } else if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      DenseSet<Value> dependentSet, processedSet;
      if (failed(getDependentPointers(loadOp.getPtr(), dependentSet,
                                      processedSet)))
        return WalkResult::interrupt();
      for (Value v : dependentSet)
        if (ptrStoreMap.find(v) != ptrStoreMap.end())
          return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return result.wasInterrupted();
}

} // namespace

//===----------------------------------------------------------------------===//
// Helper functions for async agent
//===----------------------------------------------------------------------===//

SmallVector<AgentId> getAgentIds(Operation *op) {
  SmallVector<AgentId> agentIds;
  if (auto attr = op->getAttrOfType<DenseIntElementsAttr>("async_agent"))
    for (AgentId agentId : attr.getValues<AgentId>())
      agentIds.push_back(agentId);
  return agentIds;
}

bool hasAgentId(Operation *op, AgentId agentId) {
  for (AgentId candidate : getAgentIds(op))
    if (candidate == agentId)
      return true;
  return false;
}

void setAgentIds(Operation *op, ArrayRef<AgentId> agentIds) {
  SmallVector<AgentId> sortedAgentIds(agentIds.begin(), agentIds.end());
  sort(sortedAgentIds);
  auto i32Ty = IntegerType::get(op->getContext(), 32);
  auto size = static_cast<int64_t>(sortedAgentIds.size());
  auto vecTy = VectorType::get(size, i32Ty);
  op->setAttr("async_agent", DenseIntElementsAttr::get(vecTy, sortedAgentIds));
}

SmallVector<AgentId> collectAgentIds(Operation *op) {
  SetVector<AgentId> agentIds;
  op->walk([&](Operation *curOp) {
    for (AgentId agentId : getAgentIds(curOp))
      agentIds.insert(agentId);
  });
  SmallVector<AgentId> res(agentIds.begin(), agentIds.end());
  llvm::sort(res);
  return res;
}

void addAgentIds(Operation *op, ArrayRef<int> agents) {
  auto agentsVec = getAgentIds(op);
  DenseSet<int> agentsSet(agentsVec.begin(), agentsVec.end());
  for (int a : agents) {
    if (!agentsSet.contains(a)) {
      agentsVec.push_back(a);
    }
  }
  if (agentsVec.size() > 0) {
    setAgentIds(op, agentsVec);
  }
}

//===----------------------------------------------------------------------===//
// Implementations for general auto WS
//===----------------------------------------------------------------------===//

bool isWSSupported(ModuleOp mod, int computeCapability) {
  // Early return if the target device is not feasible.
  if (computeCapability / 10 < 9) {
    return false;
  }

  // TODO: support function call.
  triton::FuncOp funcOp;
  if (mod->walk([&](triton::FuncOp op) {
           if (funcOp)
             return WalkResult::interrupt();
           funcOp = op;
           return WalkResult::advance();
         })
          .wasInterrupted() ||
      !funcOp)
    return false;

  // Triton programs with global barrier are much harder to do auto warp
  // specialization. Here we do some conservative checks to skip the bad cases.
  if (hasUnsafeBarrier(funcOp))
    return false;

  // load after store for the same buffer forces an implicit dependency, which
  // may break auto WS. Here we do some conservative checks to skip the bad
  // cases.
  if (hasUnsafeLoadAfterStore(funcOp))
    return false;

  return true;
}

} // namespace mlir
