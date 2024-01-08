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

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

using namespace mlir;
namespace ttng = mlir::triton::nvidia_gpu;

class Heuristics {
  //===--------------------------Label rules--------------------------===//
  //  - Op without agent attr: operations shared by all agents but will NOT
  //    be copied into each agent region
  //  - Op with one agent: exclusive for one agent
  //  - Op with agents: shared by agents and will be copied into each agent
  //  region
  //===---------------------------------------------------------------===//
public:
  Heuristics(MLIRContext *context, ModuleOp mod, const int &computeCapability)
      : context(context), mod(mod), computeCapability(computeCapability),
        builder(OpBuilder(context)) {}
  virtual bool run() = 0;

protected:
  // Set agentId when condition is satisfied
  virtual void
  setAgentId_if(int agentId,
                const std::function<bool(Operation *)> &condition) {
    mod.walk([&](Operation *op) -> void {
      if (condition(op)) {
        setAgentIds(op, {agentId});
      }
    });
  }

  static bool isTritonLoadOp(Operation *op) { return isa<triton::LoadOp>(op); }

  static bool isTritonDotOp(Operation *op) { return isa<triton::DotOp>(op); }

  static bool isTritonStoreOp(Operation *op) {
    return isa<triton::StoreOp>(op);
  }

  /// Because we set some special filter rules in populateAgentRegion,
  /// there may be unlabeled Ops, e.g. YieldOps, some definingOps of ForOps.
  /// or Ops without relations to agentOps
  virtual void populateUnlabledOpsAtLast(ArrayRef<int> allAgents) {
    // Label agents' parentOps
    for (int i : allAgents) {
      DenseSet<Operation *> agentParentOps;
      getAllParentOps(agentParentOps, i);
      for (auto op : agentParentOps) {
        addAgentIds(op, {i});
      }
    }

    // Get unlabeled Ops
    DenseSet<Operation *> unlabeledOps;
    mod.walk([&](Operation *op) -> void {
      if (isa<ModuleOp>(op) || isa<triton::FuncOp>(op) ||
          isa<triton::ReturnOp>(op)) {
        return;
      }
      if (!op->hasAttr("async_agent")) {
        unlabeledOps.insert(op);
      }
    });

    // Label Ops using its parentOp
    for (auto op : unlabeledOps) {
      if (auto parent = op->getParentOp()) {
        if (!isa<triton::FuncOp>(parent)) {
          assert(parent->hasAttr("async_agent"));
          auto agents = getAgentIds(parent);
          setAgentIds(op, agents);
          unlabeledOps.erase(op);
        }
      }
    }

    // Label Ops using dependency
    for (auto op : unlabeledOps) {
      labelByUsers(op, allAgents);
      unlabeledOps.erase(op);
    }
    assert(unlabeledOps.size() == 0);
  }

  // Return all Ops that are marked with target agent
  void getAgentOps(DenseSet<Operation *> &agentOps, int agentId) {
    SmallVector tmpArray{agentId};
    auto agentAttr = builder.getI32VectorAttr(ArrayRef<int>(tmpArray));
    mod.walk([&](Operation *op) -> void {
      if (op->hasAttr("async_agent") &&
          op->getAttr("async_agent") == agentAttr) {
        agentOps.insert(op);
      }
    });
  }

  void getAllParentOps(DenseSet<Operation *> &parentOps, int agentId) {
    DenseSet<Operation *> targetOps;
    getAgentOps(targetOps, agentId);
    for (auto op : targetOps) {
      getAllParentOps(parentOps, op);
    }
  }

  void getAllParentOps(DenseSet<Operation *> &parentOps, Operation *targetOp) {
    auto op = targetOp;
    while (auto parent = op->getParentOp()) {
      if (!isa<ModuleOp>(parent) && !isa<triton::FuncOp>(parent)) {
        parentOps.insert(parent);
        op = parent;
      } else {
        break;
      }
    }
  }

  void labelByUsers(Operation *op, ArrayRef<int> allAgents) {
    for (Value result : op->getResults()) {
      for (Operation *userOp : result.getUsers()) {
        if (!userOp->hasAttr("async_agent")) {
          labelByUsers(userOp, allAgents);
        }
        addAgentIds(op, getAgentIds(userOp));
      }
    }
    if (!op->hasAttr("async_agent")) {
      addAgentIds(op, allAgents);
    }
  }

protected:
  MLIRContext *context;
  ModuleOp mod;
  int computeCapability;
  OpBuilder builder;
};

//===------------------------heuristics list------------------------===//
// List all heuristics here:
//  - Heuristic_Load_MathStore: assign load and math+store to two
//    different agents respectively.
//===---------------------------------------------------------------===//

class Heuristic_Load_MathStore : public Heuristics {
public:
  Heuristic_Load_MathStore(MLIRContext *context, ModuleOp mod,
                           const int &computeCapability)
      : Heuristics(context, mod, computeCapability) {}
  bool run() override {
    constexpr int kLoadAgentId = 0;
    constexpr int kStoreAgentId = 1;
    constexpr int kNumAgents = 2;

    //===--------------------1. label key operations--------------------===//
    setAgentId_if(kLoadAgentId, isWSCandidateLoad);
    setAgentId_if(kStoreAgentId, isTritonStoreOp);

    //===--------------2. populate based on key operations--------------===//
    // find the roots (outputs) of LoadAgent
    DenseSet<Operation *> loadOps;
    getAgentOps(loadOps, kLoadAgentId);
    // find LoadAgent dependent ops
    DenseSet<Value> loadValues;
    DenseSet<Value> loadAgentDepValues;
    for (Operation *op : loadOps) {
      if (failed(getDependentValues(op, loadAgentDepValues)))
        return false;
      loadValues.insert(op->getResult(0));
    }
    for (Operation *op : getDependentOps(loadAgentDepValues))
      addAgentIds(op, kLoadAgentId);

    // find the roots (outputs) of StoreAgent
    DenseSet<Operation *> storeOps;
    getAgentOps(storeOps, kStoreAgentId);
    // find StoreAgent dependent ops
    DenseSet<Value> storeAgentDepValues;
    for (Operation *op : storeOps)
      if (failed(getDependentValues(op, storeAgentDepValues, loadValues)))
        return false;
    for (Operation *op : getDependentOps(storeAgentDepValues))
      addAgentIds(op, kStoreAgentId);

    //===---------------------3. label unlabeld Ops---------------------===//
    populateUnlabledOpsAtLast({kLoadAgentId, kDotAgentId});

    // Erase labels of MakeTensorPtrOp and its definingOps,
    // because we don't want them to be copied in each agent
    SetVector<Operation *> backwardSlice;
    mod.walk([&](triton::MakeTensorPtrOp op) -> void {
      assert(isa<triton::FuncOp>(op->getParentOp()));
      mlir::BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      getBackwardSlice(op.getOperation(), &backwardSlice, opt);
      op->removeAttr("async_agent");
    });
    for (auto op : backwardSlice) {
      op->removeAttr("async_agent");
    }
    // Set num-agents for wsmaterialization pass
    mod->setAttr("async.num-agents", builder.getI32IntegerAttr(kNumAgents));
    return true;
  }
};

class TritonGPUWSDecomposingPass
    : public TritonGPUWSDecomposingBase<TritonGPUWSDecomposingPass> {
public:
  TritonGPUWSDecomposingPass() = default;
  TritonGPUWSDecomposingPass(int computeCapability) {
    this->computeCapability = computeCapability;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    if (!ttng::TritonNvidiaGPUDialect::getWSSupportedAttr(mod))
      return signalPassFailure();

    // Build Heuristics
    Heuristic_Load_MathStore hLoadMathBasic(context, mod, computeCapability);
    if (!(hLoadMathBasic.run())) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass>
mlir::createTritonNvidiaGPUWSDecomposingPass(int computeCapability) {
  return std::make_unique<TritonGPUWSDecomposingPass>(computeCapability);
}
