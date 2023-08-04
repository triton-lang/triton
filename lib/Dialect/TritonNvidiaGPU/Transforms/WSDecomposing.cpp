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
  //  - Op without agent attr: opeations shared by all agents but will NOT
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

  void getBackwardSliceInForLoop(Operation *t,
                                 SetVector<Operation *> *backwardSlice,
                                 SetVector<Operation *> *iterArgSlice,
                                 TransitiveFilter backwardFilter = nullptr) {
    for (auto v : t->getOperands()) {
      if (v.getDefiningOp() == nullptr) {
        auto blockArg = v.cast<BlockArgument>();
        Operation *op = blockArg.getOwner()->getParentOp();
        // scf.for: the dependency traversing would be based on
        // specific arguments rather than whole ForOp.
        if (scf::ForOp forOp = dyn_cast<scf::ForOp>(op)) {
          // IterArgs
          if (blockArg.getArgNumber() >= forOp.getNumInductionVars()) {
            auto operand = forOp.getOperand(forOp.getNumControlOperands() +
                                            blockArg.getArgNumber() -
                                            forOp.getNumInductionVars());
            if (auto iterArgsDefiningOp = operand.getDefiningOp()) {
              iterArgSlice->insert(iterArgsDefiningOp);
            }
            // In-loop depenency
            auto yieldOp =
                dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
            auto yieldResult =
                yieldOp.getResults()[blockArg.getArgNumber() -
                                     forOp.getNumInductionVars()];
            auto defOp = yieldResult.getDefiningOp();
            if (defOp->getBlock() == forOp.getBody() &&
                (!backwardFilter || backwardFilter(defOp))) {
              backwardSlice->insert(defOp);
            }
          }
        }
      }
    }
    return;
  }

  // Extend region of an agent to operations that are associated with target
  // agent, but stop at the endOps (not include endOps).
  virtual bool populateAgentRegion(
      DenseSet<Operation *> &targetOps, int agentId,
      const DenseSet<Operation *> &endOps = DenseSet<Operation *>(),
      DenseSet<Operation *> *definedOps = nullptr) {
    DenseSet<Operation *> definedOpsObj;
    if (!definedOps) {
      definedOps = &definedOpsObj;
    }
    auto filter = [&](Operation *op) -> bool {
      if (endOps.contains(op) || isa<triton::FuncOp>(op) || isa<ModuleOp>(op) ||
          isa<scf::ForOp>(op)) {
        return false;
      }
      return true;
    };

    for (auto op : targetOps) {
      assert(hasAgentId(op, agentId) && "Discrepancy between op and agent id");
      definedOps->insert(op);
      SetVector<Operation *> backwardSlice, sliceInLoop, iterArgSlice;
      DenseSet<Operation *> tmpOps;

      auto labelForOp = [&](scf::ForOp forOp) -> void {
        // Add agent for forOp in case sliceInLoop is empty.
        addAgentIds(forOp, agentId);
        addAgentIds(forOp.getBody()->getTerminator(), agentId);
        // ControlOperands
        for (int i = 0; i < forOp.getNumControlOperands(); ++i) {
          if (auto controlDefiningOp = forOp.getOperand(i).getDefiningOp()) {
            addAgentIds(controlDefiningOp, agentId);
            if (!definedOps->contains(controlDefiningOp) &&
                !endOps.contains(controlDefiningOp)) {
              tmpOps.insert(controlDefiningOp);
            }
          }
        }
      };

      auto labelOp = [&](Operation *x) -> void {
        x->walk([&](Operation *y) -> void {
          addAgentIds(y, agentId);
          if (x != y && !isa<scf::YieldOp>(y)) {
            if (!definedOps->contains(y) && !endOps.contains(y)) {
              tmpOps.insert(y);
            }
          }
        });
      };
      getBackwardSlice(op, &backwardSlice, filter);
      getBackwardSliceInForLoop(op, &sliceInLoop, &iterArgSlice, filter);
      if (scf::ForOp forOp = dyn_cast<scf::ForOp>(op->getParentOp())) {
        labelForOp(forOp);
      }
      for (auto s : backwardSlice) {
        getBackwardSliceInForLoop(s, &sliceInLoop, &iterArgSlice, filter);
        labelOp(s);
      }
      for (auto argOp : iterArgSlice) {
        // addAgentIds(argOp, agentId);
        labelOp(argOp);
        if (!definedOps->contains(argOp) && !endOps.contains(argOp)) {
          tmpOps.insert(argOp);
        }
      }
      for (auto sl : sliceInLoop) {
        labelOp(sl);
        // addAgentIds(sl, agentId);
        if (sl != op && !definedOps->contains(sl) && !endOps.contains(sl)) {
          tmpOps.insert(sl);
        }
        scf::ForOp forOp = cast<scf::ForOp>(sl->getParentOp());
        labelForOp(forOp);
      }
      populateAgentRegion(tmpOps, agentId, endOps, definedOps);
    }
    return true;
  }

  /// Becuase we set some special filter rules in populateAgentRegion,
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
    constexpr int kDotAgentId = 1;
    constexpr int kStoreAgentId = kDotAgentId;

    //===--------------------1. label key operations--------------------===//
    setAgentId_if(kLoadAgentId, isTritonLoadOp);
    setAgentId_if(kDotAgentId, isTritonDotOp);
    setAgentId_if(kStoreAgentId, isTritonStoreOp);

    DenseSet<Operation *> agent1ForbiddenOps, agent0ParentOps;
    // loadOps are forbiddenOps for dotAgent.
    getAgentOps(agent1ForbiddenOps, kLoadAgentId);

    //===--------------2. populate based on key operations--------------===//
    getAllParentOps(agent0ParentOps, kLoadAgentId);

    /// Filter out the operation which is targetOps' parentOp.
    /// agent0targetOps' parentOps are also inserted into agent1ForbiddenOps.
    /// Otherwise, these parentOps will be labeled with agent1 and keep
    /// populating. Take simple gemm as an example: since
    /// kStoreAgentId=kDotAgentId, and storeOps are dependent on an forOp which
    /// contains both loadOps and DotOps, if this forOp is not in agent1's
    /// forbiddenOps, then its all definingOps will be labeled with agentId 1,
    /// which is not true, because some of them are only be used by loadOps.
    for (auto f1 : agent0ParentOps) {
      agent1ForbiddenOps.insert(f1);
    }

    // loadOps have no forbiddenOps.
    DenseSet<Operation *> agent0KeyOps, agent1KeyOps;
    getAgentOps(agent0KeyOps, kLoadAgentId);
    getAgentOps(agent1KeyOps, kDotAgentId);
    bool loadSuccess = populateAgentRegion(agent0KeyOps, kLoadAgentId);
    bool dotSuccess =
        populateAgentRegion(agent1KeyOps, kDotAgentId, agent1ForbiddenOps);

    //===---------------------3. label unlabeld Ops---------------------===//
    populateUnlabledOpsAtLast({kLoadAgentId, kDotAgentId});

    // Erase labels of MakeTensorPtrOp and its definingOps,
    // because we don't want them to be copied in each agent
    SetVector<Operation *> backwardSlice;
    mod.walk([&](triton::MakeTensorPtrOp op) -> void {
      assert(isa<triton::FuncOp>(op->getParentOp()));
      getBackwardSlice(op.getOperation(), &backwardSlice);
      op->removeAttr("async_agent");
    });
    for (auto op : backwardSlice) {
      op->removeAttr("async_agent");
    }
    return loadSuccess && dotSuccess;
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
