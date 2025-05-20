/*
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
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

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/WSUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#include <stack>

#define GEN_PASS_CLASSES
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-propagate-groups"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;
using namespace triton::nvws;

// Assign groups to an operation
void addOperationGroups(Operation *op, const std::set<std::string> &groups) {
  if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
    // for reduce op, assign all ops in the combine op to the groups
    reduceOp.walk([&](Operation *op) { addGroups(op, groups); });
    addGroups(op, groups);
    return;
  }

  addGroups(op, groups);
}

// Assign groups to a value
void addValueGroups(const Value &value, const std::set<std::string> &groups) {
  // to for loop iter args
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto parentOp = blockArg.getOwner()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      // ignore loop variable
      if (blockArg.getArgNumber() > 0) {
        auto idx = blockArg.getArgNumber() - 1;
        addGroups(cast<OpResult>(forOp.getResult(idx)), groups);
      }
    }
  }

  // to for and if op results
  if (auto result = dyn_cast<OpResult>(value)) {
    auto op = result.getOwner();
    if (isa<scf::ForOp, scf::IfOp>(op)) {
      addGroups(result, groups);
    }
  }
}

bool isControlFlowOp(Operation *op) {
  return isa<scf::IfOp, scf::ForOp, scf::YieldOp>(op);
}

SetVector<Operation *> findRoots(ModuleOp module) {
  // find all "roots"
  // i.e. ops with an initial group assignment
  // that are not control flow
  SetVector<Operation *> roots;
  module.walk([&](Operation *op) {
    if (op->hasAttr(ATTR_WSGROUPS) && !isControlFlowOp(op)) {
      roots.insert(op);
    }
  });
  return roots;
}

// get predecessors for a value:
//  - if the value is a function argument, return nothing
//  - if the value is defined by an IfOp,
//    return the corresponding operands from the yield ops in the then and
//    else branch
//  - if the value is an iter_arg of a ForOp,
//    return the corresponding init arg and operand from the yield in the
//    ForOp
SetVector<Value> getPredecessors(Value value) {
  SetVector<Value> preds;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    // value is a block arg, need to find where it is actually defined
    auto parentOp = blockArg.getOwner()->getParentOp();

    // defined in FuncOp, no predecessors
    if (isa<FuncOp>(parentOp)) {
      return {};
    }

    // defined as an iter arg of a for loop,
    // find the corresponding values from the init args and yield
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      // ignore induction variable
      if (blockArg.getArgNumber() == 0) {
        return {};
      }
      auto idx = blockArg.getArgNumber() - forOp.getNumInductionVars();
      preds.insert(forOp.getOperand(idx + forOp.getNumControlOperands()));
      preds.insert(forOp.getBody()->getTerminator()->getOperand(idx));
      return preds;
    }

    assert(false && "block arg not supported for this parent op");
  }

  auto op = value.getDefiningOp();
  assert(op);

  // is defined by an if op, get the operands for the yields
  // from the then and else branches
  // and the condition
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    preds.insert(ifOp.getCondition());
    auto idx = cast<OpResult>(value).getResultNumber();
    preds.insert(ifOp.thenYield()->getOperand(idx));
    if (ifOp.elseBlock()) {
      preds.insert(ifOp.elseYield()->getOperand(idx));
    }
    return preds;
  }

  // is defined by a for op, get the operand for the yield
  // and the lb/ub/step
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    preds.insert(forOp.getLowerBound());
    preds.insert(forOp.getUpperBound());
    preds.insert(forOp.getStep());
    auto idx = cast<OpResult>(value).getResultNumber();
    preds.insert(forOp.getBody()->getTerminator()->getOperand(idx));
    return preds;
  }

  // normal op, predecessors are simply its operands
  for (auto operand : op->getOperands()) {
    preds.insert(operand);
  }
  return preds;
}

// Note: we return values and operations here, as the successor of a value may
// not define any values. When this is the case, it is returned in the
// operations list
std::pair<SmallVector<Value>, SmallVector<Operation *>>
getSuccessors(Value value) {
  std::pair<SmallVector<Value>, SmallVector<Operation *>> succs;
  for (auto &use : value.getUses()) {
    auto op = use.getOwner();
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // used as confition in if op
      // data flow ends here so do nothing
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // argument to for op
      // if its an init arg
      if (use.getOperandNumber() >= forOp.getNumControlOperands()) {
        // get corresponding iter arg
        succs.first.push_back(forOp.getRegionIterArg(
            use.getOperandNumber() - forOp.getNumControlOperands()));
      }
      // if its lb/ub/step, data flow ends here, so do nothing
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      auto parentOp = yieldOp.getOperation()->getParentOp();
      if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
        // yield from an if
        // get corresponding value returned by if
        succs.first.push_back(ifOp.getResult(use.getOperandNumber()));
      } else if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        // yield from a for
        // get corresponding value returned by for and iter_arg
        succs.first.push_back(forOp.getResult(use.getOperandNumber()));
        succs.first.push_back(forOp.getRegionIterArg(use.getOperandNumber()));
      } else {
        assert(false && "yield op not supported with this parent op");
      }
    } else {
      // normal op
      for (auto result : op->getResults()) {
        succs.first.push_back(result);
      }
      if (op->getResults().empty()) {
        succs.second.push_back(op);
      }
    }
  }
  return succs;
}

class RootPropagation {
public:
  void propagate(const SetVector<Operation *> &roots) {
    this->roots = roots;

    // backward propagate groups from root ops
    for (auto op : roots) {
      backwardPropagateGroups(op);
    }

    // forward/backward propagate groups from root ops to ops with no groups
    for (auto op : roots) {
      bidirectionalPropagateGroups(op);
    }
  }

  // backward propagate groups from a root op to all its predecessors
  //  - stop if we hit another root op
  void backwardPropagateGroups(Operation *root) {
    assert(roots.count(root) > 0);
    auto groups = getGroups(root);

    llvm::SmallPtrSet<Value, 16> visited;
    std::stack<Value> stack;
    for (auto operand : root->getOperands()) {
      stack.push(operand);
      visited.insert(operand);
    }

    while (!stack.empty()) {
      auto value = stack.top();
      stack.pop();

      auto op = value.getDefiningOp();
      if (op) {
        // stop at roots
        if (roots.count(op) > 0) {
          continue;
        }

        // assign groups to non-control flow ops
        if (!isControlFlowOp(op)) {
          addOperationGroups(op, groups);
        }
      }

      addValueGroups(value, groups);

      // continue search into predecessors
      auto preds = getPredecessors(value);
      for (auto pred : preds) {
        if (visited.count(pred) == 0) {
          stack.push(pred);
          visited.insert(pred);
        }
      }
    }
  }

  // propagate groups forwards and backwards from a start op,
  // stopping if at ops that have groups assigned
  void bidirectionalPropagateGroups(Operation *root) {
    assert(roots.count(root) > 0);
    auto groups = getGroups(root);

    llvm::SmallPtrSet<Value, 16> visited;
    std::stack<Value> stack;
    for (auto value : root->getOperands()) {
      // note: if operand is defined by an op that has groups,
      // don't search it
      auto op = value.getDefiningOp();
      if (op && op->hasAttr(ATTR_WSGROUPS)) {
        continue;
      }
      stack.push(value);
      visited.insert(value);
    }
    for (auto value : root->getResults()) {
      stack.push(value);
      visited.insert(value);
    }

    while (!stack.empty()) {
      auto value = stack.top();
      stack.pop();

      // invariant: value popped from stack:
      //  - should be assigned to current groups
      //  - should be searched forwards/backwards

      auto op = value.getDefiningOp();

      // set groups for defining op
      if (op && !op->hasAttr(ATTR_WSGROUPS) && !isControlFlowOp(op)) {
        // assign groups
        addOperationGroups(op, groups);
      }

      addValueGroups(value, groups);

      // search predecessors
      for (auto it : getPredecessors(value)) {
        auto op = it.getDefiningOp();
        if (op && op->hasAttr(ATTR_WSGROUPS)) {
          continue;
        }
        if (visited.count(it) == 0) {
          stack.push(it);
          visited.insert(it);
        }
      }

      // search successors
      {
        auto [values, ops] = getSuccessors(value);
        for (auto it : values) {
          auto op = it.getDefiningOp();
          if (op && op->hasAttr(ATTR_WSGROUPS)) {
            continue;
          }
          if (visited.count(it) == 0) {
            stack.push(it);
            visited.insert(it);
          }
        }
        // special case: ops that define no values
        // we assign them to the group, if they have none,
        // then search their operands
        for (auto op : ops) {
          if (op->hasAttr(ATTR_WSGROUPS)) {
            continue;
          }
          addOperationGroups(op, groups);
          for (auto value : op->getOperands()) {
            if (visited.count(value) == 0) {
              stack.push(value);
              visited.insert(value);
            }
          }
        }
      }
    }
  }

private:
  SetVector<Operation *> roots;
};

class ControlFlowPropagation {
public:
  ControlFlowPropagation() {}

  void propagate(const SmallVector<Operation *> &controlFlowOps) {
    // propagate group information for control ops until we reach a fixed point
    bool first_pass = true;

    while (true) {
      bool changed = false;
      for (auto op : controlFlowOps) {
        // set groups to the union of the ops current groups and
        // all ops it contains
        auto groups = getGroups(op);
        op->walk([&](Operation *op) {
          auto subGroups = getGroups(op);
          groups.insert(subGroups.begin(), subGroups.end());
        });
        bool op_changed = getGroups(op) != groups;
        changed |= op_changed;
        // if groups changes, or this is the first pass, propagate groups
        if (first_pass || op_changed) {
          setGroups(op, groups);
          doPropagate(op);
        }
      }
      if (!first_pass && !changed) {
        // reached fixed point, so done
        break;
      }
      first_pass = false;
    }
  }

  void doPropagate(Operation *root) {
    auto groups = getGroups(root);

    SmallVector<Value> stack;
    if (auto forOp = dyn_cast<scf::ForOp>(root)) {
      stack.push_back(forOp.getLowerBound());
      stack.push_back(forOp.getUpperBound());
      stack.push_back(forOp.getStep());
    } else if (auto ifOp = dyn_cast<scf::IfOp>(root)) {
      stack.push_back(ifOp.getCondition());
    } else {
      assert(false && "control flow op not supported");
    }

    llvm::SmallPtrSet<Value, 16> visited;
    visited.insert(stack.begin(), stack.end());

    while (!stack.empty()) {
      auto value = stack.back();
      stack.pop_back();

      auto op = value.getDefiningOp();
      if (op) {
        addOperationGroups(op, groups);
      }

      addValueGroups(value, groups);

      // continue search into predecessors
      auto preds = getPredecessors(value);
      for (auto pred : preds) {
        if (visited.count(pred) == 0) {
          stack.push_back(pred);
          visited.insert(pred);
        }
      }
    }
  }
};

class NVWSPropagateGroups
    : public NVWSPropagateGroupsBase<NVWSPropagateGroups> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    // get all roots, based on the initial group assignment
    auto roots = findRoots(m);

    // propagate groups from root ops
    RootPropagation().propagate(roots);
    LLVM_DEBUG({
      DBGS() << "Module after group propagation for roots:\n";
      m.dump();
    });

    // set control flow ops as being in
    // union of groups mentioned in their regions
    SmallVector<Operation *> controlFlowOps;
    m.walk([&](Operation *op) {
      if (isControlFlowOp(op) && !isa<scf::YieldOp>(op)) {
        controlFlowOps.push_back(op);
      }
    });

    // propagate groups for control flow ops
    ControlFlowPropagation().propagate(controlFlowOps);
    LLVM_DEBUG({
      DBGS() << "Module after propagating control flow groups:\n";
      m.dump();
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createNVWSPropagateGroupsPass() {
  return std::make_unique<NVWSPropagateGroups>();
}
