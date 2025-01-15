#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "triton-gpu-taskid-propagate"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUTASKIDPROPAGATE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

// Return all Ops that are marked with target task
void getAsyncTaskOps(triton::FuncOp funcOp, DenseSet<Operation *> &asyncTaskOps,
                     int asyncTaskId) {
  funcOp.walk([&](Operation *op) -> void {
    if (auto attr =
            op->getAttrOfType<mlir::DenseIntElementsAttr>("async_task_id")) {
      for (auto val : attr.getValues<int>()) {
        if (val == asyncTaskId) {
          asyncTaskOps.insert(op);
          break;
        }
      }
    }
  });
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

void getAllParentOps(triton::FuncOp funcOp, DenseSet<Operation *> &parentOps,
                     int asyncTaskId) {
  DenseSet<Operation *> targetOps;
  getAsyncTaskOps(funcOp, targetOps, asyncTaskId);
  for (auto op : targetOps) {
    getAllParentOps(parentOps, op);
  }
}

void labelByUsers(Operation *op, ArrayRef<int> allAsyncTasks) {
  for (Value result : op->getResults()) {
    for (Operation *userOp : result.getUsers()) {
      if (!userOp->hasAttr("async_task_id")) {
        labelByUsers(userOp, allAsyncTasks);
      }
      addAsyncTaskIds(op, getAsyncTaskIds(userOp));
    }
  }
  if (!op->hasAttr("async_task_id")) {
    addAsyncTaskIds(op, allAsyncTasks);
  }
}

/// Because we set some special filter rules in populateAsyncTaskRegion,
/// there may be unlabeled Ops, e.g. YieldOps, some definingOps of ForOps.
/// or Ops without relations to asyncTaskOps
void populateUnlabledOpsAtLast(triton::FuncOp funcOp,
                               ArrayRef<int> allAsyncTasks) {
  // Label asyncTasks' parentOps
  for (int i : allAsyncTasks) {
    DenseSet<Operation *> asyncTaskParentOps;
    getAllParentOps(funcOp, asyncTaskParentOps, i);
    for (auto op : asyncTaskParentOps) {
      addAsyncTaskIds(op, {i});
    }
  }

  // Get unlabeled Ops
  DenseSet<Operation *> unlabeledOps;
  funcOp.walk([&](Operation *op) -> void {
    if (isa<ModuleOp>(op) || isa<triton::FuncOp>(op) ||
        isa<triton::ReturnOp>(op)) {
      return;
    }
    if (!op->hasAttr("async_task_id")) {
      unlabeledOps.insert(op);
    }
  });

  // Label Ops using its parentOp
  for (auto op : unlabeledOps) {
    if (auto parent = op->getParentOp()) {
      if (!isa<triton::FuncOp>(parent)) {
        if (!parent->hasAttr("async_task_id")) {
          LLVM_DEBUG({
            LDBG("op and parent: ");
            op->dump();
            parent->dump();
          });
          continue;
        }
        assert(parent->hasAttr("async_task_id"));
        auto asyncTasks = getAsyncTaskIds(parent);
        setAsyncTaskIds(op, asyncTasks);
        unlabeledOps.erase(op);
      }
    }
  }

  // Label Ops using dependency
  for (auto op : unlabeledOps) {
    labelByUsers(op, allAsyncTasks);
    unlabeledOps.erase(op);
  }
  assert(unlabeledOps.size() == 0);
}

#ifndef NDEBUG
static bool oneVecCoversTheOther(SmallVector<AsyncTaskId> &one,
                                 SmallVector<AsyncTaskId> &other) {
  // Every element of other appears in one.
  for (AsyncTaskId t : other) {
    // If t doesn't appear in one, return false.
    bool found = false;
    for (AsyncTaskId t2 : one) {
      if (t2 == t) {
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}

struct AsyncTaskIdsCompare {
  static SmallVector<AsyncTaskId> getEmptyKey() {
    SmallVector<AsyncTaskId> V;
    V.push_back(reinterpret_cast<AsyncTaskId>(-1));
    return V;
  }

  static SmallVector<AsyncTaskId> getTombstoneKey() {
    SmallVector<AsyncTaskId> V;
    V.push_back(reinterpret_cast<AsyncTaskId>(-2));
    return V;
  }

  static unsigned getHashValue(const SmallVector<AsyncTaskId> &V) {
    return static_cast<unsigned>(llvm::hash_combine_range(V.begin(), V.end()));
  }

  static bool isEqual(const SmallVector<AsyncTaskId> &LHS,
                      const SmallVector<AsyncTaskId> &RHS) {
    return LHS == RHS;
  }
};

// Make sure the def chain contains the right taskId.
bool verifyTaskId(triton::FuncOp &funcOp,
                  const llvm::DenseSet<Operation *> &anchorOps) {
  bool retCode = true;
  DenseSet<SmallVector<AsyncTaskId>, AsyncTaskIdsCompare> anchorAsyncTasks;
  for (auto anchorOp : anchorOps) {
    anchorAsyncTasks.insert(getAsyncTaskIds(anchorOp));
  }

  funcOp.walk([&](Operation *op) {
    // Skip control ops
    if (llvm::isa<ReturnOp, FuncOp, scf::YieldOp, scf::ForOp>(op))
      return;

    auto asyncTaskIds = getAsyncTaskIds(op);
    if (asyncTaskIds.empty()) {
      LLVM_DEBUG({
        LDBG("Op does not have task id");
        op->dump();
      });
      llvm_unreachable("Op does not have task id");
    }

    auto partitionShouldBeUsedSpecified = [](Operation *op) {
      if (isa<StoreOp, ExperimentalDescriptorLoadOp>(op))
        return true;
      if (isa<AtomicRMWOp, AtomicCASOp>(op))
        return true;
      if (op->hasTrait<OpTrait::DotLike>())
        return true;
      return false;
    };

    if (!anchorAsyncTasks.contains(asyncTaskIds)) {
      if (partitionShouldBeUsedSpecified(op)) {
        LLVM_DEBUG({
          LDBG("async tasks not specified by user");
          op->dump();
        });
        llvm_unreachable("async tasks not specified by user");
      }
    }

    assert(!asyncTaskIds.empty() && "Op does not have task id");

    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp)
        continue;
      if (llvm::isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(defOp))
        continue;
      auto defTaskIds = getAsyncTaskIds(defOp);
      // Make sure defTaskIds cover asyncTaskIds. Call addAsyncTaskIds if
      // necessary.
      LLVM_DEBUG({
        if (!oneVecCoversTheOther(defTaskIds, asyncTaskIds)) {
          // print defOp and op
          LDBG("Def op does not cover op");
          LDBG("Def op");
          defOp->dump();
          LDBG("op");
          op->dump();
        }
      });
      assert(oneVecCoversTheOther(defTaskIds, asyncTaskIds) &&
             "defTaskIds should cover asyncTaskIds");
    }
  });
  return retCode;
}
#endif

void backwardPropagateTaskIds(Operation *op,
                              const llvm::DenseSet<Operation *> &anchors) {
  SmallVector<Value> queue;
  auto asyncTasks = getAsyncTaskIds(op);
  for (Value operand : op->getOperands()) {
    queue.push_back(operand);
  }

  DenseSet<Value> seen;
  for (auto anchor : anchors) {
    if (anchor != op)
      for (auto result : anchor->getResults())
        seen.insert(result);
  }

  while (!queue.empty()) {
    auto value = queue.pop_back_val();
    if (!seen.insert(value).second) {
      continue;
    }

    // Handle BlockArguments of for loops (i.e. loop carried dependences).
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      auto parent = blockArg.getOwner()->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        // Propagate to the control operands.
        auto control =
            forOp.getOperands().take_front(forOp.getNumControlOperands());
        queue.insert(queue.end(), control.begin(), control.end());
        // Propagate to the initializer.
        if (blockArg.getArgNumber() >= forOp.getNumInductionVars()) {
          queue.push_back(forOp.getTiedLoopInit(blockArg)->get());
          // Propagate to the yield.
          auto idx = blockArg.getArgNumber() - forOp.getNumInductionVars();
          queue.push_back(forOp.getBody()->getTerminator()->getOperand(idx));
          addAsyncTaskIds(forOp, asyncTasks);
        }
      }
      continue;
    }

    auto op = value.getDefiningOp();
    addAsyncTaskIds(op, asyncTasks);

    // Handle for loops.
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Propagate to control operands.
      auto control =
          forOp.getOperands().take_front(forOp.getNumControlOperands());
      queue.insert(queue.end(), control.begin(), control.end());
      // Propagate to arguments.
      unsigned idx = cast<OpResult>(value).getResultNumber();
      queue.push_back(forOp.getOperand(idx + forOp.getNumControlOperands()));
      // Propagate to yield.
      queue.push_back(forOp.getBody()->getTerminator()->getOperand(idx));
      continue;
    }

    // Handle conditionals.
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      queue.push_back(ifOp.getCondition());
      unsigned idx = cast<OpResult>(value).getResultNumber();
      if (ifOp.elseBlock()) {
        queue.push_back(ifOp.elseYield()->getOperand(idx));
      }
      queue.push_back(ifOp.thenYield()->getOperand(idx));
      continue;
    }

    // Handle normal ops.
    for (Value operand : op->getOperands()) {
      queue.push_back(operand);
    }
  }
}

void backwardPropagateTaskIds(llvm::DenseSet<Operation *> &rootOps,
                              llvm::DenseSet<Operation *> &anchorOps) {
  for (Operation *op : rootOps) {
    backwardPropagateTaskIds(op, anchorOps);
  }
}

void forwardPropagateTaskIds(Operation *root,
                             const llvm::DenseSet<Operation *> &anchors) {
  auto asyncTasks = getAsyncTaskIds(root);
  SmallVector<Value> queue;
  for (Value result : root->getResults())
    queue.push_back(result);

  DenseSet<Value> seen;
  for (auto anchor : anchors) {
    if (anchor != root)
      for (auto result : anchor->getResults())
        seen.insert(result);
  }

  while (!queue.empty()) {
    auto v = queue.back();
    queue.pop_back();
    if (!seen.insert(v).second)
      continue;

    for (Operation *depOp : v.getUsers()) {
      auto depAsyncTasks = getAsyncTaskIds(depOp);
      // Skip depOp that already has task ids. Those could be either anchorOps
      // or propagated backward from anchor ops.
      if (!depAsyncTasks.empty() && depAsyncTasks != asyncTasks)
        continue;
      setAsyncTaskIds(depOp, asyncTasks);
      // Go through yieldOp to propagate task ids to the result of parentOp.
      if (auto yieldOp = dyn_cast<scf::YieldOp>(depOp)) {
        auto parentOp = yieldOp->getParentOp();
        for (OpOperand &operand : yieldOp->getOpOperands()) {
          if (operand.get() == v) {
            queue.push_back(parentOp->getResult(operand.getOperandNumber()));
            break;
          }
        }
      } else {
        for (Value result : depOp->getResults())
          queue.push_back(result);
      }
    }
  }
}

void forwardPropagateTaskIds(llvm::DenseSet<Operation *> &anchorOps) {
  for (Operation *op : anchorOps) {
    forwardPropagateTaskIds(op, anchorOps);
  }
}

void populateTaskIdsForControlDependencies(
    llvm::DenseSet<Operation *> &anchorOps) {
  for (auto op : anchorOps) {
    auto asyncTaskIds = getAsyncTaskIds(op);
    if (!asyncTaskIds.empty()) {
      while (auto parent = op->getParentOp()) {
        if (!isa<ModuleOp>(parent) && !isa<triton::FuncOp>(parent)) {
          setAsyncTaskIds(parent, asyncTaskIds);
          backwardPropagateTaskIds(parent, anchorOps);
          op = parent;
        } else {
          break;
        }
      }
    }
  }
}

class TritonGPUTaskIdPropagatePass
    : public impl::TritonGPUTaskIdPropagateBase<TritonGPUTaskIdPropagatePass> {
public:
  using impl::TritonGPUTaskIdPropagateBase<
      TritonGPUTaskIdPropagatePass>::TritonGPUTaskIdPropagateBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    llvm::DenseSet<Operation *> anchorOps;
    funcOp.walk([&](mlir::Operation *op) {
      auto asyncTasks = getAsyncTaskIds(op);
      if (asyncTasks.empty())
        return;
      std::sort(asyncTasks.begin(), asyncTasks.end());
      setAsyncTaskIds(op, asyncTasks);
      if (!isa<arith::ConstantOp, arith::ConstantIntOp>(op))
        anchorOps.insert(op);
    });

    populateTaskIdsForControlDependencies(anchorOps);

    LLVM_DEBUG({
      LDBG("after populateTaskIdsForControlDependencies ");
      funcOp->dump();
    });

    backwardPropagateTaskIds(anchorOps, anchorOps);

    LLVM_DEBUG({
      LDBG("after backwardPropagateTaskIds ");
      funcOp->dump();
    });

    forwardPropagateTaskIds(anchorOps);

    LLVM_DEBUG({
      LDBG("after forwardPropagateTaskIds ");
      funcOp->dump();
    });

    llvm::DenseSet<Operation *> rootOps;
    funcOp.walk([&](mlir::Operation *op) {
      auto asyncTasks = getAsyncTaskIds(op);
      if (!asyncTasks.empty() &&
          !isa<arith::ConstantOp, arith::ConstantIntOp>(op))
        rootOps.insert(op);
    });
    backwardPropagateTaskIds(rootOps, anchorOps);
    LLVM_DEBUG({
      LDBG("after final backwardPropagateTaskIds ");
      funcOp->dump();
    });

    DenseSet<int> allAsyncTasks;
    funcOp->walk([&](Operation *op) {
      auto asyncTasks = getAsyncTaskIds(op);
      allAsyncTasks.insert(asyncTasks.begin(), asyncTasks.end());
    });
    SmallVector<int> allAsyncTasksVec(allAsyncTasks.begin(),
                                      allAsyncTasks.end());
    populateUnlabledOpsAtLast(funcOp, allAsyncTasksVec);

    LLVM_DEBUG({
      LDBG("after populateUnlabledOpsAtLast ");
      funcOp->dump();
    });

#ifndef NDEBUG
    verifyTaskId(funcOp, anchorOps);
#endif
  }

  void runOnOperation() override {
    if (numConsumerGroups == 0) {
      getOperation()->walk([&](triton::FuncOp funcOp) {
        funcOp.walk([&](mlir::Operation *op) {
          auto asyncTasks = getAsyncTaskIds(op);
          if (!asyncTasks.empty())
            op->removeAttr("async_task_id");
        });
      });
      return;
    }
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
