#include "ScheduleMembar.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::triton::AMD {
namespace {
bool isGlobalRead(Operation *op) {
  auto interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface)
    return false;
  SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);
  return !effects.empty() && llvm::all_of(effects, [](const auto &effect) {
    return isa<MemoryEffects::Read>(effect.getEffect()) &&
           isa<GlobalMemory>(effect.getResource());
  });
}

// Follow only direct producers and one incoming CFG edge; this is a latency
// heuristic, not an analysis of outstanding hardware memory requests.
bool isGlobalReadResult(Value value) {
  if (Operation *producer = value.getDefiningOp())
    return isGlobalRead(producer);
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument)
    return false;
  Block *block = argument.getOwner();
  unsigned index = argument.getArgNumber();
  for (Block *pred : block->getPredecessors()) {
    auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
    if (!branch)
      continue;
    for (unsigned i = 0; i < branch->getNumSuccessors(); ++i) {
      if (branch->getSuccessor(i) != block)
        continue;
      auto operands = branch.getSuccessorOperands(i);
      if (index >= operands.size() || operands.isOperandProduced(index))
        continue;
      Operation *producer = operands[index].getDefiningOp();
      if (producer && isGlobalRead(producer))
        return true;
    }
  }
  return false;
}

bool consumesGlobalRead(Operation *op) {
  return llvm::any_of(op->getOperands(), isGlobalReadResult);
}

bool canCross(Operation *op) {
  return op->getNumRegions() == 0 &&
         (isMemoryEffectFree(op) || isGlobalRead(op));
}

bool isCompletionBoundary(Operation *op) {
  return isa<gpu::AsyncWaitOp, gpu::AsyncCommitGroupOp>(op) ||
         op->hasTrait<OpTrait::MemWaitOpTrait>();
}

bool isLocalBarrier(Operation *op) {
  auto barrier = dyn_cast<gpu::BarrierOp>(op);
  return barrier && !barrier.isWarp() &&
         barrier.getAddrSpace() == gpu::AddrSpace::Local;
}

struct WaitPlan {
  gpu::AsyncWaitOp wait;
  Operation *insertionPoint;
  llvm::SetVector<Operation *> barriers;
};

// Collect the first full local barrier on every path. Crossing shared effects
// could remove synchronization that Membar relied on for unrelated accesses.
bool findBarriers(Operation *op, WaitPlan &plan,
                  llvm::SmallPtrSetImpl<Operation *> &active) {
  if (!op || !active.insert(op).second)
    return false;
  bool found = false;
  if (isLocalBarrier(op)) {
    plan.barriers.insert(op);
    found = true;
  } else if (isa<gpu::BarrierOp>(op) || isCompletionBoundary(op)) {
    found = false;
  } else if (auto branch = dyn_cast<BranchOpInterface>(op)) {
    found = branch->getNumSuccessors() != 0;
    for (Block *successor : branch->getSuccessors()) {
      if (successor->empty() ||
          !findBarriers(&successor->front(), plan, active)) {
        found = false;
        break;
      }
    }
  } else if (!op->hasTrait<OpTrait::IsTerminator>() && canCross(op)) {
    found = findBarriers(op->getNextNode(), plan, active);
  }
  active.erase(op);
  return found;
}

bool coveredByWaits(Operation *op,
                    const llvm::SmallPtrSetImpl<Operation *> &waits,
                    llvm::SmallPtrSetImpl<Operation *> &active);

// At a block boundary, every predecessor must be covered, including loop entry.
bool predecessorsCovered(Operation *op,
                         const llvm::SmallPtrSetImpl<Operation *> &waits,
                         llvm::SmallPtrSetImpl<Operation *> &active) {
  if (Operation *previous = op->getPrevNode())
    return coveredByWaits(previous, waits, active);
  Block *block = op->getBlock();
  if (block->hasNoPredecessors())
    return false;
  return llvm::all_of(block->getPredecessors(), [&](Block *pred) {
    return coveredByWaits(pred->getTerminator(), waits, active);
  });
}

// Reject paths whose original barrier might cover effects before a group wait.
bool coveredByWaits(Operation *op,
                    const llvm::SmallPtrSetImpl<Operation *> &waits,
                    llvm::SmallPtrSetImpl<Operation *> &active) {
  if (!op || !active.insert(op).second)
    return false;
  if (waits.contains(op)) {
    active.erase(op);
    return true;
  }
  bool covered = false;
  if (!(isa<gpu::BarrierOp>(op) || isCompletionBoundary(op)) &&
      (isa<BranchOpInterface>(op) || canCross(op))) {
    covered = predecessorsCovered(op, waits, active);
  }
  active.erase(op);
  return covered;
}

void scheduleBarriers(ModuleOp module) {
  SmallVector<WaitPlan> plans;
  module.walk([&](gpu::AsyncWaitOp wait) {
    // Branch operands count as uses: transfers may lower to register copies.
    Operation *point = nullptr;
    for (Operation *op = wait->getNextNode(); op; op = op->getNextNode()) {
      if (isa<gpu::BarrierOp>(op) || isCompletionBoundary(op))
        break;
      if (consumesGlobalRead(op)) {
        point = op;
        break;
      }
      if (!isa<BranchOpInterface>(op) && !canCross(op))
        break;
    }
    if (!point)
      return;
    WaitPlan plan{wait, point, {}};
    llvm::SmallPtrSet<Operation *, 32> active;
    if (findBarriers(wait->getNextNode(), plan, active))
      plans.push_back(std::move(plan));
  });

  // Plans sharing a destination barrier must move together. Otherwise moving
  // one path would leave the original barrier required by another path.
  // Treat plans and barriers as a bipartite graph. Index each edge once and
  // visit each barrier once per component instead of rescanning all plans.
  llvm::DenseMap<Operation *, SmallVector<unsigned>> barrierUsers;
  for (unsigned i = 0; i < plans.size(); ++i)
    for (Operation *barrier : plans[i].barriers)
      barrierUsers[barrier].push_back(i);

  SmallVector<bool> processed(plans.size(), false);
  for (unsigned seed = 0; seed < plans.size(); ++seed) {
    if (processed[seed])
      continue;
    SmallVector<unsigned> group{seed};
    llvm::SetVector<Operation *> barriers;
    processed[seed] = true;
    for (unsigned cursor = 0; cursor < group.size(); ++cursor) {
      for (Operation *barrier : plans[group[cursor]].barriers) {
        if (!barriers.insert(barrier))
          continue;
        for (unsigned index : barrierUsers.find(barrier)->second) {
          if (processed[index])
            continue;
          processed[index] = true;
          group.push_back(index);
        }
      }
    }
    llvm::SmallPtrSet<Operation *, 16> waits;
    for (unsigned index : group) {
      waits.insert(plans[index].wait);
    }
    bool covered = true;
    for (Operation *barrier : barriers) {
      llvm::SmallPtrSet<Operation *, 32> active;
      if (!predecessorsCovered(barrier, waits, active)) {
        covered = false;
        break;
      }
    }
    if (!covered)
      continue;
    for (unsigned index : group) {
      OpBuilder builder(plans[index].insertionPoint);
      gpu::BarrierOp::create(builder, plans[index].wait.getLoc(),
                             gpu::AddrSpace::Local);
    }
    for (Operation *barrier : barriers)
      barrier->erase();
  }
}

} // namespace

void scheduleMembarBarriers(ModuleOp module) { scheduleBarriers(module); }
} // namespace mlir::triton::AMD
