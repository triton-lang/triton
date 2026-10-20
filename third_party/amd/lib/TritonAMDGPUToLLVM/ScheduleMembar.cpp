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

// Follow only direct producers and at most one incoming CFG edge; this is a
// latency heuristic, not an analysis of outstanding hardware memory requests.
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

// Advancing a loaded value while preserving the previous value in another
// loop argument may require a copy before reusing its register. Recognize the
// argument rotation, not physical registers, which are assigned later.
bool rotatesGlobalRead(BranchOpInterface branch) {
  for (unsigned edge = 0; edge < branch->getNumSuccessors(); ++edge) {
    Block *header = branch->getSuccessor(edge);
    auto incoming = branch.getSuccessorOperands(edge);
    for (Block *pred : header->getPredecessors()) {
      auto backedge = dyn_cast<BranchOpInterface>(pred->getTerminator());
      if (!backedge)
        continue;
      for (unsigned i = 0; i < backedge->getNumSuccessors(); ++i) {
        if (backedge->getSuccessor(i) != header)
          continue;
        auto operands = backedge.getSuccessorOperands(i);
        for (unsigned dest = 0; dest < operands.size(); ++dest) {
          if (operands.isOperandProduced(dest))
            continue;
          auto previous = dyn_cast<BlockArgument>(operands[dest]);
          if (!previous || previous.getOwner() != header ||
              previous.getArgNumber() == dest)
            continue;
          unsigned source = previous.getArgNumber();
          if (source < incoming.size() &&
              !incoming.isOperandProduced(source) &&
              isGlobalReadResult(incoming[source]))
            return true;
        }
      }
    }
  }
  return false;
}

bool consumesGlobalRead(Operation *op) {
  auto branch = dyn_cast<BranchOpInterface>(op);
  if (branch && rotatesGlobalRead(branch))
    return true;
  for (OpOperand &operand : op->getOpOperands()) {
    // Forwarding a value does not necessarily emit a register copy. Count
    // only actual uses, including a branch condition.
    bool forwarded = false;
    if (branch) {
      for (unsigned i = 0; i < branch->getNumSuccessors(); ++i) {
        OperandRange operands =
            branch.getSuccessorOperands(i).getForwardedOperands();
        if (!operands.empty() &&
            operand.getOperandNumber() >= operands.getBeginOperandIndex() &&
            operand.getOperandNumber() <
                operands.getBeginOperandIndex() + operands.size()) {
          forwarded = true;
          break;
        }
      }
    }
    if (forwarded)
      continue;
    if (isGlobalReadResult(operand.get()))
      return true;
  }
  return false;
}

bool canCross(Operation *op) {
  return op->getNumRegions() == 0 &&
         (isMemoryEffectFree(op) || isGlobalRead(op));
}

bool isCompletionBoundary(Operation *op) {
  // Crossing a TDM or non-TDM wait could change which completion a later
  // barrier synchronizes.
  return isa<gpu::AsyncWaitOp, gpu::AsyncCommitGroupOp>(op) ||
         op->hasTrait<OpTrait::MemWaitOpTrait>();
}

bool isLocalBarrier(Operation *op) {
  auto barrier = dyn_cast<gpu::BarrierOp>(op);
  return barrier && !barrier.isWarp() &&
         barrier.getAddrSpace() == gpu::AddrSpace::Local;
}

// For one wait, record where to insert a barrier and which existing barriers
// it could replace.
struct BarrierMoveCandidate {
  gpu::AsyncWaitOp wait;
  Operation *insertBefore;
  llvm::SetVector<Operation *> barriersToReplace;
};

// Find the first full local barrier on every path. Crossing shared effects
// could invalidate synchronization that Membar relied on for other accesses.
bool findBarriersToReplace(Operation *op, BarrierMoveCandidate &candidate,
                           llvm::SmallPtrSetImpl<Operation *> &activePath) {
  if (!op || !activePath.insert(op).second)
    return false;
  bool found = false;
  if (isLocalBarrier(op)) {
    candidate.barriersToReplace.insert(op);
    found = true;
  } else if (isa<gpu::BarrierOp>(op) || isCompletionBoundary(op)) {
    found = false;
  } else if (auto branch = dyn_cast<BranchOpInterface>(op)) {
    found = branch->getNumSuccessors() != 0;
    for (Block *successor : branch->getSuccessors()) {
      if (successor->empty() ||
          !findBarriersToReplace(&successor->front(), candidate, activePath)) {
        found = false;
        break;
      }
    }
  } else if (!op->hasTrait<OpTrait::IsTerminator>() && canCross(op)) {
    found = findBarriersToReplace(op->getNextNode(), candidate, activePath);
  }
  activePath.erase(op);
  return found;
}

bool coveredByWaits(Operation *op,
                    const llvm::SmallPtrSetImpl<Operation *> &waits,
                    llvm::SmallPtrSetImpl<Operation *> &activePath);

// At a block boundary, check every predecessor, including loop entry.
bool predecessorsCovered(Operation *op,
                         const llvm::SmallPtrSetImpl<Operation *> &waits,
                         llvm::SmallPtrSetImpl<Operation *> &activePath) {
  if (Operation *previous = op->getPrevNode())
    return coveredByWaits(previous, waits, activePath);
  Block *block = op->getBlock();
  if (block->hasNoPredecessors())
    return false;
  return llvm::all_of(block->getPredecessors(), [&](Block *pred) {
    return coveredByWaits(pred->getTerminator(), waits, activePath);
  });
}

// A removed barrier must be preceded by a group wait on every incoming path.
// Other effects or synchronization between them may still need that barrier.
bool coveredByWaits(Operation *op,
                    const llvm::SmallPtrSetImpl<Operation *> &waits,
                    llvm::SmallPtrSetImpl<Operation *> &activePath) {
  if (!op || !activePath.insert(op).second)
    return false;
  if (waits.contains(op)) {
    activePath.erase(op);
    return true;
  }
  bool covered = false;
  if (!(isa<gpu::BarrierOp>(op) || isCompletionBoundary(op)) &&
      (isa<BranchOpInterface>(op) || canCross(op))) {
    covered = predecessorsCovered(op, waits, activePath);
  }
  activePath.erase(op);
  return covered;
}

// Follow branch transfers without treating them as uses. All paths must reach
// the same first use so one new barrier can replace the deferred barriers.
// Stop at cycles rather than scheduling across another loop iteration.
Operation *findFirstGlobalReadUse(
    Operation *start, llvm::SmallPtrSetImpl<Operation *> &activePath) {
  if (!start || !activePath.insert(start).second)
    return nullptr;
  Operation *firstUse = nullptr;
  for (Operation *op = start; op; op = op->getNextNode()) {
    if (isa<gpu::BarrierOp>(op) || isCompletionBoundary(op))
      break;
    if (consumesGlobalRead(op)) {
      firstUse = op;
      break;
    }
    if (auto branch = dyn_cast<BranchOpInterface>(op)) {
      for (Block *successor : branch->getSuccessors()) {
        Operation *use =
            successor->empty()
                ? nullptr
                : findFirstGlobalReadUse(&successor->front(), activePath);
        if (!use || (firstUse && firstUse != use)) {
          firstUse = nullptr;
          break;
        }
        firstUse = use;
      }
      break;
    }
    if (op->hasTrait<OpTrait::IsTerminator>() || !canCross(op))
      break;
  }
  activePath.erase(start);
  return firstUse;
}

SmallVector<BarrierMoveCandidate> collectCandidates(ModuleOp module) {
  SmallVector<BarrierMoveCandidate> candidates;
  module.walk([&](gpu::AsyncWaitOp wait) {
    llvm::SmallPtrSet<Operation *, 32> activePath;
    Operation *firstUse =
        findFirstGlobalReadUse(wait->getNextNode(), activePath);
    if (!firstUse)
      return;
    BarrierMoveCandidate candidate{wait, firstUse, {}};
    if (findBarriersToReplace(wait->getNextNode(), candidate, activePath))
      candidates.push_back(std::move(candidate));
  });
  return candidates;
}

struct BarrierMoveGroup {
  SmallVector<unsigned> candidates;
  llvm::SetVector<Operation *> barriersToReplace;
};

// Candidates are connected when they would replace the same barrier.
// Process each connected group together so no path loses a shared barrier.
BarrierMoveGroup
collectConnectedGroup(unsigned seed, ArrayRef<BarrierMoveCandidate> candidates,
                      const llvm::DenseMap<Operation *, SmallVector<unsigned>>
                          &candidatesByBarrier,
                      SmallVector<bool> &grouped) {
  BarrierMoveGroup group;
  group.candidates.push_back(seed);
  grouped[seed] = true;

  // Visit candidates appended to the group as well as the first one.
  for (unsigned cursor = 0; cursor < group.candidates.size(); ++cursor) {
    for (Operation *barrier :
         candidates[group.candidates[cursor]].barriersToReplace) {
      // We have already found the candidates for this barrier.
      if (!group.barriersToReplace.insert(barrier))
        continue;

      // All candidates using this barrier must move together.
      for (unsigned index : candidatesByBarrier.find(barrier)->second) {
        if (grouped[index])
          continue;
        grouped[index] = true;
        group.candidates.push_back(index);
      }
    }
  }
  return group;
}

bool canReplaceBarriers(const BarrierMoveGroup &group,
                        ArrayRef<BarrierMoveCandidate> candidates) {
  llvm::SmallPtrSet<Operation *, 16> waits;
  for (unsigned index : group.candidates)
    waits.insert(candidates[index].wait);
  for (Operation *barrier : group.barriersToReplace) {
    llvm::SmallPtrSet<Operation *, 32> activePath;
    if (!predecessorsCovered(barrier, waits, activePath))
      return false;
  }
  return true;
}

void moveBarriers(const BarrierMoveGroup &group,
                  ArrayRef<BarrierMoveCandidate> candidates) {
  // Put one new barrier at each wait's selected use, then remove the old
  // ones.
  for (unsigned index : group.candidates) {
    OpBuilder builder(candidates[index].insertBefore);
    gpu::BarrierOp::create(builder, candidates[index].wait->getLoc(),
                           gpu::AddrSpace::Local);
  }
  for (Operation *barrier : group.barriersToReplace)
    barrier->erase();
}

} // namespace

// A use of a global-load result may trigger a long hardware memory wait. Move
// an eligible rendezvous before that use so other warps can reach it sooner.
// This is a scheduling heuristic: TTGIR does not show where LLVM will place
// its memory waits.
void scheduleMembarBarriers(ModuleOp module) {
  SmallVector<BarrierMoveCandidate> candidates = collectCandidates(module);
  llvm::DenseMap<Operation *, SmallVector<unsigned>> candidatesByBarrier;
  for (unsigned i = 0; i < candidates.size(); ++i)
    for (Operation *barrier : candidates[i].barriersToReplace)
      candidatesByBarrier[barrier].push_back(i);

  SmallVector<bool> grouped(candidates.size(), false);
  for (unsigned seed = 0; seed < candidates.size(); ++seed) {
    if (grouped[seed])
      continue;
    BarrierMoveGroup group =
        collectConnectedGroup(seed, candidates, candidatesByBarrier, grouped);
    if (canReplaceBarriers(group, candidates))
      moveBarriers(group, candidates);
  }
}

} // namespace mlir::triton::AMD
