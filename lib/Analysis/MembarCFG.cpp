#include "MembarCFG.h"

#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <utility>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::membar;
namespace ttg = mlir::triton::gpu;

static Block *getGuardArm(Operation *op) {
  auto split = dyn_cast<cf::CondBranchOp>(op);
  if (split && split.getCondition().getDefiningOp<ttg::ThreadPredicateOp>())
    return split.getTrueDest();
  return nullptr;
}

OpModel PlacementState::describe(Operation *op) {
  if (isa<CallOpInterface>(op))
    return {.boundary = Boundary::Clean};
  if (isa<triton::ReturnOp>(op))
    return {.boundary = returnBoundary};

  OpModel model;
  auto accesses = getMemoryAccesses(op);
  shared.describe(op, model, accesses);
  if (tmem)
    tmem->describe(op, model, accesses);
  return model;
}

MembarCFG PlacementState::buildMembarCFG(Region &region) {
  const auto &indices = shared.getBufferIndexAnalysis();
  MembarCFG cfg{};
  auto &entry = cfg.nodes.emplace_back();
  entry.next.push_back({1}); // The first block retains at least its terminator.
  if (isa<ttg::WarpSpecializeOp, ttg::WarpSpecializePartitionsOp>(
          region.getParentOp()))
    entry.effects.fixedAfter = fullBarrier;

  DenseMap<Block *, unsigned> entries;
  SmallVector<std::pair<unsigned, Operation *>> exits;
  const auto *loops = indices.getLoopInfo(&region);
  constexpr unsigned maxDepth = (std::numeric_limits<unsigned>::digits - 1) / 2;
  // Build the CFG for one execution region:
  //   - Insertion costs on retained ops
  //   - One node per retained op via `describe`
  //   - Numbers per node (to break equal-cost ties)
  for (Block &block : region) {
    // A merged guard arm can precede its branch in IR order. The guard's
    // incoming edge restricts the arm, but leaves the common join unrestricted.
    bool arm = llvm::any_of(block.getUsers(), [&](Operation *branch) {
      Block *active = getGuardArm(branch);
      return active && active->getSuccessor(0) != &block;
    });
    if (arm && block.getSinglePredecessor())
      continue;
    entries.try_emplace(&block, cfg.nodes.size());
    unsigned depth = loops ? loops->getLoopDepth(&block) : 0;
    unsigned cost = arm ? 0 : 1u << (2 * std::min(depth, maxDepth));
    for (Operation &op : block.without_terminator()) {
      OpModel effects = describe(&op);
      if (effects == OpModel{})
        continue;
      unsigned next = cfg.nodes.size() + 1;
      cfg.nodes.push_back({&op, cost, std::move(effects), {{next}}});
    }

    Operation *term = block.getTerminator();
    Operation *exit = term;
    OpModel effects;
    if (Block *active = getGuardArm(term)) {
      // A guarded notification must join pending waits before lanes split.
      // The branch is the collective insertion point. Only the active arm has
      // memory effects, and its synchronization cannot order the whole group.
      active->walk([&](Operation *op) {
        if (isa<ttg::ThreadScopeOp>(op))
          return;
        OpModel inner = describe(op);
        effects.sharedIn.join(inner.sharedIn);
        effects.sharedOut.join(inner.sharedOut);
        if (inner.boundary == Boundary::Publication)
          effects.boundary = Boundary::Publication;
      });
      exit = active->getTerminator();
    } else {
      effects = describe(term);
    }
    cfg.nodes.push_back({term, cost, std::move(effects), {}});
    exits.emplace_back(cfg.nodes.size() - 1, exit);
  }

  for (auto [id, term] : exits)
    for (Block *succ : term->getSuccessors())
      cfg.nodes[id].next.push_back(
          {entries.at(succ), indices.isBackedgeSuccessor(term, succ)});
  return cfg;
}
