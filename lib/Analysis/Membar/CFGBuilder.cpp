#include "triton/Analysis/Membar/CFGBuilder.h"
#include "triton/Analysis/Membar/TmemModel.h"

#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "triton/Dialect/Triton/IR/Utility.h"
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

OpModel MembarCFGBuilder::describe(Operation *op, SMemModel &shared) {
  if (isa<CallOpInterface>(op))
    return {};
  if (isa<triton::ReturnOp>(op))
    return {.boundary = isKernel(op->getParentOfType<FunctionOpInterface>())
                            ? Boundary::KernelReturn
                            : Boundary::None};

  OpModel model;
  auto accesses = getMemoryAccesses(op);
  shared.describe(op, model, accesses);
  if (tmem)
    tmem->describe(op, model, accesses);
  return model;
}

void MembarCFGBuilder::addRegion(MembarCFG &cfg, Region &region) {
  auto function = region.getParentOfType<FunctionOpInterface>();
  SMemModel &shared = models.at(function);
  const auto &indices = shared.getBufferIndexAnalysis();
  unsigned body = cfg.nodes.size();
  // A helper called per element cannot execute a collective.
  bool callback = callbacks.contains(function);
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
    unsigned cost =
        (arm || callback) ? 0 : 1u << (2 * std::min(depth, maxDepth));
    for (Operation &op : block.without_terminator()) {
      OpModel effects = describe(&op, shared);
      if (effects == OpModel{} && !isa<CallOpInterface>(&op))
        continue;
      unsigned next = cfg.nodes.size() + 1;
      cfg.nodes.push_back({&op, cost, body, std::move(effects), {{next}}});
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
        OpModel inner = describe(op, shared);
        effects.sharedIn.join(inner.sharedIn);
        effects.sharedOut.join(inner.sharedOut);
        if (inner.boundary == Boundary::Publication)
          effects.boundary = Boundary::Publication;
      });
      exit = active->getTerminator();
    } else {
      effects = describe(term, shared);
    }
    cfg.nodes.push_back({term, cost, body, std::move(effects), {}});
    exits.emplace_back(cfg.nodes.size() - 1, exit);
  }

  for (auto [id, term] : exits)
    for (Block *succ : term->getSuccessors())
      cfg.nodes[id].next.push_back(
          {entries.at(succ), indices.isBackedgeSuccessor(term, succ)});
}

MembarCFG MembarCFGBuilder::build(Region &root) {
  struct Callee {
    unsigned entry;
    SmallVector<Edge, 2> continuations;
  };
  DenseMap<Region *, Callee> callees;
  callees.try_emplace(&root);
  MembarCFG cfg;
  addRegion(cfg, root);

  // Append each reachable callee once, in first-call order.
  for (unsigned id = 0; id < cfg.nodes.size(); ++id) {
    auto call = dyn_cast<CallOpInterface>(cfg.nodes[id].op);
    if (!call)
      continue;
    Region &region =
        cast<FunctionOpInterface>(call.resolveCallable()).getFunctionBody();
    auto [it, inserted] = callees.try_emplace(&region);
    auto &callee = it->second;
    if (inserted) {
      callee.entry = cfg.nodes.size();
      addRegion(cfg, region);
    }
    Edge &next = cfg.nodes[id].next.front();
    callee.continuations.push_back(next);
    next = {callee.entry, true};
  }
  for (Node &node : cfg.nodes)
    if (isa<triton::ReturnOp>(node.op))
      node.next = callees.at(node.op->getParentRegion()).continuations;
  return cfg;
}
