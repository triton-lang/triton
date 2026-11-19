#include "triton/Analysis/Membar/CFGBuilder.h"
#include "triton/Analysis/Membar/AsyncFenceModel.h"
#include "triton/Analysis/Membar/TmemModel.h"

#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <utility>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::membar;
namespace ttg = mlir::triton::gpu;

// Include loops enclosing a warp role or a compiler-generated thread scope.
unsigned
mlir::triton::membar::getBlockWeight(Block *block,
                                     const BufferIndexAnalysis &indices,
                                     unsigned invocationWeight) {
  unsigned depth = 0;
  for (;;) {
    if (auto *loops = indices.getLoopInfo(block->getParent()))
      depth += loops->getLoopDepth(block);
    Operation *parent = block->getParentOp();
    if (isa<FunctionOpInterface>(parent))
      break;
    block = parent->getBlock();
  }
  constexpr unsigned maxDepth = (std::numeric_limits<unsigned>::digits - 1) / 2;
  unsigned loopWeight = 1u << (2 * std::min(depth, maxDepth));
  bool overflow;
  unsigned weight =
      llvm::SaturatingMultiply(invocationWeight, loopWeight, &overflow);
  assert(depth <= maxDepth && !overflow && "Membar block weight overflow");
  return weight;
}

static Block *getGuardArm(Operation *op) {
  auto split = dyn_cast<cf::CondBranchOp>(op);
  if (split && split.getCondition().getDefiningOp<ttg::ThreadPredicateOp>())
    return split.getTrueDest();
  return nullptr;
}

const BufferIndexAnalysis &
MembarModel::getIndices(FunctionOpInterface function) {
  return models.at(function).getBufferIndexAnalysis();
}

std::optional<OpModel> MembarModel::describe(Operation *op, FixedSync &fixed,
                                             Block *active) {
  OpModel effects;
  if (active) {
    // Join pending waits before lanes split. Synchronization in the active arm
    // cannot order the whole group.
    active->walk([&](Operation *inner) {
      if (isa<ttg::ThreadScopeOp>(inner))
        return;
      FixedSync armFixed;
      if (auto arm = describe(inner, armFixed)) {
        effects.sharedIn.join(arm->sharedIn);
        effects.sharedOut.join(arm->sharedOut);
        if (arm->boundary == Boundary::Publication)
          effects.boundary = Boundary::Publication;
      }
    });
    return effects;
  }
  if (isa<CallOpInterface>(op))
    return std::nullopt;
  auto function = op->getParentOfType<FunctionOpInterface>();
  if (isa<triton::ReturnOp>(op)) {
    if (isKernel(function))
      effects.boundary = Boundary::KernelReturn;
    return effects;
  }

  auto accesses = getMemoryAccesses(op);
  models.at(function).describe(op, effects, fixed, accesses);
  if (tmem)
    tmem->describe(op, effects, fixed, accesses);
  if (effects == OpModel{})
    return std::nullopt;
  return effects;
}

template <typename Model>
void MembarCFGBuilder<Model>::addRegion(MembarCFG<Facts> &cfg, Region &region) {
  auto function = region.getParentOfType<FunctionOpInterface>();
  unsigned invocationWeight = calleeWeights.lookup_or(function, 1u);
  const auto &indices = model.getIndices(function);
  unsigned body = cfg.nodes.size();
  // A helper called per element cannot execute a collective.
  bool callback = callbacks.contains(function);
  DenseMap<Block *, unsigned> entries;
  SmallVector<std::pair<unsigned, Operation *>> exits;
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
    unsigned cost = (arm || callback)
                        ? 0
                        : getBlockWeight(&block, indices, invocationWeight);
    Operation *term = block.getTerminator();
    Block *active = getGuardArm(term);
    for (Operation &op : block) {
      FixedSync fixed;
      auto effects = model.describe(&op, fixed, &op == term ? active : nullptr);
      if (&op != term && !effects && fixed == FixedSync{} &&
          !isa<CallOpInterface>(&op))
        continue;
      unsigned id = cfg.nodes.size();
      cfg.nodes.push_back({&op,
                           cost,
                           body,
                           fixed,
                           effects ? std::move(*effects) : Facts{},
                           {}});
      if (&op == term)
        exits.emplace_back(id, active ? active->getTerminator() : term);
      else
        cfg.nodes.back().next.push_back({id + 1});
    }
  }

  for (auto [id, term] : exits)
    for (Block *succ : term->getSuccessors())
      cfg.nodes[id].next.push_back(
          {entries.at(succ), indices.isBackedgeSuccessor(term, succ)});
}

template <typename Model>
void MembarCFGBuilder<Model>::appendWithCallees(MembarCFG<Facts> &cfg,
                                                Region &root,
                                                std::optional<Edge> roleExit) {
  struct Callee {
    unsigned entry;
    SmallVector<Edge, 2> continuations;
  };
  DenseMap<Region *, Callee> callees;
  unsigned entry = cfg.nodes.size();
  callees.try_emplace(&root, Callee{entry, {}});
  addRegion(cfg, root);

  // Append each callee once per role so its returns stay in the calling role.
  for (unsigned id = entry; id < cfg.nodes.size(); ++id) {
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
  for (auto &node : MutableArrayRef(cfg.nodes).drop_front(entry)) {
    node.inWorker = roleExit && roleExit->warp == WarpEdge::ExitWorker;
    if (isa<triton::ReturnOp>(node.op))
      node.next = callees.at(node.op->getParentRegion()).continuations;
    else if (roleExit && isa<ttg::WarpYieldOp, ttg::WarpReturnOp>(node.op))
      node.next.push_back(*roleExit);
  }
}

template <typename Model>
auto MembarCFGBuilder<Model>::build(Region &root, bool expandWarpRoles)
    -> MembarCFG<Facts> {
  MembarCFG<Facts> cfg;
  appendWithCallees(cfg, root);
  if (expandWarpRoles) {
    for (unsigned id = 0, end = cfg.nodes.size(); id < end; ++id) {
      auto ws = dyn_cast<ttg::WarpSpecializeOp>(cfg.nodes[id].op);
      if (!ws)
        continue;
      SmallVector<Region *> roles{&ws.getDefaultRegion()};
      llvm::append_range(roles, ws.getPartitionRegions());
      // The launch is a non-terminator with one fallthrough edge.
      Edge continuation = cfg.nodes[id].next.pop_back_val();
      for (Region *region : roles) {
        bool worker = region != roles.front();
        unsigned entry = cfg.nodes.size();
        cfg.nodes[id].next.push_back(
            {entry, false, worker ? WarpEdge::EnterWorker : WarpEdge::None});
        continuation.warp = worker ? WarpEdge::ExitWorker : WarpEdge::None;
        appendWithCallees(cfg, *region, continuation);
      }
    }
  }

  DenseMap<Operation *, unsigned> gapIds;
  for (auto [id, node] : llvm::enumerate(cfg.nodes))
    node.gap = gapIds.try_emplace(node.op, id).first->second;
  return cfg;
}

template class mlir::triton::membar::MembarCFGBuilder<MembarModel>;
template class mlir::triton::membar::MembarCFGBuilder<AsyncFenceModel>;
