#include "triton/Analysis/Membar/CFGBuilder.h"
#include "triton/Analysis/Membar/AsyncFenceModel.h"
#include "triton/Analysis/Membar/ClusterBarrierModel.h"
#include "triton/Analysis/Membar/TmemModel.h"

#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <limits>
#include <utility>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::membar;
using mlir::presburger::Fraction;
namespace ttg = mlir::triton::gpu;

template <typename Model>
MembarCFGBuilder<Model>::MembarCFGBuilder(Model &model, ModuleOp module)
    : model(model), callbacks(getTensorCallbackOps(module)) {
  SymbolTableCollection symbols;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa_and_nonnull<scf::SCFDialect>(op->getDialect()))
      llvm::report_fatal_error("unexpected " + op->getName().getStringRef() +
                               " in Membar graph placement. Run "
                               "--convert-scf-to-cf first.");
    if (auto call = dyn_cast<CallOpInterface>(op))
      callers[call.resolveCallableInTable(&symbols)].push_back(op->getBlock());
  });
}

template <typename Model>
auto MembarCFGBuilder<Model>::getBlockWeight(Block *block) -> BlockWeight {
  if (auto it = blockWeights.find(block); it != blockWeights.end())
    return it->second;

  // Every role executes once per launch. Sum the frequencies of all calls
  // to a helper and keep its deepest calling loop for the loop tie-break.
  Region *region = block->getParent();
  Operation *parent = region->getParentOp();
  BlockWeight entry{1, 0};
  if (auto it = callers.find(parent); it != callers.end()) {
    entry = {};
    for (Block *caller : it->second) {
      BlockWeight call = getBlockWeight(caller);
      entry.cost += call.cost;
      entry.loopDepth = std::max(entry.loopDepth, call.loopDepth);
    }
  } else if (!isa<FunctionOpInterface>(parent)) {
    entry = getBlockWeight(parent->getBlock());
  }

  auto fn = region->getParentOfType<FunctionOpInterface>();
  const auto *loops = model.getIndices(fn).getLoopInfo(region);
  DenseMap<Block *, Fraction> incoming;
  incoming[&region->front()] = entry.cost;
  // Split the incoming weight across mutually exclusive successors and add
  // their shares at joins. Price each block once, before backedges contribute.
  for (Block *b : llvm::ReversePostOrderTraversal<Region *>(region)) {
    auto *loop = loops ? loops->getLoopFor(b) : nullptr;
    unsigned depth = loop ? loop->getLoopDepth() : 0;
    Fraction share = incoming[b];
    Fraction cost = share;
    for (unsigned i = 0; i < depth; ++i)
      cost *= 4;
    blockWeights[b] = {cost, entry.loopDepth + depth};
    // The loop body and its continuation can both execute. The depth factor
    // already accounts for their relative frequency.
    if (b->getNumSuccessors() > 1 && !(loop && loop->isLoopExiting(b)))
      share /= b->getNumSuccessors();
    for (Block *succ : b->getSuccessors())
      incoming[succ] += share;
  }
  return blockWeights.try_emplace(block, entry).first->second;
}

template <typename Model> void MembarCFGBuilder<Model>::normalizeCosts() {
  // The GCD of reduced fractions is gcd(numerators) / lcm(denominators).
  Fraction unit, peak;
  for (Block *block : llvm::make_first_range(insertionCosts)) {
    const Fraction &cost = blockWeights.at(block).cost;
    unit = {gcd(unit.num, cost.num), lcm(unit.den, cost.den)};
    peak = std::max(peak, cost);
  }
  // If the smallest integer ratios overflow, discard low bits uniformly.
  // Rounding up keeps every legal site positive.
  while (unit * std::numeric_limits<unsigned>::max() < peak)
    unit *= 2;
  for (auto &[block, cost] : insertionCosts)
    cost = int64_t(ceil(blockWeights.at(block).cost / unit));
}

static Block *getGuardArm(Operation *op) {
  auto split = dyn_cast<cf::CondBranchOp>(op);
  if (split && split.getCondition().getDefiningOp<ttg::ThreadPredicateOp>())
    return split.getTrueDest();
  return nullptr;
}

// Compare the entry guard with the loop test on its initial operands.
static bool firstLoopTestIsTrue(cf::BranchOp enter, cf::CondBranchOp header) {
  Block *preheader = enter->getBlock();
  auto *pred = preheader->getSinglePredecessor();
  if (!pred)
    return false;
  auto guard = dyn_cast<cf::CondBranchOp>(pred->getTerminator());
  if (!guard)
    return false;
  auto known = guard.getCondition().getDefiningOp<arith::CmpIOp>();
  auto test = header.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!known || !test)
    return false;
  auto condition = known.getPredicate();
  if (guard.getFalseDest() == preheader)
    condition = arith::invertPredicate(condition);
  if (test.getPredicate() != condition)
    return false;
  auto initial = [&](Value v) -> Value {
    if (auto arg = dyn_cast<BlockArgument>(v);
        arg && arg.getOwner() == enter.getDest())
      return enter.getDestOperands()[arg.getArgNumber()];
    return v;
  };
  Value lhs = initial(test.getLhs()), rhs = initial(test.getRhs());
  // A loop can widen both bounds while preserving their ordering.
  if (lhs.getType() != known.getLhs().getType()) {
    auto a = lhs.getDefiningOp<arith::ExtSIOp>();
    auto b = rhs.getDefiningOp<arith::ExtSIOp>();
    if (!a || !b)
      return false;
    lhs = a.getIn();
    rhs = b.getIn();
  }
  return lhs == known.getLhs() && rhs == known.getRhs();
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
        effects.shared.merge(arm->shared);
        if (arm->boundary == Boundary::Publication)
          effects.boundary = Boundary::Publication;
      }
    });
    return effects;
  }
  auto function = op->getParentOfType<FunctionOpInterface>();
  if (isa<triton::ReturnOp>(op)) {
    if (isKernel(function)) {
      effects.boundary = Boundary::KernelReturn;
      effects.shared.threads.joins = ThreadEffects::FullCompletion;
    }
    return effects;
  }

  auto accesses = getMemoryAccesses(op);
  effects.shared = models.at(function).describe(op, fixed, accesses);
  if (auto barrier = dyn_cast<ttg::MBarrierOpInterface>(op);
      barrier && barrier.isPerWarp())
    effects.arrivalMode = ArrivalMode::PerWarp;
  if (tmem)
    tmem->describe(op, effects, fixed, accesses);
  if (effects == OpModel{})
    return std::nullopt;
  return effects;
}

template <typename Model>
void MembarCFGBuilder<Model>::addRegion(MembarCFG<Facts> &cfg, Region &region) {
  auto function = region.getParentOfType<FunctionOpInterface>();
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
    unsigned depth = 0;
    if (!arm && !callback) {
      depth = getBlockWeight(&block).loopDepth;
      insertionCosts.try_emplace(&block);
    }
    Operation *term = block.getTerminator();
    Block *active = getGuardArm(term);
    for (Operation &op : block) {
      // A disabled wait only keeps its dependent allocations live.
      if (auto wait = dyn_cast<nvidia_gpu::WaitBarrierOp>(&op);
          wait && wait.getPred() && matchPattern(wait.getPred(), m_Zero()))
        continue;
      FixedSync fixed;
      // A call links to the callee's effects, including its scratch.
      auto effects =
          isa<CallOpInterface>(&op)
              ? Facts{}
              : model.describe(&op, fixed, &op == term ? active : nullptr);
      if (&op != term && !effects && fixed == FixedSync{} &&
          !isa<ttg::WarpSpecializeOp>(&op))
        continue;
      unsigned id = cfg.nodes.size();
      cfg.nodes.push_back({&op,
                           0,
                           depth,
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

  // Split a branch-only header on entry when the first test must pass.
  // Both copies share the same insertion gap. Backedges keep both outcomes.
  const auto *loops = indices.getLoopInfo(&region);
  if (!loops)
    return;
  for (auto [id, term] : exits) {
    auto enter = dyn_cast<cf::BranchOp>(term);
    if (!enter || cfg.nodes[id].op != term)
      continue;
    Block *dest = enter.getDest();
    auto *loop = loops->getLoopFor(dest);
    if (!loop || loop->getHeader() != dest ||
        loop->getLoopPredecessor() != enter->getBlock())
      continue;
    const auto &first = cfg.nodes[entries.at(dest)];
    auto header = dyn_cast<cf::CondBranchOp>(first.op);
    if (!header || first.next.size() != 2 ||
        !firstLoopTestIsTrue(enter, header))
      continue;
    auto entry = first;
    entry.next.resize(1);
    cfg.nodes[id].next.front().to = cfg.nodes.size();
    cfg.nodes.push_back(std::move(entry));
  }
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
  insertionCosts.clear();
  appendWithCallees(cfg, root);
  if (!cfg.nodes.empty())
    cfg.multiWarp = ttg::maybeLookupNumWarps(cfg.nodes.front().op) != 1;
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

  normalizeCosts();
  DenseMap<Operation *, unsigned> gapIds;
  for (auto [id, node] : llvm::enumerate(cfg.nodes)) {
    node.insertionCost = insertionCosts.lookup(node.op->getBlock());
    node.gap = gapIds.try_emplace(node.op, id).first->second;
  }
  return cfg;
}

template class mlir::triton::membar::MembarCFGBuilder<MembarModel>;
template class mlir::triton::membar::MembarCFGBuilder<AsyncFenceModel>;
template class mlir::triton::membar::MembarCFGBuilder<ClusterBarrierModel>;
