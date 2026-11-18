#include "triton/Analysis/Membar/ClusterBarrierModel.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"

namespace mlir::triton::membar {
namespace {

using ClusterNode = Node<ClusterBarrierFacts>;
using ClusterCFG = MembarCFG<ClusterBarrierFacts>;
constexpr unsigned cluster = coverage(Action::ClusterBarrier);
constexpr unsigned live = 1, reallocated = 2, carried = 4;

bool overlapsWithinAllocation(const SharedAccess &src,
                              const SharedAccess &dst) {
  return mayOverlap(src.footprint, dst.footprint, AllocationOrigin::Same);
}

// Impure assembly can wait for a transfer or signal a sibling.
bool hasOpaqueSync(Operation *op) {
  return isa<gpu::InlineAsmOp, ElementwiseInlineAsmOp>(op) &&
         !isMemoryEffectFree(op);
}

bool conflicts(const SharedAccess &src, const SharedAccess &dst,
               bool reallocated) {
  // The programmer orders accesses within an explicit allocation's
  // lifetime. The compiler orders reuse, including implicit scratch.
  return (src.crossCTA || dst.crossCTA) && (src.isWrite || dst.isWrite) &&
         mayOverlap(src.footprint, dst.footprint,
                    reallocated ? AllocationOrigin::Any
                                : AllocationOrigin::Different);
}

bool initializesPeerCounter(const ClusterCFG &cfg,
                            const ClusterNode &producer) {
  if (!isa<nvidia_gpu::InitBarrierOp>(producer.op))
    return false;
  const auto &slot = producer.effects.accesses.front();
  for (const auto &node : cfg.nodes)
    for (const auto &access : node.effects.accesses)
      if (access.crossCTA && overlapsWithinAllocation(slot, access))
        return true;
  return false;
}

enum class Completion { None, Possible, Complete };

Completion getCompletion(const ClusterNode &issuer, const ClusterNode &node) {
  using namespace nvidia_gpu;
  Operation *op = node.op;
  // A sibling can relay completion through a different counter or at join.
  if (hasOpaqueSync(op) || isa<gpu::WarpYieldOp, gpu::WarpReturnOp>(op) ||
      (node.body && isa<WaitBarrierOp>(op)))
    return Completion::Possible;
  if (isa<TMAStoreLikeOpInterface>(issuer.op)) {
    if (auto wait = dyn_cast<TMAStoreWaitOp>(op))
      return wait.getPendings() ? Completion::Possible : Completion::Complete;
  } else if (isa<WarpGroupDotOp>(issuer.op)) {
    if (auto wait = dyn_cast<WarpGroupDotWaitOp>(op))
      return wait.getPendings() ? Completion::Possible : Completion::Complete;
    if (auto dot = dyn_cast<WarpGroupDotOp>(op); dot && !dot.getIsAsync())
      return Completion::Complete;
  } else if (isa<WaitBarrierOp>(op)) {
    // TMA loads, CLC and st.async name the counter on the payload write.
    // Copies and tcgen05 operations can signal a later commit's counter.
    if (!isa<TMALoadLikeOpInterface, CLCTryCancelOp, AsyncSharedStoreOp>(
            issuer.op) ||
        llvm::any_of(issuer.effects.accesses, [&](const auto &access) {
          return overlapsWithinAllocation(node.effects.accesses.front(),
                                          access);
        }))
      return Completion::Possible;
  } else if (isa<gpu::AsyncWaitOp>(op) &&
             isa<gpu::AsyncCopyGlobalToLocalOp>(issuer.op))
    return Completion::Possible;
  return Completion::None;
}

unsigned advanceLifetime(unsigned state, Operation *op,
                         const SharedAccess &access) {
  if (access.footprint)
    for (const auto &view : access.footprint->regionInfo.views)
      if (view.allocation == op)
        return state | reallocated;
  return state;
}

// A cluster rendezvous orders synchronous accesses. For async accesses the
// rendezvous must follow completion, so start at each possible observation too.
void recordCompletions(ClusterCFG &cfg) {
  for (const auto &producer : cfg.nodes) {
    if (!producer.effects.async)
      continue;
    for (auto [access, state] : producer.effects.sources)
      walkEdges(
          cfg, producer, state,
          [&](unsigned point, unsigned state,
              bool canExitWorker) -> std::optional<unsigned> {
            auto &node = cfg.nodes[point];
            bool sameIssuer = producer.inWorker == node.inWorker &&
                              (!node.inWorker || canExitWorker);
            // A valid synchronous read of the written descriptor requires
            // completion. Rebinding the descriptor can change its origin.
            if (auto load = dyn_cast<gpu::LocalLoadOp>(node.op);
                load && !(state & carried) &&
                load.getSrc() == producer.effects.payloadWrite)
              return std::nullopt;
            auto completion = getCompletion(producer, node);
            // The path starts here, still referring to the issuer's allocation.
            auto effect = std::pair{access, state & ~carried};
            if (completion != Completion::None &&
                !llvm::is_contained(node.effects.sources, effect))
              node.effects.sources.push_back(effect);
            if (completion == Completion::Complete && sameIssuer)
              return std::nullopt;
            return advanceLifetime(state, node.op, access);
          },
          {},
          [](Edge edge, unsigned state) {
            return edge.invalidatesIteration ? state | carried : state;
          });
  }
}

} // namespace

std::optional<ClusterBarrierFacts>
ClusterBarrierModel::describe(Operation *op, FixedSync &fixed, Block *) {
  using namespace nvidia_gpu;
  // Physical guards keep their distributed scratch handoffs outside the arms.
  fixed = FixedSync::fromStages(getClusterBarrierStages(op), cluster);
  Facts facts;
  facts.initDeadline = isa<gpu::WarpSpecializeOp, ClusterBarrierOp>(op) ||
                       hasEffect<MemoryEffects::Write>(op, Value{});
  for (const MemoryAccess &access : getMemoryAccesses(op)) {
    if (!access.isShared())
      continue;
    facts.accesses.push_back(getSharedAccess(op, access));
    // A signal on one counter can let a sibling use another.
    if (access.isShared(gpu::SharedKind::Barrier))
      facts.initDeadline = true;
    else if (access.isWrite)
      facts.payloadWrite = access.value;
    facts.async |= access.isShared(gpu::SharedKind::Async);
  }
  facts.async |=
      isa<gpu::AsyncCopyGlobalToLocalOp, AsyncSharedStoreOp, TCGen5CommitOp>(
          op);
  if (auto dot = dyn_cast<WarpGroupDotOp>(op))
    facts.async = dot.getIsAsync();
  const auto *scratch = getScratchFootprint(op);
  if (scratch)
    facts.accesses.push_back({scratch, /*isWrite=*/true});
  if (fixed.inside) {
    // Each CTA writes its own scratch. Only the reads after an internal
    // cluster rendezvous can access a peer's scratch.
    facts.sources.emplace_back(
        SharedAccess{scratch, /*isWrite=*/false, /*crossCTA=*/true}, live);
  } else {
    // Only the lead CTA waits on a paired TMA completion barrier. Its
    // peer must rendezvous before reusing even a CTA-local destination.
    auto tma = dyn_cast<TMALoadLikeOpInterface>(op);
    bool pairedWait =
        tma &&
        hasCGABroadcast(cast<gpu::MemDescType>(tma.getBarrier().getType()));
    for (SharedAccess access : facts.accesses) {
      access.crossCTA |= pairedWait;
      facts.sources.emplace_back(access, live);
    }
  }
  if (facts.accesses.empty() && !facts.initDeadline &&
      !isa<gpu::LocalAllocOp, gpu::AsyncWaitOp, TMAStoreWaitOp,
           WarpGroupDotWaitOp>(op))
    return std::nullopt;
  return facts;
}

SmallVector<Demand>
ClusterBarrierModel::discoverDemands(MembarCFG<Facts> &cfg) {
  recordCompletions(cfg);
  SmallVector<Demand> demands;
  for (auto [start, producer] : llvm::enumerate(cfg.nodes)) {
    bool init = initializesPeerCounter(cfg, producer);
    for (auto [access, initial] : producer.effects.sources) {
      membar::discoverDemands(
          cfg, demands, start, initial,
          [](unsigned state, unsigned fixed) {
            return fixed & bit(Need::ClusterAcquire) ? 0u : state;
          },
          [&](const auto &node, unsigned state) -> Conflict {
            Path path = state & carried ? Path::LoopCarried : Path::Any;
            if (node.next.empty() && access.crossCTA)
              return {Need::ClusterAcquire, path, true};
            for (const auto &other : node.effects.accesses)
              if (conflicts(access, other, state & reallocated))
                return {Need::ClusterAcquire, path, true};
            if (init && node.effects.initDeadline)
              return {producer.body == 0 ? Need::ClusterInit
                                         : Need::ClusterAcquire,
                      path};
            return {};
          },
          [&](Edge edge, unsigned state) {
            if (cfg.invalidatesIteration(edge, start))
              state |= carried;
            return advanceLifetime(state, cfg.nodes[edge.to].op, access);
          });
    }
  }
  coalesceDemands(demands);

  // A conflict with the kernel can be cut at the call or its continuation.
  // Open helper sites only for conflicts between private operations. This
  // keeps CTA-local scratch helpers summarizable by ConSan at their calls.
  auto function = [](const Node<Facts> &node) {
    return node.op->getParentOfType<FunctionOpInterface>();
  };
  auto kernel = function(cfg.nodes.front());
  DenseSet<FunctionOpInterface> eligible{kernel};
  for (const auto &d : demands)
    if (function(cfg.nodes[d.source]) != kernel)
      eligible.insert(function(cfg.nodes[d.sink]));
  for (auto &node : cfg.nodes)
    if (!eligible.contains(function(node)))
      node.insertionCost = 0;
  return demands;
}

} // namespace mlir::triton::membar
