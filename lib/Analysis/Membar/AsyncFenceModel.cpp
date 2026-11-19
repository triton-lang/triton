#include "triton/Analysis/Membar/AsyncFenceModel.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/MBarrierUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::membar {
namespace {

constexpr unsigned proxyScopes = bit(Need::ProxyCTA) | bit(Need::ProxyCluster);

bool genericAccessMayCrossCTA(Operation *op) {
  using namespace nvidia_gpu;
  if (gpu::lookupNumCTAs(op) == 1 ||
      isa<InitBarrierOp, InvalBarrierOp, CLCLoadResultOp,
          gpu::AsyncCopyGlobalToLocalOp>(op))
    return false;
  if (auto store = dyn_cast<AsyncSharedStoreOp>(op))
    return isCrossCTALoadStore(store.getDst().getType(),
                               store.getSrc().getType());
  return hasCrossCTASharedAccess(op).value_or(true);
}

using FenceCFG = MembarCFG<AsyncFenceFacts>;

bool mayObserveBarrier(const Node<AsyncFenceFacts> &node,
                       ArrayRef<const BufferRegionFootprint *> barriers) {
  auto wait = dyn_cast<nvidia_gpu::WaitBarrierOp>(node.op);
  return wait && (!wait.getPred() || !matchPattern(wait.getPred(), m_Zero())) &&
         llvm::any_of(barriers, [&](const auto *barrier) {
           return mayOverlap(barrier, node.effects.completionBarrier);
         });
}

// A cluster fence covers both windows. A CTA fence covers a peer generic
// access once the owner has acquired it.
unsigned applyFixed(unsigned state, unsigned sync) {
  if (sync & bit(Need::ProxyCluster))
    return 0;
  if (!(state & bit(Need::ClusterAcquire)))
    state &= ~(sync & bit(Need::ProxyCTA));
  if (sync & bit(Need::ClusterAcquire))
    state &= ~bit(Need::ClusterAcquire);
  return state & proxyScopes ? state : 0;
}

} // namespace

const BufferIndexAnalysis &
AsyncFenceModel::getIndices(FunctionOpInterface function) {
  return indices.try_emplace(function, function).first->second;
}

const BufferRegionFootprint *AsyncFenceModel::getFootprint(Value value) {
  return regions.getKernelFootprint(regions.getFootprint(value));
}

std::optional<AsyncFenceFacts>
AsyncFenceModel::describe(Operation *op, FixedSync &fixed, Block *active) {
  Facts facts;
  if (active) {
    // The guard may execute these accesses. Its fences only cover the
    // participating threads.
    active->walk([&](Operation *nested) {
      FixedSync ignored;
      if (auto arm = describe(nested, ignored)) {
        assert(arm->proxy == gpu::SharedKind::Generic &&
               "unexpected async payload in compiler-generated thread guard");
        llvm::append_range(facts.accesses, arm->accesses);
      }
    });
    return facts;
  }
  if (isa<CallOpInterface>(op))
    return std::nullopt; // The CFG links the callee's payload accesses.
  if (auto fence = dyn_cast<nvidia_gpu::FenceAsyncSharedOp>(op)) {
    fixed.after =
        coverage(fence.getBCluster() ? Action::ProxyCluster : Action::ProxyCTA);
    return std::nullopt;
  }
  if (auto barrier = dyn_cast<nvidia_gpu::ClusterBarrierOp>(op)) {
    if (!barrier.getRelaxed())
      fixed.after = bit(Need::ClusterAcquire);
    return std::nullopt;
  }
  bool usesCompletionBarrier =
      isa<nvidia_gpu::AsyncSharedStoreOp, nvidia_gpu::AsyncCopyMbarrierArriveOp,
          nvidia_gpu::WaitBarrierOp>(op);
  if (usesCompletionBarrier) {
    Value barrier = cast<gpu::MBarrierOpInterface>(op).getBarrier();
    facts.completionBarrier = getFootprint(barrier);
  }
  for (const MemoryAccess &access : getMemoryAccesses(op)) {
    if (!access.isShared() || access.isShared(gpu::SharedKind::Barrier))
      continue;
    assert((facts.accesses.empty() || facts.proxy == *access.sharedKind) &&
           "mixed shared payload proxies");
    facts.proxy = *access.sharedKind;
    bool crossCTA = facts.proxy == gpu::SharedKind::Async
                        ? nvidia_gpu::hasCrossCTAAsyncSharedAccess(op)
                        : genericAccessMayCrossCTA(op);
    facts.accesses.push_back(
        {getFootprint(access.value), access.isWrite, crossCTA});
  }
  if (op->hasAttr("allocation.size")) {
    assert(facts.proxy == gpu::SharedKind::Generic &&
           "async proxy operation with generic scratch");
    auto *scratch = regions.getKernelFootprint(regions.getScratchFootprint(op));
    // Each CTA writes its own scratch. A distributed layout can read a peer's.
    facts.accesses.push_back({scratch, /*isWrite=*/true});
    if (hasCrossCTAScratch(op))
      facts.accesses.push_back({scratch, /*isWrite=*/false, /*crossCTA=*/true});
  }
  if (facts.accesses.empty() && !usesCompletionBarrier &&
      !isa<gpu::AsyncCommitGroupOp, gpu::AsyncWaitOp, gpu::WarpSpecializeOp>(
          op))
    return std::nullopt;
  return facts;
}

static void recordCompletedWrites(FenceCFG &cfg) {
  using namespace nvidia_gpu;
  llvm::SetVector<const BufferRegionFootprint *> copyBarriers;
  for (const auto &node : cfg.nodes)
    if (isa<AsyncCopyMbarrierArriveOp>(node.op))
      copyBarriers.insert(node.effects.completionBarrier);

  for (const auto &producer : cfg.nodes) {
    bool copy = isa<gpu::AsyncCopyGlobalToLocalOp>(producer.op);
    if (!copy && !isa<AsyncSharedStoreOp>(producer.op))
      continue;
    // st.async names its barrier. A cp.async may signal any copy barrier.
    auto barriers = copy ? copyBarriers.getArrayRef()
                         : ArrayRef(producer.effects.completionBarrier);
    const ProxyAccess &write = producer.effects.accesses.front();
    walkEdges(
        cfg, producer, /*committed=*/false,
        [&](unsigned point, bool committed,
            bool canExitWorker) -> std::optional<unsigned> {
          auto &node = cfg.nodes[point];
          // A worker that inherited the copy may reuse its issuer's warps.
          // Only a wait by the original issuer can retire it.
          bool inherited = node.inWorker && !canExitWorker;
          auto wait = dyn_cast<gpu::AsyncWaitOp>(node.op);
          bool groupWait = copy && wait && producer.inWorker == node.inWorker &&
                           (committed || inherited);
          // A sibling may have observed the write before the roles rejoin.
          bool roleExit =
              !inherited && isa<gpu::WarpYieldOp, gpu::WarpReturnOp>(node.op);
          if ((groupWait || roleExit || mayObserveBarrier(node, barriers)) &&
              !llvm::is_contained(node.effects.accesses, write))
            node.effects.accesses.push_back(write);
          if (groupWait && !inherited && wait.getNum() == 0)
            return std::nullopt;
          return committed || (copy && isa<gpu::AsyncCommitGroupOp>(node.op));
        });
  }
}

SmallVector<Demand> AsyncFenceModel::discoverDemands(FenceCFG &cfg) {
  recordCompletedWrites(cfg);
  unsigned scopes = numCTAs == 1 ? bit(Need::ProxyCTA) : proxyScopes;
  SmallVector<Demand> demands;
  for (auto [source, producer] : llvm::enumerate(cfg.nodes)) {
    if (producer.effects.proxy != gpu::SharedKind::Generic)
      continue;
    for (const ProxyAccess &access : producer.effects.accesses) {
      unsigned state =
          scopes | (access.crossCTA ? bit(Need::ClusterAcquire) : 0);
      membar::discoverDemands(
          cfg, demands, source, state, applyFixed,
          [&](const auto &sink, unsigned live) -> Conflict {
            if (sink.effects.proxy == gpu::SharedKind::Async)
              for (const ProxyAccess &other : sink.effects.accesses) {
                Need need =
                    other.crossCTA ? Need::ProxyCluster : Need::ProxyCTA;
                if ((live & bit(need)) && (access.isWrite || other.isWrite) &&
                    mayOverlap(access.footprint, other.footprint))
                  return {need, !other.crossCTA && access.crossCTA
                                    ? Path::ClusterThenProxy
                                    : Path::Any};
              }
            return {};
          });
    }
  }
  coalesceDemands(demands);
  return demands;
}

} // namespace mlir::triton::membar
