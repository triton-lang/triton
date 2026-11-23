#include "triton/Analysis/Membar/AsyncFenceModel.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::membar {
namespace {

constexpr unsigned proxyScopes = coverage(SyncKind::ClusterProxyFence);

using FenceCFG = MembarCFG<AsyncFenceFacts>;

bool mayObserveBarrier(const Node<AsyncFenceFacts> &node,
                       ArrayRef<const BufferRegionFootprint *> barriers) {
  return isa<nvidia_gpu::WaitBarrierOp, nvidia_gpu::BarrierTestWaitReportOp>(
             node.op) &&
         llvm::any_of(barriers, [&](const auto *barrier) {
           return mayOverlap(barrier, node.effects.completionBarrier);
         });
}

// A cluster fence covers both windows. A CTA fence covers a peer generic
// access once the owner has acquired it.
unsigned applyFixed(unsigned state, unsigned sync) {
  if (sync & bit(SyncKind::ClusterProxyFence))
    return 0;
  if (!(state & bit(SyncKind::ClusterBarrier)))
    state &= ~(sync & bit(SyncKind::CTAProxyFence));
  if (sync & bit(SyncKind::ClusterBarrier))
    state &= ~bit(SyncKind::ClusterBarrier);
  return state & proxyScopes ? state : 0;
}

} // namespace

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
  if (auto fence = dyn_cast<nvidia_gpu::FenceAsyncSharedOp>(op)) {
    fixed.after = coverage(fence.getBCluster() ? SyncKind::ClusterProxyFence
                                               : SyncKind::CTAProxyFence);
    return std::nullopt;
  }
  if (auto barrier = dyn_cast<nvidia_gpu::ClusterBarrierOp>(op)) {
    if (!barrier.getRelaxed())
      fixed.after = bit(SyncKind::ClusterBarrier);
    return std::nullopt;
  }
  bool usesCompletionBarrier =
      isa<nvidia_gpu::AsyncSharedStoreOp, nvidia_gpu::AsyncCopyMbarrierArriveOp,
          nvidia_gpu::WaitBarrierOp, nvidia_gpu::BarrierTestWaitReportOp>(op);
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
    facts.accesses.push_back(getSharedAccess(op, access));
  }
  if (op->hasAttr("allocation.size")) {
    assert(facts.proxy == gpu::SharedKind::Generic &&
           "async proxy operation with generic scratch");
    auto *scratch = getScratchFootprint(op);
    // Each CTA writes its own scratch. A distributed layout can read a peer's.
    facts.accesses.push_back({scratch, /*isWrite=*/true});
    if (hasCrossCTAScratch(op))
      facts.accesses.push_back({scratch, /*isWrite=*/false, /*crossCTA=*/true});
  }
  if (facts.accesses.empty() && !usesCompletionBarrier &&
      !isa<gpu::AsyncCommitGroupOp, gpu::AsyncWaitOp>(op))
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
    const SharedAccess &write = producer.effects.accesses.front();
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
  unsigned scopes = numCTAs == 1 ? bit(SyncKind::CTAProxyFence) : proxyScopes;
  SmallVector<Demand> demands;
  for (unsigned source = 0; source < cfg.nodes.size(); ++source) {
    const auto &producer = cfg.nodes[source];
    if (producer.effects.proxy != gpu::SharedKind::Generic)
      continue;
    for (const SharedAccess &access : producer.effects.accesses) {
      unsigned state =
          scopes | (access.crossCTA ? bit(SyncKind::ClusterBarrier) : 0);
      walkEdges(
          cfg, producer, state,
          [&](unsigned sink, unsigned live, bool) -> std::optional<unsigned> {
            const auto &in = cfg.nodes[sink].effects;
            if (in.proxy == gpu::SharedKind::Async)
              for (const SharedAccess &other : in.accesses) {
                SyncKind need = other.crossCTA ? SyncKind::ClusterProxyFence
                                               : SyncKind::CTAProxyFence;
                if ((live & bit(need)) && (access.isWrite || other.isWrite) &&
                    mayOverlap(access.footprint, other.footprint)) {
                  Path path = !other.crossCTA && access.crossCTA
                                  ? Path::ClusterThenProxy
                                  : Path::Any;
                  demands.push_back({source, sink, need, path});
                  break;
                }
              }
            return live;
          },
          applyFixed);
    }
  }
  coalesceDemands(demands);
  return demands;
}

} // namespace mlir::triton::membar
