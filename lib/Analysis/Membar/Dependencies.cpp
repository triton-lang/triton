#include "triton/Analysis/Membar/Dependencies.h"
#include "triton/Analysis/Membar/TmemModel.h"
#include "llvm/ADT/STLExtras.h"
#include <tuple>

namespace mlir::triton::membar {
namespace {

auto stopAt(Need need) {
  // A TMEM wait can still need a later cross-warp rendezvous.
  return [mask = bit(need) & fullBarrier](unsigned state, unsigned fixed) {
    return fixed & mask ? 0u : state;
  };
}

std::optional<Path> sharedNeed(const BlockInfo &out, const OpModel &in,
                               bool carried,
                               std::optional<BlockInfo> &carriedOut,
                               Allocation &allocation,
                               const MembarFilterFn &filter) {
  // A kernel return has no subsequent CTA-local consumer.
  if (in.boundary == Boundary::KernelReturn)
    return out.threadSync.completion == CompletionSync::All
               ? std::optional(Path::Any)
               : std::nullopt;
  if (BlockInfo::requiresThreadSync(out, in.sharedIn) ||
      out.isIntersected(in.sharedIn, filter, &allocation))
    return Path::Any;
  if (!carried)
    return std::nullopt;
  if (!carriedOut) {
    carriedOut.emplace(out);
    carriedOut->invalidateIterationInfo();
  }
  if (carriedOut->isIntersected(in.sharedIn, filter, &allocation))
    return Path::LoopCarried;
  return std::nullopt;
}

bool tmemConflicts(ArrayRef<TMemAccess> before, ArrayRef<TMemAccess> after) {
  for (const TMemAccess &a : before)
    for (const TMemAccess &b : after)
      if ((a.isWrite || b.isWrite) && mayOverlap(a.footprint, b.footprint))
        return true;
  return false;
}

std::optional<Path> tmemNeed(const Node<OpModel> &producer,
                             const Node<OpModel> &consumer, TMemModel &tmem,
                             bool needsWait) {
  auto boundary = consumer.effects.boundary;
  Path ordered = needsWait ? Path::WaitThenOrder : Path::Any;
  // Peer-visible TMEM accesses require a rendezvous before publication.
  if (boundary == Boundary::Publication && !producer.effects.privateTMem)
    return ordered;
  // A completion wait orders same-warp RAW, WAR, and WAW dependencies.
  if (tmemConflicts(producer.effects.tmemAccesses,
                    consumer.effects.tmemAccesses))
    return tmem.isSameWarp(producer.op, consumer.op) ? Path::Any : ordered;
  if (needsWait &&
      (boundary == Boundary::WaitAll || boundary == Boundary::KernelReturn))
    return Path::Any;
  return std::nullopt;
}

} // namespace

SmallVector<Demand> discoverDemands(const MembarCFG<OpModel> &cfg,
                                    Allocation &allocation,
                                    const MembarFilterFn &sharedFilter,
                                    TMemModel *tmem) {
  bool needsWarp = llvm::any_of(
      cfg.nodes, [](const auto &node) { return node.effects.needsWarp; });
  bool ordersThreadEffects = llvm::any_of(cfg.nodes, [](const auto &node) {
    return node.effects.sharedIn.threadDemands.publicationIssuer !=
           ThreadIssuer::None;
  });
  SmallVector<Demand> demands;
  for (auto [source, producer] : llvm::enumerate(cfg.nodes)) {
    const OpModel &out = producer.effects;
    const BlockInfo &shared = out.sharedOut;
    const auto &sync = shared.threadSync;
    bool hasSlices =
        !shared.syncReadSlices.empty() || !shared.syncWriteSlices.empty();
    if (hasSlices || sync.completion != CompletionSync::None ||
        (ordersThreadEffects && sync.effectIssuer != ThreadIssuer::None)) {
      constexpr unsigned fresh = 1, carried = 2;
      std::optional<BlockInfo> carriedOut;
      discoverDemands(
          cfg, demands, source, fresh, stopAt(Need::RegionOrder),
          [&](const auto &sink, unsigned state) -> Conflict {
            return {Need::RegionOrder,
                    sharedNeed(shared, sink.effects, state == carried,
                               carriedOut, allocation, sharedFilter),
                    /*stopOnMatch=*/true};
          },
          [&](Edge edge, unsigned state) {
            return hasSlices && cfg.invalidatesIteration(edge, source) ? carried
                                                                       : state;
          });
    }

    Need producerNeed = out.tmemProducerNeed.value_or(Need::RegionOrder);
    bool needsWait = producerNeed != Need::RegionOrder;
    if (out.tmemProducerNeed)
      discoverDemands(cfg, demands, source, 1, stopAt(producerNeed),
                      [&](const auto &sink, unsigned) -> Conflict {
                        return {producerNeed,
                                tmemNeed(producer, sink, *tmem, needsWait)};
                      });

    // A completion wait leaves an outgoing issuer but has no input to order.
    if (needsWarp && shared.hasEffects()) {
      Need warpNeed = needsWait || !out.sharedIn.hasEffects()
                          ? Need::WarpConvergence
                          : Need::WarpOrder;
      discoverDemands(cfg, demands, source, 1, stopAt(warpNeed),
                      [&](const auto &sink, unsigned) -> Conflict {
                        return {warpNeed, sink.effects.needsWarp
                                              ? std::optional(Path::Any)
                                              : std::nullopt};
                      });
    }
  }
  coalesceDemands(demands);
  return demands;
}

void coalesceDemands(SmallVectorImpl<Demand> &demands) {
  // Ordered handoffs subsume ordinary demands, which subsume loop-carried ones.
  auto key = [](const Demand &d) {
    return std::tuple(d.source, d.sink, d.first, d.path == Path::LoopCarried,
                      d.path == Path::Any);
  };
  llvm::sort(demands,
             [&](const Demand &a, const Demand &b) { return key(a) < key(b); });
  auto sameTerminals = [](const Demand &a, const Demand &b) {
    return a.source == b.source && a.sink == b.sink && a.first == b.first;
  };
  demands.erase(llvm::unique(demands, sameTerminals), demands.end());
}

} // namespace mlir::triton::membar
