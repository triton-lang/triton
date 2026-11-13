#include "MembarCFG.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SparseSet.h"
#include <tuple>

namespace mlir::triton::membar {
namespace {

// A producer's shared slices have different alias facts across iterations.
enum Fact : unsigned { Shared, CarriedShared, TMem, Warp, NumFacts };

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
  if (in.boundary == Boundary::Clean ||
      BlockInfo::requiresThreadSync(out, in.sharedIn) ||
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

std::optional<Path> tmemNeed(const Node &producer, const Node &consumer,
                             TMemModel &tmem, bool needsWait) {
  auto boundary = consumer.effects.boundary;
  Path ordered = needsWait ? Path::WaitThenOrder : Path::Any;
  // Peer-visible TMEM accesses require a rendezvous before publication.
  if (boundary == Boundary::Clean ||
      (boundary == Boundary::Publication && !producer.effects.privateTMem))
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

SmallVector<Demand> discoverDemands(const MembarCFG &cfg,
                                    Allocation &allocation,
                                    const MembarFilterFn &sharedFilter,
                                    TMemModel *tmem) {
  SmallVector<Demand> demands;
  bool needsWarp = llvm::any_of(
      cfg.nodes, [](const Node &node) { return node.effects.needsWarp; });
  bool ordersThreadEffects = llvm::any_of(cfg.nodes, [](const Node &node) {
    return node.effects.boundary == Boundary::Clean ||
           node.effects.sharedIn.threadDemands.publicationIssuer !=
               ThreadIssuer::None;
  });
  // The sparse set queues each (consumer, fact) once per producer.
  llvm::SparseSet<unsigned> arrivals;
  arrivals.setUniverse(NumFacts * cfg.nodes.size());
  for (unsigned source = 0; source < cfg.nodes.size(); ++source) {
    const Node &producer = cfg.nodes[source];
    const OpModel &out = producer.effects;
    const BlockInfo &shared = out.sharedOut;
    const auto &sync = shared.threadSync;
    bool hasSlices =
        !shared.syncReadSlices.empty() || !shared.syncWriteSlices.empty();
    Need producerNeed = out.tmemProducerNeed.value_or(Need::RegionOrder);
    bool needsWait = producerNeed != Need::RegionOrder;
    // A completion wait leaves an outgoing issuer but has no input to order.
    Need warpNeed = needsWait || !out.sharedIn.hasEffects()
                        ? Need::WarpConvergence
                        : Need::WarpOrder;
    const Need needs[] = {Need::RegionOrder, Need::RegionOrder, producerNeed,
                          warpNeed};
    arrivals.clear();
    auto propagate = [&](const Node &node, Fact fact,
                         unsigned fixedInside = 0) {
      // A TMEM wait can still need a later cross-warp rendezvous.
      unsigned stop = bit(needs[fact]) & fullBarrier;
      if ((fixedInside | node.effects.fixedAfter) & stop)
        return;
      for (Edge edge : node.next) {
        if (cfg.nodes[edge.to].effects.fixedBefore & stop)
          continue;
        Fact next = fact == Shared && hasSlices && edge.invalidatesIteration
                        ? CarriedShared
                        : fact;
        arrivals.insert(NumFacts * edge.to + next);
      }
    };
    // Seed outgoing edges so a self-dependency must traverse a loop.
    if (hasSlices || sync.completion != CompletionSync::None ||
        (ordersThreadEffects && sync.effectIssuer != ThreadIssuer::None))
      propagate(producer, Shared);
    if (out.tmemProducerNeed)
      propagate(producer, TMem);
    // An entry or call can require convergence even without effects.
    if (needsWarp && (source == cfg.entry || out.boundary == Boundary::Clean ||
                      shared.hasEffects()))
      propagate(producer, Warp);

    std::optional<BlockInfo> carriedOut;
    for (unsigned i = 0; i < arrivals.size(); ++i) {
      unsigned arrival = arrivals.begin()[i];
      unsigned sink = arrival / NumFacts;
      Fact fact = static_cast<Fact>(arrival % NumFacts);
      const Node &consumer = cfg.nodes[sink];
      const OpModel &in = consumer.effects;
      bool isShared = fact == Shared || fact == CarriedShared;
      std::optional<Path> path;
      if (isShared)
        path = sharedNeed(shared, in, fact == CarriedShared, carriedOut,
                          allocation, sharedFilter);
      else if (fact == TMem)
        path = tmemNeed(producer, consumer, *tmem, needsWait);
      else if (in.needsWarp)
        path = Path::Any;

      if (path) {
        demands.push_back({source, sink, needs[fact], *path});
        // Ordering the first shared conflict covers later ones on this path.
        if (isShared)
          continue;
      }
      if (in.boundary != Boundary::Clean)
        propagate(consumer, fact, in.fixedInside);
    }
  }

  // Any region order subsumes loop-carried order at the same endpoints.
  llvm::sort(demands, [](const Demand &a, const Demand &b) {
    return std::tie(a.source, a.sink, a.first, a.path) <
           std::tie(b.source, b.sink, b.first, b.path);
  });
  auto sameTerminals = [](const Demand &a, const Demand &b) {
    return a.source == b.source && a.sink == b.sink && a.first == b.first;
  };
  demands.erase(llvm::unique(demands, sameTerminals), demands.end());
  return demands;
}

} // namespace mlir::triton::membar
