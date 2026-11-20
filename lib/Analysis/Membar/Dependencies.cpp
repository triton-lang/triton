#include "triton/Analysis/Membar/Dependencies.h"
#include "triton/Analysis/Membar/TmemModel.h"
#include "llvm/ADT/STLExtras.h"
#include <tuple>

namespace mlir::triton::membar {
namespace {

auto stopAt(SyncKind need) {
  // A TMEM wait can still need a later cross-warp rendezvous.
  return [mask = bit(need) & fullBarrier](unsigned state, unsigned fixed) {
    return fixed & mask ? 0u : state;
  };
}

void addDemand(MembarDemands &demands, ArrivalMode mode, Demand demand) {
  if (mode == ArrivalMode::Either)
    demands.optional.push_back(demand);
  if (mode != ArrivalMode::Scalar) {
    if (demand.first == SyncKind::RegionBarrier)
      demand.first = SyncKind::WarpBarrier;
    demand.path = Path::Any;
  }
  demands.required.push_back(demand);
}

std::optional<Path> sharedNeed(const BlockInfo &out, const OpModel &in,
                               bool threadSync, bool carried,
                               std::optional<BlockInfo> &carriedOut,
                               Allocation &allocation,
                               const MembarFilterFn &filter) {
  // A kernel return has no subsequent CTA-local consumer.
  if (in.boundary == Boundary::KernelReturn)
    return out.threadSync.completion == CompletionSync::All
               ? std::optional(Path::Any)
               : std::nullopt;
  if ((threadSync && in.arrivalMode == ArrivalMode::Scalar) ||
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
                             const Node<OpModel> &consumer, TMemModel &tmem) {
  bool needsWait =
      *producer.effects.tmemProducerNeed != SyncKind::RegionBarrier;
  auto boundary = consumer.effects.boundary;
  Path ordered = needsWait ? Path::WaitThenOrder : Path::Any;
  // A scalar publication must observe all issuing warps' completions.
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

MembarDemands discoverDemands(const MembarCFG<OpModel> &cfg,
                              Allocation &allocation,
                              const MembarFilterFn &sharedFilter,
                              TMemModel *tmem) {
  bool ordersThreadEffects = llvm::any_of(cfg.nodes, [](const auto &node) {
    return node.effects.sharedIn.threadDemands.publicationIssuer !=
           ThreadIssuer::None;
  });
  MembarDemands demands;
  for (unsigned source = 0; source < cfg.nodes.size(); ++source) {
    const auto &producer = cfg.nodes[source];
    const OpModel &out = producer.effects;
    const BlockInfo &shared = out.sharedOut;
    const auto &sync = shared.threadSync;
    bool hasSlices =
        !shared.syncReadSlices.empty() || !shared.syncWriteSlices.empty();
    if (hasSlices || sync.completion != CompletionSync::None ||
        (ordersThreadEffects && sync.effectIssuer != ThreadIssuer::None)) {
      constexpr unsigned fresh = 1, carried = 2;
      std::optional<BlockInfo> carriedOut;
      walkEdges(
          cfg, producer, fresh,
          [&](unsigned sink, unsigned state, bool) -> std::optional<unsigned> {
            const auto &in = cfg.nodes[sink].effects;
            bool threadSync =
                BlockInfo::requiresThreadSync(shared, in.sharedIn);
            if (auto path = sharedNeed(shared, in, threadSync, state == carried,
                                       carriedOut, allocation, sharedFilter)) {
              demands.required.push_back(
                  {source, sink, SyncKind::RegionBarrier, *path});
              return std::nullopt;
            }
            // A pending effect reaches consecutive direct notifications until a
            // CTA rendezvous. Lane zero in each warp can issue both in order.
            if (threadSync)
              addDemand(demands, in.arrivalMode,
                        {source, sink, SyncKind::RegionBarrier});
            return state;
          },
          stopAt(SyncKind::RegionBarrier),
          [&](Edge edge, unsigned state) {
            return hasSlices && cfg.invalidatesIteration(edge, source) ? carried
                                                                       : state;
          });
    }

    if (auto need = out.tmemProducerNeed)
      walkEdges(
          cfg, producer, 1,
          [&](unsigned sink, unsigned state, bool) {
            const auto &consumer = cfg.nodes[sink];
            if (auto path = tmemNeed(producer, consumer, *tmem))
              addDemand(demands, consumer.effects.arrivalMode,
                        {source, sink, *need, *path});
            return std::optional(state);
          },
          stopAt(*need));
  }
  coalesceDemands(demands.required);
  coalesceDemands(demands.optional);
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
