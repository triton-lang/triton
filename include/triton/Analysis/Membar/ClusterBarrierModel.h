#ifndef TRITON_ANALYSIS_MEMBAR_CLUSTERBARRIERMODEL_H
#define TRITON_ANALYSIS_MEMBAR_CLUSTERBARRIERMODEL_H

#include "triton/Analysis/Membar/Dependencies.h"

namespace mlir::triton::membar {

struct ClusterBarrierFacts {
  SmallVector<SharedAccess, 1> accesses;
  // Producer effects and their lifetime flags, including accesses whose
  // completion can be observed here.
  SmallVector<std::pair<SharedAccess, unsigned>, 1> sources;
  Value payloadWrite;
  bool async = false;
  std::optional<SyncKind> initDeadline;
};

/// Order peer accesses before shared storage is reused or a CTA exits.
/// Publish peer-visible mbarrier initializations before their first use.
class ClusterBarrierModel : public SharedAccessModel {
public:
  using Facts = ClusterBarrierFacts;
  using SharedAccessModel::SharedAccessModel;
  std::optional<Facts> describe(Operation *op, FixedSync &fixed,
                                Block *active = nullptr);
  SmallVector<Demand> discoverDemands(MembarCFG<Facts> &cfg);
};

} // namespace mlir::triton::membar

#endif // TRITON_ANALYSIS_MEMBAR_CLUSTERBARRIERMODEL_H
