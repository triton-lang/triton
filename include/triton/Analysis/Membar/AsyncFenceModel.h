#ifndef TRITON_ANALYSIS_MEMBAR_ASYNCFENCEMODEL_H
#define TRITON_ANALYSIS_MEMBAR_ASYNCFENCEMODEL_H

#include "triton/Analysis/Membar/Dependencies.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

namespace mlir::triton::membar {

struct AsyncFenceFacts {
  gpu::SharedKind proxy = gpu::SharedKind::Generic;
  SmallVector<SharedAccess, 1> accesses;
  const BufferRegionFootprint *completionBarrier = nullptr;
};

class AsyncFenceModel : public SharedAccessModel {
public:
  using Facts = AsyncFenceFacts;
  AsyncFenceModel(BufferRegionAnalysis &regions, int numCTAs)
      : SharedAccessModel(regions), numCTAs(numCTAs) {}

  std::optional<Facts> describe(Operation *op, FixedSync &fixed,
                                Block *active = nullptr);
  SmallVector<Demand> discoverDemands(MembarCFG<Facts> &cfg);

private:
  int numCTAs;
};

} // namespace mlir::triton::membar

#endif // TRITON_ANALYSIS_MEMBAR_ASYNCFENCEMODEL_H
