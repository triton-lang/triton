#ifndef TRITON_ANALYSIS_MEMBAR_ASYNCFENCEMODEL_H
#define TRITON_ANALYSIS_MEMBAR_ASYNCFENCEMODEL_H

#include "triton/Analysis/Membar/Dependencies.h"
#include <map>

namespace mlir::triton::membar {

struct ProxyAccess {
  const BufferRegionFootprint *footprint;
  bool isWrite;
  bool crossCTA = false;

  bool operator==(const ProxyAccess &) const = default;
};

struct AsyncFenceFacts {
  gpu::SharedKind proxy = gpu::SharedKind::Generic;
  SmallVector<ProxyAccess, 1> accesses;
  const BufferRegionFootprint *completionBarrier = nullptr;
};

class AsyncFenceModel {
public:
  using Facts = AsyncFenceFacts;
  AsyncFenceModel(BufferRegionAnalysis &regions, int numCTAs)
      : regions(regions), numCTAs(numCTAs) {}

  const BufferIndexAnalysis &getIndices(FunctionOpInterface function);
  std::optional<Facts> describe(Operation *op, FixedSync &fixed,
                                Block *active = nullptr);
  SmallVector<Demand> discoverDemands(MembarCFG<Facts> &cfg);

private:
  const BufferRegionFootprint *getFootprint(Value value);
  BufferRegionAnalysis &regions;
  int numCTAs;
  std::map<Operation *, BufferIndexAnalysis> indices;
};

} // namespace mlir::triton::membar

#endif // TRITON_ANALYSIS_MEMBAR_ASYNCFENCEMODEL_H
