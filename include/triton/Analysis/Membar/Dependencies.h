#ifndef TRITON_ANALYSIS_MEMBAR_DEPENDENCIES_H
#define TRITON_ANALYSIS_MEMBAR_DEPENDENCIES_H

#include "triton/Analysis/Membar/CFGBuilder.h"

namespace mlir::triton::membar {

// Loop-carried shared hazards and ordered TMEM handoffs each need one extra
// state. The latter always ends with a region barrier.
enum class Path { Any, LoopCarried, WaitThenOrder };

struct Demand {
  unsigned source, sink;
  Need first;
  Path path = Path::Any;
};

SmallVector<Demand> discoverDemands(const MembarCFG &cfg,
                                    Allocation &allocation,
                                    const MembarFilterFn &sharedFilter,
                                    TMemModel *tmem);

} // namespace mlir::triton::membar

#endif // TRITON_ANALYSIS_MEMBAR_DEPENDENCIES_H
