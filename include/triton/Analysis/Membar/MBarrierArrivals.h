#ifndef TRITON_ANALYSIS_MEMBAR_MBARRIERARRIVALS_H
#define TRITON_ANALYSIS_MEMBAR_MBARRIERARRIVALS_H

#include "triton/Analysis/Membar/SmemModel.h"

namespace mlir::triton::membar {

/// Choose notification modes while placing barriers, then scale each counter.
LogicalResult placeBarriersAndArrivals(ModuleAllocation &allocation,
                                       BufferRegionAnalysis &regions,
                                       MembarFilterFn filter,
                                       int computeCapability);

} // namespace mlir::triton::membar

#endif // TRITON_ANALYSIS_MEMBAR_MBARRIERARRIVALS_H
