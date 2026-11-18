#pragma once

#include "triton/Analysis/Membar/SmemModel.h"

namespace mlir::triton::nvidia_gpu {

/// Choose notification modes while placing barriers, then scale each counter.
LogicalResult placeBarriersAndArrivals(ModuleAllocation &allocation,
                                       BufferRegionAnalysis &regions,
                                       MembarFilterFn filter,
                                       int computeCapability);

} // namespace mlir::triton::nvidia_gpu
