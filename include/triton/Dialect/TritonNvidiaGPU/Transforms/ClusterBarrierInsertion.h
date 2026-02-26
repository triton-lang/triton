#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_CLUSTERBARRIERINSERTION_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_CLUSTERBARRIERINSERTION_H_

#include "triton/Analysis/Allocation.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

/// Inserts cluster barriers (cluster_barrier) using the provided
/// shared-memory allocation analysis.
void runClusterBarrierInsertion(ModuleAllocation &moduleAllocation,
                                int computeCapability);

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_CLUSTERBARRIERINSERTION_H_
