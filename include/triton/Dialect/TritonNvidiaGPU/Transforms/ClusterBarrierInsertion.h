#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_CLUSTERBARRIERINSERTION_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_CLUSTERBARRIERINSERTION_H_

#include "mlir/Support/LogicalResult.h"
#include "triton/Analysis/Allocation.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

/// Inserts cluster synchronization ops for distributed shared-memory
/// dependencies using the provided allocation analysis.
void runClusterBarrierInsertion(ModuleAllocation &moduleAllocation,
                                int computeCapability);

/// Inserts the mbarrier-init sequencing ops
/// (fence_mbarrier_init_release_cluster + cluster_arrive/wait(relaxed=true))
/// for cross-CTA mbarriers using the provided allocation analysis.
LogicalResult
runCrossCTAMBarrierInitSyncInsertion(ModuleAllocation &moduleAllocation,
                                     int computeCapability);

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_CLUSTERBARRIERINSERTION_H_
