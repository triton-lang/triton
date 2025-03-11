#ifndef TRITON_TRITONGPU_TRANSFORM_PIPELINE_WARPSPECIALIZATION_H_
#define TRITON_TRITONGPU_TRANSFORM_PIPELINE_WARPSPECIALIZATION_H_

#include "mlir/Support/LogicalResult.h"

namespace mlir::triton::gpu {
// This is the final step to prepare a loop for warp specialization. This takes
// a loop with a partition schedule and rewrites the loop such that all SSA
// dependencies between partitions are passed through shared memory and
// multibuffers them according to partition stages.
LogicalResult rewritePartitionDependencies(scf::ForOp loop);
} // namespace mlir::triton::gpu

#endif // TRITON_TRITONGPU_TRANSFORM_PIPELINE_WARPSPECIALIZATION_H_
