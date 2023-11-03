#ifndef TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_
#define TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_

#include "PipelineExpander.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include <vector>

namespace mlir {
namespace triton {

/// This fill out the pipelining options including schedule and annotations for
/// wait ops. This also does pre-processing by converting some of the loads into
/// async loads so that the IR is ready to be pipelined.
bool preProcessLoopAndGetSchedule(scf::ForOp &forOp, int numStages,
                                  mlir::triton::PipeliningOption &options);

/// This does post-processing on the pipelined loop to try to pipeline wgmma
/// ops.
// TODO: this should be included as part of the pipeline but currently the wgmma
// wait modeling is problematic.
void asyncLaunchDots(scf::ForOp forOp);

} // namespace triton
} // namespace mlir
#endif // TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_
