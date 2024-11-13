#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-pipeline-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPIPELINESCHEDULER
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

void scheduleLoop(scf::ForOp forOp, int numStages) {
  if (!preCondition(forOp))
    return;

  // 2. Schedule key ops
  //    Based on the latencies, schedule the key ops to the stages.
  // 3. Schedule dependencies
  //    Schedule the dependencies (regular and dist 1)
  // 4. Schedule the rest of the ops to the last stage
}

struct PipelineScheduler
    : public impl::TritonGPUPipelineSchedulerBase<PipelineScheduler> {
  using impl::TritonGPUPipelineSchedulerBase<
      PipelineScheduler>::TritonGPUPipelineSchedulerBase;

  void runOnOperation() override {
    // 1. Assign latencies
    //    Go over the interesting ops and assign latencies (based on the
    //    numStages) to the them, trying to populate the allowed stages. This
    //    step will be at some point extracted to separate pass that will be run
    //    only for loops missing the latency information.
    DenseMap<Operation *, int> opLatency =
        assignLatencies(getOperation(), numStages);

    // numStages should not be used below this point. We should know everything
    // based on the assigned stages
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
