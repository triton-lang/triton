#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-loop-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPULOOPSCHEDULING
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPULoopSchedulingPass
    : public impl::TritonGPULoopSchedulingBase<TritonGPULoopSchedulingPass> {
public:
  using impl::TritonGPULoopSchedulingBase<
      TritonGPULoopSchedulingPass>::TritonGPULoopSchedulingBase;

  void runOnOperation() override {
    // Go over the interesting ops and assign latencies (based on the
    // numStages) to the them, trying to populate the allowed stages. This
    // step will be at some point extracted to separate pass that will be run
    // only for loops missing the latency information.
    DenseMap<Operation *, int> opLatency =
        assignLatencies(getOperation(), numStages);
    LLVM_DEBUG({
      LDBG("Assigned latencies:\n");
      for (auto [op, latency] : opLatency)
        LDBG("  " << latency << " : " << *op);
    });
    // numStages should not be used below this point. We should know everything
    // based on the assigned stages

    // Schedule the loops
    scheduleLoops(getOperation(), opLatency);
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
