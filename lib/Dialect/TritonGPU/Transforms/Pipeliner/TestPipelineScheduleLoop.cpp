#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUTESTPIPELINESCHEDULELOOP
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

static const char *kLatencyAttrName = "tt.latency";

struct TestPipelineScheduleLoop
    : public impl::TritonGPUTestPipelineScheduleLoopBase<
          TestPipelineScheduleLoop> {
  using impl::TritonGPUTestPipelineScheduleLoopBase<
      TestPipelineScheduleLoop>::TritonGPUTestPipelineScheduleLoopBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    DenseMap<Operation *, int> opLatencies;

    // Deserialize latencies from the IR.
    m.walk([&](Operation *op) {
      if (op->hasAttr(kLatencyAttrName)) {
        int latency =
            mlir::cast<IntegerAttr>(op->getAttr(kLatencyAttrName)).getInt();
        op->removeAttr(kLatencyAttrName);
        opLatencies[op] = latency;
      }
    });

    SmallVector<scf::ForOp> loops;
    m.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

    for (auto forOp : loops) {
      scheduleLoop(forOp, opLatencies);
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
