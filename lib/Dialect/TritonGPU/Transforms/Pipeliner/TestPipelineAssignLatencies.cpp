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

#define GEN_PASS_DEF_TRITONGPUTESTPIPELINEASSIGNLATENCIES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

static const char *kLatencyAttrName = "tt.latency";

struct TestPipelineAssignLatencies
    : public impl::TritonGPUTestPipelineAssignLatenciesBase<
          TestPipelineAssignLatencies> {
  using impl::TritonGPUTestPipelineAssignLatenciesBase<
      TestPipelineAssignLatencies>::TritonGPUTestPipelineAssignLatenciesBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

    if (loops.empty())
      return;

    for (scf::ForOp forOp : loops) {
      DenseMap<Operation *, int> opLatencies =
          assignLatencies(forOp, numStages);
      for (auto [op, latency] : opLatencies) {
        op->setAttr(
            kLatencyAttrName,
            IntegerAttr::get(IntegerType::get(m.getContext(), 32), latency));
      }
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
