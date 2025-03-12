#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUTESTPIPELINEHOISTTMEMALLOC
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

void hoistTMEMAlloc(triton::nvidia_gpu::TMEMAllocOp alloc, scf::ForOp forOp,
                    CoarseSchedule &schedule);

struct TestPipelineHoistTMEMAlloc
    : public impl::TritonGPUTestPipelineHoistTMEMAllocBase<
          TestPipelineHoistTMEMAlloc> {
  using impl::TritonGPUTestPipelineHoistTMEMAllocBase<
      TestPipelineHoistTMEMAlloc>::TritonGPUTestPipelineHoistTMEMAllocBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    m.walk([&](scf::ForOp forOp) {
      CoarseSchedule schedule;
      if (failed(schedule.deSerialize(forOp))) {
        llvm_unreachable("Failed to deserialize schedule");
      }
      m.walk([&](triton::nvidia_gpu::TMEMAllocOp alloc) {
        hoistTMEMAlloc(alloc, forOp, schedule);
      });
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
