#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUAUTOMATICWARPSPECIALIZATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct AutomaticWarpSpecialization
    : triton::gpu::impl::TritonGPUAutomaticWarpSpecializationBase<
          AutomaticWarpSpecialization> {
  using TritonGPUAutomaticWarpSpecializationBase::
      TritonGPUAutomaticWarpSpecializationBase;

  void runOnOperation() override;
};
} // namespace

void AutomaticWarpSpecialization::runOnOperation() {
  OpPassManager pm;
  pm.addPass(createTritonGPUPartitionScheduling());
  pm.addPass(createNVWSInsertAref());
  pm.addPass(createNVWSInsertTmemAref());
  // `int-range-optimizations` and SCCP are good at cleaning up loop arithmetic.
  // FIXME: Re-enable integer range analysis once it is fixed.
  // pm.addPass(arith::createIntRangeOptimizationsPass());
  pm.addPass(createSCCPPass());
  pm.addPass(createCSEPass());
  pm.addPass(createNVWSLowerAref({numStages}));
  pm.addPass(createTritonGPUPartitionLoops());
  pm.addPass(createNVWSLowerWarpGroup());
  pm.addPass(createTritonGPUScheduleLoops());
  if (failed(runPipeline(pm, getOperation())))
    return signalPassFailure();

  {
    // Multi-buffer TMA descriptors. We cannot rely on SWP to do it, to support
    // desc updates in nested loops.
    SetVector<scf::ForOp> descUpdateLoops;
    getOperation().walk([&](scf::ForOp loop) {
      if (loop->hasAttr(kWarpSpecializeAttrName)) {
        loop.walk([&](triton::MakeTensorDescOp op) {
          if (auto forOp = op->getParentOfType<scf::ForOp>()) {
            descUpdateLoops.insert(forOp);
          }
        });
      }
    });

    // CoarseSchedule's notion of numStages is the maximuim loop-pipelining
    // stage + 1. See CoarseSchedule::deSerialize()
    const int coarseScheduleNumStages = numStages + 1;
    // +1 to make sure that overlapping of the next desc update and the oldest
    // inflight TMA load is safe
    const int numDescs = coarseScheduleNumStages + 1;
    triton::CoarseSchedule schedule(numDescs);

    for (auto loop : descUpdateLoops) {
      triton::lowerTMADescriptors(loop, schedule);
    }
  }
}
