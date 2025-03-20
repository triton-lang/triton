#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

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
  pm.addPass(createTritonGPULoadMMASpecialization());
  pm.addPass(createTritonGPURewritePartitionDependencies());
  pm.addPass(createSCCPPass());
  pm.addPass(createCSEPass());
  pm.addPass(createTritonGPUPartitionLoops());
  pm.addPass(createTritonGPULoopDCE());
  if (failed(runPipeline(pm, getOperation())))
    return signalPassFailure();
}
