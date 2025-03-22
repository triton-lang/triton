#include "mlir/Dialect/Arith/Transforms/Passes.h"
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
  // `int-range-optimizations` combines SCCP with integer range analysis. It's
  // good at cleaning up loop arithmetic.
  pm.addPass(arith::createIntRangeOptimizationsPass());
  pm.addPass(createCSEPass());
  pm.addPass(createTritonGPUPartitionLoops());
  // Strip dead code that might have been created by loop partitioning.
  pm.addPass(createTritonGPULoopDCE());
  if (failed(runPipeline(pm, getOperation())))
    return signalPassFailure();
}
