#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

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
  // Collect for loops to warp specialize. This pass expects the loop to already
  // be scheduled.
  SmallVector<scf::ForOp> loops;
  getOperation().walk([&](scf::ForOp loop) {
    if (llvm::any_of(loop.getOps(), [](Operation &op) {
          return op.hasAttr(kLoopStageAttrName);
        }))
      loops.push_back(loop);
  });

  for (scf::ForOp loop : loops) {
    int loopNumStages = numStages;
    if (auto attr = loop->getAttrOfType<IntegerAttr>(kNumStagesAttrName))
      loopNumStages = attr.getInt();

    PipeliningOption result;
    // Preprocess the loop by lowering async ops.
    if (!preProcessLoopAndGetSchedule(loop, loopNumStages, result))
      continue;
  }
}
