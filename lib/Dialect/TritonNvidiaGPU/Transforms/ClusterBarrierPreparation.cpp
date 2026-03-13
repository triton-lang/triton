#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONGPUCLUSTERBARRIERPREPARATION
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

constexpr StringLiteral kClusterBarrierPreparationDoneAttr =
    "ttng.cluster_barrier_preparation_done";

class TritonGPUClusterBarrierPreparationPass
    : public impl::TritonGPUClusterBarrierPreparationBase<
          TritonGPUClusterBarrierPreparationPass> {
public:
  using impl::TritonGPUClusterBarrierPreparationBase<
      TritonGPUClusterBarrierPreparationPass>::
      TritonGPUClusterBarrierPreparationBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module->hasAttr(kClusterBarrierPreparationDoneAttr))
      return;

    bool hasMultiCTA = false;
    module.walk(
        [&](Operation *op) { hasMultiCTA |= ttg::lookupNumCTAs(op) > 1; });
    if (!hasMultiCTA) {
      module->setAttr(kClusterBarrierPreparationDoneAttr,
                      UnitAttr::get(module.getContext()));
      return;
    }

    ModuleAllocation allocation(module);
    runClusterBarrierInsertion(allocation, computeCapability);
    if (failed(runCrossCTAMBarrierInitSyncInsertion(allocation,
                                                    computeCapability)))
      return signalPassFailure();

    module->setAttr(kClusterBarrierPreparationDoneAttr,
                    UnitAttr::get(module.getContext()));
  }
};

} // namespace

} // namespace mlir::triton::nvidia_gpu
