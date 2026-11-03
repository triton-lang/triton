#include "../third_party/nvidia/include/TritonNVIDIAGPUToLLVM/Utility.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.h"

using namespace mlir;

namespace {

struct TestMembarPass
    : public PassWrapper<TestMembarPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMembarPass);

  StringRef getArgument() const final { return "test-print-membar"; }
  StringRef getDescription() const final {
    return "print the result of the allocation pass";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
  }

  void runOnOperation() override {
    Operation *operation = getOperation();
    ModuleOp moduleOp = cast<ModuleOp>(operation);
    ModuleAllocation allocation(moduleOp);
    int computeCapability = 0;
    if (moduleOp->hasAttr("ttg.target")) {
      computeCapability = getNVIDIAComputeCapability(moduleOp);
      int ptxVersion = computeCapability;
      triton::NVIDIA::TargetInfo targetInfo(computeCapability, ptxVersion);
      allocation = ModuleAllocation(
          moduleOp,
          triton::nvidia_gpu::getNvidiaAllocationAnalysisScratchSizeFn(
              targetInfo));
    }
    auto solver = createDataFlowSolver();
    auto *regions = solver->load<triton::BufferRegionAnalysis>(
        triton::BufferRegionAnalysis::Mode::AllMemory, &allocation);
    if (failed(solver->initializeAndRun(moduleOp)))
      llvm::report_fatal_error("failed to analyze allocated buffer regions");

    triton::nvidia_gpu::runClusterBarrierInsertion(allocation,
                                                   computeCapability, *regions);
    if (failed(triton::nvidia_gpu::runCrossCTAMBarrierInitSyncInsertion(
            allocation, computeCapability)))
      return signalPassFailure();
    ModuleMembarAnalysis membarPass(allocation,
                                    mlir::triton::NVIDIA::canSkipBarSync);
    membarPass.runAnalysis<MembarAnalysis>(*regions);
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestMembarPass() { PassRegistration<TestMembarPass>(); }
} // namespace test
} // namespace mlir
