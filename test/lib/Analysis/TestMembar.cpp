#include "../third_party/nvidia/include/TritonNVIDIAGPUToLLVM/Utility.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
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
    registry.insert<triton::nvgpu::NVGPUDialect>();
  }

  void runOnOperation() override {
    Operation *operation = getOperation();
    ModuleOp moduleOp = cast<ModuleOp>(operation);
    ModuleAllocation allocation(moduleOp);
    if (moduleOp->hasAttr("ttg.target")) {
      int computeCapability = getNVIDIAComputeCapability(moduleOp);
      int ptxVersion = computeCapability;
      triton::NVIDIA::TargetInfo targetInfo(computeCapability, ptxVersion);
      allocation = ModuleAllocation(
          moduleOp,
          triton::nvidia_gpu::getNvidiaAllocationAnalysisScratchSizeFn(
              targetInfo));
      triton::nvidia_gpu::runClusterBarrierInsertion(allocation,
                                                     computeCapability);
    }
    ModuleMembarAnalysis membarPass(&allocation,
                                    mlir::triton::NVIDIA::canSkipBarSync);
    membarPass.run();
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestMembarPass() { PassRegistration<TestMembarPass>(); }
} // namespace test
} // namespace mlir
