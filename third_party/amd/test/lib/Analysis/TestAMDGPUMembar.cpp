#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"

using namespace mlir;

namespace {

struct TestAMDGPUMembarPass
    : public PassWrapper<TestAMDGPUMembarPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAMDGPUMembarPass);

  StringRef getArgument() const final { return "test-tritonamdgpu-membar"; }
  StringRef getDescription() const final {
    return "print the result of the membar analysis as run in the amdgpu "
           "backend";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    // Print all ops after membar pass
    ModuleAllocation allocation(moduleOp);
    ModuleMembarAnalysis membarPass(&allocation,
                                    mlir::triton::AMD::membarFilter);
    membarPass.run();
  }
};

} // namespace

namespace mlir::test {
void registerTestAMDGPUMembarPass() {
  PassRegistration<TestAMDGPUMembarPass>();
}
} // namespace mlir::test
