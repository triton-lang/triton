#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "amd/lib/TritonAMDGPUToLLVM/AsyncUtility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
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
  void getDependentDialects(DialectRegistry &registry) const override {
    // AMDMembarAnalysis may emit rocdl.sched.barrier.
    registry.insert<ROCDL::ROCDLDialect>();
  }
  StringRef getDescription() const final {
    return "print the result of the membar analysis as run in the amdgpu "
           "backend";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    triton::AMD::annotateLocalLoadsSyncedViaAsyncWait(moduleOp);
    // Print all ops after membar pass
    ModuleAllocation allocation(moduleOp);
    ModuleMembarAnalysis membarPass(allocation, triton::AMD::membarFilter);
    membarPass.run<triton::AMD::AMDMembarAnalysis>();
  }
};

} // namespace

namespace mlir::test {
void registerTestAMDGPUMembarPass() {
  PassRegistration<TestAMDGPUMembarPass>();
}
} // namespace mlir::test
