#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "amd/lib/TritonAMDGPUToLLVM/AsyncUtility.h"
#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"

using namespace mlir;

namespace {

struct TestAMDGPUMembarPass : public OperationPass<ModuleOp> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAMDGPUMembarPass);

  StringRef getArgument() const final { return "test-tritonamdgpu-membar"; }
  StringRef getDescription() const final {
    return "print the result of the membar analysis as run in the amdgpu "
           "backend";
  }
  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<TestAMDGPUMembarPass>(
        *static_cast<const TestAMDGPUMembarPass *>(this));
  }
  StringRef getName() const override { return "TestAMDGPUMembarPass"; }
  TestAMDGPUMembarPass()
      : OperationPass<ModuleOp>(TypeID::get<TestAMDGPUMembarPass>()) {}
  TestAMDGPUMembarPass(const TestAMDGPUMembarPass &other)
      : OperationPass<ModuleOp>(other) {}
  TestAMDGPUMembarPass &operator=(const TestAMDGPUMembarPass &) = delete;
  TestAMDGPUMembarPass(TestAMDGPUMembarPass &&) = delete;
  TestAMDGPUMembarPass &operator=(TestAMDGPUMembarPass &&) = delete;
  ~TestAMDGPUMembarPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ROCDL::ROCDLDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    triton::AMD::annotateLocalLoadsSyncedViaAsyncWait(moduleOp);
    // Print all ops after membar pass
    ModuleAllocation allocation(moduleOp);
    MembarInsertBarrierFn insertBarrier = nullptr;
    auto isaFamily = triton::AMD::deduceISAFamily(archGenerationName);
    if (isCDNA(isaFamily))
      insertBarrier = triton::AMD::membarInsertBarrierCDNA;
    ModuleMembarAnalysis membarPass(&allocation, triton::AMD::membarFilter,
                                    insertBarrier);
    membarPass.run();
  }

protected:
  Pass::Option<std::string> archGenerationName{
      *this, "arch-generation-name",
      llvm::cl::desc("GFX generation name of target device."),
      llvm::cl::init(std::string{})};
};

} // namespace

namespace mlir::test {
void registerTestAMDGPUMembarPass() {
  PassRegistration<TestAMDGPUMembarPass>();
}
} // namespace mlir::test
