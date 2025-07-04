#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestAMDMemDescSubviewSplit
    : public PassWrapper<TestAMDMemDescSubviewSplit, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAMDMemDescSubviewSplit);

  TestAMDMemDescSubviewSplit() = default;
  TestAMDMemDescSubviewSplit(const TestAMDMemDescSubviewSplit &) {}

  StringRef getArgument() const final {
    return "test-tritonamdgpu-split-memdescsubview";
  }
  StringRef getDescription() const final {
    return "test splitting ttg::MemDescSubviewOp using the AMD backend";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    Location loc = mod->getLoc();
    IRRewriter rewriter(mod->getContext());

    auto numSplits = llvm::to_vector(clPassArg);
    mod->walk([&](triton::gpu::MemDescSubviewOp view) {
      rewriter.setInsertionPointAfter(view);
      triton::AMD::splitMemDescView(rewriter, loc, view, numSplits);
    });
  }

  ListOption<unsigned> clPassArg{
      *this, "num-splits", llvm::cl::desc("number of MemDescSubview splits"),
      llvm::cl::Required};
};
} // namespace

namespace mlir::test {
void registerTestAMDCommonUtils() {
  PassRegistration<TestAMDMemDescSubviewSplit>();
}
} // namespace mlir::test
