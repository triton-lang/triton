#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"

using namespace mlir;

namespace {

struct TestMembarPass
    : public PassWrapper<TestMembarPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMembarPass);

  StringRef getArgument() const final { return "test-print-membar"; }
  StringRef getDescription() const final {
    return "print the result of the allocation pass";
  }

  void runOnOperation() override {
    Operation *operation = getOperation();
    auto &os = llvm::errs();
    // Convert to std::string can remove quotes from op_name
    auto opName = SymbolTable::getSymbolName(operation).getValue().str();
    os << opName << "\n";

    // Lower the module to the cf dialect
    auto *context = operation->getContext();
    RewritePatternSet scfPatterns(context);
    mlir::populateLoopToStdConversionPatterns(scfPatterns);
    mlir::ConversionTarget scfTarget(*context);
    scfTarget.addIllegalOp<scf::ForOp, scf::IfOp, scf::ParallelOp, scf::WhileOp,
                           scf::ExecuteRegionOp>();
    scfTarget.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(operation, scfTarget,
                                      std::move(scfPatterns))))
      return;

    // Print all ops after membar pass
    Allocation allocation(operation);
    MembarAnalysis membarPass(&allocation);
    membarPass.run();

    os << *operation << "\n";
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestMembarPass() { PassRegistration<TestMembarPass>(); }
} // namespace test
} // namespace mlir
