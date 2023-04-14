#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"

using namespace mlir;

namespace {

struct TestMembarPass
    : public PassWrapper<TestMembarPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMembarPass);

  StringRef getArgument() const final { return "test-print-membar"; }
  StringRef getDescription() const final {
    return "print the result of the allocation pass";
  }

  void runOnOperation() override {
    Operation *operation = getOperation();
    ModuleOp moduleOp = cast<ModuleOp>(operation);
    // Print all ops after membar pass
    ModuleAllocation allocation(moduleOp);
    MembarAnalysis membarPass(&allocation);
    membarPass.run();
    // Print all ops before membar pass
    auto &os = llvm::errs();
    moduleOp.walk([&](triton::FuncOp funcOp) {
      // Convert to std::string can remove quotes from op_name
      auto opName = SymbolTable::getSymbolName(funcOp).getValue().str();
      os << opName << "\n";
      funcOp.walk([&](Operation *op) { os << *op << "\n"; });
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestMembarPass() { PassRegistration<TestMembarPass>(); }
} // namespace test
} // namespace mlir
