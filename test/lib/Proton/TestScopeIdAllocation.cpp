#include "mlir/Pass/Pass.h"
#include "third_party/proton/dialect/include/Analysis/ScopeIdAllocation.h"

using namespace mlir;
using namespace triton::proton;

namespace {

struct TestScopeIdAllocationPass
    : public PassWrapper<TestScopeIdAllocationPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestScopeIdAllocationPass);

  TestScopeIdAllocationPass() = default;
  TestScopeIdAllocationPass(const TestScopeIdAllocationPass &other)
      : PassWrapper<TestScopeIdAllocationPass, OperationPass<ModuleOp>>(other) {
  }

  StringRef getArgument() const final {
    return "test-print-scope-id-allocation";
  }
  StringRef getDescription() const final {
    return "print the result of the scope id allocation pass";
  }

  void runOnOperation() override {
    auto &os = llvm::errs();
    ModuleOp moduleOp = getOperation();
    // Convert to std::string can remove quotes from opName
    ModuleScopeIdAllocation moduleScopeIdAllocation(moduleOp);
    moduleOp.walk([&](triton::FuncOp funcOp) {
      auto opName = SymbolTable::getSymbolName(funcOp).getValue().str();
      os << opName << "\n";
      funcOp.walk([&](RecordOp recordOp) {
        auto scopeId = moduleScopeIdAllocation.getOpScopeId(recordOp);
        os << "scope id = " << scopeId << "\n";
      });
    });
  }
};

} // namespace

namespace mlir {
namespace test {
namespace proton {
void registerTestScopeIdAllocationPass() {
  PassRegistration<TestScopeIdAllocationPass>();
}
} // namespace proton
} // namespace test
} // namespace mlir
