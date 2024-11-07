#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"

using namespace mlir;

namespace {

unsigned getScratchSize128(Operation *) { return 128; }
unsigned getScratchSizeInvalid(Operation *) {
  return mlir::triton::invalidAllocationSize;
}

enum class GetScratchSizeFunction {
  None,
  ValidConstant,
  Invalid,
};

struct TestAllocationPass
    : public PassWrapper<TestAllocationPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAllocationPass);

  TestAllocationPass() = default;
  TestAllocationPass(const TestAllocationPass &other)
      : PassWrapper<TestAllocationPass, OperationPass<ModuleOp>>(other) {}

  StringRef getArgument() const final { return "test-print-allocation"; }
  StringRef getDescription() const final {
    return "print the result of the allocation pass";
  }

  ModuleAllocation getModuleAllocation() {
    switch (getScratchSizeFunction) {
    case GetScratchSizeFunction::None:
      return {getOperation()};
    case GetScratchSizeFunction::ValidConstant:
      return {getOperation(), getScratchSize128};
    case GetScratchSizeFunction::Invalid:
      return {getOperation(), getScratchSizeInvalid};
    }
    llvm_unreachable("Unhandled case");
  }

  void runOnOperation() override {
    auto &os = llvm::errs();
    ModuleOp moduleOp = getOperation();
    // Convert to std::string can remove quotes from opName
    ModuleAllocation moduleAllocation = getModuleAllocation();
    moduleOp.walk([&](triton::FuncOp funcOp) {
      auto opName = SymbolTable::getSymbolName(funcOp).getValue().str();
      os << opName << "\n";
      auto *allocation = moduleAllocation.getFuncData(funcOp);
      funcOp.walk([&](Operation *op) {
        auto scratchBufferId = allocation->getBufferId(op);
        if (scratchBufferId != Allocation::InvalidBufferId) {
          size_t offset = allocation->getOffset(scratchBufferId);
          size_t size = allocation->getAllocatedSize(scratchBufferId);
          if (allocation->isVirtualBuffer(scratchBufferId))
            os << "virtual offset = " << offset << ", size = " << size << "\n";
          else
            os << "scratch offset = " << offset << ", size = " << size << "\n";
        }
        if (op->getNumResults() < 1)
          return;
        for (Value result : op->getResults()) {
          auto bufferId = allocation->getBufferId(result);
          if (bufferId != Allocation::InvalidBufferId) {
            size_t offset = allocation->getOffset(bufferId);
            size_t size = allocation->getAllocatedSize(bufferId);
            os << "offset = " << offset << ", size = " << size << "\n";
          }
        }
      });
      os << "size = " << allocation->getSharedMemorySize() << "\n";
    });
  }

  Option<GetScratchSizeFunction> getScratchSizeFunction{
      *this, "get-scratch-size-function",
      llvm::cl::desc("Custom scratch size function to use"),
      llvm::cl::init(GetScratchSizeFunction::None),
      llvm::cl::values(
          clEnumValN(GetScratchSizeFunction::None, "None", "None (default)"),
          clEnumValN(GetScratchSizeFunction::ValidConstant, "ValidConstant",
                     "ValidConstant"),
          clEnumValN(GetScratchSizeFunction::Invalid, "Invalid", "Invalid"))};
};

} // namespace

namespace mlir {
namespace test {
void registerTestAllocationPass() { PassRegistration<TestAllocationPass>(); }
} // namespace test
} // namespace mlir
