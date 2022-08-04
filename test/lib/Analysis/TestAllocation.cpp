#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"

using namespace mlir;

namespace {

struct TestAllocationPass
    : public PassWrapper<TestAllocationPass, OperationPass<FuncOp>> {

  // LLVM15+
  // MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAllocationPass);

  StringRef getArgument() const final { return "test-print-allocation"; }
  StringRef getDescription() const final {
    return "print the result of the allocation pass";
  }

  void runOnOperation() override {
    Operation *operation = getOperation();
    auto &os = llvm::errs();
    os << "Testing: " << operation->getName() << "\n";
    AllocationAnalysis analysis(operation);
    operation->walk([&](Operation *op) {
      if (op->getNumResults() < 1)
        return;
      for (Value result : op->getResults()) {
        Type type = result.getType();
        if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
          Attribute encoding = tensorType.getEncoding();
          if (encoding.isa<triton::gpu::TritonGPUSharedEncodingAttr>()) {
            size_t offset = analysis.getOffset(result);
            size_t size = analysis.getAllocatedSize(result);
            os << "offset = " << offset << ", size = " << size << "\n";
          }
        }
      }
    });
    os << "size = " << analysis.getSharedMemorySize() << "\n";
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestAllocationPass() { PassRegistration<TestAllocationPass>(); }
} // namespace test
} // namespace mlir
