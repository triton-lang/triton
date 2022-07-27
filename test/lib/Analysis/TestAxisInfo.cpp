#include "mlir/Pass/Pass.h"
#include "triton/Analysis/AxisInfo.h"

using namespace mlir;

namespace {

struct TestAxisInfoPass
    : public PassWrapper<TestAxisInfoPass, OperationPass<FuncOp>> {

  // LLVM15+
  // MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAlignmentPass);

  void print(const std::string &name, raw_ostream &os, ArrayRef<int> vals) {
    os << name << ": [";
    for (size_t d = 0; d < vals.size(); d++) {
      if (d != 0)
        os << ", ";
      os << vals[d];
    }
    os << "]";
  }

  StringRef getArgument() const final { return "test-print-alignment"; }
  StringRef getDescription() const final {
    return "print the result of the alignment analysis pass";
  }

  void runOnOperation() override {
    Operation *operation = getOperation();
    auto &os = llvm::errs();
    os << "Testing: " << operation->getName() << "\n";
    AxisInfoAnalysis analysis(&getContext());
    analysis.run(operation);
    operation->walk([&](Operation *op) {
      if (op->getNumResults() < 1)
        return;
      for (Value result : op->getResults()) {
        // std::ostringstream oss;
        // result.print(oss);
        // os << " => ";
        LatticeElement<AxisInfo> *latticeElement =
            analysis.lookupLatticeElement(result);
        if (!latticeElement) {
          os << "None\n";
          return;
        }
        AxisInfo &info = latticeElement->getValue();
        print("Contiguity", os, info.getContiguity());
        os << " ; ";
        print("Divisibility", os, info.getDivisibility());
        os << " ; ";
        print("Constancy", os, info.getConstancy());
        os << " ( ";
        result.print(os);
        os << " ) ";
        os << "\n";
      }
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestAlignmentPass() { PassRegistration<TestAxisInfoPass>(); }
} // namespace test
} // namespace mlir
