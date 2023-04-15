#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Alias.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace {

struct TestAliasPass
    : public PassWrapper<TestAliasPass, OperationPass<triton::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAliasPass);

  static void print(StringRef name, SmallVector<std::string, 4> &vals,
                    raw_ostream &os) {
    if (vals.empty())
      return;
    os << name << " -> ";
    size_t i = 0;
    for (auto val : vals) {
      if (i != 0)
        os << ",";
      os << val;
      ++i;
    }
    os << "\n";
  }

  StringRef getArgument() const final { return "test-print-alias"; }
  StringRef getDescription() const final {
    return "print the result of the alias analysis pass";
  }

  void runOnOperation() override {
    Operation *operation = getOperation();
    auto &os = llvm::errs();
    auto opName = SymbolTable::getSymbolName(operation).getValue().str();
    os << opName << "\n";

    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    SharedMemoryAliasAnalysis *analysis =
        solver->load<SharedMemoryAliasAnalysis>();
    if (failed(solver->initializeAndRun(operation)))
      return signalPassFailure();

    AsmState state(operation->getParentOfType<ModuleOp>());
    // Get operation ids of value's aliases
    auto getAllocOpNames = [&](Value value) {
      dataflow::Lattice<AliasInfo> *latticeElement =
          analysis->getLatticeElement(value);
      SmallVector<std::string, 4> opNames;
      if (latticeElement) {
        auto &info = latticeElement->getValue();
        for (auto &alias : info.getAllocs()) {
          auto opName =
              getValueOperandName(alias.getDefiningOp()->getResult(0), state);
          opNames.push_back(std::move(opName));
        }
      }
      // Ensure deterministic output
      std::sort(opNames.begin(), opNames.end());
      return opNames;
    };

    operation->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op->getNumResults() < 1) {
        // cond br, br
        if (auto branch = dyn_cast<BranchOpInterface>(op)) {
          auto *block = branch->getBlock();
          for (auto arg : llvm::enumerate(block->getArguments())) {
            auto operand = block->getArgument(arg.index());
            auto opNames = getAllocOpNames(operand);
            auto argName = getValueOperandName(arg.value(), state);
            print(argName, opNames, os);
          }
        }
        return;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        for (auto arg : llvm::enumerate(forOp.getRegionIterArgs())) {
          auto operand = forOp.getOpOperandForRegionIterArg(arg.value()).get();
          auto opNames = getAllocOpNames(operand);
          auto argName = getValueOperandName(arg.value(), state);
          print(argName, opNames, os);
        }
      }
      for (auto result : llvm::enumerate(op->getResults())) {
        auto opNames = getAllocOpNames(result.value());
        auto resultName = getValueOperandName(result.value(), state);
        print(resultName, opNames, os);
      }
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestAliasPass() { PassRegistration<TestAliasPass>(); }
} // namespace test
} // namespace mlir
