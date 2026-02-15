#pragma once

#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/AxisInfo.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::test {

struct TestAxisInfoPass
    : public PassWrapper<TestAxisInfoPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAxisInfoPass);

  StringRef getArgument() const override { return "test-print-alignment"; }
  StringRef getDescription() const final {
    return "print the result of the alignment analysis pass";
  }

  void runOnOperation() override {
    Operation *operation = this->getOperation();
    ModuleOp moduleOp = cast<ModuleOp>(operation);
    auto moduleAxisInfoAnalysis = getAnalysis(moduleOp);
    moduleOp.walk([&](FuncOp funcOp) {
      funcOp.walk([&](Operation *op) {
        if (op->getNumResults() < 1)
          return;
        for (Value result : op->getResults()) {
          InFlightDiagnostic diag = mlir::emitRemark(op->getLoc());
          diag << result;
          diag << " => ";
          auto *axisInfo = moduleAxisInfoAnalysis.getAxisInfo(result);
          if (axisInfo) {
            std::string str;
            llvm::raw_string_ostream os(str);
            axisInfo->print(os);
            diag << str;
          }
        }
      });
    });
  }

protected:
  virtual ModuleAxisInfoAnalysis getAnalysis(ModuleOp moduleOp) const {
    return ModuleAxisInfoAnalysis(moduleOp);
  }
};

} // namespace mlir::test
