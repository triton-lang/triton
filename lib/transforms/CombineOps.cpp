#include "triton/transforms/Passes.h"
#include <memory>

using namespace mlir;

namespace {
// <patterns>
struct CombineDotOp : public RewritePattern {
  explicit CombineDotOp(MLIRContext *context)
    : RewritePattern(/*rootName*/FuncOp::getOperationName(), /*Benefit*/1, context);

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    //
    
  }
};
// </patterns>

/// Passes.td (?)
struct CombineOpsPass { // : public mlir::OperationPass<FuncOp>
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);

    patterns.add<CombineDotOp>();
    patterns.add<CombineSelectMaskedLoadOp>();
    patterns.add<CombineGEPOp>();
    patterns.add<CombineReduceOp>();

    // TODO: populate xxx Patter(?)

    // TODO: should be use applyPartialConversion(...) ?
    if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  };
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::triton::createCombineOpsPass() {
  return std::make_unique<CombineOpsPass>();
}
