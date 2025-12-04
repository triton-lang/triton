#include "LoweringDialectPlugin/LoweringDialectPluginDialect.h"
#include "LoweringDialectPlugin/LoweringDialectPluginOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

#include "LoweringDialectPlugin/LoweringDialectPluginPasses.h"

namespace mlir::triton::loweringdialectplugin {
#define GEN_PASS_DEF_LOWERINGDIALECTPLUGINMAGICOP
#include "LoweringDialectPlugin/LoweringDialectPluginPasses.h.inc"

namespace {


LogicalResult PluginMagicOpLowering(FunctionOpInterface func) {
  MLIRContext *context = func.getContext();
  Location loc = func->getLoc();
  OpBuilder builder(context);
  func.walk([&](mlir::triton::loweringdialectplugin::MagicOp op) {
      builder.setInsertionPoint(op);
      auto a = op.getInput();
      auto newOp = arith::AddIOp::create(builder, loc, a, a);
      op->erase();
  });
  return success();
}

} // namespace

class LoweringDialectPluginMagicOpPass
    : public impl::LoweringDialectPluginMagicOpBase<
          LoweringDialectPluginMagicOpPass> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();
    Location loc = m->getLoc();
    FunctionOpInterface func = *m.getOps<FunctionOpInterface>().begin();
    if (failed(PluginMagicOpLowering(func))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createLoweringDialectPluginMagicOpPass() {
  return std::make_unique<LoweringDialectPluginMagicOpPass>();
}

} // namespace mlir::triton::loweringdialectplugin
