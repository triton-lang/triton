#include "DialectPlugin/DialectPluginDialect.h"
#include "DialectPlugin/DialectPluginOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

#include "DialectPlugin/DialectPluginPasses.h"

namespace mlir::triton::dialectplugin {
#define GEN_PASS_DEF_DIALECTPLUGINMAGICOP
#include "DialectPlugin/DialectPluginPasses.h.inc"

namespace {


LogicalResult PluginMagicOp(FunctionOpInterface func) {
  MLIRContext *context = func.getContext();
  Location loc = func->getLoc();
  OpBuilder builder(context);
  func.walk([&](mlir::triton::dialectplugin::MagicOp op) {
      builder.setInsertionPoint(op);
      auto a = op.getInput();
      auto newOp = arith::AddIOp::create(builder, loc, a, a);
      op->erase();
  });
  return success();
}

} // namespace

class DialectPluginMagicOpPass
    : public impl::DialectPluginMagicOpBase<
          DialectPluginMagicOpPass> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();
    Location loc = m->getLoc();
    FunctionOpInterface func = *m.getOps<FunctionOpInterface>().begin();
    if (failed(PluginMagicOp(func))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createDialectPluginMagicOpPass() {
  return std::make_unique<DialectPluginMagicOpPass>();
}

} // namespace mlir::triton::dialectplugin
