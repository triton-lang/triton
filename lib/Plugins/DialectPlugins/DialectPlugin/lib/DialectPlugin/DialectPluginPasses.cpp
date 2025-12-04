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

LogicalResult ConvertPluginMagicOp(FunctionOpInterface func) {
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

class ConvertPluginGPUToTritonGPUPass
    : public impl::DialectPluginMagicOpBase<
          ConvertPluginGPUToTritonGPUPass> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = &getContext();
    Location loc = m->getLoc();
    FunctionOpInterface func = *m.getOps<FunctionOpInterface>().begin();
    RewritePatternSet patterns(context);
    if (failed(ConvertPluginMagicOp(func))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createConvertPluginGPUToTritonGPUPass() {
  return std::make_unique<ConvertPluginGPUToTritonGPUPass>();
}

} // namespace mlir::triton::dialectplugin
