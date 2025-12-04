#include "DialectPlugin/DialectPluginDialect.h"
#include "DialectPlugin/DialectPluginOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "DialectPlugin/DialectPluginPasses.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::dialectplugin {
#define GEN_PASS_DEF_DIALECTPLUGINMAGICOP
#include "DialectPlugin/DialectPluginPasses.h.inc"
}

namespace {

LogicalResult ConvertPluginMagicOp(FunctionOpInterface func) {
  MLIRContext *context = func.getContext();
  Location loc = func->getLoc();
  OpBuilder builder(context);
  func.walk([&](mlir::triton::dialectplugin::MagicOp op) {
      builder.setInsertionPoint(op);
      auto a = op.getInput();
      auto newOp = mlir::LLVM::ZeroOp::create(builder, loc, a.getType());
      op.replaceAllUsesWith(newOp.getResult());
      op.erase();
  });
  return success();
}

} // namespace

class ConvertPluginGPUToLLVMPass
    : public mlir::triton::dialectplugin::impl::DialectPluginMagicOpBase<
          ConvertPluginGPUToLLVMPass> {
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

namespace mlir::triton::dialectplugin {
std::unique_ptr<OperationPass<ModuleOp>> createConvertPluginGPUToLLVMPass() {
  return std::make_unique<ConvertPluginGPUToLLVMPass>();
}

} // namespace mlir::triton::dialectplugin
