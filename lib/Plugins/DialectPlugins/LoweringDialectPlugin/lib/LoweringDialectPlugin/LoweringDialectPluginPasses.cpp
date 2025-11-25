#include "LoweringDialectPlugin/LoweringDialectPluginDialect.h"
#include "LoweringDialectPlugin/LoweringDialectPluginOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

#include "LoweringDialectPlugin/LoweringDialectPluginPasses.h"

namespace mlir::triton::loweringdialectplugin {
#define GEN_PASS_DEF_LOWERINGDIALECTPLUGINSWITCHBARFOO
#define GEN_PASS_DEF_LOWERINGDIALECTPLUGINLOWERFOOOP
#include "LoweringDialectPlugin/LoweringDialectPluginPasses.h.inc"

namespace {
class LoweringDialectPluginSwitchBarFooRewriter
    : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getSymName() == "bar") {
      rewriter.modifyOpInPlace(op, [&op]() { op.setSymName("foo"); });
      return success();
    }
    return failure();
  }
};

class LoweringDialectPluginLowerFooOpRewriter
    : public OpRewritePattern<mlir::triton::loweringdialectplugin::FooOp> {
public:
  using OpRewritePattern<
      mlir::triton::loweringdialectplugin::FooOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::triton::loweringdialectplugin::FooOp op,
                                PatternRewriter &rewriter) const final {
    llvm::errs() << "FOUND FOO OP!!: ";
    op.dump();
    auto a = op.getInput();
    auto newOp = arith::AddIOp::create(rewriter, op.getLoc(), a, a);
    op->replaceAllUsesWith(newOp);
    return success();
  }
};

class LoweringDialectPluginSwitchBarFoo
    : public impl::LoweringDialectPluginSwitchBarFooBase<
          LoweringDialectPluginSwitchBarFoo> {
public:
  using impl::LoweringDialectPluginSwitchBarFooBase<
      LoweringDialectPluginSwitchBarFoo>::LoweringDialectPluginSwitchBarFooBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<LoweringDialectPluginSwitchBarFooRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

class LoweringDialectPluginLowerFooOp
    : public impl::LoweringDialectPluginLowerFooOpBase<
          LoweringDialectPluginLowerFooOp> {
public:
  using impl::LoweringDialectPluginLowerFooOpBase<
      LoweringDialectPluginLowerFooOp>::LoweringDialectPluginLowerFooOpBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<LoweringDialectPluginLowerFooOpRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::triton::loweringdialectplugin
