#include "PluginLowering/PluginLoweringDialect.h"
#include "PluginLowering/PluginLoweringOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "PluginLowering/PluginLoweringPasses.h"

namespace mlir::pluginlowering {
#define GEN_PASS_DEF_PLUGINLOWERINGSWITCHBARFOO
#define GEN_PASS_DEF_PLUGINLOWERINGLOWERFOOOP
#include "PluginLowering/PluginLoweringPasses.h.inc"

namespace {
class PluginLoweringSwitchBarFooRewriter
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

class PluginLoweringLowerFooOpRewriter
    : public OpRewritePattern<mlir::pluginlowering::FooOp> {
public:
  using OpRewritePattern<mlir::pluginlowering::FooOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::pluginlowering::FooOp op,
                                PatternRewriter &rewriter) const final {
    llvm::errs() << "FOUND FOO OP!!: ";
    op.dump();
    auto a = op.getInput();
    auto newOp = arith::AddIOp::create(rewriter, op.getLoc(), a, a);
    op->replaceAllUsesWith(newOp);
    return success();
  }
};

class PluginLoweringSwitchBarFoo
    : public impl::PluginLoweringSwitchBarFooBase<PluginLoweringSwitchBarFoo> {
public:
  using impl::PluginLoweringSwitchBarFooBase<
      PluginLoweringSwitchBarFoo>::PluginLoweringSwitchBarFooBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<PluginLoweringSwitchBarFooRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

class PluginLoweringLowerFooOp
    : public impl::PluginLoweringLowerFooOpBase<PluginLoweringLowerFooOp> {
public:
  using impl::PluginLoweringLowerFooOpBase<
      PluginLoweringLowerFooOp>::PluginLoweringLowerFooOpBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<PluginLoweringLowerFooOpRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::pluginlowering
