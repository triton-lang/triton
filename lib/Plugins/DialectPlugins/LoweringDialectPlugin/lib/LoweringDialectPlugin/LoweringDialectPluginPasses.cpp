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
#define GEN_PASS_DEF_LOWERINGDIALECTPLUGINMAGICOPINSERTER
#define GEN_PASS_DEF_LOWERINGDIALECTPLUGINMAGICOP
#include "LoweringDialectPlugin/LoweringDialectPluginPasses.h.inc"

namespace {
// class LoweringDialectPluginMagicOpInserterRewriter
//     : public OpRewritePattern<func::FuncOp> {
// public:
//   using OpRewritePattern<func::FuncOp>::OpRewritePattern;
//   LogicalResult matchAndRewrite(func::FuncOp op,
//                                 PatternRewriter &rewriter) const final {
//     // Note: This is what I want this pass to do, no luck so far:
//     // auto magicOp = mlir::triton::loweringdialectplugin::MagicOp::create(
//     //     builder, op.getLoc(), op.getOpResult(0));
//     // op.replaceAllUsesWith(magicOp);
//     if (op.getSymName() == "bar") {
//       rewriter.modifyOpInPlace(op, [&op]() { op.setSymName("foo"); });
//       return success();
//     }
//     return failure();
//   }
// };

class LoweringDialectPluginMagicOpRewriter
    : public OpRewritePattern<mlir::triton::loweringdialectplugin::MagicOp> {
public:
  using OpRewritePattern<
      mlir::triton::loweringdialectplugin::MagicOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::triton::loweringdialectplugin::MagicOp op,
                                PatternRewriter &rewriter) const final {
    auto a = op.getInput();
    auto newOp = arith::AddIOp::create(rewriter, op.getLoc(), a, a);
    op->replaceAllUsesWith(newOp);
    return success();
  }
};

// class LoweringDialectPluginMagicOpInserter
//     : public impl::LoweringDialectPluginMagicOpInserterBase<
//           LoweringDialectPluginMagicOpInserter> {
// public:
//   using impl::LoweringDialectPluginMagicOpInserterBase<
//       LoweringDialectPluginMagicOpInserter>::LoweringDialectPluginMagicOpInserterBase;
//   void runOnOperation() final {
//     RewritePatternSet patterns(&getContext());
//     patterns.add<LoweringDialectPluginMagicOpInserterRewriter>(&getContext());
//     FrozenRewritePatternSet patternSet(std::move(patterns));
//     if (failed(applyPatternsGreedily(getOperation(), patternSet)))
//       signalPassFailure();
//   }
// };

class LoweringDialectPluginMagicOp
    : public impl::LoweringDialectPluginMagicOpBase<
          LoweringDialectPluginMagicOp> {
public:
  using impl::LoweringDialectPluginMagicOpBase<
      LoweringDialectPluginMagicOp>::LoweringDialectPluginMagicOpBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<LoweringDialectPluginMagicOpRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::triton::loweringdialectplugin
