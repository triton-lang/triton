#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonInstrument/IR/Ops.cpp.inc"

#include "triton/Dialect/TritonInstrument/IR/OpsEnums.cpp.inc"

namespace mlir {
namespace triton {
namespace instrument {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

template <typename ViewOp, typename FPSanOp>
struct PushFPSanThroughViewPattern : public OpRewritePattern<ViewOp> {
  using OpRewritePattern<ViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ViewOp view,
                                PatternRewriter &rewriter) const override {
    auto fpsan = view->getOperand(0).template getDefiningOp<FPSanOp>();
    if (!fpsan)
      return failure();

    auto resultTy = dyn_cast<RankedTensorType>(view->getResult(0).getType());
    auto payloadTy = dyn_cast<RankedTensorType>(fpsan.getVal().getType());
    if (!resultTy || !payloadTy)
      return failure();

    auto payloadViewTy = resultTy.clone(payloadTy.getElementType());
    OperationState state(view.getLoc(), view->getName());
    state.addOperands(fpsan.getVal());
    state.addTypes(payloadViewTy);
    state.addAttributes(view->getAttrs());
    Operation *payloadView = rewriter.create(state);
    auto moved = FPSanOp::create(rewriter, view.getLoc(), resultTy,
                                 payloadView->getResult(0));
    rewriter.replaceOp(view, moved->getResults());
    return success();
  }
};

template <typename FPSanOp>
void addPushFPSanThroughViewPatterns(RewritePatternSet &patterns,
                                     MLIRContext *context) {
  patterns.add<PushFPSanThroughViewPattern<ttg::ConvertLayoutOp, FPSanOp>,
               PushFPSanThroughViewPattern<tt::TransOp, FPSanOp>,
               PushFPSanThroughViewPattern<tt::ReshapeOp, FPSanOp>,
               PushFPSanThroughViewPattern<tt::BroadcastOp, FPSanOp>,
               PushFPSanThroughViewPattern<tt::ExpandDimsOp, FPSanOp>>(context);
}

void ExperimentalFPSanEmbedOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  // view(embed(x)) -> embed(view(x))
  addPushFPSanThroughViewPatterns<ExperimentalFPSanEmbedOp>(patterns, context);
}

OpFoldResult ExperimentalFPSanEmbedOp::fold(FoldAdaptor adaptor) {
  if (auto unembed = getVal().getDefiningOp<ExperimentalFPSanUnembedOp>())
    if (unembed.getVal().getType() == getType())
      return unembed.getVal();
  return {};
}

void ExperimentalFPSanUnembedOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  // view(unembed(x)) -> unembed(view(x))
  addPushFPSanThroughViewPatterns<ExperimentalFPSanUnembedOp>(patterns,
                                                              context);
}

OpFoldResult ExperimentalFPSanUnembedOp::fold(FoldAdaptor adaptor) {
  if (auto embed = getVal().getDefiningOp<ExperimentalFPSanEmbedOp>())
    if (embed.getVal().getType() == getType())
      return embed.getVal();
  return {};
}

} // namespace instrument
} // namespace triton
} // namespace mlir
