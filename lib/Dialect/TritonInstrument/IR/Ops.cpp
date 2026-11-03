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

LogicalResult DotI8Op::verify() {
  auto aEnc =
      dyn_cast<ttg::DotOperandEncodingAttr>(getA().getType().getEncoding());
  auto bEnc =
      dyn_cast<ttg::DotOperandEncodingAttr>(getB().getType().getEncoding());
  if (!aEnc || !bEnc)
    return emitError("requires dot operand encodings for A and B");

  auto aMma = dyn_cast<ttg::NvidiaMmaEncodingAttr>(aEnc.getParent());
  auto bMma = dyn_cast<ttg::NvidiaMmaEncodingAttr>(bEnc.getParent());
  auto dMma =
      dyn_cast<ttg::NvidiaMmaEncodingAttr>(getD().getType().getEncoding());
  if (!aMma || !bMma || !dMma || aMma.getVersionMajor() != 2 ||
      bMma.getVersionMajor() != 2 || dMma.getVersionMajor() != 2)
    return emitError("requires NVIDIA MMAv2 operand and result layouts");
  if (aMma != bMma || aMma != dMma)
    return emitError("requires matching NVIDIA MMAv2 layouts");

  auto layoutInterface =
      cast<tt::DialectInferLayoutInterface>(&dMma.getDialect());
  return layoutInterface->verifyDotOpEncodingCompatibility(getOperation(), aEnc,
                                                           bEnc);
}

LogicalResult ExperimentalLocalGatherOp::verify() {
  auto srcTy = getSrc().getType();
  auto indicesTy = cast<RankedTensorType>(getIndices().getType());
  auto dstTy = cast<RankedTensorType>(getType());
  unsigned axis = getAxis();

  if (!isa<ttg::SharedEncodingTrait>(srcTy.getEncoding()))
    return emitError("source must have shared memory encoding");

  if (!indicesTy.getElementType().isInteger())
    return emitError("indices must have integer element type");

  if (dstTy.getShape() != indicesTy.getShape())
    return emitError("result shape must match indices shape");

  if (srcTy.getRank() != indicesTy.getRank())
    return emitError("source and indices must have the same rank");

  if (axis >= srcTy.getRank())
    return emitError("axis ")
           << axis << " is out of bounds for source rank " << srcTy.getRank();

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitError("result element type must match source element type");

  if (indicesTy.getEncoding() != dstTy.getEncoding())
    return emitError("indices and result must have the same layout");

  if (static_cast<int64_t>(getOffsets().size()) != srcTy.getRank())
    return emitError("offset count must match source rank");

  return success();
}

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
