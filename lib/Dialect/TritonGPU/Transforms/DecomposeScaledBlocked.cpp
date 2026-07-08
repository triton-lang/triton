#include "triton/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.h"

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace mlir::triton::gpu {

SmallVector<int, 2> DecomposeScaledBlocked::getTransposeOrder(int rank) {
  assert(rank >= 2);
  auto transOrder = llvm::to_vector<2>(llvm::seq<int>(rank - 2));
  transOrder.push_back(rank - 1);
  transOrder.push_back(rank - 2);
  return transOrder;
}

LogicalResult
DecomposeScaledBlocked::matchAndRewrite(DotScaledOp scaledDotOp,
                                        PatternRewriter &rewriter) const {
  if (isa_and_nonnull<MmaEncodingTrait>(
          scaledDotOp.getResult().getType().getEncoding()))
    return failure();

  // Types
  auto computeType = getComputeType(scaledDotOp.getAElemType(),
                                    scaledDotOp.getBElemType(), rewriter);

  auto scaledA = scaleArg(rewriter, scaledDotOp, 0, computeType);
  scaledA = cvtDotOperand(rewriter, scaledDotOp, 0, scaledA);
  auto scaledB = scaleArg(rewriter, scaledDotOp, 1, computeType);
  scaledB = cvtDotOperand(rewriter, scaledDotOp, 1, scaledB);
  auto newDot = DotOp::create(rewriter, scaledDotOp.getLoc(), scaledA, scaledB,
                              scaledDotOp.getC());

  rewriter.replaceOpWithNewOp<ConvertLayoutOp>(scaledDotOp,
                                               scaledDotOp.getType(), newDot);
  return success();
}

FloatType
DecomposeScaledBlocked::getComputeType(ScaleDotElemType aType,
                                       ScaleDotElemType bType,
                                       PatternRewriter &rewriter) const {
  if (aType == ScaleDotElemType::FP16 || bType == ScaleDotElemType::FP16)
    return rewriter.getF16Type();
  return rewriter.getBF16Type();
}

TypedValue<RankedTensorType>
DecomposeScaledBlocked::scaleTo16(PatternRewriter &rewriter,
                                  TypedValue<RankedTensorType> scale,
                                  FloatType computeType) const {
  auto loc = scale.getLoc();
  auto scaleTy = scale.getType();
  assert(computeType == rewriter.getBF16Type() ||
         computeType == rewriter.getF16Type());

  if (isa<FloatType>(scaleTy.getElementType())) {
    auto scaleType = scaleTy.clone(computeType);
    return cast<TypedValue<RankedTensorType>>(
        FpToFpOp::create(rewriter, loc, scaleType, scale).getResult());
  }

  // Choose an fp type that can fit the scale value.
  FloatType largeFpType = computeType == rewriter.getF16Type()
                              ? rewriter.getF32Type()
                              : computeType;
  int intWidth = largeFpType.getIntOrFloatBitWidth();
  auto intType = rewriter.getIntegerType(intWidth);

  auto zexted =
      arith::ExtUIOp::create(rewriter, loc, scaleTy.clone(intType), scale);
  // getFpMantissaWidth() returns the number of bits in the mantissa plus the
  // sign bit!
  int shiftValue = largeFpType.getFPMantissaWidth() - 1;
  auto shiftConst =
      arith::ConstantIntOp::create(rewriter, loc, shiftValue, intWidth);
  auto shift =
      SplatOp::create(rewriter, loc, scaleTy.clone(intType), shiftConst);
  auto shlRes = arith::ShLIOp::create(rewriter, loc, zexted, shift);
  Value scaleFP =
      BitcastOp::create(rewriter, loc, scaleTy.clone(largeFpType), shlRes);
  if (largeFpType != computeType) {
    scaleFP = arith::TruncFOp::create(rewriter, loc, scaleTy.clone(computeType),
                                      scaleFP);
  }
  return cast<TypedValue<RankedTensorType>>(scaleFP);
}

TypedValue<RankedTensorType> DecomposeScaledBlocked::broadcastScale(
    PatternRewriter &rewriter, DotScaledOp scaledDotOp,
    TypedValue<RankedTensorType> scale, int dim, Attribute dstEncoding) const {
  auto loc = scale.getLoc();
  auto scaleTy = scale.getType();
  int32_t scaleFactor = scaledDotOp.deduceScaleFactor();

  // We want to broadcast the scales along dim. To do this:
  //   1. Introduce a size 1 dimension right after dim
  //   2. Broadcast the new dim to the scale factor
  //   3. Reshape it to get the result shape.

  // Compute the shape after each step.
  auto expandedShape = to_vector(scaleTy.getShape());
  expandedShape.insert(expandedShape.begin() + dim + 1, 1);
  auto broadcastShape = expandedShape;
  broadcastShape[dim + 1] = scaleFactor;
  auto resultShape = to_vector(scaleTy.getShape());
  resultShape[dim] *= scaleFactor;

  // It is more efficient to perform a layout conversion before broadcasting,
  // since there are fewer elements. Infer the source encoding that will produce
  // dstEncoding after the broadcast and reshape.
  auto interface = cast<DialectInferLayoutInterface>(&dstEncoding.getDialect());
  Attribute broadcastEncoding;
  auto result = interface->inferReshapeOpEncoding(
      resultShape, dstEncoding, broadcastShape, broadcastEncoding,
      /*allowReorder=*/false, loc);
  assert(succeeded(result));
  Attribute srcEncoding;
  result = interface->inferReshapeOpEncoding(expandedShape, broadcastEncoding,
                                             scaleTy.getShape(), srcEncoding,
                                             /*allowReorder=*/false, loc);
  assert(succeeded(result));

  auto srcType = scaleTy.cloneWithEncoding(srcEncoding);

  // Convert the scales to the desired type.
  scale = ConvertLayoutOp::create(rewriter, loc, srcType, scale);

  // Introduce the new dimension.
  auto expandType = RankedTensorType::get(
      expandedShape, scaleTy.getElementType(), broadcastEncoding);
  // We know this layout avoids having any layout conversions after we expand
  // the scales, so mark the layout as efficient. Otherwise, forward layout
  // propagation may try to sink the convert layout.
  auto expandScale = ReshapeOp::create(
      rewriter, loc, expandType, scale,
      /*allow_reorder=*/nullptr, /*efficient_layout=*/rewriter.getUnitAttr());
  // Broadcast the dimension to the microscaling factor.
  auto broadcastType = RankedTensorType::get(
      broadcastShape, scaleTy.getElementType(), broadcastEncoding);
  auto broadcastScale =
      BroadcastOp::create(rewriter, loc, broadcastType, expandScale);
  auto resultType =
      RankedTensorType::get(resultShape, scaleTy.getElementType(), dstEncoding);
  // Reshape to fold in the broadcasted dimension and get the final result.
  return ReshapeOp::create(rewriter, loc, resultType, broadcastScale);
}

TypedValue<RankedTensorType> DecomposeScaledBlocked::maskNan(
    PatternRewriter &rewriter, DotScaledOp scaledDotOp,
    TypedValue<RankedTensorType> mxfp, TypedValue<RankedTensorType> scale,
    int dim) const {
  // Skip NaN checks if fastMath
  if (scaledDotOp.getFastMath())
    return mxfp;

  // Implement tl.where(scale == 0xFF, float("nan"), mxfp)
  auto loc = scale.getLoc();

  // Scale is NaN
  auto scaleTy = scale.getType();
  TypedValue<RankedTensorType> scaleIsNan;
  if (isa<FloatType>(scaleTy.getElementType())) {
    auto computeType = cast<FloatType>(mxfp.getType().getElementType());
    auto scaleFp = scaleTo16(rewriter, scale, computeType);
    scaleIsNan = cast<TypedValue<RankedTensorType>>(
        arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::UNO, scaleFp,
                              scaleFp)
            .getResult());
  } else {
    auto constFF = arith::ConstantOp::create(
        rewriter, loc, scaleTy,
        DenseElementsAttr::get(scaleTy,
                               APInt(scaleTy.getElementTypeBitWidth(), 0xff)));
    scaleIsNan = cast<TypedValue<RankedTensorType>>(
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, scale,
                              constFF)
            .getResult());
  }
  auto cond = broadcastScale(rewriter, scaledDotOp, scaleIsNan, dim,
                             mxfp.getType().getEncoding());

  // Create NaN
  auto mxfpTy = mxfp.getType();
  auto nan = APFloat::getNaN(
      cast<FloatType>(mxfpTy.getElementType()).getFloatSemantics());
  auto constNan = arith::ConstantOp::create(
      rewriter, loc, mxfpTy, DenseElementsAttr::get(mxfpTy, nan));

  auto result = arith::SelectOp::create(rewriter, loc, cond, constNan, mxfp);
  return cast<TypedValue<RankedTensorType>>(result.getResult());
}

TypedValue<RankedTensorType>
DecomposeScaledBlocked::scaleArg(PatternRewriter &rewriter,
                                 DotScaledOp scaledDotOp, int opIdx,
                                 FloatType computeType) const {
  auto v = opIdx == 0 ? scaledDotOp.getA() : scaledDotOp.getB();
  auto scale = opIdx == 0 ? scaledDotOp.getAScale() : scaledDotOp.getBScale();
  auto isFp4 =
      ScaleDotElemType::E2M1 ==
      (opIdx == 0 ? scaledDotOp.getAElemType() : scaledDotOp.getBElemType());

  auto loc = v.getLoc();
  auto rank = v.getType().getRank();
  auto kDim = opIdx == 0 ? rank - 1 : rank - 2;

  // 0) Upcast value to computeType (fp16/bf16)
  if (isFp4) {
    bool kPack =
        opIdx == 0 ? scaledDotOp.getLhsKPack() : scaledDotOp.getRhsKPack();
    int packedDim = kPack ? kDim : (opIdx == 0 ? rank - 2 : rank - 1);
    v = Fp4ToFpOp::create(rewriter, loc, v, computeType, packedDim);
  } else {
    auto vType16 = v.getType().clone(computeType);
    v = cast<TypedValue<RankedTensorType>>(
        FpToFpOp::create(rewriter, loc, vType16, v).getResult());
  }
  if (!scale)
    return v;

  // 1) Cast scale to fp16/bf16, broadcast it and convert its layout
  auto reshapeScale = extendAndBroadcastScale(rewriter, scaledDotOp, scale,
                                              computeType, v.getType(), opIdx);

  // 2) Multiply
  auto mxfp = cast<TypedValue<RankedTensorType>>(
      arith::MulFOp::create(rewriter, loc, v, reshapeScale).getResult());

  // 3) If the scale is NaN, return NaN, else return the scaled value.
  return maskNan(rewriter, scaledDotOp, mxfp, scale, kDim);
}

TypedValue<RankedTensorType> DecomposeScaledBlocked::extendAndBroadcastScale(
    PatternRewriter &rewriter, DotScaledOp scaledDotOp,
    TypedValue<RankedTensorType> &scale, FloatType computeType,
    RankedTensorType dstType, int opIdx) const {
  auto loc = scale.getLoc();
  auto v = opIdx == 0 ? scaledDotOp.getA() : scaledDotOp.getB();
  auto rank = v.getType().getRank();
  auto kDim = opIdx == 0 ? rank - 1 : rank - 2;

  // For some weird reason, we take the scale with shape as if it were coming
  // from the lhs even when it's the rhs. In a normal world, we should accept
  // this parameter transposed, as we do with the mxfp.
  //
  // Notice: this is an inplace change.
  if (opIdx == 1) {
    auto order = getTransposeOrder(rank);
    scale = TransOp::create(rewriter, loc, scale, order);
  }

  // 1) Cast scale to compute type (fp16/bf16)
  auto scale16 = scaleTo16(rewriter, scale, computeType);

  // 2) Broadcast scale to the same shape as v and convert the layout
  return broadcastScale(rewriter, scaledDotOp, scale16, kDim,
                        dstType.getEncoding());
}

TypedValue<RankedTensorType>
DecomposeScaledBlocked::cvtDotOperand(PatternRewriter &rewriter,
                                      DotScaledOp scaledDotOp, int opIdx,
                                      TypedValue<RankedTensorType> v) const {
  auto *ctx = rewriter.getContext();
  auto retEnc = scaledDotOp.getType().getEncoding();
  auto vType = v.getType();
  auto encoding =
      DotOperandEncodingAttr::get(ctx, opIdx, retEnc, vType.getElementType());
  auto retTy = vType.cloneWithEncoding(encoding);
  return ConvertLayoutOp::create(rewriter, v.getLoc(), retTy, v);
}

void populateDecomposeScaledBlockedPatterns(RewritePatternSet &patterns,
                                            int benefit) {
  patterns.add<DecomposeScaledBlocked>(patterns.getContext(), benefit);
}

} // namespace mlir::triton::gpu
