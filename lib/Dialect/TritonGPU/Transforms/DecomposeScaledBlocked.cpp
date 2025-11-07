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

  // TODO: add support for m/n packed formats.
  if (!scaledDotOp.getLhsKPack() || !scaledDotOp.getRhsKPack())
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
    PatternRewriter &rewriter, DotScaledOp scaledDotOp, ModuleOp mod,
    TypedValue<RankedTensorType> scale, int dim) const {
  auto *ctx = rewriter.getContext();
  auto loc = scale.getLoc();
  auto scaleTy = scale.getType();
  auto rank = scaleTy.getRank();
  // 2.1) Expand dims along the last dimension
  {
    // 2.1.1) Find default encoding for ExpandDims
    auto shape = to_vector(scaleTy.getShape());
    shape.insert(shape.end(), 1);
    auto nWarps = lookupNumWarps(scaledDotOp);
    auto threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
    auto numCTAs = TritonGPUDialect::getNumCTAs(mod);
    auto blockedEnc =
        getDefaultBlockedEncoding(ctx, shape, nWarps, threadsPerWarp, numCTAs);
    // 2.1.2) Cast scale16 to SliceEncoding
    auto sliceEnc = SliceEncodingAttr::get(ctx, rank, blockedEnc);
    auto sliceType = scaleTy.cloneWithEncoding(sliceEnc);
    scale = ConvertLayoutOp::create(rewriter, loc, sliceType, scale);
  }
  auto expandScale = ExpandDimsOp::create(rewriter, loc, scale, rank);
  // 2.2) Broadcast the dimension to size 32
  auto scaleShape = to_vector(scaleTy.getShape());
  scaleShape.push_back(32);
  auto broadcastScale = BroadcastOp::create(
      rewriter, loc, expandScale.getType().clone(scaleShape), expandScale);
  // 2.3) Transpose the dimension to the scaled dimension
  auto transposeOrder = llvm::to_vector(llvm::seq<int32_t>(rank));
  transposeOrder.insert(transposeOrder.begin() + dim + 1, rank);
  auto transposedScale =
      TransOp::create(rewriter, loc, broadcastScale, transposeOrder);
  // 2.4) Reshape to the shape of v
  scaleShape.pop_back();
  scaleShape[dim] *= 32;
  auto reshapeScale =
      ReshapeOp::create(rewriter, loc, scaleShape, transposedScale);
  return reshapeScale;
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
  auto mod = scaledDotOp->getParentOfType<ModuleOp>();

  // Scale is NaN
  auto scaleTy = scale.getType();
  auto constFF = arith::ConstantOp::create(
      rewriter, loc, scaleTy,
      DenseElementsAttr::get(scaleTy,
                             APInt(scaleTy.getElementTypeBitWidth(), 0xff)));
  auto scaleIsNan = cast<TypedValue<RankedTensorType>>(
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, scale,
                            constFF)
          .getResult());
  auto cond = broadcastScale(rewriter, scaledDotOp, mod, scaleIsNan, dim);
  // Make scale is NaN compatible with mxfp
  auto condTy = cond.getType();
  condTy = condTy.cloneWithEncoding(mxfp.getType().getEncoding());
  cond = ConvertLayoutOp::create(rewriter, loc, condTy, cond);

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
  auto fastMath = scaledDotOp.getFastMath();

  auto loc = v.getLoc();
  auto rank = v.getType().getRank();
  auto kDim = opIdx == 0 ? rank - 1 : rank - 2;

  // 0) Upcast value to computeType (fp16/bf16)
  if (isFp4) {
    // We always pack along the fastest moving dimension, kDim
    v = Fp4ToFpOp::create(rewriter, loc, v, computeType, kDim);
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
  auto mod = scaledDotOp->getParentOfType<ModuleOp>();
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
  auto reshapeScale = broadcastScale(rewriter, scaledDotOp, mod, scale16, kDim);
  return ConvertLayoutOp::create(rewriter, loc, dstType, reshapeScale);
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
