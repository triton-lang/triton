#include "triton/Dialect/TritonInstrument/Transforms/FpSanitizer.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"

namespace mlir {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace tti = mlir::triton::instrument;
namespace amdgpu = mlir::triton::amdgpu;

#define GEN_PASS_DEF_TRITONAMDGPUFPSANITIZER
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// ------------------------------------------------------------
// Utility functions
// ------------------------------------------------------------

Value convertScaleElemType(PatternRewriter &rewriter, Location loc, Value scale,
                           FloatType dstElemTy) {
  auto scaleTy = cast<RankedTensorType>(scale.getType());
  auto elemTy = scaleTy.getElementType();

  if (isa<FloatType>(elemTy)) {
    if (elemTy == dstElemTy)
      return scale;
    return tt::FpToFpOp::create(rewriter, loc, scaleTy.clone(dstElemTy), scale);
  }

  auto elemIntTy = dyn_cast<IntegerType>(elemTy);
  if (!elemIntTy || elemIntTy.getWidth() != 8)
    return {};

  FloatType largeFpType = dstElemTy.isF16() ? rewriter.getF32Type() : dstElemTy;
  int intWidth = largeFpType.getIntOrFloatBitWidth();
  auto largeIntTy = rewriter.getIntegerType(intWidth);

  auto ext =
      arith::ExtUIOp::create(rewriter, loc, scaleTy.clone(largeIntTy), scale);
  int shiftValue = largeFpType.getFPMantissaWidth() - 1;
  Value shift = arith::ConstantOp::create(
      rewriter, loc, scaleTy.clone(largeIntTy),
      DenseElementsAttr::get(scaleTy.clone(largeIntTy),
                             rewriter.getIntegerAttr(largeIntTy, shiftValue)));
  Value scaleBits = arith::ShLIOp::create(rewriter, loc, ext, shift);
  if (dstElemTy.isBF16()) {
    Value minScale = arith::ConstantOp::create(
        rewriter, loc, scaleTy.clone(largeIntTy),
        DenseElementsAttr::get(scaleTy.clone(largeIntTy),
                               rewriter.getIntegerAttr(largeIntTy, 0x0040)));
    scaleBits = arith::MaxUIOp::create(rewriter, loc, scaleBits, minScale);
  }
  Value scaleFP = tt::BitcastOp::create(rewriter, loc,
                                        scaleTy.clone(largeFpType), scaleBits);
  if (largeFpType != dstElemTy)
    scaleFP = arith::TruncFOp::create(rewriter, loc, scaleTy.clone(dstElemTy),
                                      scaleFP);
  return scaleFP;
}

// Converts a scale to dstTy, repeating each compact scale over its group of
// elements along axis. ConvertLayoutOp preserves shape, so first insert the
// repeat dimension after axis to keep each group contiguous. For axis=1 and
// groups of 32:
//   [8, 16] -> [8, 16, 1] -> [8, 16, 32] -> [8, 512].
Value expandScale(PatternRewriter &rewriter, Location loc, Value scale,
                  RankedTensorType dstTy, int axis) {
  scale = convertScaleElemType(rewriter, loc, scale,
                               cast<FloatType>(dstTy.getElementType()));
  if (!scale)
    return {};

  auto scaleTy = cast<RankedTensorType>(scale.getType());
  if (scaleTy.getShape() != dstTy.getShape()) {
    auto expandedShape = llvm::to_vector(scaleTy.getShape());
    expandedShape.insert(expandedShape.begin() + axis + 1, 1);
    scale = tt::ReshapeOp::create(rewriter, loc, expandedShape, scale);
    // The op verifiers guarantee an integral number of elements per scale.
    expandedShape[axis + 1] = dstTy.getShape()[axis] / scaleTy.getShape()[axis];
    auto broadcastTy =
        cast<RankedTensorType>(scale.getType()).clone(expandedShape);
    scale = tt::BroadcastOp::create(rewriter, loc, broadcastTy, scale);
    scale = tt::ReshapeOp::create(rewriter, loc, dstTy.getShape(), scale);
  }
  if (scale.getType() != dstTy)
    scale = ttg::ConvertLayoutOp::create(rewriter, loc, dstTy, scale);
  return scale;
}

Value scaleDowncastInput(PatternRewriter &rewriter, Location loc, Value input,
                         Value scale, int axis) {
  auto inputTy = cast<RankedTensorType>(input.getType());
  scale = expandScale(rewriter, loc, scale, inputTy, axis);
  return arith::DivFOp::create(rewriter, loc, input, scale);
}

Value downcastFp4Payload(PatternRewriter &rewriter, Location loc, Value input,
                         bool homomorphicCasts) {
  auto inputTy = cast<RankedTensorType>(input.getType());
  unsigned width = inputTy.getElementType().getIntOrFloatBitWidth();
  auto payloadTy = inputTy.clone(rewriter.getIntegerType(width));
  auto fp4PayloadTy = inputTy.clone(rewriter.getIntegerType(4));
  Value payload =
      tti::ExperimentalFPSanEmbedOp::create(rewriter, loc, payloadTy, input);
  // Fp4ToFpOp unpacks raw nibbles as payloads 0 through 15, so narrow to an
  // unsigned payload to round-trip them.
  Value fp4Payload =
      tti::castFloatPayloadToType(rewriter, loc, payload, fp4PayloadTy,
                                  homomorphicCasts, /*isUnsigned=*/true);
  return arith::ExtUIOp::create(
      rewriter, loc, inputTy.clone(rewriter.getI8Type()), fp4Payload);
}

//----------------------------------------
// Patterns
//----------------------------------------

struct ScaledUpcastFp8OpPattern
    : public OpRewritePattern<amdgpu::ScaledUpcastFp8Op> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(amdgpu::ScaledUpcastFp8Op op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstTy = op.getOutput().getType();
    auto dstElemTy = cast<FloatType>(dstTy.getElementType());

    Value upcasted = tt::FpToFpOp::create(
        rewriter, loc, op.getInput().getType().clone(dstElemTy), op.getInput());

    auto scale = convertScaleElemType(rewriter, loc, op.getScale(), dstElemTy);
    if (!scale)
      return failure();

    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, upcasted, scale);
    return success();
  }
};

struct ScaledUpcastFp4OpPattern
    : public OpRewritePattern<amdgpu::ScaledUpcastFp4Op> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(amdgpu::ScaledUpcastFp4Op op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstTy = op.getOutput().getType();
    auto dstElemTy = cast<FloatType>(dstTy.getElementType());

    Value upcasted = ttg::Fp4ToFpOp::create(rewriter, loc, op.getInput(),
                                            dstElemTy, op.getAxis());

    auto scale = expandScale(rewriter, loc, op.getScale(), dstTy, op.getAxis());
    if (!scale)
      return failure();

    // ScaledUpcastFp4Op does not have SameOperandsAndResultEncoding, so
    // maybe convert_layout to dstTy.
    if (upcasted.getType() != dstTy)
      upcasted = ttg::ConvertLayoutOp::create(rewriter, loc, dstTy, upcasted);

    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, upcasted, scale);
    return success();
  }
};

struct ScaledDowncastFp8OpPattern
    : public OpRewritePattern<amdgpu::ScaledDowncastFp8Op> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(amdgpu::ScaledDowncastFp8Op op,
                                PatternRewriter &rewriter) const override {
    Value scaled = scaleDowncastInput(rewriter, op.getLoc(), op.getInput(),
                                      op.getScale(), op.getAxis());
    // FpSan ignores the rounding mode; the verifier requires one.
    rewriter.replaceOpWithNewOp<tt::FpToFpOp>(
        op, op.getType(), scaled,
        tt::RoundingModeAttr::get(rewriter.getContext(),
                                  tt::RoundingMode::RTNE));
    return success();
  }
};

struct ScaledDowncastFp4OpPattern
    : public OpRewritePattern<amdgpu::ScaledDowncastFp4Op> {
  ScaledDowncastFp4OpPattern(MLIRContext *context, bool homomorphicCasts)
      : OpRewritePattern(context), homomorphicCasts(homomorphicCasts) {}

  LogicalResult matchAndRewrite(amdgpu::ScaledDowncastFp4Op op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto inputTy = op.getInput().getType();
    int axis = op.getAxis();
    Value scaled =
        scaleDowncastInput(rewriter, loc, op.getInput(), op.getScale(), axis);
    Value fp4Payload =
        downcastFp4Payload(rewriter, loc, scaled, homomorphicCasts);

    // Split pairs along axis, moving the pair dimension last for tt.split.
    auto pairShape = llvm::to_vector(inputTy.getShape());
    pairShape[axis] /= 2;
    pairShape.insert(pairShape.begin() + axis + 1, 2);
    Value pairs = tt::ReshapeOp::create(rewriter, loc, pairShape, fp4Payload);
    auto order = llvm::to_vector(llvm::seq<int32_t>(axis + 1));
    llvm::append_range(order, llvm::seq<int32_t>(axis + 2, pairShape.size()));
    order.push_back(axis + 1);
    pairs = tt::TransOp::create(rewriter, loc, pairs, order);
    auto split = tt::SplitOp::create(rewriter, loc, pairs);
    Value evenPayload = split.getOutLHS();
    Value oddPayload = split.getOutRHS();
    auto packedTy = cast<RankedTensorType>(oddPayload.getType());
    Value four = arith::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(packedTy, rewriter.getI8IntegerAttr(4)));
    oddPayload = arith::ShLIOp::create(rewriter, loc, oddPayload, four);
    Value packedBytes =
        arith::OrIOp::create(rewriter, loc, evenPayload, oddPayload);
    if (packedBytes.getType() != op.getType())
      packedBytes = ttg::ConvertLayoutOp::create(rewriter, loc, op.getType(),
                                                 packedBytes);
    rewriter.replaceOp(op, packedBytes);
    return success();
  }

private:
  bool homomorphicCasts;
};

void populateAmdFpSanPatterns(RewritePatternSet &patterns,
                              bool homomorphicCasts) {
  patterns.add<ScaledUpcastFp4OpPattern, ScaledUpcastFp8OpPattern,
               ScaledDowncastFp8OpPattern>(patterns.getContext());
  patterns.add<ScaledDowncastFp4OpPattern>(patterns.getContext(),
                                           homomorphicCasts);
}

struct TritonAMDGPUFpSanitizerPass
    : impl::TritonAMDGPUFpSanitizerBase<TritonAMDGPUFpSanitizerPass> {
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateAmdFpSanPatterns(patterns, homomorphicCasts);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      getOperation()->emitError(
          "FpSanitizer error: Failed to apply AMD patterns");
      signalPassFailure();
      return;
    }

    bool hasUnsupported = false;
    getOperation()->walk([&](Operation *op) {
      if (isa<amdgpu::ScaledUpcastFp8Op, amdgpu::ScaledUpcastFp4Op,
              amdgpu::ScaledDowncastFp8Op, amdgpu::ScaledDowncastFp4Op>(op)) {
        op->emitError("FpSanitizer error: unsupported AMD op remaining: ")
            << op->getName();
        hasUnsupported = true;
      }
    });
    if (hasUnsupported)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir
