#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
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
  Value shifted = arith::ShLIOp::create(rewriter, loc, ext, shift);
  Value scaleFP =
      tt::BitcastOp::create(rewriter, loc, scaleTy.clone(largeFpType), shifted);
  if (largeFpType != dstElemTy)
    scaleFP = arith::TruncFOp::create(rewriter, loc, scaleTy.clone(dstElemTy),
                                      scaleFP);
  return scaleFP;
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

    auto scale = convertScaleElemType(rewriter, loc, op.getScale(), dstElemTy);
    if (!scale)
      return failure();

    // ScaledUpcastFp4Op does not have SameOperandsAndResultEncoding, so
    // maybe convert_layout to dstTy.
    if (upcasted.getType() != dstTy)
      upcasted = ttg::ConvertLayoutOp::create(rewriter, loc, dstTy, upcasted);
    if (scale.getType() != dstTy)
      scale = ttg::ConvertLayoutOp::create(rewriter, loc, dstTy, scale);

    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, upcasted, scale);
    return success();
  }
};

void populateAmdFpSanPatterns(RewritePatternSet &patterns) {
  patterns.add<ScaledUpcastFp4OpPattern, ScaledUpcastFp8OpPattern>(
      patterns.getContext());
}

class TritonAMDGPUFpSanitizerPass
    : public impl::TritonAMDGPUFpSanitizerBase<TritonAMDGPUFpSanitizerPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateAmdFpSanPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      getOperation()->emitError(
          "FpSanitizer error: Failed to apply AMD patterns");
      signalPassFailure();
      return;
    }

    bool hasUnsupported = false;
    getOperation()->walk([&](Operation *op) {
      if (isa<amdgpu::ScaledUpcastFp8Op, amdgpu::ScaledUpcastFp4Op>(op)) {
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
