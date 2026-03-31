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
  unsigned dstWidth = dstElemTy.getIntOrFloatBitWidth();
  auto intTy = rewriter.getIntegerType(dstWidth);

  if (isa<BFloat16Type>(elemTy)) {
    // Hack to align ScaledUpcast*Op sanitization with that of DotScaledOp.
    // Original i8 scale was zext to i16 then shl by 7. We recover the input.
    auto i16Ty = rewriter.getI16Type();
    auto scaleI16Ty = scaleTy.clone(i16Ty);
    Value scaleI = tt::BitcastOp::create(rewriter, loc, scaleI16Ty, scale);
    Value shift = arith::ConstantOp::create(
        rewriter, loc, scaleI16Ty,
        DenseElementsAttr::get(scaleI16Ty, rewriter.getIntegerAttr(i16Ty, 7)));
    Value shifted = arith::ShRUIOp::create(rewriter, loc, scaleI, shift);
    auto ext =
        arith::ExtUIOp::create(rewriter, loc, scaleTy.clone(intTy), shifted);
    return tt::BitcastOp::create(rewriter, loc, scaleTy.clone(dstElemTy), ext);
  }

  auto elemIntTy = dyn_cast<IntegerType>(elemTy);
  if (!elemIntTy || elemIntTy.getWidth() != 8)
    return {};

  auto ext = arith::ExtUIOp::create(rewriter, loc, scaleTy.clone(intTy), scale);
  return tt::BitcastOp::create(rewriter, loc, scaleTy.clone(dstElemTy), ext);
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
