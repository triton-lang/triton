#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

class UseScaledUpcastFp4 : public OpRewritePattern<arith::MulFOp> {
public:
  UseScaledUpcastFp4(MLIRContext *context, const AMD::TargetInfo &targetInfo,
                     PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), targetInfo(targetInfo) {}

  LogicalResult matchAndRewrite(arith::MulFOp mulOp,
                                PatternRewriter &rewriter) const override {
    if (targetInfo.getISAFamily() != AMD::ISAFamily::CDNA4)
      return failure();

    auto scaleTy = dyn_cast<RankedTensorType>(mulOp.getRhs().getType());
    if (!scaleTy || !scaleTy.getElementType().isBF16())
      return failure();
    auto fp4ToFpOp = mulOp.getLhs().getDefiningOp<Fp4ToFpOp>();
    if (!fp4ToFpOp)
      return failure();

    rewriter.replaceOpWithNewOp<amdgpu::ScaledUpcastFp4Op>(
        mulOp, mulOp.getType(), fp4ToFpOp.getSrc(), mulOp.getRhs(),
        fp4ToFpOp.getAxis());
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

class UseScaledUpcastFp8 : public OpRewritePattern<arith::MulFOp> {
public:
  UseScaledUpcastFp8(MLIRContext *context, const AMD::TargetInfo &targetInfo,
                     PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), targetInfo(targetInfo) {}

  LogicalResult matchAndRewrite(arith::MulFOp mulOp,
                                PatternRewriter &rewriter) const override {
    if (targetInfo.getISAFamily() != AMD::ISAFamily::CDNA4)
      return failure();

    auto scaleTy = dyn_cast<RankedTensorType>(mulOp.getRhs().getType());
    if (!scaleTy || !scaleTy.getElementType().isBF16())
      return failure();
    auto fpToFpOp = mulOp.getLhs().getDefiningOp<FpToFpOp>();
    if (!fpToFpOp)
      return failure();

    rewriter.replaceOpWithNewOp<amdgpu::ScaledUpcastFp8Op>(
        mulOp, mulOp.getType(), fpToFpOp.getSrc(), mulOp.getRhs());
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

} // namespace

namespace mlir::triton::AMD {

void populatePeepholeOptimizationPatterns(RewritePatternSet &patterns,
                                          const AMD::TargetInfo &targetInfo,
                                          PatternBenefit benefit) {
  patterns.add<UseScaledUpcastFp4>(patterns.getContext(), targetInfo, benefit);
}

} // namespace mlir::triton::AMD

