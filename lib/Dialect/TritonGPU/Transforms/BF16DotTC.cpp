#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUBF16DOTTC
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

template <typename T>
auto convertValue(Value value, const FloatType &scalarToType,
                  PatternRewriter &rewriter) -> mlir::Value {
  auto fromType = cast<RankedTensorType>(value.getType());
  auto toType = fromType.cloneWith(std::nullopt, scalarToType);
  return rewriter.create<T>(value.getLoc(), toType, value).getResult();
}

auto splitF32(Value input, unsigned N, PatternRewriter &rewriter)
    -> llvm::SmallVector<Value, 3> {
  llvm::SmallVector<Value, 3> splitInputs;
  for (unsigned i = 0; i < N; ++i) {
    Value inputAsBF16 =
        convertValue<arith::TruncFOp>(input, rewriter.getBF16Type(), rewriter);
    if (i != N - 1) {
      Value inputAsF32 = convertValue<arith::ExtFOp>(
          inputAsBF16, rewriter.getF32Type(), rewriter);
      input = rewriter.create<arith::SubFOp>(input.getLoc(), input, inputAsF32);
    }
    splitInputs.push_back(inputAsBF16);
  }
  return splitInputs;
}

Value IEEEDot(PatternRewriter &rewriter, Value lhs, Value rhs, Value acc) {
  return rewriter.create<DotOp>(lhs.getLoc(), lhs, rhs, acc,
                                /*inputPrecision=*/InputPrecision::IEEE,
                                /*maxNumImpreciseAcc=*/0);
}

auto getBF16Count(triton::InputPrecision precision) -> unsigned {
  switch (precision) {
  default:
    return 0;
  case InputPrecision::BF16x3:
    // BF16x3 only needs the first 2 values derived from splitting an F32
    return 2;
  case InputPrecision::BF16x6:
    return 3;
  }
}

// Implements 3xBF16 https://arxiv.org/abs/1904.06376
// See also
// https://github.com/openxla/xla/blob/e33f93fb7220d408811afdc926cf10baaf49c64e/xla/backends/gpu/codegen/triton/dot_algorithms.cc#L152
// As well as
// https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblaslt/tensilelite/Tensile/Components/LocalRead.py#L288-L330
struct BF16xN : public OpRewritePattern<DotOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    // BF16 indices and count
    const unsigned hi = 0;
    const unsigned mid = 1;
    const unsigned lo = 2;
    const unsigned N = getBF16Count(dotOp.getInputPrecision());
    Location loc = dotOp.getLoc();
    auto typeA = dotOp.getA().getType();
    auto typeB = dotOp.getB().getType();

    if (!cast<RankedTensorType>(typeA).getElementType().isF32() ||
        !cast<RankedTensorType>(typeB).getElementType().isF32() || !N)
      return failure();

    // Aux functions
    auto zeroLike = [&](Value c) -> Value {
      return rewriter.create<SplatOp>(
          dotOp->getLoc(), c.getType(),
          rewriter.create<arith::ConstantOp>(dotOp->getLoc(),
                                             rewriter.getF32FloatAttr(0)));
    };
    auto replaceNansWithZeros = [&](Value value) -> Value {
      auto nans = rewriter.create<arith::CmpFOp>(
          dotOp->getLoc(), arith::CmpFPredicate::UNO, value, value);
      auto zero = zeroLike(value);
      return rewriter.create<arith::SelectOp>(dotOp->getLoc(), nans, zero,
                                              value);
    };

    // Starting Values: a(0), a(1), a(2), b(0), b(1), b(2) and zero accumulator
    const auto lhs_parts = splitF32(dotOp.getA(), N, rewriter);
    const auto rhs_parts = splitF32(dotOp.getB(), N, rewriter);
    auto result = zeroLike(dotOp.getC());

    switch (dotOp.getInputPrecision()) {
    default:
      assert(false && "BF16DotTCPass expects BF16x6 or BF16x3");
      return failure();

      // NOTE: 9 dots possible; handled like so if not for lack of speedup:
      // case InputPrecision::BF16x9:
      //   result = IEEEDot(rewriter, lhs_parts[lo], rhs_parts[lo], result);
      //   result = IEEEDot(rewriter, lhs_parts[mid], rhs_parts[lo], result);
      //   result = IEEEDot(rewriter, lhs_parts[lo], rhs_parts[mid], result);

    case InputPrecision::BF16x6:
      result = IEEEDot(rewriter, lhs_parts[mid], rhs_parts[mid], result);

      result = IEEEDot(rewriter, lhs_parts[lo], rhs_parts[hi], result);
      result = IEEEDot(rewriter, lhs_parts[hi], rhs_parts[lo], result);

    case InputPrecision::BF16x3:
      result = IEEEDot(rewriter, lhs_parts[mid], rhs_parts[hi], result);
      result = IEEEDot(rewriter, lhs_parts[hi], rhs_parts[mid], result);
      result = replaceNansWithZeros(result);

      // NOTE: For BF16x1 bail without replaceNansWithZeros
      // case InputPrecision::BF16x1: break;
    }

    result = IEEEDot(rewriter, lhs_parts[hi], rhs_parts[hi], result);
    result = rewriter.create<arith::AddFOp>(loc, result, dotOp.getC());

    rewriter.replaceOp(dotOp, result);
    return success();
  }
};

} // anonymous namespace

struct BF16DotTCPass : public impl::TritonGPUBF16DotTCBase<BF16DotTCPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    RewritePatternSet decomposePatterns(context);
    decomposePatterns.add<BF16xN>(context);
    if (applyPatternsGreedily(m, std::move(decomposePatterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::triton::gpu
