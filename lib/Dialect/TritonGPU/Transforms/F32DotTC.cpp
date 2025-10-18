#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUF32DOTTC
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

bool isF32(Value operand) {
  return cast<RankedTensorType>(operand.getType()).getElementType().isF32();
};

Value zeroLike(Value c, PatternRewriter &rewriter) {
  return rewriter.create<SplatOp>(c.getLoc(), c.getType(),
                                  rewriter.create<arith::ConstantOp>(
                                      c.getLoc(), rewriter.getF32FloatAttr(0)));
};

Value dot(Value lhs, Value rhs, Value acc, PatternRewriter &rewriter,
          InputPrecision precision = InputPrecision::IEEE,
          uint32_t maxNumImpreciseAcc = 0) {
  return rewriter.create<DotOp>(lhs.getLoc(), lhs, rhs, acc, precision,
                                maxNumImpreciseAcc);
};

Value replaceNansWithZeros(Value value, PatternRewriter &rewriter) {
  auto nans = rewriter.create<arith::CmpFOp>(
      value.getLoc(), arith::CmpFPredicate::UNO, value, value);
  auto zero = zeroLike(value, rewriter);
  return rewriter.create<arith::SelectOp>(value.getLoc(), nans, zero, value);
};

unsigned getBF16Count(triton::InputPrecision precision) {
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

    if (!isF32(dotOp.getA()) || !isF32(dotOp.getB()) || !N)
      return failure();

    // Starting Values: a(0), a(1), a(2), b(0), b(1), b(2) and zero accumulator
    const auto lhs_parts = splitF32(dotOp.getA(), N, rewriter);
    const auto rhs_parts = splitF32(dotOp.getB(), N, rewriter);
    auto result = zeroLike(dotOp.getC(), rewriter);

    switch (dotOp.getInputPrecision()) {
    default:
      assert(false && "BF16DotTCPass expects BF16x6 or BF16x3");
      return failure();

      // clang-format off
    // NOTE: 9 dots possible; handled like so if not for lack of speedup:
    // case InputPrecision::BF16x9:
    //   result = dot(lhs_parts[lo], rhs_parts[lo], result, rewriter);
    //   result = dot(lhs_parts[mid], rhs_parts[lo], result, rewriter);
    //   result = dot(lhs_parts[lo], rhs_parts[mid], result, rewriter);
      // clang-format on

    case InputPrecision::BF16x6:
      result = dot(lhs_parts[mid], rhs_parts[mid], result, rewriter);

      result = dot(lhs_parts[lo], rhs_parts[hi], result, rewriter);
      result = dot(lhs_parts[hi], rhs_parts[lo], result, rewriter);

    case InputPrecision::BF16x3:
      result = dot(lhs_parts[mid], rhs_parts[hi], result, rewriter);
      result = dot(lhs_parts[hi], rhs_parts[mid], result, rewriter);
      result = replaceNansWithZeros(result, rewriter);

      // NOTE: For BF16x1 bail without replaceNansWithZeros
      // case InputPrecision::BF16x1: break;
    }

    result = dot(lhs_parts[hi], rhs_parts[hi], result, rewriter);
    result =
        rewriter.create<arith::AddFOp>(dotOp.getLoc(), result, dotOp.getC());

    rewriter.replaceOp(dotOp, result);
    return success();
  }
};

// nb. We call the trick TF32x3 as C++ disallows variables starting with numbers
// Implement 3xTF32 trick https://github.com/NVIDIA/cutlass/discussions/385
// For a, b f32
// dot(a, b, inputPrecision="tf32x3") ->
//  let aBig = f32ToTF32(a), aSmall = a - aBig;
//  let bBig = f32ToTF32(b), bSmall = b - bBig;
//  let small = dot(aSmall, bBig, inputPrecision="tf32") +
//              dot(aBig, bSmall, inputPrecision="tf32")
//  let masked_nans = replaceNansWithZeros(small)
//  let big = dot(aBig, bBig, inputPrecision="tf32")
//  return big + masked_nans;
class TF32x3 : public OpRewritePattern<DotOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    if (!(dotOp.getInputPrecision() == InputPrecision::TF32x3 &&
          isF32(dotOp.getA()) && isF32(dotOp.getB()))) {
      return failure();
    }

    // Aux functions
    auto f32ToTF32 = [&](Value value) -> Value {
      return rewriter
          .create<ElementwiseInlineAsmOp>(dotOp.getLoc(), value.getType(),
                                          "cvt.rna.tf32.f32 $0, $1;", "=r,r",
                                          /*isPure=*/true, /*pack=*/1,
                                          ArrayRef<Value>{value})
          .getResult()[0];
    };
    auto add = [&](Value a, Value b) -> Value {
      return rewriter.create<arith::AddFOp>(dotOp.getLoc(), a, b);
    };
    auto sub = [&](Value a, Value b) -> Value {
      return rewriter.create<arith::SubFOp>(dotOp.getLoc(), a, b);
    };

    auto aBig = f32ToTF32(dotOp.getA());
    auto aSmall = sub(dotOp.getA(), aBig);

    auto bBig = f32ToTF32(dotOp.getB());
    auto bSmall = sub(dotOp.getB(), bBig);

    auto zero = zeroLike(dotOp.getC(), rewriter);

    auto dot1 = dot(aSmall, bBig, zero, rewriter, InputPrecision::TF32,
                    dotOp.getMaxNumImpreciseAcc());
    auto dot2 = dot(aBig, bSmall, dot1, rewriter, InputPrecision::TF32,
                    dotOp.getMaxNumImpreciseAcc());

    // If lhs is 1.0, we will have lhs_high = 1.0 and lhs_low = 0.0.
    // If rhs is +infinity, we will have:
    // +infinity * 1.0 = +infinity
    // +infinity * 0.0 = NaN
    // We would get the wrong result if we sum these partial products. Instead,
    // we must override any accumulated result if the last partial product is
    // non-finite.
    auto dot2withZeroedNans = replaceNansWithZeros(dot2, rewriter);
    auto dot3 = dot(aBig, bBig, dot2withZeroedNans, rewriter,
                    InputPrecision::TF32, dotOp.getMaxNumImpreciseAcc());

    auto sum = add(dot3, dotOp.getC());

    rewriter.replaceOp(dotOp, sum);
    return success();
  }
};

} // anonymous namespace

struct F32DotTCPass : public impl::TritonGPUF32DotTCBase<F32DotTCPass> {
  using impl::TritonGPUF32DotTCBase<F32DotTCPass>::TritonGPUF32DotTCBase;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    RewritePatternSet decomposePatterns(context);
    if (this->emuTF32) {
      decomposePatterns.add<TF32x3>(context);
    }
    decomposePatterns.add<BF16xN>(context);
    if (applyPatternsGreedily(m, std::move(decomposePatterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::triton::gpu
