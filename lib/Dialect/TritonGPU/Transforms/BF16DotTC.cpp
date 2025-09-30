#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUBF16DOTTC
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

template <typename T>
auto convertValue(const Value &value, const FloatType &scalarToType, PatternRewriter &rewriter) -> mlir::Value {
  auto fromType = cast<RankedTensorType>(value.getType());
  auto toType = RankedTensorType::get(fromType.getShape(), scalarToType, fromType.getEncoding());
  return rewriter.create<T>(value.getLoc(), toType, value).getResult();
}

auto SplitF32(Value input, unsigned N, PatternRewriter &rewriter) -> llvm::SmallVector<Value, 3> {
  llvm::SmallVector<Value, 3> split_inputs;
  for (unsigned i = 0; i < N; ++i) {
    Value input_as_bf16 = convertValue<arith::TruncFOp>(input, rewriter.getBF16Type(), rewriter);
    if (i != N - 1) {
      Value input_as_f32 = convertValue<arith::ExtFOp>(input_as_bf16, rewriter.getF32Type(), rewriter);
      input = rewriter.create<arith::SubFOp>(input.getLoc(), input,
                                             input_as_f32);
    }
    split_inputs.push_back(input_as_bf16);
  }
  return split_inputs;
}

// Implement 3xBF16 https://arxiv.org/abs/1904.06376
class BF16x3 : public OpRewritePattern<DotOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    switch (dotOp.getInputPrecision()) {
      case InputPrecision::BF16:
      case InputPrecision::BF16x3:
      case InputPrecision::BF16x6:
      case InputPrecision::BF16x9:
        break;
      default:
        return failure();
    }

    auto isF32 = [](Value operand) {
      return cast<RankedTensorType>(operand.getType()).getElementType().isF32();
    };
    if (!isF32(dotOp.getA()) || !isF32(dotOp.getB())) {
      return failure();
    }

    Value zero = rewriter.create<SplatOp>(
        dotOp->getLoc(), dotOp.getC().getType(),
        rewriter.create<arith::ConstantOp>(dotOp->getLoc(),
                                           rewriter.getF32FloatAttr(0)));
    auto dot = [&](Value a, Value b, Value c) -> Value {
      return rewriter.create<DotOp>(dotOp->getLoc(), c.getType(), a, b, c,
                                    InputPrecision::BF16,
                                    dotOp.getMaxNumImpreciseAcc());
    };
    auto replaceNansWithZeros = [&](Value value) -> Value {
      auto nans = rewriter.create<arith::CmpFOp>(
          dotOp->getLoc(), arith::CmpFPredicate::UNO, value, value);
      return rewriter.create<arith::SelectOp>(dotOp->getLoc(), nans, zero,
                                              value);
    };

    const unsigned hi = 0;
    const unsigned mid = 1;
    const unsigned lo = 2;

    const unsigned N = 3;
    auto lhs_parts = SplitF32(dotOp.getA(), N, rewriter);
    auto rhs_parts = SplitF32(dotOp.getB(), N, rewriter);

    auto result = zero;

    if (dotOp.getInputPrecision() == InputPrecision::BF16x9) {
      result = dot(lhs_parts[lo], rhs_parts[lo], result);
      result = dot(lhs_parts[mid], rhs_parts[lo], result);
      result = dot(lhs_parts[lo], rhs_parts[mid], result);

      result = dot(lhs_parts[mid], rhs_parts[mid], result);

      result = dot(lhs_parts[lo], rhs_parts[hi], result);
      result = dot(lhs_parts[hi], rhs_parts[lo], result);

    } else if (dotOp.getInputPrecision() == InputPrecision::BF16x6) {
      result = dot(lhs_parts[mid], rhs_parts[mid], result);

      result = dot(lhs_parts[lo], rhs_parts[hi], result);
      result = dot(lhs_parts[hi], rhs_parts[lo], result);
    }

    // BF16x3, BF16x6, BF16x9 all need this
    if (dotOp.getInputPrecision() != InputPrecision::BF16) {
      result = dot(lhs_parts[mid], rhs_parts[hi], result);
      result = dot(lhs_parts[hi], rhs_parts[mid], result);
    }

    result = replaceNansWithZeros(result);
    result = dot(lhs_parts[hi], rhs_parts[hi], result);
    result = rewriter.create<arith::AddFOp>(dotOp.getLoc(), result, dotOp.getC());

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
    decomposePatterns.add<BF16x3>(context);
    if (applyPatternsGreedily(m, std::move(decomposePatterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
