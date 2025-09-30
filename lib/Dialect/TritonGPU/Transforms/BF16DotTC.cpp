#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUBF16DOTTC
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

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

    // Aux functions
    auto f32ToBF16 = [&](Value value) -> Value {
      auto fp32Type = cast<RankedTensorType>(value.getType());
      auto bf16Type =
          RankedTensorType::get(fp32Type.getShape(), rewriter.getBF16Type(), fp32Type.getEncoding());
      return rewriter.create<arith::TruncFOp>(dotOp.getLoc(), bf16Type, value)
          .getResult();
    };
    auto bf16ToF32 = [&](Value value) -> Value {
      auto bf16Type = cast<RankedTensorType>(value.getType());
      auto fp32Type =
          RankedTensorType::get(bf16Type.getShape(), rewriter.getF32Type(), bf16Type.getEncoding());
      return rewriter.create<arith::ExtFOp>(dotOp.getLoc(), fp32Type, value)
          .getResult();
    };
    auto zeroLike = [&](Value c) -> Value {
      return rewriter.create<SplatOp>(
          dotOp->getLoc(), c.getType(),
          rewriter.create<arith::ConstantOp>(dotOp->getLoc(),
                                             rewriter.getF32FloatAttr(0)));
    };
    auto dot = [&](Value a, Value b, Value c) -> Value {
      return rewriter.create<DotOp>(dotOp->getLoc(), c.getType(), a, b, c,
                                    InputPrecision::BF16,
                                    dotOp.getMaxNumImpreciseAcc());
    };
    auto replaceNansWithZeros = [&](Value value) -> Value {
      auto nans = rewriter.create<arith::CmpFOp>(
          dotOp->getLoc(), arith::CmpFPredicate::UNO, value, value);
      auto zero = zeroLike(value);
      return rewriter.create<arith::SelectOp>(dotOp->getLoc(), nans, zero,
                                              value);
    };

    auto SplitF32 = [&](Value input, unsigned N) -> std::vector<Value> {
      std::vector<Value> split_inputs;
      split_inputs.reserve(N);
      for (int i = 0; i < N; ++i) {
        Value input_as_bf16 = f32ToBF16(input);
        if (i != N - 1) {
          Value input_as_f32 = bf16ToF32(input_as_bf16);
          input = rewriter.create<arith::SubFOp>(dotOp->getLoc(), input,
                                                 input_as_f32);
        }
        split_inputs.push_back(input_as_bf16);
      }
      return split_inputs;
    };

    const int hi = 0;
    const int med = 1;
    const int lo = 2;

    const unsigned N = 3;
    auto lhs_parts = SplitF32(dotOp.getA(), N);
    auto rhs_parts = SplitF32(dotOp.getB(), N);

    auto result = zeroLike(dotOp.getC());

    if (dotOp.getInputPrecision() == InputPrecision::BF16x9) {
      result = dot(lhs_parts[lo], rhs_parts[lo], result);
      result = dot(lhs_parts[med], rhs_parts[lo], result);
      result = dot(lhs_parts[lo], rhs_parts[med], result);

      result = dot(lhs_parts[med], rhs_parts[med], result);

      result = dot(lhs_parts[lo], rhs_parts[hi], result);
      result = dot(lhs_parts[hi], rhs_parts[lo], result);

    } else if (dotOp.getInputPrecision() == InputPrecision::BF16x6) {
      result = dot(lhs_parts[med], rhs_parts[med], result);

      result = dot(lhs_parts[lo], rhs_parts[hi], result);
      result = dot(lhs_parts[hi], rhs_parts[lo], result);
    }

    // BF16x3, BF16x6, BF16x9 all need this
    result = dot(lhs_parts[med], rhs_parts[hi], result);
    result = dot(lhs_parts[hi], rhs_parts[med], result);

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
