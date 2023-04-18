#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include <optional>

using namespace mlir;

namespace {

using triton::ReduceOp;
using triton::gpu::BlockedEncodingAttr;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::SliceEncodingAttr;

std::optional<BlockedEncodingAttr>
getBlockedLayoutParentOfSliceLayout(const Attribute &inputLayout) {
  Attribute layout = inputLayout;
  while (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
    Attribute parentLayout = sliceLayout.getParent();
    if (parentLayout == layout)
      return std::nullopt;
    layout = parentLayout;
  }

  return layout.dyn_cast<BlockedEncodingAttr>();
}

class SliceReductionToBlockReductionPattern
    : public mlir::OpRewritePattern<ReduceOp> {

public:
  using OpRewritePattern<ReduceOp>::OpRewritePattern;
  SliceReductionToBlockReductionPattern(MLIRContext *context,
                                        PatternBenefit benefit = 1)
      : OpRewritePattern<ReduceOp>(context, benefit) {}

  LogicalResult matchAndRewrite(ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {

    auto loc = reduceOp.getLoc();
    SmallVector<Value> newArgs;
    SmallVector<Type> newReduceOpResultTypes;

    bool matched = false;

    for (const auto &[operand, result] :
         llvm::zip(reduceOp.getOperands(), reduceOp.getResults())) {
      auto operandType = operand.getType().cast<RankedTensorType>();
      Attribute operandEncoding = operandType.getEncoding();

      auto resultType = result.getType().cast<RankedTensorType>();

      if (auto sliceEncoding = operandEncoding.dyn_cast<SliceEncodingAttr>()) {
        std::optional<BlockedEncodingAttr> blockedEncoding =
            getBlockedLayoutParentOfSliceLayout(sliceEncoding);

        if (!blockedEncoding.has_value())
          return failure();

        RankedTensorType newOperandType = RankedTensorType::get(
            operandType.getShape(), operandType.getElementType(),
            blockedEncoding.value());

        Value newOperand =
            rewriter.create<ConvertLayoutOp>(loc, newOperandType, operand);
        newArgs.push_back(newOperand);

        Attribute newResultEncoding = SliceEncodingAttr::get(
            getContext(), reduceOp.getAxis(), blockedEncoding.value());
        Type newResultType = RankedTensorType::get(resultType.getShape(),
                                                   resultType.getElementType(),
                                                   newResultEncoding);
        newReduceOpResultTypes.push_back(newResultType);

        matched = true;
      } else {
        newArgs.push_back(operand);
        newReduceOpResultTypes.push_back(resultType);
      }
    }

    if (!matched)
      return failure();

    auto newReduceOp = rewriter.create<ReduceOp>(loc, newReduceOpResultTypes,
                                                 newArgs, reduceOp.getAxis());
    newReduceOp.getRegion().takeBody(reduceOp.getRegion());

    SmallVector<Value> newResults;

    const auto getEncoding = [](Value v) {
      return v.getType().cast<RankedTensorType>().getEncoding();
    };

    for (const auto &[oldResult, newResult] :
         llvm::zip(reduceOp.getResults(), newReduceOp.getResults())) {
      Attribute oldEncoding = getEncoding(oldResult);
      Attribute newEncoding = getEncoding(newResult);

      if (oldEncoding == newEncoding)
        newResults.push_back(newResult);
      else {
        auto convertedResult = rewriter.create<ConvertLayoutOp>(
            loc, oldResult.getType().cast<RankedTensorType>(), newResult);
        newResults.push_back(convertedResult);
      }
    }

    rewriter.replaceOp(reduceOp, newResults);

    return success();
  }
};

} // namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUConvertSliceReductionToBlockReductionPass
    : public TritonGPUConvertSliceReductionToBlockReductionBase<
          TritonGPUConvertSliceReductionToBlockReductionPass> {
public:
  TritonGPUConvertSliceReductionToBlockReductionPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<::SliceReductionToBlockReductionPattern>(context);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass>
mlir::createTritonGPUConvertSliceReductionToBlockReductionPass() {
  return std::make_unique<TritonGPUConvertSliceReductionToBlockReductionPass>();
}
