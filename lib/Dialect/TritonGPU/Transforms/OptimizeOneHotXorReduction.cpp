#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUOPTIMIZEONEHOTXORREDUCTION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

static bool isZero(Value value) {
  if (auto splat = value.getDefiningOp<triton::SplatOp>())
    value = splat.getSrc();
  return matchPattern(value, m_Zero());
}

static std::optional<llvm::APInt> getSplatInteger(Value value) {
  if (auto splat = value.getDefiningOp<triton::SplatOp>())
    value = splat.getSrc();

  llvm::APInt integer;
  if (matchPattern(value, m_ConstantInt(&integer)))
    return integer;

  DenseElementsAttr elements;
  if (matchPattern(value, m_Constant(&elements)) && elements.isSplat()) {
    if (auto integerAttr =
            dyn_cast<IntegerAttr>(elements.getSplatValue<Attribute>()))
      return integerAttr.getValue();
  }
  return std::nullopt;
}

// A 32-element tensor with one value in each warp lane can provide an indexed
// element through a single warp shuffle. The common source-level idiom
//
//   xor_sum(where(arange(0, 32) == bit, values, 0), axis=0)
//
// otherwise lowers as a full warp XOR reduction for every selected bit.
// Replace the independent one-hot reduction with a singleton gather; the
// existing gather lowering translates the logical index into its owning lane.
class OptimizeOneHotXorReductionPattern
    : public OpRewritePattern<triton::ReduceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ReduceOp reduction,
                                PatternRewriter &rewriter) const override {
    Region &combiner = reduction.getCombineOp();
    if (reduction.getAxis() != 0 || reduction.getNumOperands() != 1 ||
        reduction.getNumResults() != 1 || !combiner.hasOneBlock() ||
        combiner.front().getOperations().size() != 2 ||
        !isa_and_nonnull<arith::XOrIOp>(reduction.getSingleCombiner()))
      return failure();

    auto selectedValues =
        reduction.getOperand(0).getDefiningOp<arith::SelectOp>();
    if (!selectedValues || !isZero(selectedValues.getFalseValue()))
      return failure();

    Value values = selectedValues.getTrueValue();
    auto valuesType = dyn_cast<RankedTensorType>(values.getType());
    if (!valuesType || valuesType.getRank() != 1 ||
        valuesType.getShape().front() != 32 ||
        !valuesType.getElementType().isInteger(32) ||
        !isa_and_nonnull<DistributedEncodingTrait>(valuesType.getEncoding()) ||
        !hasOneValuePerLanePerWarp(valuesType))
      return failure();

    auto comparison =
        selectedValues.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!comparison || comparison.getPredicate() != arith::CmpIPredicate::eq)
      return failure();

    auto offsets = comparison.getLhs().getDefiningOp<triton::MakeRangeOp>();
    Value selectedIndex = comparison.getRhs();
    if (!offsets) {
      offsets = comparison.getRhs().getDefiningOp<triton::MakeRangeOp>();
      selectedIndex = comparison.getLhs();
    }
    if (!offsets || offsets.getStart() != 0 || offsets.getEnd() != 32)
      return failure();

    auto index = getSplatInteger(selectedIndex);
    if (!index || index->uge(32))
      return failure();

    auto singletonType = RankedTensorType::get({1}, valuesType.getElementType(),
                                               valuesType.getEncoding());
    Value scalarIndex = arith::ConstantIntOp::create(
        rewriter, reduction.getLoc(), index->getZExtValue(), 32);
    Value gatherIndex = triton::SplatOp::create(rewriter, reduction.getLoc(),
                                                singletonType, scalarIndex);
    auto gathered = triton::GatherOp::create(
        rewriter, reduction.getLoc(), singletonType, values, gatherIndex,
        /*axis=*/0, /*efficient_layout=*/true);
    gathered->setAttr(AttrOneHotXorReduction, rewriter.getUnitAttr());
    Value result = triton::UnsplatOp::create(rewriter, reduction.getLoc(),
                                             valuesType.getElementType(),
                                             gathered.getResult());

    rewriter.replaceOp(reduction, result);
    if (selectedValues->use_empty())
      rewriter.eraseOp(selectedValues);
    return success();
  }

private:
  static bool hasOneValuePerLanePerWarp(RankedTensorType valuesType) {
    LinearLayout layout = toLinearLayout(valuesType);
    auto *context = valuesType.getContext();
    auto kRegister = StringAttr::get(context, "register");
    auto kLane = StringAttr::get(context, "lane");
    auto kWarp = StringAttr::get(context, "warp");
    auto kBlock = StringAttr::get(context, "block");
    auto kDim = StringAttr::get(context, "dim0");

    // Indexed shuffle lowering is valid only when every warp owns the complete
    // tensor. Reject register selection and cross-warp/cross-CTA ownership.
    return layout.hasInDim(kRegister) && layout.hasInDim(kLane) &&
           layout.hasInDim(kWarp) && layout.hasInDim(kBlock) &&
           layout.getInDimSize(kRegister) == 1 &&
           layout.getInDimSize(kLane) == 32 &&
           layout.sublayout({kLane}, {kDim}).isSurjective() &&
           layout.sublayoutIsZero({kWarp, kBlock}, {kDim});
  }

};

class TritonGPUOptimizeOneHotXorReductionPass
    : public impl::TritonGPUOptimizeOneHotXorReductionBase<
          TritonGPUOptimizeOneHotXorReductionPass> {
public:
  void runOnOperation() override {
    auto module = getOperation();
    auto target = module->getAttrOfType<StringAttr>(AttrTargetName);
    if (!target || !target.getValue().starts_with("cuda:"))
      return;

    GreedyRewriteConfig config;
    config.enableFolding(false);
    config.enableConstantCSE(false);
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);

    RewritePatternSet reductionPatterns(&getContext());
    reductionPatterns.add<OptimizeOneHotXorReductionPattern>(&getContext());
    if (failed(applyPatternsGreedily(module, std::move(reductionPatterns),
                                     config)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::triton::gpu
