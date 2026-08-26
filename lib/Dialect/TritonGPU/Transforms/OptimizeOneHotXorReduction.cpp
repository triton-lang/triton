#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"

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
  OptimizeOneHotXorReductionPattern(
      MLIRContext *context,
      llvm::SmallPtrSetImpl<Operation *> &optimizedGathers)
      : OpRewritePattern(context), optimizedGathers(optimizedGathers) {}

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
    optimizedGathers.insert(gathered.getOperation());
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

  llvm::SmallPtrSetImpl<Operation *> &optimizedGathers;
};

// After a one-hot reduction becomes a gather, the surrounding source-level
// accumulator update can be implemented with one NVIDIA ternary instruction:
//
//   accumulator ^ where(condition, splat(gathered_value), 0)
//     => lop3(accumulator, splat(freeze(gathered_value)), sext(condition),
//     0x78)
class FuseOneHotXorAccumulationPattern
    : public OpRewritePattern<arith::XOrIOp> {
public:
  FuseOneHotXorAccumulationPattern(
      MLIRContext *context,
      const llvm::SmallPtrSetImpl<Operation *> &optimizedGathers)
      : OpRewritePattern(context), optimizedGathers(optimizedGathers) {}

  LogicalResult matchAndRewrite(arith::XOrIOp accumulation,
                                PatternRewriter &rewriter) const override {
    auto contribution = accumulation.getRhs().getDefiningOp<arith::SelectOp>();
    Value accumulator = accumulation.getLhs();
    if (!contribution) {
      contribution = accumulation.getLhs().getDefiningOp<arith::SelectOp>();
      accumulator = accumulation.getRhs();
    }
    if (!contribution || !contribution->hasOneUse() ||
        !isZero(contribution.getFalseValue()))
      return failure();

    auto broadcast =
        contribution.getTrueValue().getDefiningOp<triton::SplatOp>();
    auto unsplat = broadcast
                       ? broadcast.getSrc().getDefiningOp<triton::UnsplatOp>()
                       : triton::UnsplatOp{};
    auto gathered = unsplat ? unsplat.getSrc().getDefiningOp<triton::GatherOp>()
                            : triton::GatherOp{};
    if (!gathered || gathered.getAxis() != 0 ||
        !optimizedGathers.contains(gathered.getOperation()))
      return failure();

    auto accumulationType = dyn_cast<RankedTensorType>(accumulation.getType());
    auto conditionType =
        dyn_cast<RankedTensorType>(contribution.getCondition().getType());
    if (!accumulationType || !accumulationType.getElementType().isInteger(32) ||
        !conditionType || !conditionType.getElementType().isInteger(1) ||
        conditionType.getShape() != accumulationType.getShape() ||
        conditionType.getEncoding() != accumulationType.getEncoding())
      return failure();

    // Select masks a poison basis when the condition is false, whereas inline
    // assembly eagerly consumes every operand. Freeze only the scalar used by
    // this fusion: defined values are unchanged, and choosing a value for
    // poison preserves the defined zero contribution of the original false
    // branch. The scalar is already an LLVM-compatible i32 at this late TTGIR
    // stage.
    Value frozenBasis = LLVM::FreezeOp::create(rewriter, accumulation.getLoc(),
                                               unsplat.getResult());
    Value frozenBroadcast = triton::SplatOp::create(
        rewriter, accumulation.getLoc(), accumulationType, frozenBasis);
    Value signMask =
        arith::ExtSIOp::create(rewriter, accumulation.getLoc(),
                               accumulationType, contribution.getCondition());
    auto fused = triton::ElementwiseInlineAsmOp::create(
        rewriter, accumulation.getLoc(), TypeRange{accumulationType},
        "lop3.b32 $0, $1, $2, $3, 0x78;", "=r,r,r,r", /*pure=*/true,
        /*packed_element=*/1,
        ValueRange{accumulator, frozenBroadcast, signMask});
    rewriter.replaceOp(accumulation, fused.getResult());
    if (contribution->use_empty())
      rewriter.eraseOp(contribution);
    return success();
  }

private:
  const llvm::SmallPtrSetImpl<Operation *> &optimizedGathers;
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

    llvm::SmallPtrSet<Operation *, 32> optimizedGathers;
    RewritePatternSet reductionPatterns(&getContext());
    reductionPatterns.add<OptimizeOneHotXorReductionPattern>(&getContext(),
                                                             optimizedGathers);
    if (failed(applyPatternsGreedily(module, std::move(reductionPatterns),
                                     config)))
      return signalPassFailure();
    if (optimizedGathers.empty())
      return;

    // Only fuse accumulations that consume gathers created in this invocation.
    RewritePatternSet accumulationPatterns(&getContext());
    accumulationPatterns.add<FuseOneHotXorAccumulationPattern>(
        &getContext(), optimizedGathers);
    if (failed(applyPatternsGreedily(module, std::move(accumulationPatterns),
                                     config)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::triton::gpu
