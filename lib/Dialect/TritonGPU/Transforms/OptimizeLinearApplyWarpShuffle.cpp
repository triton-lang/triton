#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUOPTIMIZELINEARAPPLYWARPSHUFFLE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

// A canonical 32-element basis holds exactly one value in each warp lane and
// repeats the same logical tensor in every warp. DecomposeLinearApply extracts
// each basis value with an expensive one-hot XOR reduction:
//
//   reduce_xor(select(arange(32) == bit, bases, 0))
//
// Gathering one element from the existing basis instead lowers to one indexed
// warp shuffle, even when the basis was computed directly in registers:
//
//   unsplat(gather(bases, splat(bit, tensor<1xi32>), axis=0))
//
// Fuse the subsequent conditional XOR with NVIDIA's ternary logic operation:
//
//   accumulator ^ select(condition, basis, 0)
//     => lop3(accumulator, basis, sign_extend(condition), 0x78)
class OptimizeLinearApplyWarpShufflePattern
    : public OpRewritePattern<triton::ReduceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ReduceOp reduction,
                                PatternRewriter &rewriter) const override {
    // Match only the scalar XOR reductions emitted to select one basis value.
    if (reduction.getAxis() != 0 || reduction.getNumOperands() != 1 ||
        reduction.getNumResults() != 1 || !reduction->hasOneUse() ||
        !isa_and_nonnull<arith::XOrIOp>(reduction.getSingleCombiner()))
      return failure();

    auto selectedBasis =
        reduction.getOperand(0).getDefiningOp<arith::SelectOp>();
    if (!selectedBasis || !selectedBasis->hasOneUse() ||
        !isZero(selectedBasis.getFalseValue()))
      return failure();

    // The basis may be loaded, passed as an argument, or computed in lane
    // registers. Its origin is irrelevant as long as each warp owns all values
    // and every lane contributes exactly one value.
    Value bases = selectedBasis.getTrueValue();
    auto basisType = dyn_cast<RankedTensorType>(bases.getType());
    if (!basisType || basisType.getRank() != 1 ||
        basisType.getShape().front() != 32 ||
        !basisType.getElementType().isInteger(32) ||
        !isa_and_nonnull<DistributedEncodingTrait>(basisType.getEncoding()) ||
        !hasOneBasisPerLanePerWarp(basisType))
      return failure();

    // The one-hot comparison supplies the logical basis index to gather. Gather
    // lowering translates that index through the layout into its owning lane,
    // so non-identity but invertible lane permutations remain valid.
    auto comparison =
        selectedBasis.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!comparison || comparison.getPredicate() != arith::CmpIPredicate::eq)
      return failure();
    auto selectionOffsets =
        comparison.getLhs().getDefiningOp<triton::MakeRangeOp>();
    auto selectedBit = comparison.getRhs().getDefiningOp<triton::SplatOp>();
    if (!selectionOffsets || !selectedBit || selectionOffsets.getStart() != 0 ||
        selectionOffsets.getEnd() != 32)
      return failure();
    llvm::APInt bit;
    if (!matchPattern(selectedBit.getSrc(), m_ConstantInt(&bit)) || bit.uge(32))
      return failure();

    // Keep this pattern specific to linear_apply by also recognizing its
    // accumulator ^ select(condition, splat(reduced_basis), 0) consumer.
    auto broadcastBasis = dyn_cast<triton::SplatOp>(
        *reduction.getResult().front().getUsers().begin());
    if (!broadcastBasis || !broadcastBasis->hasOneUse())
      return failure();
    auto contribution = dyn_cast<arith::SelectOp>(
        *broadcastBasis.getResult().getUsers().begin());
    if (!contribution ||
        contribution.getTrueValue() != broadcastBasis.getResult() ||
        !isZero(contribution.getFalseValue()) || !contribution->hasOneUse())
      return failure();

    auto accumulation =
        dyn_cast<arith::XOrIOp>(*contribution.getResult().getUsers().begin());
    auto accumulationType =
        accumulation ? dyn_cast<RankedTensorType>(accumulation.getType())
                     : RankedTensorType{};
    auto conditionType =
        dyn_cast<RankedTensorType>(contribution.getCondition().getType());
    if (!accumulation || !accumulationType ||
        !accumulationType.getElementType().isInteger(32) || !conditionType ||
        !conditionType.getElementType().isInteger(1) ||
        conditionType.getShape() != accumulationType.getShape() ||
        conditionType.getEncoding() != accumulationType.getEncoding())
      return failure();

    // A singleton gather keeps the source tensor in its existing registers.
    // Since the entire basis is lane-distributed and warp-replicated, gather
    // lowering emits exactly one shfl.sync.idx rather than shared-memory
    // communication or a fresh global load.
    rewriter.setInsertionPoint(reduction);
    auto singletonType = RankedTensorType::get({1}, basisType.getElementType(),
                                               basisType.getEncoding());
    Value gatherIndex = triton::SplatOp::create(
        rewriter, reduction.getLoc(), singletonType, selectedBit.getSrc());
    auto gathered = triton::GatherOp::create(
        rewriter, reduction.getLoc(), singletonType, bases, gatherIndex,
        /*axis=*/0, /*efficient_layout=*/true);
    Value scalarBasis = triton::UnsplatOp::create(rewriter, reduction.getLoc(),
                                                  basisType.getElementType(),
                                                  gathered.getResult());

    rewriter.replaceOp(reduction, scalarBasis);
    rewriter.eraseOp(selectedBasis);

    // Sign extension maps an i1 predicate to either zero or all ones; the 0x78
    // truth table computes accumulator ^ (broadcastBasis & signMask).
    rewriter.setInsertionPoint(accumulation);
    Value accumulator = accumulation.getLhs() == contribution.getResult()
                            ? accumulation.getRhs()
                            : accumulation.getLhs();
    Value signMask =
        arith::ExtSIOp::create(rewriter, accumulation.getLoc(),
                               accumulationType, contribution.getCondition());
    auto fused = triton::ElementwiseInlineAsmOp::create(
        rewriter, accumulation.getLoc(), TypeRange{accumulationType},
        "lop3.b32 $0, $1, $2, $3, 0x78;", "=r,r,r,r", /*pure=*/true,
        /*packed_element=*/1,
        ValueRange{accumulator, broadcastBasis.getResult(), signMask});
    rewriter.replaceOp(accumulation, fused.getResult());
    rewriter.eraseOp(contribution);
    return success();
  }

private:
  static bool hasOneBasisPerLanePerWarp(RankedTensorType basisType) {
    LinearLayout layout = toLinearLayout(basisType);
    auto *context = basisType.getContext();
    auto kRegister = StringAttr::get(context, "register");
    auto kLane = StringAttr::get(context, "lane");
    auto kWarp = StringAttr::get(context, "warp");
    auto kBlock = StringAttr::get(context, "block");
    auto kDim = StringAttr::get(context, "dim0");

    // Full lane coverage plus one register per lane guarantees one indexed
    // shuffle per gather. Zero warp/block contributions prove every warp owns
    // the same complete basis without cross-warp or cross-CTA communication.
    return layout.hasInDim(kRegister) && layout.hasInDim(kLane) &&
           layout.hasInDim(kWarp) && layout.hasInDim(kBlock) &&
           layout.getInDimSize(kRegister) == 1 &&
           layout.getInDimSize(kLane) == 32 &&
           layout.sublayout({kLane}, {kDim}).isSurjective() &&
           layout.sublayoutIsZero({kWarp, kBlock}, {kDim});
  }

  static bool isZero(Value value) {
    if (auto splat = value.getDefiningOp<triton::SplatOp>())
      value = splat.getSrc();
    llvm::APInt integer;
    return matchPattern(value, m_ConstantInt(&integer)) && integer.isZero();
  }
};

class TritonGPUOptimizeLinearApplyWarpShufflePass
    : public impl::TritonGPUOptimizeLinearApplyWarpShuffleBase<
          TritonGPUOptimizeLinearApplyWarpShufflePass> {
public:
  void runOnOperation() override {
    // The fused ternary operation uses NVIDIA-specific inline PTX.
    auto module = getOperation();
    auto target = module->getAttrOfType<StringAttr>(AttrTargetName);
    if (!target || !target.getValue().starts_with("cuda:"))
      return;

    RewritePatternSet patterns(&getContext());
    patterns.add<OptimizeLinearApplyWarpShufflePattern>(&getContext());

    // Keep each independent reduction's consumer chain intact until its
    // pattern runs; folding the initial XOR with zero would hide that chain.
    GreedyRewriteConfig config;
    config.enableFolding(false);
    config.enableConstantCSE(false);
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(applyPatternsGreedily(module, std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::triton::gpu
