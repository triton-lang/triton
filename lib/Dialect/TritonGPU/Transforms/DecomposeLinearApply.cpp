#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUDECOMPOSELINEARAPPLY
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

class DecomposeLinearApplyPattern
    : public OpRewritePattern<triton::LinearApplyOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LinearApplyOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(validateBasisLayout(op)))
      return failure();

    Location loc = op.getLoc();
    RankedTensorType indexType = op.getIndex().getType();
    Value bases = op.getBases();
    RankedTensorType basisType = op.getBases().getType();

    Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
    Value zeroBases = triton::SplatOp::create(rewriter, loc, basisType, zero);
    Value zeroIndex = triton::SplatOp::create(rewriter, loc, indexType, zero);
    Value oneIndex = triton::SplatOp::create(rewriter, loc, indexType, one);
    Value basisOffsets =
        triton::MakeRangeOp::create(rewriter, loc, basisType, 0, 32);
    Value result = zeroIndex;

    for (unsigned bit = 0; bit < 32; ++bit) {
      Value bitValue = arith::ConstantIntOp::create(rewriter, loc, bit, 32);
      Value basisBit =
          triton::SplatOp::create(rewriter, loc, basisType, bitValue);
      Value isBasisBit = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, basisOffsets, basisBit);
      Value selectedBases =
          arith::SelectOp::create(rewriter, loc, isBasisBit, bases, zeroBases);
      Value selectedBasis = createXorReduction(loc, selectedBases, rewriter);

      Value indexBit =
          triton::SplatOp::create(rewriter, loc, indexType, bitValue);
      Value shiftedIndex =
          arith::ShRUIOp::create(rewriter, loc, op.getIndex(), indexBit);
      Value maskedIndex =
          arith::AndIOp::create(rewriter, loc, shiftedIndex, oneIndex);
      Value bitIsSet = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::ne, maskedIndex, zeroIndex);
      Value basisForIndex =
          triton::SplatOp::create(rewriter, loc, indexType, selectedBasis);
      Value contribution = arith::SelectOp::create(rewriter, loc, bitIsSet,
                                                   basisForIndex, zeroIndex);
      result = arith::XOrIOp::create(rewriter, loc, result, contribution);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static LogicalResult validateBasisLayout(triton::LinearApplyOp op) {
    RankedTensorType basisType = op.getBases().getType();
    LinearLayout layout = toLinearLayout(basisType);
    MLIRContext *context = op.getContext();
    auto kRegister = StringAttr::get(context, "register");
    auto kLane = StringAttr::get(context, "lane");
    auto kWarp = StringAttr::get(context, "warp");
    auto kDim = StringAttr::get(context, "dim0");

    if (layout.sublayout({kRegister, kLane, kWarp}, {kDim}).isSurjective())
      return success();

    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto target = module->getAttrOfType<StringAttr>(AttrTargetName);
    if (target && target.getValue().starts_with("cuda:"))
      return success();

    if (target && (target.getValue().starts_with("hip:") ||
                   target.getValue().starts_with("gfx")))
      return op.emitOpError("linear_apply with a cross-CTA basis layout is "
                            "unsupported on AMD; use a CTA-local basis layout");

    return op.emitOpError("linear_apply with a cross-CTA basis layout "
                          "requires an explicit CUDA target; use a CTA-local "
                          "basis layout");
  }

  static Value createXorReduction(Location loc, Value input,
                                  PatternRewriter &rewriter) {
    auto reduce = triton::ReduceOp::create(rewriter, loc, ValueRange{input}, 0);
    Block &block = reduce.getCombineOp().emplaceBlock();
    Type elementType = getElementTypeOrSelf(input.getType());
    block.addArguments({elementType, elementType}, {loc, loc});

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);
      Value combined = arith::XOrIOp::create(
          rewriter, loc, block.getArgument(0), block.getArgument(1));
      triton::ReduceReturnOp::create(rewriter, loc, ValueRange{combined});
    }

    return reduce.getResult().front();
  }
};

class TritonGPUDecomposeLinearApplyPass
    : public impl::TritonGPUDecomposeLinearApplyBase<
          TritonGPUDecomposeLinearApplyPass> {
public:
  void runOnOperation() override {
    SmallVector<triton::LinearApplyOp> ops;
    getOperation().walk([&](triton::LinearApplyOp op) { ops.push_back(op); });

    PatternRewriter rewriter(&getContext());
    DecomposeLinearApplyPattern pattern(&getContext());
    for (triton::LinearApplyOp op : ops) {
      rewriter.setInsertionPoint(op);
      if (failed(pattern.matchAndRewrite(op, rewriter))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace mlir::triton::gpu
