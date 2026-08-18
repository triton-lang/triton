#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/DiscardableAttributes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONCOMBINEOPS
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

bool isZero(Value val) {
  return (matchPattern(val, m_Zero()) || matchPattern(val, m_AnyZeroFloat()));
}

bool isAddPtrOffsetCombinable(Value first, Value second) {
  auto GetConstantIntValue = [](Value val) -> std::optional<llvm::APInt> {
    DenseElementsAttr constAttr;
    auto defOp = val.getDefiningOp();
    if (defOp) {
      if (auto splatOp = llvm::dyn_cast<SplatOp>(defOp))
        val = splatOp.getSrc();
      else if (matchPattern(defOp, m_Constant(&constAttr)) &&
               constAttr.isSplat()) {
        auto attr = constAttr.getSplatValue<Attribute>();
        // Check IntegerAttr
        if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
          return intAttr.getValue();
      }
    }

    // Check constant value.
    llvm::APInt intVal;
    if (matchPattern(val, m_ConstantInt(&intVal)))
      return intVal;

    return std::nullopt;
  };

  if (first.getType() == second.getType()) {
    // Whether bitwidth of element type is equal to pointer
    if (getElementTypeOrSelf(first.getType()).getIntOrFloatBitWidth() == 64)
      return true;

    // first + second does not overflow
    auto firstVal = GetConstantIntValue(first);
    auto secondVal = GetConstantIntValue(second);
    if (firstVal && secondVal) {
      bool overflow = false;
      auto resVal = firstVal->sadd_ov(*secondVal, overflow);
      return !overflow;
    }
  }
  return false;
}

// TODO(csigg): remove after next LLVM integrate.
using FastMathFlags = arith::FastMathFlags;

#include "TritonCombine.inc"

// select(cond, load(ptrs, splat(cond), ???), other)
//   => load(ptrs, splat(cond), other)
class CombineSelectMaskedLoadPattern : public RewritePattern {
public:
  CombineSelectMaskedLoadPattern(MLIRContext *context)
      : RewritePattern(arith::SelectOp::getOperationName(), 3, context,
                       {LoadOp::getOperationName()}) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto selectOp = llvm::dyn_cast<arith::SelectOp>(op);
    if (!selectOp)
      return failure();

    Value trueValue = selectOp.getTrueValue();
    Value falseValue = selectOp.getFalseValue();
    Value condSelect = selectOp.getCondition();

    auto loadOp = trueValue.getDefiningOp<LoadOp>();
    if (!loadOp)
      return failure();

    Value mask = loadOp.getMask();
    if (!mask)
      return failure();

    auto splatOp = mask.getDefiningOp<SplatOp>();
    if (!splatOp)
      return failure();

    auto splatCond = splatOp.getSrc();
    if (splatCond != condSelect)
      return failure();

    rewriter.replaceOpWithNewOp<LoadOp>(
        op, loadOp.getPtr(), loadOp.getMask(), /*other=*/falseValue,
        loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
    return success();
  }
};

// sum(x[:, :, None] * y[None, :, :], 1)
// -> dot(x, y)
class CombineBroadcastMulReducePattern : public RewritePattern {
private:
  static bool isAddF32(const Operation *op) {
    if (auto addf = dyn_cast_or_null<arith::AddFOp>(op))
      return addf.getType().getIntOrFloatBitWidth() <= 32;
    return false;
  }

  /// Return true if \p op broadcasts only along \p axis, false otherwise.
  static bool isBroadcastAlongAxis(BroadcastOp op, unsigned axis) {
    auto srcShape = op.getSrc().getType().getShape();
    auto dstShape = op.getType().getShape();
    for (unsigned i = 0; i < srcShape.size(); ++i) {
      if ((srcShape[i] != dstShape[i]) != (i == axis))
        return false;
    }
    return true;
  }

public:
  CombineBroadcastMulReducePattern(MLIRContext *context)
      : RewritePattern(ReduceOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto reduceOp = llvm::dyn_cast<ReduceOp>(op);
    if (!reduceOp)
      return failure();
    if (cast<RankedTensorType>(reduceOp.getOperand(0).getType()).getRank() != 3)
      return failure();
    // We must be reducing along the middle dim.
    if (reduceOp.getAxis() != 1)
      return failure();
    // only support reduce with simple addition
    Region &combineOp = reduceOp.getCombineOp();
    bool isReduceAdd = combineOp.hasOneBlock() &&
                       combineOp.front().getOperations().size() == 2 &&
                       isAddF32(&*combineOp.front().getOperations().begin());
    if (!isReduceAdd)
      return failure();
    // operand of reduce has to be mul
    auto mulOp = reduceOp.getOperand(0).getDefiningOp<arith::MulFOp>();
    if (!mulOp)
      return failure();
    // mul operand has to be broadcast
    auto broadcastLhsOp = mulOp.getOperand(0).getDefiningOp<BroadcastOp>();
    if (!broadcastLhsOp)
      return failure();
    auto broadcastRhsOp = mulOp.getOperand(1).getDefiningOp<BroadcastOp>();
    if (!broadcastRhsOp)
      return failure();
    // The first operand must be broadcasted from (M, K, 1) to (M, K, N), and
    // the second operand must go from (1, K, N) to (M, K, N).
    if (!isBroadcastAlongAxis(broadcastLhsOp, 2) ||
        !isBroadcastAlongAxis(broadcastRhsOp, 0))
      return failure();
    auto broadcastLhsShape =
        cast<ShapedType>(broadcastLhsOp.getType()).getShape();
    auto broadcastRhsShape =
        cast<ShapedType>(broadcastRhsOp.getType()).getShape();
    if (broadcastLhsShape[2] < 16 || broadcastRhsShape[0] < 16)
      return failure();
    Type newAccType = RankedTensorType::get(
        {broadcastLhsShape[0], broadcastRhsShape[2]},
        cast<ShapedType>(broadcastLhsOp.getSrc().getType()).getElementType());
    rewriter.setInsertionPoint(op);
    Value lhs = ReshapeOp::create(
        rewriter, op->getLoc(),
        broadcastLhsOp.getSrc().getType().getShape().drop_back(),
        broadcastLhsOp.getSrc());
    Value rhs = ReshapeOp::create(
        rewriter, op->getLoc(),
        broadcastRhsOp.getSrc().getType().getShape().drop_front(),
        broadcastRhsOp.getSrc());
    auto newAcc =
        SplatOp::create(rewriter, op->getLoc(), newAccType,
                        arith::ConstantOp::create(rewriter, op->getLoc(),
                                                  rewriter.getF32FloatAttr(0)));
    rewriter.replaceOpWithNewOp<DotOp>(op, lhs, rhs, newAcc,
                                       InputPrecision::IEEE, 0);
    return success();
  }
};

// When reducing a 1D tensor the order of elements of the tensor doesn't matter.
// Therefore we can relax the reshape to allow it to re-order elements.
class CombineReshapeReducePatterns : public mlir::OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::ReshapeOp reshapeOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (reshapeOp.getAllowReorder())
      return failure();
    if (reshapeOp.getType().getRank() != 1)
      return failure();
    for (Operation *user : reshapeOp->getUsers()) {
      if (!isa<triton::ReduceOp, triton::HistogramOp>(user))
        return failure();
    }
    rewriter.modifyOpInPlace(reshapeOp,
                             [&]() { reshapeOp.setAllowReorder(true); });
    return success();
  }
};

class RankedReduceDescriptorLoads : public mlir::OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::ReshapeOp reshapeOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto loadDef = reshapeOp.getSrc().getDefiningOp<triton::DescriptorLoadOp>();
    if (!loadDef || !loadDef->hasOneUse())
      return failure();
    int loadRank = loadDef.getType().getRank();
    int reshapeRank = reshapeOp.getType().getRank();
    if (!(reshapeRank < loadRank))
      return failure();
    ArrayRef<int64_t> loadShape = loadDef.getType().getShape();
    ArrayRef<int64_t> reshapeShape = reshapeOp.getType().getShape();
    for (int i = 0; i < loadRank - reshapeRank; ++i) {
      // Only rank reduce unit dims.
      if (loadShape[i] != 1)
        return failure();
    }
    if (loadShape.take_back(reshapeRank) != reshapeShape)
      return failure();
    rewriter.modifyOpInPlace(
        loadDef, [&]() { loadDef.getResult().setType(reshapeOp.getType()); });
    rewriter.replaceOp(reshapeOp, loadDef.getResult());
    return success();
  }
};

template <typename DotOpType, typename AddOpType>
class CombineDotAddPattern : public mlir::OpRewritePattern<AddOpType> {
public:
  using OpRewritePattern<AddOpType>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(AddOpType addOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto dotOp = addOp.getRhs().template getDefiningOp<DotOpType>();
    bool isDotLHS = false;
    if (!dotOp) {
      dotOp = addOp.getLhs().template getDefiningOp<DotOpType>();
      if (!dotOp) {
        return failure();
      }
      isDotLHS = true;
    }
    if (!dotOp->hasOneUse()) {
      return failure();
    }
    if (!isZero(dotOp.getC()))
      return failure();
    if constexpr (std::is_same_v<DotOpType, DotOp> &&
                  std::is_same_v<AddOpType, arith::AddFOp>) {
      if (dotOp.getMaxNumImpreciseAcc() != 0) {
        return failure();
      }
    }
    rewriter.modifyOpInPlace(dotOp, [&] {
      dotOp.getCMutable().assign(isDotLHS ? addOp.getRhs() : addOp.getLhs());
      dotOp->moveBefore(addOp);
    });
    rewriter.replaceAllUsesWith(addOp, dotOp.getResult());
    return success();
  }
};

// AddIOp(DotOp(a, b, c), d) and c==0 => DotOp(a, b, d)
// AddFOp(DotOp(a, b, c), d) and c==0 => DotOp(a, b, d)
// AddIOp(d, DotOp(a, b, c)) and c==0 => DotOp(a, b, d)
// AddFOp(d, DotOp(a, b, c)) and c==0 => DotOp(a, b, d)
using CombineDotAddIPattern = CombineDotAddPattern<DotOp, arith::AddIOp>;
using CombineDotAddFPattern = CombineDotAddPattern<DotOp, arith::AddFOp>;
using CombineDotScaledAddFPattern =
    CombineDotAddPattern<DotScaledOp, arith::AddFOp>;

// Return the value of p val if it is statically known to be a float
// constant
static std::optional<llvm::APFloat> getSplatFloatConstant(Value val) {
  if (auto splatOp = val.getDefiningOp<SplatOp>())
    val = splatOp.getSrc();
  Attribute attr;
  if (!matchPattern(val, m_Constant(&attr)))
    return std::nullopt;
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
    if (!denseAttr.isSplat())
      return std::nullopt;
    attr = denseAttr.getSplatValue<Attribute>();
  }
  if (auto floatAttr = dyn_cast_or_null<FloatAttr>(attr))
    return floatAttr.getValue();
  return std::nullopt;
}


//   truncf(maxnumf(extf(x), c)) => maxnumf(x, c_narrow)
template <typename OpTy>
class CombineTruncMinMaxExtPattern
    : public mlir::OpRewritePattern<arith::TruncFOp> {
private:
  // A wide operand expressible in the narrow type: either the source of an
  // extf from that type, or a constant that converts to it exactly.
  struct NarrowOperand {
    Value val;
    std::optional<llvm::APFloat> cst;
  };

  static std::optional<NarrowOperand> matchNarrowOperand(Value operand,
                                                         Type narrowTy) {
    if (auto extOp = operand.getDefiningOp<arith::ExtFOp>()) {
      if (extOp.getIn().getType() != narrowTy)
        return std::nullopt;
      return NarrowOperand{extOp.getIn(), std::nullopt};
    }
    auto cVal = getSplatFloatConstant(operand);
    if (!cVal || cVal->isNaN())
      return std::nullopt;
    auto elemTy = cast<FloatType>(getElementTypeOrSelf(narrowTy));
    bool losesInfo = false;
    if (cVal->convert(elemTy.getFloatSemantics(),
                      llvm::APFloat::rmNearestTiesToEven,
                      &losesInfo) != llvm::APFloat::opOK ||
        losesInfo)
      return std::nullopt;
    return NarrowOperand{Value(), *cVal};
  }

public:
  using OpRewritePattern<arith::TruncFOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(arith::TruncFOp truncOp,
                  mlir::PatternRewriter &rewriter) const override {
    // The fold produces exactly representable values, but stay conservative
    // with explicit rounding modes.
    if (truncOp.getRoundingmode())
      return failure();
    auto innerOp = truncOp.getIn().getDefiningOp<OpTy>();
    if (!innerOp || !innerOp->hasOneUse())
      return failure();
    Type narrowTy = truncOp.getType();
    Type elemTy = getElementTypeOrSelf(narrowTy);
    if (!elemTy.isF16() && !elemTy.isBF16() && !elemTy.isF32())
      return failure();
    auto lhs = matchNarrowOperand(innerOp.getLhs(), narrowTy);
    auto rhs = matchNarrowOperand(innerOp.getRhs(), narrowTy);
    if (!lhs || !rhs)
      return failure();
    // Both operands constant is the arith folder's job.
    if (!lhs->val && !rhs->val)
      return failure();
    auto materialize = [&](NarrowOperand &operand) -> Value {
      if (operand.val)
        return operand.val;
      Attribute attr = FloatAttr::get(elemTy, *operand.cst);
      if (auto shapedTy = dyn_cast<ShapedType>(narrowTy))
        attr = DenseElementsAttr::get(shapedTy, attr);
      return arith::ConstantOp::create(rewriter, truncOp.getLoc(),
                                       cast<TypedAttr>(attr));
    };
    rewriter.replaceOpWithNewOp<OpTy>(truncOp, materialize(*lhs),
                                      materialize(*rhs));
    return success();
  }
};

using CombineTruncMaxNumFExtPattern =
    CombineTruncMinMaxExtPattern<arith::MaxNumFOp>;
using CombineTruncMinNumFExtPattern =
    CombineTruncMinMaxExtPattern<arith::MinNumFOp>;
using CombineTruncMaximumFExtPattern =
    CombineTruncMinMaxExtPattern<arith::MaximumFOp>;
using CombineTruncMinimumFExtPattern =
    CombineTruncMinMaxExtPattern<arith::MinimumFOp>;

//   truncf(divf(extf(x), c)) => mulf(x, 1/c)
class CombineTruncDivPow2Pattern
    : public mlir::OpRewritePattern<arith::TruncFOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(arith::TruncFOp truncOp,
                  mlir::PatternRewriter &rewriter) const override {
    // Only the default (round-to-nearest-even) truncation rounds like mulf.
    if (truncOp.getRoundingmode())
      return failure();
    auto divOp = truncOp.getIn().getDefiningOp<arith::DivFOp>();
    if (!divOp || !divOp->hasOneUse())
      return failure();
    auto extOp = divOp.getLhs().getDefiningOp<arith::ExtFOp>();
    if (!extOp)
      return failure();
    Value x = extOp.getIn();

    if (x.getType() != truncOp.getType())
      return failure();
    Type elemTy = getElementTypeOrSelf(x.getType());
    if (!elemTy.isF16() && !elemTy.isBF16() && !elemTy.isF32())
      return failure();
    auto divisor = getSplatFloatConstant(divOp.getRhs());
    if (!divisor)
      return failure();
    llvm::APFloat recip(divisor->getSemantics());
    if (!divisor->getExactInverse(&recip))
      return failure();
    bool losesInfo = false;
    if (recip.convert(cast<FloatType>(elemTy).getFloatSemantics(),
                      llvm::APFloat::rmNearestTiesToEven,
                      &losesInfo) != llvm::APFloat::opOK ||
        losesInfo)
      return failure();
    Attribute recipAttr = FloatAttr::get(elemTy, recip);
    if (auto shapedTy = dyn_cast<ShapedType>(truncOp.getType()))
      recipAttr = DenseElementsAttr::get(shapedTy, recipAttr);
    auto cst = arith::ConstantOp::create(rewriter, truncOp.getLoc(),
                                         cast<TypedAttr>(recipAttr));
    rewriter.replaceOpWithNewOp<arith::MulFOp>(truncOp, x, cst);
    return success();
  }
};

} // anonymous namespace

class CombineOpsPass : public impl::TritonCombineOpsBase<CombineOpsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp m = getOperation();

    patterns.add<CombineDotAddIPattern>(context);
    patterns.add<CombineDotAddFPattern>(context);
    patterns.add<CombineDotScaledAddFPattern>(context);
    patterns.add<CombineSelectMaskedLoadPattern>(context);
    patterns.add<CombineTruncMaxNumFExtPattern, CombineTruncMinNumFExtPattern,
                 CombineTruncMaximumFExtPattern, CombineTruncMinimumFExtPattern>(
        context);
    patterns.add<CombineTruncDivPow2Pattern>(context);
    patterns.add<CombineAddPtrPattern>(context);
    patterns.add<CombineBroadcastMulReducePattern>(context);
    patterns.add<CombineReshapeReducePatterns>(context);
    patterns.add<RankedReduceDescriptorLoads>(context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir::triton
