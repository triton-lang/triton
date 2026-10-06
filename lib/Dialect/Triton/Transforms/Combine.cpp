#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/DiscardableAttributes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/MathExtras.h"

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

// A constant affine tensor: base + start + sum(index[d] * steps[d]). The
// optional base is a scalar pointer, never a tensor of unknown addresses.
// Check every integer intermediate, not just the final address: sign extending
// a wrapping i32 offset is not equivalent to doing the arithmetic in i64.
struct JoinAddress {
  Value base;
  int64_t start = 0;
  SmallVector<int64_t> steps;

  std::optional<std::pair<int64_t, int64_t>>
  bounds(ArrayRef<int64_t> shape) const {
    int64_t lo = start, hi = start;
    for (auto [size, step] : llvm::zip_equal(shape, steps)) {
      int64_t span;
      if (llvm::MulOverflow(size - 1, step, span) ||
          llvm::AddOverflow(lo, std::min<int64_t>(span, 0), lo) ||
          llvm::AddOverflow(hi, std::max<int64_t>(span, 0), hi))
        return std::nullopt;
    }
    return std::pair(lo, hi);
  }

  std::optional<int64_t> linearStep(ArrayRef<int64_t> shape) const {
    std::optional<int64_t> step;
    int64_t size = 1;
    for (int d = shape.size() - 1; d >= 0; --d) {
      if (shape[d] == 1)
        continue;
      if (!step)
        step = steps[d];
      int64_t expected;
      if (llvm::MulOverflow(*step, size, expected) || expected != steps[d] ||
          llvm::MulOverflow(size, shape[d], size))
        return std::nullopt;
    }
    return step.value_or(0);
  }
};

// JIT can produce duplicated scalar block bases before CSE. Compare only pure
// regionless expressions, with a work budget to bound matching cost.
static bool sameJoinBase(Value a, Value b, unsigned &budget) {
  if (a == b)
    return true;
  if (!budget || a.getType() != b.getType())
    return false;
  --budget;
  auto x = dyn_cast<OpResult>(a), y = dyn_cast<OpResult>(b);
  if (!x || !y || x.getResultNumber() != y.getResultNumber())
    return false;
  auto *lhs = x.getOwner(), *rhs = y.getOwner();
  if (lhs->getNumRegions() || rhs->getNumRegions() ||
      !isMemoryEffectFree(lhs) || !isMemoryEffectFree(rhs))
    return false;
  return OperationEquivalence::isEquivalentTo(
      lhs, rhs,
      [&](Value a, Value b) { return success(sameJoinBase(a, b, budget)); },
      nullptr, OperationEquivalence::IgnoreLocations);
}

class JoinAddressAnalysis {
public:
  std::optional<JoinAddress> get(Value value, unsigned depth = 0) {
    auto it = cache.find(value);
    if (it != cache.end())
      return it->second;
    // Bound recursion and total work, independently of tensor element count.
    if (depth == 64 || cache.size() >= 256)
      return std::nullopt;
    auto result = compute(value, depth + 1);
    if (result) {
      auto ty = dyn_cast<RankedTensorType>(value.getType());
      ArrayRef<int64_t> shape = ty ? ty.getShape() : ArrayRef<int64_t>();
      for (auto [size, step] : llvm::zip_equal(shape, result->steps))
        if (size == 1)
          step = 0;
      auto bounds = result->bounds(shape);
      unsigned width =
          result->base
              ? 64
              : getElementTypeOrSelf(value.getType()).getIntOrFloatBitWidth();
      if (!bounds || !llvm::isIntN(width, bounds->first) ||
          !llvm::isIntN(width, bounds->second))
        result = std::nullopt;
    }
    cache[value] = result;
    return result;
  }

private:
  DenseMap<Value, std::optional<JoinAddress>> cache;

  std::optional<JoinAddress> compute(Value v, unsigned depth) {
    auto ty = dyn_cast<RankedTensorType>(v.getType());
    Type elem = getElementTypeOrSelf(v.getType());
    if ((ty && ty.getEncoding()) ||
        (!isa<PointerType>(elem) && !isa<IntegerType>(elem)))
      return std::nullopt;
    if (!ty && isa<PointerType>(elem))
      return JoinAddress{v, 0, {}};
    if (auto i = dyn_cast<IntegerType>(elem); i && i.getWidth() > 64)
      return std::nullopt;
    APInt constant;
    if (matchPattern(v, m_ConstantInt(&constant)))
      return JoinAddress{Value(), constant.getSExtValue(),
                         SmallVector<int64_t>(ty ? ty.getRank() : 0, 0)};
    if (auto range = v.getDefiningOp<MakeRangeOp>())
      return JoinAddress{Value(), range.getStart(), {1}};
    if (auto splat = v.getDefiningOp<SplatOp>()) {
      auto src = get(splat.getSrc(), depth);
      if (src)
        src->steps.assign(ty.getRank(), 0);
      return src;
    }
    if (auto expand = v.getDefiningOp<ExpandDimsOp>()) {
      auto src = get(expand.getSrc(), depth);
      if (src)
        src->steps.insert(src->steps.begin() + expand.getAxis(), 0);
      return src;
    }
    if (auto broadcast = v.getDefiningOp<BroadcastOp>())
      return get(broadcast.getSrc(), depth);
    if (auto trans = v.getDefiningOp<TransOp>()) {
      auto src = get(trans.getSrc(), depth);
      if (src) {
        auto steps = src->steps;
        for (auto [d, axis] : llvm::enumerate(trans.getOrder()))
          src->steps[d] = steps[axis];
      }
      return src;
    }
    if (auto reshape = v.getDefiningOp<ReshapeOp>()) {
      auto src = get(reshape.getSrc(), depth);
      if (!src)
        return std::nullopt;
      // Partition into contiguous chunks, from minor to major. A reshape may
      // split or merge within a chunk but must not cross a row-pitch gap.
      SmallVector<std::pair<int64_t, int64_t>> chunks;
      auto shape = reshape.getSrc().getType().getShape();
      for (int d = shape.size() - 1; d >= 0; --d) {
        if (shape[d] == 1)
          continue;
        int64_t nextStep;
        if (!chunks.empty() &&
            !llvm::MulOverflow(chunks.back().first, chunks.back().second,
                               nextStep) &&
            nextStep == src->steps[d]) {
          if (llvm::MulOverflow(chunks.back().first, shape[d],
                                chunks.back().first))
            return std::nullopt;
        } else {
          chunks.emplace_back(shape[d], src->steps[d]);
        }
      }
      src->steps.assign(ty.getRank(), 0);
      unsigned chunk = 0;
      for (int d = ty.getRank() - 1; d >= 0; --d) {
        int64_t size = ty.getShape()[d];
        if (size == 1)
          continue;
        if (chunk == chunks.size() || chunks[chunk].first % size)
          return std::nullopt;
        auto &[remaining, step] = chunks[chunk];
        src->steps[d] = step;
        remaining /= size;
        if (remaining == 1)
          ++chunk;
        else if (llvm::MulOverflow(step, size, step))
          return std::nullopt;
      }
      return src;
    }
    auto *op = v.getDefiningOp();
    if (!op)
      return std::nullopt;
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op)) {
      auto src = get(op->getOperand(0), depth);
      if (src && isa<arith::ExtUIOp>(op)) {
        auto srcTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
        auto bounds =
            src->bounds(srcTy ? srcTy.getShape() : ArrayRef<int64_t>());
        if (!bounds)
          return std::nullopt;
        if (bounds->first < 0) {
          // Zero extension of an entirely negative narrow range is a constant
          // translation. A range crossing zero instead has a discontinuity.
          unsigned width = getElementTypeOrSelf(op->getOperand(0).getType())
                               .getIntOrFloatBitWidth();
          if (bounds->second >= 0 || width >= 63 ||
              llvm::AddOverflow(src->start, int64_t(1) << width, src->start))
            return std::nullopt;
        }
      }
      return src;
    }
    if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::ShLIOp,
             AddPtrOp, JoinOp>(op))
      return std::nullopt;
    auto lhs = get(op->getOperand(0), depth);
    auto rhs = get(op->getOperand(1), depth);
    if (!lhs || !rhs)
      return std::nullopt;
    if (isa<JoinOp>(op)) {
      unsigned budget = 64;
      int64_t delta;
      if (!sameJoinBase(lhs->base, rhs->base, budget) ||
          lhs->steps != rhs->steps ||
          llvm::SubOverflow(rhs->start, lhs->start, delta))
        return std::nullopt;
      lhs->steps.push_back(delta);
      return lhs;
    }
    if (rhs->base)
      return std::nullopt;
    if (isa<arith::AddIOp, arith::SubIOp, AddPtrOp>(op)) {
      bool subtract = isa<arith::SubIOp>(op);
      auto combine = [&](int64_t a, int64_t b, int64_t &out) {
        return subtract ? llvm::SubOverflow(a, b, out)
                        : llvm::AddOverflow(a, b, out);
      };
      if (combine(lhs->start, rhs->start, lhs->start))
        return std::nullopt;
      for (auto [a, b] : llvm::zip_equal(lhs->steps, rhs->steps))
        if (combine(a, b, a))
          return std::nullopt;
      return lhs;
    }
    auto isConstant = [](const JoinAddress &a) {
      return llvm::all_of(a.steps, [](int64_t step) { return step == 0; });
    };
    if (isa<arith::MulIOp>(op) && !isConstant(*rhs))
      std::swap(lhs, rhs);
    if (!isConstant(*rhs))
      return std::nullopt;
    int64_t factor = rhs->start;
    if (isa<arith::ShLIOp>(op)) {
      if (factor < 0 ||
          factor >= std::min<unsigned>(63, elem.getIntOrFloatBitWidth()))
        return std::nullopt;
      factor = int64_t(1) << factor;
    }
    if (llvm::MulOverflow(lhs->start, factor, lhs->start))
      return std::nullopt;
    for (auto &step : lhs->steps)
      if (llvm::MulOverflow(step, factor, step))
        return std::nullopt;
    return lhs;
  }
};

// Expose increasing contiguous pairs to AxisInfo, independent of the source
// spelling (mul/shift, casts, nonzero ranges, reshapes, broadcast row strides).
class CombineAdjacentPointerJoin : public OpRewritePattern<JoinOp> {
public:
  using OpRewritePattern<JoinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(JoinOp op,
                                PatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<RankedTensorType>(op.getLhs().getType());
    if (!srcTy || srcTy.getRank() == 0 || srcTy.getEncoding() ||
        !isa<PointerType>(srcTy.getElementType()))
      return failure();
    JoinAddressAnalysis analysis;
    auto address = analysis.get(op.getResult());
    if (!address || !address->base || address->steps.back() != 1 ||
        (srcTy.getShape().back() != 1 &&
         address->steps[srcTy.getRank() - 1] != 2))
      return failure();
    auto loc = op.getLoc();
    auto pairTy = cast<RankedTensorType>(op.getType());
    SmallVector<int64_t> shape(srcTy.getShape());
    if (llvm::MulOverflow(shape.back(), int64_t(2), shape.back()))
      return failure();
    auto steps = address->steps;
    steps.pop_back();
    steps.back() = 1;
    if (address->linearStep(pairTy.getShape()) == 1) {
      shape = {pairTy.getNumElements()};
      steps = {1};
    }
    if (llvm::any_of(shape, [](int64_t n) { return n > INT32_MAX; }))
      return failure();
    auto bounds = address->bounds(pairTy.getShape());
    bool useI32 =
        llvm::isInt<32>(bounds->first) && llvm::isInt<32>(bounds->second);
    for (auto [size, step] : llvm::zip_equal(shape, steps))
      useI32 &= llvm::isInt<32>(step) && llvm::isInt<32>((size - 1) * step);
    Type indexTy = useI32 ? rewriter.getI32Type() : rewriter.getI64Type();
    auto offsetsTy = RankedTensorType::get(shape, indexTy);
    Value offsets = arith::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(
            offsetsTy, rewriter.getIntegerAttr(indexTy, address->start)));
    for (unsigned d = 0; d < shape.size(); ++d) {
      if (!steps[d] || shape[d] == 1)
        continue;
      Value axis = MakeRangeOp::create(
          rewriter, loc,
          RankedTensorType::get({shape[d]}, rewriter.getI32Type()), 0,
          shape[d]);
      auto axisTy = RankedTensorType::get({shape[d]}, indexTy);
      if (!useI32)
        axis = arith::ExtSIOp::create(rewriter, loc, axisTy, axis);
      if (steps[d] != 1) {
        auto scale = arith::ConstantOp::create(
            rewriter, loc,
            DenseElementsAttr::get(axisTy,
                                   rewriter.getIntegerAttr(indexTy, steps[d])));
        axis = arith::MulIOp::create(rewriter, loc, axis, scale);
      }
      for (unsigned e = 0; e < shape.size(); ++e)
        if (e != d)
          axis = ExpandDimsOp::create(rewriter, loc, axis, e);
      if (axis.getType() != offsetsTy)
        axis = BroadcastOp::create(rewriter, loc, offsetsTy, axis);
      offsets = arith::AddIOp::create(rewriter, loc, offsets, axis);
    }
    auto bases = SplatOp::create(rewriter, loc, pairTy, address->base);
    auto fullOffsets =
        ReshapeOp::create(rewriter, loc, pairTy.clone(indexTy), offsets, false);
    rewriter.replaceOpWithNewOp<AddPtrOp>(op, pairTy, bases, fullOffsets);
    return success();
  }
};

// In particular, a duplicated mask is constant along the new pair axis.
class CombineIdenticalJoin : public OpRewritePattern<JoinOp> {
public:
  using OpRewritePattern<JoinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(JoinOp op,
                                PatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<RankedTensorType>(op.getLhs().getType());
    if (!srcTy || srcTy.getEncoding() || !srcTy.getElementType().isInteger(1) ||
        op.getLhs() != op.getRhs())
      return failure();
    SmallVector<int64_t> shape(srcTy.getShape());
    shape.push_back(1);
    auto expanded = ExpandDimsOp::create(
        rewriter, op.getLoc(),
        RankedTensorType::get(shape, srcTy.getElementType()), op.getLhs(),
        srcTy.getRank());
    rewriter.replaceOpWithNewOp<BroadcastOp>(op, op.getType(), expanded);
    return success();
  }
};

// Collapse the contiguous innermost pair into its preceding dimension so
// coalescing does not choose a different row ownership from the subsequent
// split/output computation. Preserve outer dimensions and their row pitches.
class FlattenContiguousPairLoad : public OpRewritePattern<LoadOp> {
public:
  using OpRewritePattern<LoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LoadOp op,
                                PatternRewriter &rewriter) const override {
    auto ty = dyn_cast<RankedTensorType>(op.getType());
    if (!ty || ty.getEncoding() || ty.getRank() < 2 ||
        ty.getShape().back() != 2 || op.getIsVolatile() ||
        op->hasAttr("tt.contiguity") || op->hasAttr("tt.divisibility") ||
        op->hasAttr("tt.constancy"))
      return failure();
    // This is a pair-unpacking optimization, not a general layout heuristic.
    // Leave matrix/reduction consumers and shared values to layout assignment.
    if (!op.getResult().hasOneUse())
      return failure();
    Operation *consumer = *op.getResult().getUsers().begin();
    while (
        isa<arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncFOp,
            arith::TruncIOp, arith::BitcastOp, arith::SIToFPOp, arith::UIToFPOp,
            arith::FPToSIOp, arith::FPToUIOp, FpToFpOp>(consumer)) {
      if (!consumer->getResult(0).hasOneUse())
        return failure();
      consumer = *consumer->getResult(0).getUsers().begin();
    }
    if (!isa<SplitOp>(consumer))
      return failure();
    auto ptr = op.getPtr().getDefiningOp<AddPtrOp>();
    if (!ptr)
      return failure();
    auto base = ptr.getPtr().getDefiningOp<SplatOp>();
    JoinAddressAnalysis analysis;
    auto address = analysis.get(ptr.getOffset());
    if (!base || !address || address->steps.back() != 1 ||
        (ty.getShape()[ty.getRank() - 2] != 1 &&
         address->steps[ty.getRank() - 2] != 2))
      return failure();
    SmallVector<int64_t> flatShape(ty.getShape());
    flatShape.pop_back();
    if (llvm::MulOverflow(flatShape.back(), int64_t(2), flatShape.back()))
      return failure();
    auto flatTy = RankedTensorType::get(flatShape, ty.getElementType());
    auto ptrTy = flatTy.clone(
        cast<RankedTensorType>(op.getPtr().getType()).getElementType());
    auto flatBase =
        SplatOp::create(rewriter, op.getLoc(), ptrTy, base.getSrc());
    auto flatOffset = ReshapeOp::create(
        rewriter, op.getLoc(),
        flatTy.clone(getElementTypeOrSelf(ptr.getOffset().getType())),
        ptr.getOffset(), false);
    auto flatPtr =
        AddPtrOp::create(rewriter, op.getLoc(), ptrTy, flatBase, flatOffset);
    IRMapping mapping;
    mapping.map(op.getPtr(), flatPtr);
    for (Value v : {op.getMask(), op.getOther()}) {
      if (!v)
        continue;
      auto vTy = cast<RankedTensorType>(v.getType());
      auto flat = ReshapeOp::create(
          rewriter, op.getLoc(), flatTy.clone(vTy.getElementType()), v, false);
      mapping.map(v, flat);
    }
    auto clone = cast<LoadOp>(rewriter.clone(*op, mapping));
    clone.getResult().setType(flatTy);
    auto result = ReshapeOp::create(rewriter, op.getLoc(), ty, clone, false);
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

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

    if (!loadOp.getResult().hasOneUse() ||
        !DominanceInfo().properlyDominates(falseValue, loadOp))
      return failure();

    rewriter.modifyOpInPlace(
        loadOp, [&] { loadOp.getOtherMutable().assign(falseValue); });
    rewriter.replaceOp(op, loadOp.getResult());
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
    if (!isAddF32(reduceOp.getSingleCombiner()))
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
    patterns.add<CombineAddPtrPattern>(context);
    patterns.add<CombineBroadcastMulReducePattern>(context);
    patterns.add<RankedReduceDescriptorLoads>(context);
    patterns.add<CombineAdjacentPointerJoin, CombineIdenticalJoin>(context);
    patterns.add<FlattenContiguousPairLoad>(context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir::triton
