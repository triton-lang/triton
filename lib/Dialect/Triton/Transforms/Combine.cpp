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
#include "triton/Analysis/AffineAddress.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/DiscardableAttributes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"
#include <functional>

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

// Materialize a proven logical access. Both pointer-expression normalization
// and independent-load coalescing use this; neither path depends on the other.
static Value materializeContiguousAccess(const ContiguousAccessPlan &plan,
                                         RankedTensorType ptrTy,
                                         PatternRewriter &rewriter,
                                         Location loc) {
  auto shape = plan.shape;
  auto steps = plan.address.steps;
  if (plan.address.linearStep(shape) == 1 &&
      ptrTy.getNumElements() <= INT32_MAX) {
    shape = {ptrTy.getNumElements()};
    steps = {1};
  }
  Type indexTy = rewriter.getIntegerType(plan.indexBitWidth);
  auto offsetsTy = RankedTensorType::get(shape, indexTy);
  Value offsets = arith::ConstantOp::create(
      rewriter, loc,
      DenseElementsAttr::get(
          offsetsTy, rewriter.getIntegerAttr(indexTy, plan.address.start)));
  for (unsigned d = 0; d < shape.size(); ++d) {
    if (!steps[d] || shape[d] == 1)
      continue;
    Value axis = MakeRangeOp::create(
        rewriter, loc, RankedTensorType::get({shape[d]}, rewriter.getI32Type()),
        0, shape[d]);
    auto axisTy = RankedTensorType::get({shape[d]}, indexTy);
    if (plan.indexBitWidth != 32)
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
  auto bases = SplatOp::create(rewriter, loc, ptrTy, plan.address.base);
  auto fullOffsets =
      ReshapeOp::create(rewriter, loc, ptrTy.clone(indexTy), offsets, false);
  return AddPtrOp::create(rewriter, loc, ptrTy, bases, fullOffsets);
}

// Join is one client of the common access analysis, not the canonical form
// that other memory transformations have to construct.
class CombineAdjacentPointerJoin : public OpRewritePattern<JoinOp> {
public:
  using OpRewritePattern<JoinOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(JoinOp op,
                                PatternRewriter &rewriter) const override {
    AffineAddressAnalysis analysis;
    auto plan = analysis.planInterleavedAccess({op.getLhs(), op.getRhs()});
    if (!plan)
      return failure();
    rewriter.replaceOp(op, materializeContiguousAccess(
                               *plan, cast<RankedTensorType>(op.getType()),
                               rewriter, op.getLoc()));
    return success();
  }
};

// Pack tensor values in coordinate order. This helper only rearranges mask or
// fallback data; pointer construction uses the access plan directly.
static Value packInterleavedValues(ArrayRef<Value> values,
                                   RankedTensorType packedTy,
                                   PatternRewriter &rewriter, Location loc) {
  auto ty = cast<RankedTensorType>(values.front().getType());
  Value packed;
  if (llvm::all_of(values,
                   [&](Value value) { return value == values.front(); })) {
    auto expanded =
        ExpandDimsOp::create(rewriter, loc, values.front(), ty.getRank());
    SmallVector<int64_t> shape(ty.getShape());
    shape.push_back(values.size());
    packed = BroadcastOp::create(
        rewriter, loc, RankedTensorType::get(shape, ty.getElementType()),
        expanded);
  } else if (values.size() == 2) {
    packed = JoinOp::create(rewriter, loc, values[0], values[1]);
  } else {
    auto even = JoinOp::create(rewriter, loc, values[0], values[2]);
    auto odd = JoinOp::create(rewriter, loc, values[1], values[3]);
    packed = JoinOp::create(rewriter, loc, even, odd);
  }
  return ReshapeOp::create(rewriter, loc, packedTy, packed, false);
}

// Combine independent reads before layout assignment. Address equivalence is
// necessary but not sufficient: do not cross writes, atomics, barriers, unknown
// effects or control-flow regions, and do not speculate load-dependent
// operands.
class CoalesceAdjacentLoads {
public:
  CoalesceAdjacentLoads(DenseMap<Operation *, int64_t> &alignments,
                        int numThreads)
      : alignments(alignments), numThreads(numThreads) {}
  LogicalResult matchAndRewrite(LoadOp first, PatternRewriter &rewriter) const {
    auto ty = dyn_cast<RankedTensorType>(first.getType());
    if (!ty || !ty.getRank() || ty.getEncoding() || first.getIsVolatile() ||
        first->hasAttr("tt.contiguity") || first->hasAttr("tt.divisibility") ||
        first->hasAttr("tt.constancy"))
      return failure();
    // Otherwise grouping can spread a logical pair across physical threads,
    // requiring shared-memory conversions just to reconstruct its components.
    if (ty.getNumElements() < numThreads)
      return failure();
    Type elem = ty.getElementType();
    if ((!isa<IntegerType, FloatType>(elem)) ||
        elem.getIntOrFloatBitWidth() < 8 || elem.getIntOrFloatBitWidth() > 32)
      return failure();
    AffineAddressAnalysis analysis;
    auto address = analysis.get(first.getPtr());
    if (!address || !address->base)
      return failure();
    int64_t stride = ty.getShape().back() == 1 ? 2 : address->steps.back();
    if (stride != 2 && stride != 4)
      return failure();
    unsigned width = stride;

    SmallVector<std::pair<int64_t, LoadOp>> group{{0, first}};
    unsigned budget = 64;
    for (Operation *op = first->getNextNode(); op && budget--;
         op = op->getNextNode()) {
      auto next = dyn_cast<LoadOp>(op);
      if (!next) {
        if (op->getNumRegions() || !isPure(op))
          break;
        continue;
      }
      if (next.getIsVolatile())
        break;
      // Equal masks give pair/quad-uniform validity, so grouping can expose a
      // wide load. Different masks remain independent, even when grouping them
      // would be semantically legal, until there is a profitability model.
      if (next.getType() != ty ||
          !haveEquivalentPureExpressions(next.getMask(), first.getMask()) ||
          bool(next.getOther()) != bool(first.getOther()) ||
          next->getAttrs() != first->getAttrs())
        continue;
      auto delta =
          analysis.getConstantOffsetDifference(first.getPtr(), next.getPtr());
      if (!delta || *delta <= -int64_t(width) || *delta >= int64_t(width) ||
          llvm::any_of(group,
                       [&](auto entry) { return entry.first == *delta; }))
        continue;
      group.emplace_back(*delta, next);
      if (group.size() == width)
        break;
    }
    if (group.size() != width)
      return failure();
    llvm::sort(group, [](auto a, auto b) { return a.first < b.first; });
    SmallVector<Value> ptrs;
    for (auto [offset, load] : group)
      ptrs.push_back(load.getPtr());
    auto plan = analysis.planInterleavedAccess(ptrs);
    if (!plan)
      return failure();
    if (alignments.lookup(group.front().second) <
        width * elem.getIntOrFloatBitWidth() / 8)
      return failure();

    // Overlapping windows need a joint reuse/layout cost model. Combining one
    // pair in isolation can give neighboring windows incompatible vector
    // widths. Leave those reads alone, even though their addresses are legally
    // packable.
    auto packedBounds = plan->address.bounds(plan->shape);
    for (bool forward : {false, true}) {
      unsigned remaining = 64;
      for (Operation *op = forward ? first->getNextNode()
                                   : first->getPrevNode();
           op && remaining--;
           op = forward ? op->getNextNode() : op->getPrevNode()) {
        if (auto load = dyn_cast<LoadOp>(op)) {
          if (load.getIsVolatile())
            break;
          if (llvm::any_of(group,
                           [&](auto entry) { return entry.second == load; }))
            continue;
          if (!analysis.getConstantOffsetDifference(ptrs.front(),
                                                    load.getPtr()))
            continue;
          auto other = analysis.get(load.getPtr());
          auto bounds = other->bounds(ty.getShape());
          if (bounds && bounds->first <= packedBounds->second &&
              packedBounds->first <= bounds->second)
            return failure();
        } else if (op->getNumRegions() || !isPure(op)) {
          break;
        }
      }
    }

    // Validate all operands before mutating IR. Only clone a bounded DAG of
    // speculatable, regionless pure expressions to the first load's position.
    DominanceInfo dominance;
    SmallVector<Operation *> hoist;
    llvm::SmallPtrSet<Operation *, 16> visited;
    std::function<LogicalResult(Value, unsigned)> available =
        [&](Value value, unsigned depth) {
          if (dominance.properlyDominates(value, first))
            return success();
          auto *op = value.getDefiningOp();
          if (!op || op->getBlock() != first->getBlock() || depth == 32 ||
              op->getNumRegions() || !isPure(op))
            return failure();
          if (!visited.insert(op).second)
            return success();
          if (visited.size() > 64)
            return failure();
          for (Value operand : op->getOperands())
            if (failed(available(operand, depth + 1)))
              return failure();
          hoist.push_back(op);
          return success();
        };
    if (failed(available(plan->address.base, 0)))
      return failure();
    for (auto [offset, load] : group)
      if (load.getOther() && failed(available(load.getOther(), 0)))
        return failure();

    rewriter.setInsertionPoint(first);
    IRMapping mapping;
    for (Operation *op : hoist)
      rewriter.clone(*op, mapping);
    plan->address.base = mapping.lookupOrDefault(plan->address.base);
    auto packedTy = RankedTensorType::get(plan->shape, elem);
    Value ptr = materializeContiguousAccess(
        *plan,
        packedTy.clone(
            cast<RankedTensorType>(first.getPtr().getType()).getElementType()),
        rewriter, first.getLoc());
    mapping.map(first.getPtr(), ptr);
    for (bool isMask : {true, false}) {
      if (!(isMask ? first.getMask() : first.getOther()))
        continue;
      SmallVector<Value> values;
      for (auto [offset, load] : group)
        values.push_back(mapping.lookupOrDefault(isMask ? first.getMask()
                                                        : load.getOther()));
      Value packed = packInterleavedValues(
          values,
          packedTy.clone(getElementTypeOrSelf(values.front().getType())),
          rewriter, first.getLoc());
      mapping.map(isMask ? first.getMask() : first.getOther(), packed);
    }
    auto load = cast<LoadOp>(rewriter.clone(*first, mapping));
    load.getResult().setType(packedTy);
    SmallVector<int64_t> unpackShape(ty.getShape());
    unpackShape.push_back(2);
    if (width == 4)
      unpackShape.push_back(2);
    auto unpacked = ReshapeOp::create(rewriter, first.getLoc(),
                                      RankedTensorType::get(unpackShape, elem),
                                      load, false);
    auto split = SplitOp::create(rewriter, first.getLoc(), unpacked);
    SmallVector<Value> outputs;
    if (width == 2) {
      outputs = {split.getOutLHS(), split.getOutRHS()};
    } else {
      auto even = SplitOp::create(rewriter, first.getLoc(), split.getOutLHS());
      auto odd = SplitOp::create(rewriter, first.getLoc(), split.getOutRHS());
      outputs = {even.getOutLHS(), odd.getOutLHS(), even.getOutRHS(),
                 odd.getOutRHS()};
    }
    for (auto [entry, value] : llvm::zip_equal(group, outputs)) {
      alignments.erase(entry.second);
      rewriter.replaceOp(entry.second, value);
    }
    return success();
  }

private:
  DenseMap<Operation *, int64_t> &alignments;
  int numThreads;
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
    AffineAddressAnalysis analysis;
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
  using TritonCombineOpsBase::TritonCombineOpsBase;
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
    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
      return;
    }
    if (numThreads <= 0)
      return;

    // Snapshot alignment only after pointer normalization. Consume each
    // original load at most once; do not query a stale dataflow analysis after
    // mutations or let newly allocated operations reuse worklist entries.
    SmallVector<LoadOp> worklist;
    m.walk([&](LoadOp load) {
      auto ty = dyn_cast<RankedTensorType>(load.getType());
      if (!ty || !ty.getRank() || ty.getEncoding() ||
          ty.getNumElements() < numThreads || load.getIsVolatile())
        return;
      AffineAddressAnalysis analysis;
      auto address = analysis.get(load.getPtr());
      if (address && address->base &&
          (ty.getShape().back() == 1 || address->steps.back() == 2 ||
           address->steps.back() == 4))
        worklist.push_back(load);
    });
    // Most kernels have no interleaved accesses. Do not run another module
    // dataflow analysis unless there are at least two potential candidates.
    if (worklist.size() < 2)
      return;
    DenseMap<Operation *, int64_t> alignments;
    {
      ModuleAxisInfoAnalysis axisInfo(m);
      for (LoadOp load : worklist) {
        auto ty = cast<RankedTensorType>(load.getType());
        auto *info = axisInfo.getAxisInfo(load.getPtr());
        if (!info)
          continue;
        alignments[load] = info->getDivisibility(ty.getRank() - 1);
      }
    }
    CoalesceAdjacentLoads combine(alignments, numThreads);
    PatternRewriter rewriter(context);
    for (LoadOp load : worklist)
      if (alignments.contains(load)) {
        rewriter.setInsertionPoint(load);
        (void)combine.matchAndRewrite(load, rewriter);
      }
  }
};

} // namespace mlir::triton
