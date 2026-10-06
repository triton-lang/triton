#include "triton/Analysis/AffineAddress.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton {

std::optional<std::pair<int64_t, int64_t>>
AffineAddress::bounds(ArrayRef<int64_t> shape) const {
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

std::optional<int64_t>
AffineAddress::linearStep(ArrayRef<int64_t> shape) const {
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

// JIT can produce duplicated bases and masks before CSE. Compare only pure
// regionless expressions, with a work budget to bound matching cost.
static bool equivalentExpressionsImpl(Value a, Value b, unsigned &budget) {
  if (a == b)
    return true;
  if (!a || !b || !budget || a.getType() != b.getType())
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
      [&](Value a, Value b) {
        return success(equivalentExpressionsImpl(a, b, budget));
      },
      nullptr, OperationEquivalence::IgnoreLocations);
}

bool haveEquivalentPureExpressions(Value lhs, Value rhs, unsigned budget) {
  return equivalentExpressionsImpl(lhs, rhs, budget);
}

std::optional<AffineAddress> AffineAddressAnalysis::get(Value value,
                                                        unsigned depth) {
  auto it = cache.find(value);
  if (it != cache.end())
    return it->second;
  // Bound recursion and total work, independently of tensor element count.
  if (depth >= 64 || cache.size() >= 256)
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

std::optional<AffineAddress> AffineAddressAnalysis::compute(Value v,
                                                            unsigned depth) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  Type elem = getElementTypeOrSelf(v.getType());
  if ((ty && ty.getEncoding()) ||
      (!isa<PointerType>(elem) && !isa<IntegerType>(elem)))
    return std::nullopt;
  if (!ty && isa<PointerType>(elem))
    return AffineAddress{v, 0, {}};
  if (auto i = dyn_cast<IntegerType>(elem); i && i.getWidth() > 64)
    return std::nullopt;
  APInt constant;
  if (matchPattern(v, m_ConstantInt(&constant)))
    return AffineAddress{Value(), constant.getSExtValue(),
                         SmallVector<int64_t>(ty ? ty.getRank() : 0, 0)};
  if (auto range = v.getDefiningOp<MakeRangeOp>())
    return AffineAddress{Value(), range.getStart(), {1}};
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
      auto bounds = src->bounds(srcTy ? srcTy.getShape() : ArrayRef<int64_t>());
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
  if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::ShLIOp, AddPtrOp,
           JoinOp>(op))
    return std::nullopt;
  auto lhs = get(op->getOperand(0), depth);
  auto rhs = get(op->getOperand(1), depth);
  if (!lhs || !rhs)
    return std::nullopt;
  if (isa<JoinOp>(op)) {
    unsigned budget = 64;
    int64_t delta;
    if (!equivalentExpressionsImpl(lhs->base, rhs->base, budget) ||
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
  auto isConstant = [](const AffineAddress &a) {
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

std::optional<int64_t>
AffineAddressAnalysis::getConstantOffsetDifference(Value a, Value b) {
  if (a.getType() != b.getType())
    return std::nullopt;
  auto lhs = get(a), rhs = get(b);
  if (!lhs || !rhs || lhs->steps != rhs->steps)
    return std::nullopt;
  unsigned budget = 64;
  int64_t delta;
  if (!equivalentExpressionsImpl(lhs->base, rhs->base, budget) ||
      llvm::SubOverflow(rhs->start, lhs->start, delta))
    return std::nullopt;
  return delta;
}

std::optional<ContiguousAccessPlan>
AffineAddressAnalysis::planInterleavedAccess(ArrayRef<Value> ptrs) {
  if (ptrs.size() < 2 || ptrs.size() > INT32_MAX ||
      !llvm::isPowerOf2_64(ptrs.size()))
    return std::nullopt;
  auto ty = dyn_cast<RankedTensorType>(ptrs.front().getType());
  if (!ty || !ty.getRank() || ty.getEncoding() ||
      !isa<PointerType>(ty.getElementType()))
    return std::nullopt;
  auto address = get(ptrs.front());
  if (!address || !address->base ||
      (ty.getShape().back() != 1 && address->steps.back() != ptrs.size()))
    return std::nullopt;
  for (auto [i, ptr] : llvm::enumerate(ptrs)) {
    auto delta = getConstantOffsetDifference(ptrs.front(), ptr);
    if (!delta || *delta != i)
      return std::nullopt;
  }
  SmallVector<int64_t> shape(ty.getShape());
  if (llvm::MulOverflow(shape.back(), int64_t(ptrs.size()), shape.back()) ||
      llvm::any_of(shape, [](int64_t size) { return size > INT32_MAX; }))
    return std::nullopt;
  address->steps.back() = 1;
  auto bounds = address->bounds(shape);
  if (!bounds)
    return std::nullopt;
  bool useI32 =
      llvm::isInt<32>(bounds->first) && llvm::isInt<32>(bounds->second);
  for (auto [size, step] : llvm::zip_equal(shape, address->steps)) {
    int64_t span;
    if (llvm::MulOverflow(size - 1, step, span))
      return std::nullopt;
    useI32 &= llvm::isInt<32>(step) && llvm::isInt<32>(span);
  }
  return ContiguousAccessPlan{*address, shape, useI32 ? 32u : 64u};
}

} // namespace mlir::triton
