#include "triton/Analysis/BufferIndexAnalysis.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CheckedArithmetic.h"
#include <optional>

namespace mlir {

static int64_t normalizeModuloOffset(int64_t offset, int64_t modulus) {
  return ((offset % modulus) + modulus) % modulus;
}

/// A buffer index decomposed as `baseValue + constantOffset`, optionally
/// under a known modulus (recorded when the index is wrapped by remsi or
/// the pipeliner's select/cmpi idiom). Two expressions rooted at the
/// same base value with different offsets (modulo the same recorded modulus,
/// when both carry one) provably target different slots.
struct BufferIndexExpr {
  Value baseValue;
  int64_t constantOffset = 0;
  std::optional<int64_t> modulus;
  Value indexedSource;

  bool operator==(const BufferIndexExpr &other) const {
    return baseValue == other.baseValue &&
           constantOffset == other.constantOffset && modulus == other.modulus &&
           indexedSource == other.indexedSource;
  }

  bool isProvablyDifferentFrom(const BufferIndexExpr &other) const {
    return baseValue == other.baseValue &&
           indexedSource == other.indexedSource && modulus == other.modulus &&
           constantOffset != other.constantOffset;
  }
};

namespace {

enum class DFSState { Visiting, Visited };

/// A CFG is reducible iff every retreating edge in a depth-first traversal is
/// a backedge, i.e. its target dominates its source. Only blocks reachable
/// from the function entry matter to membar's forward dataflow analysis.
bool isReducibleCFG(Block *block, DenseMap<Block *, DFSState> &states,
                    DominanceInfo &dominanceInfo,
                    llvm::SmallPtrSetImpl<Block *> &loopHeaders) {
  states[block] = DFSState::Visiting;

  for (Block *successor : block->getTerminator()->getSuccessors()) {
    auto it = states.find(successor);
    if (it == states.end()) {
      if (!isReducibleCFG(successor, states, dominanceInfo, loopHeaders))
        return false;
      continue;
    }
    if (it->second == DFSState::Visiting) {
      if (!dominanceInfo.dominates(successor, block))
        return false;
      loopHeaders.insert(successor);
    }
  }

  states[block] = DFSState::Visited;
  return true;
}

bool isReducibleCFG(FunctionOpInterface funcOp, DominanceInfo &dominanceInfo,
                    llvm::SmallPtrSetImpl<Block *> &loopHeaders) {
  DenseMap<Block *, DFSState> states;
  return isReducibleCFG(&funcOp.getBlocks().front(), states, dominanceInfo,
                        loopHeaders);
}

std::optional<int64_t> getConstantIntValue(Value v) {
  APInt val;
  if (matchPattern(v, m_ConstantInt(&val)))
    return val.getSExtValue();
  return std::nullopt;
}

BufferIndexExpr analyzeBufferIndex(Value indexValue);
std::optional<BufferIndexExpr> matchModuloPattern(arith::SelectOp selectOp);

using SignedRange = std::pair<int64_t, int64_t>;

/// Conservative signed range [lo, hi] of a scalar integer. Fails rather than
/// guess: a block argument depending on itself, or any op not listed below.
std::optional<SignedRange>
getSignedRange(Value v, llvm::SmallPtrSetImpl<Value> &visiting,
               unsigned depth = 8) {
  if (auto c = getConstantIntValue(v))
    return SignedRange{*c, *c};
  if (depth == 0)
    return std::nullopt;

  // remsi(x, N) lies in (-N, N) whatever x is.
  if (auto remOp = v.getDefiningOp<arith::RemSIOp>()) {
    auto mod = getConstantIntValue(remOp.getRhs());
    if (mod && *mod > 0)
      return SignedRange{-(*mod - 1), *mod - 1};
    return std::nullopt;
  }

  if (auto selectOp = v.getDefiningOp<arith::SelectOp>()) {
    // The pipeliner's wrap, already proven to stay in [0, N).
    if (auto expr = matchModuloPattern(selectOp))
      return SignedRange{0, *expr->modulus - 1};
    auto t = getSignedRange(selectOp.getTrueValue(), visiting, depth - 1);
    auto f = getSignedRange(selectOp.getFalseValue(), visiting, depth - 1);
    if (!t || !f)
      return std::nullopt;
    return SignedRange{std::min(t->first, f->first),
                       std::max(t->second, f->second)};
  }

  if (auto addOp = v.getDefiningOp<arith::AddIOp>()) {
    auto lhs = getSignedRange(addOp.getLhs(), visiting, depth - 1);
    auto rhs = getSignedRange(addOp.getRhs(), visiting, depth - 1);
    if (!lhs || !rhs)
      return std::nullopt;
    auto lo = llvm::checkedAdd(lhs->first, rhs->first);
    auto hi = llvm::checkedAdd(lhs->second, rhs->second);
    auto intTy = dyn_cast<IntegerType>(v.getType());
    if (!lo || !hi || !intTy || intTy.getWidth() > 64 ||
        *lo < APInt::getSignedMinValue(intTy.getWidth()).getSExtValue() ||
        *hi > APInt::getSignedMaxValue(intTy.getWidth()).getSExtValue())
      return std::nullopt;
    return SignedRange{*lo, *hi};
  }

  // A block argument is bounded by the union of its incoming values, as long
  // as none of them needs its own bound.
  auto arg = dyn_cast<BlockArgument>(v);
  if (!arg || !visiting.insert(v).second)
    return std::nullopt;
  Block *block = arg.getOwner();
  std::optional<SignedRange> range;
  for (auto predIt = block->pred_begin(), e = block->pred_end(); predIt != e;
       ++predIt) {
    auto branch = dyn_cast<BranchOpInterface>((*predIt)->getTerminator());
    if (!branch) {
      range.reset();
      break;
    }
    Value incoming = branch.getSuccessorOperands(
        predIt.getSuccessorIndex())[arg.getArgNumber()];
    auto r =
        incoming ? getSignedRange(incoming, visiting, depth - 1) : std::nullopt;
    if (!r) {
      range.reset();
      break;
    }
    range = range ? SignedRange{std::min(range->first, r->first),
                                std::max(range->second, r->second)}
                  : *r;
  }
  visiting.erase(v);
  return range;
}

/// True if `x + c` (the given addi) cannot overflow its signed integer type:
/// either the op says so (nsw), or the range of `x` proves it.
bool isAddWithoutSignedOverflow(arith::AddIOp addOp, Value x, int64_t c) {
  if (arith::bitEnumContainsAny(addOp.getOverflowFlags(),
                                arith::IntegerOverflowFlags::nsw))
    return true;
  llvm::SmallPtrSet<Value, 8> visiting;
  auto range = getSignedRange(x, visiting);
  auto intTy = dyn_cast<IntegerType>(addOp.getType());
  if (!range || !intTy || intTy.getWidth() > 64)
    return false;
  auto lo = llvm::checkedAdd(range->first, c);
  auto hi = llvm::checkedAdd(range->second, c);
  return lo && hi &&
         *lo >= APInt::getSignedMinValue(intTy.getWidth()).getSExtValue() &&
         *hi <= APInt::getSignedMaxValue(intTy.getWidth()).getSExtValue();
}

bool isCFBlockArgProvablyBounded(BlockArgument blockArg, int64_t modulus,
                                 arith::SelectOp selectOp) {
  // For cf-form loops, the loop-carried value is a block argument. Each
  // incoming value must be either an in-range initial value or the matched
  // select that advances the counter.
  Block *header = blockArg.getOwner();
  unsigned argIdx = blockArg.getArgNumber();
  // Entry block arguments have no incoming operands to prove the bound.
  if (header->pred_begin() == header->pred_end())
    return false;

  for (auto predIt = header->pred_begin(), e = header->pred_end(); predIt != e;
       ++predIt) {
    Block *pred = *predIt;
    auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
    if (!branch)
      return false;

    auto operands = branch.getSuccessorOperands(predIt.getSuccessorIndex());
    Value incoming = operands[argIdx];

    if (incoming == selectOp.getResult())
      continue;

    auto c = getConstantIntValue(incoming);
    if (!c || *c < -1 || *c >= modulus)
      return false;
  }

  return true;
}

/// Verify that `base` is provably bounded by -1 <= base < N so that
///   select(cmpi sge/slt (addi(base, 1), N), ...)
/// truly equals (base + 1) % N. The select form returns 0 on the wrap arm
/// rather than (base + 1) - N; outside that range the two expressions
/// diverge and the match would be unsound.
///
/// Constants are checked directly. For cf-form loop-carried counters we prove
/// the bound inductively: incoming values to the loop-header block argument
/// must each be either an in-range constant or the matched select itself.
/// Given -1 <= base < N we have 0 <= base + 1 <= N, and the select maps N to 0
/// and otherwise returns base + 1, so the next value satisfies 0 <= next < N.
bool isBaseProvablyBounded(Value base, int64_t modulus,
                           arith::SelectOp selectOp) {
  assert(modulus > 0);

  if (auto c = getConstantIntValue(base))
    return *c >= -1 && *c < modulus;

  auto blockArg = dyn_cast<BlockArgument>(base);
  if (!blockArg)
    return false;

  return isCFBlockArgProvablyBounded(blockArg, modulus, selectOp);
}

/// Match the one-step modular wrap the pipeliner emits on its iter_arg:
///   select(cmpi sge (addi(base, 1), N), zero, addi(base, 1))
///   select(cmpi slt (addi(base, 1), N), addi(base, 1), zero)
/// Both equal (base + 1) % N only when -1 <= base < N; outside that range
/// the wrap arm would need to be (base + 1) - N, not 0, so accepting the
/// match unconditionally would be unsound. We require C == 1 and verify
/// the range assumption via isBaseProvablyBounded.
std::optional<BufferIndexExpr> matchModuloPattern(arith::SelectOp selectOp) {
  auto cmp = selectOp.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return std::nullopt;

  Value wrapVal, noWrapVal;
  if (cmp.getPredicate() == arith::CmpIPredicate::sge) {
    wrapVal = selectOp.getTrueValue();
    noWrapVal = selectOp.getFalseValue();
  } else if (cmp.getPredicate() == arith::CmpIPredicate::slt) {
    noWrapVal = selectOp.getTrueValue();
    wrapVal = selectOp.getFalseValue();
  } else {
    return std::nullopt;
  }

  auto wrapConst = getConstantIntValue(wrapVal);
  if (!wrapConst || *wrapConst != 0)
    return std::nullopt;

  auto addOp = noWrapVal.getDefiningOp<arith::AddIOp>();
  if (!addOp || cmp.getLhs() != addOp.getResult())
    return std::nullopt;

  // Try constant on RHS then LHS (addi is commutative).
  std::optional<int64_t> c = getConstantIntValue(addOp.getRhs());
  Value base = addOp.getLhs();
  if (!c) {
    c = getConstantIntValue(addOp.getLhs());
    base = addOp.getRhs();
  }
  if (!c || *c != 1)
    return std::nullopt;

  // Modulus must be a positive compile-time constant.
  auto mod = getConstantIntValue(cmp.getRhs());
  if (!mod || *mod <= 0)
    return std::nullopt;

  // The (base + 1) % N rewrite is only valid for -1 <= base < N.
  if (!isBaseProvablyBounded(base, *mod, selectOp))
    return std::nullopt;

  auto baseExpr = analyzeBufferIndex(base);
  // Nested moduli ((x mod M) + 1) mod N don't reduce to (x + 1) mod N in
  // general; keep the full select as the expression root.
  auto offset = llvm::checkedAdd(baseExpr.constantOffset, *c);
  if (baseExpr.modulus || !offset)
    return std::nullopt;
  BufferIndexExpr result{baseExpr.baseValue, *offset};
  result.modulus = *mod;
  return result;
}

// An addi is split into base + constant only when it cannot overflow (nsw, or
// a range proof); a wrapped sum is not base + constant modulo N. Anything else
// stays an opaque base, which only ever compares equal to itself.
BufferIndexExpr analyzeBufferIndex(Value indexValue) {
  if (auto c = getConstantIntValue(indexValue))
    return BufferIndexExpr{nullptr, *c};

  if (auto addOp = indexValue.getDefiningOp<arith::AddIOp>()) {
    auto composeWithConstant = [&](Value nonConst,
                                   int64_t constant) -> BufferIndexExpr {
      if (!isAddWithoutSignedOverflow(addOp, nonConst, constant))
        return BufferIndexExpr{indexValue, 0};
      auto baseExpr = analyzeBufferIndex(nonConst);
      // (x mod N) + C is not represented as (base, offset, mod); keep the
      // full addi as the expression root.
      auto offset = llvm::checkedAdd(baseExpr.constantOffset, constant);
      if (baseExpr.modulus || !offset)
        return BufferIndexExpr{indexValue, 0};
      return {baseExpr.baseValue, *offset};
    };
    if (auto offset = getConstantIntValue(addOp.getRhs()))
      return composeWithConstant(addOp.getLhs(), *offset);
    if (auto offset = getConstantIntValue(addOp.getLhs()))
      return composeWithConstant(addOp.getRhs(), *offset);
    return BufferIndexExpr{indexValue, 0};
  }

  if (auto selectOp = indexValue.getDefiningOp<arith::SelectOp>())
    if (auto result = matchModuloPattern(selectOp))
      return *result;

  // arith.remsi(x, N): strip the remainder and record N as the modulus.
  // N must be a positive compile-time constant.
  if (auto remOp = indexValue.getDefiningOp<arith::RemSIOp>()) {
    if (auto mod = getConstantIntValue(remOp.getRhs()); mod && *mod > 0) {
      auto result = analyzeBufferIndex(remOp.getLhs());
      // Nested modulus: keep the full remsi as the expression root.
      if (result.modulus)
        return BufferIndexExpr{indexValue, 0};
      result.modulus = *mod;
      return result;
    }
  }

  return BufferIndexExpr{indexValue, 0};
}

// Bounds the walk through chains of loop-carried slot views (one link per
// pipeline stage) and stops it on cycles such as a view carried unchanged.
constexpr unsigned kMaxCarriedViewDepth = 8;

std::optional<BufferIndexExpr>
analyzeSlot(Value value, DominanceInfo &dominanceInfo,
            const llvm::SmallPtrSetImpl<Block *> &loopHeaders, unsigned depth);

/// Derive the slot of a memdesc carried around a cf-form loop as a header
/// block argument, as the pipeliner does when it rotates the slot views of
/// the stages in flight. The result is relative to the current iteration's
/// loop counter, so it is comparable with the slots the body indexes itself.
///
/// The derivation is an induction over the loop's iterations. Let `c` be a
/// header argument advanced by exactly one slot along every backedge,
/// c' = (c + 1) mod N (the matched select/cmpi wrap). If every backedge passes
/// a view whose slot is (c + k) mod N, the carried view in the next iteration
/// is at (c' - 1 + k) mod N, i.e. (c + k - 1) mod N relative to its own
/// counter. This holds in the first iteration too only if every entry edge
/// passes a constant slot k0 with k0 = (c0 + k - 1) mod N, where c0 is the
/// counter's (constant) entry value; otherwise the proof fails.
std::optional<BufferIndexExpr>
analyzeLoopCarriedSlot(BlockArgument arg, DominanceInfo &dominanceInfo,
                       const llvm::SmallPtrSetImpl<Block *> &loopHeaders,
                       unsigned depth) {
  Block *header = arg.getOwner();
  if (!loopHeaders.contains(header))
    return std::nullopt;

  struct Incoming {
    Value view;
    SuccessorOperands operands;
  };
  SmallVector<Incoming> entries;
  std::optional<BufferIndexExpr> carried;
  for (auto predIt = header->pred_begin(), e = header->pred_end(); predIt != e;
       ++predIt) {
    Block *pred = *predIt;
    auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
    if (!branch)
      return std::nullopt;
    auto operands = branch.getSuccessorOperands(predIt.getSuccessorIndex());
    Value view = operands[arg.getArgNumber()];
    if (!view)
      return std::nullopt;
    if (!dominanceInfo.dominates(header, pred)) {
      entries.push_back({view, operands});
      continue;
    }

    // Backedge: the view handed to the next iteration.
    auto next = analyzeSlot(view, dominanceInfo, loopHeaders, depth);
    if (!next || !next->modulus)
      return std::nullopt;
    auto counter = dyn_cast_or_null<BlockArgument>(next->baseValue);
    if (!counter || counter.getOwner() != header)
      return std::nullopt;
    // The counter must advance by exactly one slot, modulo the same N, on
    // this backedge.
    Value counterNext = operands[counter.getArgNumber()];
    if (!counterNext)
      return std::nullopt;
    BufferIndexExpr step = analyzeBufferIndex(counterNext);
    if (step.baseValue != counter || step.modulus != next->modulus ||
        normalizeModuloOffset(step.constantOffset, *next->modulus) != 1)
      return std::nullopt;

    BufferIndexExpr shifted = *next;
    auto previous = llvm::checkedSub(next->constantOffset, int64_t(1));
    if (!previous)
      return std::nullopt;
    shifted.constantOffset = normalizeModuloOffset(*previous, *next->modulus);
    if (carried && !(*carried == shifted))
      return std::nullopt;
    carried = shifted;
  }
  if (!carried || entries.empty())
    return std::nullopt;

  // Base case: the first iteration must see the same slot.
  int64_t modulus = *carried->modulus;
  auto counter = cast<BlockArgument>(carried->baseValue);
  for (const Incoming &entry : entries) {
    auto initial = analyzeSlot(entry.view, dominanceInfo, loopHeaders, depth);
    if (!initial || initial->baseValue || initial->modulus ||
        initial->indexedSource != carried->indexedSource ||
        initial->constantOffset < 0 || initial->constantOffset >= modulus)
      return std::nullopt;
    Value counterInit = entry.operands[counter.getArgNumber()];
    auto c0 = counterInit ? getConstantIntValue(counterInit) : std::nullopt;
    auto first =
        c0 ? llvm::checkedAdd(*c0, carried->constantOffset) : std::nullopt;
    if (!first ||
        initial->constantOffset != normalizeModuloOffset(*first, modulus))
      return std::nullopt;
  }
  return carried;
}

/// Returns the slot expression of a shared-memory view, if it selects one
/// slot of a multi-buffered allocation.
std::optional<BufferIndexExpr>
analyzeSlot(Value value, DominanceInfo &dominanceInfo,
            const llvm::SmallPtrSetImpl<Block *> &loopHeaders, unsigned depth) {
  // MemDescIndexOp selects a whole slot of a multi-buffered allocation; its
  // index operand identifies the slot. MemDescViewTrait producers (trans,
  // reshape, reinterpret, subslice) are slot-preserving, so we can walk
  // through them to find the underlying MemDescIndexOp.
  Value v = value;
  while (auto *def = v.getDefiningOp()) {
    if (auto indexOp = dyn_cast<triton::gpu::MemDescIndexOp>(def)) {
      BufferIndexExpr expr = analyzeBufferIndex(indexOp.getIndex());
      expr.indexedSource = indexOp.getSrc();
      return expr;
    }
    if (!def->hasTrait<OpTrait::MemDescViewTrait>())
      return std::nullopt;
    v = def->getOperand(0);
  }
  auto arg = dyn_cast<BlockArgument>(v);
  if (!arg || depth == 0)
    return std::nullopt;
  return analyzeLoopCarriedSlot(arg, dominanceInfo, loopHeaders, depth - 1);
}

} // namespace

BufferIndexAnalysis::BufferIndexAnalysis(FunctionOpInterface funcOp)
    : dominanceInfo(funcOp),
      hasReducibleCFG(isReducibleCFG(funcOp, dominanceInfo, loopHeaders)) {}

BufferIndexAnalysis::~BufferIndexAnalysis() = default;

bool areBufferIndicesProvablyDifferent(const AllocationSlice &a,
                                       const AllocationSlice &b) {
  auto *aExpr = a.bufferIndexExpr;
  auto *bExpr = b.bufferIndexExpr;
  if (!aExpr || !bExpr)
    return false;
  return aExpr->isProvablyDifferentFrom(*bExpr);
}

const BufferIndexExpr *BufferIndexAnalysis::intern(BufferIndexExpr expr) {
  // Euclidean normalization: make 0 <= constantOffset < m so negative constants
  // compare correctly against positive ones. Equivalent expressions (e.g.
  // offset 0 and offset m) share a single interned entry.
  if (expr.modulus) {
    int64_t m = *expr.modulus;
    expr.constantOffset = normalizeModuloOffset(expr.constantOffset, m);
  }

  for (const auto &existing : expressions)
    if (*existing == expr)
      return existing.get();

  auto owned = std::make_unique<BufferIndexExpr>(expr);
  const BufferIndexExpr *result = owned.get();
  expressions.push_back(std::move(owned));
  return result;
}

AllocationSlice
BufferIndexAnalysis::makeSlice(Value value, Interval<size_t> allocationInterval,
                               Allocation::BufferId bufferId) {
  AllocationSlice slice(value, allocationInterval, bufferId);
  attachBufferIndex(slice, value);
  return slice;
}

void BufferIndexAnalysis::attachBufferIndex(AllocationSlice &slice,
                                            Value value) {
  if (!hasReducibleCFG)
    return;

  auto expr =
      analyzeSlot(value, dominanceInfo, loopHeaders, kMaxCarriedViewDepth);
  if (!expr)
    return;
  slice.bufferIndexExpr = intern(std::move(*expr));
}

bool BufferIndexAnalysis::isBackedgeSuccessor(Operation *terminator,
                                              Block *successor) const {
  if (isa<BranchOpInterface>(terminator))
    return !hasReducibleCFG ||
           !isa<FunctionOpInterface>(terminator->getParentOp()) ||
           dominanceInfo.dominates(successor, terminator->getBlock());
  // Region-to-region edges can start another dynamic iteration (scf.for and
  // scf.while). Conservatively invalidate on all such edges; exits to the
  // parent operation's continuation do not revisit the region's definitions.
  return isa<RegionBranchTerminatorOpInterface>(terminator) &&
         successor->getParentOp() == terminator->getParentOp();
}

bool BufferIndexAnalysis::entersLoop(Operation *terminator) const {
  if (!isa<BranchOpInterface>(terminator))
    return false;
  return llvm::any_of(terminator->getSuccessors(), [&](Block *successor) {
    return loopHeaders.contains(successor) &&
           !isBackedgeSuccessor(terminator, successor);
  });
}

} // namespace mlir
