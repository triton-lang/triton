#include "triton/Analysis/BufferIndexAnalysis.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
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

  bool operator==(const BufferIndexExpr &other) const {
    return baseValue == other.baseValue &&
           constantOffset == other.constantOffset && modulus == other.modulus;
  }

  bool isProvablyDifferentFrom(const BufferIndexExpr &other) const {
    if (baseValue != other.baseValue)
      return false;
    if (modulus || other.modulus) {
      if (modulus != other.modulus)
        return false;
      int64_t m = *modulus;
      // Euclidean normalization: make 0 <= offset < m so that
      // negative constants compare correctly against positive ones.
      int64_t a = normalizeModuloOffset(constantOffset, m);
      int64_t b = normalizeModuloOffset(other.constantOffset, m);
      return a != b;
    }
    return constantOffset != other.constantOffset;
  }
};

namespace {

std::optional<int64_t> getConstantIntValue(Value v) {
  APInt val;
  if (matchPattern(v, m_ConstantInt(&val)))
    return val.getSExtValue();
  return std::nullopt;
}

BufferIndexExpr analyzeBufferIndex(Value indexValue,
                                   const DominanceInfo &dominanceInfo);

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
/// Constants are checked directly. For loop-carried counters we prove the
/// bound inductively: initial values must satisfy -1 <= init < N, and
/// recurrent incoming values must be the select itself. Given -1 <= base < N
/// we have 0 <= base + 1 <= N, and the select maps N to 0 and otherwise
/// returns base + 1, so the next value satisfies 0 <= next < N.
bool isBaseProvablyBounded(Value base, int64_t modulus,
                           arith::SelectOp selectOp) {
  assert(modulus > 0);

  if (auto c = getConstantIntValue(base))
    return *c >= -1 && *c < modulus;

  auto blockArg = dyn_cast<BlockArgument>(base);
  if (!blockArg)
    return false;

  auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
  if (forOp && blockArg.getOwner() == forOp.getBody()) {
    // Argument 0 is the induction variable; iter_args start at 1.
    unsigned argIdx = blockArg.getArgNumber();
    if (argIdx == 0)
      return false;
    unsigned iterIdx = argIdx - 1;

    auto initC = getConstantIntValue(forOp.getInitArgs()[iterIdx]);
    if (!initC || *initC < -1 || *initC >= modulus)
      return false;

    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    return yieldOp.getOperand(iterIdx) == selectOp.getResult();
  }

  return isCFBlockArgProvablyBounded(blockArg, modulus, selectOp);
}

/// Match the one-step modular wrap the pipeliner emits on its iter_arg:
///   select(cmpi sge (addi(base, 1), N), zero, addi(base, 1))
///   select(cmpi slt (addi(base, 1), N), addi(base, 1), zero)
/// Both equal (base + 1) % N only when -1 <= base < N; outside that range
/// the wrap arm would need to be (base + 1) - N, not 0, so accepting the
/// match unconditionally would be unsound. We require C == 1 and verify
/// the range assumption via isBaseProvablyBounded.
std::optional<BufferIndexExpr>
matchModuloPattern(arith::SelectOp selectOp,
                   const DominanceInfo &dominanceInfo) {
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

  auto baseExpr = analyzeBufferIndex(base, dominanceInfo);
  // Nested moduli ((x mod M) + 1) mod N don't reduce to (x + 1) mod N in
  // general; keep the full select as the expression root.
  if (baseExpr.modulus)
    return std::nullopt;
  BufferIndexExpr result{baseExpr.baseValue, baseExpr.constantOffset + *c};
  result.modulus = *mod;
  return result;
}

// Slot indices are assumed not to overflow signed integer arithmetic; use a
// wider index type if the pipeline counter can reach the integer range.
BufferIndexExpr analyzeBufferIndex(Value indexValue,
                                   const DominanceInfo &dominanceInfo) {
  if (auto c = getConstantIntValue(indexValue))
    return BufferIndexExpr{nullptr, *c};

  if (auto addOp = indexValue.getDefiningOp<arith::AddIOp>()) {
    auto composeWithConstant = [&](Value nonConst,
                                   int64_t constant) -> BufferIndexExpr {
      auto baseExpr = analyzeBufferIndex(nonConst, dominanceInfo);
      // (x mod N) + C is not represented as (base, offset, mod); keep the
      // full addi as the expression root.
      if (baseExpr.modulus)
        return BufferIndexExpr{indexValue, 0};
      return {baseExpr.baseValue, baseExpr.constantOffset + constant};
    };
    if (auto offset = getConstantIntValue(addOp.getRhs()))
      return composeWithConstant(addOp.getLhs(), *offset);
    if (auto offset = getConstantIntValue(addOp.getLhs()))
      return composeWithConstant(addOp.getRhs(), *offset);
  }

  if (auto selectOp = indexValue.getDefiningOp<arith::SelectOp>())
    if (auto result = matchModuloPattern(selectOp, dominanceInfo))
      return *result;

  // arith.remsi(x, N): strip the remainder and record N as the modulus.
  // N must be a positive compile-time constant.
  if (auto remOp = indexValue.getDefiningOp<arith::RemSIOp>()) {
    if (auto mod = getConstantIntValue(remOp.getRhs()); mod && *mod > 0) {
      auto result = analyzeBufferIndex(remOp.getLhs(), dominanceInfo);
      // Nested modulus: keep the full remsi as the expression root.
      if (result.modulus)
        return BufferIndexExpr{indexValue, 0};
      result.modulus = *mod;
      return result;
    }
  }

  return BufferIndexExpr{indexValue, 0};
}

Value extractBufferIndex(Value value) {
  // MemDescIndexOp selects a whole slot of a multi-buffered allocation; its
  // index operand identifies the slot. MemDescViewTrait producers (trans,
  // reshape, reinterpret, subslice) are slot-preserving, so we can walk
  // through them to find the underlying MemDescIndexOp.
  Value v = value;
  while (auto *def = v.getDefiningOp()) {
    if (auto indexOp = dyn_cast<triton::gpu::MemDescIndexOp>(def))
      return indexOp.getIndex();
    if (!def->hasTrait<OpTrait::MemDescViewTrait>())
      break;
    v = def->getOperand(0);
  }
  return Value();
}

} // namespace

BufferIndexAnalysis::BufferIndexAnalysis(FunctionOpInterface funcOp)
    : dominanceInfo(funcOp) {}

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
  // Canonicalize the modular offset so 0 <= constantOffset < m. Equivalent
  // expressions (e.g. offset 0 and offset m) share a single interned entry.
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
  Value index = extractBufferIndex(value);
  if (!index)
    return;
  slice.bufferIndexExpr = intern(analyzeBufferIndex(index, dominanceInfo));
}

bool BufferIndexAnalysis::isBackedgeSuccessor(Operation *terminator,
                                              Block *successor) const {
  auto br = dyn_cast<RegionBranchTerminatorOpInterface>(terminator);
  if (br && isa<RegionBranchOpInterface>(br->getParentOp())) {
    Region *succRegion = successor->getParent();
    if (succRegion == br->getParentOp()->getParentRegion())
      return false;
    // A successor region whose number is <= the terminator's region number
    // denotes re-entry into the same or an earlier region: scf.for yield ->
    // body (same region), scf.while after -> before.
    return succRegion->getRegionNumber() <=
           br->getParentRegion()->getRegionNumber();
  }

  if (isa<BranchOpInterface>(terminator))
    return dominanceInfo.dominates(successor, terminator->getBlock());
  return false;
}

void BufferIndexAnalysis::invalidateBufferIndices(BlockInfo &info) const {
  auto rebuild = [](BlockInfo::SliceMapT &m) {
    BlockInfo::SliceMapT rebuilt;
    for (const auto &[slice, ops] : m) {
      AllocationSlice key = slice;
      key.bufferIndexExpr = nullptr;
      rebuilt[key].insert(ops.begin(), ops.end());
    }
    m = std::move(rebuilt);
  };
  rebuild(info.syncReadSlices);
  rebuild(info.syncWriteSlices);
}

} // namespace mlir
