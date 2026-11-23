#include "triton/Analysis/BufferIndexAnalysis.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <optional>

namespace mlir {

/// A slot index of the form `baseValue + constantOffset`, optionally reduced
/// by a remainder, a low-bit mask or the pipeliner's one-step wrap.
struct BufferIndexExpr {
  // Shared SSA base of the index arithmetic. Null for a constant index.
  Value baseValue;
  // Keep signed offsets until we know the modulus of the slot expression.
  int64_t constantOffset = 0;
  std::optional<int64_t> modulus;
  // The descriptor determines both the origin and the stride of each slot.
  Value indexedSource;
  // In the block owning baseValue, the slot is congruent to
  // loopPhase + iteration * loopStep modulo N when loopPhase is known.
  std::optional<int64_t> loopPhase;
  int64_t loopStep = 0;

  bool operator==(const BufferIndexExpr &) const = default;
};

namespace {

std::optional<int64_t> getConstantIntValue(Value v) {
  APInt val;
  if (matchPattern(v, m_ConstantInt(&val)))
    return val.getSExtValue();
  return std::nullopt;
}

bool allIncomingValues(BlockArgument blockArg,
                       llvm::function_ref<bool(Block *, Value)> check) {
  Block *header = blockArg.getOwner();
  unsigned argIdx = blockArg.getArgNumber();
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
    if (!incoming || !check(pred, incoming))
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

  return allIncomingValues(blockArg, [&](Block *, Value incoming) {
    if (incoming == selectOp.getResult())
      return true;
    auto c = getConstantIntValue(incoming);
    return c && *c >= -1 && *c < modulus;
  });
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

  if (auto seed = getConstantIntValue(base))
    return BufferIndexExpr{nullptr, *seed + 1, *mod};
  return BufferIndexExpr{base, 1, *mod};
}

BufferIndexExpr analyzeBufferIndex(Value indexValue) {
  if (auto c = getConstantIntValue(indexValue))
    return BufferIndexExpr{nullptr, *c};

  if (auto addOp = indexValue.getDefiningOp<arith::AddIOp>()) {
    auto composeWithConstant = [&](Value nonConst,
                                   int64_t constant) -> BufferIndexExpr {
      auto baseExpr = analyzeBufferIndex(nonConst);
      // (x mod N) + C is not represented as (base, offset, mod); keep the
      // full addi as the expression root.
      int64_t offset;
      if (baseExpr.modulus ||
          llvm::AddOverflow(baseExpr.constantOffset, constant, offset))
        return BufferIndexExpr{indexValue, 0};
      return {baseExpr.baseValue, offset};
    };
    if (auto offset = getConstantIntValue(addOp.getRhs()))
      return composeWithConstant(addOp.getLhs(), *offset);
    if (auto offset = getConstantIntValue(addOp.getLhs()))
      return composeWithConstant(addOp.getRhs(), *offset);
  }

  if (auto selectOp = indexValue.getDefiningOp<arith::SelectOp>())
    if (auto result = matchModuloPattern(selectOp))
      return *result;

  auto wrap = [&](Value value, int64_t modulus) -> BufferIndexExpr {
    auto result = analyzeBufferIndex(value);
    // Nested moduli do not generally reduce to one modulus.
    if (result.modulus)
      return {indexValue, 0};
    result.modulus = modulus;
    return result;
  };
  // For power-of-two N the low bits give the residue for either signedness.
  Operation *rem = indexValue.getDefiningOp();
  if (isa_and_nonnull<arith::RemSIOp, arith::RemUIOp>(rem)) {
    if (auto mod = getConstantIntValue(rem->getOperand(1));
        mod && *mod > 0 &&
        (isa<arith::RemSIOp>(rem) || llvm::isPowerOf2_64(*mod)))
      return wrap(rem->getOperand(0), *mod);
  }
  if (auto andOp = indexValue.getDefiningOp<arith::AndIOp>()) {
    for (unsigned i : {0u, 1u}) {
      auto mask = getConstantIntValue(andOp->getOperand(i));
      if (mask && llvm::isPowerOf2_64(*mask + 1))
        return wrap(andOp->getOperand(1 - i), *mask + 1);
    }
  }

  return BufferIndexExpr{indexValue, 0};
}

// Each entry must establish the same seed and each backedge add the same step.
void analyzeLoopPhase(BufferIndexExpr &expr, DominanceInfo &dominance) {
  auto arg = dyn_cast_or_null<BlockArgument>(expr.baseValue);
  if (!arg || !expr.modulus)
    return;
  std::optional<int64_t> seed, step;
  auto check = [&](Block *pred, Value incoming) {
    auto input = analyzeBufferIndex(incoming);
    bool backedge = dominance.dominates(arg.getOwner(), pred);
    if (input.baseValue != (backedge ? expr.baseValue : Value{}) ||
        (input.modulus && input.modulus != expr.modulus))
      return false;
    int64_t residue = llvm::mod(input.constantOffset, *expr.modulus);
    auto &expected = backedge ? step : seed;
    if (expected && *expected != residue)
      return false;
    expected = residue;
    return true;
  };
  if (allIncomingValues(arg, check) && seed) {
    expr.loopPhase = llvm::mod(*seed + expr.constantOffset, *expr.modulus);
    expr.loopStep = step.value_or(0);
  }
}

triton::gpu::MemDescIndexOp extractBufferIndex(Value value) {
  // Views and single-block pipeline stages preserve the selected slot.
  // Stop at block arguments, which may carry a different slot on each edge.
  while (auto *def = value.getDefiningOp()) {
    if (auto index = dyn_cast<triton::gpu::MemDescIndexOp>(def))
      return index;
    if (def->hasTrait<OpTrait::MemDescViewTrait>()) {
      value = def->getOperand(0);
      continue;
    }
    auto stage = dyn_cast<scf::ExecuteRegionOp>(def);
    if (!stage || !stage.getRegion().hasOneBlock())
      break;
    auto yield =
        dyn_cast<scf::YieldOp>(stage.getRegion().front().getTerminator());
    if (!yield)
      break;
    value = yield.getOperand(cast<OpResult>(value).getResultNumber());
  }
  return {};
}

} // namespace

BufferIndexAnalysis::BufferIndexAnalysis(FunctionOpInterface funcOp)
    : dominanceInfo(funcOp), hasReducibleCFG(true) {
  Region &body = funcOp.getFunctionBody();
  if (const auto *info = getLoopInfo(&body)) {
    // A CFG is reducible iff every retreating edge in a depth-first traversal
    // is a backedge, i.e. its target dominates its source. Only blocks
    // reachable from the function entry matter to membar's forward dataflow
    // analysis.
    llvm::ReversePostOrderTraversal<Block *> blocks(&body.front());
    hasReducibleCFG = !llvm::containsIrreducibleCFG<Block *>(blocks, *info);
  }
}

BufferIndexAnalysis::~BufferIndexAnalysis() = default;

const triton::membar::Subslice *BufferIndexAnalysis::getSubslice(Value value) {
  using triton::membar::Subslice;
  auto view = value.getDefiningOp<triton::gpu::MemDescSubsliceOp>();
  if (!view)
    return nullptr;
  auto &coords = subslices[value];
  if (!coords)
    coords = std::make_unique<Subslice>(
        Subslice{view.getSrc().getAsOpaquePointer(), view.getOffsets(),
                 view.getType().getShape()});
  return coords.get();
}

bool triton::membar::disjointSubslices(const Subslice *a, const Subslice *b) {
  if (!a || !b || a->source != b->source)
    return false;
  for (auto [startA, sizeA, startB, sizeB] :
       llvm::zip_equal(a->offsets, a->shape, b->offsets, b->shape))
    if (startA + sizeA <= startB || startB + sizeB <= startA)
      return true;
  return false;
}

const CFGLoopInfo *BufferIndexAnalysis::getLoopInfo(Region *region) const {
  if (region->hasOneBlock())
    return nullptr;
  auto &info = loops[region];
  if (!info)
    info = std::make_unique<CFGLoopInfo>(dominanceInfo.getDomTree(region));
  return info.get();
}

bool areBufferIndicesProvablyDifferent(const BufferIndexExpr *a,
                                       const BufferIndexExpr *b) {
  // For shared slots (p + offset) % N with positive non-power-of-two N,
  // signed i32 overflow in the additions is undefined. We deliberately omit
  // the wraparound case so Membar can prove small dynamic indices disjoint
  // even when their bounds are unknown.
  if (!a || !b || a->indexedSource != b->indexedSource ||
      a->modulus != b->modulus)
    return false;
  if (a->baseValue == b->baseValue)
    return a->constantOffset != b->constantOffset;
  return a->loopPhase && b->loopPhase && a->loopPhase != b->loopPhase &&
         a->loopStep == b->loopStep &&
         cast<BlockArgument>(a->baseValue).getOwner() ==
             cast<BlockArgument>(b->baseValue).getOwner();
}

const BufferIndexExpr *BufferIndexAnalysis::getIndexExpr(Value value) {
  // In irreducible CFGs, a cycle can rebind an index without a dominance
  // backedge that invalidates the proof.
  if (!hasReducibleCFG)
    return nullptr;
  auto index = extractBufferIndex(value);
  if (!index || !isa<triton::gpu::SharedMemorySpaceAttr>(
                    index.getSrc().getType().getMemorySpace()))
    return nullptr;
  auto expr = analyzeBufferIndex(index.getIndex());
  expr.indexedSource = index.getSrc();
  // Plain i32 sums wrap modulo 2^32, which also preserves low-bit masks.
  expr.constantOffset =
      llvm::mod(expr.constantOffset, expr.modulus.value_or(int64_t{1} << 32));
  analyzeLoopPhase(expr, dominanceInfo);
  for (const auto &existing : expressions)
    if (*existing == expr)
      return existing.get();
  expressions.push_back(std::make_unique<BufferIndexExpr>(expr));
  return expressions.back().get();
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
  if (!isa<BranchOpInterface>(terminator) ||
      !isa<FunctionOpInterface>(terminator->getParentOp()))
    return false;
  const auto *info = getLoopInfo(terminator->getParentRegion());
  return info &&
         llvm::any_of(terminator->getSuccessors(), [&](Block *successor) {
           return info->isLoopHeader(successor) &&
                  !isBackedgeSuccessor(terminator, successor);
         });
}

} // namespace mlir
