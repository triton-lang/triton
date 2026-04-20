//===- BufferIndexAnalysis.cpp - Decompose buffer-index SSA values --------===//
//
// Support for the membar analysis's dynamic-buffer-index disjointness
// check: given an SSA value used as a slot index into a multi-buffered
// shared-memory allocation, recover a symbolic form
// `baseValue + constantOffset (mod modulus?)` so the membar analysis can
// prove two accesses hit different slots.
//
// The entry point is analyzeBufferIndex (declared in Membar.h). Everything
// else here is internal to this file.
//
//===----------------------------------------------------------------------===//

#include "triton/Analysis/Membar.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

namespace mlir {
namespace {

std::optional<int64_t> getConstantIntValue(Value v) {
  APInt val;
  if (matchPattern(v, m_ConstantInt(&val)))
    return val.getSExtValue();
  return std::nullopt;
}

/// Verify that `base` is provably in [-1, N) so that
///   select(cmpi sge/slt (addi(base, 1), N), ...)
/// truly equals (base + 1) % N. The select form returns 0 on the wrap arm
/// rather than (base + 1) - N; outside that range the two expressions
/// diverge and the match would be unsound.
///
/// Three shapes are accepted:
///   (a) `base` is a compile-time constant in [-1, N).
///   (b) `base` is an `arith.remsi _, N`, which pins the value to [0, N).
///   (c) `base` is an `scf.for` iter_arg whose init is a compile-time
///       constant in [-1, N) and whose yield-back operand is exactly
///       `selectOp`. In that case the invariant is inductive: the init
///       is in range by construction, and every later iteration's value
///       is `select(...) ∈ [0, N)`, which stays in [-1, N).
bool isBaseProvablyBounded(Value base, int64_t modulus,
                           arith::SelectOp selectOp) {
  assert(modulus > 0);

  if (auto c = getConstantIntValue(base))
    return *c >= -1 && *c < modulus;

  if (auto remOp = base.getDefiningOp<arith::RemSIOp>())
    if (auto m = getConstantIntValue(remOp.getRhs()); m && *m == modulus)
      return true;

  auto blockArg = dyn_cast<BlockArgument>(base);
  if (!blockArg)
    return false;
  auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
  if (!forOp || blockArg.getOwner() != forOp.getBody())
    return false;
  // Argument 0 is the induction variable; iter_args start at 1.
  unsigned argIdx = blockArg.getArgNumber();
  if (argIdx == 0)
    return false;
  unsigned iterIdx = argIdx - 1;
  if (iterIdx >= forOp.getNumRegionIterArgs())
    return false;

  auto initC = getConstantIntValue(forOp.getInitArgs()[iterIdx]);
  if (!initC || *initC < -1 || *initC >= modulus)
    return false;

  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (iterIdx >= yieldOp.getNumOperands())
    return false;
  return yieldOp.getOperand(iterIdx) == selectOp.getResult();
}

/// Match the one-step modular wrap the pipeliner emits on its iter_arg:
///   select(cmpi sge (addi(base, 1), N), zero, addi(base, 1))
///   select(cmpi slt (addi(base, 1), N), addi(base, 1), zero)
/// Both equal (base + 1) % N only when base ∈ [-1, N); outside that range
/// the wrap arm would need to be (base + 1) - N, not 0, so accepting the
/// match unconditionally would be unsound. We require C == 1 (general
/// offsets with modular wrap should go through arith.remsi, handled in
/// analyzeBufferIndex) and verify the range assumption via
/// isBaseProvablyBounded.
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

  // Modulus must be a positive compile-time constant for comparisons to
  // be meaningful.
  auto mod = getConstantIntValue(cmp.getRhs());
  if (!mod || *mod <= 0)
    return std::nullopt;

  // The (base + 1) % N rewrite is only valid for base ∈ [-1, N).
  if (!isBaseProvablyBounded(base, *mod, selectOp))
    return std::nullopt;

  auto baseExpr = analyzeBufferIndex(base);
  // Nested moduli ((x mod M) + 1) mod N don't reduce to (x + 1) mod N in
  // general; bail out to an opaque expression.
  if (baseExpr.modulus)
    return std::nullopt;
  BufferIndexExpr result{baseExpr.baseValue, baseExpr.constantOffset + *c};
  result.modulus = *mod;
  return result;
}

} // namespace

BufferIndexExpr analyzeBufferIndex(Value indexValue) {
  if (auto c = getConstantIntValue(indexValue))
    return BufferIndexExpr{nullptr, *c};

  if (auto addOp = indexValue.getDefiningOp<arith::AddIOp>()) {
    auto composeWithConstant = [&](Value nonConst,
                                   int64_t constant) -> BufferIndexExpr {
      auto baseExpr = analyzeBufferIndex(nonConst);
      // (x mod N) + C is not representable as (base, offset, mod); bail
      // out to an opaque expression.
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
    if (auto result = matchModuloPattern(selectOp))
      return *result;

  // arith.remsi(x, N): strip the remainder and record N as the modulus.
  // N must be a positive compile-time constant.
  if (auto remOp = indexValue.getDefiningOp<arith::RemSIOp>()) {
    if (auto mod = getConstantIntValue(remOp.getRhs()); mod && *mod > 0) {
      auto result = analyzeBufferIndex(remOp.getLhs());
      // Nested modulus: bail to opaque (see matchModuloPattern).
      if (result.modulus)
        return BufferIndexExpr{indexValue, 0};
      result.modulus = *mod;
      return result;
    }
  }

  return BufferIndexExpr{indexValue, 0};
}

} // namespace mlir
