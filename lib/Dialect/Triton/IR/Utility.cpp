#include "triton/Dialect/Triton/IR/Utility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;

Value tt::getPredMask(RewriterBase &rewriter, Type typeLike, Value currentMask,
                      Value pred) {
  Type maskType = tt::getI1SameShape(typeLike);
  Location loc = pred.getLoc();
  Value mask = pred;
  if (isa<RankedTensorType>(maskType)) {
    mask = tt::SplatOp::create(rewriter, loc, maskType, pred);
  }
  if (currentMask) {
    mask = arith::AndIOp::create(rewriter, loc, mask, currentMask);
  }
  return mask;
}

Value tt::getLastInductionValue(OpBuilder &b, scf::ForOp loop) {
  Location loc = loop.getLoc();
  // (ub - lb -1) // step * step + lb
  Value diff =
      arith::SubIOp::create(b, loc, loop.getUpperBound(), loop.getLowerBound());
  diff = arith::SubIOp::create(
      b, loc, diff,
      arith::ConstantOp::create(b, loc, b.getIntegerAttr(diff.getType(), 1)));
  Value ceilStep = arith::MulIOp::create(
      b, loc, arith::DivSIOp::create(b, loc, diff, loop.getStep()),
      loop.getStep());
  return arith::AddIOp::create(b, loc, ceilStep, loop.getLowerBound());
}

bool tt::isKernel(FunctionOpInterface funcOp) {
  return funcOp.getVisibility() == SymbolTable::Visibility::Public;
}

bool tt::isHostSideDescriptor(Value v) {
  auto arg = dyn_cast<BlockArgument>(v);
  if (!arg)
    return false;
  auto funcOp = dyn_cast<FunctionOpInterface>(arg.getOwner()->getParentOp());
  if (!funcOp)
    return false;
  return tt::isKernel(funcOp);
}

unsigned tt::getBitwidth(RankedTensorType ty) {
  auto isPtr = isa<PointerType>(ty.getElementType());
  return isPtr ? kPtrBitWidth : std::max(ty.getElementTypeBitWidth(), 8u);
}

std::optional<ConstantIntRanges>
tt::getBoundFromCmpPredicate(arith::CmpIPredicate predicate,
                             const ConstantIntRanges &other, bool isLhs) {
  using P = arith::CmpIPredicate;
  if (predicate == P::eq)
    return other;

  unsigned width = other.umin().getBitWidth();
  if (predicate == P::ne) {
    auto value = other.getConstantValue();
    if (!value)
      return ConstantIntRanges::maxRange(width);
    APInt umin = APInt::getZero(width), umax = APInt::getMaxValue(width);
    APInt smin = APInt::getSignedMinValue(width);
    APInt smax = APInt::getSignedMaxValue(width);
    if (*value == umin)
      ++umin;
    if (*value == umax)
      --umax;
    if (*value == smin)
      ++smin;
    if (*value == smax)
      --smax;
    return ConstantIntRanges(umin, umax, smin, smax);
  }

  bool isSigned = predicate == P::slt || predicate == P::sle ||
                  predicate == P::sgt || predicate == P::sge;
  bool less = predicate == P::slt || predicate == P::sle ||
              predicate == P::ult || predicate == P::ule;
  bool strict = predicate == P::slt || predicate == P::sgt ||
                predicate == P::ult || predicate == P::ugt;
  bool upperBound = less == isLhs;
  APInt min =
      isSigned ? APInt::getSignedMinValue(width) : APInt::getZero(width);
  APInt max =
      isSigned ? APInt::getSignedMaxValue(width) : APInt::getMaxValue(width);
  if (upperBound) {
    APInt bound = isSigned ? other.smax() : other.umax();
    if (strict && bound == min)
      return std::nullopt;
    max = strict ? bound - 1 : bound;
  } else {
    APInt bound = isSigned ? other.smin() : other.umin();
    if (strict && bound == max)
      return std::nullopt;
    min = strict ? bound + 1 : bound;
  }
  return ConstantIntRanges::range(min, max, isSigned);
}

std::optional<ConstantIntRanges> tt::getBoundFromCmpOp(arith::CmpIOp cmpOp,
                                                       Value anchor) {
  // Assumption ranges replace inferred ranges, so an unrepresentable hole
  // must not introduce an otherwise uninformative assumption.
  if (cmpOp.getPredicate() == arith::CmpIPredicate::ne)
    return std::nullopt;
  bool isLhs = cmpOp.getLhs() == anchor;
  auto value = getConstantIntValue(
      getAsOpFoldResult(isLhs ? cmpOp.getRhs() : cmpOp.getLhs()));
  if (!value)
    return std::nullopt;

  unsigned width = ConstantIntRanges::getStorageBitwidth(anchor.getType());
  APInt bound(width, static_cast<uint64_t>(*value), /*isSigned=*/true);
  auto result = getBoundFromCmpPredicate(
      cmpOp.getPredicate(), ConstantIntRanges::constant(bound), isLhs);
  auto predicate = cmpOp.getPredicate();
  bool isSigned = predicate == arith::CmpIPredicate::slt ||
                  predicate == arith::CmpIPredicate::sle ||
                  predicate == arith::CmpIPredicate::sgt ||
                  predicate == arith::CmpIPredicate::sge;
  if (result && isSigned && result->smin().isMinSignedValue() &&
      isa_and_nonnull<GetProgramIdOp, GetNumProgramsOp>(anchor.getDefiningOp()))
    result = result->intersection(ConstantIntRanges::fromSigned(
        APInt::getZero(width), APInt::getSignedMaxValue(width)));
  return result;
}
