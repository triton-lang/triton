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

std::optional<ConstantIntRanges> tt::getBoundFromCmpOp(arith::CmpIOp cmpOp,
                                                       Value anchor) {
  bool isSigned = true;
  switch (cmpOp.getPredicate()) {
  case arith::CmpIPredicate::uge:
  case arith::CmpIPredicate::ugt:
  case arith::CmpIPredicate::ule:
  case arith::CmpIPredicate::ult:
    isSigned = false;
  default:
    break;
  }

  bool anchorIsLhs = cmpOp.getLhs() == anchor;
  auto maybeConstantIntValue = getConstantIntValue(
      getAsOpFoldResult(anchorIsLhs ? cmpOp.getRhs() : cmpOp.getLhs()));
  if (auto constValue = maybeConstantIntValue) {
    unsigned bitWidth = ConstantIntRanges::getStorageBitwidth(anchor.getType());
    assert(bitWidth > 0 && "expected non-zero bitwdith");
    APInt apVal = {bitWidth, static_cast<uint64_t>(*constValue), isSigned};
    APInt min, max;
    if (isSigned) {
      min = APInt::getSignedMinValue(bitWidth);
      if (llvm::isa_and_nonnull<mlir::triton::GetProgramIdOp,
                                mlir::triton::GetNumProgramsOp>(
              anchor.getDefiningOp())) {
        min = APInt::getZero(bitWidth);
      } else
        min = APInt::getSignedMinValue(bitWidth);
      max = APInt::getSignedMaxValue(bitWidth);
    } else {
      min = APInt::getMinValue(bitWidth);
      max = APInt::getMaxValue(bitWidth);
    }

    switch (cmpOp.getPredicate()) {
    case arith::CmpIPredicate::eq:
      return mlir::ConstantIntRanges::constant(apVal);
    case arith::CmpIPredicate::uge:
    case arith::CmpIPredicate::sge: {
      // K >= apVal implies K ∈ [apVal, max]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(apVal, max, isSigned);
      // apVal >= K implies K ∈ [min, apVal]
      return mlir::ConstantIntRanges::range(min, apVal, isSigned);
    }
    case arith::CmpIPredicate::ugt:
    case arith::CmpIPredicate::sgt: {
      // K > apVal implies K >= apVal + 1 implies K ∈ [apVal + 1, max]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(apVal + 1, max, isSigned);
      // apVal > K implies apVal - 1 >= K implies K ∈ [min, apVal - 1]
      return mlir::ConstantIntRanges::range(min, apVal - 1, isSigned);
    }
    case arith::CmpIPredicate::ule:
    case arith::CmpIPredicate::sle: {
      // K <= apVal implies K ∈ [min, apVal]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(min, apVal, isSigned);
      // apVal <= K implies K ∈ [apVal, max]
      return mlir::ConstantIntRanges::range(apVal, max, isSigned);
    }
    case arith::CmpIPredicate::ult:
    case arith::CmpIPredicate::slt: {
      // K < apVal implies K <= apVal -1 implies K ∈ [min, apVal - 1]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(min, apVal - 1, isSigned);
      // apVal < K implies apVal + 1 <= K implies K ∈ [apVal + 1, max]
      return mlir::ConstantIntRanges::range(apVal + 1, max, isSigned);
    }
    default:
      emitRemark(cmpOp.getLoc(), "unsupported cmp predicate for assumption");
      return {};
    }
  }
  return {};
}
