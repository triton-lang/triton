//===----------------------------------------------------------------------===//
//
// Copyright (c) Triton Project Contributors.
//
//===----------------------------------------------------------------------===//

#include "triton/Analysis/OpFoldResultUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

// Return integer if ofr is an IntegerAttr. Note that this function differs
// from getConstantIntValue, which returns an integer if ofr is the constant
// result of an operation too.
std::optional<int64_t> getIntAttr(const OpFoldResult ofr) {
  if (ofr.is<Attribute>() && ofr.get<Attribute>().isa<IntegerAttr>())
    return ofr.get<Attribute>().dyn_cast<IntegerAttr>().getInt();

  return std::nullopt;
}

// Process addition of two OFRs. If both OFRs are Integer Attributes, result
// is an Integer Attribute. Otherwise, insert the arith.addi instruction if
// needed and use its result Value.
OpFoldResult addOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, ConversionPatternRewriter &rewriter) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // shortcut for special cases
  if (!lhsIntAttr && rhsIntAttr && rhsIntAttr.value() == 0)
    return lhs;
  if (!rhsIntAttr && lhsIntAttr && lhsIntAttr.value() == 0)
    return rhs;

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return rewriter.getIndexAttr(lhsIntAttr.value() + rhsIntAttr.value());

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = lhs.dyn_cast<Value>();
  if (lhsIntAttr) {
    auto lhsOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  } else {
    assert(lhsValue.getType().isa<IndexType>());
  }

  auto rhsValue = rhs.dyn_cast<Value>();
  if (rhsIntAttr) {
    auto rhsOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  } else {
    assert(lhsValue.getType().isa<IndexType>());
  }

  return rewriter.create<arith::AddIOp>(loc, lhsValue, rhsValue).getResult();
}

// Produce result = lhs - rhs. If both OFRs are Integer Attributes, result
// is an Integer Attribute. Otherwise, insert the arith.addi instruction if
// needed and use its result Value.
OpFoldResult subOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, ConversionPatternRewriter &rewriter) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // shortcut for special cases
  if (!lhsIntAttr && rhsIntAttr && rhsIntAttr.value() == 0)
    return lhs;

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return rewriter.getIndexAttr(lhsIntAttr.value() - rhsIntAttr.value());

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = lhs.dyn_cast<Value>();
  if (lhsIntAttr) {
    auto lhsOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = rhs.dyn_cast<Value>();
  if (rhsIntAttr) {
    auto rhsOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto sumOp = rewriter.create<arith::SubIOp>(loc, lhsValue, rhsValue);
  return sumOp.getResult();
}

// Process multiplication of two OFRs. If both OFRs are Integer Attributes,
// result is an Integer Attribtue. Otherwise, insert the arith.muli
// instruction if needed and use its result Value.
OpFoldResult mulOFRValue(const OpFoldResult lhs, const Value rhs,
                         const Location loc,
                         ConversionPatternRewriter &rewriter) {
  auto lhsIntAttr = getIntAttr(lhs);

  auto rhsIsConst = false;
  // if rhs is not a const, use max value since min is used to represent
  // dynamic size or stride
  auto rhsConstValue = std::numeric_limits<int64_t>::max();
  auto rhsOp = rhs.getDefiningOp<arith::ConstantOp>();
  if (rhsOp) {
    rhsIsConst = true;
    rhsConstValue = rhsOp.getValue().cast<IntegerAttr>().getInt();
  }

  // shortcuts for special cases
  if (lhsIntAttr) {
    if (lhsIntAttr.value() == 0)
      return lhs;
    if (lhsIntAttr.value() == 1)
      return rhs;
  }
  if (rhsIsConst) {
    if (rhsConstValue == 0)
      return rhsOp.getResult();
    if (rhsConstValue == 1)
      return lhs;
  }

  // 0. both lhs and rhs are constants
  if (lhsIntAttr && rhsIsConst)
    return rewriter.getIndexAttr(lhsIntAttr.value() * rhsConstValue);

  // 1. if lhs is constant but rhs is not
  if (lhsIntAttr && !rhsIsConst) {
    auto lhsConstOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(lhsIntAttr.value()));
    auto mulOp =
        rewriter.create<arith::MulIOp>(loc, lhsConstOp.getResult(), rhs);
    return mulOp.getResult();
  }

  // 2. if lhs is not constant
  assert(!lhsIntAttr);
  auto mulOp = rewriter.create<arith::MulIOp>(loc, lhs.get<Value>(), rhs);
  return mulOp.getResult();
}

OpFoldResult minOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, ConversionPatternRewriter &rewriter) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return rewriter.getIndexAttr(
        std::min(lhsIntAttr.value(), rhsIntAttr.value()));

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = lhs.dyn_cast<Value>();
  if (lhsIntAttr) {
    auto lhsOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = rhs.dyn_cast<Value>();
  if (rhsIntAttr) {
    auto rhsOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto minOp = rewriter.create<arith::MinSIOp>(loc, lhsValue, rhsValue);
  return minOp.getResult();
}

} // namespace mlir
