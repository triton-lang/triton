//===----------------------------------------------------------------------===//
//
// Copyright (c) Triton Project Contributors.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ANALYSIS_OPFOLDRESULT_UTILS_H
#define TRITON_ANALYSIS_OPFOLDRESULT_UTILS_H

#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"

#include <optional>

namespace mlir {

class ConversionPatternRewriter;

// Return integer if ofr is an IntegerAttr. Note that this function differs
// from getConstantIntValue, which returns an integer if ofr is the constant
// result of an operation too.
std::optional<int64_t> getIntAttr(const OpFoldResult ofr);

// Process addition of two OFRs. If both OFRs are Integer Attributes, result
// is an Integer Attribute. Otherwise, insert the arith.addi instruction if
// needed and use its result Value.
OpFoldResult addOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, ConversionPatternRewriter &rewriter);

// Produce result = lhs - rhs. If both OFRs are Integer Attributes, result
// is an Integer Attribute. Otherwise, insert the arith.addi instruction if
// needed and use its result Value.
OpFoldResult subOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, ConversionPatternRewriter &rewriter);

// Process multiplication of two OFRs. If both OFRs are Integer Attributes,
// result is an Integer Attribtue. Otherwise, insert the arith.muli
// instruction if needed and use its result Value.
OpFoldResult mulOFRValue(const OpFoldResult lhs, const Value rhs,
                         const Location loc,
                         ConversionPatternRewriter &rewriter);

OpFoldResult minOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, ConversionPatternRewriter &rewriter);

} // namespace mlir

#endif
