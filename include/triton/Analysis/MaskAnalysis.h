//===----------------------------------------------------------------------===//
//
// Copyright (c) Triton Project Contributors.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ANALYSIS_MASKANALYSIS_H
#define TRITON_ANALYSIS_MASKANALYSIS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {

class ConversionPatternRewriter;

namespace triton {
// Data structure used to decode the pattern in a mask used for load and store.
// start and end field represent the start and end index of a range (produced
// by make_range, addi, etc.). While multi-dimensional data is possible, we
// assume range comparison can only be done on 1 dimension at a time (and
// results of range comparions across dimensions can be combined), hence start
// and end are not vectors. dims represents the real access size for ld/st
// (instead of the tensor/memref size specified by the IR). scalar is a shortcut
// used when the entire state contains a single scalar value.
//
// The general lifetime of this data structure is roughly:
// 1. A range is created by make_range and optionally operated on by addi w/
// result of splat, expand_dims, etc. During this phase, either (1) both start
// and end are populated, or (2) scalar is populated. Only one of the dimensions
// (that contains the range) can have dim > 1.
// 2. Result from step 1 is compared with a another MaskState that represents a
// scalar value. The resulting state only has dims populated.
// 3. Optionally, result from step 2 can be broadcasted and anded with other
// results from step 2. The resulting state only has dims populated.
//
// Example of creating 2D mask:
//  mask = (rows[:, None] < M) & (cols[None, :] < N)
struct MaskState {
  OpFoldResult start;
  OpFoldResult end;
  SmallVector<OpFoldResult> dims;
  OpFoldResult scalar;

  int64_t getRank() const { return dims.size(); }

  bool isEmpty() const { return getRank() == 0 && !scalar && !start && !end; }

  bool isMask() const { return !start && !end && !scalar && dims.size() != 0; }

  // Recursively parse a Value; call the coresponding function based on the
  // defining operation and Value type
  LogicalResult parse(Value operand, const Location loc,
                      ConversionPatternRewriter &rewriter);

  tensor::ExtractSliceOp
  getExtractSlice(Value source, const Location loc,
                  ConversionPatternRewriter &rewriter) const;

  memref::SubViewOp getSubview(Value source, const Location loc,
                               ConversionPatternRewriter &rewriter) const;

private:
  // -------
  // Utility functions to operate on MaskState
  // -------
  LogicalResult addStateScalar(const MaskState &state,
                               const OpFoldResult scalar, Location loc,
                               ConversionPatternRewriter &rewriter);

  LogicalResult addStates(const MaskState &lhsState, const MaskState &rhsState,
                          Location loc, ConversionPatternRewriter &rewriter);

  LogicalResult minStates(const MaskState &lhsState, const MaskState &rhsState,
                          Location loc, ConversionPatternRewriter &rewriter);
  // -------
  // Helper functions to parse values to populate MaskState
  // -------

  // Operand is the result of a constant
  // Get the value of the constant and assign it to scalar.
  LogicalResult parseConstant(arith::ConstantOp constOp, const Location loc,
                              ConversionPatternRewriter &rewriter);

  // Operand is an integer scalar
  LogicalResult parseIntScalar(Value scalar, const Location loc,
                               ConversionPatternRewriter &rewriter);

  // Operand is the result of addi
  // One and only one of the operands should be a scalar. Increment both start
  // and end, dims remains unchanged, and scalar is empty.
  LogicalResult parseAdd(arith::AddIOp addOp, const Location loc,
                         ConversionPatternRewriter &rewriter);
  // Operand is the result of andi
  // Each of the result state dims is smaller of the two operands' dims.
  // Insert instruction if needed to get new dims.
  LogicalResult parseAnd(arith::AndIOp andOp, const Location loc,
                         ConversionPatternRewriter &rewriter);

  // Operand is the result of cmpi
  // Assume only of the dimensions have size > 1. Only support slt for now.
  // For that dimension, calculate this new dim as: dim = min(end, value) -
  // start
  LogicalResult parseCmp(arith::CmpIOp cmpOp, const Location loc,
                         ConversionPatternRewriter &rewriter);
  // Operand is the result of make_range
  // Set start and end accordingly; step size must be 1.
  LogicalResult parseMakeRange(triton::MakeRangeOp rangeOp, const Location loc,
                               ConversionPatternRewriter &rewriter);
  // Operand is the result of broadcast
  // Change dims only; assume only applies to tensors.
  LogicalResult parseBroadcast(triton::BroadcastOp broadcastOp,
                               const Location loc,
                               ConversionPatternRewriter &rewriter);
  // Operand is the result of splat
  // Assume only applies to scalar. start and end are left empty; scalar will
  // be assigned, and dims will be updated.
  LogicalResult parseSplat(triton::SplatOp splatOp, const Location loc,
                           ConversionPatternRewriter &rewriter);
  // Operand is the result of expand_dims
  // Insert additional dims; start and end do not change and correspond to the
  // dimension that contains the range.
  LogicalResult parseExpandDims(triton::ExpandDimsOp expandDimsOp,
                                const Location loc,
                                ConversionPatternRewriter &rewriter);
};

} // namespace triton

} // namespace mlir

#endif
