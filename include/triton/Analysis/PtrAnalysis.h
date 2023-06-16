//===----------------------------------------------------------------------===//
//
// Copyright (c) Triton Project Contributors.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ANALYSIS_PTRANALYSIS_H
#define TRITON_ANALYSIS_PTRANALYSIS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include <set>

namespace mlir {

class ConversionPatternRewriter;

namespace triton {

// Data structure used to decode pointer arithmetics and potentially to be
// translate it into memref. offsets, sizes, and strides are in unit of elements
// in a linearly laid-out memory, which is the same as pointer arithmetic
// operations in Triton language. scalar is a shortcut used when the entire
// state describes a single scalar value. source is the base pointer.
struct PtrState {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  Value source;
  Value scalar;

  int64_t getRank() const;

  bool isEmpty() const;

  // Process addition of two PtrStates.
  void addState(const PtrState &lhsState, const PtrState &rhsState,
                Location loc, ConversionPatternRewriter &rewriter);

  // Process multiplication of two PtrStates
  void mulState(const PtrState &lhsState, const PtrState &rhsState,
                const Location loc, ConversionPatternRewriter &rewriter);

  // Produce a reinterpret cast based on the current PtrState. Additional
  // instructions may be inserted in calculating the final offset.
  memref::ReinterpretCastOp createCastOp(ArrayRef<int64_t> resultShape,
                                         const Location loc,
                                         ConversionPatternRewriter &rewriter);
};

class PtrAnalysis {
public:
  using IndexMapSet = std::map<int, std::set<int>>;

  // Recursively parse a Value; call the corresponding
  // function based on the defining operation and argument type.
  static void
  visitOperand(Value operand, PtrState &state, const Location loc,
               ConversionPatternRewriter &rewriter,
               const llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of arith.addi. Process both arguments and insert any
  // arith.addi instruction as needed.
  // Main assumptions:
  //  Only one of lhsState and rhsState has source field set
  //  Current PtrState should be empty
  // Expected result:
  //  source = lhsState.source ? lhsState.source : rhsState.source
  //  sizes[i] = lhsState.sizes[i] (which should match rhsState.sizes[i])
  //  offsets[i] = lhsState.offsets[i] + rhsState.offsets[i]
  //  strides[i] = lhsState.strides[i] + rhsState.strides[i]
  static void
  visitOperandAdd(arith::AddIOp addOp, PtrState &state, const Location loc,
                  ConversionPatternRewriter &rewriter,
                  const llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of arith.muli. Process both arguments and insert any
  // arith.muli instruction as needed.
  // Main assumptions:
  //  Neither lhsState nor rhsState has source field set
  //  Current PtrState should be empty
  //  Currently only support one of the operand is a scalar index
  // Expected result (scalar and tensorState represent the two operands):
  //  source = null
  //  sizes[i] = tensorState.sizes[i]
  //  offsets[i] = tensorState.offsets[i] * scalar
  //  strides[i] = tensorState.strides[i] * scalar
  static void
  visitOperandMul(arith::MulIOp mulOp, PtrState &state, const Location loc,
                  ConversionPatternRewriter &rewriter,
                  const llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of make_range.
  // Main assumptions:
  //  start, end, and shape are all statically known
  //  The output of make_range is 1-dimensional
  //  Does not check validity of inputs (e.g., stride > 0)
  // Expected result:
  //  source = null
  //  sizes[0] = shape[0]
  //  offset[0] = start
  //  strides[0] = ceiling( (end - start) / shape[0] )
  static void
  visitOperandMakeRange(triton::MakeRangeOp rangeOp, PtrState &state,
                        Location loc, ConversionPatternRewriter &rewriter,
                        const llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of expand_dims
  // Main assumptions:
  //  Only 1 dimension changes for each invocation of reshape
  //  The changed dimension must have size of 1
  // Expected result:
  //  Insert a dimension of size 1, stride 0, and offset 0
  static void
  visitOperandExpandDims(triton::ExpandDimsOp expandDimsOp, PtrState &state,
                         const Location loc,
                         ConversionPatternRewriter &rewriter,
                         const llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of broadcast
  // Main assumptions:
  //  Rank of soure and result is the same
  // Expected result:
  //  Update sizes[i] only, no changes to other fields
  static void
  visitOperandBroadcast(triton::BroadcastOp broadcastOp, PtrState &state,
                        const Location loc, ConversionPatternRewriter &rewriter,
                        const llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of splat
  // Main assumptions:
  //  Source is a scalar value (i.e., an integer or a pointer, not a tensor)
  // Expected result:
  //  sizes[i] reflect the shape of the result, strides[i] = 0,  offsets[i] = 0
  //  if source is an integer, offset[0] = scalar = source
  static void
  visitOperandSplat(triton::SplatOp splatOp, PtrState &state,
                    const Location loc, ConversionPatternRewriter &rewriter,
                    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of arith.constant that is a splat
  // Main assumptions:
  //  Source is a constant op that produces a constant dense tensor where all
  //  elements are the same (i.e.: a constant that is splatted)
  // Expected result:
  //  sizes[i] reflect the shape of the result, strides[i] = 0,  offsets[i] =
  //  splat value if i == 0, otherwise 0
  static void
  visitOperandConstSplat(arith::ConstantOp op, PtrState &state,
                         const Location loc,
                         ConversionPatternRewriter &rewriter,
                         const llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of addptr.
  // Main assumptions:
  //  The ptr field should populate the source field
  //  ptr and offset fields should result in same rank
  // Expected result:
  //  The resulting state for ptr and offset wil be added
  static void
  visitOperandAddptr(triton::AddPtrOp addptrOp, PtrState &state,
                     const Location loc, ConversionPatternRewriter &rewriter,
                     const llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of reinterpret_cast.
  // Main assumptions:
  //  None
  // Expected result:
  //  Directly grab all corresponding fields from reinterpret_cast.
  static void
  visitOperandReintCast(memref::ReinterpretCastOp reintCastOp, PtrState &state,
                        const Location loc, ConversionPatternRewriter &rewriter,
                        const llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Parse the state of AddPtrOp, insert any instruction needed to
  // calculate strides and offsets, build PtrState for this operand, and record
  // PtrState for knownPtrs.
  static void rewriteAddptrOp(triton::AddPtrOp op,
                              ConversionPatternRewriter &rewriter,
                              llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Parse the state of YieldOp, insert any instruction needed to calculate
  // strides and offsets, build PtrState for this operand, and record PtrState
  // in knownPtrs.
  static void
  rewriteYieldOp(scf::YieldOp op, ConversionPatternRewriter &rewriter,
                 const IndexMapSet &levelToBlockArgIndex, const int level,
                 const llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  static void rewriteForOp(scf::ForOp op, ConversionPatternRewriter &rewriter,
                           IndexMapSet &levelToBlockArgIndex, const int level,
                           llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  static Value getScalarMemRef(Value ptr, Value memRef, const Location loc,
                               ConversionPatternRewriter &rewriter);
};

} // namespace triton

} // namespace mlir

#endif
