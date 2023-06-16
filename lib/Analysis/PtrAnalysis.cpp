//===----------------------------------------------------------------------===//
//
// Copyright (c) Triton Project Contributors.
//
//===----------------------------------------------------------------------===//

#include "triton/Analysis/PtrAnalysis.h"
#include "triton/Analysis/OpFoldResultUtils.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include <set>

#define DEBUG_TYPE "triton-ptr-analysis"

namespace mlir {

namespace triton {

int64_t PtrState::getRank() const {
  assert(offsets.size() == sizes.size() && offsets.size() == strides.size());
  return offsets.size();
}

bool PtrState::isEmpty() const {
  return (getRank() == 0 && !source && !scalar);
}

void PtrState::addState(const PtrState &lhsState, const PtrState &rhsState,
                        Location loc, ConversionPatternRewriter &rewriter) {
  assert(isEmpty() && lhsState.getRank() == rhsState.getRank());

  // at most one of lhs and rhs should have valid source, since otherwise we
  // will be losing information
  assert(!(lhsState.source && rhsState.source));
  source = lhsState.source ? lhsState.source : rhsState.source;

  if (lhsState.scalar && rhsState.scalar) {
    auto addOp =
        rewriter.create<arith::AddIOp>(loc, lhsState.scalar, rhsState.scalar);
    scalar = addOp.getResult();
  } else if (lhsState.getRank() == 0) { // both lhs and rhs are scalars
    scalar = lhsState.scalar ? lhsState.scalar : rhsState.scalar;
  }

  for (uint64_t i = 0; i < lhsState.sizes.size(); i++) {
    auto newOffset =
        addOFRs(lhsState.offsets[i], rhsState.offsets[i], loc, rewriter);
    offsets.push_back(newOffset);

    auto newStride =
        addOFRs(lhsState.strides[i], rhsState.strides[i], loc, rewriter);
    strides.push_back(newStride);

    sizes.push_back(lhsState.sizes[i]);
  }
}

void PtrState::mulState(const PtrState &lhsState, const PtrState &rhsState,
                        const Location loc,
                        ConversionPatternRewriter &rewriter) {
  bool rhsScalar = true;
  assert(isEmpty() && lhsState.getRank() == rhsState.getRank());

  // neither lhs nor rhs should have source, since multiplying base pointer
  // does not make sense
  assert(!(lhsState.source && rhsState.source));

  source = lhsState.source ? lhsState.source : rhsState.source;

  assert((lhsState.scalar || rhsState.scalar) &&
         !(lhsState.scalar && rhsState.scalar) &&
         "currently does not support both tensors are effectively non-scalar");

  if (!rhsState.scalar && lhsState.scalar)
    rhsScalar = false;

  for (uint64_t i = 0; i < lhsState.sizes.size(); i++) {
    OpFoldResult newOffset;
    OpFoldResult newStride;
    if (rhsScalar) {
      newOffset =
          mulOFRValue(lhsState.offsets[i], rhsState.scalar, loc, rewriter);
      newStride =
          mulOFRValue(lhsState.strides[i], rhsState.scalar, loc, rewriter);
    } else {
      newOffset =
          mulOFRValue(rhsState.offsets[i], lhsState.scalar, loc, rewriter);
      newStride =
          mulOFRValue(rhsState.strides[i], lhsState.scalar, loc, rewriter);
    }
    offsets.push_back(newOffset);
    strides.push_back(newStride);
    sizes.push_back(lhsState.sizes[i]);
  }
}

memref::ReinterpretCastOp
PtrState::createCastOp(ArrayRef<int64_t> resultShape, const Location loc,
                       ConversionPatternRewriter &rewriter) {
  // Accumulate final offset
  OpFoldResult targetOffset = rewriter.getIndexAttr(0);
  for (auto o : offsets)
    targetOffset = addOFRs(targetOffset, o, loc, rewriter);

  // Create result MemRefType
  SmallVector<int64_t> staticOffset;
  SmallVector<Value> dynamicOffset;
  SmallVector<int64_t> staticStrides;
  SmallVector<Value> dynamicStrides;
  dispatchIndexOpFoldResult(targetOffset, dynamicOffset, staticOffset);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);

  auto elementType = source.getType().cast<BaseMemRefType>().getElementType();
  auto layout = StridedLayoutAttr::get(source.getContext(), staticOffset[0],
                                       staticStrides);
  auto resultType = MemRefType::get(resultShape, elementType, layout);

  // Create reinterpret cast
  return rewriter.create<memref::ReinterpretCastOp>(
      loc, resultType, source, targetOffset, sizes, strides);
}

void PtrAnalysis::visitOperandAdd(
    arith::AddIOp addOp, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  PtrState lhsState;
  visitOperand(addOp.getLhs(), lhsState, loc, rewriter, knownPtrs);

  PtrState rhsState;
  visitOperand(addOp.getRhs(), rhsState, loc, rewriter, knownPtrs);

  state.addState(lhsState, rhsState, loc, rewriter);
}

void PtrAnalysis::visitOperandMul(
    arith::MulIOp mulOp, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  PtrState lhsState;
  visitOperand(mulOp.getLhs(), lhsState, loc, rewriter, knownPtrs);

  PtrState rhsState;
  visitOperand(mulOp.getRhs(), rhsState, loc, rewriter, knownPtrs);

  state.mulState(lhsState, rhsState, loc, rewriter);
}

void PtrAnalysis::visitOperandMakeRange(
    triton::MakeRangeOp rangeOp, PtrState &state, Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  auto shape = rangeOp.getType().cast<ShapedType>().getShape();

  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];

  state.offsets.push_back(rewriter.getIndexAttr(start));
  state.sizes.push_back(rewriter.getIndexAttr(shape[0]));
  state.strides.push_back(rewriter.getIndexAttr(stride));
}

void PtrAnalysis::visitOperandExpandDims(
    triton::ExpandDimsOp expandDimsOp, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  visitOperand(expandDimsOp.getSrc(), state, loc, rewriter, knownPtrs);

  auto dstShape =
      expandDimsOp.getResult().getType().cast<ShapedType>().getShape();
  auto axis = expandDimsOp.getAxis();

  assert(dstShape[axis] == 1 &&
         "expect changed dimension to be 1 in expand_dims");

  // insert dimension info
  state.offsets.insert(state.offsets.begin() + axis, rewriter.getIndexAttr(0));
  state.sizes.insert(state.sizes.begin() + axis, rewriter.getIndexAttr(1));
  state.strides.insert(state.strides.begin() + axis, rewriter.getIndexAttr(0));
}

void PtrAnalysis::visitOperandBroadcast(
    triton::BroadcastOp broadcastOp, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();
  assert(src.getType().isa<ShapedType>() &&
         "input to tt.broadcast should be a tensor");

  auto srcShape = src.getType().cast<ShapedType>().getShape();
  auto dstShape = dst.getType().cast<ShapedType>().getShape();
  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  visitOperand(src, state, loc, rewriter, knownPtrs);

  for (size_t i = 0; i < srcShape.size(); i++) {
    if (srcShape[i] == dstShape[i])
      continue;
    else if (srcShape[i] < dstShape[i])
      state.sizes[i] = rewriter.getIndexAttr(dstShape[i]);
    else
      llvm_unreachable("unexpected dimensions used in broadcast");
  }

  return;
}

void PtrAnalysis::visitOperandSplat(
    triton::SplatOp splatOp, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = dst.getType().cast<ShapedType>().getShape();

  visitOperand(src, state, loc, rewriter, knownPtrs);

  if (src.getType().isa<IntegerType>() ||
      src.getType().isa<triton::PointerType>()) {
    for (auto s : dstShape) {
      state.offsets.push_back(rewriter.getIndexAttr(0));
      state.sizes.push_back(rewriter.getIndexAttr(s));
      state.strides.push_back(rewriter.getIndexAttr(0));
    }
  } else {
    // src is a memref that represent a scalar pointer; it should have one
    // dimension of size 1. This happens inside a for loop that originally has
    // an init arg that is a tensor of pointers; this arg would have been
    // replaced by rewriteForOp.
    auto srcType = src.getType().cast<MemRefType>();
    assert(srcType.getRank() == 1 && state.getRank() == 1 &&
           "splat MemRef source should have rank 1");
    assert(srcType.getShape()[0] == 1 &&
           getIntAttr(state.sizes[0]).value() == 1 &&
           "splat MemRef source should have size 1");

    // Stride[0] will have value of 1 set in visitOperandAddPtr. This value will
    // be represented by a constOp. Clear this value.
    state.strides[0] = rewriter.getIndexAttr(0);

    for (auto [i, s] : llvm::enumerate(dstShape)) {
      if (i == 0) {
        state.sizes[i] = rewriter.getIndexAttr(s);
        continue;
      }
      state.offsets.push_back(rewriter.getIndexAttr(0));
      state.sizes.push_back(rewriter.getIndexAttr(s));
      state.strides.push_back(rewriter.getIndexAttr(0));
    }
  }

  // If we splat a integer value, scalar should become the offset of the outer
  // most dimension
  if (state.scalar)
    state.offsets[0] = state.scalar;

  return;
}

void PtrAnalysis::visitOperandAddptr(
    triton::AddPtrOp addptrOp, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  PtrState ptrState;
  visitOperand(addptrOp.getPtr(), ptrState, addptrOp.getLoc(), rewriter,
               knownPtrs);

  PtrState offsetState;
  visitOperand(addptrOp.getOffset(), offsetState, addptrOp.getLoc(), rewriter,
               knownPtrs);

  assert(ptrState.source && "ptr field should provide source / base pointer");

  // Handle the special case when we are in a for loop, ptr is originally a
  // scalar pointer but replaced with a memref. In this case, ptrState will have
  // rank 1 and offsetState will have rank 0.
  // TODO:
  //  Passing a block argument pointer directly into a for loop not supported
  if (ptrState.getRank() == 1 && offsetState.getRank() == 0) {
    offsetState.sizes.push_back(rewriter.getIndexAttr(1));
    offsetState.offsets.push_back(offsetState.scalar);
    offsetState.strides.push_back(rewriter.getIndexAttr(0));
  }

  assert(ptrState.getRank() == offsetState.getRank() &&
         "ptr and offset field should have the same rank");

  state.addState(ptrState, offsetState, addptrOp.getLoc(), rewriter);
}

void PtrAnalysis::visitOperandReintCast(
    memref::ReinterpretCastOp reintCastOp, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  state.offsets = reintCastOp.getMixedOffsets();
  state.sizes = reintCastOp.getMixedSizes();
  state.strides = reintCastOp.getMixedStrides();
  state.source = reintCastOp.getSource();

  // getMixedOffsets produces staticOffsets (which is the result of collapsing
  // multiple dimensions). Populate the rest of the dimensions with zeroes.
  assert(state.offsets.size() == 1);
  for (size_t i = 1; i < state.sizes.size(); i++) {
    state.offsets.push_back(rewriter.getIndexAttr(0));
  }
}

void PtrAnalysis::visitOperand(
    Value operand, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {

  if (knownPtrs.find(operand) != knownPtrs.end()) {
    state = knownPtrs.lookup(operand);
    return;
  }

  if (operand.getType().isa<IntegerType>()) {
    auto castOp = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), operand);
    state.scalar = castOp.getResult();
    return;
  }

  if (operand.getType().isa<triton::PointerType>()) {
    auto remappedPtr = rewriter.getRemappedValue(operand);
    assert(remappedPtr);

    // A scalar pointer can either be produced by AddPtrOp or a block argument
    if (auto op = operand.getDefiningOp()) {
      assert(operand.getDefiningOp<triton::AddPtrOp>() &&
             "Assume only addptr can produce a scalar pointer");
      visitOperandAddptr(cast<triton::AddPtrOp>(op), state, loc, rewriter,
                         knownPtrs);
    } else {
      state.source = remappedPtr;
    }
    return;
  }

  if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    visitOperandAdd(op, state, loc, rewriter, knownPtrs);
  } else if (auto op = operand.getDefiningOp<arith::MulIOp>()) {
    visitOperandMul(op, state, loc, rewriter, knownPtrs);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    visitOperandMakeRange(op, state, loc, rewriter, knownPtrs);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    visitOperandBroadcast(op, state, loc, rewriter, knownPtrs);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    visitOperandSplat(op, state, loc, rewriter, knownPtrs);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    visitOperandExpandDims(op, state, loc, rewriter, knownPtrs);
  } else if (auto op = operand.getDefiningOp<triton::AddPtrOp>()) {
    visitOperandAddptr(op, state, loc, rewriter, knownPtrs);
  } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    visitOperandConstSplat(op, state, loc, rewriter, knownPtrs);
  } else {
    operand.getDefiningOp()->dump();
    llvm_unreachable("encountered addptr operand produced by an "
                     "unsupported operation");
  }
}

void PtrAnalysis::visitOperandConstSplat(
    arith::ConstantOp op, PtrState &state, const Location loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());
  // this condition is to handle cases where tt.broadcast and tt.splat are
  // folded
  auto attr = cast<DenseElementsAttr>(op.getValue());
  auto elementType = attr.getElementType();
  assert(attr.isSplat() && elementType.isa<IntegerType>());
  auto values = attr.getValues<IntegerAttr>();
  auto value = values[0].getValue();
  auto constAttr = rewriter.getIndexAttr(value.getSExtValue());
  auto constOp = rewriter.create<arith::ConstantOp>(loc, constAttr,
                                                    rewriter.getIndexType());

  state.scalar = constOp;

  auto resultType = cast<ShapedType>(op.getResult().getType());
  for (auto i = 0; i < resultType.getShape().size(); i++) {
    if (i == 0) {
      state.offsets.push_back(constOp.getResult());
    } else {
      state.offsets.push_back(rewriter.getIndexAttr(0));
    }

    state.sizes.push_back(rewriter.getIndexAttr(resultType.getShape()[i]));
    state.strides.push_back(rewriter.getIndexAttr(0));
  }
}

void PtrAnalysis::rewriteAddptrOp(
    triton::AddPtrOp op, ConversionPatternRewriter &rewriter,
    llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  // any inserted instruction should be before this addptr
  auto origIp = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(op);

  PtrState state;
  visitOperandAddptr(op, state, op.getLoc(), rewriter, knownPtrs);

  // If the result is a scalar pointer, visitOperandAddptr will not populate
  // sizes, strides, and offsets. We need to do it here.
  if (state.sizes.size() == 0) {
    state.sizes.push_back(rewriter.getIndexAttr(1));
    state.strides.push_back(rewriter.getIndexAttr(0));
    state.offsets.push_back(state.scalar);
  }

  SmallVector<int64_t> scalarShape(1, 1);
  ArrayRef<int64_t> resultShape;
  if (auto shapedType = op.getResult().getType().dyn_cast<ShapedType>()) {
    resultShape = shapedType.getShape();
  } else {
    // scalar pointer, should produce a one dimensional memref
    resultShape = scalarShape;
    assert(state.getRank() == 1);
  }

  // If there are dimensions with size 1 and stride 0, replace stride 0 with 1
  // so inferResultType below works as expected.
  for (size_t i = 0; i < state.sizes.size(); i++) {
    auto strideIntAttr = getIntAttr(state.strides[i]);
    auto sizeIntAttr = getIntAttr(state.sizes[i]);

    if (!strideIntAttr || strideIntAttr != 0)
      continue;

    if (sizeIntAttr && sizeIntAttr.value() == 1)
      state.strides[i] = rewriter.getIndexAttr(1);
  }

  auto castOp = state.createCastOp(resultShape, op.getLoc(), rewriter);
  LLVM_DEBUG({
    llvm::dbgs() << "cast MemRefType:\n";
    castOp.getOperation()->print(llvm::dbgs(),
                                 OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });

  state.source = castOp.getResult();
  rewriter.replaceOp(op, castOp.getResult());

  knownPtrs[op.getResult()] = state;

  rewriter.restoreInsertionPoint(origIp);
}

void PtrAnalysis::rewriteYieldOp(
    scf::YieldOp op, ConversionPatternRewriter &rewriter,
    const IndexMapSet &levelToBlockArgIndex, const int level,
    const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  // any inserted instruction should be before this yield
  OpBuilder::InsertionGuard insertionGuard{rewriter};
  rewriter.setInsertionPoint(op);

  auto adaptor = scf::YieldOp::Adaptor(op);

  SmallVector<PtrState> initArgState;
  SmallVector<Value> operands(adaptor.getOperands());

  // For each of the init arg that we added additional Values in for loop, we
  // need to add corresponding Values as yield operands. The loop below gathers
  // PtrState for those values.
  for (auto [i, v] : llvm::enumerate(adaptor.getOperands())) {
    if (auto mappedV = rewriter.getRemappedValue(v)) {
      // If this value is a tensor of pointers produced by AddPtrOp,
      // TritonTypeConverter should have converted to MemRefType without layout
      // information. Since it doesn't match with the MemRefType that we
      // produced in rewriteAddptrOp (which is in canonical form with layout
      // information), an unrealized_conversion_cast should have been added. We
      // need to trace it back through this unrealized_conversion_cast to get
      // the original reinterpret_cast. Also see comments in
      // TritonTypeConverter::addConversion.
      //
      // For TritonToLinalg, we do not use any TypeConverters, hence we can
      // access the reinterpret_cast directly.
      if (v.getDefiningOp<triton::AddPtrOp>()) {
        if (auto castOp = mappedV.getDefiningOp<UnrealizedConversionCastOp>()) {
          auto castInputs = castOp.getInputs();
          assert(castInputs.size() == 1 &&
                 "only expect 1:1 mapping for unrealized_conversion_cast that "
                 "were "
                 "automatically inserted during legalizing");
          v = castInputs[0];
        } else if (auto castOp =
                       mappedV.getDefiningOp<memref::ReinterpretCastOp>()) {
          v = castOp;
        } else {
          llvm_unreachable("mapped value defined by an unexpected op");
        }
      } else {
        // If this value is not a tensor of pointers, we will use the mapped
        // value, and rely on the conversion will happen later automatically
        // when we legalize loop body.

        // TODO:
        //  The scenario where a value is a tensor of pointers but not produced
        //  by AddPtrOp is not supported
        if (mappedV.getType().isa<TensorType>() &&
            mappedV.getType()
                .dyn_cast<TensorType>()
                .getElementType()
                .isa<triton::PointerType>())
          llvm_unreachable("unsupported scenario where a value is a tensor of "
                           "pointers but not produced by AddPtrOp");
        v = mappedV;
      }
    }

    if (levelToBlockArgIndex.find(level) == levelToBlockArgIndex.end())
      continue;
    auto thisSet = levelToBlockArgIndex.find(level)->second;
    if (thisSet.find(i) == thisSet.end())
      continue;

    auto reintCastOp = v.getDefiningOp<memref::ReinterpretCastOp>();
    assert(
        reintCastOp ||
        (v.getType().isa<TensorType>() &&
         v.getType().dyn_cast<TensorType>().getElementType().isa<IndexType>()));

    PtrState state;
    if (reintCastOp) {
      visitOperandReintCast(reintCastOp, state, op.getLoc(), rewriter,
                            knownPtrs);
    } else {
      visitOperand(v, state, op.getLoc(), rewriter, knownPtrs);
    }
    initArgState.push_back(state);
  }

  // For each of the PtrState recorded in the last step, extract value
  // that correspond to offset and stride for each dimension and append
  // them to yield operands.
  for (auto state : initArgState) {
    for (auto s : state.offsets) {
      // offsets can be IntAttr zeroes, since reinterpret_cast collapses them
      // for the input memref, and the for loop may not update offsets other
      // than offsets[0]. Create constants Values for those zeroes.
      if (auto sIntAttr = getIntAttr(s)) {
        assert(sIntAttr.value() == 0 && "attribute offsets should be zeroes");
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(0));
        operands.push_back(constOp.getResult());
      } else {
        operands.push_back(s.get<Value>());
      }
    }

    for (auto s : state.strides) {
      assert(!getIntAttr(s) &&
             "PtrState strides for yield within for loop not expected to be "
             "attribute.");
      operands.push_back(s.get<Value>());
    }
  }

  // Yield is a terminator op that must be at the end of the function
  rewriter.setInsertionPointAfter(op);
  auto newOp = rewriter.replaceOpWithNewOp<scf::YieldOp>(op, operands);
  assert(op->getNumResults() == 0);

  LLVM_DEBUG({
    llvm::dbgs() << "new yield:";
    newOp.getOperation()->print(llvm::dbgs(),
                                OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });
}

void PtrAnalysis::rewriteForOp(
    scf::ForOp op, ConversionPatternRewriter &rewriter,
    IndexMapSet &levelToBlockArgIndex, const int level,
    llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  SmallVector<Value> newInitArgs;

  SmallVector<std::pair<int, PtrState>> initArgIndexState;
  SmallVector<std::pair<int, PtrState>> knownPtrsTmp;

  // Create a new list of init args
  for (auto [i, arg] : llvm::enumerate(op.getInitArgs())) {
    auto mappedV = rewriter.getRemappedValue(arg);

    // Trace back the original value. See comments in rewriteYieldOp.
    // This block is unreachable for TritonToLinalg because we don't use
    // TypeConverters.
    if (mappedV && mappedV.getDefiningOp<UnrealizedConversionCastOp>()) {
      auto castOp = mappedV.getDefiningOp<UnrealizedConversionCastOp>();
      assert(castOp && "expected unrealized_conversion_cast");
      auto castInputs = castOp.getInputs();
      assert(castInputs.size() == 1 &&
             "only expect 1:1 mapping for unrealized_conversion_cast that were "
             "automatically inserted during legalizing");
      mappedV = castInputs[0];
    }

    memref::ReinterpretCastOp reintCastOp;

    // If this init arg is supposed to be remapped, use the remapped value
    // instead. In addition, if this init arg is a memref created by a
    // reinterpret_cast or a tensor of index, there is a chance that it will be
    // used in addptr. Create PtrState for each such init arg.
    if (mappedV) {
      // TODO:
      //  Passing a block argument pointer directly into a for loop not
      assert(!(mappedV.dyn_cast<BlockArgument>() &&
               mappedV.getType().isa<UnrankedMemRefType>()) &&
             "cannot take pointer block argument as init arg for for loop");
      reintCastOp = mappedV.getDefiningOp<memref::ReinterpretCastOp>();
      newInitArgs.push_back(mappedV);
    } else {
      newInitArgs.push_back(arg);
    }

    auto indexTensor =
        arg.getType().isa<TensorType>() &&
        arg.getType().dyn_cast<TensorType>().getElementType().isa<IndexType>();

    if (!reintCastOp && !indexTensor)
      continue;

    PtrState state;
    if (reintCastOp) {
      visitOperandReintCast(reintCastOp, state, op.getLoc(), rewriter,
                            llvm::SmallDenseMap<Value, PtrState>(0));
    } else {
      // TODO:
      visitOperand(arg, state, op.getLoc(), rewriter,
                   llvm::SmallDenseMap<Value, PtrState>(0));
    }

    // Record the PtrState for later processing
    initArgIndexState.push_back(std::make_pair(i, state));
  }

  // Set insertion point to be before the for loop for new variables passed
  // into the new loop.
  auto origIp = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(op);

  // For each of the PtrState recorded in the last step, insert new
  // instructions to describe offset and stride for each dimension and append
  // them to init args
  for (auto [i, state] : initArgIndexState) {
    // For each dimension, if the corresponding offset and stride is an
    // integer attribute, create a constant value and append them at the end
    // of init arg list.
    for (auto [j, s] : llvm::enumerate(state.offsets)) {
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(sIntAttr.value()));
        newInitArgs.push_back(constOp.getResult());
        state.offsets[j] = constOp.getResult();
      } else {
        newInitArgs.push_back(s.get<Value>());
      }
    }

    for (auto [j, s] : llvm::enumerate(state.strides)) {
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(sIntAttr.value()));
        newInitArgs.push_back(constOp.getResult());
        state.strides[j] = constOp.getResult();
      } else {
        newInitArgs.push_back(s.get<Value>());
      }
    }

    // Note that we want the knownPtrs to be indexed by block arg, but we only
    // have index for now. Also, the state we record is the init arg, but want
    // to to use newly created block arg. These block args are not created yet.
    // We will translate this mapping later.
    knownPtrsTmp.push_back(std::make_pair(i, state));
    levelToBlockArgIndex[level].insert(i);

    // If the original init arg is a memref produced by reinterpret_cast, create
    // a new memref using new strides and offsets created above. This produces a
    // canonicalized memref, which will match what the for loop generates if it
    // modifies the memref. E.g., original reinterpret_cast can produce a memref
    // with const stride:
    //  - memref<4x256xbf16, affine_map<(d0, d1)[s0, s1] -> (d0 * 256 + s0 + d1
    //  * s1)>>
    // The new reinterpret_cast will always have dynamic stride and offset:
    //  - memref<4x256xbf16, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 +
    //  d1 * s2)>>
    if (auto reintCastOp =
            newInitArgs[i].getDefiningOp<memref::ReinterpretCastOp>()) {
      SmallVector<int64_t> resultShape;
      for (auto s : state.sizes) {
        auto sIntAttr = getIntAttr(s);
        assert(sIntAttr && "expected constant size");
        resultShape.push_back(sIntAttr.value());
      }
      auto castOp = state.createCastOp(resultShape, op.getLoc(), rewriter);

      LLVM_DEBUG({
        llvm::dbgs() << "new reinterpret_cast with dynamic sizes "
                        "and offsets:";
        castOp->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
        llvm::dbgs() << "\n";
      });

      newInitArgs[i] = castOp.getResult();
    }
  }

  rewriter.restoreInsertionPoint(origIp);

  // create a new scf::ForOp that uses updated init args and same loop body
  auto newOp = rewriter.create<scf::ForOp>(
      op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(),
      newInitArgs, [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        IRMapping mapping;
        mapping.map(op.getInductionVar(), iv);
        mapping.map(op.getInitArgs(), newInitArgs);
        mapping.map(op.getRegionIterArgs(), args);
        for (auto &bodyOp : op.getLoopBody().getOps()) {
          b.clone(bodyOp, mapping);
        }
      });

  // Convert the book-keeping data structure to use the correct key and value.
  // Key is converted from init arg index to newly created block arg, and
  // Value's PtrState fields are converted from init arg to newly created block
  // arg
  int cnt = op.getRegionIterArgs().size();
  for (auto [i, state] : knownPtrsTmp) {
    for (auto it = state.offsets.begin(); it != state.offsets.end(); it++) {
      *it = newOp.getRegionIterArgs()[cnt];
      cnt++;
    }

    for (auto it = state.strides.begin(); it != state.strides.end(); it++) {
      *it = newOp.getRegionIterArgs()[cnt];
      cnt++;
    }

    auto key = newOp.getRegionIterArgs()[i];
    knownPtrs.insert(std::make_pair(key, state));
  }
  assert(static_cast<size_t>(cnt) == newOp.getRegionIterArgs().size() &&
         "expect to remap all new block args");

  // replace only the results that correspond to the original scf.for
  auto resultsToReplaceWith = ResultRange(
      newOp.result_begin(), newOp.result_begin() + op.getNumResults());
  rewriter.replaceOp(op, resultsToReplaceWith);

  // Update the loop body. Manually invoke the rewrite logic on addptr and yield
  // in the loop body, so we can take advantage of the states we built up
  for (auto &bodyOp : newOp.getLoopBody().getOps()) {
    if (auto addptrOp = dyn_cast<triton::AddPtrOp>(bodyOp)) {
      rewriteAddptrOp(addptrOp, rewriter, knownPtrs);
    } else if (auto forOp = dyn_cast<scf::ForOp>(bodyOp)) {
      // TODO:
      //  Nested for loops are not supported at the moment
      assert(0 && "nested loops currently not supported");
      // rewriteForOp(forOp, rewriter, levelToBlockArgIndex, level+1,
      // knownPtrs); levelToBlockArgIndex.erase(level+1);
    }
  }

  if (op.getNumRegionIterArgs()) {
    auto yieldOp = cast<scf::YieldOp>(newOp.getBody()->getTerminator());
    rewriteYieldOp(yieldOp, rewriter, levelToBlockArgIndex, level, knownPtrs);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "new for\n";
    newOp.getOperation()->print(llvm::dbgs(),
                                OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });
}

Value PtrAnalysis::getScalarMemRef(Value ptr, Value memRef, const Location loc,
                                   ConversionPatternRewriter &rewriter) {
  assert(ptr.getType().cast<triton::PointerType>() &&
         "expected scalar pointer");

  // If pointer is generated by tt.addptr, TypeConverter will have inserted an
  // unrealized conversion cast for ptr to cast its type from tt.ptr to unranked
  // memref. Input of this cast is the actual source memref.
  //
  // For TritonToLinalg, we can access the reinterpret_cast directly due to no
  // usages of TypeConverters.
  if (ptr.getDefiningOp<triton::AddPtrOp>()) {
    if (auto uCast = memRef.getDefiningOp<UnrealizedConversionCastOp>()) {
      assert(uCast && "expected unrealized conversion inserted by type "
                      "converter not found");
      return uCast.getInputs()[0];
    } else if (auto castOp =
                   memRef.getDefiningOp<memref::ReinterpretCastOp>()) {
      return castOp.getResult();
    } else {
      llvm_unreachable("pointer value is defined by an unexpected op");
    }
  }

  assert(isa<BlockArgument>(ptr) &&
         "pointer is neither produced by addptr nor a block argument");
  PtrState state;
  state.source = memRef;
  state.offsets.push_back(rewriter.getIndexAttr(0));
  state.sizes.push_back(rewriter.getIndexAttr(1));
  state.strides.push_back(rewriter.getIndexAttr(1));
  auto castOp = state.createCastOp(SmallVector<int64_t>(1, 1), loc, rewriter);
  return castOp.getResult();
}

} // namespace triton
} // namespace mlir
