//===----------------------------------------------------------------------===//
//
// Copyright (c) Triton Project Contributors.
//
//===----------------------------------------------------------------------===//

#include "triton/Analysis/MaskAnalysis.h"
#include "triton/Analysis/OpFoldResultUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

namespace triton {

LogicalResult MaskState::parse(Value operand, const Location loc,
                               ConversionPatternRewriter &rewriter) {
  if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    return this->parseConstant(op, loc, rewriter);
  } else if (operand.getType().isa<IntegerType>()) {
    return this->parseIntScalar(operand, loc, rewriter);
  } else if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    return this->parseAdd(op, loc, rewriter);
  } else if (auto op = operand.getDefiningOp<arith::AndIOp>()) {
    return this->parseAnd(op, loc, rewriter);
  } else if (auto op = operand.getDefiningOp<arith::CmpIOp>()) {
    return this->parseCmp(op, loc, rewriter);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    return this->parseMakeRange(op, loc, rewriter);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    return this->parseBroadcast(op, loc, rewriter);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    return this->parseSplat(op, loc, rewriter);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    return this->parseExpandDims(op, loc, rewriter);
  } else {
    return failure();
  }
}

tensor::ExtractSliceOp
MaskState::getExtractSlice(Value source, const Location loc,
                           ConversionPatternRewriter &rewriter) const {
  auto sourceType = source.getType().cast<RankedTensorType>();
  SmallVector<OpFoldResult> offsets(getRank(), rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(getRank(), rewriter.getIndexAttr(1));

  auto dstType = tensor::ExtractSliceOp::inferResultType(sourceType, offsets,
                                                         dims, strides);

  return rewriter.create<tensor::ExtractSliceOp>(loc, dstType, source, offsets,
                                                 dims, strides);
}

memref::SubViewOp
MaskState::getSubview(Value source, const Location loc,
                      ConversionPatternRewriter &rewriter) const {
  auto sourceType = source.getType().cast<MemRefType>();
  SmallVector<OpFoldResult> offsets(getRank(), rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(getRank(), rewriter.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(sourceType, offsets, dims, strides);

  return rewriter.create<memref::SubViewOp>(loc, dstType.cast<MemRefType>(),
                                            source, offsets, dims, strides);
}

LogicalResult MaskState::addStateScalar(const MaskState &state,
                                        const OpFoldResult scalar, Location loc,
                                        ConversionPatternRewriter &rewriter) {
  start = addOFRs(state.start, scalar, loc, rewriter);
  end = addOFRs(state.end, scalar, loc, rewriter);
  dims = state.dims;
  return success();
}

LogicalResult MaskState::addStates(const MaskState &lhsState,
                                   const MaskState &rhsState, Location loc,
                                   ConversionPatternRewriter &rewriter) {
  if (lhsState.scalar && rhsState.scalar) {
    InFlightDiagnostic diag =
        emitError(loc) << "Unexpected case where both lhs and rhs are scalars";
    return failure();
  }

  if (!lhsState.scalar && !rhsState.scalar) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "Unsupported scenario where neither lhs nor rhs is a scalar";
    return failure();
  }

  if (lhsState.scalar)
    return addStateScalar(rhsState, lhsState.scalar, loc, rewriter);
  else
    return addStateScalar(lhsState, rhsState.scalar, loc, rewriter);
}

LogicalResult MaskState::minStates(const MaskState &lhsState,
                                   const MaskState &rhsState, Location loc,
                                   ConversionPatternRewriter &rewriter) {
  if (lhsState.getRank() != rhsState.getRank()) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "Unexpected case where lhs and rhs have different ranks";
    return failure();
  }

  for (uint32_t i = 0; i < lhsState.getRank(); i++) {
    auto lhsDim = lhsState.dims[i];
    auto rhsDim = rhsState.dims[i];
    dims.push_back(minOFRs(lhsDim, rhsDim, loc, rewriter));
  }
  return success();
}

LogicalResult MaskState::parseConstant(arith::ConstantOp constOp,
                                       const Location loc,
                                       ConversionPatternRewriter &rewriter) {
  assert(this->isEmpty());

  if (isa<DenseElementsAttr>(constOp.getValue())) {
    auto attr = cast<DenseElementsAttr>(constOp.getValue());
    auto elementType = attr.getElementType();
    assert(attr.isSplat() && elementType.isa<IntegerType>() &&
           "All elements must share a single integer constant value");
    auto values = attr.getValues<IntegerAttr>();
    auto value = values[0].getValue();
    auto constAttr = rewriter.getIndexAttr(value.getSExtValue());
    auto op = rewriter.create<arith::ConstantOp>(loc, constAttr,
                                                 rewriter.getIndexType());
    this->scalar = op.getValue();
  } else {
    auto value = constOp.getValue().cast<IntegerAttr>().getInt();
    this->scalar = rewriter.getIndexAttr(value);
  }

  return success();
}

LogicalResult MaskState::parseIntScalar(Value scalar, const Location loc,
                                        ConversionPatternRewriter &rewriter) {
  assert(this->isEmpty());
  auto castOp =
      rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), scalar);
  this->scalar = castOp.getResult();
  return success();
}

LogicalResult MaskState::parseAdd(arith::AddIOp addOp, const Location loc,
                                  ConversionPatternRewriter &rewriter) {
  assert(this->isEmpty());

  MaskState lhsState;
  if (failed(lhsState.parse(addOp.getLhs(), loc, rewriter)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(addOp.getRhs(), loc, rewriter)))
    return failure();

  return this->addStates(lhsState, rhsState, loc, rewriter);
}

LogicalResult MaskState::parseAnd(arith::AndIOp andOp, const Location loc,
                                  ConversionPatternRewriter &rewriter) {
  assert(this->isEmpty());

  MaskState lhsState;
  if (failed(lhsState.parse(andOp.getLhs(), loc, rewriter)) ||
      !lhsState.isMask())
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(andOp.getRhs(), loc, rewriter)) ||
      !rhsState.isMask())
    return failure();

  return this->minStates(lhsState, rhsState, loc, rewriter);
}

LogicalResult MaskState::parseCmp(arith::CmpIOp cmpOp, const Location loc,
                                  ConversionPatternRewriter &rewriter) {
  assert(this->isEmpty());

  if (cmpOp.getPredicate() != arith::CmpIPredicate::slt) {
    InFlightDiagnostic diag = emitError(loc) << "Unsupported cmpi predicate";
    return failure();
  }

  MaskState lhsState;
  if (failed(lhsState.parse(cmpOp.getLhs(), loc, rewriter)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(cmpOp.getRhs(), loc, rewriter)))
    return failure();

  assert((!lhsState.scalar && rhsState.scalar) && "Unsupported cmpi scenario");

  int32_t cmpDim = -1;
  for (int32_t i = 0; i < lhsState.getRank(); i++) {
    auto dimIntAttr = getIntAttr(lhsState.dims[i]);
    if (!dimIntAttr || dimIntAttr.value() != 1) {
      if (cmpDim != -1) {
        InFlightDiagnostic diag = emitError(loc)
                                  << "Unsupported cmpi with more than one "
                                     "dimension with size larger than 1";
        return failure();
      }
      cmpDim = i;
    }
  }
  assert(cmpDim != -1 &&
         "Unexpected case where no dimension has size larger than 1");

  auto newEnd = minOFRs(lhsState.end, rhsState.scalar, loc, rewriter);
  auto newDim = subOFRs(newEnd, lhsState.start, loc, rewriter);

  for (int32_t i = 0; i < lhsState.getRank(); i++) {
    if (i == cmpDim)
      this->dims.push_back(newDim);
    else
      this->dims.push_back(lhsState.dims[i]);
  }

  return success();
}

LogicalResult MaskState::parseMakeRange(triton::MakeRangeOp rangeOp,
                                        const Location loc,
                                        ConversionPatternRewriter &rewriter) {
  assert(this->isEmpty());

  auto shape = rangeOp.getType().cast<ShapedType>().getShape();
  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];

  if (stride != 1) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "stride must be 1 for make_range whose result is used "
           "as load or store masks";
    return failure();
  }

  this->start = rewriter.getIndexAttr(start);
  this->end = rewriter.getIndexAttr(end);
  this->dims.push_back(rewriter.getIndexAttr(shape[0]));

  return success();
}

LogicalResult MaskState::parseBroadcast(triton::BroadcastOp broadcastOp,
                                        const Location loc,
                                        ConversionPatternRewriter &rewriter) {
  assert(this->isEmpty());

  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();
  assert(src.getType().isa<ShapedType>() &&
         "input to tt.broadcast should be a tensor");

  auto srcShape = src.getType().cast<ShapedType>().getShape();
  auto dstShape = dst.getType().cast<ShapedType>().getShape();
  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  if (failed(parse(src, loc, rewriter)))
    return failure();

  for (size_t i = 0; i < srcShape.size(); i++) {
    if (srcShape[i] == dstShape[i])
      continue;
    else if (srcShape[i] < dstShape[i])
      this->dims[i] = rewriter.getIndexAttr(dstShape[i]);
    else
      llvm_unreachable("unexpected dimensions used in broadcast");
  }

  return success();
}

LogicalResult MaskState::parseSplat(triton::SplatOp splatOp, const Location loc,
                                    ConversionPatternRewriter &rewriter) {
  assert(this->isEmpty());

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = dst.getType().cast<ShapedType>().getShape();

  if (!src.getType().isa<IntegerType>()) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "splat source must be an integer scalar for load/store masks";
    return failure();
  }

  if (failed(this->parse(src, loc, rewriter)))
    return failure();

  for (auto s : dstShape)
    this->dims.push_back(rewriter.getIndexAttr(s));

  return success();
}

LogicalResult MaskState::parseExpandDims(triton::ExpandDimsOp expandDimsOp,
                                         const Location loc,
                                         ConversionPatternRewriter &rewriter) {
  assert(this->isEmpty());

  if (failed(this->parse(expandDimsOp.getSrc(), loc, rewriter)))
    return failure();

  auto dstShape =
      expandDimsOp.getResult().getType().cast<ShapedType>().getShape();
  auto axis = expandDimsOp.getAxis();
  assert(dstShape[axis] == 1 &&
         "expect changed dimension to be 1 in expand_dims");
  this->dims.insert(this->dims.begin() + axis, rewriter.getIndexAttr(1));

  return success();
}

} // namespace triton
} // namespace mlir
