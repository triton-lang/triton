//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace triton {
namespace xpu {
//-- MakeRangeOp --
// OpFoldResult MakeRangeOp::fold(FoldAdaptor adaptor) {
//   // make_range(start, start + 1) -> constant(start)
//   if (adaptor.getStart() + 1 == adaptor.getEnd()) {
//     auto shapedType = cast<ShapedType>(getType());
//     return SplatElementsAttr::get(shapedType, adaptor.getStartAttr());
//   }
//   return {};
// }

LogicalResult MakeRangeOp::verify() {
  int64_t start = getStartAttr().getInt();
  int64_t end = getEndAttr().getInt();
  if (start > end) {
    return this->emitOpError() << "start must be less than or equal to end";
  }
  auto ty = getType();
  if (ty.getShape().size() != 1) {
    return this->emitOpError() << "return type must be a 1D tensor";
  }
  if (end - start < ty.getShape()[0]) { // loopIdx change the verify logic
    return this->emitOpError()
           << "number of elements in returned tensor, " << ty.getShape()[0]
           << ", must not exceed the size of range [" << start << ", " << end
           << "), which has " << end - start << " elements";
  }
  if (!ty.getElementType().isInteger(32)) {
    return this->emitOpError() << "returned tensor must have i32 elements";
  }
  return success();
}

// //-- InterleaveOp --
// OpFoldResult InterleaveOp::fold(FoldAdaptor adaptor) {
//   // make_range(start, start + 1) -> constant(start)
//   if (adaptor.getStart() + 1 == adaptor.getEnd()) {
//     auto shapedType = cast<ShapedType>(getType());
//     return SplatElementsAttr::get(shapedType, adaptor.getStartAttr());
//   }
//   return {};
// }

LogicalResult InterleaveOp::verify() {
  int64_t start = getStartAttr().getInt();
  int64_t end = getEndAttr().getInt();
  if (start > end) {
    return this->emitOpError() << "start must be less than or equal to end";
  }
  auto ty = getType();
  if (ty.getShape().size() != 1) {
    return this->emitOpError() << "return type must be a 1D tensor";
  }
  if (end - start < ty.getShape()[0]) { // loopIdx change the verify logic
    return this->emitOpError()
           << "number of elements in returned tensor, " << ty.getShape()[0]
           << ", must not exceed the size of range [" << start << ", " << end
           << "), which has " << end - start << " elements";
  }
  if (!ty.getElementType().isInteger(32) &&
      !ty.getElementType().isInteger(64)) {
    return this->emitOpError() << "returned tensor must have i32/i64 elements";
  }
  return success();
}

//-- ReduceOp --
static LogicalResult
inferReduceReturnShape(const RankedTensorType &argTy, const Type &retEltTy,
                       int axis, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto retShape = argTy.getShape().vec();
  retShape.erase(retShape.begin() + axis);
  if (retShape.empty()) {
    // 0d-tensor -> scalar
    inferredReturnTypes.push_back(retEltTy);
  } else {
    // nd-tensor where n >= 1
    // infer encoding
    Attribute argEncoding = argTy.getEncoding();
    Attribute retEncoding;
    if (argEncoding) {
      Dialect &dialect = argEncoding.getDialect();
      auto inferLayoutInterface =
          dyn_cast<DialectInferLayoutInterface>(&dialect);
      if (inferLayoutInterface
              ->inferReduceOpEncoding(argEncoding, axis, retEncoding)
              .failed()) {
        llvm::report_fatal_error("failed to infer layout for ReduceOp");
        return failure();
      }
    }
    // create type
    inferredReturnTypes.push_back(
        RankedTensorType::get(retShape, retEltTy, retEncoding));
  }
  return success();
}

// Helpers for Reductions and Scans
template <class Op> LogicalResult verifyReduceScan(Op &op) {
  if (op.getOperands().empty()) {
    return op.emitOpError() << "must have at least 1 operand";
  }
  if ((op.getNumOperands() - 1) != op.getNumResults()) { // -1 for loopIndex
    return op.emitOpError() << "must have the same number of inputs as outputs";
  }

  auto getElementType = [](Type ty) {
    if (auto tensorType = dyn_cast<RankedTensorType>(ty)) {
      return tensorType.getElementType();
    }
    return ty;
  };

  for (auto [opElemTy, resTy] :
       llvm::zip(op.getElementTypes(), op.getResultTypes())) {
    auto _opElemTy = getElementTypeOrSelf(opElemTy);
    if (_opElemTy != getElementType(resTy)) {
      return op.emitOpError() << "operand types and result types must agree";
    }
  }
  return success();
}

template <class ReturnOp, class Op>
static LogicalResult verifyRegionsImpl(Op &op) {
  auto argElementTypes = op.getElementTypes();
  const auto &operands = op.getOperands();
  const auto numArgs = 2 * (operands.size() - 1); // -1 for loopIndex
  auto &block = *op.getBody();
  if (block.getNumArguments() != numArgs) {
    return op.emitOpError() << "nested block must take " << numArgs
                            << " arguments, but given block with "
                            << block.getNumArguments() << " arguments";
  }
  unsigned i = 0;
  const auto &blockArgTypes = block.getArgumentTypes();
  for (unsigned i = 0; i < numArgs; ++i) {
    const auto &blockArgTy = blockArgTypes[i];
    const auto &argElemTy =
        argElementTypes[i % (operands.size() - 1)]; // -1 for loopIndex
    if (blockArgTy != argElemTy) {
      return op.emitOpError()
             << "type mismatch on combine operation. Expected argument " << i
             << " to have type " << argElemTy << " but got " << blockArgTy;
    }
  }

  auto terminator = dyn_cast<ReturnOp>(block.getTerminator());
  if (!terminator) {
    return op.emitOpError()
           << "combine operation must be terminated "
           << "with a ReduceReturnOp but got " << block.getTerminator();
  }
  const auto &combineResults = terminator->getOperands();
  if (combineResults.size() != (operands.size() - 1)) { // -1 for loopIndex
    return op.emitOpError()
           << "expected combine operation to return " << operands.size()
           << " values but got " << combineResults.size();
  }
  for (unsigned i = 0; i < combineResults.size(); ++i) {
    const auto &resultTy = combineResults[i].getType();
    const auto &argElemTy = argElementTypes[i];
    if (resultTy != argElemTy) {
      return op.emitOpError()
             << "type mismatch on combine operation. Expected argument " << i
             << " to have type " << argElemTy << " but got " << resultTy;
    }
  }
  return success();
}

static llvm::SmallVector<RankedTensorType>
getInputTypesImpl(const Operation::operand_range &operands) {
  llvm::SmallVector<RankedTensorType> srcTys;
  srcTys.reserve(operands.size());
  for (const auto &[i, ty] : llvm::enumerate(operands.getTypes())) {
    if (i == (operands.size() - 1))
      continue; // skip loopIndex
    srcTys.push_back(cast<RankedTensorType>(ty));
  }
  return srcTys;
}

static llvm::SmallVector<Type>
getElementTypesImpl(const Operation::operand_range &operands) {
  llvm::SmallVector<Type> srcElemTys;
  srcElemTys.reserve(operands.size());
  for (const auto &[i, op] : llvm::enumerate(operands)) {
    if (i == (operands.size() - 1))
      continue; // skip loopIndex
    srcElemTys.push_back(cast<RankedTensorType>(op.getType()).getElementType());
  }
  return srcElemTys;
}

LogicalResult ReduceOp::verify() { return verifyReduceScan(*this); }

LogicalResult ReduceOp::verifyRegions() {
  return verifyRegionsImpl<ReduceReturnOp>(*this);
}

llvm::SmallVector<RankedTensorType> ReduceOp::getInputTypes() {
  return getInputTypesImpl(this->getOperands());
}

llvm::SmallVector<Type> ReduceOp::getElementTypes() {
  return getElementTypesImpl(this->getOperands());
}

unsigned ReduceOp::getNumOperands() { return this->getOperands().size(); }

LogicalResult ReduceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  Properties *prop = properties.as<Properties *>();
  int axis = prop->axis.getInt();
  for (auto [i, arg] : llvm::enumerate(operands)) {
    if (i == (operands.size() - 1))
      continue; // skip loopIndex
    auto argTy = cast<RankedTensorType>(arg.getType());
    auto retEltTy = getElementTypeOrSelf(argTy.getElementType());
    if (inferReduceReturnShape(argTy, retEltTy, axis, inferredReturnTypes)
            .failed()) {
      return failure();
    }
  }
  return success();
}

//-- BroadcastOp --
template <typename OpType>
LogicalResult canonicalizeViewOrBroadcast(OpType op,
                                          PatternRewriter &rewriter) {
  auto definingOp = op.getSrc().getDefiningOp();
  if (!definingOp) {
    return failure();
  }

  // view(view) -> view
  if (auto parentView = dyn_cast<OpType>(definingOp)) {
    rewriter.replaceOpWithNewOp<OpType>(op, TypeRange({op.getType()}),
                                        parentView->getOperands(),
                                        parentView->getAttrs());
    return success();
  }

  // view(splat) -> splat
  if (auto splat = dyn_cast<SplatOp>(definingOp)) {
    rewriter.replaceOpWithNewOp<SplatOp>(op, op.getType(), splat.getSrc());
    return success();
  }

  return failure();
}

LogicalResult BroadcastOp::canonicalize(BroadcastOp op,
                                        PatternRewriter &rewriter) {
  return canonicalizeViewOrBroadcast(op, rewriter);
}

OpFoldResult BroadcastOp::fold(FoldAdaptor adaptor) {
  if (getType() == getSrc().getType()) {
    // no-op
    return getSrc();
  }

  auto value = adaptor.getSrc();
  if (!value)
    return {};

  if (auto denseElemsAttr = dyn_cast<SplatElementsAttr>(value)) {
    auto shapedType = cast<ShapedType>(getType());
    return denseElemsAttr.resizeSplat(shapedType);
  }
  return {};
}

} // namespace xpu
} // namespace triton
} // namespace mlir
