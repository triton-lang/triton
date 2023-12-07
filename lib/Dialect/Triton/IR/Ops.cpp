#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir {
namespace triton {

// Parser & printer for assembly forms
ParseResult LoadOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse operands
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon())
    return failure();

  // Operand types
  SmallVector<Type> operandTypes;

  // Parse `optional(type(ptr)) -> type(result)`
  Type ptrType, resultType;
  if (parser.parseType(resultType))
    return failure();
  if (parser.parseOptionalArrow().succeeded()) {
    ptrType = resultType;
    if (parser.parseType(resultType))
      return failure();
    operandTypes.push_back(ptrType);
    result.addTypes(resultType);
  } else {
    operandTypes.push_back(getPointerTypeSameShape(resultType));
    result.addTypes(resultType);
  }

  // Determine `mask` and `other`
  int hasMask = 0, hasOther = 0;
  if (allOperands.size() >= 2) {
    operandTypes.push_back(getI1SameShape(resultType));
    hasMask = 1;
  }
  if (allOperands.size() >= 3) {
    operandTypes.push_back(resultType);
    hasOther = 1;
  }

  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();

  // Deduce `operandSegmentSizes` from the number of the operands
  auto operandSegmentSizesAttrName =
      LoadOp::getOperandSegmentSizesAttrName(result.name);
  result.addAttribute(
      operandSegmentSizesAttrName,
      parser.getBuilder().getDenseI32ArrayAttr({1, hasMask, hasOther}));

  return success();
}

void LoadOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer << getOperation()->getOperands();

  // `operandSegmentSizes` can be deduced, so we don't print it.
  printer.printOptionalAttrDict(getOperation()->getAttrs(),
                                {getOperandSegmentSizesAttrName()});

  // `type(ptr) -> type(result)`
  printer << " : ";
  // `type(ptr)` is optional during parsing, we only print for tensor pointers
  if (isTensorPointerType(getPtr().getType())) {
    printer.printStrippedAttrOrType(getPtr().getType());
    printer << " -> ";
  }
  printer.printStrippedAttrOrType(getResult().getType());
}

void LoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), getPtr(),
                       SideEffects::DefaultResource::get());
  if (getIsVolatile())
    effects.emplace_back(MemoryEffects::Write::get(),
                         SideEffects::DefaultResource::get());
}

ParseResult StoreOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse operands
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon())
    return failure();

  // Operand types
  SmallVector<Type> operandTypes;

  // Parse `optional(type(ptr)), type(val)`
  // Pointer type
  Type ptrType, valType;
  if (parser.parseType(valType))
    return failure();
  if (parser.parseOptionalComma().succeeded()) {
    ptrType = valType;
    if (parser.parseType(valType))
      return failure();
    operandTypes.push_back(ptrType);
  } else {
    operandTypes.push_back(getPointerTypeSameShape(valType));
  }

  // Value type
  operandTypes.push_back(valType);

  // Determine `mask`
  if (allOperands.size() >= 3)
    operandTypes.push_back(getI1SameShape(valType));

  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void StoreOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer << getOperation()->getOperands();
  printer.printOptionalAttrDict(getOperation()->getAttrs(), /*elidedAttrs=*/{});

  // `type(ptr), type(value)`
  printer << " : ";
  // `type(ptr)` is optional during parsing, we only print for tensor pointers
  if (isTensorPointerType(getPtr().getType())) {
    printer.printStrippedAttrOrType(getPtr().getType());
    printer << ", ";
  }
  printer.printStrippedAttrOrType(getValue().getType());
}

} // namespace triton
} // namespace mlir

#define GET_OP_CLASSES
#include "triton/Dialect/Triton/IR/Ops.cpp.inc"

// enum attribute definitions
#include "triton/Dialect/Triton/IR/OpsEnums.cpp.inc"

namespace mlir {
namespace triton {

//-- LoadOp --
static Type getLoadOpResultType(::mlir::OpBuilder &builder, Type ptrType) {
  auto ptrTensorType = ptrType.dyn_cast<RankedTensorType>();
  if (!ptrTensorType)
    return ptrType.cast<PointerType>().getPointeeType();
  auto shape = ptrTensorType.getShape();
  Type elementType =
      ptrTensorType.getElementType().cast<PointerType>().getPointeeType();
  return RankedTensorType::get(shape, elementType);
}

void LoadOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                   ::mlir::Value ptr, ::mlir::triton::CacheModifier cache,
                   ::mlir::triton::EvictionPolicy evict, bool isVolatile) {
  LoadOp::build(builder, state, ptr, /*mask=*/{}, /*other=*/{},
                /*boundaryCheck=*/{}, /*padding=*/{}, cache, evict, isVolatile);
}

void LoadOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                   ::mlir::Value ptr, ArrayRef<int32_t> boundaryCheck,
                   std::optional<::mlir::triton::PaddingOption> padding,
                   ::mlir::triton::CacheModifier cache,
                   ::mlir::triton::EvictionPolicy evict, bool isVolatile) {
  LoadOp::build(builder, state, ptr, /*mask=*/{}, /*other=*/{}, boundaryCheck,
                padding, cache, evict, isVolatile);
}

void LoadOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                   ::mlir::Value ptr, ::mlir::Value mask,
                   ::mlir::triton::CacheModifier cache,
                   ::mlir::triton::EvictionPolicy evict, bool isVolatile) {
  LoadOp::build(builder, state, ptr, mask, /*other=*/{}, /*boundaryCheck=*/{},
                /*padding=*/{}, cache, evict, isVolatile);
}

void LoadOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                   ::mlir::Value ptr, ::mlir::Value mask, ::mlir::Value other,
                   ::mlir::triton::CacheModifier cache,
                   ::mlir::triton::EvictionPolicy evict, bool isVolatile) {
  LoadOp::build(builder, state, ptr, mask, other, /*boundaryCheck=*/{},
                /*padding=*/{}, cache, evict, isVolatile);
}

void LoadOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                   ::mlir::Value ptr, ::mlir::Value mask, ::mlir::Value other,
                   std::optional<ArrayRef<int32_t>> boundaryCheck,
                   std::optional<::mlir::triton::PaddingOption> padding,
                   ::mlir::triton::CacheModifier cache,
                   ::mlir::triton::EvictionPolicy evict, bool isVolatile) {
  // Operands
  state.addOperands(ptr);
  if (mask) {
    state.addOperands(mask);
    if (other) {
      state.addOperands(other);
    }
  }

  // Attributes
  state.addAttribute(
      getOperandSegmentSizesAttrName(state.name),
      builder.getDenseI32ArrayAttr({1, (mask ? 1 : 0), (other ? 1 : 0)}));
  if (boundaryCheck.has_value()) {
    state.addAttribute(getBoundaryCheckAttrName(state.name),
                       builder.getDenseI32ArrayAttr(boundaryCheck.value()));
  }
  if (padding.has_value()) {
    state.addAttribute(getPaddingAttrName(state.name),
                       ::mlir::triton::PaddingOptionAttr::get(
                           builder.getContext(), padding.value()));
  }
  state.addAttribute(
      getCacheAttrName(state.name),
      ::mlir::triton::CacheModifierAttr::get(builder.getContext(), cache));
  state.addAttribute(
      getEvictAttrName(state.name),
      ::mlir::triton::EvictionPolicyAttr::get(builder.getContext(), evict));
  state.addAttribute(getIsVolatileAttrName(state.name),
                     builder.getBoolAttr(isVolatile));

  // Result type
  Type resultType = getLoadOpResultType(builder, ptr.getType());
  state.addTypes({resultType});
}

// load(ptr, splat(1), ...)        -> load(ptr, ...)
// load(ptr, splat(0), other, ...) -> other
struct CanonicalizeMaskedLoadPattern
    : public mlir::OpRewritePattern<triton::LoadOp> {
  CanonicalizeMaskedLoadPattern(mlir::MLIRContext *context)
      : OpRewritePattern<triton::LoadOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(triton::LoadOp loadOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto mask = loadOp.getMask();
    if (!mask)
      return mlir::failure();

    auto constantMask =
        llvm::dyn_cast_or_null<arith::ConstantOp>(mask.getDefiningOp());
    if (!constantMask)
      return mlir::failure();

    auto splatMask = constantMask.getValue().dyn_cast<SplatElementsAttr>();
    if (!splatMask)
      return mlir::failure();

    if (splatMask.getSplatValue<IntegerAttr>().getValue() == true) {
      // mask = splat(1)
      rewriter.replaceOpWithNewOp<triton::LoadOp>(
          loadOp, loadOp.getType(), loadOp.getPtr(), Value(), Value(),
          loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
          loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
    } else {
      // mask = splat(0)

      // If there's no "other", the value is "undef".  Perhaps we want to
      // optimize it in the future.x
      auto otherVal = loadOp.getOther();
      if (!otherVal)
        return mlir::failure();
      rewriter.replaceOp(loadOp, otherVal);
    }
    return mlir::success();
  }
};

void triton::LoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<CanonicalizeMaskedLoadPattern>(context);
}

//-- StoreOp --
void StoreOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                    ::mlir::Value ptr, ::mlir::Value value,
                    ::mlir::triton::CacheModifier cache,
                    ::mlir::triton::EvictionPolicy evict) {
  return StoreOp::build(builder, state, ptr, value, /*mask=*/{},
                        /*boundaryCheck=*/{}, cache, evict);
}

void StoreOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                    ::mlir::Value ptr, ::mlir::Value value, ::mlir::Value mask,
                    ::mlir::triton::CacheModifier cache,
                    ::mlir::triton::EvictionPolicy evict) {
  return StoreOp::build(builder, state, ptr, value, mask, /*boundaryCheck=*/{},
                        cache, evict);
}

void StoreOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                    ::mlir::Value ptr, ::mlir::Value value,
                    ArrayRef<int32_t> boundaryCheck,
                    ::mlir::triton::CacheModifier cache,
                    ::mlir::triton::EvictionPolicy evict) {
  return StoreOp::build(builder, state, ptr, value, /*mask=*/{},
                        builder.getDenseI32ArrayAttr(boundaryCheck), cache,
                        evict);
}

// store(ptr, value, splat(1), ...) -> store(ptr, value, ...)
// store(ptr, value, splat(0), ...) -> [none]
struct CanonicalizeMaskedStorePattern
    : public mlir::OpRewritePattern<triton::StoreOp> {
  CanonicalizeMaskedStorePattern(mlir::MLIRContext *context)
      : OpRewritePattern<triton::StoreOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp storeOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto mask = storeOp.getMask();
    if (!mask)
      return mlir::failure();

    auto constantMask =
        llvm::dyn_cast_or_null<arith::ConstantOp>(mask.getDefiningOp());
    if (!constantMask)
      return mlir::failure();

    auto splatMask = constantMask.getValue().dyn_cast<SplatElementsAttr>();
    if (!splatMask)
      return mlir::failure();

    if (splatMask.getSplatValue<IntegerAttr>().getValue() == true) {
      // mask = splat(1)
      rewriter.replaceOpWithNewOp<triton::StoreOp>(
          storeOp, storeOp.getPtr(), storeOp.getValue(), storeOp.getCache(),
          storeOp.getEvict());
    } else {
      // mask = splat(0)
      rewriter.eraseOp(storeOp);
    }
    return mlir::success();
  }
};

void triton::StoreOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<CanonicalizeMaskedStorePattern>(context);
}

//-- TransOp --
mlir::LogicalResult mlir::triton::TransOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // type is the same as the input
  auto argTy = operands[0].getType().cast<RankedTensorType>();
  SmallVector<int64_t> retShape(argTy.getShape().begin(),
                                argTy.getShape().end());
  std::reverse(retShape.begin(), retShape.end());
  auto retEltTy = argTy.getElementType();
  Attribute argEncoding = argTy.getEncoding();
  Attribute retEncoding;
  if (argEncoding) {
    Dialect &dialect = argEncoding.getDialect();
    auto inferLayoutInterface = dyn_cast<DialectInferLayoutInterface>(&dialect);
    if (inferLayoutInterface->inferTransOpEncoding(argEncoding, retEncoding)
            .failed()) {
      llvm::report_fatal_error("failed to infer layout for ReduceOp");
      return mlir::failure();
    }
  }
  inferredReturnTypes.push_back(
      RankedTensorType::get(retShape, retEltTy, retEncoding));
  return mlir::success();
}

//-- DotOp --
mlir::LogicalResult mlir::triton::DotOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // type is the same as the accumulator
  auto accTy = operands[2].getType().cast<RankedTensorType>();
  inferredReturnTypes.push_back(accTy);

  // verify encodings
  auto aEnc = operands[0].getType().cast<RankedTensorType>().getEncoding();
  auto bEnc = operands[1].getType().cast<RankedTensorType>().getEncoding();
  auto retEnc = accTy.getEncoding();
  if (aEnc) {
    assert(bEnc);
    Dialect &dialect = aEnc.getDialect();
    auto interface = dyn_cast<DialectInferLayoutInterface>(&dialect);
    if (interface->inferDotOpEncoding(aEnc, 0, retEnc, location).failed())
      return mlir::failure();
    if (interface->inferDotOpEncoding(bEnc, 1, retEnc, location).failed())
      return mlir::failure();
  }
  return mlir::success();
}

LogicalResult mlir::triton::DotOp::verify() {
  auto aTy = getOperand(0).getType().cast<RankedTensorType>();
  auto bTy = getOperand(1).getType().cast<RankedTensorType>();
  if (aTy.getElementType().getIntOrFloatBitWidth() !=
      bTy.getElementType().getIntOrFloatBitWidth())
    return emitError(
        "element types of operands A and B must have same bit width");
  auto aEncoding = aTy.getEncoding();
  auto bEncoding = bTy.getEncoding();
  if (!aEncoding && !bEncoding)
    return mlir::success();
  // Verify that the encodings are valid.
  if (!aEncoding || !bEncoding)
    return emitError("mismatching encoding between A and B operands");
  Dialect &dialect = aEncoding.getDialect();
  auto interface = cast<DialectInferLayoutInterface>(&dialect);
  return interface->verifyDotOpEncodingCompatibility(getOperation(), aEncoding,
                                                     bEncoding);
}

//-- MakeRangeOp --
OpFoldResult MakeRangeOp::fold(FoldAdaptor adaptor) {
  // make_range(start, start + 1) -> constant(start)
  if (adaptor.getStart() + 1 == adaptor.getEnd()) {
    auto shapedType = getType().cast<ShapedType>();
    return SplatElementsAttr::get(shapedType, adaptor.getStartAttr());
  }
  return {};
}

LogicalResult MakeRangeOp::verify() {
  int64_t start = getStartAttr().getInt();
  int64_t end = getEndAttr().getInt();
  if (start > end) {
    return this->emitOpError() << "start must be less than or equal to end";
  }
  auto ty = getType().dyn_cast<RankedTensorType>();
  if (!ty) {
    return this->emitOpError() << "return type must be a ranked tensor";
  }
  if (ty.getShape().size() != 1) {
    return this->emitOpError() << "return type must be a 1D tensor";
  }
  if (end - start != ty.getShape()[0]) {
    return this->emitOpError()
           << "number of elements in returned tensor, " << ty.getShape()[0]
           << ", must match size of range [" << start << ", " << end
           << "), which has " << end - start << " elements";
  }
  if (!ty.getElementType().isInteger(32)) {
    return this->emitOpError() << "returned tensor must have i32 elements";
  }
  return success();
}

//-- ReduceOp --
static mlir::LogicalResult
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
        return mlir::failure();
      }
    }
    // create type
    inferredReturnTypes.push_back(
        RankedTensorType::get(retShape, retEltTy, retEncoding));
  }
  return mlir::success();
}

void ReduceOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     mlir::ValueRange operands, int axis) {
  SmallVector<Type> inferredReturnTypes;
  for (unsigned i = 0; i < operands.size(); ++i) {
    auto argTy = operands[i].getType().cast<RankedTensorType>();
    auto retEltTy = argTy.getElementType();
    (void)inferReduceReturnShape(argTy, retEltTy, axis, inferredReturnTypes);
  }

  ReduceOp::build(builder, state, inferredReturnTypes, operands, axis);
}

mlir::LogicalResult mlir::triton::ReduceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  Properties *prop = properties.as<Properties *>();
  int axis = prop->axis.getInt();
  for (auto arg : operands) {
    auto argTy = arg.getType().cast<RankedTensorType>();
    auto retEltTy = argTy.getElementType();
    if (inferReduceReturnShape(argTy, retEltTy, axis, inferredReturnTypes)
            .failed()) {
      return failure();
    }
  }
  return success();
}

mlir::LogicalResult mlir::triton::ReduceOp::verify() {
  if (this->getOperands().size() < 1) {
    return this->emitOpError() << "must have at least 1 operand";
  }
  for (const auto &operand : this->getOperands()) {
    if (!dyn_cast<RankedTensorType>(operand.getType())) {
      return this->emitOpError() << "operands must be RankedTensorType";
    }
  }
  return success();
}

mlir::LogicalResult mlir::triton::ReduceOp::verifyRegions() {
  auto argElementTypes = this->getElementTypes();
  const auto &operands = this->getOperands();
  const auto numArgs = 2 * operands.size();
  auto &block = *this->getBody();
  if (block.getNumArguments() != numArgs) {
    return this->emitOpError() << "nested block must take " << numArgs
                               << " arguments, but given block with "
                               << block.getNumArguments() << " arguments";
  }
  unsigned i = 0;
  const auto &blockArgTypes = block.getArgumentTypes();
  for (unsigned i = 0; i < numArgs; ++i) {
    const auto &blockArgTy = blockArgTypes[i];
    const auto &argElemTy = argElementTypes[i % operands.size()];
    if (blockArgTy != argElemTy) {
      return this->emitOpError()
             << "type mismatch on combine operation. Expected argument " << i
             << " to have type " << argElemTy << " but got " << blockArgTy;
    }
  }

  auto terminator =
      dyn_cast<mlir::triton::ReduceReturnOp>(block.getTerminator());
  if (!terminator) {
    return this->emitOpError()
           << "combine operation must be terminated "
           << "with a ReduceReturnOp but got " << block.getTerminator();
  }
  const auto &combineResults = terminator->getOperands();
  if (combineResults.size() != operands.size()) {
    return this->emitOpError()
           << "expected combine operation to return " << operands.size()
           << " values but got " << combineResults.size();
  }
  for (unsigned i = 0; i < combineResults.size(); ++i) {
    const auto &resultTy = combineResults[i].getType();
    const auto &argElemTy = argElementTypes[i];
    if (resultTy != argElemTy) {
      return this->emitOpError()
             << "type mismatch on combine operation. Expected argument " << i
             << " to have type " << argElemTy << " but got " << resultTy;
    }
  }
  return mlir::success();
}

llvm::SmallVector<mlir::RankedTensorType> ReduceOp::getInputTypes() {
  llvm::SmallVector<RankedTensorType> srcTys;
  srcTys.reserve(this->getNumOperands());
  for (const auto &ty : this->getOperands().getTypes()) {
    srcTys.push_back(ty.cast<RankedTensorType>());
  }
  return srcTys;
}

llvm::SmallVector<Type> ReduceOp::getElementTypes() {
  llvm::SmallVector<Type> srcElemTys;
  srcElemTys.reserve(this->getNumOperands());
  for (const auto &op : this->getOperands()) {
    srcElemTys.push_back(
        op.getType().cast<RankedTensorType>().getElementType());
  }
  return srcElemTys;
}

unsigned ReduceOp::getNumOperands() { return this->getOperands().size(); }

//-- ScanOp --
void ScanOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::ValueRange operands, int axis) {
  SmallVector<Type> inferredReturnTypes;
  for (auto arg : operands)
    inferredReturnTypes.push_back(arg.getType());
  ReduceOp::build(builder, state, inferredReturnTypes, operands, axis);
}

mlir::LogicalResult mlir::triton::ScanOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  for (auto arg : operands)
    inferredReturnTypes.push_back(arg.getType());
  return success();
}

mlir::LogicalResult mlir::triton::ScanOp::verify() {
  if (this->getOperands().size() < 1) {
    return this->emitOpError() << "must have at least 1 operand";
  }
  for (const auto &operand : this->getOperands()) {
    if (!dyn_cast<RankedTensorType>(operand.getType())) {
      return this->emitOpError() << "operands must be RankedTensorType";
    }
  }
  return success();
}

//-- SplatOp --
OpFoldResult SplatOp::fold(FoldAdaptor adaptor) {
  auto value = adaptor.getSrc();
  if (!value)
    return {};
  auto shapedType = getType().cast<ShapedType>();
  auto ret = SplatElementsAttr::get(shapedType, ArrayRef<Attribute>(value));
  return ret;
}

//-- ExpandDimsOp --
mlir::LogicalResult mlir::triton::ExpandDimsOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // infer shape
  auto arg = operands[0];
  auto argTy = arg.getType().cast<RankedTensorType>();
  auto retShape = argTy.getShape().vec();
  Properties *prop = properties.as<Properties *>();
  int axis = prop->axis.getInt();
  retShape.insert(retShape.begin() + axis, 1);
  // infer encoding
  Attribute argEncoding = argTy.getEncoding();
  Attribute retEncoding;
  if (argEncoding) {
    Dialect &dialect = argEncoding.getDialect();
    auto inferLayoutInterface = dyn_cast<DialectInferLayoutInterface>(&dialect);
    if (inferLayoutInterface
            ->inferExpandDimsOpEncoding(argEncoding, axis, retEncoding, loc)
            .failed())
      return emitOptionalError(loc, "failed to infer layout for ExpandDimsOp");
  }
  // create type
  auto argEltTy = argTy.getElementType();
  inferredReturnTypes.push_back(
      RankedTensorType::get(retShape, argEltTy, retEncoding));
  return mlir::success();
}

LogicalResult ExpandDimsOp::canonicalize(ExpandDimsOp op,
                                         PatternRewriter &rewriter) {
  auto definingOp = op.getOperand().getDefiningOp();
  if (!definingOp) {
    return mlir::failure();
  }
  // expand_dims(splat) -> splat
  if (auto splat = dyn_cast<triton::SplatOp>(definingOp)) {
    rewriter.replaceOpWithNewOp<triton::SplatOp>(op, op.getType(),
                                                 splat.getOperand());
    return mlir::success();
  }
  // expand_dims(broadcast) -> broadcast(expand_dims)
  //
  // On it's own this doesn't do much, but consider
  //    broadcast(expand_dims(broadcast))
  // -> broadcast(broadcast(expand_dims))
  // -> broadcast(expand_dims)
  if (auto broadcast = dyn_cast<triton::BroadcastOp>(definingOp)) {
    auto src = broadcast.getSrc();
    auto srcTy = src.getType().dyn_cast<RankedTensorType>();
    auto elemTy = srcTy.getElementType();
    auto srcShape = srcTy.getShape();

    llvm::SmallVector<int64_t, 4> newExpandShape(srcShape.begin(),
                                                 srcShape.end());
    newExpandShape.insert(newExpandShape.begin() + op.getAxis(), 1);
    auto newExpandTy = RankedTensorType::get(newExpandShape, elemTy);

    auto newExpand = rewriter.create<triton::ExpandDimsOp>(
        op.getLoc(), newExpandTy, src, op.getAxis());
    auto newBroadcast = rewriter.create<triton::BroadcastOp>(
        broadcast.getLoc(), op.getType(), newExpand.getResult());
    rewriter.replaceOp(op, {newBroadcast.getResult()});
    return mlir::success();
  }

  return mlir::failure();
}

template <typename ViewLikeOp>
static OpFoldResult foldViewLikeOp(ViewLikeOp op, Attribute value) {
  if (!value)
    return {};

  auto shapedType = op.getType().template cast<mlir::ShapedType>();
  if (auto denseElemsAttr = value.dyn_cast<DenseElementsAttr>()) {
    if (denseElemsAttr.isSplat()) {
      return denseElemsAttr.resizeSplat(shapedType);
    } else {
      return denseElemsAttr.reshape(shapedType);
    }
  }
  return {};
}

OpFoldResult ExpandDimsOp::fold(FoldAdaptor adaptor) {
  return foldViewLikeOp(*this, adaptor.getSrc());
}

//-- ReshapeOp --
template <typename OpType>
LogicalResult canonicalizeViewOrBroadcast(OpType op,
                                          PatternRewriter &rewriter) {
  auto definingOp = op.getOperand().getDefiningOp();
  if (!definingOp) {
    return mlir::failure();
  }

  // view(view) -> view
  if (auto parentView = dyn_cast<OpType>(definingOp)) {
    rewriter.replaceOpWithNewOp<OpType>(op, TypeRange({op.getType()}),
                                        parentView->getOperands(),
                                        parentView->getAttrs());
    return mlir::success();
  }

  // view(splat) -> splat
  if (auto splat = dyn_cast<triton::SplatOp>(definingOp)) {
    rewriter.replaceOpWithNewOp<triton::SplatOp>(op, op.getType(),
                                                 splat.getOperand());
    return mlir::success();
  }

  return mlir::failure();
}

LogicalResult ReshapeOp::canonicalize(ReshapeOp op, PatternRewriter &rewriter) {
  if (!op.getAllowReorder() || op.getEfficientLayout().has_value())
    return failure();
  return canonicalizeViewOrBroadcast(op, rewriter);
}

OpFoldResult ReshapeOp::fold(FoldAdaptor adaptor) {
  if (getType() == getOperand().getType()) {
    // no-op
    return getOperand();
  }

  return foldViewLikeOp(*this, adaptor.getSrc());
}

mlir::LogicalResult mlir::triton::ReshapeOp::verify() {
  auto dstType = getType().cast<RankedTensorType>();
  auto srcType = getSrc().getType().cast<RankedTensorType>();
  if (dstType.getNumElements() != srcType.getNumElements()) {
    return emitError(
        "number of src and dst elements of reshape must be the same");
  }
  return mlir::success();
}

//-- BroadcastOp --
LogicalResult BroadcastOp::canonicalize(BroadcastOp op,
                                        PatternRewriter &rewriter) {
  return canonicalizeViewOrBroadcast(op, rewriter);
}

OpFoldResult BroadcastOp::fold(FoldAdaptor adaptor) {
  if (getType() == getOperand().getType()) {
    // no-op
    return getOperand();
  }

  auto value = adaptor.getSrc();
  if (!value)
    return {};

  if (auto denseElemsAttr = value.dyn_cast<SplatElementsAttr>()) {
    auto shapedType = getType().cast<ShapedType>();
    return denseElemsAttr.resizeSplat(shapedType);
  }
  return {};
}

//-- MakeTensorPtrOp --
void MakeTensorPtrOp::build(::mlir::OpBuilder &builder,
                            ::mlir::OperationState &state, ::mlir::Value base,
                            ::mlir::ValueRange shape,
                            ::mlir::ValueRange strides,
                            ::mlir::ValueRange offsets,
                            ArrayRef<int32_t> tensorShape,
                            ArrayRef<int32_t> order) {
  // Get pointer type from `base`
  auto pointerType = base.getType().cast<PointerType>();
  assert(pointerType != nullptr);

  // Build type `tt.ptr<tensor<tensorShape, base.pointeeType>>`
  auto tensorType = RankedTensorType::get(
      SmallVector<int64_t>(tensorShape.begin(), tensorShape.end()),
      pointerType.getPointeeType());
  auto result = PointerType::get(tensorType, 1);

  return build(builder, state, result, base, shape, strides, offsets,
               builder.getDenseI32ArrayAttr(order));
}

// The following ops, including `call`, `func`, and `return` are copied and
// modified from
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Func/IR/FuncOps.cpp
// We could revert it back once MLIR has a better inliner interface.
//-- FuncOp --
void triton::FuncOp::build(OpBuilder &builder, OperationState &state,
                           StringRef name, FunctionType type,
                           ArrayRef<NamedAttribute> attrs,
                           ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

ParseResult triton::FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void triton::FuncOp::print(OpAsmPrinter &printer) {
  function_interface_impl::printFunctionOp(
      printer, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

// -- CallOp --
LogicalResult
triton::CallOp::verifySymbolUses(mlir::SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this).getProperties().callee;
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

// -- ReturnOp --
LogicalResult triton::ReturnOp::verify() {
  auto function = cast<triton::FuncOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match function result type ("
                         << results[i] << ")"
                         << " in function @" << function.getName();

  return success();
}

// -- ElementwiseInlineAsmOp --
void ElementwiseInlineAsmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (getPure())
    return;
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

// -- ExternElementwiseOp --
void ExternElementwiseOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (getPure())
    return;
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

} // namespace triton
} // namespace mlir
