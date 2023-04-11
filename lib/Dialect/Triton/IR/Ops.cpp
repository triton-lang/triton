#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OperationSupport.h"
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

//-- TransOp --
mlir::LogicalResult mlir::triton::TransOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
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
    DictionaryAttr attributes, RegionRange regions,
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

//-- ReduceOp --
mlir::LogicalResult mlir::triton::ReduceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // infer shape
  Value arg = operands[0];
  auto argTy = arg.getType().cast<RankedTensorType>();
  auto argEltTy = argTy.getElementType();
  auto i32Ty = IntegerType::get(argEltTy.getContext(), 32);
  auto redOp =
      attributes.get("redOp").cast<mlir::triton::RedOpAttr>().getValue();
  bool withIndex = mlir::triton::ReduceOp::withIndex(redOp);
  auto retEltTy = withIndex ? i32Ty : argEltTy;
  auto retShape = argTy.getShape().vec();
  int axis = attributes.get("axis").cast<IntegerAttr>().getInt();
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

bool mlir::triton::ReduceOp::withIndex(mlir::triton::RedOp redOp) {
  return redOp == mlir::triton::RedOp::ARGMIN ||
         redOp == mlir::triton::RedOp::ARGMAX ||
         redOp == mlir::triton::RedOp::ARGUMIN ||
         redOp == mlir::triton::RedOp::ARGUMAX ||
         redOp == mlir::triton::RedOp::ARGFMIN ||
         redOp == mlir::triton::RedOp::ARGFMAX;
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
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // infer shape
  auto arg = operands[0];
  auto argTy = arg.getType().cast<RankedTensorType>();
  auto retShape = argTy.getShape().vec();
  int axis = attributes.get("axis").cast<IntegerAttr>().getInt();
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

//-- ViewOp --
template <typename OpType>
LogicalResult canonicalizeViewOrBroadcast(OpType op,
                                          PatternRewriter &rewriter) {
  auto definingOp = op.getOperand().getDefiningOp();
  if (!definingOp) {
    return mlir::failure();
  }

  // view(view) -> view
  if (auto parent_view = dyn_cast<OpType>(definingOp)) {
    rewriter.replaceOpWithNewOp<OpType>(op, op.getType(),
                                        parent_view.getOperand());
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
LogicalResult ViewOp::canonicalize(ViewOp op, PatternRewriter &rewriter) {
  return canonicalizeViewOrBroadcast(op, rewriter);
}

OpFoldResult ViewOp::fold(FoldAdaptor adaptor) {
  if (getType() == getOperand().getType()) {
    // no-op
    return getOperand();
  }

  return foldViewLikeOp(*this, adaptor.getSrc());
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
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
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

} // namespace triton
} // namespace mlir
