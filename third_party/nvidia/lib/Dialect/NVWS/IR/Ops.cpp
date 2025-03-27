#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"

#define GET_OP_CLASSES
#include "Dialect/NVWS/IR/Ops.cpp.inc"

namespace mlir::triton::nvws {

template <typename T>
static std::optional<Twine> verifySlice(T &origType, T &newType, size_t rank) {
  if (!origType || !newType)
    return "MLIR Types don't match";
  if (origType.getElementType() != newType.getElementType() ||
      origType.getRank() - rank != newType.getRank()) {
    return "Ranks don't match";
  }
  for (size_t i = 0, e = newType.getShape().size(); i < e; i++) {
    if (origType.getShape()[i + rank] != newType.getShape()[i])
      return "Dimensions don't match";
  }
  return std::nullopt;
}

std::optional<Twine> static arefRegionVerify(
    ArefType aref, ValueTypeRange<MutableArrayRef<BlockArgument>> blockArgTypes,
    size_t rank) {
  auto numBatchAxes = aref.getNumBatchAxes();
  if (numBatchAxes ? *numBatchAxes != rank : rank == 0)
    return "The Number of Batch axes on the aref type does not match the "
           "number of indexes";
  auto typeArray = aref.getBaseType();
  if (typeArray.size() != blockArgTypes.size())
    return "Aref has different number of arguments than region";
  // This should probably rely on the memdescSubviewOp verifier?
  for (auto [orig, arg] : llvm::zip(typeArray, blockArgTypes)) {
    if (rank == 0) {
      if (orig != arg)
        return "MLIR Types don't match";
    } else {
      if (auto origT = dyn_cast<RankedTensorType>(orig)) {
        auto argT = dyn_cast<RankedTensorType>(arg);
        if (auto result = verifySlice(origT, argT, rank))
          return result;
      } else if (auto origT = dyn_cast<triton::gpu::MemDescType>(orig)) {
        auto argT = dyn_cast<triton::gpu::MemDescType>(arg);
        if (auto result = verifySlice(origT, argT, rank))
          return result;
      } else {
        return "Slicing not Implemented for this type";
      }
    }
  }
  return std::nullopt;
}

LogicalResult ArefPutOp::verify() {
  if (auto result =
          arefRegionVerify(getOperand().getType(),
                           getRegion().getArgumentTypes(), getIndexes().size()))
    return emitError(*result);
  return success();
}

ParseResult ArefRegionParse(OpAsmParser &p, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<OpAsmParser::Argument> blockArgs;
  SMLoc operandLoc = p.getCurrentLocation();
  if (p.parseOperandList(operands, AsmParser::Delimiter::None) ||
      p.parseOperandList(operands, AsmParser::Delimiter::OptionalSquare) ||
      p.parseKeyword("as") ||
      p.parseArgumentList(blockArgs, AsmParser::Delimiter::Paren, true) ||
      p.parseRegion(*result.addRegion(), blockArgs) ||
      p.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  FunctionType types;
  if (p.parseColon() || p.parseType(types) ||
      p.resolveOperands(operands, types.getInputs(), operandLoc,
                        result.operands))
    return failure();

  result.addTypes(types.getResults());

  return success();
}

ParseResult ArefPutOp::parse(OpAsmParser &p, OperationState &result) {
  return ArefRegionParse(p, result);
}

void ArefPutOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printOperand(getOperand());
  auto indexes = getIndexes();
  if (indexes.size() > 0) {
    p << '[';
    p.printOperands(getIndexes());
    p << ']';
  }

  p << " as ";
  p << '(';
  llvm::interleaveComma(getRegion().getArguments(), p,
                        [&](BlockArgument arg) { p.printRegionArgument(arg); });
  p << ')';
  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);

  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  p << " : ";
  p.printFunctionalType(*this);
}

LogicalResult ArefGetOp::verify() {
  if (auto result =
          arefRegionVerify(getOperand().getType(),
                           getRegion().getArgumentTypes(), getIndexes().size()))
    return emitError(*result);
  return success();
}

ParseResult ArefGetOp::parse(OpAsmParser &p, OperationState &result) {
  return ArefRegionParse(p, result);
}

void ArefGetOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printOperand(getOperand());
  auto indexes = getIndexes();
  if (indexes.size() > 0) {
    p << '[';
    p.printOperands(getIndexes());
    p << ']';
  }

  p << " as ";
  p << '(';
  llvm::interleaveComma(getRegion().getArguments(), p,
                        [&](BlockArgument arg) { p.printRegionArgument(arg); });
  p << ')';
  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());

  p << " : ";
  p.printFunctionalType(*this);
}

LogicalResult ArefReturnOp::verify() {
  Operation *op = getOperation();
  auto operandTypes = getOperandTypes();
  if (auto get = dyn_cast<ArefGetOp>(op->getBlock()->getParentOp())) {
    auto resultTypes = get.getResultTypes();
    if (operandTypes.size() != resultTypes.size()) {
      return emitError("Mismatching number of returns");
    }
    for (auto [returnType, parentType] : llvm::zip(operandTypes, resultTypes)) {
      if (returnType != parentType) {
        return emitError(
            "Return sources and Parent Op Results have different types");
      }
    }
  }
  if (isa<ArefPutOp>(op->getBlock()->getParentOp())) {
    auto argTypes = op->getBlock()->getArgumentTypes();
    SmallVector<Type> inRegTypes =
        llvm::filter_to_vector(argTypes, [](const Type &type) {
          if (isa<triton::gpu::MemDescType>(type))
            return false;
          return true;
        });

    if (operandTypes.size() != inRegTypes.size()) {
      return emitError("Mismatching number of returns");
    }
    for (auto [returnType, parentType] : llvm::zip(operandTypes, inRegTypes)) {
      if (returnType != parentType) {
        return emitError(
            "Return sources and Block Arguments have different types");
      }
    }
  }
  return success();
}

LogicalResult WarpGroupOp::verify() {
  auto numWarps = getNumWarps();
  auto regions = getRegions();
  if (numWarps.size() != regions.size())
    return emitError("Must supply numWarps for each Warp Group");
  return success();
}

ParseResult WarpGroupOp::parse(OpAsmParser &p, OperationState &result) {
  auto ctx = p.getBuilder().getContext();

  SMLoc operandLoc = p.getCurrentLocation();
  if (p.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  SmallVector<int32_t> partitionNumWarps;
  while (succeeded(p.parseOptionalKeyword(
      ("partition" + Twine(partitionNumWarps.size()).str())))) {
    SMLoc regionLoc = p.getCurrentLocation();
    if (p.parseKeyword("num_warps") || p.parseLParen() ||
        p.parseInteger(partitionNumWarps.emplace_back()) || p.parseRParen() ||
        p.parseRegion(*result.addRegion()))
      return failure();
  }

  result.addAttribute(getNumWarpsAttrName(result.name),
                      p.getBuilder().getDenseI32ArrayAttr(partitionNumWarps));

  return success();
}

void WarpGroupOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                     {getNumWarpsAttrName()});

  for (auto [i, region, numWarps] :
       llvm::enumerate(getPartitionRegions(), getNumWarps())) {
    p.printNewline();
    p << "partition" << i;
    p << " num_warps(" << numWarps << ") ";
    p.printRegion(region, /*printEntryBlockArgs=*/false);
  }
}

} // namespace mlir::triton::nvws
