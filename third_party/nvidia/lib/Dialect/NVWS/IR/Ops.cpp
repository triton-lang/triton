#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/DebugStringHelper.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "Dialect/NVWS/IR/Ops.cpp.inc"

namespace mlir::triton::nvws {

template <typename T>
std::optional<Twine> verifySlice(T &origType, T &newType, size_t rank) {
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

std::optional<Twine>
ArefRegionVerify(ArefType aref,
                 ValueTypeRange<MutableArrayRef<BlockArgument>> blockArgTypes,
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
      }
    }
  }
  return std::nullopt;
}

LogicalResult ArefPutOp::verify() {
  if (auto result =
          ArefRegionVerify(getOperand().getType(),
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
          ArefRegionVerify(getOperand().getType(),
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

} // namespace mlir::triton::nvws
