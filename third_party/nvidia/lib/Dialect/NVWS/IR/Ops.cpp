#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeRange.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/NVWS/IR/NVWSAttrEnums.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/NVWS/IR/Ops.cpp.inc"

namespace mlir::triton::nvws {

LogicalResult ArefCreateOp::verify() {
  SmallVector<int> dims;
  for (auto operand : getOperands()) {
    SmallVector<Operation *> users(operand.user_begin(), operand.user_end());
    if (users.size() != 1)
      return emitError("Aref buffer is used elsewhere, Aref cannot guarantee "
                       "async safety");
    auto type = operand.getType();
    if (auto mType = dyn_cast<gpu::MemDescType>(type)) {
      dims.push_back(mType.getShape()[0]);
    } else if (auto rType = dyn_cast<RankedTensorType>(type)) {
      dims.push_back(rType.getShape()[0]);
    } else {
      return emitError("Aref is sliced, but input type isn't supported.");
    }
  }
  if (!llvm::all_equal(dims))
    return emitError("Leading dims of sliced aref inputs don't match.");

  return success();
}

template <typename T>
static std::optional<Twine> verifySlice(T &origType, T &newType) {
  if (!origType || !newType)
    return "MLIR Types don't match";
  if (origType.getElementType() != newType.getElementType() ||
      origType.getRank() - 1 != newType.getRank()) {
    return "Ranks don't match";
  }
  for (size_t i = 0, e = newType.getShape().size(); i < e; i++) {
    if (origType.getShape()[i + 1] != newType.getShape()[i])
      return "Dimensions don't match";
  }
  return std::nullopt;
}

std::optional<Twine> static arefEnterVerify(
    ArefType aref, mlir::ValueTypeRange<ResultRange> resultTypes) {
  auto typeArray = aref.getBaseType();
  if (typeArray.size() != resultTypes.size())
    return "Aref has different number of arguments than enter";
  // This should probably rely on the memdescSubviewOp verifier?
  for (auto [orig, arg] : llvm::zip(typeArray, resultTypes)) {
    if (auto origT = dyn_cast<RankedTensorType>(orig)) {
      auto argT = dyn_cast<RankedTensorType>(arg);
      if (auto result = verifySlice(origT, argT))
        return result;
    } else if (auto origT = dyn_cast<triton::gpu::MemDescType>(orig)) {
      auto argT = dyn_cast<triton::gpu::MemDescType>(arg);
      if (auto result = verifySlice(origT, argT))
        return result;
    } else {
      return "Slicing not Implemented for this type";
    }
  }
  return std::nullopt;
}

LogicalResult ArefPutEnterOp::verify() {
  if (auto result = arefEnterVerify(getAref().getType(), getResultTypes()))
    return emitError(*result);
  return success();
}

LogicalResult ArefGetEnterOp::verify() {
  if (auto result = arefEnterVerify(getAref().getType(), getResultTypes()))
    return emitError(*result);
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

void CreateTokenOp::build(::mlir::OpBuilder &builder,
                          ::mlir::OperationState &state, uint32_t num,
                          TokenLoadType loadType) {
  auto tokenType = TokenType::get(builder.getContext());
  auto resultType = RankedTensorType::get({num}, tokenType);
  build(builder, state, resultType, num, loadType);
}

} // namespace mlir::triton::nvws
