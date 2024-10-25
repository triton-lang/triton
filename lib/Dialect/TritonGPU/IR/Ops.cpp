#include "mlir/IR/BuiltinTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Types.h"
#include "llvm/Support/raw_ostream.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonGPU/IR/Ops.cpp.inc"

namespace mlir::triton::gpu {

LogicalResult UpcastMXFPOp::verify() {
  auto fpType = getFpType();

  auto xTy = getSrc().getType();
  auto scaleTy = getScale().getType();

  if (xTy.getElementType() != FloatType::getBF16(getContext()) &&
      xTy.getElementType() != IntegerType::get(getContext(), 8)) {
    return emitOpError("element type of the first operand must be bf16 or i8");
  }

  if (scaleTy.getElementType() != IntegerType::get(getContext(), 8)) {
    return emitOpError("element type of the second operand must be uint8");
  }

  auto xShape = xTy.getShape();
  auto scaleShape = scaleTy.getShape();

  if (xShape.size() != scaleShape.size() || xShape.size() < 2) {
    return emitOpError(
        "operands must have the same number of dimensions, at least 2");
  }

  if (!(fpType == ScaleType::E2M1 || fpType == ScaleType::E4M3 ||
        fpType == ScaleType::E5M2)) {
    return emitOpError("NYI: fpType must be E2M1, E4M3, or E5M2");
  }

  // Change to support fp8 types
  const auto elems_packed = fpType == ScaleType::E2M1 ? 2 : 1;

  if (xShape.back() != (32 / elems_packed) * scaleShape.back()) {
    return emitOpError("last dimension of first operand must be 16 times "
                       "larger than that of the second operand");
  }

  if (!std::equal(xShape.begin(), xShape.end() - 1, scaleShape.begin())) {
    return emitOpError(
        "all dimensions except the last must match between operands");
  }

  auto dotEncoding =
      dyn_cast_or_null<DotOperandEncodingAttr>(xTy.getEncoding());
  if (!dotEncoding) {
    return emitOpError("Expected a DotOperandEncodingAttr for values");
  }

  auto blockedScale =
      dyn_cast_or_null<BlockedEncodingAttr>(scaleTy.getEncoding());
  if (!blockedScale) {
    return emitOpError("Expected a BlockOperandEncoding for scales");
  }

  if (isa<NvidiaMmaEncodingAttr>(dotEncoding.getParent())) {
    // Necessary to keep all of the scales of a given block of values in the
    // same warp
    auto threadsPerWarp = blockedScale.getThreadsPerWarp();
    if (threadsPerWarp != ArrayRef<unsigned>({16, 2})) {
      return emitOpError("Expected threads per warp to be {16, 2}");
    }
  }

  return success();
}

LogicalResult UpcastMXFPOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties opaqueProperties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto xTy = cast<RankedTensorType>(operands[0].getType());
  auto properties = opaqueProperties.as<const Properties *>();
  auto typeEncoded = properties->fp_type.getValue();
  auto xShape = xTy.getShape();

  auto encoding = xTy.getEncoding();
  if (!encoding) {
    return emitOptionalError(loc, "expected an encoding");
  }
  if (!mlir::isa<DotOperandEncodingAttr>(encoding)) {
    return emitOptionalError(loc, "expected a dotOperand encoding");
  }

  if (typeEncoded == ScaleType::E2M1) {
    auto oldEncoding = cast<DotOperandEncodingAttr>(encoding);
    auto newVEncoding = DotOperandEncodingAttr::get(
        ctx, oldEncoding.getOpIdx(), oldEncoding.getParent(),
        oldEncoding.getKWidth() * 2);
    auto newShape = SmallVector<int64_t>(xShape);
    newShape.back() *= 2;
    inferredReturnTypes.push_back(
        RankedTensorType::get(newShape, FloatType::getBF16(ctx), newVEncoding));
  } else {
    inferredReturnTypes.push_back(xTy);
  }

  return success();
}

} // namespace mlir::triton::gpu
