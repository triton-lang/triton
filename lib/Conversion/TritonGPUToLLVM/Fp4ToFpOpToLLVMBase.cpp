#include "triton/Conversion/TritonGPUToLLVM/Fp4ToFpOpToLLVMBase.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::gpu {

Fp4ToFpOpConversionBase::Fp4ToFpOpConversionBase(
    LLVMTypeConverter &typeConverter, PatternBenefit benefit)
    : ConvertOpToLLVMPattern<Fp4ToFpOp>(typeConverter, benefit) {}

LogicalResult Fp4ToFpOpConversionBase::matchAndRewrite(
    Fp4ToFpOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto resTy = op.getType();
  auto elemType = resTy.getElementType();
  assert(elemType == f16_ty || elemType == bf16_ty);

  auto xVals = unpackUniqueTensorElements(loc, adaptor.getSrc(), rewriter);

  SmallVector<Value> results;
  results.reserve(xVals.size() * 2);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  for (int i = 0; i < xVals.size(); i += 4) {
    unsigned numBytes = std::min<size_t>(4, xVals.size() - i);
    Value packedVec = b.undef(vec_ty(i8_ty, 4));
    // Use zero instead of undef for unused bytes to simplify later
    // optimizations.
    for (unsigned j = 0; j < 4; ++j) {
      Value byte = j < numBytes ? xVals[i + j] : b.i8_val(0);
      packedVec = b.insert_element(packedVec, byte, b.i32_val(j));
    }
    auto upcast = upcastPackedFp4(op, rewriter, packedVec, elemType);
    results.append(upcast.begin(), upcast.begin() + 2 * numBytes);
  }

  Value result = packUniqueTensorElements(loc, getTypeConverter(), results,
                                          rewriter, resTy);
  rewriter.replaceOp(op, result);
  return success();
}

} // namespace mlir::triton::gpu
