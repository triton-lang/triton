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
  auto *ctx = op.getContext();
  auto kRegister = str_attr("register");
  auto srcTy = op.getSrc().getType();
  auto resTy = op.getType();
  auto elemType = resTy.getElementType();
  assert(elemType == f16_ty || elemType == bf16_ty);

  auto xVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);

  SmallVector<Value> results;
  results.reserve(xVals.size() * 2);
  assert(xVals.size() % 4 == 0);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  for (int i = 0; i < xVals.size(); i += 4) {
    Value packedVec = b.undef(vec_ty(i8_ty, 4));
    for (int j = 0; j < 4; ++j)
      packedVec = b.insert_element(packedVec, xVals[i + j], b.i32_val(j));
    auto upcast = upcastPackedFp4(op, rewriter, packedVec, elemType);
    results.append(upcast.begin(), upcast.end());
  }

  auto srcLayout = toLinearLayout(srcTy);
  auto axisDim = *(srcLayout.getOutDimNames().begin() + op.getAxis());
  // Expand the source layout to reflect the unpacked elements in results.
  auto fullSrcLayout =
      LinearLayout::identity1D(2, kRegister, axisDim) * srcLayout;
  // Create a mapping to get the source location associated with each result.
  auto resToFullSrc = toLinearLayout(resTy)
                          .invertAndCompose(fullSrcLayout)
                          .sublayout({kRegister}, {kRegister});

  // Apply the mapping to get the final result.
  SmallVector<Value> mappedResults(results.size());
  for (int i = 0; i < results.size(); ++i) {
    auto srcIndex = resToFullSrc.apply({{kRegister, i}}).front().second;
    mappedResults[i] = results[srcIndex];
  }
  Value result =
      packLLElements(loc, getTypeConverter(), mappedResults, rewriter, resTy);
  rewriter.replaceOp(op, result);
  return success();
}

} // namespace mlir::triton::gpu
