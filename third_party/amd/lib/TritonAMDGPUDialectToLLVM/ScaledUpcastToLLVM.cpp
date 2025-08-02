#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/PatternTritonAMDGPUToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

template <typename ConvertOp>
SmallVector<Value, 4> upcast8xMxfp4_HW(RewriterBase &rewriter, Location loc,
                                       ArrayRef<Value> xVals, int idx,
                                       Value scale) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value packedVec = b.undef(vec_ty(i8_ty, 4));
  for (int i : llvm::seq(4))
    packedVec = b.insert_element(packedVec, xVals[idx + i], b.i32_val(i));
  packedVec = b.bitcast(packedVec, i32_ty);
  Type retElemType = bf16_ty;
  if constexpr (std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkF16Fp4Op>)
    retElemType = f16_ty;
  Type resType = vec_ty(retElemType, 2);
  Value scaleF32 = b.bitcast(
      b.shl(b.zext(i32_ty, b.bitcast(scale, i16_ty)), b.i32_val(16)), f32_ty);
  SmallVector<Value, 4> results;
  // Intentionally swap the byte indices 1 and 2 to align with how the LLVM
  // backend accesses them
  for (int srcSelIndex : {0, 2, 1, 3})
    results.push_back(rewriter.create<ConvertOp>(loc, resType, packedVec,
                                                 scaleF32, srcSelIndex));
  return results;
}

struct ScaledUpcastFp4Pattern
    : ConvertOpToLLVMPattern<amdgpu::ScaledUpcastFp4Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(amdgpu::ScaledUpcastFp4Op upcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = upcastOp.getLoc();
    auto elemType = upcastOp.getType().getElementType();

    auto inputVals = unpackLLElements(loc, adaptor.getInput(), rewriter);
    auto scaleVals = unpackLLElements(loc, adaptor.getScale(), rewriter);

    assert(inputVals.size() % 4 == 0);
    SmallVector<Value> results;
    results.reserve(inputVals.size() * 2);

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    for (int i = 0; i < inputVals.size(); i += 4) {
      SmallVector<Value, 4> v4i32 =
          upcast8xMxfp4_HW<ROCDL::CvtScaleF32PkBf16Fp4Op>(
              rewriter, loc, inputVals, i, scaleVals[i * 2]);
      for (int j = 0; j < 4; j++) {
        Value elements = b.bitcast(v4i32[j], vec_ty(elemType, 2));
        results.push_back(b.extract_element(elements, b.i32_val(0)));
        results.push_back(b.extract_element(elements, b.i32_val(1)));
      }
    }

    Value result = packLLElements(loc, getTypeConverter(), results, rewriter,
                                  upcastOp.getType());
    rewriter.replaceOp(upcastOp, result);
    return success();
  }
};
} // anonymous namespace

void mlir::triton::AMD::populateScaledUpcastOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ScaledUpcastFp4Pattern>(typeConverter, benefit);
}
