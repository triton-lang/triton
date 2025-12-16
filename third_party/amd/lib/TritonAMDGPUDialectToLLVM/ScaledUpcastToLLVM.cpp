#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/PatternTritonAMDGPUToLLVM.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using mlir::LLVM::AMD::upcast4xMxfp8_HW;
using mlir::LLVM::AMD::upcast8xMxfp4_HW;
using mlir::LLVM::AMD::upcast8xMxfp8fp4_HW;

// TODO: using if-then-else to repalce ternary operator on template
namespace {
struct ScaledUpcastFp4OpPattern
    : ConvertOpToLLVMPattern<amdgpu::ScaledUpcastFp4Op> {

  ScaledUpcastFp4OpPattern(const LLVMTypeConverter &converter,
                           const AMD::TargetInfo &targetInfo,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

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
    if (targetInfo.getISAFamily() == AMD::ISAFamily::GFX1250) {
      assert(scaleVals.size() == 2 * inputVals.size());
      for (int i = 0; i < inputVals.size(); i += 4) {

        const auto &converted =
            elemType.isF16()
                ? upcast8xMxfp8fp4_HW<ROCDL::CvtPkScalePk8F16Fp4Op>(
                      rewriter, loc, inputVals, i, scaleVals)
                : upcast8xMxfp8fp4_HW<ROCDL::CvtPkScalePk8Bf16Fp4Op>(
                      rewriter, loc, inputVals, i, scaleVals);

        results.append(converted.begin(), converted.end());
      }
    } else {
      for (int i = 0; i < inputVals.size(); i += 4) {
        SmallVector<Value, 4> v4i32 =
            elemType.isF16()
                ? upcast8xMxfp4_HW<ROCDL::CvtScaleF32PkF16Fp4Op>(
                      rewriter, loc, inputVals, i, scaleVals[i * 2],
                      /*useShiftedScale=*/true)
                : upcast8xMxfp4_HW<ROCDL::CvtScaleF32PkBf16Fp4Op>(
                      rewriter, loc, inputVals, i, scaleVals[i * 2],
                      /*useShiftedScale=*/true);
        for (int j = 0; j < 4; j++) {
          Value elements = b.bitcast(v4i32[j], vec_ty(elemType, 2));
          results.push_back(b.extract_element(elements, b.i32_val(0)));
          results.push_back(b.extract_element(elements, b.i32_val(1)));
        }
      }
    }

    Value result = packLLElements(loc, getTypeConverter(), results, rewriter,
                                  upcastOp.getType());
    rewriter.replaceOp(upcastOp, result);
    return success();
  }

  const AMD::TargetInfo &targetInfo;
};

struct ScaledUpcastFp8OpPattern
    : ConvertOpToLLVMPattern<amdgpu::ScaledUpcastFp8Op> {

  ScaledUpcastFp8OpPattern(const LLVMTypeConverter &converter,
                           const AMD::TargetInfo &targetInfo,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(amdgpu::ScaledUpcastFp8Op upcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = upcastOp.getLoc();
    auto elemType = upcastOp.getType().getElementType();
    auto fp8ElemType = upcastOp.getInput().getType().getElementType();

    auto inputVals = unpackLLElements(loc, adaptor.getInput(), rewriter);
    auto scaleVals = unpackLLElements(loc, adaptor.getScale(), rewriter);

    assert(inputVals.size() % 4 == 0);
    assert(inputVals.size() == scaleVals.size());

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto isaFamily = targetInfo.getISAFamily();
    SmallVector<Value> results;
    results.reserve(inputVals.size());
    if (isaFamily == AMD::ISAFamily::GFX1250) {
      for (int i = 0; i < inputVals.size(); i += 8) {
        const auto &converted =
            elemType.isF16()
                ? (isa<Float8E4M3FNType>(fp8ElemType)
                       ? upcast8xMxfp8fp4_HW<ROCDL::CvtPkScalePk8F16Fp8Op>(
                             rewriter, loc, inputVals, i, scaleVals)
                       : upcast8xMxfp8fp4_HW<ROCDL::CvtPkScalePk8F16Bf8Op>(
                             rewriter, loc, inputVals, i, scaleVals))
                : (isa<Float8E4M3FNType>(fp8ElemType)
                       ? upcast8xMxfp8fp4_HW<ROCDL::CvtPkScalePk8Bf16Fp8Op>(
                             rewriter, loc, inputVals, i, scaleVals)
                       : upcast8xMxfp8fp4_HW<ROCDL::CvtPkScalePk8Bf16Bf8Op>(
                             rewriter, loc, inputVals, i, scaleVals));

        results.append(converted.begin(), converted.end());
      }
    } else {
      for (int i = 0; i < inputVals.size(); i += 4) {
        SmallVector<Value, 2> v2i32 =
            elemType.isF16()
                ? (isa<Float8E4M3FNType>(fp8ElemType)
                       ? upcast4xMxfp8_HW<ROCDL::CvtScaleF32PkF16Fp8Op>(
                             rewriter, loc, inputVals, i, scaleVals[i],
                             /*useShiftedScale=*/true)
                       : upcast4xMxfp8_HW<ROCDL::CvtScaleF32PkF16Bf8Op>(
                             rewriter, loc, inputVals, i, scaleVals[i],
                             /*useShiftedScale=*/true))
                : (isa<Float8E4M3FNType>(fp8ElemType)
                       ? upcast4xMxfp8_HW<ROCDL::CvtScaleF32PkBf16Fp8Op>(
                             rewriter, loc, inputVals, i, scaleVals[i],
                             /*useShiftedScale=*/true)
                       : upcast4xMxfp8_HW<ROCDL::CvtScaleF32PkBf16Bf8Op>(
                             rewriter, loc, inputVals, i, scaleVals[i],
                             /*useShiftedScale=*/true));
        for (int j = 0; j < 2; j++) {
          Value elements = b.bitcast(v2i32[j], vec_ty(elemType, 2));
          results.push_back(b.extract_element(elements, b.i32_val(0)));
          results.push_back(b.extract_element(elements, b.i32_val(1)));
        }
      }
    }

    Value result = packLLElements(loc, getTypeConverter(), results, rewriter,
                                  upcastOp.getType());
    rewriter.replaceOp(upcastOp, result);
    return success();
  }

  const AMD::TargetInfo &targetInfo;
};
} // anonymous namespace

void mlir::triton::AMD::populateScaledUpcastOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const AMD::TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<ScaledUpcastFp4OpPattern>(typeConverter, targetInfo, benefit);
  patterns.add<ScaledUpcastFp8OpPattern>(typeConverter, targetInfo, benefit);
}
