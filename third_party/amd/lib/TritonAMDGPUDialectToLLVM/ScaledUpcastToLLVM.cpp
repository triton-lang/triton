#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/PatternTritonAMDGPUToLLVM.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using mlir::LLVM::AMD::convertF8ToF32_SW;
using mlir::LLVM::AMD::upcast4xMxfp8_HW;
using mlir::LLVM::AMD::upcast8xMxfp4_HW;
using mlir::LLVM::AMD::upcast8xMxfp4_SW;
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
    if (targetInfo.supportsCvtPkScalePk8()) {
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
    } else if (targetInfo.supportsHwScaledUpcast()) {
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
    } else {
      // Software emulation: upcast fp4 via LUT, then multiply by scale.
      bool toFp16 = elemType.isF16();
      auto isaFamily = targetInfo.getISAFamily();
      for (size_t i = 0; i < inputVals.size(); i += 4) {
        Value packedVec = b.undef(vec_ty(i8_ty, 4));
        for (int j : llvm::seq(4))
          packedVec =
              b.insert_element(packedVec, inputVals[i + j], b.i32_val(j));

        SmallVector<Value> v8vals =
            upcast8xMxfp4_SW(rewriter, upcastOp, toFp16, packedVec, isaFamily);

        // The bf16 scale was left-shifted by 7 (scaleTo16); shift by 16 more
        // to get f32.
        Value scaleBf16 = scaleVals[i * 2];
        Value scaleF32 = b.bitcast(
            b.shl(b.zext(i32_ty, b.bitcast(scaleBf16, i16_ty)), b.i32_val(16)),
            f32_ty);

        for (int j : llvm::seq(8)) {
          Value vF32;
          if (toFp16) {
            vF32 = b.fpext(f32_ty, v8vals[j]);
          } else {
            // bf16 → f32 via bit manipulation (gfx9 lacks native bf16 VALU)
            vF32 = b.bitcast(b.shl(b.zext(i32_ty, b.bitcast(v8vals[j], i16_ty)),
                                   b.i32_val(16)),
                             f32_ty);
          }
          Value mulF32 = b.fmul(vF32, scaleF32);
          if (toFp16) {
            results.push_back(b.fptrunc(f16_ty, mulF32));
          } else {
            Value mulI16 = b.trunc(
                i16_ty, b.lshr(b.bitcast(mulF32, i32_ty), b.i32_val(16)));
            results.push_back(b.bitcast(mulI16, bf16_ty));
          }
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
    SmallVector<Value> results;
    results.reserve(inputVals.size());
    if (targetInfo.supportsCvtPkScalePk8()) {
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
    } else if (targetInfo.supportsHwScaledUpcast()) {
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
    } else {
      // Software emulation: convert fp8 to f32, then multiply by scale.
      bool isE4M3FN = isa<Float8E4M3FNType>(fp8ElemType);
      bool toFp16 = elemType.isF16();
      for (size_t i = 0; i < inputVals.size(); i += 4) {
        // The bf16 scale was left-shifted by 7 (scaleTo16); shift by 16 more
        // to get f32.
        Value scaleBf16 = scaleVals[i];
        Value scaleF32 = b.bitcast(
            b.shl(b.zext(i32_ty, b.bitcast(scaleBf16, i16_ty)), b.i32_val(16)),
            f32_ty);

        for (int j : llvm::seq(4)) {
          Value f32Val =
              convertF8ToF32_SW(rewriter, loc, inputVals[i + j], isE4M3FN);
          Value mulF32 = b.fmul(f32Val, scaleF32);
          if (toFp16) {
            results.push_back(b.fptrunc(f16_ty, mulF32));
          } else {
            Value mulI16 = b.trunc(
                i16_ty, b.lshr(b.bitcast(mulF32, i32_ty), b.i32_val(16)));
            results.push_back(b.bitcast(mulI16, bf16_ty));
          }
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
