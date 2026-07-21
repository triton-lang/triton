#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/PatternTritonAMDGPUToLLVM.h"
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

// For each v_cvt_scale_pk8 group of 8 fp4 (4 consecutive input bytes) that a
// thread holds, return the index into the thread's scale registers of the scale
// to apply to that group.
SmallVector<int> computeFp4GroupScaleRegisters(amdgpu::ScaledUpcastFp4Op op,
                                               int64_t numInputBytes) {
  MLIRContext *ctx = op.getContext();
  auto outTy = op.getType();
  auto scaleTy = op.getScale().getType();
  int64_t axis = op.getAxis();
  int64_t elementsPerScale = outTy.getShape()[axis] / scaleTy.getShape()[axis];

  LinearLayout outLL = triton::gpu::toLinearLayout(outTy);
  LinearLayout scaleLL = triton::gpu::toLinearLayout(scaleTy);
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kBlock = StringAttr::get(ctx, "block");

  auto outToScaleLL = amdgpu::ScaledUpcastFp4Op::computeScaleLayout(
      outLL, axis, elementsPerScale);
  assert(outToScaleLL && "expected valid scale layout after verifier");
  LinearLayout outRegToScaleReg = outToScaleLL->invertAndCompose(scaleLL);

  // One intrinsic spans 8 fp4 values
  int64_t numGroups = (2 * numInputBytes) / 8;
  SmallVector<int> groupScaleReg;
  groupScaleReg.reserve(numGroups);
  for (int g = 0; g < numGroups; ++g) {
    auto scaleCoord =
        outRegToScaleReg.apply({{kReg, static_cast<int32_t>(8 * g)},
                                {kLane, 0},
                                {kWarp, 0},
                                {kBlock, 0}});
    auto regIt = llvm::find_if(
        scaleCoord, [&](const auto &dimVal) { return dimVal.first == kReg; });
    assert(regIt != scaleCoord.end() && "scale register mapping missing");
    groupScaleReg.push_back(regIt->second);
  }
  return groupScaleReg;
}

// Returns true if the scale layout gives lane `j` and lane `j^16` the same
// scale value (i.e. the lane basis for the value-16 bit is all zeros). In that
// case both output-lane halves of a v_cvt_scale_pk8 already share a scale.
bool isScaleLane16Broadcast(RankedTensorType scaleTy) {
  MLIRContext *ctx = scaleTy.getContext();
  LinearLayout scaleLL = triton::gpu::toLinearLayout(scaleTy);
  auto kLane = StringAttr::get(ctx, "lane");
  // Lane value 16 corresponds to basis position log2(16) = 4.
  constexpr int kLane16Pos = 4;
  if (scaleLL.getInDimSizeLog2(kLane) <= kLane16Pos)
    return true;
  ArrayRef<int32_t> basis16 = scaleLL.getBasis(kLane, kLane16Pos);
  return llvm::all_of(basis16, [](int32_t v) { return v == 0; });
}

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

    auto inputVals =
        unpackUniqueTensorElements(loc, adaptor.getInput(), rewriter);
    auto scaleVals =
        unpackUniqueTensorElements(loc, adaptor.getScale(), rewriter);

    assert(inputVals.size() % 4 == 0);
    SmallVector<Value> results;
    results.reserve(inputVals.size() * 2);

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    if (targetInfo.supportsCvtPkScalePk8()) {
      auto groupScaleReg =
          computeFp4GroupScaleRegisters(upcastOp, inputVals.size());

      // FP4/FP6 v_cvt_scale_pk8 with opSel=0 sources the scale for output
      // lanes 16..31 from byte 1 of the *lower* 16 lanes' Vscale while output
      // lanes 0..15 use byte 0. When the scale layout is not broadcast across
      // the lane^16 split, lane j and lane j+16 carry different scales, so we
      // must co-locate lane (j^16)'s scale into byte 1 of lane j via a
      // cross-lane exchange. Broadcast (e.g. wmma) layouts already share the
      // scale, so we skip the exchange there.
      // TODO: we could also check if lane0 holds the scale lane16 requires to
      // avoid the v_permlane16_swap.
      bool broadcast = isScaleLane16Broadcast(upcastOp.getScale().getType());
      SmallVector<Value> crossScaleVals(scaleVals.size());
      auto getCrossScale = [&](int scaleIdx) -> Value {
        if (broadcast)
          return Value();
        if (!crossScaleVals[scaleIdx]) {
          Value s32 = b.zext(i32_ty, scaleVals[scaleIdx]);
          // Will be lowered to v_permlane16_swap so it's quite cheap.
          Value partner = targetInfo.shuffleXor(rewriter, loc, s32, 16);
          crossScaleVals[scaleIdx] = b.trunc(i8_ty, partner);
        }
        return crossScaleVals[scaleIdx];
      };

      for (int i = 0; i < inputVals.size(); i += 4) {
        int scaleIdx = groupScaleReg[i / 4];
        Value crossScale = getCrossScale(scaleIdx);

        const auto &converted =
            elemType.isF16()
                ? upcast8xMxfp8fp4_HW<ROCDL::CvtPkScalePk8F16Fp4Op>(
                      rewriter, loc, inputVals, i, scaleVals, scaleIdx,
                      crossScale)
                : upcast8xMxfp8fp4_HW<ROCDL::CvtPkScalePk8Bf16Fp4Op>(
                      rewriter, loc, inputVals, i, scaleVals, scaleIdx,
                      crossScale);

        results.append(converted.begin(), converted.end());
      }
    } else if (targetInfo.supportsHwScaledUpcast()) {
      assert(scaleVals.size() == 2 * inputVals.size() &&
             "compact fp4 scales are only supported on pk scale path");
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
      assert(scaleVals.size() == 2 * inputVals.size() &&
             "compact fp4 scales are only supported on pk scale path");
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

    Value result = packUniqueTensorElements(loc, getTypeConverter(), results,
                                            rewriter, upcastOp.getType());
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

    auto inputVals =
        unpackUniqueTensorElements(loc, adaptor.getInput(), rewriter);
    auto scaleVals =
        unpackUniqueTensorElements(loc, adaptor.getScale(), rewriter);

    assert(inputVals.size() % 4 == 0);
    assert(inputVals.size() == scaleVals.size());

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value> results;
    results.reserve(inputVals.size());
    if (targetInfo.supportsCvtPkScalePk8()) {
      // Broadcast layouts can use FP8 Block32 (opSel=0). Otherwise use Block16
      // (opSel=8) and pack lane j^16's scale into byte 1.
      bool broadcast = isScaleLane16Broadcast(upcastOp.getScale().getType());
      SmallVector<Value> crossScaleVals(scaleVals.size());
      auto getCrossScale = [&](int scaleIdx) -> Value {
        if (broadcast)
          return Value();
        if (!crossScaleVals[scaleIdx]) {
          Value s32 = b.zext(i32_ty, scaleVals[scaleIdx]);
          Value partner = targetInfo.shuffleXor(rewriter, loc, s32, 16);
          crossScaleVals[scaleIdx] = b.trunc(i8_ty, partner);
        }
        return crossScaleVals[scaleIdx];
      };
      for (int i = 0; i < inputVals.size(); i += 8) {
        Value crossScale = getCrossScale(i);
        const auto &converted =
            elemType.isF16()
                ? (isa<Float8E4M3FNType>(fp8ElemType)
                       ? upcast8xMxfp8fp4_HW<ROCDL::CvtPkScalePk8F16Fp8Op>(
                             rewriter, loc, inputVals, i, scaleVals, i,
                             crossScale)
                       : upcast8xMxfp8fp4_HW<ROCDL::CvtPkScalePk8F16Bf8Op>(
                             rewriter, loc, inputVals, i, scaleVals, i,
                             crossScale))
                : (isa<Float8E4M3FNType>(fp8ElemType)
                       ? upcast8xMxfp8fp4_HW<ROCDL::CvtPkScalePk8Bf16Fp8Op>(
                             rewriter, loc, inputVals, i, scaleVals, i,
                             crossScale)
                       : upcast8xMxfp8fp4_HW<ROCDL::CvtPkScalePk8Bf16Bf8Op>(
                             rewriter, loc, inputVals, i, scaleVals, i,
                             crossScale));

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

    Value result = packUniqueTensorElements(loc, getTypeConverter(), results,
                                            rewriter, upcastOp.getType());
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
