#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/PatternTritonAMDGPUToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using mlir::LLVM::AMD::convertF8ToF32_SW;
using mlir::LLVM::AMD::upcast8xMxfp4_SW;

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
  scaleLL = scaleLL.removeZeroBasesAlongDim(kReg);

  // One intrinsic spans 8 fp4 values
  int64_t numGroups = (2 * numInputBytes) / 8;
  SmallVector<int> groupScaleReg;
  groupScaleReg.reserve(numGroups);
  for (int g = 0; g < numGroups; ++g) {
    auto scaleCoord = outToScaleLL->apply({{kReg, static_cast<int32_t>(8 * g)},
                                           {kLane, 0},
                                           {kWarp, 0},
                                           {kBlock, 0}});
    auto scaleReg = AMD::getRegFromCoordinates(scaleLL, scaleCoord, ctx);
    assert(scaleReg && "scale register mapping missing");
    groupScaleReg.push_back(*scaleReg);
  }
  return groupScaleReg;
}

bool scaleIsPreShifted(RankedTensorType scaleTy) {
  return scaleTy.getElementType().isBF16();
}

// Software multiplication needs the numeric scale, unlike hardware conversions
// that read only its exponent field.
Value scaleToF32(RewriterBase &rewriter, Location loc, Value scale,
                 bool preShifted) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value value;
  if (preShifted) {
    value = b.bitcast(
        b.shl(b.zext(i32_ty, b.bitcast(scale, i16_ty)), b.i32_val(16)), f32_ty);
  } else {
    Value bits = b.shl(b.zext(i32_ty, scale), b.i32_val(23));
    value = b.bitcast(b.umax(bits, b.i32_val(0x00400000)), f32_ty);
  }
  return b.fma(value, b.f32_val(0.0), value);
}

// Packs the 4 i8 values starting at `vals[idx]` into an i32.
Value packI8x4ToI32(RewriterBase &rewriter, Location loc, ArrayRef<Value> vals,
                    int idx) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value packedVec = b.undef(vec_ty(i8_ty, 4));
  for (int i : llvm::seq(4))
    packedVec = b.insert_element(packedVec, vals[idx + i], b.i32_val(i));
  return b.bitcast(packedVec, i32_ty);
}

// Hardware scaled conversions read only the exponent field of the f32 scale.
// A pre-shifted (bf16) scale has already been left-shifted by 7 in the
// DotScaledOp decomposition to fit the bf16 exponent, so it only needs a
// further left shift by 16.
Value scaleExponentToF32(RewriterBase &rewriter, Location loc, Value scale,
                         bool preShifted) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (preShifted)
    return b.bitcast(
        b.shl(b.zext(i32_ty, b.bitcast(scale, i16_ty)), b.i32_val(16)), f32_ty);
  return b.bitcast(b.shl(b.zext(i32_ty, scale), b.i32_val(23)), f32_ty);
}

// Splits each packed 2-element conversion result into scalar values.
SmallVector<Value> unpackPairs(RewriterBase &rewriter, Location loc,
                               ArrayRef<Value> pairs, Type elemType) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> results;
  for (Value pair : pairs) {
    Value elements = b.bitcast(pair, vec_ty(elemType, 2));
    results.push_back(b.extract_element(elements, b.i32_val(0)));
    results.push_back(b.extract_element(elements, b.i32_val(1)));
  }
  return results;
}

// Upcasts the 8 fp4 values packed in the 4 bytes starting at `xVals[idx]`
// with v_cvt_scalef32_pk_{f16,bf16}_fp4.
template <typename ConvertOp>
SmallVector<Value> upcast8xMxfp4_HW(RewriterBase &rewriter, Location loc,
                                    ArrayRef<Value> xVals, int idx, Value scale,
                                    bool preShifted) {
  Value packedVec = packI8x4ToI32(rewriter, loc, xVals, idx);
  Type elemType = std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkF16Fp4Op>
                      ? Type(f16_ty)
                      : Type(bf16_ty);
  Value scaleF32 = scaleExponentToF32(rewriter, loc, scale, preShifted);
  SmallVector<Value, 4> pairs;
  for (int srcSelIndex : llvm::seq(4))
    pairs.push_back(ConvertOp::create(rewriter, loc, vec_ty(elemType, 2),
                                      packedVec, scaleF32, srcSelIndex));
  return unpackPairs(rewriter, loc, pairs, elemType);
}

// Upcasts the 4 fp8 values starting at `xVals[idx]` with
// v_cvt_scalef32_pk_{f16,bf16}_{fp8,bf8}.
template <typename ConvertOp>
SmallVector<Value> upcast4xMxfp8_HW(RewriterBase &rewriter, Location loc,
                                    ArrayRef<Value> xVals, int idx, Value scale,
                                    bool preShifted) {
  Value packedVec = packI8x4ToI32(rewriter, loc, xVals, idx);
  Type elemType =
      std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkF16Fp8Op> ||
              std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkF16Bf8Op>
          ? Type(f16_ty)
          : Type(bf16_ty);
  Value scaleF32 = scaleExponentToF32(rewriter, loc, scale, preShifted);
  SmallVector<Value, 2> pairs;
  for (bool srcLoHiSel : {false, true})
    pairs.push_back(ConvertOp::create(rewriter, loc, vec_ty(elemType, 2),
                                      packedVec, scaleF32, srcLoHiSel));
  return unpackPairs(rewriter, loc, pairs, elemType);
}

// 1) for the parameter `inputVals`
// The fp8 tensor `inputVals` is upcasted to a [b]f16 tensor in the same shape,
// as an operand of 16x16x32_[b]f16 WMMA instruction and the layout is:
// clang-format off
//
// --------------------------------------------------------------------------------------------------------------
// \Row    0,1   2,3   4,5   6,7  |  8,9  10,11  12,13 14,15 | 16,17 18,19 20,21 22,23 | 24,25 26,27  28,29 30,31
// \__
// Col                            |                          |                         |
// 0      t0r0  t0r1  t0r2  t0r3  | t16r0 t16r1  t16r2 t16r3 | t0r4  t0r5  t0r6  t0r7  | t16r4 t16r5  t16r6 t16r7
// 1      t1r0  t1r1  t1r2  t1r3  | t17r0 t17r1  t17r2 t17r3 | t1r4  t1r5  t1r6  t1r7  | t17r4 t17r5  t17r6 t17r7
// ...                            |                           ...... .....
// 15     t15r0 t15r1 t15r2 t15r3 | t31r0 t31r1  t31r2 t31r3 | t15r4 t15r5 t15r6 t15r7 | t31r4 t31r5  t31r6 t31r7
// --------------------------------------------------------------------------------------------------------------
//
// clang-format on

// The points here are:
// Lane and lane+16 co-hold one row
// Input tensor of upcast `inputVals` is with same layout yet element type is
// fp8;
//
// 2) for the parameter `scales`
//   For scale tensor, e.g. if input shape is (32, 4) and block mode is 32,
// it is already transformed via `reshape(broadcast_to(expand_dims(a_scale, 2),
// (32, 4, 32)), (32, 128))` and output layout in the wave is `register = [[0,
// 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[0, 32], [0, 64], [1, 0], [2,
// 0], [4, 0]]` which means every lane will hold continous 32 elements and these
// 32 elements share one scale since the block mode is 32.
//
// 3) for `opSel` used in the rocdl.cvt.scale.pk8
//
// Scale selection for v_cvt_scale_pk8 is driven by opSel and four Vscale bytes.
// For the modes used here, byte 0 holds the local lane's scale. For F4/F6
// opSel=0, lanes 0..15 read byte 0 and lanes 16..31 read byte 1, so
// crossLaneScale (when present) is packed into byte 1. For F8, opSel=0 uses
// byte 0 for both lane halves when the scale layout is lane^16-broadcast;
// otherwise opSel=8 selects Block16 mode and byte 1 supplies the lane^16 scale.
//
template <typename ConvertOp>
SmallVector<Value, 8> upcast8xMxfp8fp4_HW(RewriterBase &rewriter, Location loc,
                                          ArrayRef<Value> inputVals, int idx,
                                          ArrayRef<Value> scales, int scaleIdx,
                                          Value crossLaneScale = Value()) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  bool toFp16 = (std::is_same_v<ConvertOp, ROCDL::CvtPkScalePk8F16Fp8Op> ||
                 std::is_same_v<ConvertOp, ROCDL::CvtPkScalePk8F16Bf8Op> ||
                 std::is_same_v<ConvertOp, ROCDL::CvtPkScalePk8F16Fp4Op>);
  Type resElemType = toFp16 ? f16_ty : bf16_ty;
  Type resType = vec_ty(resElemType, 8);

  bool fromFP4 = (std::is_same_v<ConvertOp, ROCDL::CvtPkScalePk8F16Fp4Op> ||
                  std::is_same_v<ConvertOp, ROCDL::CvtPkScalePk8Bf16Fp4Op>);

  auto packedSize = fromFP4 ? 4 : 8;
  Value packedVec = b.undef(vec_ty(i8_ty, packedSize));
  for (int ii : llvm::seq(packedSize))
    packedVec = b.insert_element(packedVec, inputVals[idx + ii], b.i32_val(ii));
  packedVec =
      fromFP4 ? b.bitcast(packedVec, i32_ty)
              : b.bitcast(packedVec, vec_ty(i32_ty, packedSize / sizeof(int)));

  assert(scaleIdx >= 0 && scaleIdx < static_cast<int>(scales.size()));
  Value localScale = scales[scaleIdx];
  Value partnerScale = crossLaneScale ? crossLaneScale : localScale;
  unsigned scaleSel = !fromFP4 && crossLaneScale ? 8 : 0;
  Value packedScale = b.undef(vec_ty(i8_ty, 4));
  packedScale = b.insert_element(packedScale, localScale, b.i32_val(0));
  if (fromFP4 || crossLaneScale)
    packedScale = b.insert_element(packedScale, partnerScale, b.i32_val(1));
  Value scaleInt32 = b.bitcast(packedScale, i32_ty);
  auto res =
      ConvertOp::create(rewriter, loc, resType, packedVec, scaleInt32, scaleSel)
          .getRes();
  Value elements = b.bitcast(res, vec_ty(resElemType, 8));

  SmallVector<Value, 8> results;
  for (auto ii : llvm::seq(8)) {
    results.push_back(b.extract_element(elements, b.i32_val(ii)));
  }

  return results;
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
    bool preShifted = scaleIsPreShifted(upcastOp.getScale().getType());
    auto groupScaleReg =
        computeFp4GroupScaleRegisters(upcastOp, inputVals.size());

    if (targetInfo.supportsCvtPkScalePk8Upcast()) {
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
      for (int i = 0; i < inputVals.size(); i += 4) {
        int scaleIdx = groupScaleReg[i / 4];
        SmallVector<Value> converted =
            elemType.isF16() ? upcast8xMxfp4_HW<ROCDL::CvtScaleF32PkF16Fp4Op>(
                                   rewriter, loc, inputVals, i,
                                   scaleVals[scaleIdx], preShifted)
                             : upcast8xMxfp4_HW<ROCDL::CvtScaleF32PkBf16Fp4Op>(
                                   rewriter, loc, inputVals, i,
                                   scaleVals[scaleIdx], preShifted);
        results.append(converted.begin(), converted.end());
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

        Value scaleF32 = scaleToF32(
            rewriter, loc, scaleVals[groupScaleReg[i / 4]], preShifted);

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

    // The op verifier guarantees whole register-consecutive groups.
    assert(inputVals.size() %
               (targetInfo.supportsCvtPkScalePk8Upcast() ? 8 : 4) ==
           0);
    assert(inputVals.size() == scaleVals.size());

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    bool preShifted = scaleIsPreShifted(upcastOp.getScale().getType());
    SmallVector<Value> results;
    results.reserve(inputVals.size());
    // Broadcast layouts can use FP8 Block32 (opSel=0). Otherwise use Block16
    // (opSel=8) and pack lane j^16's scale into byte 1.
    bool broadcast = isScaleLane16Broadcast(upcastOp.getScale().getType());
    if (targetInfo.supportsCvtPkScalePk8Upcast() &&
        (broadcast ||
         targetInfo.supportsCvtPkScalePk8Block16())) { // b32 vs b16 needed
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
    } else if (targetInfo.supportsHwScaledUpcast() &&
               !targetInfo.supportsCvtPkScalePk8()) {
      for (int i = 0; i < inputVals.size(); i += 4) {
        SmallVector<Value> converted =
            elemType.isF16()
                ? (isa<Float8E4M3FNType>(fp8ElemType)
                       ? upcast4xMxfp8_HW<ROCDL::CvtScaleF32PkF16Fp8Op>(
                             rewriter, loc, inputVals, i, scaleVals[i],
                             preShifted)
                       : upcast4xMxfp8_HW<ROCDL::CvtScaleF32PkF16Bf8Op>(
                             rewriter, loc, inputVals, i, scaleVals[i],
                             preShifted))
                : (isa<Float8E4M3FNType>(fp8ElemType)
                       ? upcast4xMxfp8_HW<ROCDL::CvtScaleF32PkBf16Fp8Op>(
                             rewriter, loc, inputVals, i, scaleVals[i],
                             preShifted)
                       : upcast4xMxfp8_HW<ROCDL::CvtScaleF32PkBf16Bf8Op>(
                             rewriter, loc, inputVals, i, scaleVals[i],
                             preShifted));
        results.append(converted.begin(), converted.end());
      }
    } else {
      // Software emulation: convert fp8 to f32, then multiply by scale.
      bool isE4M3FN = isa<Float8E4M3FNType>(fp8ElemType);
      bool toFp16 = elemType.isF16();
      for (size_t i = 0; i < inputVals.size(); ++i) {
        Value scaleF32 = scaleToF32(rewriter, loc, scaleVals[i], preShifted);
        Value f32Val = convertF8ToF32_SW(rewriter, loc, inputVals[i], isE4M3FN);
        Value mulF32 = b.fmul(f32Val, scaleF32);
        if (toFp16) {
          results.push_back(b.fptrunc(f16_ty, mulF32));
        } else {
          Value mulI16 =
              b.trunc(i16_ty, b.lshr(b.bitcast(mulF32, i32_ty), b.i32_val(16)));
          results.push_back(b.bitcast(mulI16, bf16_ty));
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
