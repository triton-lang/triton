//===----------------------------------------------------------------------===//
//
// This file lowers Triton `triton::FpToFpOp` casts for the AMD backend to LLVM
// dialect operations and ROCDL conversion intrinsics.
//
// It provides:
//  - software fallback conversion paths for FP32/FP16/BF16/FP8 variants,
//  - hardware-accelerated packed conversion paths,
//  - round-mode aware handling (RTNE/RTZ), saturation, NaN/Inf and subnormal
//    semantics.
//
// Conversions are described by a table of rules (see getFpCvtRules). Each rule
// lowers one (src, dst, rounding) conversion with a fixed vector width and may
// require a hardware feature (see amdgpu::TargetFeatures). The first rule whose
// feature the target has is used.
//
//===----------------------------------------------------------------------===//

#include "third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TargetInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <type_traits>

using namespace mlir;

using mlir::getElementTypeOrSelf;
using mlir::triton::gpu::ElementwiseOpConversionBase;
using mlir::triton::gpu::MultipleOperandsRange;
using triton::amdgpu::TargetFeatures;

//===----------------------------------------------------------------------===//
// Common utilities
//===----------------------------------------------------------------------===//

namespace {
template <typename FPType> struct FPTypeInfo {
  FPTypeInfo(Location loc, ConversionPatternRewriter &rewriter)
      : loc(loc), rewriter(rewriter), b(loc, rewriter) {}
  constexpr IntegerType getIntType() {
    if constexpr (std::is_same_v<FPType, Float32Type>) {
      return i32_ty;
    }
    if constexpr (std::is_same_v<FPType, Float16Type> ||
                  std::is_same_v<FPType, BFloat16Type>) {
      return i16_ty;
    }
    if constexpr (std::is_same_v<FPType, Float8E4M3FNType> ||
                  std::is_same_v<FPType, Float8E5M2Type> ||
                  std::is_same_v<FPType, Float8E4M3FNUZType> ||
                  std::is_same_v<FPType, Float8E5M2FNUZType>) {
      return i8_ty;
    }
    return nullptr;
  }

  auto getHalfwayPointsForDstType(TypeID dstTyID) {
    using VecType =
        std::conditional_t<std::is_same_v<FPType, Float32Type>,
                           SmallVector<int32_t>, SmallVector<int16_t>>;
    if constexpr (std::is_same_v<FPType, Float32Type>) {
      if (dstTyID == TypeID::get<Float8E4M3FNType>())
        return VecType{0x3a800000,  // halfway between [0/8 * 2^-6, 1/8 * 2^-6]
                       0x3b400000,  // halfway between [1/8 * 2^-6, 2/8 * 2^-6]
                       0x3ba00000,  // halfway between [2/8 * 2^-6, 3/8 * 2^-6]
                       0x3be00000,  // halfway between [3/8 * 2^-6, 4/8 * 2^-6]
                       0x3c100000,  // halfway between [4/8 * 2^-6, 5/8 * 2^-6]
                       0x3c300000,  // halfway between [5/8 * 2^-6, 6/8 * 2^-6]
                       0x3c500000,  // halfway between [6/8 * 2^-6, 7/8 * 2^-6]
                       0x3c700000}; // halfway between [7/8 * 2^-6, 8/8 * 2^-6]
      if (dstTyID == TypeID::get<Float8E5M2Type>())
        return VecType{
            0x37000000,  // halfway between [0/4 * 2^(-14), 1/4 * 2^(-14)]
            0x37c00000,  // halfway between [1/4 * 2^(-14), 2/4 * 2^(-14)]
            0x38200000,  // halfway between [2/4 * 2^(-14), 3/4 * 2^(-14)]
            0x38600000}; // halfway between [3/4 * 2^(-14), 4/4 * 2^(-14)]
      if (dstTyID == TypeID::get<Float8E4M3FNUZType>())
        // We divide the range of subnormals in 2^3 subranges.
        // Each i entry in the LUT corresponds to the midpoint of the ith
        // subrange represented in the src format (here float32)
        return VecType{0x3a000000,  // halfway between [0/8 * 2^-7, 1/8 * 2^-7]
                       0x3ac00000,  // halfway between [1/8 * 2^-7, 2/8 * 2^-7]
                       0x3b200000,  // halfway between [2/8 * 2^-7, 3/8 * 2^-7]
                       0x3b600000,  // halfway between [3/8 * 2^-7, 4/8 * 2^-7]
                       0x3b900000,  // halfway between [4/8 * 2^-7, 5/8 * 2^-7]
                       0x3bb00000,  // halfway between [5/8 * 2^-7, 6/8 * 2^-7]
                       0x3bd00000,  // halfway between [6/8 * 2^-7, 7/8 * 2^-7]
                       0x3bf00000}; // halfway between [7/8 * 2^-7, 8/8 * 2^-7]
      if (dstTyID == TypeID::get<Float8E5M2FNUZType>())
        // Minimum normal for E5M2FNUZ is 0x38000000 (2^-15)
        // We divide the range of subnormals in 2^2 subranges.
        // Each i entry in the LUT corresponds to the midpoint of the ith
        // subrange represented in the src format (here float32)
        return VecType{
            0x36800000,  // halfway between [0/4 * 2^-15, 1/4 * 2^-15]
            0x37400000,  // halfway between [1/4 * 2^-15, 2/4 * 2^-15]
            0x37a00000,  // halfway between [2/4 * 2^-15, 3/4 * 2^-15]
            0x37e00000}; // halfway between [3/4 * 2^-15, 4/4 * 2^-15]
    }
    if constexpr (std::is_same_v<FPType, Float16Type>) {
      if (dstTyID == TypeID::get<Float8E4M3FNType>())
        return VecType{0x1400, 0x1A00, 0x1D00, 0x1F00,
                       0x2080, 0x2180, 0x2280, 0x2380};
      if (dstTyID == TypeID::get<Float8E5M2Type>())
        return VecType{0x0080, 0x0180, 0x0200, 0x0380};
      if (dstTyID == TypeID::get<Float8E4M3FNUZType>())
        // Minimum normal for E4M3FNUZ is 0x2000 (2^-7)
        // We divide the range of subnormals in 2^3 subranges.
        // Each i entry in the LUT corresponds to the midpoint of the ith
        // subrange represented in the src format (here float16)
        return VecType{0x1000,  // halfway between [0/8 * 2^-7, 1/8 * 2^-7]
                       0x1600,  // halfway between [1/8 * 2^-7, 2/8 * 2^-7]
                       0x1900,  // halfway between [2/8 * 2^-7, 3/8 * 2^-7]
                       0x1b00,  // halfway between [3/8 * 2^-7, 4/8 * 2^-7]
                       0x1c80,  // halfway between [4/8 * 2^-7, 5/8 * 2^-7]
                       0x1d80,  // halfway between [5/8 * 2^-7, 6/8 * 2^-7]
                       0x1e80,  // halfway between [6/8 * 2^-7, 7/8 * 2^-7]
                       0x1f80}; // halfway between [7/8 * 2^-7, 8/8 * 2^-7]
    }
    if constexpr (std::is_same_v<FPType, BFloat16Type>) {
      if (dstTyID == TypeID::get<Float8E4M3FNUZType>())
        // Minimum normal for E4M3FNUZ is 0x3c00 (2^-7)
        // We divide the range of subnormals in 2^3 subranges.
        // Each i entry in the LUT corresponds to the midpoint of the ith
        // subrange represented in the src format (here bfloat16)
        return VecType{0x3a00,  // halfway between [0/8 * 2^-7, 1/8 * 2^-7]
                       0x3ac0,  // halfway between [1/8 * 2^-7, 2/8 * 2^-7]
                       0x3b20,  // halfway between [2/8 * 2^-7, 3/8 * 2^-7]
                       0x3b60,  // halfway between [3/8 * 2^-7, 4/8 * 2^-7]
                       0x3b90,  // halfway between [4/8 * 2^-7, 5/8 * 2^-7]
                       0x3bb0,  // halfway between [5/8 * 2^-7, 6/8 * 2^-7]
                       0x3bd0,  // halfway between [6/8 * 2^-7, 7/8 * 2^-7]
                       0x3bf0}; // halfway between [7/8 * 2^-7, 8/8 * 2^-7]
      if (dstTyID == TypeID::get<Float8E5M2FNUZType>()) {
        // Minimum normal for E5M2FNUZ is 0x3800 (2^-15)
        // We divide the range of subnormals in 2^2 subranges.
        // Each i entry in the LUT corresponds to the midpoint of the ith
        // subrange represented in the src format (here bfloat16)
        // 2^-18 =
        return VecType{0x3680,  // halfway between [0/4 * 2^-15, 1/4 * 2^-15]
                       0x3740,  // halfway between [1/4 * 2^-15, 2/4 * 2^-15]
                       0x37a0,  // halfway between [2/4 * 2^-15, 3/4 * 2^-15]
                       0x37e0}; // halfway between [3/4 * 2^-15, 4/4 * 2^-15]
      }
      if (dstTyID == TypeID::get<Float8E4M3FNType>())
        return VecType{0x3a80, 0x3b40, 0x3ba0, 0x3be0,
                       0x3c10, 0x3c30, 0x3c50, 0x3c70};
      if (dstTyID == TypeID::get<Float8E5M2Type>())
        return VecType{0x3700, 0x37c0, 0x3820, 0x3860};
    }
    return VecType{};
  }

  constexpr Value toLLVMIntValue(int32_t val) {
    if constexpr (std::is_same_v<FPType, Float32Type>) {
      return b.i32_val(val);
    }
    if constexpr (std::is_same_v<FPType, Float16Type> ||
                  std::is_same_v<FPType, BFloat16Type>) {
      return b.i16_val(val);
    }
    if constexpr (std::is_same_v<FPType, Float8E4M3FNType> ||
                  std::is_same_v<FPType, Float8E5M2Type> ||
                  std::is_same_v<FPType, Float8E4M3FNUZType> ||
                  std::is_same_v<FPType, Float8E5M2FNUZType>) {
      return b.i8_val(val);
    }
    return nullptr;
  }

  const llvm::fltSemantics &getFPSemantics() {
    if constexpr (std::is_same_v<FPType, Float32Type>) {
      return llvm::APFloat::IEEEsingle();
    }
    if constexpr (std::is_same_v<FPType, Float16Type>) {
      return llvm::APFloat::IEEEhalf();
    }
    if constexpr (std::is_same_v<FPType, BFloat16Type>) {
      return llvm::APFloat::BFloat();
    }
    if constexpr (std::is_same_v<FPType, Float8E4M3FNType>) {
      return llvm::APFloat::Float8E4M3FN();
    }
    if constexpr (std::is_same_v<FPType, Float8E4M3FNUZType>) {
      return llvm::APFloat::Float8E4M3FNUZ();
    }
    if constexpr (std::is_same_v<FPType, Float8E5M2FNUZType>) {
      return llvm::APFloat::Float8E5M2FNUZ();
    }

    return llvm::APFloat::Bogus();
  }

  Location loc;
  ConversionPatternRewriter &rewriter;
  TritonLLVMOpBuilder b;
};
} // namespace

static Value checkIsNan(TritonLLVMOpBuilder &builder, Value v) {
  Location loc = builder.loc;
  OpBuilder &rewriter = *builder.builder;

  // bits 0 and 1 indicate signaling Nan and quiet Nan, respectively
  IntegerAttr controlBits = rewriter.getIntegerAttr(i32_ty, 0b11);
  return LLVM::IsFPClass::create(rewriter, loc, i1_ty, v, controlBits);
}

// Fp16 -> Fp32
static Value Fp16ToFp32OneValue(Location loc,
                                ConversionPatternRewriter &rewriter,
                                const Value &v) {

  TritonLLVMOpBuilder b(loc, rewriter);
  return b.fpext(f32_ty, v);
}

static Value Fp32ToFp16rtneOneValue(Location loc, RewriterBase &rewriter,
                                    const Value &v) {
  return LLVM::FPTruncOp::create(rewriter, loc, f16_ty, v);
}

// Fp16/Bf16 -> Fp32
template <typename SrcFPType>
  requires llvm::is_one_of<SrcFPType, Float16Type, BFloat16Type>::value
static SmallVector<Value> extToFp32(Location loc,
                                    ConversionPatternRewriter &rewriter,
                                    const SmallVector<Value> &v) {
  SmallVector<Value> result;
  for (Value x : v) {
    if constexpr (std::is_same_v<SrcFPType, Float16Type>)
      result.push_back(Fp16ToFp32OneValue(loc, rewriter, x));
    else
      result.push_back(AMD::convertBf16ToFp32(loc, rewriter, x));
  }
  return result;
}

// Converts a fixed number of values, given by the numElements of the rule that
// uses it (see FpCvtRule).
using CvtImpl = SmallVector<Value> (*)(Location, ConversionPatternRewriter &,
                                       const SmallVector<Value> &);

//===----------------------------------------------------------------------===//
// Fp8 upcasts: hardware
//===----------------------------------------------------------------------===//

// Convert eight OCP Fp8/Bf8 values to Fp16/Bf16/Fp32 with one scaled
// v_cvt_scale_pk8_* (unit scale). Currently used on CDNA5 targets that support
// it (see TargetFeatures::supportsCvtPkScalePk8Upcast).
template <typename ConvertOp>
static SmallVector<Value>
scalePk8UpcastFromFp8(Location loc, ConversionPatternRewriter &rewriter,
                      const SmallVector<Value> &v) {
  assert(v.size() == 8);
  const size_t inSize = v.size();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Type vInTy = nullptr;
  Type resTy = nullptr;
  Type vResTy = nullptr;

  const size_t intVecSize = (inSize * i8_ty.getWidth()) / i32_ty.getWidth();
  vInTy = vec_ty(i32_ty, intVecSize);

  if constexpr ((std::is_same_v<ConvertOp, ROCDL::CvtPkScalePk8F32Fp8Op>) ||
                (std::is_same_v<ConvertOp, ROCDL::CvtPkScalePk8F32Bf8Op>)) {
    resTy = f32_ty;
  } else if constexpr ((std::is_same_v<ConvertOp,
                                       ROCDL::CvtPkScalePk8F16Bf8Op>) ||
                       (std::is_same_v<ConvertOp,
                                       ROCDL::CvtPkScalePk8F16Fp8Op>)) {
    resTy = f16_ty;
  } else if constexpr ((std::is_same_v<ConvertOp,
                                       ROCDL::CvtPkScalePk8Bf16Bf8Op>) ||
                       (std::is_same_v<ConvertOp,
                                       ROCDL::CvtPkScalePk8Bf16Fp8Op>)) {
    resTy = bf16_ty;
  }
  assert(resTy != nullptr);

  vResTy = vec_ty(resTy, inSize);

  auto vI8InTy = vec_ty(i8_ty, inSize);

  Value vI8In = b.undef(vI8InTy);
  SmallVector<Value, 8> idx;
  for (size_t i = 0; i < inSize; ++i) {
    idx.push_back(b.i32_val(i));
    vI8In = b.insert_element(vI8InTy, vI8In, v[i], idx[i]);
  }
  auto vIn = b.bitcast(vI8In, vInTy);

  // Create fp8(1.0) as scale
  Value scale = b.i32_val(127);
  // OpScale 0 = use bits [0:7] from scale
  IntegerAttr opscale = rewriter.getI32IntegerAttr(0);

  auto result = ConvertOp::create(rewriter, loc, vResTy, vIn, scale, opscale);
  SmallVector<Value> ret(inSize);
  for (auto [i, value] : llvm::enumerate(ret)) {
    value = b.extract_element(resTy, result, idx[i]);
  }

  return ret;
}

// Convert four OCP Fp8/Bf8 values to Fp16/Bf16/Fp32 with the scaled
// v_cvt_scalef32_pk_* (two values per instruction, unit scale). Currently used
// on CDNA4 (see TargetFeatures::supportsCvtScaleF32Pk).
template <typename ConvertOp>
static SmallVector<Value>
scalePk4UpcastFromFp8(Location loc, ConversionPatternRewriter &rewriter,
                      const SmallVector<Value> &v) {
  assert(v.size() == 4);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value fp8x4Vec = b.undef(fp8x4VecTy);
  SmallVector<Value, 4> idx;
  for (size_t i = 0; i < 4; i++) {
    idx.push_back(b.i32_val(i));
    fp8x4Vec = b.insert_element(fp8x4VecTy, fp8x4Vec, v[i], idx[i]);
  }
  auto i32v = b.bitcast(fp8x4Vec, i32_ty);

  Type resElemType;
  if constexpr (std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkF32Fp8Op> ||
                std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkF32Bf8Op>) {
    resElemType = f32_ty;
  } else if constexpr (std::is_same_v<ConvertOp,
                                      ROCDL::CvtScaleF32PkF16Fp8Op> ||
                       std::is_same_v<ConvertOp,
                                      ROCDL::CvtScaleF32PkF16Bf8Op>) {
    resElemType = f16_ty;
  } else {
    resElemType = bf16_ty;
  }
  Type resType = vec_ty(resElemType, 2);
  Value scale = b.f32_val(1);
  auto result1 = ConvertOp::create(rewriter, loc, resType, i32v, scale,
                                   /*srcLoHiSel=*/false);
  auto result2 = ConvertOp::create(rewriter, loc, resType, i32v, scale,
                                   /*srcLoHiSel=*/true);
  SmallVector<Value> ret(4);
  ret[0] = b.extract_element(resElemType, result1, idx[0]);
  ret[1] = b.extract_element(resElemType, result1, idx[1]);
  ret[2] = b.extract_element(resElemType, result2, idx[0]);
  ret[3] = b.extract_element(resElemType, result2, idx[1]);
  return ret;
}

// Convert four packed Fp8/Bf8 values to Fp32 with the packed unscaled cvt
// (v_cvt_pk_f32_{fp8,bf8}). The target decides the fp8 format; currently used
// only for FNUZ formats on CDNA3 (see TargetFeatures::supportsCvtFp8Fnuz).
template <typename ConvertOp>
static SmallVector<Value> Pk4F8ToFp32(Location loc,
                                      ConversionPatternRewriter &rewriter,
                                      const SmallVector<Value> &v) {
  assert(v.size() == 4);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value fp8x4Vec = b.undef(fp8x4VecTy);
  SmallVector<Value, 4> idx;
  for (size_t i = 0; i < 4; i++) {
    idx.push_back(b.i32_val(i));
    fp8x4Vec = b.insert_element(fp8x4VecTy, fp8x4Vec, v[i], idx[i]);
  }
  auto i32v = b.bitcast(fp8x4Vec, i32_ty);

  auto resType = i64_ty;
  auto dstType = f32_ty;

  auto resultLo =
      ConvertOp::create(rewriter, loc, resType, i32v, /*wordSel=*/false);
  auto resultHi =
      ConvertOp::create(rewriter, loc, resType, i32v, /*wordSel=*/true);
  auto f32x2VecTy = vec_ty(dstType, 2);
  SmallVector<Value> ret(4);
  auto retVec = b.bitcast(resultLo, f32x2VecTy);
  ret[0] = b.extract_element(dstType, retVec, idx[0]);
  ret[1] = b.extract_element(dstType, retVec, idx[1]);
  retVec = b.bitcast(resultHi, f32x2VecTy);
  ret[2] = b.extract_element(dstType, retVec, idx[0]);
  ret[3] = b.extract_element(dstType, retVec, idx[1]);
  return ret;
}

// Convert four packed Fp8/Bf8 values to Fp32 with the unscaled cvt
// (v_cvt_f32_{fp8,bf8}), one byte per instruction. ConvertOp chooses Fp8 versus
// Bf8; currently used for OCP formats on RDNA4 (see
// TargetFeatures::supportsCvtFp8Ocp).
template <typename ConvertOp>
static SmallVector<Value> F8x4ToFp32(Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     const SmallVector<Value> &v) {
  assert(v.size() == 4);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value fp8x4Vec = b.undef(fp8x4VecTy);
  for (size_t i = 0; i < 4; i++)
    fp8x4Vec = b.insert_element(fp8x4VecTy, fp8x4Vec, v[i], b.i32_val(i));
  Value i32v = b.bitcast(fp8x4Vec, i32_ty);

  SmallVector<Value> ret(4);
  for (int i = 0; i < 4; i++)
    ret[i] = ConvertOp::create(rewriter, loc, f32_ty, i32v, /*byteSel=*/i);
  return ret;
}

// Fp8/Bf8 -> Fp32 via the unscaled cvt. FNUZ formats use the packed form, OCP
// formats one byte per instruction (see TargetFeatures::supportsCvtFp8Ocp).
// The packed v_cvt_pk_f32_{fp8,bf8} is avoided for OCP formats because LLVM
// currently selects its word-0 form as a fake16 VOP1 that the true16 assembler
// rejects for VGPR sources.
template <typename SrcFPType>
static SmallVector<Value> unscaledFp8ToFp32(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            const SmallVector<Value> &v) {
  if constexpr (std::is_same_v<SrcFPType, Float8E4M3FNUZType>)
    return Pk4F8ToFp32<ROCDL::CvtPkF32Fp8Op>(loc, rewriter, v);
  else if constexpr (std::is_same_v<SrcFPType, Float8E5M2FNUZType>)
    return Pk4F8ToFp32<ROCDL::CvtPkF32Bf8Op>(loc, rewriter, v);
  else if constexpr (std::is_same_v<SrcFPType, Float8E4M3FNType>)
    return F8x4ToFp32<ROCDL::CvtF32Fp8Op>(loc, rewriter, v);
  else {
    static_assert(std::is_same_v<SrcFPType, Float8E5M2Type>);
    return F8x4ToFp32<ROCDL::CvtF32Bf8Op>(loc, rewriter, v);
  }
}

// Fp8/Bf8 -> Fp16 via the unscaled cvt
template <typename SrcFPType>
static SmallVector<Value> unscaledFp8ToFp16(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            const SmallVector<Value> &v) {
  SmallVector<Value> ret = unscaledFp8ToFp32<SrcFPType>(loc, rewriter, v);
  for (Value &x : ret)
    x = Fp32ToFp16rtneOneValue(loc, rewriter, x);
  return ret;
}

// Fp8/Bf8 -> Bf16 via the unscaled cvt
template <typename SrcFPType>
static SmallVector<Value> unscaledFp8ToBf16(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            const SmallVector<Value> &v) {
  SmallVector<Value> ret = unscaledFp8ToFp32<SrcFPType>(loc, rewriter, v);
  for (Value &x : ret)
    x = AMD::convertFp32ToBf16(loc, rewriter, x, RoundingMode::RTZ);
  return ret;
}

//===----------------------------------------------------------------------===//
// Fp8 upcasts: software
//===----------------------------------------------------------------------===//

static Value Fp8E4M3fnuzToFp16OneValue(Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       Value v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  Value a = b.undef(fp8x2VecTy);
  a = b.insert_element(fp8x2VecTy, a, b.i8_val(0), b.i32_val(0));
  a = b.insert_element(fp8x2VecTy, a, v, b.i32_val(1));
  a = b.bitcast(a, i16_ty);

  // Get sign and absolute value
  Value sign = b.and_(a, b.i16_val(0x8000));
  a = b.and_(a, b.i16_val(0x7FFF));

  // Right shift 1 bit to adjust the positions of exponent and mantissa
  a = b.lshr(a, b.i16_val(1));

  // Adjust exponent, (15 - 8) << 10 === 0x1C00
  a = b.add(a, b.i16_val(0x1C00));

  Value v8 = b.bitcast(v, i8_ty);
  Value vAbs = b.and_(v8, b.i8_val(0x7F));

  // Check denorms and zero
  // Here we use a LUT to map S.0000.000 ~ S.0000.111 to its corresponding
  // fp16 value Minimum subnormal value in E4M3FNUZ is 2^-10
  constexpr size_t lutSize = 8;
  static constexpr int denormsAndZeroLut[lutSize] = {0x0000,  // 0 * 2^-10
                                                     0x1400,  // 1 * 2^-10
                                                     0x1800,  // 2 * 2^-10
                                                     0x1a00,  // 3 * 2^-10
                                                     0x1c00,  // 4 * 2^-10
                                                     0x1d00,  // 5 * 2^-10
                                                     0x1e00,  // 6 * 2^-10
                                                     0x1f00}; // 7 * 2^-10

  for (int i = 0; i < lutSize; i++) {
    a = b.select(b.icmp_eq(vAbs, b.i8_val(i)), b.i16_val(denormsAndZeroLut[i]),
                 a);
  }

  // Set sign
  a = b.or_(a, sign);
  // 0x80 is the fnuz NaN; guard it last, else the zero LUT and sign-or
  // clobber it.
  a = b.select(b.icmp_eq(v8, b.i8_val(0x80)), b.i16_val(0x7E00), a);
  a = b.bitcast(a, f16_ty);

  return a;
}

static SmallVector<Value>
Fp8E4M3fnuzToFp16SW(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 4);
  SmallVector<Value> result(4);
  for (size_t i = 0; i < 4; i++)
    result[i] = Fp8E4M3fnuzToFp16OneValue(loc, rewriter, v[i]);
  return result;
}

// OCP Fp8 -> Fp16
static SmallVector<Value> Fp8E4M3fnToFp16SW(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            const SmallVector<Value> &v) {
  assert(v.size() == 4);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  auto i16x2VecTy = vec_ty(i16_ty, 2);

  Value scale = b.undef(fp16x2VecTy);
  scale = b.insert_element(fp16x2VecTy, scale, b.f16_val(0x1p+8), b.i32_val(0));
  scale = b.insert_element(fp16x2VecTy, scale, b.f16_val(0x1p+8), b.i32_val(1));

  SmallVector<Value> results;
  for (size_t i = 0; i < 4; i += 2) {
    // Pack two fp8 values into the high bytes of the 16-bit halves of an i32
    Value a = b.undef(fp8x4VecTy);
    a = b.insert_element(fp8x4VecTy, a, b.i8_val(0), b.i32_val(0));
    a = b.insert_element(fp8x4VecTy, a, v[i], b.i32_val(1));
    a = b.insert_element(fp8x4VecTy, a, b.i8_val(0), b.i32_val(2));
    a = b.insert_element(fp8x4VecTy, a, v[i + 1], b.i32_val(3));
    a = b.bitcast(a, i32_ty);

    // Get sign and absolute value
    Value sign = b.and_(i32_ty, a, b.i32_val(0x80008000));
    a = b.and_(i32_ty, a, b.i32_val(0x7FFF7FFF));

    // Right shift 1 bit to adjust the positions of exponent and mantissa
    a = b.lshr(i32_ty, a, b.i32_val(1));
    a = b.bitcast(a, fp16x2VecTy);

    // Multiplying by 2^(15-7) rebiases the exponent and normalizes
    // subnormals. fp8 subnormals are fp16 denormals before the multiply, so
    // this requires fp16 denormals to be preserved, not flushed to zero.
    a = b.fmul(fp16x2VecTy, a, scale);
    a = b.bitcast(a, i32_ty);

    // Set sign
    a = b.or_(i32_ty, a, sign);
    a = b.bitcast(a, i16x2VecTy);

    for (size_t j = 0; j < 2; j++) {
      Value r = b.extract_element(i16_ty, a, b.i32_val(j));
      // Check NaN
      Value vAbs = b.and_(v[i + j], b.i8_val(0x7F));
      r = b.select(b.icmp_eq(vAbs, b.i8_val(0x7F)), b.i16_val(0x7E00), r);
      results.push_back(b.bitcast(r, f16_ty));
    }
  }
  return results;
}

static Value Fp8E5M2fnuzToFp16OneValue(Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       Value v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  Value a = b.undef(fp8x2VecTy);
  a = b.insert_element(fp8x2VecTy, a, b.int_val(8, 0), b.i32_val(0));
  a = b.insert_element(fp8x2VecTy, a, v, b.i32_val(1));
  a = b.bitcast(a, i16_ty);

  auto e = b.and_(i16_ty, a, b.int_val(16, 0x7C00));
  auto m = b.and_(i16_ty, a, b.int_val(16, 0x0300));
  auto sign = b.and_(i16_ty, a, b.int_val(16, 0x8000));

  // check whether all exponents are zeros
  auto e_is_zero = b.icmp_eq(e, b.int_val(16, 0x0));

  // case 1, e is zero, need to move m right by 1 bit
  auto m1 = b.lshr(i16_ty, m, b.int_val(16, 1));
  auto o0 = b.or_(i16_ty, sign, m1);

  // case 2, e is nonzero, sub exponent by 1
  auto e1 = b.sub(i16_ty, e, b.int_val(16, 0x0400));

  auto e_is_one = b.icmp_eq(e, b.int_val(16, 0x0400));
  auto m2 = b.add(i16_ty, m1, b.int_val(16, 0x0200));

  auto o1 = b.or_(i16_ty, sign, b.or_(i16_ty, m, e1));
  auto o2 = b.or_(i16_ty, sign, m2);

  auto o12 = b.select(e_is_one, o2, o1);
  auto o = b.select(e_is_zero, o0, o12);

  // 0x80 is the fnuz NaN; map it to an fp16 NaN.
  o = b.select(b.icmp_eq(v, b.i8_val(0x80)), b.i16_val(0x7E00), o);

  return b.bitcast(o, f16_ty);
}

static SmallVector<Value>
Fp8E5M2fnuzToFp16SW(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 4);
  SmallVector<Value> result(4);
  for (size_t i = 0; i < 4; i++)
    result[i] = Fp8E5M2fnuzToFp16OneValue(loc, rewriter, v[i]);
  return result;
}

// OCP Bf8 -> Fp16
static SmallVector<Value> Fp8E5M2ToFp16SW(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = b.undef(fp8x4VecTy);
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(0));
  a0 = b.insert_element(fp8x4VecTy, a0, v[0], b.i32_val(1));
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(2));
  a0 = b.insert_element(fp8x4VecTy, a0, v[1], b.i32_val(3));
  a0 = b.bitcast(a0, i32_ty);
  Value a1 = b.undef(fp8x4VecTy);
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(0));
  a1 = b.insert_element(fp8x4VecTy, a1, v[2], b.i32_val(1));
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(2));
  a1 = b.insert_element(fp8x4VecTy, a1, v[3], b.i32_val(3));
  a1 = b.bitcast(a1, i32_ty);

  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  auto fp16x2Vec0 = b.bitcast(a0, fp16x2VecTy);
  auto fp16x2Vec1 = b.bitcast(a1, fp16x2VecTy);

  return {b.extract_element(f16_ty, fp16x2Vec0, b.i32_val(0)),
          b.extract_element(f16_ty, fp16x2Vec0, b.i32_val(1)),
          b.extract_element(f16_ty, fp16x2Vec1, b.i32_val(0)),
          b.extract_element(f16_ty, fp16x2Vec1, b.i32_val(1))};
}

// OCP Bf8/Fp8 -> Bf16
template <typename SrcFPType>
  requires llvm::is_one_of<SrcFPType, Float8E4M3FNType, Float8E5M2Type>::value
static SmallVector<Value> OcpF8ToBf16SW(Location loc,
                                        ConversionPatternRewriter &rewriter,
                                        const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = b.undef(fp8x4VecTy);
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(0));
  a0 = b.insert_element(fp8x4VecTy, a0, v[0], b.i32_val(1));
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(2));
  a0 = b.insert_element(fp8x4VecTy, a0, v[1], b.i32_val(3));
  a0 = b.bitcast(a0, i32_ty);

  Value a1 = b.undef(fp8x4VecTy);
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(0));
  a1 = b.insert_element(fp8x4VecTy, a1, v[2], b.i32_val(1));
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(2));
  a1 = b.insert_element(fp8x4VecTy, a1, v[3], b.i32_val(3));
  a1 = b.bitcast(a1, i32_ty);

  Value b0 = b.and_(i32_ty, a0, b.i32_val(0x7fff7fff));
  Value b1 = b.and_(i32_ty, a1, b.i32_val(0x7fff7fff));
  uint32_t reducedMantissaBits;
  float upcastBias;
  if constexpr (std::is_same_v<SrcFPType, Float8E4M3FNType>) {
    reducedMantissaBits = 4; // 3 + 8 - 7
    upcastBias = 0x1p+120;   // 2^(127-7)
  } else {
    reducedMantissaBits = 3; // 2 + 8 - 7
    upcastBias = 0x1p+112;   // 2^(127-15)
  }
  b0 = b.lshr(i32_ty, b0, b.i32_val(reducedMantissaBits));
  b1 = b.lshr(i32_ty, b1, b.i32_val(reducedMantissaBits));

  Value c0 = b.shl(i32_ty, b0, b.i32_val(16));
  Value c1 = b.and_(i32_ty, b0, b.i32_val(0xFFFF0000));
  Value c2 = b.shl(i32_ty, b1, b.i32_val(16));
  Value c3 = b.and_(i32_ty, b1, b.i32_val(0xFFFF0000));

  c0 = b.bitcast(c0, f32_ty);
  c1 = b.bitcast(c1, f32_ty);
  c2 = b.bitcast(c2, f32_ty);
  c3 = b.bitcast(c3, f32_ty);

  Value d0 = b.fmul(f32_ty, c0, b.f32_val(upcastBias));
  Value d1 = b.fmul(f32_ty, c1, b.f32_val(upcastBias));
  Value d2 = b.fmul(f32_ty, c2, b.f32_val(upcastBias));
  Value d3 = b.fmul(f32_ty, c3, b.f32_val(upcastBias));

  d0 = b.bitcast(d0, i32_ty);
  d1 = b.bitcast(d1, i32_ty);
  d2 = b.bitcast(d2, i32_ty);
  d3 = b.bitcast(d3, i32_ty);

  Value out0 = b.or_(i32_ty, b.lshr(i32_ty, d0, b.i32_val(16)), d1);
  Value out1 = b.or_(i32_ty, b.lshr(i32_ty, d2, b.i32_val(16)), d3);

  Value sign0 = b.and_(i32_ty, a0, b.i32_val(0x80008000));
  Value sign1 = b.and_(i32_ty, a1, b.i32_val(0x80008000));

  out0 = b.or_(i32_ty, out0, sign0);
  out1 = b.or_(i32_ty, out1, sign1);

  auto bf16x2VecTy = vec_ty(bf16_ty, 2);
  out0 = b.bitcast(out0, bf16x2VecTy);
  out1 = b.bitcast(out1, bf16x2VecTy);

  return {b.extract_element(bf16_ty, out0, b.i32_val(0)),
          b.extract_element(bf16_ty, out0, b.i32_val(1)),
          b.extract_element(bf16_ty, out1, b.i32_val(0)),
          b.extract_element(bf16_ty, out1, b.i32_val(1))};
}

static SmallVector<Value>
Fp8E4M3fnuzToBf16SW(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  // Create a packed representation of both fp8 values:
  // Each i halfword (16bit) has the upper byte set to v[i] and the lower byte
  // to 0 byte3             byte0 | v[1] | 0 | v[0] | 0 |
  Value a0 = b.undef(fp8x4VecTy);
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(0));
  a0 = b.insert_element(fp8x4VecTy, a0, v[0], b.i32_val(1));
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(2));
  a0 = b.insert_element(fp8x4VecTy, a0, v[1], b.i32_val(3));
  a0 = b.bitcast(a0, i32_ty);

  // Clear sign bits and align the 3bit mantissa fields of each halfword with
  // the mantissa position in bfloat16
  constexpr uint32_t signMask = 1U << (16 - 1U);
  constexpr uint32_t absMask = signMask - 1;
  constexpr uint32_t absHalfwordMask = absMask | (absMask << 16U);
  constexpr uint32_t dstMantBitWidth = 7;
  constexpr uint32_t srcMantBitWidth = 3;

  Value b0 = b.and_(i32_ty, a0, b.i32_val(absHalfwordMask));
  b0 = b.lshr(i32_ty, b0, b.i32_val(dstMantBitWidth - srcMantBitWidth));

  // Split the 2 halfwords into separate 32bit words in order to convert them
  Value c0 = b.shl(i32_ty, b0, b.i32_val(16));
  Value c1 = b.and_(i32_ty, b0, b.i32_val(0xFFFF0000));
  c0 = b.bitcast(c0, f32_ty);
  c1 = b.bitcast(c1, f32_ty);

  // Adjust exponent bias (expBias = dstExpBias - srcExpBias = 127 - 8 = 119)
  Value d0 = b.fmul(f32_ty, c0, b.f32_val(0x1p+119));
  Value d1 = b.fmul(f32_ty, c1, b.f32_val(0x1p+119));
  d0 = b.bitcast(d0, i32_ty);
  d1 = b.bitcast(d1, i32_ty);

  // Add the signs and place the halfwords in the proper place in order to
  // pack them
  constexpr uint32_t signHalfwordMask = signMask | (signMask << 16U);
  Value out0 = b.or_(i32_ty, b.lshr(i32_ty, d0, b.i32_val(16)), d1);
  Value sign0 = b.and_(i32_ty, a0, b.i32_val(signHalfwordMask));
  out0 = b.or_(i32_ty, out0, sign0);

  // Unpack the 2 bfloat16 values and return them
  auto bf16x2VecTy = vec_ty(bf16_ty, 2);
  out0 = b.bitcast(out0, bf16x2VecTy);
  Value r0 = b.extract_element(bf16_ty, out0, b.i32_val(0));
  Value r1 = b.extract_element(bf16_ty, out0, b.i32_val(1));
  // 0x80 is the fnuz NaN; the bit-trick above makes it -0.0, so map it to a
  // bf16 NaN per lane.
  Value bf16Nan = b.bitcast(b.i16_val(0x7FC0), bf16_ty);
  Value isNan0 = b.icmp_eq(v[0], b.i8_val(0x80));
  Value isNan1 = b.icmp_eq(v[1], b.i8_val(0x80));
  return {b.select(isNan0, bf16Nan, r0), b.select(isNan1, bf16Nan, r1)};
}

static SmallVector<Value>
Fp8E5M2fnuzToBf16SW(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 4);
  SmallVector<Value> fp16Vec = Fp8E5M2fnuzToFp16SW(loc, rewriter, v);
  SmallVector<Value> result(4);
  for (size_t i = 0; i < 4; i++) {
    Value fp32 = Fp16ToFp32OneValue(loc, rewriter, fp16Vec[i]);
    result[i] = AMD::convertFp32ToBf16(loc, rewriter, fp32, RoundingMode::RTZ);
  }
  return result;
}

// Fp8/Bf8 -> Fp32 through a software Fp8/Bf8 -> Fp16 conversion
template <CvtImpl ToFp16>
static SmallVector<Value> fp8ToFp32ViaFp16(Location loc,
                                           ConversionPatternRewriter &rewriter,
                                           const SmallVector<Value> &v) {
  SmallVector<Value> result = ToFp16(loc, rewriter, v);
  for (Value &x : result)
    x = Fp16ToFp32OneValue(loc, rewriter, x);
  return result;
}

//===----------------------------------------------------------------------===//
// Fp32 -> Fp16/Bf16 downcasts
//===----------------------------------------------------------------------===//

// Fp32 -> Fp16/Bf16 (RTNE). Two values are converted with a <2 x f32> fptrunc,
// which CDNA4 and later lower to the packed v_cvt_pk_[b]f16_f32; one value with
// a scalar fptrunc.
static SmallVector<Value>
convertFp32ToFp16rtne(Location loc, ConversionPatternRewriter &rewriter,
                      ArrayRef<Value> v, Type outElemTy) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (v.size() == 1)
    return {b.fptrunc(outElemTy, v.front())};

  assert(v.size() == 2);
  auto inVecTy = vec_ty(f32_ty, 2);
  auto retVecTy = vec_ty(outElemTy, 2);
  Value inVec = b.undef(inVecTy);
  auto idx0 = b.i32_val(0);
  auto idx1 = b.i32_val(1);
  inVec = b.insert_element(inVecTy, inVec, v[0], idx0);
  inVec = b.insert_element(inVecTy, inVec, v[1], idx1);
  Value retVec = b.fptrunc(retVecTy, inVec);
  SmallVector<Value> ret(2);
  ret[0] = b.extract_element(outElemTy, retVec, idx0);
  ret[1] = b.extract_element(outElemTy, retVec, idx1);
  return ret;
}

// Fp32 -> Fp16 (RTNE), one value at a time
static SmallVector<Value> Fp32ToFp16rtne(Location loc,
                                         ConversionPatternRewriter &rewriter,
                                         const SmallVector<Value> &v) {
  SmallVector<Value> result;
  for (Value x : v)
    result.push_back(Fp32ToFp16rtneOneValue(loc, rewriter, x));
  return result;
}

// Fp32 -> Fp16/Bf16 (RTNE) with the packed v_cvt_pk_[b]f16_f32 (see
// convertFp32ToFp16rtne)
template <typename DstFPType>
static SmallVector<Value>
Fp32ToFp16Bf16rtnePacked(Location loc, ConversionPatternRewriter &rewriter,
                         const SmallVector<Value> &v) {
  Type dstTy = DstFPType::get(rewriter.getContext());
  return convertFp32ToFp16rtne(loc, rewriter, v, dstTy);
}

// Fp32 -> Fp16 (RTZ)
static SmallVector<Value> Fp32ToFp16rtz(Location loc,
                                        ConversionPatternRewriter &rewriter,
                                        const SmallVector<Value> &v) {
  assert(v.size() == 2);

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type v2f16Ty = vec_ty(f16_ty, 2);

  Value result;
  result = ROCDL::CvtPkRtz::create(rewriter, loc, v2f16Ty, v[0], v[1]);
  SmallVector<Value> ret(2);
  auto idx0 = b.i32_val(0);
  auto idx1 = b.i32_val(1);
  ret[0] = b.extract_element(f16_ty, result, idx0);
  ret[1] = b.extract_element(f16_ty, result, idx1);
  return ret;
}

// Fp32 -> Bf16, one value
template <RoundingMode rounding>
static SmallVector<Value>
Fp32ToBf16OneValue(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  assert(v.size() == 1);
  return {AMD::convertFp32ToBf16(loc, rewriter, v[0], rounding)};
}

// Fp32 -> Bf16 (RTZ), packing two results with v_perm_b32
static SmallVector<Value>
Fp32ToBf16rtzPacked(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value f0 = b.bitcast(v[0], i32_ty);
  Value f1 = b.bitcast(v[1], i32_ty);
  Value sel = b.i32_val(0x07060302);
  Value packed =
      LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.amdgcn.perm", i32_ty,
                                      ValueRange{f1, f0, sel})
          .getResult(0);
  Value v2 = b.bitcast(packed, vec_ty(bf16_ty, 2));
  return {b.extract_element(v2, b.i32_val(0)),
          b.extract_element(v2, b.i32_val(1))};
}

//===----------------------------------------------------------------------===//
// Fp8 downcasts: hardware
//===----------------------------------------------------------------------===//

// Convert eight Fp16/Bf16/Fp32 values to OCP Fp8/Bf8 with one scaled
// v_cvt_scalef32_pk8_* (unit scale). Currently used on CDNA5 (see
// TargetFeatures::supportsCvtPkScalePk8).
template <typename ConvertOp>
static SmallVector<Value>
scalePk8DowncastToFp8(Location loc, ConversionPatternRewriter &rewriter,
                      const SmallVector<Value> &v) {
  const size_t inSize = 8;
  assert(v.size() == inSize);

  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Type vFPInTy = nullptr;
  if constexpr ((std::is_same_v<ConvertOp, ROCDL::CvtScaleF32Pk8Fp8F32Op>) ||
                (std::is_same_v<ConvertOp, ROCDL::CvtScaleF32Pk8Bf8F32Op>)) {
    vFPInTy = vec_ty(f32_ty, inSize);
  } else if constexpr ((std::is_same_v<ConvertOp,
                                       ROCDL::CvtScaleF32Pk8Fp8F16Op>) ||
                       (std::is_same_v<ConvertOp,
                                       ROCDL::CvtScaleF32Pk8Bf8F16Op>)) {
    vFPInTy = vec_ty(f16_ty, inSize);
  } else if constexpr ((std::is_same_v<ConvertOp,
                                       ROCDL::CvtScaleF32Pk8Fp8Bf16Op>) ||
                       (std::is_same_v<ConvertOp,
                                       ROCDL::CvtScaleF32Pk8Bf8Bf16Op>)) {
    vFPInTy = vec_ty(bf16_ty, inSize);
  }

  Type vFPResTy = nullptr;
  Type vFPOutTy = nullptr;
  if constexpr ((std::is_same_v<ConvertOp, ROCDL::CvtScaleF32Pk8Fp8F32Op>) ||
                (std::is_same_v<ConvertOp, ROCDL::CvtScaleF32Pk8Bf8F32Op>) ||
                (std::is_same_v<ConvertOp, ROCDL::CvtScaleF32Pk8Fp8F16Op>) ||
                (std::is_same_v<ConvertOp, ROCDL::CvtScaleF32Pk8Bf8F16Op>) ||
                (std::is_same_v<ConvertOp, ROCDL::CvtScaleF32Pk8Fp8Bf16Op>) ||
                (std::is_same_v<ConvertOp, ROCDL::CvtScaleF32Pk8Bf8Bf16Op>)) {
    vFPResTy = vec_ty(i32_ty, 2);
    vFPOutTy = vec_ty(i8_ty, inSize);
  }

  // make sure that all types were selected
  assert(vFPInTy && vFPResTy && vFPOutTy);

  // convert SmallVector to llvm vector
  Value inVec = b.undef(vFPInTy);
  for (size_t i = 0; i < inSize; ++i) {
    inVec = b.insert_element(vFPInTy, inVec, v[i], b.i32_val(i));
  }

  auto resVec = ConvertOp::create(rewriter, loc, vFPResTy, inVec, b.f32_val(1));
  auto outVec = b.bitcast(resVec, vFPOutTy);

  // convert llvm vector to SmallVector
  SmallVector<Value> result(inSize);
  for (size_t i = 0; i < inSize; i++) {
    result[i] = b.extract_element(i8_ty, outVec, b.i32_val(i));
  }
  return result;
}

// Convert four Fp16/Bf16/Fp32 values to OCP Fp8/Bf8 with the scaled
// v_cvt_scalef32_pk_* (two values per instruction, unit scale). Currently used
// on CDNA4 (see TargetFeatures::supportsCvtScaleF32Pk).
template <typename ConvertOp>
static SmallVector<Value>
scalePk4DowncastToFp8(Location loc, ConversionPatternRewriter &rewriter,
                      const SmallVector<Value> &v) {
  assert(v.size() == 4);
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Type v2I16Ty = vec_ty(i16_ty, 2);
  Value v2I16Vec = b.undef(v2I16Ty);
  Value scale = b.f32_val(1);

  if constexpr (std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkFp8F32Op> ||
                std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkBf8F32Op>) {
    v2I16Vec =
        ConvertOp::create(rewriter, loc, v2I16Ty, v2I16Vec, v[0], v[1], scale,
                          /*dstLoHiSel=*/false);
    v2I16Vec =
        ConvertOp::create(rewriter, loc, v2I16Ty, v2I16Vec, v[2], v[3], scale,
                          /*dstLoHiSel=*/true);
  } else {
    Type v2F16Ty = vec_ty(v[0].getType(), 2);
    Value srcVec = b.undef(v2F16Ty);
    auto idx0 = b.i32_val(0);
    auto idx1 = b.i32_val(1);
    srcVec = b.insert_element(v2F16Ty, srcVec, v[0], idx0);
    srcVec = b.insert_element(v2F16Ty, srcVec, v[1], idx1);
    v2I16Vec =
        ConvertOp::create(rewriter, loc, v2I16Ty, v2I16Vec, srcVec, scale,
                          /*dstLoHiSel=*/false);
    srcVec = b.insert_element(v2F16Ty, srcVec, v[2], idx0);
    srcVec = b.insert_element(v2F16Ty, srcVec, v[3], idx1);
    v2I16Vec =
        ConvertOp::create(rewriter, loc, v2I16Ty, v2I16Vec, srcVec, scale,
                          /*dstLoHiSel=*/true);
  }

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  auto fp8x4Vec = b.bitcast(v2I16Vec, fp8x4VecTy);
  SmallVector<Value> ret(4);
  for (size_t i = 0; i < 4; i++) {
    auto idx = b.i32_val(i);
    ret[i] = b.extract_element(i8_ty, fp8x4Vec, idx);
  }
  return ret;
}

// Convert four Fp32 values to packed Fp8/Bf8 with the unscaled cvt
// (v_cvt_pk_{fp8,bf8}_f32). ConvertOp chooses Fp8 versus Bf8 and the target
// decides the format: FNUZ on CDNA3, OCP on RDNA4 (see
// TargetFeatures::supportsCvtFp8Fnuz and TargetFeatures::supportsCvtFp8Ocp).
template <typename ConvertOp>
static SmallVector<Value> Pk4Fp32ToF8(Location loc,
                                      ConversionPatternRewriter &rewriter,
                                      bool saturateInf, SmallVector<Value> v) {
  assert(v.size() == 4);
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  // The fp16_ovfl mode bit (see adjustModeRegister) saturates finite overflow,
  // but the unscaled cvt keeps infinities: OCP fp8 turns them into NaN and bf8
  // keeps them as inf. OCP saturation maps +-inf to +-max as well, so clamp the
  // inputs first; the NaN-propagating minimum/maximum keep NaNs intact. FNUZ
  // formats have no infinity and convert it to NaN, which is what we want.
  if (saturateInf) {
    constexpr bool isBf8 = std::is_same_v<ConvertOp, ROCDL::CvtPkBf8F32Op>;
    const llvm::fltSemantics &dstSemantic =
        isBf8 ? llvm::APFloat::Float8E5M2() : llvm::APFloat::Float8E4M3FN();
    float dstMax = llvm::APFloat::getLargest(dstSemantic).convertToFloat();
    Value maxVal = b.f32_val(dstMax);
    Value minVal = b.f32_val(-dstMax);
    for (Value &x : v) {
      x = LLVM::MinimumOp::create(rewriter, loc, x, maxVal);
      x = LLVM::MaximumOp::create(rewriter, loc, x, minVal);
    }
  }

  Value result = b.undef(i32_ty);

  result = ConvertOp::create(rewriter, loc, i32_ty, v[0], v[1], result,
                             /*wordSel=*/false);
  result = ConvertOp::create(rewriter, loc, i32_ty, v[2], v[3], result,
                             /*wordSel=*/true);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  auto fp8x4Vec = b.bitcast(result, fp8x4VecTy);
  SmallVector<Value> ret(4);
  for (size_t i = 0; i < 4; i++) {
    auto idx = b.i32_val(i);
    ret[i] = b.extract_element(i8_ty, fp8x4Vec, idx);
  }
  return ret;
}

// Fp32 -> Fp8/Bf8 via the unscaled cvt
template <typename DstFPType>
static SmallVector<Value> unscaledFp32ToFp8(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            const SmallVector<Value> &v) {
  using ConvertOp = std::conditional_t<
      llvm::is_one_of<DstFPType, Float8E4M3FNType, Float8E4M3FNUZType>::value,
      ROCDL::CvtPkFp8F32Op, ROCDL::CvtPkBf8F32Op>;
  constexpr bool isOcp =
      llvm::is_one_of<DstFPType, Float8E4M3FNType, Float8E5M2Type>::value;
  return Pk4Fp32ToF8<ConvertOp>(loc, rewriter, /*saturateInf=*/isOcp, v);
}

// Fp16/Bf16 -> Fp8/Bf8 via the unscaled cvt
template <typename SrcFPType, typename DstFPType>
static SmallVector<Value>
unscaledToFp8ViaFp32(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &v) {
  assert(v.size() == 4);
  return unscaledFp32ToFp8<DstFPType>(loc, rewriter,
                                      extToFp32<SrcFPType>(loc, rewriter, v));
}

//===----------------------------------------------------------------------===//
// Fp8 downcasts: software
//===----------------------------------------------------------------------===//

// Downcast one Fp32, Fp16 or Bf16 value to OCP Fp8 (E4M3FN) or to an FNUZ
// Fp8/Bf8 format, in saturation and round-to-nearest-even mode. OCP Bf8 (E5M2)
// has its own routines. According to
// https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1,
// in saturation mode inf and out-of-range numbers are converted to the largest
// normal number, i.e. ±448 for E4M3FN. NaNs are converted to NaNs.
// FNUZ formats have no infinity, so inf converts to NaN; see
// https://onnx.ai/onnx/technical/float8.html
template <typename SrcFPType, typename DstFPType>
  requires llvm::is_one_of<SrcFPType, Float32Type, Float16Type,
                           BFloat16Type>::value &&
           llvm::is_one_of<DstFPType, Float8E4M3FNType, Float8E4M3FNUZType,
                           Float8E5M2FNUZType>::value
static Value downcastToFp8rtneOneValue(Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       Value v) {
  constexpr bool isFp8UZ =
      llvm::is_one_of<DstFPType, Float8E4M3FNUZType, Float8E5M2FNUZType>::value;
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  FPTypeInfo<SrcFPType> srcFpInfo(loc, rewriter);
  FPTypeInfo<DstFPType> dstFpInfo(loc, rewriter);

  const llvm::fltSemantics &srcSemantic = srcFpInfo.getFPSemantics();
  auto srcWidth = llvm::APFloat::getSizeInBits(srcSemantic);
  auto srcMantissaBits = llvm::APFloat::semanticsPrecision(srcSemantic) - 1;
  auto srcExponentBits = srcWidth - srcMantissaBits - 1;
  auto srcBias = (1 << (srcExponentBits - 1)) - 1;

  const llvm::fltSemantics &dstSemantic = dstFpInfo.getFPSemantics();
  auto dstWidth = llvm::APFloat::getSizeInBits(dstSemantic);
  auto dstMantissaBits = llvm::APFloat::semanticsPrecision(dstSemantic) - 1;
  auto dstExponentBits = dstWidth - dstMantissaBits - 1;
  auto dstBias = (1 << (dstExponentBits - 1)) - 1;
  if (isFp8UZ) {
    dstBias++;
  }

  auto srcIntType = srcFpInfo.getIntType();
  Value isNaN = checkIsNan(b, v);

  uint32_t reducedMantissaBits = srcMantissaBits - dstMantissaBits;
  Value reducedMantissaValue = srcFpInfo.toLLVMIntValue(reducedMantissaBits);

  // Get sign and absolute value
  Value intVal = b.bitcast(v, srcIntType);
  uint32_t signMask = 1 << (srcWidth - 1);
  Value sign =
      b.trunc(i8_ty, b.lshr(b.and_(intVal, srcFpInfo.toLLVMIntValue(signMask)),
                            srcFpInfo.toLLVMIntValue(srcWidth - 8)));

  uint32_t absoluteMask = signMask - 1;
  intVal = b.and_(intVal, srcFpInfo.toLLVMIntValue(absoluteMask));

  // Rounding to nearest even
  uint32_t baseRoundingBias = (1 << (reducedMantissaBits - 1)) - 1;

  // For Fp16, S.EEEEE.MMMMMMMMMM => 0.00000.00M0000000 => 0.00000.000000000M
  uint32_t mantissaLSB = 1 << reducedMantissaBits;
  Value mantissaLSBValue = srcFpInfo.toLLVMIntValue(mantissaLSB);
  Value remainingMantissaLSB =
      b.lshr(b.and_(intVal, mantissaLSBValue), reducedMantissaValue);
  Value roundingBias =
      b.add(remainingMantissaLSB, srcFpInfo.toLLVMIntValue(baseRoundingBias));
  Value vFp8 = b.add(intVal, roundingBias);

  // Reduce mantissa to number of bits of the destination format
  // Example: For Fp16 to FP8E4M3FN, reduceMantissaMask == 1.11111.1110000000
  uint32_t reduceMantissaMask =
      ((1 << (1 + srcExponentBits + dstMantissaBits + 1)) - 1)
      << reducedMantissaBits;
  Value reduceMantissa = srcFpInfo.toLLVMIntValue(reduceMantissaMask);
  vFp8 = b.and_(vFp8, reduceMantissa);

  // We round numbers smaller than the minimal normal number in Fp8 to make
  // it easier to handle subnormals
  auto dstSmallest = llvm::APFloat::getSmallestNormalized(dstSemantic);
  // Get the srcFpType representation of the minimal normal number in Fp8
  bool losesInfo;
  dstSmallest.convert(srcSemantic, APFloat::rmNearestTiesToEven, &losesInfo);
  uint32_t dstMinimal =
      static_cast<uint32_t>(dstSmallest.bitcastToAPInt().getZExtValue());
  vFp8 = b.umax(vFp8, srcFpInfo.toLLVMIntValue(dstMinimal));

  // Adjust exponent bias
  uint32_t expBias = (srcBias - dstBias) << srcMantissaBits;
  vFp8 = b.sub(vFp8, srcFpInfo.toLLVMIntValue(expBias));

  // Shift right and truncate
  vFp8 = b.trunc(i8_ty, b.lshr(vFp8, reducedMantissaValue));

  // Any numbers larger than the max normal number(including infinity) in FP8
  // after rounding will cause overflow
  auto dstLargest = llvm::APFloat::getLargest(dstSemantic);
  uint32_t dstMaxPositive =
      static_cast<uint32_t>(dstLargest.bitcastToAPInt().getZExtValue());
  // Get the srcFpType representation of the maximal normal number in Fp8
  dstLargest.convert(srcSemantic, APFloat::rmNearestTiesToEven, &losesInfo);
  uint32_t dstMaxOfSrcType =
      static_cast<uint32_t>(dstLargest.bitcastToAPInt().getZExtValue());

  // For Fp16, 0x5F7F == 0.10111.1101111111 is the largest possible normal
  // number(including infinity) after rounding in FP8E4M3
  // For Fp8 UZ types, conversion with saturation converts infinity to NaN
  if constexpr (!isFp8UZ) {
    // Include infinity
    if constexpr (std::is_same_v<SrcFPType, Float32Type>)
      dstMaxOfSrcType |= 0x7ffff;
    else if constexpr (std::is_same_v<SrcFPType, Float16Type>)
      dstMaxOfSrcType |= 0x7f;
    else
      dstMaxOfSrcType |= 0x7;
  } else {
    uint32_t expFullMask = ((1U << srcExponentBits) - 1U) << srcMantissaBits;
    // In case the exponent is full (all ones), then we have either a NaN or Inf
    Value isNaNOrInf =
        b.icmp_eq(b.and_(intVal, srcFpInfo.toLLVMIntValue(expFullMask)),
                  srcFpInfo.toLLVMIntValue(expFullMask));
    isNaN = isNaNOrInf;
  }

  Value isOverflow =
      b.icmp_ugt(intVal, srcFpInfo.toLLVMIntValue(dstMaxOfSrcType));
  vFp8 = b.select(isOverflow, dstFpInfo.toLLVMIntValue(dstMaxPositive), vFp8);

  // Round subnormals to nearest even. Ref:
  // https://github.com/openxla/xla/blob/f20c6fe2/xla/service/elemental_ir_emitter.cc#L272
  auto dstTyID = TypeID::get<DstFPType>();
  auto halfwayPointsLUT = srcFpInfo.getHalfwayPointsForDstType(dstTyID);
  size_t lutSize = halfwayPointsLUT.size();

  for (int i = lutSize - 1; i >= 0; i--) {
    Value cmp;
    if (i % 2 == 0) {
      cmp = b.icmp_ule(intVal, srcFpInfo.toLLVMIntValue(halfwayPointsLUT[i]));
    } else {
      cmp = b.icmp_ult(intVal, srcFpInfo.toLLVMIntValue(halfwayPointsLUT[i]));
    }

    vFp8 = b.select(cmp, b.i8_val(i), vFp8);
  }

  int32_t positiveNan = 0;
  if constexpr (isFp8UZ) {
    // Only one NaN value which is represented with sign = 1
    positiveNan = (1 << (dstExponentBits + dstMantissaBits));
  } else {
    positiveNan = (1 << (dstExponentBits + dstMantissaBits)) - 1;
  }

  // NaN remains NaN after conversion
  vFp8 = b.select(isNaN, dstFpInfo.toLLVMIntValue(positiveNan), vFp8);

  // Set sign bit
  vFp8 = b.or_(vFp8, sign);
  // In UZ formats there is only 1 zero (positive zero)
  // Correct negative zero to 0
  if constexpr (isFp8UZ) {
    Value isNegativeZero =
        b.and_(b.icmp_eq(vFp8, b.i8_val(0x80)), b.icmp_eq(isNaN, b.i1_val(0)));
    vFp8 = b.select(isNegativeZero, b.i8_val(0), vFp8);
  }

  return vFp8;
}

// Fp32/Fp16/Bf16 -> OCP Fp8 or FNUZ Fp8/Bf8 (RTNE), one value at a time
template <typename SrcFPType, typename DstFPType>
static SmallVector<Value> swDowncastToFp8(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          const SmallVector<Value> &v) {
  SmallVector<Value> result;
  for (Value x : v)
    result.push_back(
        downcastToFp8rtneOneValue<SrcFPType, DstFPType>(loc, rewriter, x));
  return result;
}

// Fp16 -> FNUZ Bf8 (RTNE)
static SmallVector<Value>
Fp16ToFp8E5M2fnuzSW(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return swDowncastToFp8<Float32Type, Float8E5M2FNUZType>(
      loc, rewriter, extToFp32<Float16Type>(loc, rewriter, v));
}

// Fp16 -> OCP Bf8 (RTNE)
static SmallVector<Value>
Fp16ToFp8E5M2rtneSW(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {

  assert(v.size() == 4);
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  SmallVector<Value> result(4);
  for (size_t i = 0; i < 4; ++i) {
    Value fp16 = v[i];
    Value i16 = b.bitcast(fp16, i16_ty);

    Value s = b.and_(i16_ty, i16, b.i16_val(0x8000));
    Value exp =
        b.and_(i16_ty, b.lshr(i16_ty, i16, b.i16_val(10)), b.i16_val(0x1F));
    Value man = b.and_(i16_ty, i16, b.i16_val(0x03FF));
    Value sig = b.and_(i16_ty, i16, b.i16_val(0x7FFF));

    // Round 10-bit mantissa to 2-bit nearest, ties to even
    Value bias = b.add(
        i16_ty,
        b.lshr(i16_ty, b.and_(i16_ty, sig, b.i16_val(0x0100)), b.i16_val(8)),
        b.i16_val(0x007F));
    i16 = b.add(i16_ty, sig, bias);

    // Handle overflow using saturation mode, by setting sig to be the max.
    // Any number equal or larger than 0x7B80 after rounding (including
    // infinite 0x7C00) will cause overflow
    i16 = b.select(b.icmp_uge(sig, b.i16_val(0x7B80)), b.i16_val(0x7B00), i16);

    // Handle NaN value by keeping it Nan
    i16 = b.select(
        b.and_(b.icmp_eq(exp, b.i16_val(0x1F)), b.icmp_ne(man, b.i16_val(0x0))),
        b.i16_val(0x7E00), i16);

    // Add sign bit
    i16 = b.or_(i16_ty, s, i16);

    // Truncate to 8-bit
    result[i] = b.trunc(i8_ty, b.lshr(i16_ty, i16, b.i16_val(8)));
  }

  return result;
}

// Bf16 -> OCP Bf8 (RTNE)
static SmallVector<Value> Bf16ToFp8E5M2SW(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          const SmallVector<Value> &v) {
  assert(v.size() == 4);
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  SmallVector<Value> result(4);
  for (size_t i = 0; i < 4; ++i) {
    Value fp16 = v[i];
    Value i16 = b.bitcast(fp16, i16_ty);

    Value s = b.and_(i16_ty, i16, b.i16_val(0x8000));
    Value exp =
        b.and_(i16_ty, b.lshr(i16_ty, i16, b.i16_val(7)), b.i16_val(0xFF));
    Value man = b.and_(i16_ty, i16, b.i16_val(0x7F));

    // Convert 8-bit exponent to 5-bit exponent
    Value exp5 = b.select(b.icmp_ult(exp, b.i16_val(0x71)), b.i16_val(0),
                          b.sub(i16_ty, exp, b.i16_val(0x70)));

    // Handle subnormal values (exp5 = 0)
    // - exp <  0x6e: mantissa = 0x0000 (0)
    // - exp == 0x6e: mantissa = 0x0000 (0),
    //                           0x0020 (1/4)
    // - exp == 0x6f: mantissa = 0x0020 (1/4),
    //                           0x0040 (1/2)
    // - exp == 0x70: mantissa = 0x0040 (1/2),
    //                           0x0060 (3/4),
    //                           0x0080 (1)
    man = b.select(b.icmp_ult(exp, b.i16_val(0x6e)), b.i16_val(0), man);
    man = b.select(
        b.icmp_eq(exp, b.i16_val(0x6e)),
        b.select(b.icmp_ne(man, b.i16_val(0)), b.i16_val(0x0020), b.i16_val(0)),
        man);
    man = b.select(b.icmp_eq(exp, b.i16_val(0x6f)),
                   b.select(b.icmp_uge(man, b.i16_val(0x0040)),
                            b.i16_val(0x0040), b.i16_val(0x0020)),
                   man);
    man = b.select(b.icmp_eq(exp, b.i16_val(0x70)),
                   b.select(b.icmp_ugt(man, b.i16_val(0x0020)),
                            b.select(b.icmp_uge(man, b.i16_val(0x0060)),
                                     b.i16_val(0x0080), b.i16_val(0x0060)),
                            b.i16_val(0x0040)),
                   man);

    // Round 7-bit mantissa to 2-bit
    Value sig = b.or_(i16_ty, b.shl(i16_ty, exp5, b.i16_val(7)), man);
    Value bias = b.add(
        i16_ty,
        b.lshr(i16_ty, b.and_(i16_ty, sig, b.i16_val(0x0020)), b.i16_val(5)),
        b.i16_val(0x000F));
    i16 = b.add(i16_ty, sig, bias);

    // Handle overflow using saturation mode, by setting sig to be the max.
    // Overflow will happe for the following cases:
    // - Any number equal or larger than 0x0F70 after rounding
    // - Exponent larged than 0x8E (including infinite 0xFF)
    i16 = b.select(b.or_(b.icmp_ugt(exp, b.i16_val(0x8E)),
                         b.icmp_uge(sig, b.i16_val(0x0F70))),
                   b.i16_val(0x0F7F), i16);

    // Handle NaN value by keeping it Nan
    i16 = b.select(
        b.and_(b.icmp_eq(exp, b.i16_val(0xFF)), b.icmp_ne(man, b.i16_val(0x0))),
        b.i16_val(0x0FC0), i16);

    // Add sign bit
    i16 = b.or_(i16_ty, b.lshr(i16_ty, s, b.i16_val(3)), i16);

    // Truncate to 8-bit
    result[i] = b.trunc(i8_ty, b.lshr(i16_ty, i16, b.i16_val(5)));
  }

  return result;
}

// Fp32 -> OCP Bf8 (RTNE)
static SmallVector<Value>
Fp32ToFp8E5M2rtneSW(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 4);
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  SmallVector<Value> result(4);
  for (size_t i = 0; i < 4; ++i) {
    Value fp32 = v[i];
    Value i32 = b.bitcast(fp32, i32_ty);

    Value s = b.and_(i32_ty, i32, b.i32_val(0x80000000));
    Value exp =
        b.and_(i32_ty, b.lshr(i32_ty, i32, b.i32_val(23)), b.i32_val(0xFF));
    Value man = b.and_(i32_ty, i32, b.i32_val(0x007FFFFF));

    // Convert 8-bit exponent to 5-bit
    Value exp5 = b.select(b.icmp_ult(exp, b.i32_val(0x71)), b.i32_val(0),
                          b.sub(i32_ty, exp, b.i32_val(0x70)));

    // Handle subnormal values (exp5 = 0)
    // - exp <  0x6e: mantissa = 0x00000000 (0)
    // - exp == 0x6e: mantissa = 0x00000000 (0),
    //                           0x00200000 (1/4)
    // - exp == 0x6f: mantissa = 0x00200000 (1/4),
    //                           0x00400000 (1/2)
    // - exp == 0x70: mantissa = 0x00400000 (1/2),
    //                           0x00600000 (3/4),
    //                           0x00800000 (1)
    man = b.select(b.icmp_ult(exp, b.i32_val(0x6e)), b.i32_val(0), man);
    man = b.select(b.icmp_eq(exp, b.i32_val(0x6e)),
                   b.select(b.icmp_ne(man, b.i32_val(0)), b.i32_val(0x00200000),
                            b.i32_val(0)),
                   man);
    man = b.select(b.icmp_eq(exp, b.i32_val(0x6f)),
                   b.select(b.icmp_uge(man, b.i32_val(0x00400000)),
                            b.i32_val(0x00400000), b.i32_val(0x00200000)),
                   man);
    man = b.select(
        b.icmp_eq(exp, b.i32_val(0x70)),
        b.select(b.icmp_ugt(man, b.i32_val(0x00200000)),
                 b.select(b.icmp_uge(man, b.i32_val(0x00600000)),
                          b.i32_val(0x00800000), b.i32_val(0x00600000)),
                 b.i32_val(0x00400000)),
        man);

    // Round 23-bit mantissa to 2-bit nearest, ties to even
    Value sig = b.or_(i32_ty, b.shl(i32_ty, exp5, b.i32_val(23)), man);
    Value bias =
        b.add(i32_ty,
              b.lshr(i32_ty, b.and_(i32_ty, sig, b.i32_val(0x00200000)),
                     b.i32_val(21)),
              b.i32_val(0x000FFFFF));
    i32 = b.add(i32_ty, sig, bias);

    // Handle overflow using saturation mode, by setting sig to be the max.
    // Overflow will happe for the following cases:
    // - Any number equal or larger than 0x0F700000 after rounding
    // - Exponent larged than 0x8E (including infinite 0xFF)
    i32 = b.select(b.or_(b.icmp_ugt(exp, b.i32_val(0x8E)),
                         b.icmp_uge(sig, b.i32_val(0x0F700000))),
                   b.i32_val(0x0F7FFFFF), i32);

    // Handle NaN value by keeping it Nan
    i32 = b.select(
        b.and_(b.icmp_eq(exp, b.i32_val(0xFF)), b.icmp_ne(man, b.i32_val(0x0))),
        b.i32_val(0x0FC00000), i32);

    // Add sign bit
    i32 = b.or_(i32_ty, b.lshr(i32_ty, s, b.i32_val(3)), i32);

    // Truncate to 8-bit
    result[i] = b.trunc(i8_ty, b.lshr(i32_ty, i32, b.i32_val(21)));
  }
  return result;
}

// Fp16 -> OCP Bf8 (RTZ)
static SmallVector<Value> Fp16ToFp8E5M2rtz(Location loc,
                                           ConversionPatternRewriter &rewriter,
                                           const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec0 = b.undef(fp16x2VecTy);
  Value fp16x2Vec1 = b.undef(fp16x2VecTy);
  fp16x2Vec0 = b.insert_element(fp16x2VecTy, fp16x2Vec0, v[0], b.i32_val(0));
  fp16x2Vec0 = b.insert_element(fp16x2VecTy, fp16x2Vec0, v[1], b.i32_val(1));
  fp16x2Vec1 = b.insert_element(fp16x2VecTy, fp16x2Vec1, v[2], b.i32_val(0));
  fp16x2Vec1 = b.insert_element(fp16x2VecTy, fp16x2Vec1, v[3], b.i32_val(1));

  Value a0 = b.bitcast(fp16x2Vec0, i32_ty);
  Value a1 = b.bitcast(fp16x2Vec1, i32_ty);

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  a0 = b.bitcast(a0, fp8x4VecTy);
  a1 = b.bitcast(a1, fp8x4VecTy);

  return {b.extract_element(i8_ty, a0, b.i32_val(1)),
          b.extract_element(i8_ty, a0, b.i32_val(3)),
          b.extract_element(i8_ty, a1, b.i32_val(1)),
          b.extract_element(i8_ty, a1, b.i32_val(3))};
}

// Fp32 -> OCP Bf8 (RTZ), through Fp16 (RTZ)
static SmallVector<Value> Fp32ToFp8E5M2rtz(Location loc,
                                           ConversionPatternRewriter &rewriter,
                                           const SmallVector<Value> &v) {
  assert(v.size() == 4);
  SmallVector<Value> lo = Fp32ToFp16rtz(loc, rewriter, {v[0], v[1]});
  SmallVector<Value> hi = Fp32ToFp16rtz(loc, rewriter, {v[2], v[3]});
  return Fp16ToFp8E5M2rtz(loc, rewriter, {lo[0], lo[1], hi[0], hi[1]});
}

//===----------------------------------------------------------------------===//
// Conversion rules
//===----------------------------------------------------------------------===//

namespace {
// Target feature that gates a rule. Rules are selected only through these
// features, never through the ISA family, so supporting a new architecture
// only requires describing its features in TargetFeatures.
using FpCvtFeature = bool (TargetFeatures::*)() const;

// Rounding modes accepted by a rule.
enum class CvtRounding {
  RTNE,
  RTZ,
  // Any rounding mode; used by exact conversions.
  Any,
};

// One way of lowering a (src, dst, rounding) conversion.
struct FpCvtRule {
  TypeID srcTy;
  TypeID dstTy;
  CvtRounding rounding;
  // Target feature this rule requires, or nullptr if it requires none and is
  // therefore available on every target.
  FpCvtFeature feature;
  // Number of values converted by one call of `convert`.
  size_t numElements;
  // Only use the rule if a thread holds at least `numElements` contiguous
  // values. Rules are always correct with fewer values, since missing inputs
  // are padded with undef and their outputs dropped. Set this only for rules
  // that are profitable just when fully filled and are followed by a cheaper
  // rule for fewer values.
  bool needsContiguousElements;
  CvtImpl convert;
};
} // namespace

static bool acceptsRounding(CvtRounding accepted,
                            std::optional<RoundingMode> rounding) {
  switch (accepted) {
  case CvtRounding::RTNE:
    return rounding == RoundingMode::RTNE;
  case CvtRounding::RTZ:
    return rounding == RoundingMode::RTZ;
  case CvtRounding::Any:
    return true;
  }
  llvm_unreachable("unknown CvtRounding");
}

// Whether every finite value of `srcTy` is exactly representable in `dstTy`,
// so the conversion never rounds. Unlike APFloat::isRepresentableBy, this
// accounts for source normals that land in the destination's subnormal range
// (e.g. f8E5M2FNUZ -> f16).
[[maybe_unused]] static bool isExactConversion(Type srcTy, Type dstTy) {
  const llvm::fltSemantics &src = cast<FloatType>(srcTy).getFloatSemantics();
  const llvm::fltSemantics &dst = cast<FloatType>(dstTy).getFloatSemantics();
  // Exponent of the smallest subnormal.
  auto minUlpExponent = [](const llvm::fltSemantics &sem) {
    return llvm::APFloat::semanticsMinExponent(sem) -
           static_cast<int>(llvm::APFloat::semanticsPrecision(sem)) + 1;
  };
  return llvm::APFloat::semanticsMaxExponent(src) <=
             llvm::APFloat::semanticsMaxExponent(dst) &&
         llvm::APFloat::semanticsPrecision(src) <=
             llvm::APFloat::semanticsPrecision(dst) &&
         minUlpExponent(src) >= minUlpExponent(dst);
}

template <typename SrcFPType, typename DstFPType>
static FpCvtRule rule(CvtRounding rounding, FpCvtFeature feature,
                      size_t numElements, CvtImpl convert,
                      bool needsContiguousElements = false) {
  return {TypeID::get<SrcFPType>(),
          TypeID::get<DstFPType>(),
          rounding,
          feature,
          numElements,
          needsContiguousElements,
          convert};
}

// All supported conversions. For a given (src, dst, rounding) the first rule
// whose feature is available on the target is used, so rules are listed from
// most to least preferred and every conversion ends with a rule that is always
// available.
static const SmallVector<FpCvtRule> &getFpCvtRules() {
  using F32 = Float32Type;
  using F16 = Float16Type;
  using BF16 = BFloat16Type;
  using Fp8 = Float8E4M3FNType;
  using Bf8 = Float8E5M2Type;
  using Fp8Fnuz = Float8E4M3FNUZType;
  using Bf8Fnuz = Float8E5M2FNUZType;

  constexpr CvtRounding Exact = CvtRounding::Any;
  constexpr CvtRounding RTNE = CvtRounding::RTNE;
  constexpr CvtRounding RTZ = CvtRounding::RTZ;

  constexpr FpCvtFeature Always = nullptr;
  constexpr FpCvtFeature FnuzHW = &TargetFeatures::supportsCvtFp8Fnuz;
  constexpr FpCvtFeature OcpHW = &TargetFeatures::supportsCvtFp8Ocp;
  constexpr FpCvtFeature ScaledPk4 = &TargetFeatures::supportsCvtScaleF32Pk;
  constexpr FpCvtFeature ScaledPk8Up =
      &TargetFeatures::supportsCvtPkScalePk8Upcast;
  constexpr FpCvtFeature ScaledPk8Down = &TargetFeatures::supportsCvtPkScalePk8;
  constexpr FpCvtFeature PkFp32ToFp16 =
      &TargetFeatures::supportsCvtPkFp32ToFp16Bf16;

  static const SmallVector<FpCvtRule> rules = {
      // Fp8 -> Fp16
      rule<Fp8Fnuz, F16>(Exact, FnuzHW, 4, unscaledFp8ToFp16<Fp8Fnuz>),
      rule<Fp8Fnuz, F16>(Exact, Always, 4, Fp8E4M3fnuzToFp16SW),
      rule<Fp8, F16>(Exact, ScaledPk8Up, 8,
                     scalePk8UpcastFromFp8<ROCDL::CvtPkScalePk8F16Fp8Op>),
      rule<Fp8, F16>(Exact, ScaledPk4, 4,
                     scalePk4UpcastFromFp8<ROCDL::CvtScaleF32PkF16Fp8Op>),
      rule<Fp8, F16>(Exact, OcpHW, 4, unscaledFp8ToFp16<Fp8>),
      rule<Fp8, F16>(Exact, Always, 4, Fp8E4M3fnToFp16SW),
      rule<Bf8Fnuz, F16>(Exact, FnuzHW, 4, unscaledFp8ToFp16<Bf8Fnuz>),
      rule<Bf8Fnuz, F16>(Exact, Always, 4, Fp8E5M2fnuzToFp16SW),
      rule<Bf8, F16>(Exact, ScaledPk8Up, 8,
                     scalePk8UpcastFromFp8<ROCDL::CvtPkScalePk8F16Bf8Op>),
      rule<Bf8, F16>(Exact, ScaledPk4, 4,
                     scalePk4UpcastFromFp8<ROCDL::CvtScaleF32PkF16Bf8Op>),
      rule<Bf8, F16>(Exact, Always, 4, Fp8E5M2ToFp16SW),

      // Fp8 -> Bf16
      rule<Fp8Fnuz, BF16>(Exact, FnuzHW, 4, unscaledFp8ToBf16<Fp8Fnuz>),
      rule<Fp8Fnuz, BF16>(Exact, Always, 2, Fp8E4M3fnuzToBf16SW),
      rule<Fp8, BF16>(Exact, ScaledPk8Up, 8,
                      scalePk8UpcastFromFp8<ROCDL::CvtPkScalePk8Bf16Fp8Op>),
      rule<Fp8, BF16>(Exact, ScaledPk4, 4,
                      scalePk4UpcastFromFp8<ROCDL::CvtScaleF32PkBf16Fp8Op>),
      rule<Fp8, BF16>(Exact, OcpHW, 4, unscaledFp8ToBf16<Fp8>),
      rule<Fp8, BF16>(Exact, Always, 4, OcpF8ToBf16SW<Fp8>),
      rule<Bf8Fnuz, BF16>(Exact, FnuzHW, 4, unscaledFp8ToBf16<Bf8Fnuz>),
      rule<Bf8Fnuz, BF16>(Exact, Always, 4, Fp8E5M2fnuzToBf16SW),
      rule<Bf8, BF16>(Exact, ScaledPk8Up, 8,
                      scalePk8UpcastFromFp8<ROCDL::CvtPkScalePk8Bf16Bf8Op>),
      rule<Bf8, BF16>(Exact, ScaledPk4, 4,
                      scalePk4UpcastFromFp8<ROCDL::CvtScaleF32PkBf16Bf8Op>),
      rule<Bf8, BF16>(Exact, OcpHW, 4, unscaledFp8ToBf16<Bf8>),
      rule<Bf8, BF16>(Exact, Always, 4, OcpF8ToBf16SW<Bf8>),

      // Fp8 -> Fp32
      rule<Fp8Fnuz, F32>(Exact, FnuzHW, 4, unscaledFp8ToFp32<Fp8Fnuz>),
      rule<Fp8Fnuz, F32>(Exact, Always, 4,
                         fp8ToFp32ViaFp16<Fp8E4M3fnuzToFp16SW>),
      rule<Fp8, F32>(Exact, ScaledPk8Up, 8,
                     scalePk8UpcastFromFp8<ROCDL::CvtPkScalePk8F32Fp8Op>),
      rule<Fp8, F32>(Exact, ScaledPk4, 4,
                     scalePk4UpcastFromFp8<ROCDL::CvtScaleF32PkF32Fp8Op>),
      rule<Fp8, F32>(Exact, OcpHW, 4, unscaledFp8ToFp32<Fp8>),
      rule<Fp8, F32>(Exact, Always, 4, fp8ToFp32ViaFp16<Fp8E4M3fnToFp16SW>),
      rule<Bf8Fnuz, F32>(Exact, FnuzHW, 4, unscaledFp8ToFp32<Bf8Fnuz>),
      rule<Bf8Fnuz, F32>(Exact, Always, 4,
                         fp8ToFp32ViaFp16<Fp8E5M2fnuzToFp16SW>),
      rule<Bf8, F32>(Exact, ScaledPk8Up, 8,
                     scalePk8UpcastFromFp8<ROCDL::CvtPkScalePk8F32Bf8Op>),
      rule<Bf8, F32>(Exact, ScaledPk4, 4,
                     scalePk4UpcastFromFp8<ROCDL::CvtScaleF32PkF32Bf8Op>),
      rule<Bf8, F32>(Exact, OcpHW, 4, unscaledFp8ToFp32<Bf8>),
      rule<Bf8, F32>(Exact, Always, 4, fp8ToFp32ViaFp16<Fp8E5M2ToFp16SW>),

      // Fp16/Bf16 -> Fp32
      rule<F16, F32>(Exact, Always, 4, extToFp32<F16>),
      rule<BF16, F32>(Exact, Always, 4, extToFp32<BF16>),

      // Fp16 -> Fp8
      rule<F16, Fp8Fnuz>(RTNE, FnuzHW, 4, unscaledToFp8ViaFp32<F16, Fp8Fnuz>),
      rule<F16, Fp8Fnuz>(RTNE, Always, 2, swDowncastToFp8<F16, Fp8Fnuz>),
      rule<F16, Fp8>(RTNE, ScaledPk8Down, 8,
                     scalePk8DowncastToFp8<ROCDL::CvtScaleF32Pk8Fp8F16Op>),
      rule<F16, Fp8>(RTNE, ScaledPk4, 4,
                     scalePk4DowncastToFp8<ROCDL::CvtScaleF32PkFp8F16Op>),
      rule<F16, Fp8>(RTNE, OcpHW, 4, unscaledToFp8ViaFp32<F16, Fp8>),
      rule<F16, Fp8>(RTNE, Always, 4, swDowncastToFp8<F16, Fp8>),
      rule<F16, Bf8Fnuz>(RTNE, FnuzHW, 4, unscaledToFp8ViaFp32<F16, Bf8Fnuz>),
      rule<F16, Bf8Fnuz>(RTNE, Always, 2, Fp16ToFp8E5M2fnuzSW),
      rule<F16, Bf8>(RTNE, ScaledPk8Down, 8,
                     scalePk8DowncastToFp8<ROCDL::CvtScaleF32Pk8Bf8F16Op>),
      rule<F16, Bf8>(RTNE, ScaledPk4, 4,
                     scalePk4DowncastToFp8<ROCDL::CvtScaleF32PkBf8F16Op>),
      rule<F16, Bf8>(RTNE, OcpHW, 4, unscaledToFp8ViaFp32<F16, Bf8>),
      rule<F16, Bf8>(RTNE, Always, 4, Fp16ToFp8E5M2rtneSW),
      rule<F16, Bf8>(RTZ, Always, 4, Fp16ToFp8E5M2rtz),

      // Bf16 -> Fp8
      rule<BF16, Fp8Fnuz>(RTNE, FnuzHW, 4, unscaledToFp8ViaFp32<BF16, Fp8Fnuz>),
      rule<BF16, Fp8Fnuz>(RTNE, Always, 4, swDowncastToFp8<BF16, Fp8Fnuz>),
      rule<BF16, Fp8>(RTNE, ScaledPk8Down, 8,
                      scalePk8DowncastToFp8<ROCDL::CvtScaleF32Pk8Fp8Bf16Op>),
      rule<BF16, Fp8>(RTNE, ScaledPk4, 4,
                      scalePk4DowncastToFp8<ROCDL::CvtScaleF32PkFp8Bf16Op>),
      rule<BF16, Fp8>(RTNE, OcpHW, 4, unscaledToFp8ViaFp32<BF16, Fp8>),
      rule<BF16, Fp8>(RTNE, Always, 4, swDowncastToFp8<BF16, Fp8>),
      rule<BF16, Bf8Fnuz>(RTNE, FnuzHW, 4, unscaledToFp8ViaFp32<BF16, Bf8Fnuz>),
      rule<BF16, Bf8Fnuz>(RTNE, Always, 4, swDowncastToFp8<BF16, Bf8Fnuz>),
      rule<BF16, Bf8>(RTNE, ScaledPk8Down, 8,
                      scalePk8DowncastToFp8<ROCDL::CvtScaleF32Pk8Bf8Bf16Op>),
      rule<BF16, Bf8>(RTNE, ScaledPk4, 4,
                      scalePk4DowncastToFp8<ROCDL::CvtScaleF32PkBf8Bf16Op>),
      rule<BF16, Bf8>(RTNE, OcpHW, 4, unscaledToFp8ViaFp32<BF16, Bf8>),
      rule<BF16, Bf8>(RTNE, Always, 4, Bf16ToFp8E5M2SW),

      // Fp32 -> Fp16/Bf16
      rule<F32, F16>(RTNE, PkFp32ToFp16, 2, Fp32ToFp16Bf16rtnePacked<F16>),
      rule<F32, F16>(RTNE, Always, 2, Fp32ToFp16rtne),
      rule<F32, F16>(RTZ, Always, 2, Fp32ToFp16rtz),
      rule<F32, BF16>(RTNE, PkFp32ToFp16, 2, Fp32ToFp16Bf16rtnePacked<BF16>),
      rule<F32, BF16>(RTNE, Always, 1, Fp32ToBf16OneValue<RoundingMode::RTNE>),
      rule<F32, BF16>(RTZ, Always, 2, Fp32ToBf16rtzPacked,
                      /*needsContiguousElements=*/true),
      rule<F32, BF16>(RTZ, Always, 1, Fp32ToBf16OneValue<RoundingMode::RTZ>),

      // Fp32 -> Fp8
      rule<F32, Fp8Fnuz>(RTNE, FnuzHW, 4, unscaledFp32ToFp8<Fp8Fnuz>),
      rule<F32, Fp8Fnuz>(RTNE, Always, 2, swDowncastToFp8<F32, Fp8Fnuz>),
      rule<F32, Fp8>(RTNE, ScaledPk8Down, 8,
                     scalePk8DowncastToFp8<ROCDL::CvtScaleF32Pk8Fp8F32Op>),
      rule<F32, Fp8>(RTNE, ScaledPk4, 4,
                     scalePk4DowncastToFp8<ROCDL::CvtScaleF32PkFp8F32Op>),
      rule<F32, Fp8>(RTNE, OcpHW, 4, unscaledFp32ToFp8<Fp8>),
      rule<F32, Fp8>(RTNE, Always, 4, swDowncastToFp8<F32, Fp8>),
      rule<F32, Bf8Fnuz>(RTNE, FnuzHW, 4, unscaledFp32ToFp8<Bf8Fnuz>),
      rule<F32, Bf8Fnuz>(RTNE, Always, 2, swDowncastToFp8<F32, Bf8Fnuz>),
      rule<F32, Bf8>(RTNE, ScaledPk8Down, 8,
                     scalePk8DowncastToFp8<ROCDL::CvtScaleF32Pk8Bf8F32Op>),
      rule<F32, Bf8>(RTNE, ScaledPk4, 4,
                     scalePk4DowncastToFp8<ROCDL::CvtScaleF32PkBf8F32Op>),
      rule<F32, Bf8>(RTNE, OcpHW, 4, unscaledFp32ToFp8<Bf8>),
      rule<F32, Bf8>(RTNE, Always, 4, Fp32ToFp8E5M2rtneSW),
      rule<F32, Bf8>(RTZ, Always, 4, Fp32ToFp8E5M2rtz),
  };
  return rules;
}

//===----------------------------------------------------------------------===//
// FpToFpOp lowering
//===----------------------------------------------------------------------===//

namespace {
struct FpToFpOpConversion
    : public ElementwiseOpConversionBase<triton::FpToFpOp, FpToFpOpConversion> {
  explicit FpToFpOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              const AMD::TargetInfo &targetInfo,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        targetFeatures(targetInfo.getArch()) {}

  const FpCvtRule *selectRule(Type srcTy, Type dstTy,
                              std::optional<RoundingMode> roundingMode,
                              size_t maxElementsPerThread) const {
    for (const FpCvtRule &rule : getFpCvtRules()) {
      if (rule.srcTy != srcTy.getTypeID() || rule.dstTy != dstTy.getTypeID())
        continue;
      if (!acceptsRounding(rule.rounding, roundingMode))
        continue;
      if (rule.feature && !(targetFeatures.*rule.feature)())
        continue;
      if (rule.needsContiguousElements &&
          maxElementsPerThread < rule.numElements)
        continue;
      // A rule that ignores the requested rounding mode is only correct if
      // the conversion never rounds.
      assert((rule.rounding != CvtRounding::Any ||
              isExactConversion(srcTy, dstTy)) &&
             "rule accepting any rounding mode must be an exact conversion");
      return &rule;
    }
    return nullptr;
  }

  void setFunctionAttributes(triton::FpToFpOp op) const {
    auto dstElementType = getElementTypeOrSelf(op.getResult());

    // set clamping attribute for FP8 data types
    if (dstElementType.isFloat() &&
        (dstElementType.getIntOrFloatBitWidth() == 8)) {
      auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
      if (func) {
        using attrType = triton::amdgpu::SetFP8ClampingAttr;
        auto attrName = attrType::getMnemonic();
        if (!func->hasAttrOfType<attrType>(attrName)) {
          func->setAttr(attrName, attrType::get(op->getContext()));
        }
      }
    }
  }

  std::optional<size_t> getNumContiguousElementsPerThread(Operation *op) const {
    auto tensorTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    // Scalar operand: one element per thread.
    if (!tensorTy)
      return 1;
    auto layoutTy = mlir::dyn_cast<triton::gpu::DistributedEncodingTrait>(
        tensorTy.getEncoding());
    if (!layoutTy)
      return std::nullopt;
    auto order = triton::gpu::getThreadOrder(layoutTy, tensorTy.getShape());
    auto elemsPerThread = triton::gpu::getElemsPerThread(tensorTy);
    return elemsPerThread[order.back()];
  }

  SmallVector<Value> createDestOps(triton::FpToFpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcElementType = getElementTypeOrSelf(op.getSrc());
    auto dstElementType = getElementTypeOrSelf(op.getResult());

    auto roundingMode = op.getRounding();

    auto maybeMaxElementsPerThread = getNumContiguousElementsPerThread(op);
    if (!maybeMaxElementsPerThread) {
      op->emitError(
          "Expected an operand with a distributed encoding attribute");
      return SmallVector<Value>{};
    }
    const size_t maxElementsPerThread = maybeMaxElementsPerThread.value();

    // Fp16 <-> Bf16 is lowered as an exact upcast to Fp32 followed by the
    // Fp32 -> dst rule.
    bool useFp32Intermediate =
        (srcElementType.isBF16() && dstElementType.isF16()) ||
        (srcElementType.isF16() && dstElementType.isBF16());
    if (useFp32Intermediate)
      roundingMode = roundingMode.value_or(RoundingMode::RTNE);
    const FpCvtRule *rule =
        selectRule(useFp32Intermediate ? f32_ty : srcElementType,
                   dstElementType, roundingMode, maxElementsPerThread);
    if (rule == nullptr) {
      std::string rmError;
      if (roundingMode.has_value())
        rmError = std::string(" with rounding mode ") +
                  stringifyRoundingMode(roundingMode.value()).str();
      op->emitError("Unsupported conversion from ")
          << srcElementType << " to " << dstElementType << rmError;
      return SmallVector<Value>{};
    }

    setFunctionAttributes(op);

    const size_t numElements = rule->numElements;

    // extract a chunk of input element defined by the rule (i.e., the
    // instruction property) filled the rest of the values with undefs if the
    // input vector is too small
    SmallVector<Value> inVals;
    inVals.reserve(std::min(numElements, operands.size()));
    for (unsigned i = 0; i < std::min(numElements, operands.size()); i++) {
      inVals.push_back(operands[i][0]);
    }
    inVals.resize(numElements,
                  b.undef(typeConverter->convertType(srcElementType)));

    if (useFp32Intermediate)
      for (Value &v : inVals)
        v = srcElementType.isBF16() ? AMD::convertBf16ToFp32(loc, rewriter, v)
                                    : Fp16ToFp32OneValue(loc, rewriter, v);

    SmallVector<Value> outVals = rule->convert(loc, rewriter, inVals);
    assert(outVals.size() == inVals.size());
    outVals.resize(std::min(numElements, operands.size()));
    return outVals;
  }

private:
  TargetFeatures targetFeatures;
};
} // namespace

//===----------------------------------------------------------------------===//
// Public interface
//===----------------------------------------------------------------------===//

namespace mlir::triton::AMD {
Value convertBf16ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                        const Value &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto as_int16 = b.bitcast(v, i16_ty);
  auto as_int32 = b.zext(i32_ty, as_int16);
  auto shifted = b.shl(i32_ty, as_int32, b.i32_val(16));
  return b.bitcast(shifted, f32_ty);
}

Value convertFp32ToBf16(Location loc, ConversionPatternRewriter &rewriter,
                        const Value &v, const RoundingMode rounding) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto as_int32 = b.bitcast(v, i32_ty);
  if (rounding == RoundingMode::RTZ) {
    auto shifted = b.lshr(i32_ty, as_int32, b.i32_val(16));
    auto truncated = b.trunc(i16_ty, shifted);
    return b.bitcast(truncated, bf16_ty);
  }

  // This implementation is a faster version for fp32 to bf16 type conversion
  // It is from CK:
  // https://github.com/cgmillette/composable_kernel/commit/24e75bef6aa5
  // It uses less VGPR and less number of instructions compared to the
  // previous implementation
  Value isNan = checkIsNan(b, v);
  Value v16 = b.i32_val(16);
  Value tmp = b.and_(i32_ty, b.lshr(i32_ty, as_int32, v16), b.i32_val(1));

  Value v7FFF = b.i32_val(0x7FFF);
  Value s1 = b.add(as_int32, tmp);
  Value s2 = b.add(s1, v7FFF);

  Value vNan = b.i32_val(0x7FFF0000);
  Value res = b.select(isNan, vNan, s2);

  Value shifted = b.lshr(i32_ty, res, v16);
  Value truncated = b.trunc(i16_ty, shifted);
  return b.bitcast(truncated, bf16_ty);
}

// Fp32ToF16/Bf16 RTNE
SmallVector<Value> convertFp32ToF16rtne(Location loc,
                                        ConversionPatternRewriter &rewriter,
                                        Type inElemTy, Type outElemTy,
                                        MultipleOperandsRange operands,
                                        const TargetFeatures &targetFeatures) {
  if (targetFeatures.supportsCvtPkFp32ToFp16Bf16()) {
    SmallVector<Value> inVals;
    size_t numElem = std::min(size_t(2), operands.size());
    inVals.reserve(numElem);
    for (unsigned i = 0; i < numElem; i++) {
      inVals.push_back(operands[i][0]);
    }
    return convertFp32ToFp16rtne(loc, rewriter, inVals, outElemTy);
  }

  if (outElemTy.isBF16()) {
    assert(inElemTy.isF32() && "unsupported conversion");
    return {AMD::convertFp32ToBf16(loc, rewriter, operands[0][0],
                                   RoundingMode::RTNE)};
  }
  return {LLVM::FPTruncOp::create(rewriter, loc, outElemTy, operands[0][0])};
}

void populateFpCastOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                    const TargetInfo &targetInfo,
                                    PatternBenefit benefit) {

  patterns.add<FpToFpOpConversion>(typeConverter, axisInfoAnalysis, targetInfo,
                                   benefit);
}
} // namespace mlir::triton::AMD
