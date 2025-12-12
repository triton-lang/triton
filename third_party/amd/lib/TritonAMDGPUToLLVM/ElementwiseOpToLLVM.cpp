#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <type_traits>

using namespace mlir;

using mlir::triton::gpu::appendOrGetExternFuncOp;
using mlir::triton::gpu::ElementwiseOpConversion;
using mlir::triton::gpu::ElementwiseOpConversionBase;
using mlir::triton::gpu::getElementType;
using mlir::triton::gpu::getFunctionType;
using mlir::triton::gpu::MultipleOperandsRange;

using ConverterT = std::function<SmallVector<Value>(
    Location, ConversionPatternRewriter &, const SmallVector<Value> &)>;

namespace {
bool isCDNA4(AMD::ISAFamily family) { return family == AMD::ISAFamily::CDNA4; }
bool isCDNA4OrHigher(AMD::ISAFamily family) {
  return family == AMD::ISAFamily::CDNA4 || family == AMD::ISAFamily::GFX1250;
}

//===----------------------------------------------------------------------===//
// Data type conversion utility functions
//===----------------------------------------------------------------------===//
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

  std::optional<std::pair<Value, Value>> getPlusMinusInf() {
    if constexpr (std::is_same_v<FPType, Float32Type>) {
      return std::make_pair(b.i32_val(0x7F800000), b.i32_val(0xFF800000));
    }
    if constexpr (std::is_same_v<FPType, Float16Type>) {
      return std::make_pair(b.i16_val(0x7C00), b.i16_val(0xFC00));
    }
    if constexpr (std::is_same_v<FPType, BFloat16Type>) {
      return std::make_pair(b.i16_val(0x7F80), b.i16_val(0xFF80));
    }

    return std::nullopt;
  }

  std::optional<std::pair<Value, Value>> getPlusMinusMax() {
    if constexpr (std::is_same_v<FPType, Float8E4M3FNType>) {
      return std::make_pair(b.i8_val(0x7E), b.i8_val(0xFE));
    }
    if constexpr (std::is_same_v<FPType, Float8E4M3FNUZType>) {
      return std::make_pair(b.i8_val(0x7E), b.i8_val(0xFE));
    }
    if constexpr (std::is_same_v<FPType, Float8E5M2Type>) {
      return std::make_pair(b.i8_val(0x7B), b.i8_val(0xFB));
    }
    if constexpr (std::is_same_v<FPType, Float8E5M2FNUZType>) {
      return std::make_pair(b.i8_val(0x7B), b.i8_val(0xFB));
    }

    return std::nullopt;
  }

  Location loc;
  ConversionPatternRewriter &rewriter;
  TritonLLVMOpBuilder b;
};

// Convert Ocp Fp8/Bf8 to Fp16/Bf16/Fp32 on CDNA4
template <typename ConvertOp>
static SmallVector<Value>
cvtScalePkUpcastFromFp8(Location loc, ConversionPatternRewriter &rewriter,
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

// Convert Fp16/Bf16/Fp32 to OCP Fp8/Bf8 on CDNA4
template <typename ConvertOp>
static SmallVector<Value>
cvtScalePk4DowncastToFp8(Location loc, ConversionPatternRewriter &rewriter,
                         const SmallVector<Value> &v) {
  assert(v.size() == 4);
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Type v2I16Ty = vec_ty(i16_ty, 2);
  Value v2I16Vec = b.undef(v2I16Ty);
  Value scale = b.f32_val(1);

  Value result;
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

template <typename ConvertOp>
static SmallVector<Value>
cvtScalePk8DowncastToFp8(Location loc, ConversionPatternRewriter &rewriter,
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

  Type v4I8Ty = vec_ty(i8_ty, 4);

  auto resVec = ConvertOp::create(rewriter, loc, vFPResTy, inVec, b.f32_val(1));
  auto outVec = b.bitcast(resVec, vFPOutTy);

  // convert llvm vector to SmallVector
  SmallVector<Value> result(inSize);
  for (size_t i = 0; i < inSize; i++) {
    result[i] = b.extract_element(i8_ty, outVec, b.i32_val(i));
  }
  return result;
}

static SmallVector<Value>
Fp16_to_Fp8E5M2_RTNE_SW(Location loc, ConversionPatternRewriter &rewriter,
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

static SmallVector<Value>
Fp16_to_Fp8E5M2_RTNE_HW(Location loc, ConversionPatternRewriter &rewriter,
                        const SmallVector<Value> &v) {
  if (v.size() == 8) {
    return cvtScalePk8DowncastToFp8<ROCDL::CvtScaleF32Pk8Bf8F16Op>(loc,
                                                                   rewriter, v);
  }
  assert(v.size() == 4);
  return cvtScalePk4DowncastToFp8<ROCDL::CvtScaleF32PkBf8F16Op>(loc, rewriter,
                                                                v);
}

ConverterT Fp16_to_Fp8E5M2_RTNE(AMD::ISAFamily isaFamily) {
  return isCDNA4OrHigher(isaFamily) ? Fp16_to_Fp8E5M2_RTNE_HW
                                    : Fp16_to_Fp8E5M2_RTNE_SW;
}

// Fp16 -> OCP Bf8 (RTZ)
static SmallVector<Value>
Fp16_to_Fp8E5M2_RTZ(Location loc, ConversionPatternRewriter &rewriter,
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

static Value checkIsNan(TritonLLVMOpBuilder &builder, Value v) {
  StringRef intrinsic = "llvm.is.fpclass";
  // bits 0 and 1 indicate signaling Nan and quiet Nan, respectively
  Location loc = builder.loc;
  OpBuilder &rewriter = *builder.builder;
  Value nanBits = builder.i32_val(3);

  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, i1_ty,
                                         ValueRange{v, nanBits})
      ->getResult(0);
}

// Downcast from Fp32, FP16 or BFloat16 to FP8 formats in saturation and
// round-to-nearest-even mode. According to
// https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1,
// In saturation mode, inf and out-of-range numbers are converted to the largest
// normal number, i.e. ±448. NaNs are converted to NaNs.
// For UZ formats please check: https://onnx.ai/onnx/technical/float8.html
template <typename SrcFPType, typename DstFPType>
static Value downcastToFp8_RTNE_oneValue(Location loc,
                                         ConversionPatternRewriter &rewriter,
                                         Value v) {
  static_assert((std::is_same_v<SrcFPType, Float32Type>) ||
                (std::is_same_v<SrcFPType, Float16Type>) ||
                (std::is_same_v<SrcFPType, BFloat16Type>));
  static_assert((std::is_same_v<DstFPType, Float8E4M3FNType> ||
                 std::is_same_v<DstFPType, Float8E4M3FNUZType> ||
                 std::is_same_v<DstFPType, Float8E5M2FNUZType>));
  constexpr bool isFp8UZ = (std::is_same_v<DstFPType, Float8E4M3FNUZType> ||
                            std::is_same_v<DstFPType, Float8E5M2FNUZType>);
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

// Fp16 -> OCP Fp8 (RTNZ)
static SmallVector<Value>
Fp16_to_Fp8E4M3FN_RTNE_SW(Location loc, ConversionPatternRewriter &rewriter,
                          const SmallVector<Value> &v) {
  assert(v.size() == 4);
  SmallVector<Value> result(4);
  for (size_t i = 0; i < 4; i++)
    result[i] = downcastToFp8_RTNE_oneValue<Float16Type, Float8E4M3FNType>(
        loc, rewriter, v[i]);
  return result;
}

static SmallVector<Value>
Fp16_to_Fp8E4M3FN_RTNE_HW(Location loc, ConversionPatternRewriter &rewriter,
                          const SmallVector<Value> &v) {
  if (v.size() == 8) {
    return cvtScalePk8DowncastToFp8<ROCDL::CvtScaleF32Pk8Fp8F16Op>(loc,
                                                                   rewriter, v);
  }
  assert(v.size() == 4);
  return cvtScalePk4DowncastToFp8<ROCDL::CvtScaleF32PkFp8F16Op>(loc, rewriter,
                                                                v);
}

ConverterT Fp16_to_Fp8E4M3FN_RTNE(AMD::ISAFamily isaFamily) {
  return isCDNA4OrHigher(isaFamily) ? Fp16_to_Fp8E4M3FN_RTNE_HW
                                    : Fp16_to_Fp8E4M3FN_RTNE_SW;
}

// Fp16 -> Fp32
static Value cvtFp16ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v) {

  TritonLLVMOpBuilder b(loc, rewriter);
  return b.fpext(f32_ty, v);
}

// Convert Bf8/Fp8 to Fp32 on CDNA3
template <typename ConvertOp>
static SmallVector<Value> cvtPkF8ToFp32(Location loc,
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

// Convert Fp32 to Bf8/Fp8 on CDNA3
template <typename ConvertOp>
static SmallVector<Value> cvtPkFp32ToF8(Location loc,
                                        ConversionPatternRewriter &rewriter,
                                        const SmallVector<Value> &v) {
  assert(v.size() == 4);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type v2I16Ty = vec_ty(i16_ty, 2);
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

// Convert OCP Fp8 to Fp32 on CDNA4
static SmallVector<Value> Fp8E4M3FN_to_Fp32(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            const SmallVector<Value> &v) {
  assert(v.size() == 4);
  return cvtScalePkUpcastFromFp8<ROCDL::CvtScaleF32PkF32Fp8Op>(loc, rewriter,
                                                               v);
}

// Convert OCP Bf8 to Fp32 on CDNA4
static SmallVector<Value> Fp8E5M2_to_Fp32(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          const SmallVector<Value> &v) {
  assert(v.size() == 4);
  return cvtScalePkUpcastFromFp8<ROCDL::CvtScaleF32PkF32Bf8Op>(loc, rewriter,
                                                               v);
}

// Fp32 -> OCP Fp8 (RTNZ)
static SmallVector<Value>
Fp32_to_Fp8E4M3FN_RTNE_SW(Location loc, ConversionPatternRewriter &rewriter,
                          const SmallVector<Value> &v) {
  SmallVector<Value> result(4);
  for (size_t i = 0; i < 4; i++)
    result[i] = downcastToFp8_RTNE_oneValue<Float32Type, Float8E4M3FNType>(
        loc, rewriter, v[i]);
  return result;
}

// Convert Fp32 to OCP Fp8 on CDNA4
static SmallVector<Value>
Fp32_to_Fp8E4M3FN_RTNE_HW(Location loc, ConversionPatternRewriter &rewriter,
                          const SmallVector<Value> &v) {
  if (v.size() == 8) {
    return cvtScalePk8DowncastToFp8<ROCDL::CvtScaleF32Pk8Fp8F32Op>(loc,
                                                                   rewriter, v);
  }
  assert(v.size() == 4);
  return cvtScalePk4DowncastToFp8<ROCDL::CvtScaleF32PkFp8F32Op>(loc, rewriter,
                                                                v);
}

ConverterT Fp32_to_Fp8E4M3FN_RTNE(AMD::ISAFamily isaFamily) {
  return isCDNA4OrHigher(isaFamily) ? Fp32_to_Fp8E4M3FN_RTNE_HW
                                    : Fp32_to_Fp8E4M3FN_RTNE_SW;
}

// Fp32 -> OCP Bf8 (RTNE)

static SmallVector<Value>
Fp32_to_Fp8E5M2_RTNE_SW(Location loc, ConversionPatternRewriter &rewriter,
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

static SmallVector<Value>
Fp32_to_Fp8E5M2_RTNE_HW(Location loc, ConversionPatternRewriter &rewriter,
                        const SmallVector<Value> &v) {
  if (v.size() == 8) {
    return cvtScalePk8DowncastToFp8<ROCDL::CvtScaleF32Pk8Bf8F32Op>(loc,
                                                                   rewriter, v);
  }
  assert(v.size() == 4);
  return cvtScalePk4DowncastToFp8<ROCDL::CvtScaleF32PkBf8F32Op>(loc, rewriter,
                                                                v);
}

ConverterT Fp32_to_Fp8E5M2_RTNE(AMD::ISAFamily isaFamily) {
  return isCDNA4OrHigher(isaFamily) ? Fp32_to_Fp8E5M2_RTNE_HW
                                    : Fp32_to_Fp8E5M2_RTNE_SW;
}

// Fp32 -> Nanoo Bf8 on CDNA3
static SmallVector<Value>
Fp32_to_Fp8E5M2FNUZ_HW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  assert(v.size() == 4);
  return cvtPkFp32ToF8<ROCDL::CvtPkBf8F32Op>(loc, rewriter, v);
}

static SmallVector<Value>
Fp32_to_Fp8E5M2FNUZ_SW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  assert(v.size() == 2);
  SmallVector<Value> result(2);
  result[0] = downcastToFp8_RTNE_oneValue<Float32Type, Float8E5M2FNUZType>(
      loc, rewriter, v[0]);
  result[1] = downcastToFp8_RTNE_oneValue<Float32Type, Float8E5M2FNUZType>(
      loc, rewriter, v[1]);
  return result;
}

ConverterT Fp32_to_Fp8E5M2FNUZ(AMD::ISAFamily isaFamily) {
  return isaFamily == AMD::ISAFamily::CDNA3 ? Fp32_to_Fp8E5M2FNUZ_HW
                                            : Fp32_to_Fp8E5M2FNUZ_SW;
}

// Fp32 -> Nanoo Fp8 on CDNA3
static SmallVector<Value>
Fp32_to_Fp8E4M3FNUZ_HW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  assert(v.size() == 4);
  return cvtPkFp32ToF8<ROCDL::CvtPkFp8F32Op>(loc, rewriter, v);
}

static SmallVector<Value>
Fp32_to_Fp8E4M3FNUZ_SW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  assert(v.size() == 2);
  SmallVector<Value> result(2);
  result[0] = downcastToFp8_RTNE_oneValue<Float32Type, Float8E4M3FNUZType>(
      loc, rewriter, v[0]);
  result[1] = downcastToFp8_RTNE_oneValue<Float32Type, Float8E4M3FNUZType>(
      loc, rewriter, v[1]);
  return result;
}

static ConverterT Fp32_to_Fp8E4M3FNUZ(AMD::ISAFamily isaFamily) {
  return isCDNA4(isaFamily) ? Fp32_to_Fp8E4M3FNUZ_SW : Fp32_to_Fp8E4M3FNUZ_HW;
}

// Nanoo Bf8 -> Fp32 on CDNA3
static SmallVector<Value>
Fp8E5M2FNUZ_to_Fp32(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 4);
  return cvtPkF8ToFp32<ROCDL::CvtPkF32Bf8Op>(loc, rewriter, v);
}

// Nanoo Fp8 -> Fp32 on CDNA3
static SmallVector<Value>
Fp8E4M3FNUZ_to_Fp32(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 4);
  return cvtPkF8ToFp32<ROCDL::CvtPkF32Fp8Op>(loc, rewriter, v);
}

static SmallVector<Value>
Fp16_to_Fp8E5M2FNUZ_SW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  assert(v.size() == 2);
  SmallVector<Value> vFp32 = {cvtFp16ToFp32(loc, rewriter, v[0]),
                              cvtFp16ToFp32(loc, rewriter, v[1])};
  return Fp32_to_Fp8E5M2FNUZ_SW(loc, rewriter, vFp32);
}

static SmallVector<Value>
Fp16_to_Fp8E5M2FNUZ_HW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  SmallVector<Value> f32Vec(4);
  for (size_t i = 0; i < 4; i++)
    f32Vec[i] = cvtFp16ToFp32(loc, rewriter, v[i]);

  // Convert fp32 to bf8
  return cvtPkFp32ToF8<ROCDL::CvtPkBf8F32Op>(loc, rewriter, f32Vec);
}

ConverterT Fp16_to_Fp8E5M2FNUZ(AMD::ISAFamily isaFamily) {
  return isaFamily == AMD::ISAFamily::CDNA3 ? Fp16_to_Fp8E5M2FNUZ_HW
                                            : Fp16_to_Fp8E5M2FNUZ_SW;
}

static Value Fp8E4M3FN_to_Fp16_oneValue(Location loc,
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

  // Adjust exponent, (15 - 7) << 10 === 0x2000
  a = b.add(a, b.i16_val(0x2000));

  // Check NaN
  Value vAbs = b.and_(b.bitcast(v, i8_ty), b.i8_val(0x7F));
  a = b.select(b.icmp_eq(vAbs, b.i8_val(0x7F)), b.i16_val(0x7E00), a);

  // Check denorms and zero
  // Here we use a LUT to map S.0000.000 ~ S.0000.111 to its corresponding fp16
  // value
  constexpr size_t lutSize = 8;
  static constexpr int denormsAndZeroLut[lutSize] = {
      0x0000, 0x1800, 0x1C00, 0x1E00, 0x2000, 0x2100, 0x2200, 0x2300};

  for (int i = 0; i < lutSize; i++) {
    a = b.select(b.icmp_eq(vAbs, b.i8_val(i)), b.i16_val(denormsAndZeroLut[i]),
                 a);
  }

  // Set sign
  a = b.or_(a, sign);
  a = b.bitcast(a, f16_ty);

  return a;
}

// Ocp Fp8->Fp16
static SmallVector<Value>
Fp8E4M3FN_to_Fp16_SW(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &values) {
  SmallVector<Value> results(4);
  for (size_t i = 0; i < 4; i++)
    results[i] = Fp8E4M3FN_to_Fp16_oneValue(loc, rewriter, values[i]);
  return results;
}

static SmallVector<Value>
Fp8E4M3FN_to_Fp16_HW(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &v) {
  assert(v.size() == 4);
  return cvtScalePkUpcastFromFp8<ROCDL::CvtScaleF32PkF16Fp8Op>(loc, rewriter,
                                                               v);
}

ConverterT Fp8E4M3FN_to_Fp16(AMD::ISAFamily isaFamily) {
  return isCDNA4(isaFamily) ? Fp8E4M3FN_to_Fp16_HW : Fp8E4M3FN_to_Fp16_SW;
}

// Ocp Bf8->Fp16
static SmallVector<Value>
Fp8E5M2_to_Fp16_SW(Location loc, ConversionPatternRewriter &rewriter,
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

static SmallVector<Value>
Fp8E5M2_to_Fp16_HW(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  assert(v.size() == 4);
  return cvtScalePkUpcastFromFp8<ROCDL::CvtScaleF32PkF16Bf8Op>(loc, rewriter,
                                                               v);
}

ConverterT Fp8E5M2_to_Fp16(AMD::ISAFamily isaFamily) {
  return isCDNA4(isaFamily) ? Fp8E5M2_to_Fp16_HW : Fp8E5M2_to_Fp16_SW;
}

static SmallVector<Value>
convertFp32ToFp16RTZ(Location loc, ConversionPatternRewriter &rewriter,
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

// Fp32->Fp16/Bf16 (RTNE) in GFX950
static SmallVector<Value>
convertFp32ToFp16RTNE(Location loc, ConversionPatternRewriter &rewriter,
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

static SmallVector<Value>
Fp32_to_Fp8E5M2_RTZ(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 4);
  SmallVector<Value> inVals(2);
  inVals[0] = v[0];
  inVals[1] = v[1];
  auto f16Vec = convertFp32ToFp16RTZ(loc, rewriter, inVals);
  SmallVector<Value> vec(4);
  vec[0] = f16Vec[0];
  vec[1] = f16Vec[1];
  inVals[0] = v[2];
  inVals[1] = v[3];
  f16Vec = convertFp32ToFp16RTZ(loc, rewriter, inVals);
  vec[2] = f16Vec[0];
  vec[3] = f16Vec[1];
  return Fp16_to_Fp8E5M2_RTZ(loc, rewriter, vec);
}

static Value convertBf16ToFp32(Location loc,
                               ConversionPatternRewriter &rewriter,
                               const Value &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto as_int16 = b.bitcast(v, i16_ty);
  auto as_int32 = b.zext(i32_ty, as_int16);
  auto shifted = b.shl(i32_ty, as_int32, b.i32_val(16));
  return b.bitcast(shifted, f32_ty);
}

static Value convertFp32ToBf16(Location loc,
                               ConversionPatternRewriter &rewriter,
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

// Fp32_to_F16/Bf16 RTNE
static SmallVector<Value> Fp32_to_F16_RTNE(Location loc,
                                           ConversionPatternRewriter &rewriter,
                                           Type inElemTy, Type outElemTy,
                                           MultipleOperandsRange operands,
                                           AMD::ISAFamily isaFamily) {
  // For CDNA4 we can potentially use packed v_cvt_pk_[b]f16_f32 instructions.
  if (isCDNA4(isaFamily)) {
    SmallVector<Value> inVals;
    size_t numElem = std::min(size_t(2), operands.size());
    inVals.reserve(numElem);
    for (unsigned i = 0; i < numElem; i++) {
      inVals.push_back(operands[i][0]);
    }
    return convertFp32ToFp16RTNE(loc, rewriter, inVals, outElemTy);
  }

  if (outElemTy.isBF16()) {
    assert(inElemTy.isF32() && "unsupported conversion");
    return {
        convertFp32ToBf16(loc, rewriter, operands[0][0], RoundingMode::RTNE)};
  }
  return {LLVM::FPTruncOp::create(rewriter, loc, outElemTy, operands[0][0])};
}

static Value Fp8E5M2FNUZ_to_Fp16_oneValue(Location loc,
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

  return b.bitcast(o, f16_ty);
}

static SmallVector<Value>
Fp8E5M2FNUZ_to_Fp16_SW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  assert(v.size() == 4);
  SmallVector<Value> result(4);
  for (size_t i = 0; i < 4; i++)
    result[i] = Fp8E5M2FNUZ_to_Fp16_oneValue(loc, rewriter, v[i]);
  return result;
}

static SmallVector<Value>
Fp8E5M2FNUZ_to_Fp16_HW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  assert(v.size() == 4);
  // Convert Bf8 to fp32
  SmallVector<Value> ret =
      cvtPkF8ToFp32<ROCDL::CvtPkF32Bf8Op>(loc, rewriter, v);

  // Convert fp32 to fp16
  for (size_t i = 0; i < 4; i++)
    ret[i] = LLVM::AMD::cvtFp32ToFp16RTNE_oneValue(loc, rewriter, ret[i]);

  return ret;
}

ConverterT Fp8E5M2FNUZ_to_Fp16(AMD::ISAFamily isaFamily) {
  return isaFamily == AMD::ISAFamily::CDNA3 ? Fp8E5M2FNUZ_to_Fp16_HW
                                            : Fp8E5M2FNUZ_to_Fp16_SW;
}

// OCP Bf8/Fp8 -> Bf16
template <typename SrcFPType>
static SmallVector<Value> OcpF8_to_Bf16_SW(Location loc,
                                           ConversionPatternRewriter &rewriter,
                                           const SmallVector<Value> &v) {
  static_assert(std::is_same_v<SrcFPType, Float8E4M3FNType> ||
                std::is_same_v<SrcFPType, Float8E5M2Type>);
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
Fp8E5M2_to_Bf16_SW(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  return OcpF8_to_Bf16_SW<Float8E5M2Type>(loc, rewriter, v);
}

static SmallVector<Value>
Fp8E5M2_to_Bf16_HW(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  assert(v.size() == 4);
  return cvtScalePkUpcastFromFp8<ROCDL::CvtScaleF32PkBf16Bf8Op>(loc, rewriter,
                                                                v);
}

ConverterT Fp8E5M2_to_Bf16(AMD::ISAFamily isaFamily) {
  return isCDNA4(isaFamily) ? Fp8E5M2_to_Bf16_HW : Fp8E5M2_to_Bf16_SW;
}

// Bf16 -> OCP Bf8
static SmallVector<Value>
Bf16_to_Fp8E5M2_SW(Location loc, ConversionPatternRewriter &rewriter,
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

static SmallVector<Value>
Bf16_to_Fp8E5M2_HW(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  if (v.size() == 8) {
    return cvtScalePk8DowncastToFp8<ROCDL::CvtScaleF32Pk8Bf8Bf16Op>(
        loc, rewriter, v);
  }
  assert(v.size() == 4);
  return cvtScalePk4DowncastToFp8<ROCDL::CvtScaleF32PkBf8Bf16Op>(loc, rewriter,
                                                                 v);
}

static ConverterT Bf16_to_Fp8E5M2(AMD::ISAFamily isaFamily) {
  return isCDNA4OrHigher(isaFamily) ? Bf16_to_Fp8E5M2_HW : Bf16_to_Fp8E5M2_SW;
}

// Bf16 -> OCP Fp8 using RTNE
static SmallVector<Value>
Bf16_to_Fp8E4M3FN_RTNE_SW(Location loc, ConversionPatternRewriter &rewriter,
                          const SmallVector<Value> &v) {
  assert(v.size() == 4);
  SmallVector<Value> result(4);
  for (size_t i = 0; i < 4; ++i)
    result[i] = downcastToFp8_RTNE_oneValue<BFloat16Type, Float8E4M3FNType>(
        loc, rewriter, v[i]);
  return result;
}

static SmallVector<Value>
Bf16_to_Fp8E4M3FN_RTNE_HW(Location loc, ConversionPatternRewriter &rewriter,
                          const SmallVector<Value> &v) {
  if (v.size() == 8) {
    return cvtScalePk8DowncastToFp8<ROCDL::CvtScaleF32Pk8Fp8Bf16Op>(
        loc, rewriter, v);
  }
  assert(v.size() == 4);
  return cvtScalePk4DowncastToFp8<ROCDL::CvtScaleF32PkFp8Bf16Op>(loc, rewriter,
                                                                 v);
}

ConverterT Bf16_to_Fp8E4M3FN(AMD::ISAFamily isaFamily) {
  return isCDNA4OrHigher(isaFamily) ? Bf16_to_Fp8E4M3FN_RTNE_HW
                                    : Bf16_to_Fp8E4M3FN_RTNE_SW;
}

// fp8e4m3fn to bf16
static SmallVector<Value>
Fp8E4M3FN_to_Bf16_SW(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &v) {
  return OcpF8_to_Bf16_SW<Float8E4M3FNType>(loc, rewriter, v);
}

static SmallVector<Value>
Fp8E4M3FN_to_Bf16_HW(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &v) {
  assert(v.size() == 4);
  return cvtScalePkUpcastFromFp8<ROCDL::CvtScaleF32PkBf16Fp8Op>(loc, rewriter,
                                                                v);
}

ConverterT Fp8E4M3FN_to_Bf16(AMD::ISAFamily isaFamily) {
  return isCDNA4(isaFamily) ? Fp8E4M3FN_to_Bf16_HW : Fp8E4M3FN_to_Bf16_SW;
}

// fp8e4m3fnuz to bf16
static SmallVector<Value>
Fp8E4M3FNUZ_to_Bf16_HW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  assert(v.size() == 4);
  auto ret = cvtPkF8ToFp32<ROCDL::CvtPkF32Fp8Op>(loc, rewriter, v);
  for (size_t i = 0; i < 4; i++)
    ret[i] = convertFp32ToBf16(loc, rewriter, ret[i], RoundingMode::RTZ);
  return ret;
}

static SmallVector<Value>
Fp8E4M3FNUZ_to_Bf16_SW(Location loc, ConversionPatternRewriter &rewriter,
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

  // Add the signs and place the halfwords in the proper place in order to pack
  // them
  constexpr uint32_t signHalfwordMask = signMask | (signMask << 16U);
  Value out0 = b.or_(i32_ty, b.lshr(i32_ty, d0, b.i32_val(16)), d1);
  Value sign0 = b.and_(i32_ty, a0, b.i32_val(signHalfwordMask));
  out0 = b.or_(i32_ty, out0, sign0);

  // Unpack the 2 bfloat16 values and return them
  auto bf16x2VecTy = vec_ty(bf16_ty, 2);
  out0 = b.bitcast(out0, bf16x2VecTy);
  return {b.extract_element(bf16_ty, out0, b.i32_val(0)),
          b.extract_element(bf16_ty, out0, b.i32_val(1))};
}

static ConverterT Fp8E4M3FNUZ_to_Bf16(AMD::ISAFamily isaFamily) {
  return isCDNA4(isaFamily) ? Fp8E4M3FNUZ_to_Bf16_SW : Fp8E4M3FNUZ_to_Bf16_HW;
}

// bf16 to fp8e4m3fnuz
static SmallVector<Value>
Bf16_to_Fp8E4M3FNUZ_HW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  assert(v.size() == 4);
  SmallVector<Value> fp32Vec(4);
  for (size_t i = 0; i < 4; i++)
    fp32Vec[i] = convertBf16ToFp32(loc, rewriter, v[i]);
  return cvtPkFp32ToF8<ROCDL::CvtPkFp8F32Op>(loc, rewriter, fp32Vec);
}

static SmallVector<Value>
Bf16_to_Fp8E4M3FNUZ_SW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  assert(v.size() == 4);
  SmallVector<Value> result(4);
  for (size_t i = 0; i < 4; i++)
    result[i] = downcastToFp8_RTNE_oneValue<BFloat16Type, Float8E4M3FNUZType>(
        loc, rewriter, v[i]);
  return result;
}

static ConverterT Bf16_to_Fp8E4M3FNUZ(AMD::ISAFamily isaFamily) {
  return isCDNA4(isaFamily) ? Bf16_to_Fp8E4M3FNUZ_SW : Bf16_to_Fp8E4M3FNUZ_HW;
}

// fp8e5m2fnuz to bf16
static SmallVector<Value>
Fp8E5M2FNUZ_to_Bf16(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 4);
  auto ret = cvtPkF8ToFp32<ROCDL::CvtPkF32Bf8Op>(loc, rewriter, v);
  for (size_t i = 0; i < 4; i++)
    ret[i] = convertFp32ToBf16(loc, rewriter, ret[i], RoundingMode::RTZ);
  return ret;
}

// bf16 to fp8e5m2fnuz
static SmallVector<Value>
Bf16_to_Fp8E5M2FNUZ_HW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  assert(v.size() == 4);
  SmallVector<Value> f32Vec(4);
  for (size_t i = 0; i < 4; i++)
    f32Vec[i] = convertBf16ToFp32(loc, rewriter, v[i]);
  return cvtPkFp32ToF8<ROCDL::CvtPkBf8F32Op>(loc, rewriter, f32Vec);
}

static SmallVector<Value>
Bf16_to_Fp8E5M2FNUZ_SW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  assert(v.size() == 4);
  SmallVector<Value> result(4);
  for (size_t i = 0; i < 4; i++)
    result[i] = downcastToFp8_RTNE_oneValue<BFloat16Type, Float8E5M2FNUZType>(
        loc, rewriter, v[i]);
  return result;
}

static ConverterT Bf16_to_Fp8E5M2FNUZ(AMD::ISAFamily isaFamily) {
  return isCDNA4(isaFamily) ? Bf16_to_Fp8E5M2FNUZ_SW : Bf16_to_Fp8E5M2FNUZ_HW;
}

static Value Fp8E4M3FNUZ_to_Fp16_oneValue(Location loc,
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
  // Check NaN (1.0000.000 in E4M3FNUZ)
  // Pick an arbitrary number which represents NaN in fp16 (exp=11111 and mant
  // != 0)
  a = b.select(b.icmp_eq(v8, b.i8_val(0x80)), b.i16_val(0x7E00), a);

  // Check denorms and zero
  // Here we use a LUT to map S.0000.000 ~ S.0000.111 to its corresponding fp16
  // value
  // Minimum subnormal value in E4M3FNUZ is 2^-10
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
  a = b.bitcast(a, f16_ty);

  return a;
}

static SmallVector<Value>
Fp8E4M3FNUZ_to_Fp16_SW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  assert(v.size() == 4);
  SmallVector<Value> result(4);
  for (size_t i = 0; i < 4; i++)
    result[i] = Fp8E4M3FNUZ_to_Fp16_oneValue(loc, rewriter, v[i]);
  return result;
}

static SmallVector<Value>
Fp8E4M3FNUZ_to_Fp16_HW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  assert(v.size() == 4);
  // Convert fp8 to fp32
  SmallVector<Value> ret =
      cvtPkF8ToFp32<ROCDL::CvtPkF32Fp8Op>(loc, rewriter, v);

  // Convert fp32 to fp16
  for (size_t i = 0; i < 4; i++)
    ret[i] = LLVM::AMD::cvtFp32ToFp16RTNE_oneValue(loc, rewriter, ret[i]);

  return ret;
}

static ConverterT Fp8E4M3FNUZ_to_Fp16(AMD::ISAFamily isaFamily) {
  return isaFamily == AMD::ISAFamily::CDNA3 ? Fp8E4M3FNUZ_to_Fp16_HW
                                            : Fp8E4M3FNUZ_to_Fp16_SW;
}

static SmallVector<Value>
Fp16_to_Fp8E4M3FNUZ_SW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  assert(v.size() == 2);
  SmallVector<Value> result(2);
  result[0] = downcastToFp8_RTNE_oneValue<Float16Type, Float8E4M3FNUZType>(
      loc, rewriter, v[0]);
  result[1] = downcastToFp8_RTNE_oneValue<Float16Type, Float8E4M3FNUZType>(
      loc, rewriter, v[1]);
  return result;
}

static SmallVector<Value>
Fp16_to_Fp8E4M3FNUZ_HW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  assert(v.size() == 4);
  SmallVector<Value> f32Vec(4);
  for (size_t i = 0; i < 4; i++)
    f32Vec[i] = cvtFp16ToFp32(loc, rewriter, v[i]);

  // Convert fp32 to fp8
  return cvtPkFp32ToF8<ROCDL::CvtPkFp8F32Op>(loc, rewriter, f32Vec);
}

static ConverterT Fp16_to_Fp8E4M3FNUZ(AMD::ISAFamily isaFamily) {
  return isaFamily == AMD::ISAFamily::CDNA3 ? Fp16_to_Fp8E4M3FNUZ_HW
                                            : Fp16_to_Fp8E4M3FNUZ_SW;
}

//===----------------------------------------------------------------------===//
// Data type conversion patterns
//===----------------------------------------------------------------------===//

// Attempts to use vectorized conversions via inline PTX when possible.
struct FpToFpOpConversion
    : public ElementwiseOpConversionBase<triton::FpToFpOp, FpToFpOpConversion> {
  explicit FpToFpOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              AMD::ISAFamily isaFamily,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        isaFamily(isaFamily) {}

  static Value convertFp16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    return cvtFp16ToFp32(loc, rewriter, v);
  }

  FailureOr<ConverterT>
  getConversionFunc(Type srcTy, Type dstTy,
                    std::optional<RoundingMode> roundingMode) const {
    auto F8E4M3B15TyID = TypeID::get<Float8E4M3B11FNUZType>();
    auto F8E4M3FNUZTyID = TypeID::get<Float8E4M3FNUZType>();
    auto F8E5M2FNUZTyID = TypeID::get<Float8E5M2FNUZType>();
    auto F8E5M2TyID = TypeID::get<Float8E5M2Type>();
    auto F8E4M3FNTyID = TypeID::get<Float8E4M3FNType>();
    auto F16TyID = TypeID::get<Float16Type>();
    auto BF16TyID = TypeID::get<BFloat16Type>();
    auto F32TyID = TypeID::get<Float32Type>();
    auto F64TyID = TypeID::get<Float64Type>();

    auto undefRounding = static_cast<RoundingMode>(-1);

    static DenseMap<std::tuple<TypeID, TypeID, RoundingMode>, ConverterT>
        srcMap = {
            // F8 -> F16
            {{F8E4M3FNUZTyID, F16TyID, undefRounding},
             Fp8E4M3FNUZ_to_Fp16(isaFamily)},
            {{F8E4M3FNTyID, F16TyID, undefRounding},
             Fp8E4M3FN_to_Fp16(isaFamily)},
            {{F8E5M2FNUZTyID, F16TyID, undefRounding},
             Fp8E5M2FNUZ_to_Fp16(isaFamily)},
            {{F8E5M2TyID, F16TyID, undefRounding}, Fp8E5M2_to_Fp16(isaFamily)},
            // F16 -> F8
            {{F16TyID, F8E4M3FNTyID, RoundingMode::RTNE},
             Fp16_to_Fp8E4M3FN_RTNE(isaFamily)},
            {{F16TyID, F8E5M2FNUZTyID, RoundingMode::RTNE},
             Fp16_to_Fp8E5M2FNUZ(isaFamily)},
            {{F16TyID, F8E4M3FNUZTyID, RoundingMode::RTNE},
             Fp16_to_Fp8E4M3FNUZ(isaFamily)},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTNE},
             Fp16_to_Fp8E5M2_RTNE(isaFamily)},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTZ}, Fp16_to_Fp8E5M2_RTZ},
            // F8 -> BF16
            {{F8E5M2TyID, BF16TyID, undefRounding}, Fp8E5M2_to_Bf16(isaFamily)},
            {{F8E5M2FNUZTyID, BF16TyID, undefRounding}, Fp8E5M2FNUZ_to_Bf16},
            {{F8E4M3FNTyID, BF16TyID, undefRounding},
             Fp8E4M3FN_to_Bf16(isaFamily)},
            {{F8E4M3FNUZTyID, BF16TyID, undefRounding},
             Fp8E4M3FNUZ_to_Bf16(isaFamily)},
            // BF16 -> F8
            {{BF16TyID, F8E5M2TyID, RoundingMode::RTNE},
             Bf16_to_Fp8E5M2(isaFamily)},
            {{BF16TyID, F8E4M3FNTyID, RoundingMode::RTNE},
             Bf16_to_Fp8E4M3FN(isaFamily)},
            {{BF16TyID, F8E5M2FNUZTyID, RoundingMode::RTNE},
             Bf16_to_Fp8E5M2FNUZ(isaFamily)},
            {{BF16TyID, F8E4M3FNUZTyID, RoundingMode::RTNE},
             Bf16_to_Fp8E4M3FNUZ(isaFamily)},
            // F32 <-> F8
            {{F32TyID, F8E4M3FNUZTyID, RoundingMode::RTNE},
             Fp32_to_Fp8E4M3FNUZ(isaFamily)},
            {{F32TyID, F8E5M2FNUZTyID, RoundingMode::RTNE},
             Fp32_to_Fp8E5M2FNUZ(isaFamily)},
            {{F32TyID, F8E4M3FNTyID, RoundingMode::RTNE},
             Fp32_to_Fp8E4M3FN_RTNE(isaFamily)},
            {{F32TyID, F8E5M2TyID, RoundingMode::RTNE},
             Fp32_to_Fp8E5M2_RTNE(isaFamily)},
            {{F32TyID, F8E5M2TyID, RoundingMode::RTZ}, Fp32_to_Fp8E5M2_RTZ},
            {{F8E4M3FNUZTyID, F32TyID, undefRounding}, Fp8E4M3FNUZ_to_Fp32},
            {{F8E5M2FNUZTyID, F32TyID, undefRounding}, Fp8E5M2FNUZ_to_Fp32},
            {{F8E4M3FNTyID, F32TyID, undefRounding}, Fp8E4M3FN_to_Fp32},
            {{F8E5M2TyID, F32TyID, undefRounding}, Fp8E5M2_to_Fp32},
            // F32 -> F16 with RTZ
            {{F32TyID, F16TyID, RoundingMode::RTZ}, convertFp32ToFp16RTZ},
        };
    std::tuple<TypeID, TypeID, RoundingMode> key = {
        srcTy.getTypeID(), dstTy.getTypeID(),
        roundingMode.value_or(undefRounding)};
    if (srcMap.count(key) == 0) {
      return failure();
    }
    return srcMap.lookup(key);
  }

  SmallVector<Value> createDestOps(triton::FpToFpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcElementType = getElementType(op.getSrc());
    auto dstElementType = getElementType(op.getResult());

    auto roundingMode = op.getRounding();
    if (srcElementType.isF32() &&
        (dstElementType.isF16() || dstElementType.isBF16())) {
      assert(roundingMode.has_value() &&
             "rounding mode must be specified for fp32->fp16/bf16 conversion");
      if (roundingMode.value() == RoundingMode::RTNE) {
        return Fp32_to_F16_RTNE(loc, rewriter, srcElementType, dstElementType,
                                operands, isaFamily);
      }
    }
    if (srcElementType.isF32() && dstElementType.isBF16()) {
      return {
          convertFp32ToBf16(loc, rewriter, operands[0][0], RoundingMode::RTZ)};
    }

    size_t numElements = 4;
    // numElements = 2 for :
    // fp32 -> fp16 with RTZ
    // fp32/fp16 -> nanoo fp8/bf8 on non-CDNA3
    // nanoo fp8 -> bf16 on CDNA4
    if ((llvm::isa<Float32Type>(srcElementType) &&
         llvm::isa<Float16Type>(dstElementType) &&
         roundingMode == RoundingMode::RTZ) ||
        (llvm::isa<Float32Type, Float16Type>(srcElementType) &&
         llvm::isa<Float8E4M3FNUZType, Float8E5M2FNUZType>(dstElementType) &&
         isaFamily != AMD::ISAFamily::CDNA3) ||
        (llvm::isa<Float8E4M3FNUZType>(srcElementType) &&
         dstElementType.isBF16() && isCDNA4(isaFamily)))
      numElements = 2;

    // fp32 -> fp8 with rtne can be done in two steps:
    // - fp32 -> fp16 with rtne and
    // - fp16 -> fp8 with rtne
    // with the following exceptions:
    // 1. fp32 -> ocp fp8/bf8 on CDNA4: has hardware support
    // 2. fp32 -> nanoo fp8/bf8 on CDNA3: has hardware support
    // 3. fp32 -> ocp fp8/bf8 on non-CDNA4: has software support
    bool useFP16IntermediateSrc =
        srcElementType.isF32() && !dstElementType.isF16() &&
        !(isCDNA4(isaFamily) &&
          (llvm::isa<Float8E4M3FNType, Float8E4M3FNUZType, Float8E5M2Type,
                     Float8E5M2FNUZType>(dstElementType))) &&
        !(isaFamily == AMD::ISAFamily::CDNA3 &&
          (llvm::isa<Float8E4M3FNUZType, Float8E5M2FNUZType>(
              dstElementType))) &&
        !(!isCDNA4(isaFamily) &&
          (llvm::isa<Float8E5M2Type, Float8E4M3FNType>(dstElementType)));

    if ((isaFamily == AMD::ISAFamily::GFX1250) &&
        ((llvm::isa<Float32Type>(srcElementType)) ||
         (llvm::isa<Float16Type>(srcElementType)) ||
         (llvm::isa<BFloat16Type>(srcElementType))) &&
        ((llvm::isa<Float8E4M3FNType>(dstElementType)) ||
         (llvm::isa<Float8E5M2Type>(dstElementType))) &&
        ((roundingMode.has_value()) && (*roundingMode != RoundingMode::RTZ))) {
      numElements = 8;
      useFP16IntermediateSrc = false;
    }

    // fp8/bf8->f32, if neither nanoo fp8/bf8 on CDNA3 nor ocp fp8/bf8 on CDNA4,
    // is done in two steps: fp8/bf8->fp16 and fp16->fp32
    bool isDstFP32 = dstElementType.isF32();
    bool useFP16IntermediateDst =
        (isDstFP32 &&
         !(isCDNA4(isaFamily) &&
           (llvm::isa<Float8E4M3FNType, Float8E5M2Type>(srcElementType))) &&
         !(isaFamily == AMD::ISAFamily::CDNA3 &&
           (llvm::isa<Float8E4M3FNUZType, Float8E5M2FNUZType>(
               srcElementType))));

    Type srcType = useFP16IntermediateSrc ? f16_ty : srcElementType;
    Type dstType = useFP16IntermediateDst ? f16_ty : dstElementType;
    SmallVector<Value> inVals;
    inVals.reserve(std::min(numElements, operands.size()));
    for (unsigned i = 0; i < std::min(numElements, operands.size()); i++) {
      inVals.push_back(operands[i][0]);
    }
    bool isSrcFP16 = srcElementType.isF16();
    bool isSrcBF16 = srcElementType.isBF16();

    if ((isSrcFP16 || isSrcBF16) && isDstFP32) {
      SmallVector<Value> outVals;
      for (Value &v : inVals) {
        if (isSrcFP16)
          outVals.push_back(convertFp16ToFp32(loc, rewriter, v));
        else
          outVals.push_back(convertBf16ToFp32(loc, rewriter, v));
      }
      return outVals;
    }
    if (useFP16IntermediateSrc) {
      if (isCDNA4(isaFamily))
        inVals = convertFp32ToFp16RTNE(loc, rewriter, inVals, f16_ty);
      else {
        for (Value &v : inVals)
          v = LLVM::AMD::cvtFp32ToFp16RTNE_oneValue(loc, rewriter, v);
      }
    }

    if (dstType.isFloat() && (dstType.getIntOrFloatBitWidth() == 8)) {
      auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
      if (func) {
        using attrType = triton::amdgpu::SetFP8ClampingAttr;
        auto attrName = attrType::getMnemonic();
        if (!func->hasAttrOfType<attrType>(attrName)) {
          func->setAttr(attrName, attrType::get(op->getContext()));
        }
      }
    }

    inVals.resize(numElements, b.undef(typeConverter->convertType(srcType)));
    SmallVector<Value> outVals;
    if (srcType != dstType) {
      auto getCvtFunc = getConversionFunc(srcType, dstType, roundingMode);
      if (failed(getCvtFunc)) {
        std::string rmError;
        if (roundingMode.has_value())
          rmError = std::string(" with rounding mode ") +
                    stringifyRoundingMode(roundingMode.value()).str();
        op->emitError("Unsupported conversion from ")
            << srcType << " to " << dstType << rmError;
        return outVals;
      } else {
        auto cvtFunc = getCvtFunc.value();
        outVals = cvtFunc(loc, rewriter, inVals);
      }
    } else {
      outVals = inVals;
    }

    assert(outVals.size() == inVals.size());
    outVals.resize(std::min(numElements, operands.size()));
    if (useFP16IntermediateDst)
      for (Value &v : outVals)
        v = convertFp16ToFp32(loc, rewriter, v);
    // Pack values
    return outVals;
  }

private:
  AMD::ISAFamily isaFamily;
};

template <typename OP>
Value EmitDualBF16ElementwiseOp(Location loc,
                                ConversionPatternRewriter &rewriter,
                                MultipleOperandsRange operands) {
  auto v0 = convertBf16ToFp32(loc, rewriter, operands[0][0]);
  auto v1 = convertBf16ToFp32(loc, rewriter, operands[0][1]);
  auto result = OP::create(rewriter, loc, f32_ty, v0, v1);
  return convertFp32ToBf16(loc, rewriter, result, RoundingMode::RTNE);
}

struct FDivOpConversion
    : ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::DivFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {

    return {LLVM::FDivOp::create(rewriter, loc, elemTy, operands[0][0],
                                 operands[0][1])};
  }
};

struct FMulOpConversion
    : ElementwiseOpConversionBase<arith::MulFOp, FMulOpConversion> {

  explicit FMulOpConversion(LLVMTypeConverter &typeConverter,
                            ModuleAxisInfoAnalysis &axisAnalysisPass,
                            AMD::ISAFamily isaFamily,
                            PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        isaFamily(isaFamily) {}

  SmallVector<Value> createDestOps(arith::MulFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      if (isRDNA(isaFamily)) {
        // To avoid casting to/from fp32, we compute a dot product with one
        // element of each vector set to zero.
        auto b = TritonLLVMOpBuilder(loc, rewriter);
        Value aVal = packLLVector(
            loc, ValueRange{operands[0][0], b.bf16_val(0.0)}, rewriter);
        Value bVal = packLLVector(
            loc, ValueRange{operands[0][1], b.bf16_val(0.0)}, rewriter);
        return {LLVM::createLLVMIntrinsicCallOp(
                    rewriter, loc, "llvm.amdgcn.fdot2.bf16.bf16", bf16_ty,
                    ValueRange{aVal, bVal, b.bf16_val(0.0)})
                    ->getResult(0)};
      } else {
        return {
            EmitDualBF16ElementwiseOp<LLVM::FMulOp>(loc, rewriter, operands)};
      }
    } else {
      return {LLVM::FMulOp::create(rewriter, loc, elemTy, operands[0][0],
                                   operands[0][1])};
    }
  }

private:
  AMD::ISAFamily isaFamily;
};

struct FAddOpConversion
    : ElementwiseOpConversionBase<arith::AddFOp, FAddOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::AddFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {EmitDualBF16ElementwiseOp<LLVM::FAddOp>(loc, rewriter, operands)};
    } else {
      return {LLVM::FAddOp::create(rewriter, loc, elemTy, operands[0][0],
                                   operands[0][1])};
    }
  }
};

struct FSubOpConversion
    : ElementwiseOpConversionBase<arith::SubFOp, FSubOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::SubFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {EmitDualBF16ElementwiseOp<LLVM::FSubOp>(loc, rewriter, operands)};
    } else {
      return {LLVM::FSubOp::create(rewriter, loc, elemTy, operands[0][0],
                                   operands[0][1])};
    }
  }
};

static SmallVector<Value> S8_to_Bf16(Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> inValues = {v[0], v[1], v[2], v[3]};
  SmallVector<Value> outValues = {};
  for (Value inVal : inValues) {
    Value bf16Val = LLVM::SIToFPOp::create(rewriter, loc, bf16_ty, inVal);
    outValues.push_back(bf16Val);
  }
  return outValues;
}

struct SIToFPOpConversion
    : ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::SIToFPOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type inElemTy = getElementType(op.getIn());
    Type outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16() && inElemTy.isInteger(8) && operands.size() >= 4) {
      SmallVector<Value> inVals = {operands[0][0], operands[1][0],
                                   operands[2][0], operands[3][0]};
      auto outVals = S8_to_Bf16(loc, rewriter, inVals);
      assert(outVals.size() == 4);
      return outVals;
    } else if (outElemTy.isBF16()) {
      auto value =
          LLVM::SIToFPOp::create(rewriter, loc, f32_ty, operands[0][0]);
      return {convertFp32ToBf16(loc, rewriter, value, RoundingMode::RTNE)};
    } else {
      return {LLVM::SIToFPOp::create(rewriter, loc, elemTy, operands[0][0])};
    }
  }
};

struct FPToSIOpConversion
    : ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::FPToSIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto value = convertBf16ToFp32(loc, rewriter, operands[0][0]);
      return {LLVM::FPToSIOp::create(rewriter, loc, elemTy, value)};
    } else {
      return {LLVM::FPToSIOp::create(rewriter, loc, elemTy, operands[0][0])};
    }
  }
};

struct ExtFOpConversion
    : ElementwiseOpConversionBase<arith::ExtFOp, ExtFOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::ExtFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto outElemTy = getElementType(op.getOut());
      assert(outElemTy.isF32() && "unsupported conversion");
      return {convertBf16ToFp32(loc, rewriter, operands[0][0])};
    } else {
      return {LLVM::FPExtOp::create(rewriter, loc, elemTy, operands[0][0])};
    }
  }
};

struct TruncFOpConversion
    : ElementwiseOpConversionBase<arith::TruncFOp, TruncFOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  explicit TruncFOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              AMD::ISAFamily isaFamily,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        isaFamily(isaFamily) {}

  SmallVector<Value> createDestOps(arith::TruncFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto outElemTy = getElementType(op.getOut());
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isF32() && (outElemTy.isBF16() || outElemTy.isF16())) {
      return Fp32_to_F16_RTNE(loc, rewriter, inElemTy, outElemTy, operands,
                              isaFamily);
    }
    return {LLVM::FPTruncOp::create(rewriter, loc, elemTy, operands[0][0])};
  }

private:
  AMD::ISAFamily isaFamily;
};

struct ExpOpConversionApprox
    : ElementwiseOpConversionBase<math::ExpOp, ExpOpConversionApprox> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(math::ExpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // For non-FP32 input, call __ocml_exp_f64 for higher-precision calculation
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    const double log2e = 1.4426950408889634;
    Value prod = b.fmul(f32_ty, operands[0][0], b.f32_val(log2e));

    // Here we use llvm.exp2.f32 instead of math::Exp2Op. The latter
    // flushes denorms by default, but we want to preserve denorms by default
    // for expOp.
    StringRef funcName = "llvm.exp2.f32";
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);

    return {LLVM::createLLVMCallOp(rewriter, loc, funcOp, prod).getResult()};
  }
};

struct Exp2OpConversion
    : ElementwiseOpConversionBase<math::Exp2Op, Exp2OpConversion> {
  explicit Exp2OpConversion(LLVMTypeConverter &typeConverter,
                            ModuleAxisInfoAnalysis &axisInfoAnalysis, bool ftz,
                            PatternBenefit benefit)
      : ElementwiseOpConversionBase(typeConverter, axisInfoAnalysis, benefit),
        ftz(ftz) {}

  SmallVector<Value> createDestOps(math::Exp2Op op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // For non-FP32 input, call __ocml_exp2_f64 for higher-precision calculation
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    // On AMD backend, both intrinsics are lowered to v_exp_f32 instruction,
    // which flushes input and output denorms. `llvm.amdgcn.exp2.f32` provides
    // direct access to v_exp_f32. For `llvm.exp2.f32`, the LLVM backend inserts
    // instructions to handle denorms iff `allow_flush_denorm` is False.
    StringRef funcName = ftz ? "llvm.amdgcn.exp2.f32" : "llvm.exp2.f32";
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);

    return {
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]).getResult()};
  }

private:
  bool ftz;
};

struct RsqrtOpConversion
    : ElementwiseOpConversionBase<math::RsqrtOp, RsqrtOpConversion> {
  explicit RsqrtOpConversion(LLVMTypeConverter &typeConverter,
                             ModuleAxisInfoAnalysis &axisInfoAnalysis, bool ftz,
                             PatternBenefit benefit)
      : ElementwiseOpConversionBase(typeConverter, axisInfoAnalysis, benefit),
        ftz(ftz) {}

  SmallVector<Value> createDestOps(math::RsqrtOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // This pass only deals with FP32 input with ftz configuration. Other cases
    // are delegate to MLIR.
    //
    // For FP16/FP64 input, it's lowered to __ocml_rsqrt_f16/__ocml_rsqrt_f64.
    //
    // For FP32 input with non-ftz configuration, it's lowered to
    // __ocml_rsqrt_f32, which will check the ftz/daz settings in the backend
    // dynamically to decide to preserve/flush denorms.
    if (elemTy.getIntOrFloatBitWidth() != 32 || !ftz)
      return {};

    // `llvm.amdgcn.rsq.f32` provides direct access to v_rsq_f32_e32.
    StringRef funcName = "llvm.amdgcn.rsq.f32";

    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);

    return {
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]).getResult()};
  }

private:
  bool ftz;
};

static inline std::pair<Value, Value>
scaleUpIfDenorm(ConversionPatternRewriter &rewriter, Location loc,
                const Value &src, float scaleThreshold, float scaleFactor) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value needScale = b.fcmp_ogt(b.f32_val(scaleThreshold), src);
  Value scaledSrc = b.fmul(f32_ty, src, b.f32_val(scaleFactor));
  Value selectedSrc = b.select(needScale, scaledSrc, src);
  return {needScale, selectedSrc};
}

static inline Value scaleDownIfDenorm(ConversionPatternRewriter &rewriter,
                                      Location loc, const Value &src,
                                      Value needScale, float scaleFactor) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value scaledSrc = b.fmul(f32_ty, src, b.f32_val(scaleFactor));
  return b.select(needScale, scaledSrc, src);
}

struct SqrtOpConversion
    : ElementwiseOpConversionBase<math::SqrtOp, SqrtOpConversion> {
  explicit SqrtOpConversion(LLVMTypeConverter &typeConverter,
                            ModuleAxisInfoAnalysis &axisInfoAnalysis, bool ftz,
                            PatternBenefit benefit)
      : ElementwiseOpConversionBase(typeConverter, axisInfoAnalysis, benefit),
        ftz(ftz) {}

  SmallVector<Value> createDestOps(math::SqrtOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // This function only handles FP32 inputs. Other data types are lowered to
    // LLVM::SqrtOp by MLIR.
    //
    // On the AMDGPU backend, instructions legalized from LLVM::SqrtOp are
    // designed to produce IEEE-compliant results and always preserve denorms.
    // But what we actually need is an approximated SQRT. So we need to manually
    // lower the op.
    //
    // Differences in this approach are
    // 1. Refinement iterations following llvm.amdgcn.sqrt.f32 are removed to
    // improve performance.
    // 2. With ftz enabled, the scaling-up-and-down process is bypassed to
    // ensure denorms are flushed to zero.
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    Value needScale = b.false_val();
    Value scaledSrc = operands[0][0];
    if (!ftz) {
      // For non-ftz cases, if the input value is below 2^{-96}, it needs to be
      // scaled up by a factor of 2^{32}, to prevent it from being flushed by
      // llvm.amdgcn.sqrt.f32.
      //
      // The result is then scaled down afterward to get the correct result.
      // Reference:
      // https://github.com/llvm/llvm-project/blob/0876c11c/llvm/lib/Target/AMDGPU/AMDGPULegalizerInfo.cpp#L5235-L5314.
      std::tie(needScale, scaledSrc) = scaleUpIfDenorm(
          rewriter, loc, operands[0][0], 0x1.0p-96f, 0x1.0p+32f);
    }

    // llvm.amdgcn.sqrt.f32 provides direct access to v_sqrt_f32, which provides
    // 1ULP accuracy and flushs denorms.
    StringRef funcName = "llvm.amdgcn.sqrt.f32";

    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);

    Value intrinsicsOutput =
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]).getResult();

    if (!ftz) {
      // In case of non-ftz, we need to calibrate the results by scaling down by
      // a factor of 2^{-16}.
      return {scaleDownIfDenorm(rewriter, loc, intrinsicsOutput, needScale,
                                0x1.0p-16f)};
    } else {
      return {intrinsicsOutput};
    }
  }

private:
  bool ftz;
};
} // namespace

namespace mlir::triton::AMD {
void adjustModeRegister(ModuleOp mod, const TargetInfo &targetInfo) {
  MLIRContext *ctx = mod->getContext();
  Location loc = mod->getLoc();
  mlir::OpBuilder builder(ctx);
  auto auxBuilder = TritonLLVMOpBuilder(loc, builder);

  mod->walk([&](LLVM::LLVMFuncOp func) {
    using attrType = triton::amdgpu::SetFP8ClampingAttr;
    auto attrName = attrType::getMnemonic();
    if (!func->hasAttrOfType<attrType>(attrName))
      return;
    else
      func->removeAttr(attrName);

    if (func.getBody().empty())
      return;
    auto &body = func.getBody().front();
    builder.setInsertionPoint(&body.front());

    // This is the location of the fp16_ovfl flag in the Mode register. It's
    // calculated following this formula:
    //     (mode register ID = 1) | (Offset << 6) | ((Width - 1) << 11)
    // In this case, Offset = 23 and Width = 1.
    // When the bit is 0/1, the conversion from fp32/fp16/bf16 to fp8/bf8 is
    // in non-saturation/saturation mode.
    Value fp16OVFLModeRegLoc = auxBuilder.i32_val(1473);
    LLVM::createLLVMIntrinsicCallOp(
        builder, loc, "llvm.amdgcn.s.setreg", {},
        {fp16OVFLModeRegLoc, auxBuilder.i32_val(1)});
  });
}

void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, bool ftz,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    const TargetInfo &targetInfo, PatternBenefit benefit) {

  // fmin (return NaN if either op is NaN)
  patterns.add<ElementwiseOpConversion<arith::MinimumFOp, LLVM::MinimumOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  // fmax (return NaN if either op is NaN)
  patterns.add<ElementwiseOpConversion<arith::MaximumFOp, LLVM::MaximumOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ElementwiseOpConversion<triton::PreciseDivFOp, LLVM::FDivOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ElementwiseOpConversion<triton::PreciseSqrtOp, LLVM::SqrtOp>>(
      typeConverter, axisInfoAnalysis, benefit);

  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FSubOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FAddOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FMulOpConversion>(typeConverter, axisInfoAnalysis,
                                 targetInfo.getISAFamily(), benefit);

  patterns.add<ExtFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<TruncFOpConversion>(typeConverter, axisInfoAnalysis,
                                   targetInfo.getISAFamily(), benefit);
  patterns.add<FPToSIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<SIToFPOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FpToFpOpConversion>(typeConverter, axisInfoAnalysis,
                                   targetInfo.getISAFamily(), benefit);

  // ExpOpConversionApprox will try using __ocml_exp2_f32 if the input type is
  // FP32. For other input types, ExpOpConversionApprox will return failure and
  // later pass will call __ocml_exp_f64 for higher-precision calculation
  patterns.add<ExpOpConversionApprox>(typeConverter, axisInfoAnalysis, benefit);
  // Exp2OpConversion will use llvm.exp2.f32 or llvm.amdgcn.exp2.f32
  // based on the ftz flag if the input type is FP32. For FP64 input,
  // Exp2OpConversion will return failure and later pass will call
  // __ocml_exp2_f64 for higher-precision calculation
  patterns.add<Exp2OpConversion>(typeConverter, axisInfoAnalysis, ftz, benefit);
  patterns.add<RsqrtOpConversion>(typeConverter, axisInfoAnalysis, ftz,
                                  benefit);
  patterns.add<SqrtOpConversion>(typeConverter, axisInfoAnalysis, ftz, benefit);
  triton::populateElementwiseOpToLLVMPatterns(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
  bool hwNanPropagationSupported = targetInfo.supportMaximumMinimum();
  triton::populateMinMaxFOpToLLVMPattern(typeConverter, patterns,
                                         axisInfoAnalysis,
                                         hwNanPropagationSupported, benefit);
  triton::populateClampFOpToLLVMPattern(typeConverter, patterns,
                                        axisInfoAnalysis, targetInfo, benefit);
}
} // namespace mlir::triton::AMD
