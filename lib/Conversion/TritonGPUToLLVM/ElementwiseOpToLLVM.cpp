#include "ElementwiseOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;
using ::mlir::triton::gpu::getTotalElemsPerThread;

/* ----- FP8E5M2 ------ */
// This data-type is the standard FP8E5M2 format

const std::string Fp16_to_Fp8E5M2 =
    "{                            \n"
    ".reg .b32 a<2>;              \n"
    "and.b32 a0, $1, 0x7fff7fff;  \n"           // a0 &= 0x7fff7fff
    "and.b32 a1, $2, 0x7fff7fff;  \n"           // (strip sign)
    "add.u32 a0, a0, 0x00800080;  \n"           // a0 += 0x00800080
    "add.u32 a1, a1, 0x00800080;  \n"           // (round to nearest)
    "lop3.b32 a0, $1, 0x80008000, a0, 0xea; \n" // a0 = a0|(0x80008000&in0)
    "lop3.b32 a1, $2, 0x80008000, a1, 0xea; \n" // (restore sign)
    "prmt.b32 $0, a0, a1, 0x7531; \n\t"         // output = a1a0
    "}";

const std::string Fp8E5M2_to_Fp16 = "{                           \n"
                                    "prmt.b32 $0, 0, $2, 0x5140; \n\t"
                                    "prmt.b32 $1, 0, $2, 0x7362; \n\t"
                                    "}";

const std::string Fp8E5M2_to_Bf16 =
    "{                                      \n"
    ".reg .b32 a<2>, b<2>;                  \n" // if input = 0xf1f2f3f4
    "prmt.b32 a0, 0, $2, 0x5140;            \n" // a0 = 0xf300f400
    "prmt.b32 a1, 0, $2, 0x7362;            \n" // a1 = 0xf100f200
    "lop3.b32 b0, a0, 0x7fff7fff, 0, 0xc0;  \n" // b0 = a0 & 0x7fff7fff
    "lop3.b32 b1, a1, 0x7fff7fff, 0, 0xc0;  \n" // (strip sign)
    "shr.b32  b0, b0, 3;                    \n" // b0 >>= 3
    "shr.b32  b1, b1, 3;                    \n" // shift into bf16 position
    "add.u32  b0, b0, 0x38003800;           \n" // b0.exp += 2**7-2**4
                                                // exponent compensate = 112
    "add.u32  b1, b1, 0x38003800;           \n" // b1 += 112<<7 | 112<<7<<16
    "lop3.b32 $0, b0, 0x80008000, a0, 0xf8; \n" // out0 = b0|(0x80008000&a0)
    "lop3.b32 $1, b1, 0x80008000, a1, 0xf8; \n" // (restore sign)
    "}";

const std::string Bf16_to_Fp8E5M2 =
    "{                                           \n" // bf16=fp8>>3 + 112<<7
    ".reg .u32 sign, sign<2>, nosign, nosign<2>; \n" // fp8_min = 0b00000000
    ".reg .u32 fp8_min, fp8_max, rn_;            \n" // fp8_max = 0b11111111
    "mov.u32 fp8_min, 0x38003800;                \n" // so bf16_min = 0x3800
    "mov.u32 fp8_max, 0x57e057e0;                \n" // so bf16_max = 0x57e0
    "mov.u32 rn_, 0x00100010;                    \n" // round to nearest
    "and.b32 sign0, $1, 0x80008000;              \n" // sign0=in0&0x80008000
    "and.b32 sign1, $2, 0x80008000;              \n" // (store sign)
    "prmt.b32 sign, sign0, sign1, 0x7531;        \n"
    "and.b32 nosign0, $1, 0x7fff7fff;            \n" // nosign0=in0&0x7fff7fff
    "and.b32 nosign1, $2, 0x7fff7fff;            \n" // (strip sign)

    // nosign = clamp(nosign, min, max)
    ".reg .u32 nosign_0_<2>, nosign_1_<2>;       \n"
    "and.b32 nosign_0_0, nosign0, 0xffff0000;    \n"
    "max.u32 nosign_0_0, nosign_0_0, 0x38000000; \n"
    "min.u32 nosign_0_0, nosign_0_0, 0x57e00000; \n"
    "and.b32 nosign_0_1, nosign0, 0x0000ffff;    \n"
    "max.u32 nosign_0_1, nosign_0_1, 0x3800;     \n"
    "min.u32 nosign_0_1, nosign_0_1, 0x57e0;     \n"
    "or.b32 nosign0, nosign_0_0, nosign_0_1;     \n"
    "and.b32 nosign_1_0, nosign1, 0xffff0000;    \n"
    "max.u32 nosign_1_0, nosign_1_0, 0x38000000; \n"
    "min.u32 nosign_1_0, nosign_1_0, 0x57e00000; \n"
    "and.b32 nosign_1_1, nosign1, 0x0000ffff;    \n"
    "max.u32 nosign_1_1, nosign_1_1, 0x3800;     \n"
    "min.u32 nosign_1_1, nosign_1_1, 0x57e0;     \n"
    "or.b32 nosign1, nosign_1_0, nosign_1_1;     \n"

    "add.u32 nosign0, nosign0, rn_;              \n" // nosign0 += rn_
    "add.u32 nosign1, nosign1, rn_;              \n" // (round to nearest)
    "sub.u32 nosign0, nosign0, 0x38003800;       \n" // nosign0-=0x38003800
    "sub.u32 nosign1, nosign1, 0x38003800;       \n" // (compensate offset)
    "shl.b32 nosign0, nosign0, 3;                \n" // nosign0 <<= 3
    "shl.b32 nosign1, nosign1, 3;                \n" // shift into to fp8e4
    "prmt.b32 nosign, nosign0, nosign1, 0x7531;  \n" // nosign0 = 0xf100f200
                                                     // nosign1 = 0xf300f400
                                                     // nosign = 0xf3f4f1f2
    "or.b32 $0, nosign, sign;                    \n" // restore sign
    "}";

/* ----- FP8E4M3B15 ------ */
// This data-type is a variant of the standard FP8E4M3 format.
// It was designed for fast software conversion to FP16 on
// nvidia GPUs that do not support it natively.
// Specifically, this data-type:
//    - has infinities
//    - has multiple nans (when all exponent bits are 1)
//    - has an exponent bias of 15 (vs. 7 for fp8e4m3)

// Fp8E4M3B15 -> Fp16 (packed)
// fast conversion code provided by Scott Gray @ OpenAI
// $0 = (($2 << 1) & 0x80008000u) | (($2 << 7) & 0x3f803f80u);
// $1 = (($2 << 0) & 0x80008000u) | (($2 << 0) & 0x3f803f80u);
// WARN: subnormal (0bs0000xxx) are not handled
const std::string Fp8E4M3B15_to_Fp16 =
    "{                                      \n"
    ".reg .b32 a<2>;                        \n"
    "shl.b32 a0, $2, 1;                     \n"
    "shl.b32 a1, $2, 7;                     \n"
    "and.b32  $0, a0, 0x80008000;           \n"
    "lop3.b32 $0, $0, a1, 0x3f803f80, 0xf8; \n"
    "and.b32  $1, $2, 0x80008000;           \n"
    "lop3.b32 $1, $1, $2, 0x3f803f80, 0xf8; \n"
    "}";

// Fp16 -> Fp8E4M3B15 (packed)
// fast conversion code provided by Scott Gray @ OpenAI
// ret = ((e4.x >> 1) & (0x80008000u >> 1)) |
//       ((e4.x >> 7) & (0x3f803f80u >> 7)) |
//       ((e4.y >> 0) & (0x80008000u >> 0)) |
//       ((e4.y >> 0) & (0x3f803f80u >> 0)) ;
// WARN: subnormal (0bs0000xxx) are not handled
const std::string Fp16_to_Fp8E4M3B15 =
    "{                                       \n"
    ".reg .b32 a<2>;                         \n"
    "shr.b32  a0, $1, 1;                     \n"
    "shr.b32  a1, $1, 7;                     \n"
    "and.b32  $0,     a0, 0x40004000;        \n"
    "lop3.b32 $0, $0, a1, 0x007f007f, 0xf8;  \n"
    "lop3.b32 $0, $0, $2, 0x80008000, 0xf8;  \n"
    "lop3.b32 $0, $0, $2, 0x3f803f80, 0xf8;  \n"
    "}";

/* ----- FP8E4M3 ------ */
// Note: when handled by software, this format
// does not handle denormals and has
// more than a single NaN values.

// Fp8E4M3 -> Fp16 (packed)
const std::string Fp8E4M3_to_Fp16 =
    "{                                      \n"
    ".reg .b32 a<2>, b<2>;                  \n" // if input = 0xf1f2f3f4
    "prmt.b32 a0, 0, $2, 0x5040;            \n" // a0 = 0xf300f400
    "prmt.b32 a1, 0, $2, 0x7060;            \n" // a1 = 0xf100f200
    "lop3.b32 b0, a0, 0x7fff7fff, 0, 0xc0;  \n" // b0 = a0 & 0x7fff7fff
    "lop3.b32 b1, a1, 0x7fff7fff, 0, 0xc0;  \n" // (strip sign)
    "shr.b32  b0, b0, 1;                    \n" // b0 >>= 1
    "shr.b32  b1, b1, 1;                    \n" // shift into fp16 position
    "add.u32  b0, b0, 0x20002000;           \n" // b0.exp += 2**4-2**3
                                                // exponent compensate = 8
    "add.u32  b1, b1, 0x20002000;           \n" // b1 += 8<<10 | 8<<10<<16
    "lop3.b32 $0, b0, 0x80008000, a0, 0xf8; \n" // out0 = b0|(0x80008000&a0)
    "lop3.b32 $1, b1, 0x80008000, a1, 0xf8; \n" // (restore sign)
    "}";

// Fp16 -> Fp8E4M3 (packed)
const std::string Fp16_to_Fp8E4M3 =
    "{                                      \n"
    ".reg .b32 a<2>, b<2>;                  \n" // see Fp8E4M3x4ToFp16x4
    "sub.u32 a0, $1, 0x20002000;            \n" // a0 = input0 - 0x20002000
                                                // (compensate offset)
    "sub.u32 a1, $2, 0x20002000;            \n" // a1 = input1 - 0x20002000
                                                // (8 << 10 | 8 << 10 << 16)
    "shl.b32 a0, a0, 1;                     \n" // a0 <<= 1
    "shl.b32 a1, a1, 1;                     \n" // shift into fp8e4 position
    "lop3.b32 a0, a0, 0x7fff7fff, 0, 0xc0;  \n" // a0 &= 0x7fff7fff
    "lop3.b32 a1, a1, 0x7fff7fff, 0, 0xc0;  \n" // (strip sign)
    "add.u32 a0, a0, 0x00800080;            \n" // a0 += 0x00800080
    "add.u32 a1, a1, 0x00800080;            \n" // (round to nearest)
    "lop3.b32 b0, $1, 0x80008000, a0, 0xea; \n" // b0 = a0|(0x80008000&in0)
    "lop3.b32 b1, $2, 0x80008000, a1, 0xea; \n" // (restore sign)
    "prmt.b32 $0, b0, b1, 0x7531;           \n" // output = b1b0
    "}";

// WARN: subnormal (0bs0000xxx) are not handled
const std::string Fp8E4M3_to_Bf16 =
    "{                                      \n"
    ".reg .b32 a<2>, b<2>;                  \n" // if input = 0xf1f2f3f4
    "prmt.b32 a0, 0, $2, 0x5040;            \n" // a0 = 0xf300f400
    "prmt.b32 a1, 0, $2, 0x7060;            \n" // a1 = 0xf100f200
    "and.b32 b0, a0, 0x7fff7fff;            \n" // b0 = a0 & 0x7fff7fff
    "and.b32 b1, a1, 0x7fff7fff;            \n" // (strip sign)
    "shr.b32 b0, b0, 4;                     \n" // b0 >>= 4
    "shr.b32 b1, b1, 4;                     \n" // shift into fp16 position
    "add.u32 b0, b0, 0x3c003c00;            \n" // b0.exp += 2**7-2**3
                                                // exponent compensate = 120
    "add.u32 b1, b1, 0x3c003c00;            \n" // b1 += 120<<7 | 120<<7<<16
    "lop3.b32 $0, b0, 0x80008000, a0, 0xf8; \n" // out0 = b0|(0x80008000&a0)
    "lop3.b32 $1, b1, 0x80008000, a1, 0xf8; \n" // (restore sign)
    "}";

const std::string Bf16_to_Fp8E4M3 =
    "{                                           \n" // bf16=fp8>>4 + 120<<7
    ".reg .u32 sign, sign<2>, nosign, nosign<2>; \n" // fp8_min = 0b00000000
    ".reg .u32 fp8_min, fp8_max, rn_;            \n" // fp8_max = 0b11111111
    "mov.u32 fp8_min, 0x3c003c00;                \n" // so bf16_min = 0x3c00
    "mov.u32 fp8_max, 0x43f043f0;                \n" // so bf16_max = 0x43f0
    "mov.u32 rn_, 0x80008;                       \n" // round to nearest
    "and.b32 sign0, $1, 0x80008000;              \n" // sign0=in0&0x80008000
    "and.b32 sign1, $2, 0x80008000;              \n" // (store sign)
    "prmt.b32 sign, sign0, sign1, 0x7531;        \n"
    "and.b32 nosign0, $1, 0x7fff7fff;            \n" // nosign0=in0&0x7fff7fff
    "and.b32 nosign1, $2, 0x7fff7fff;            \n" // (strip sign)

    // nosign = clamp(nosign, min, max)
    ".reg .u32 nosign_0_<2>, nosign_1_<2>;       \n"
    "and.b32 nosign_0_0, nosign0, 0xffff0000;    \n"
    "max.u32 nosign_0_0, nosign_0_0, 0x3c000000; \n"
    "min.u32 nosign_0_0, nosign_0_0, 0x43f00000; \n"
    "and.b32 nosign_0_1, nosign0, 0x0000ffff;    \n"
    "max.u32 nosign_0_1, nosign_0_1, 0x3c00;     \n"
    "min.u32 nosign_0_1, nosign_0_1, 0x43f0;     \n"
    "or.b32 nosign0, nosign_0_0, nosign_0_1;     \n"
    "and.b32 nosign_1_0, nosign1, 0xffff0000;    \n"
    "max.u32 nosign_1_0, nosign_1_0, 0x3c000000; \n"
    "min.u32 nosign_1_0, nosign_1_0, 0x43f00000; \n"
    "and.b32 nosign_1_1, nosign1, 0x0000ffff;    \n"
    "max.u32 nosign_1_1, nosign_1_1, 0x3c00;     \n"
    "min.u32 nosign_1_1, nosign_1_1, 0x43f0;     \n"
    "or.b32 nosign1, nosign_1_0, nosign_1_1;     \n"

    "add.u32 nosign0, nosign0, rn_;              \n" // nosign0 += rn_
    "add.u32 nosign1, nosign1, rn_;              \n" // (round to nearest)
    "sub.u32 nosign0, nosign0, 0x3c003c00;       \n" // nosign0-=0x3c003c00
    "sub.u32 nosign1, nosign1, 0x3c003c00;       \n" // (compensate offset)
    "shr.u32 nosign0, nosign0, 4;                \n" // nosign0 >>= 4
    "shr.u32 nosign1, nosign1, 4;                \n" // shift into to fp8e4
    "prmt.b32 nosign, nosign0, nosign1, 0x6420;  \n" // nosign0 = 0x00f100f2
                                                     // nosign1 = 0x00f300f4
                                                     // nosign = 0xf3f4f1f2
    "or.b32 $0, nosign, sign;                    \n" // restore sign
    "}";

static SmallVector<Value> reorderValues(const SmallVector<Value> &values,
                                        Type inType, Type ouType) {
  auto inTensorTy = inType.dyn_cast<RankedTensorType>();
  auto ouTensorTy = ouType.dyn_cast<RankedTensorType>();
  if (!inTensorTy || !ouTensorTy)
    return values;
  auto inEncoding =
      dyn_cast<triton::gpu::DotOperandEncodingAttr>(inTensorTy.getEncoding());
  auto ouEncoding =
      dyn_cast<triton::gpu::DotOperandEncodingAttr>(ouTensorTy.getEncoding());
  assert(inEncoding == ouEncoding);
  if (!inEncoding)
    return values;
  size_t inBitWidth = inTensorTy.getElementType().getIntOrFloatBitWidth();
  size_t ouBitWidth = ouTensorTy.getElementType().getIntOrFloatBitWidth();
  auto ouEltTy = ouTensorTy.getElementType();
  if (inBitWidth == ouBitWidth)
    return values;
  if (inBitWidth == 16 && ouBitWidth == 32) {
    SmallVector<Value> ret;
    for (unsigned i = 0; i < values.size(); i += 8) {
      ret.push_back(values[i]);
      ret.push_back(values[i + 1]);
      ret.push_back(values[i + 4]);
      ret.push_back(values[i + 5]);
      ret.push_back(values[i + 2]);
      ret.push_back(values[i + 3]);
      ret.push_back(values[i + 6]);
      ret.push_back(values[i + 7]);
    }
    return ret;
  }
  if (inBitWidth == 8 && ouBitWidth == 16) {
    SmallVector<Value> ret;
    for (unsigned i = 0; i < values.size(); i += 16) {
      ret.push_back(values[i + 0]);
      ret.push_back(values[i + 1]);
      ret.push_back(values[i + 2]);
      ret.push_back(values[i + 3]);
      ret.push_back(values[i + 8]);
      ret.push_back(values[i + 9]);
      ret.push_back(values[i + 10]);
      ret.push_back(values[i + 11]);
      ret.push_back(values[i + 4]);
      ret.push_back(values[i + 5]);
      ret.push_back(values[i + 6]);
      ret.push_back(values[i + 7]);
      ret.push_back(values[i + 12]);
      ret.push_back(values[i + 13]);
      ret.push_back(values[i + 14]);
      ret.push_back(values[i + 15]);
    }
    return ret;
    // for (unsigned i = 0; i < values.size(); i += 16) {
    //   ret.push_back(values[i]);
    //   ret.push_back(values[i + 1]);
    //   ret.push_back(values[i + 4]);
    //   ret.push_back(values[i + 5]);
    //   ret.push_back(values[i + 8]);
    //   ret.push_back(values[i + 9]);
    //   ret.push_back(values[i + 12]);
    //   ret.push_back(values[i + 13]);

    //   ret.push_back(values[i + 2]);
    //   ret.push_back(values[i + 3]);
    //   ret.push_back(values[i + 6]);
    //   ret.push_back(values[i + 7]);
    //   ret.push_back(values[i + 10]);
    //   ret.push_back(values[i + 11]);
    //   ret.push_back(values[i + 14]);
    //   ret.push_back(values[i + 15]);
    // }
    return values;
  }
  llvm_unreachable("unimplemented code path");
}

inline SmallVector<Value> unpackI32(const SmallVector<Value> &inValues,
                                    Type srcTy,
                                    ConversionPatternRewriter &rewriter,
                                    Location loc,
                                    TypeConverter *typeConverter) {
  auto tensorTy = srcTy.dyn_cast<RankedTensorType>();
  if (!tensorTy)
    return inValues;
  auto encoding = tensorTy.getEncoding().dyn_cast<DotOperandEncodingAttr>();
  if (!(encoding && encoding.getParent().isa<MmaEncodingAttr>()))
    return inValues;
  SmallVector<Value> outValues;
  for (auto v : inValues) {
    // cast i32 to appropriate eltType vector and extract elements
    auto eltType = typeConverter->convertType(tensorTy.getElementType());
    auto vecType = vec_ty(eltType, 32 / eltType.getIntOrFloatBitWidth());
    auto vec = bitcast(v, vecType);
    for (int i = 0; i < 32 / eltType.getIntOrFloatBitWidth(); i++) {
      outValues.push_back(extract_element(vec, i32_val(i)));
    }
  }
  return outValues;
}

inline SmallVector<Value> packI32(const SmallVector<Value> &inValues,
                                  Type srcTy,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc, TypeConverter *typeConverter) {
  auto tensorTy = srcTy.dyn_cast<RankedTensorType>();
  if (!tensorTy)
    return inValues;
  auto encoding = tensorTy.getEncoding().dyn_cast<DotOperandEncodingAttr>();
  if (!(encoding && encoding.getParent().isa<MmaEncodingAttr>()))
    return inValues;
  SmallVector<Value> outValues;
  auto eltType = typeConverter->convertType(tensorTy.getElementType());
  int vecWidth = 32 / eltType.getIntOrFloatBitWidth();
  auto vecType = vec_ty(eltType, vecWidth);
  for (int i = 0; i < inValues.size(); i += vecWidth) {
    Value vec = undef(vecType);
    for (int j = 0; j < vecWidth; j++) {
      vec = insert_element(vec, inValues[i + j], i32_val(j));
    }
    outValues.push_back(bitcast(vec, i32_ty));
  }
  return outValues;
}

struct FpToFpOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::FpToFpOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::FpToFpOp>::ConvertTritonGPUOpToLLVMPattern;

  typedef std::function<SmallVector<Value>(
      Location, ConversionPatternRewriter &, const Value &, const Value &,
      const Value &, const Value &)>
      ConvertorT;
  /* ------------------ */
  // FP8 -> FP16
  /* ------------------ */

  static ConvertorT makeConverterFromPtx(const std::string &ptxAsm, Type inType,
                                         Type outType) {

    ConvertorT converter =
        [ptxAsm, inType,
         outType](Location loc, ConversionPatternRewriter &rewriter,
                  const Value &v0, const Value &v1, const Value &v2,
                  const Value &v3) -> SmallVector<Value> {
      SmallVector<Value> v = {v0, v1, v2, v3};
      auto ctx = rewriter.getContext();
      int inBitwidth = inType.getIntOrFloatBitWidth();
      int outBitwidth = outType.getIntOrFloatBitWidth();
      // first, we pack `v` into 32-bit ints
      int inVecWidth = 32 / inBitwidth;
      auto inVecTy = vec_ty(inType, inVecWidth);
      SmallVector<Value> inPacked(4 / inVecWidth, undef(inVecTy));
      for (size_t i = 0; i < 4; i++)
        inPacked[i / inVecWidth] = insert_element(
            inVecTy, inPacked[i / inVecWidth], v[i], i32_val(i % inVecWidth));
      for (size_t i = 0; i < inPacked.size(); i++)
        inPacked[i] = bitcast(inPacked[i], i32_ty);

      // then, we run the provided inline PTX
      int outVecWidth = 32 / outBitwidth;
      int outNums = 4 / outVecWidth;
      PTXBuilder builder;
      SmallVector<PTXBuilder::Operand *> operands;
      for (int i = 0; i < outNums; i++)
        operands.push_back(builder.newOperand("=r"));
      for (Value inVal : inPacked)
        operands.push_back(builder.newOperand(inVal, "r"));
      auto &ptxOp = *builder.create(ptxAsm);
      ptxOp(operands, /*onlyAttachMLIRArgs=*/true);
      auto outVecTy = vec_ty(outType, outVecWidth);
      SmallVector<Value> outPacked;
      if (outNums == 1)
        outPacked.push_back(builder.launch(rewriter, loc, outVecTy, false));
      else {
        auto outStructTy = struct_ty(SmallVector<Type>(outNums, outVecTy));
        auto outStruct = builder.launch(rewriter, loc, outStructTy, false);
        for (int i = 0; i < outNums; i++)
          outPacked.push_back(extract_val(outVecTy, outStruct, i));
      }
      // unpack the output
      SmallVector<Value> ret;
      for (size_t i = 0; i < 4; i++)
        ret.push_back(extract_element(outType, outPacked[i / outVecWidth],
                                      i32_val(i % outVecWidth)));
      return ret;
    };
    return converter;
  }

  static Value convertBf16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    PTXBuilder builder;
    auto &cvt = *builder.create("cvt.f32.bf16");
    auto res = builder.newOperand("=r");
    auto operand = builder.newOperand(v, "h");
    cvt(res, operand);
    return builder.launch(rewriter, loc, f32_ty, false);
  }

  static Value convertFp16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    PTXBuilder builder;
    auto &cvt = *builder.create("cvt.f32.f16");
    auto res = builder.newOperand("=r");
    auto operand = builder.newOperand(v, "h");
    cvt(res, operand);
    return builder.launch(rewriter, loc, f32_ty, false);
  }

  static Value convertFp32ToBf16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    PTXBuilder builder;
    auto &cvt = *builder.create("cvt.rn.bf16.f32");
    auto res = builder.newOperand("=h");
    auto operand = builder.newOperand(v, "r");
    cvt(res, operand);
    // TODO: This is a hack to get the right type. We should be able to invoke
    // the type converter
    return builder.launch(rewriter, loc, i16_ty, false);
  }

  static Value convertFp32ToFp16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    PTXBuilder builder;
    auto &cvt = *builder.create("cvt.rn.f16.f32");
    auto res = builder.newOperand("=h");
    auto operand = builder.newOperand(v, "r");
    cvt(res, operand);
    return builder.launch(rewriter, loc, f16_ty, false);
  }

  ConvertorT getConversionFunc(Type srcTy, Type dstTy) const {
    auto F8E4M3B15TyID = TypeID::get<mlir::Float8E4M3B11FNUZType>();
    auto F8E4M3TyID = TypeID::get<mlir::Float8E4M3FNUZType>();
    auto F8E5M2TyID = TypeID::get<mlir::Float8E5M2Type>();
    auto F16TyID = TypeID::get<mlir::Float16Type>();
    auto BF16TyID = TypeID::get<mlir::BFloat16Type>();
    auto F32TyID = TypeID::get<mlir::Float32Type>();
    auto F64TyID = TypeID::get<mlir::Float64Type>();
    static DenseMap<std::pair<TypeID, TypeID>, std::string> srcMap = {
        // F8 -> F16
        {{F8E4M3B15TyID, F16TyID}, Fp8E4M3B15_to_Fp16},
        {{F8E4M3TyID, F16TyID}, Fp8E4M3_to_Fp16},
        {{F8E5M2TyID, F16TyID}, Fp8E5M2_to_Fp16},
        // F16 -> F8
        {{F16TyID, F8E4M3B15TyID}, Fp16_to_Fp8E4M3B15},
        {{F16TyID, F8E4M3TyID}, Fp16_to_Fp8E4M3},
        {{F16TyID, F8E5M2TyID}, Fp16_to_Fp8E5M2},
        // F8 -> BF16
        {{F8E4M3TyID, BF16TyID}, Fp8E4M3_to_Bf16},
        {{F8E5M2TyID, BF16TyID}, Fp8E5M2_to_Bf16},
        // BF16 -> F8
        {{BF16TyID, F8E4M3TyID}, Bf16_to_Fp8E4M3},
        {{BF16TyID, F8E5M2TyID}, Bf16_to_Fp8E5M2},
    };

    std::pair<TypeID, TypeID> key = {srcTy.getTypeID(), dstTy.getTypeID()};
    if (srcMap.count(key) == 0) {
      llvm::errs() << "Unsupported conversion from " << srcTy << " to " << dstTy
                   << "\n";
      llvm_unreachable("");
    }
    return makeConverterFromPtx(srcMap.lookup(key),
                                getTypeConverter()->convertType(srcTy),
                                getTypeConverter()->convertType(dstTy));
  }

  LogicalResult
  matchAndRewrite(triton::FpToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTensorType = op.getFrom().getType().cast<mlir::RankedTensorType>();
    auto dstTensorType =
        op.getResult().getType().cast<mlir::RankedTensorType>();
    auto srcElementType = srcTensorType.getElementType();
    auto dstElementType = dstTensorType.getElementType();
    auto loc = op->getLoc();
    // check that the number of elements is divisible by 4
    // Unpack value
    auto inVals = getTypeConverter()->unpackLLElements(loc, adaptor.getFrom(),
                                                       rewriter, srcTensorType);
    inVals =
        unpackI32(inVals, srcTensorType, rewriter, loc, getTypeConverter());
    // Cast
    SmallVector<Value> outVals;
    auto elems = inVals.size();
    assert(elems % 4 == 0 &&
           "FP8 casting only support tensors with 4-aligned sizes");
    bool isFP32src = srcElementType.isF32();
    bool isFP32dst = dstElementType.isF32();
    auto cvtFunc = getConversionFunc(isFP32src ? f16_ty : srcElementType,
                                     isFP32dst ? f16_ty : dstElementType);
    if (isFP32src)
      for (Value &v : inVals)
        v = convertFp32ToFp16(loc, rewriter, v);
    for (size_t i = 0; i < elems; i += 4)
      outVals.append(cvtFunc(loc, rewriter, inVals[i], inVals[i + 1],
                             inVals[i + 2], inVals[i + 3]));
    if (isFP32dst)
      for (Value &v : outVals)
        v = convertFp16ToFp32(loc, rewriter, v);
    // Pack values
    assert(outVals.size() == elems);
    outVals = reorderValues(outVals, srcTensorType, dstTensorType);
    outVals =
        packI32(outVals, dstTensorType, rewriter, loc, getTypeConverter());
    auto result = getTypeConverter()->packLLElements(loc, outVals, rewriter,
                                                     dstTensorType);
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename SourceOp, typename ConcreteT>
class ElementwiseOpConversionBase
    : public ConvertTritonGPUOpToLLVMPattern<SourceOp> {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ElementwiseOpConversionBase(
      TritonGPUToLLVMTypeConverter &typeConverter, PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern<SourceOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = op.getType();
    Location loc = op->getLoc();
    // element type
    auto resultElementTy = getElementTypeOrSelf(resultTy);
    Type elemTy = this->getTypeConverter()->convertType(resultElementTy);
    SmallVector<Value> resultVals;
    //
    SmallVector<SmallVector<Value>> allOperands;
    for (auto operand : adaptor.getOperands()) {
      auto argTy = op->getOperand(0).getType();
      auto sub_operands = this->getTypeConverter()->unpackLLElements(
          loc, operand, rewriter, argTy);
      sub_operands = unpackI32(sub_operands, argTy, rewriter, loc,
                               this->getTypeConverter());
      allOperands.resize(sub_operands.size());
      for (auto v : llvm::enumerate(sub_operands))
        allOperands[v.index()].push_back(v.value());
    }
    if (allOperands.size() == 0)
      allOperands.push_back({});
    for (const SmallVector<Value> &operands : allOperands) {
      Value curr =
          ((ConcreteT *)(this))
              ->createDestOp(op, adaptor, rewriter, elemTy, operands, loc);
      if (!bool(curr))
        return failure();
      resultVals.push_back(curr);
    }
    if (op->getNumOperands() > 0) {
      auto argTy = op->getOperand(0).getType();
      resultVals = reorderValues(resultVals, argTy, resultTy);
    }
    resultVals =
        packI32(resultVals, resultTy, rewriter, loc, this->getTypeConverter());
    Value view = this->getTypeConverter()->packLLElements(loc, resultVals,
                                                          rewriter, resultTy);
    rewriter.replaceOp(op, view);

    return success();
  }
};

template <typename SourceOp, typename DestOp>
struct ElementwiseOpConversion
    : public ElementwiseOpConversionBase<
          SourceOp, ElementwiseOpConversion<SourceOp, DestOp>> {
  using Base =
      ElementwiseOpConversionBase<SourceOp,
                                  ElementwiseOpConversion<SourceOp, DestOp>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  explicit ElementwiseOpConversion(LLVMTypeConverter &typeConverter,
                                   PatternBenefit benefit = 1)
      : ElementwiseOpConversionBase<SourceOp, ElementwiseOpConversion>(
            typeConverter, benefit) {}

  // An interface to support variant DestOp builder.
  DestOp createDestOp(SourceOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter, Type elemTy,
                      ValueRange operands, Location loc) const {
    return rewriter.create<DestOp>(loc, elemTy, operands,
                                   adaptor.getAttributes().getValue());
  }
};

struct CmpIOpConversion
    : public ElementwiseOpConversionBase<triton::gpu::CmpIOp,
                                         CmpIOpConversion> {
  using Base =
      ElementwiseOpConversionBase<triton::gpu::CmpIOp, CmpIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  LLVM::ICmpOp createDestOp(triton::gpu::CmpIOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter, Type elemTy,
                            ValueRange operands, Location loc) const {
    return rewriter.create<LLVM::ICmpOp>(
        loc, elemTy, ArithCmpIPredicateToLLVM(op.getPredicate()), operands[0],
        operands[1]);
  }

  static LLVM::ICmpPredicate
  ArithCmpIPredicateToLLVM(arith::CmpIPredicate predicate) {
    switch (predicate) {
#define __PRED_ENUM(item__)                                                    \
  case arith::CmpIPredicate::item__:                                           \
    return LLVM::ICmpPredicate::item__

      __PRED_ENUM(eq);
      __PRED_ENUM(ne);
      __PRED_ENUM(sgt);
      __PRED_ENUM(sge);
      __PRED_ENUM(slt);
      __PRED_ENUM(sle);
      __PRED_ENUM(ugt);
      __PRED_ENUM(uge);
      __PRED_ENUM(ult);
      __PRED_ENUM(ule);

#undef __PRED_ENUM
    }
    llvm_unreachable("Unknown arith::CmpIPredicate");
  }
};

struct CmpFOpConversion
    : public ElementwiseOpConversionBase<triton::gpu::CmpFOp,
                                         CmpFOpConversion> {
  using Base =
      ElementwiseOpConversionBase<triton::gpu::CmpFOp, CmpFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  static LLVM::FCmpOp createDestOp(triton::gpu::CmpFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, ValueRange operands,
                                   Location loc) {
    return rewriter.create<LLVM::FCmpOp>(
        loc, elemTy, ArithCmpFPredicateToLLVM(op.getPredicate()), operands[0],
        operands[1]);
  }

  static LLVM::FCmpPredicate
  ArithCmpFPredicateToLLVM(arith::CmpFPredicate predicate) {
    switch (predicate) {
#define __PRED_ENUM(item__, item1__)                                           \
  case arith::CmpFPredicate::item__:                                           \
    return LLVM::FCmpPredicate::item1__

      __PRED_ENUM(OEQ, oeq);
      __PRED_ENUM(ONE, one);
      __PRED_ENUM(OGT, ogt);
      __PRED_ENUM(OGE, oge);
      __PRED_ENUM(OLT, olt);
      __PRED_ENUM(OLE, ole);
      __PRED_ENUM(ORD, ord);
      __PRED_ENUM(UEQ, ueq);
      __PRED_ENUM(UGT, ugt);
      __PRED_ENUM(UGE, uge);
      __PRED_ENUM(ULT, ult);
      __PRED_ENUM(ULE, ule);
      __PRED_ENUM(UNE, une);
      __PRED_ENUM(UNO, uno);
      __PRED_ENUM(AlwaysTrue, _true);
      __PRED_ENUM(AlwaysFalse, _false);

#undef __PRED_ENUM
    }
    llvm_unreachable("Unknown arith::CmpFPredicate");
  }
};

template <class T>
struct ExternElementwiseOpConversion
    : public ElementwiseOpConversionBase<T, ExternElementwiseOpConversion<T>> {
  using Base = ElementwiseOpConversionBase<T, ExternElementwiseOpConversion<T>>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;
  typedef typename Base::OpAdaptor OpAdaptor;

  Value createDestOp(T op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    StringRef funcName = op.getSymbol();
    if (funcName.empty())
      llvm::errs() << "ExternElementwiseOpConversion";

    Type funcType = getFunctionType(elemTy, operands);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetFuncOp(rewriter, op, funcName, funcType);
    return rewriter.create<LLVM::CallOp>(loc, funcOp, operands).getResult();
  }

private:
  Type getFunctionType(Type resultType, ValueRange operands) const {
    SmallVector<Type> operandTypes(operands.getTypes());
    return LLVM::LLVMFunctionType::get(resultType, operandTypes);
  }

  LLVM::LLVMFuncOp appendOrGetFuncOp(ConversionPatternRewriter &rewriter, T op,
                                     StringRef funcName, Type funcType) const {
    using LLVM::LLVMFuncOp;

    auto funcAttr = StringAttr::get(op->getContext(), funcName);
    Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
    if (funcOp)
      return cast<LLVMFuncOp>(*funcOp);

    auto parent = ((Operation *)op)->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    mlir::OpBuilder b(parent);
    auto ret = b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
    ret.getOperation()->setAttr(
        "libname", StringAttr::get(op->getContext(), op.getLibname()));
    ret.getOperation()->setAttr(
        "libpath", StringAttr::get(op->getContext(), op.getLibpath()));
    return ret;
  }
};

struct FDivOpConversion
    : ElementwiseOpConversionBase<mlir::arith::DivFOp, FDivOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::DivFOp, FDivOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::DivFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    PTXBuilder ptxBuilder;
    auto &fdiv = *ptxBuilder.create<PTXInstr>("div");
    unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
    if (32 == bitwidth) {
      fdiv.o("full").o("f32");
    } else if (64 == bitwidth) {
      fdiv.o("rn").o("f64");
    } else {
      assert(0 && bitwidth && "not supported");
    }

    auto res = ptxBuilder.newOperand(bitwidth == 32 ? "=r" : "=l");
    auto lhs = ptxBuilder.newOperand(operands[0], bitwidth == 32 ? "r" : "l");
    auto rhs = ptxBuilder.newOperand(operands[1], bitwidth == 32 ? "r" : "l");
    fdiv(res, lhs, rhs);

    Value ret = ptxBuilder.launch(rewriter, loc, elemTy, false);
    return ret;
  }
};

struct FMulOpConversion
    : ElementwiseOpConversionBase<mlir::arith::MulFOp, FMulOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::MulFOp, FMulOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::MulFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      PTXBuilder builder;
      auto ptxAsm = " { .reg .b16 c;        \n"
                    "    mov.b16 c, 0x8000U; \n" // 0.0
                    "    fma.rn.bf16 $0, $1, $2, c; } \n";
      auto &fMul = *builder.create<PTXInstr>(ptxAsm);
      auto res = builder.newOperand("=h");
      auto lhs = builder.newOperand(operands[0], "h");
      auto rhs = builder.newOperand(operands[1], "h");
      fMul({res, lhs, rhs}, /*onlyAttachMLIRArgs=*/true);
      return builder.launch(rewriter, loc, i16_ty, false);
    } else {
      return rewriter.create<LLVM::FMulOp>(loc, elemTy, operands[0],
                                           operands[1]);
    }
  }
};

struct FAddOpConversion
    : ElementwiseOpConversionBase<mlir::arith::AddFOp, FAddOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::AddFOp, FAddOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::AddFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      PTXBuilder builder;
      auto ptxAsm = "{ .reg .b16 c;         \n"
                    "   mov.b16 c, 0x3f80U; \n" // 1.0
                    "   fma.rn.bf16 $0, $1, c, $2; } \n";
      auto &fAdd = *builder.create<PTXInstr>(ptxAsm);
      auto res = builder.newOperand("=h");
      auto lhs = builder.newOperand(operands[0], "h");
      auto rhs = builder.newOperand(operands[1], "h");
      fAdd({res, lhs, rhs}, /*onlyAttachMLIRArgs=*/true);
      return builder.launch(rewriter, loc, i16_ty, false);
    } else {
      return rewriter.create<LLVM::FAddOp>(loc, elemTy, operands[0],
                                           operands[1]);
    }
  }
};

struct FSubOpConversion
    : ElementwiseOpConversionBase<mlir::arith::SubFOp, FSubOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::SubFOp, FSubOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::SubFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      PTXBuilder builder;
      auto ptxAsm = " { .reg .b16 c;         \n"
                    "    mov.b16 c, 0xbf80U; \n" // -1.0
                    "    fma.rn.bf16 $0, $2, c, $1;} \n";
      auto &fSub = *builder.create<PTXInstr>(ptxAsm);
      auto res = builder.newOperand("=h");
      auto lhs = builder.newOperand(operands[0], "h");
      auto rhs = builder.newOperand(operands[1], "h");
      fSub({res, lhs, rhs}, /*onlyAttachMLIRArgs=*/true);
      return builder.launch(rewriter, loc, i16_ty, false);
    } else {
      return rewriter.create<LLVM::FSubOp>(loc, elemTy, operands[0],
                                           operands[1]);
    }
  }
};

struct SIToFPOpConversion
    : ElementwiseOpConversionBase<mlir::arith::SIToFPOp, SIToFPOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::SIToFPOp, SIToFPOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::SIToFPOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16()) {
      auto value = rewriter.create<LLVM::SIToFPOp>(loc, f32_ty, operands[0]);
      return FpToFpOpConversion::convertFp32ToBf16(loc, rewriter, value);
    } else {
      return rewriter.create<LLVM::SIToFPOp>(loc, elemTy, operands[0]);
    }
  }
};

struct FPToSIOpConversion
    : ElementwiseOpConversionBase<mlir::arith::FPToSIOp, FPToSIOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::FPToSIOp, FPToSIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::FPToSIOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto value =
          FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0]);
      return rewriter.create<LLVM::FPToSIOp>(loc, elemTy, value);
    } else {
      return rewriter.create<LLVM::FPToSIOp>(loc, elemTy, operands[0]);
    }
  }
};

struct ExtFOpConversion
    : ElementwiseOpConversionBase<mlir::arith::ExtFOp, ExtFOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::ExtFOp, ExtFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::ExtFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto outElemTy = getElementType(op.getOut());
      assert(outElemTy.isF32() && "unsupported conversion");
      return FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0]);
    } else {
      return rewriter.create<LLVM::FPExtOp>(loc, elemTy, operands[0]);
    }
  }
};

struct TruncFOpConversion
    : ElementwiseOpConversionBase<mlir::arith::TruncFOp, TruncFOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::TruncFOp, TruncFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::TruncFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16()) {
      auto inElemTy = getElementType(op.getIn());
      assert(inElemTy.isF32() && "unsupported conversion");
      return FpToFpOpConversion::convertFp32ToBf16(loc, rewriter, operands[0]);
    } else {
      return rewriter.create<LLVM::FPTruncOp>(loc, elemTy, operands[0]);
    }
  }
};

struct ExpOpConversionApprox
    : ElementwiseOpConversionBase<mlir::math::ExpOp, ExpOpConversionApprox> {
  using Base =
      ElementwiseOpConversionBase<mlir::math::ExpOp, ExpOpConversionApprox>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::math::ExpOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    // For non-FP32 input, call __nv_expf for higher-precision calculation
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    const double log2e = 1.4426950408889634;
    Value prod = fmul(f32_ty, operands[0], f32_val(log2e));

    PTXBuilder ptxBuilder;
    auto &exp2 = ptxBuilder.create<PTXInstr>("ex2")->o("approx").o("f32");
    auto output = ptxBuilder.newOperand("=f");
    auto input = ptxBuilder.newOperand(prod, "f");
    exp2(output, input);
    return ptxBuilder.launch(rewriter, loc, f32_ty, false);
  }
};

struct AbsIOpConversion
    : ElementwiseOpConversionBase<mlir::math::AbsIOp, AbsIOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::math::AbsIOp, AbsIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::math::AbsIOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto boolFalse = rewriter.getBoolAttr(false);
    auto constFalse = rewriter.create<LLVM::ConstantOp>(loc, boolFalse);
    return rewriter.create<LLVM::AbsOp>(loc, elemTy, operands[0],
                                        /*is_int_min_poison=*/constFalse);
  }
};

struct AbsFOpConversion
    : ElementwiseOpConversionBase<mlir::math::AbsFOp, AbsFOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::math::AbsFOp, AbsFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::math::AbsFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    if (llvm::isa<IntegerType>(elemTy)) {
      // Mask out the sign bit
      auto num_bits =
          getElementTypeOrSelf(op.getType()).getIntOrFloatBitWidth();
      assert(num_bits <= 16);
      auto mask = (1u << (num_bits - 1u)) - 1u;
      auto maskAttr = rewriter.getIntegerAttr(elemTy, mask);
      auto maskConst = rewriter.create<LLVM::ConstantOp>(loc, maskAttr);
      return and_(operands[0], maskConst);
    }

    return rewriter.create<LLVM::FAbsOp>(loc, elemTy, operands[0]);
  }
};

/// The lowering of index_cast becomes an integer conversion since index
/// becomes an integer.  If the bit width of the source and target integer
/// types is the same, just erase the cast.  If the target type is wider,
/// sign-extend the value, otherwise truncate it.
struct IndexCastOpLowering
    : public ElementwiseOpConversionBase<arith::IndexCastOp,
                                         IndexCastOpLowering> {
  using Base =
      ElementwiseOpConversionBase<arith::IndexCastOp, IndexCastOpLowering>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(arith::IndexCastOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto inElemTy =
        this->getTypeConverter()->convertType(getElementType(op.getIn()));
    unsigned targetBits = elemTy.getIntOrFloatBitWidth();
    unsigned sourceBits = inElemTy.getIntOrFloatBitWidth();

    if (targetBits == sourceBits)
      return operands[0];
    if (targetBits < sourceBits)
      return rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, elemTy,
                                                        operands[0]);
    return rewriter.replaceOpWithNewOp<LLVM::SExtOp>(op, elemTy, operands[0]);
  }
};

void populateElementwiseOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
#define POPULATE_TERNARY_OP(SRC_OP, DST_OP)                                    \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(typeConverter, benefit);
  POPULATE_TERNARY_OP(triton::gpu::SelectOp, LLVM::SelectOp)
  POPULATE_TERNARY_OP(arith::SelectOp, LLVM::SelectOp)
#undef POPULATE_TERNARY_OP

#define POPULATE_BINARY_OP(SRC_OP, DST_OP)                                     \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(typeConverter, benefit);
  POPULATE_BINARY_OP(arith::SubIOp, LLVM::SubOp) // -
  POPULATE_BINARY_OP(arith::AddIOp, LLVM::AddOp) // +
  POPULATE_BINARY_OP(arith::MulIOp, LLVM::MulOp) // *
  POPULATE_BINARY_OP(arith::DivSIOp, LLVM::SDivOp)
  POPULATE_BINARY_OP(arith::DivUIOp, LLVM::UDivOp)
  POPULATE_BINARY_OP(arith::RemFOp, LLVM::FRemOp) // %
  POPULATE_BINARY_OP(arith::RemSIOp, LLVM::SRemOp)
  POPULATE_BINARY_OP(arith::RemUIOp, LLVM::URemOp)
  POPULATE_BINARY_OP(arith::AndIOp, LLVM::AndOp)    // &
  POPULATE_BINARY_OP(arith::OrIOp, LLVM::OrOp)      // |
  POPULATE_BINARY_OP(arith::XOrIOp, LLVM::XOrOp)    // ^
  POPULATE_BINARY_OP(arith::ShLIOp, LLVM::ShlOp)    // <<
  POPULATE_BINARY_OP(arith::ShRSIOp, LLVM::AShrOp)  // >>
  POPULATE_BINARY_OP(arith::ShRUIOp, LLVM::LShrOp)  // >>
  POPULATE_BINARY_OP(arith::MinFOp, LLVM::MinNumOp) // fmin
  POPULATE_BINARY_OP(arith::MinSIOp, LLVM::SMinOp)  // smin
#undef POPULATE_BINARY_OP

#define POPULATE_UNARY_OP(SRC_OP, DST_OP)                                      \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(typeConverter, benefit);
  POPULATE_UNARY_OP(arith::TruncIOp, LLVM::TruncOp)
  POPULATE_UNARY_OP(arith::ExtSIOp, LLVM::SExtOp)
  POPULATE_UNARY_OP(arith::ExtUIOp, LLVM::ZExtOp)
  POPULATE_UNARY_OP(arith::FPToUIOp, LLVM::FPToUIOp)
  POPULATE_UNARY_OP(arith::UIToFPOp, LLVM::UIToFPOp)
  POPULATE_UNARY_OP(math::LogOp, math::LogOp)
  POPULATE_UNARY_OP(math::CosOp, math::CosOp)
  POPULATE_UNARY_OP(math::SinOp, math::SinOp)
  POPULATE_UNARY_OP(math::SqrtOp, math::SqrtOp)
  POPULATE_UNARY_OP(math::ExpOp, math::ExpOp)
  POPULATE_UNARY_OP(triton::BitcastOp, LLVM::BitcastOp)
  POPULATE_UNARY_OP(triton::IntToPtrOp, LLVM::IntToPtrOp)
  POPULATE_UNARY_OP(triton::PtrToIntOp, LLVM::PtrToIntOp)
#undef POPULATE_UNARY_OP

  patterns.add<AbsIOpConversion>(typeConverter, benefit);
  patterns.add<AbsFOpConversion>(typeConverter, benefit);
  patterns.add<CmpIOpConversion>(typeConverter, benefit);
  patterns.add<CmpFOpConversion>(typeConverter, benefit);

  patterns.add<FDivOpConversion>(typeConverter, benefit);
  patterns.add<FSubOpConversion>(typeConverter, benefit);
  patterns.add<FAddOpConversion>(typeConverter, benefit);
  patterns.add<FMulOpConversion>(typeConverter, benefit);

  patterns.add<ExtFOpConversion>(typeConverter, benefit);
  patterns.add<TruncFOpConversion>(typeConverter, benefit);
  patterns.add<FPToSIOpConversion>(typeConverter, benefit);
  patterns.add<SIToFPOpConversion>(typeConverter, benefit);
  patterns.add<IndexCastOpLowering>(typeConverter, benefit);

  patterns.add<FpToFpOpConversion>(typeConverter, benefit);

  patterns.add<ExternElementwiseOpConversion<triton::PureExternElementwiseOp>>(
      typeConverter, benefit);
  patterns
      .add<ExternElementwiseOpConversion<triton::ImpureExternElementwiseOp>>(
          typeConverter, benefit);
  // ExpOpConversionApprox will try using ex2.approx if the input type is
  // FP32. For other input types, ExpOpConversionApprox will return failure and
  // ElementwiseOpConversion<math::ExpOp, math::ExpOp> defined below will call
  // __nv_expf for higher-precision calculation
  patterns.add<ExpOpConversionApprox>(typeConverter, benefit);
}
