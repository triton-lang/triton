#include "ElementwiseOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;
using ::mlir::triton::gpu::getElemsPerThread;

struct FpToFpOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::FpToFpOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::FpToFpOp>::ConvertTritonGPUOpToLLVMPattern;

  /* ------------------ */
  // FP8 -> FP16
  /* ------------------ */

  static SmallVector<Value>
  convertFp8x4ToFp16x4(Location loc, ConversionPatternRewriter &rewriter,
                       const char *ptxAsm, const Value &v0, const Value &v1,
                       const Value &v2, const Value &v3) {
    auto ctx = rewriter.getContext();
    auto fp8x4VecTy = vec_ty(i8_ty, 4);
    Value fp8x4Vec = undef(fp8x4VecTy);
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v0, i32_val(0));
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v1, i32_val(1));
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v2, i32_val(2));
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v3, i32_val(3));
    fp8x4Vec = bitcast(fp8x4Vec, i32_ty);

    PTXBuilder builder;
    auto &ptxOp = *builder.create(ptxAsm);

    auto *o0 = builder.newOperand("=r");
    auto *o1 = builder.newOperand("=r");
    auto *i = builder.newOperand(fp8x4Vec, "r");
    ptxOp({o0, o1, i}, /*onlyAttachMLIRArgs=*/true);

    auto fp16x2VecTy = vec_ty(f16_ty, 2);
    auto fp16x2x2StructTy =
        struct_ty(SmallVector<Type>{fp16x2VecTy, fp16x2VecTy});
    auto fp16x2x2Struct =
        builder.launch(rewriter, loc, fp16x2x2StructTy, false);
    auto fp16x2Vec0 = extract_val(fp16x2VecTy, fp16x2x2Struct, 0);
    auto fp16x2Vec1 = extract_val(fp16x2VecTy, fp16x2x2Struct, 1);
    return {extract_element(f16_ty, fp16x2Vec0, i32_val(0)),
            extract_element(f16_ty, fp16x2Vec0, i32_val(1)),
            extract_element(f16_ty, fp16x2Vec1, i32_val(0)),
            extract_element(f16_ty, fp16x2Vec1, i32_val(1))};
  }

  static SmallVector<Value>
  convertFp8E4M3x4ToFp16x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    auto *ptxAsm = // WARN: subnormal (0bs0000xxx) are not handled
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
    return convertFp8x4ToFp16x4(loc, rewriter, ptxAsm, v0, v1, v2, v3);
  }

  static SmallVector<Value>
  convertFp8E5M2x4ToFp16x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    // exponent bias of Fp8E5M2 and Fp16 are the same
    auto *ptxAsm = "{                           \n"
                   "prmt.b32 $0, 0, $2, 0x5140; \n\t"
                   "prmt.b32 $1, 0, $2, 0x7362; \n\t"
                   "}";
    return convertFp8x4ToFp16x4(loc, rewriter, ptxAsm, v0, v1, v2, v3);
  }

  /* ------------------ */
  // FP8 -> BF16
  /* ------------------ */
  static SmallVector<Value>
  convertFp8x4ToBf16x4(Location loc, ConversionPatternRewriter &rewriter,
                       const char *ptxAsm, const Value &v0, const Value &v1,
                       const Value &v2, const Value &v3) {
    auto ctx = rewriter.getContext();
    auto fp8x4VecTy = vec_ty(i8_ty, 4);
    Value fp8x4Vec = undef(fp8x4VecTy);
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v0, i32_val(0));
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v1, i32_val(1));
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v2, i32_val(2));
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v3, i32_val(3));
    fp8x4Vec = bitcast(fp8x4Vec, i32_ty);

    PTXBuilder builder;
    auto &ptxOp = *builder.create(ptxAsm);

    auto *o0 = builder.newOperand("=r");
    auto *o1 = builder.newOperand("=r");
    auto *i = builder.newOperand(fp8x4Vec, "r");
    ptxOp({o0, o1, i}, /* onlyAttachMLIRArgs */ true);

    auto bf16x2VecTy = vec_ty(i16_ty, 2);
    auto bf16x2x2StructTy =
        struct_ty(SmallVector<Type>{bf16x2VecTy, bf16x2VecTy});
    auto bf16x2x2Struct =
        builder.launch(rewriter, loc, bf16x2x2StructTy, false);
    auto bf16x2Vec0 = extract_val(bf16x2VecTy, bf16x2x2Struct, 0);
    auto bf16x2Vec1 = extract_val(bf16x2VecTy, bf16x2x2Struct, 1);
    return {extract_element(i16_ty, bf16x2Vec0, i32_val(0)),
            extract_element(i16_ty, bf16x2Vec0, i32_val(1)),
            extract_element(i16_ty, bf16x2Vec1, i32_val(0)),
            extract_element(i16_ty, bf16x2Vec1, i32_val(1))};
  }

  static SmallVector<Value>
  convertFp8E4M3x4ToBf16x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    auto *ptxAsm = // WARN: subnormal (0bs0000xxx) are not handled
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
    return convertFp8x4ToBf16x4(loc, rewriter, ptxAsm, v0, v1, v2, v3);
  };

  static SmallVector<Value>
  convertFp8E5M2x4ToBf16x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    auto *ptxAsm = // WARN: subnormal (0bs00000xx) are not handled
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
    return convertFp8x4ToBf16x4(loc, rewriter, ptxAsm, v0, v1, v2, v3);
  };

  /* ------------------ */
  // FP16 -> FP8
  /* ------------------ */

  static SmallVector<Value>
  convertFp16x4ToFp8x4(Location loc, ConversionPatternRewriter &rewriter,
                       const char *ptxAsm, const Value &v0, const Value &v1,
                       const Value &v2, const Value &v3) {
    auto fp16x2VecTy = vec_ty(f16_ty, 2);
    Value fp16x2Vec0 = undef(fp16x2VecTy);
    Value fp16x2Vec1 = undef(fp16x2VecTy);
    fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v0, i32_val(0));
    fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v1, i32_val(1));
    fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v2, i32_val(0));
    fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v3, i32_val(1));
    fp16x2Vec0 = bitcast(fp16x2Vec0, i32_ty);
    fp16x2Vec1 = bitcast(fp16x2Vec1, i32_ty);

    PTXBuilder builder;
    auto &ptxOp = *builder.create(ptxAsm);

    auto *o = builder.newOperand("=r");
    auto *i0 = builder.newOperand(fp16x2Vec0, "r");
    auto *i1 = builder.newOperand(fp16x2Vec1, "r");
    ptxOp({o, i0, i1}, /*onlyAttachMLIRArgs=*/true);

    auto fp8x4VecTy = vec_ty(i8_ty, 4);
    auto fp8x4Vec = builder.launch(rewriter, loc, fp8x4VecTy, false);
    return {extract_element(i8_ty, fp8x4Vec, i32_val(0)),
            extract_element(i8_ty, fp8x4Vec, i32_val(1)),
            extract_element(i8_ty, fp8x4Vec, i32_val(2)),
            extract_element(i8_ty, fp8x4Vec, i32_val(3))};
  }

  static SmallVector<Value>
  convertFp16x4ToFp8E4M3x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    auto *ptxAsm = // WARN: subnormal Fp8s are not handled
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
    return convertFp16x4ToFp8x4(loc, rewriter, ptxAsm, v0, v1, v2, v3);
  }

  static SmallVector<Value>
  convertFp16x4ToFp8E5M2x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    auto *ptxAsm =
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
    return convertFp16x4ToFp8x4(loc, rewriter, ptxAsm, v0, v1, v2, v3);
  }

  /* ------------------ */
  // FP32 -> FP8
  /* ------------------ */

  static SmallVector<Value>
  convertFp32x4ToFp8E4M3x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    auto c0 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v0);
    auto c1 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v1);
    auto c2 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v2);
    auto c3 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v3);
    return convertFp16x4ToFp8E4M3x4(loc, rewriter, c0, c1, c2, c3);
  }

  static SmallVector<Value>
  convertFp32x4ToFp8E5M2x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    auto c0 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v0);
    auto c1 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v1);
    auto c2 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v2);
    auto c3 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v3);
    return convertFp16x4ToFp8E5M2x4(loc, rewriter, c0, c1, c2, c3);
  }

  /* ------------------ */
  // BF16 -> FP8
  /* ------------------ */

  static SmallVector<Value>
  convertBf16x4ToFp8x4(Location loc, ConversionPatternRewriter &rewriter,
                       const char *ptxAsm, const Value &v0, const Value &v1,
                       const Value &v2, const Value &v3) {
    auto bf16x2VecTy = vec_ty(i16_ty, 2);
    Value bf16x2Vec0 = undef(bf16x2VecTy);
    Value bf16x2Vec1 = undef(bf16x2VecTy);
    bf16x2Vec0 = insert_element(bf16x2VecTy, bf16x2Vec0, v0, i32_val(0));
    bf16x2Vec0 = insert_element(bf16x2VecTy, bf16x2Vec0, v1, i32_val(1));
    bf16x2Vec1 = insert_element(bf16x2VecTy, bf16x2Vec1, v2, i32_val(0));
    bf16x2Vec1 = insert_element(bf16x2VecTy, bf16x2Vec1, v3, i32_val(1));
    bf16x2Vec0 = bitcast(bf16x2Vec0, i32_ty);
    bf16x2Vec1 = bitcast(bf16x2Vec1, i32_ty);

    PTXBuilder builder;
    auto &ptxOp = *builder.create(ptxAsm);

    auto *o = builder.newOperand("=r");
    auto *i0 = builder.newOperand(bf16x2Vec0, "r");
    auto *i1 = builder.newOperand(bf16x2Vec1, "r");
    ptxOp({o, i0, i1}, /*onlyAttachMLIRArgs=*/true);

    auto fp8x4VecTy = vec_ty(i8_ty, 4);
    auto fp8x4Vec = builder.launch(rewriter, loc, fp8x4VecTy, false);
    return {extract_element(i8_ty, fp8x4Vec, i32_val(0)),
            extract_element(i8_ty, fp8x4Vec, i32_val(1)),
            extract_element(i8_ty, fp8x4Vec, i32_val(2)),
            extract_element(i8_ty, fp8x4Vec, i32_val(3))};
  }

  static SmallVector<Value>
  convertBf16x4ToFp8E4M3x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    auto *ptxAsm = // bf16 is clamped firstly to fp8 min/max
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
    return convertBf16x4ToFp8x4(loc, rewriter, ptxAsm, v0, v1, v2, v3);
  };

  static SmallVector<Value>
  convertBf16x4ToFp8E5M2x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    auto *ptxAsm = // bf16 is clamped firstly to fp8 min/max
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
    return convertBf16x4ToFp8x4(loc, rewriter, ptxAsm, v0, v1, v2, v3);
  }

  /* ------------------ */
  // FP8 -> FP32
  /* ------------------ */

  static SmallVector<Value>
  convertFp8E4M3x4ToFp32x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    auto fp16Values = convertFp8E4M3x4ToFp16x4(loc, rewriter, v0, v1, v2, v3);
    return {convertFp16ToFp32(loc, rewriter, fp16Values[0]),
            convertFp16ToFp32(loc, rewriter, fp16Values[1]),
            convertFp16ToFp32(loc, rewriter, fp16Values[2]),
            convertFp16ToFp32(loc, rewriter, fp16Values[3])};
  }

  static SmallVector<Value>
  convertFp8E5M2x4ToFp32x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    auto fp16Values = convertFp8E5M2x4ToFp16x4(loc, rewriter, v0, v1, v2, v3);
    return {rewriter.create<LLVM::FPExtOp>(loc, f32_ty, fp16Values[0]),
            rewriter.create<LLVM::FPExtOp>(loc, f32_ty, fp16Values[1]),
            rewriter.create<LLVM::FPExtOp>(loc, f32_ty, fp16Values[2]),
            rewriter.create<LLVM::FPExtOp>(loc, f32_ty, fp16Values[3])};
  }

  //

  static SmallVector<Value>
  convertFp8E4M3x4ToFp64x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    auto fp16Values = convertFp8E4M3x4ToFp16x4(loc, rewriter, v0, v1, v2, v3);
    return {rewriter.create<LLVM::FPExtOp>(loc, f64_ty, fp16Values[0]),
            rewriter.create<LLVM::FPExtOp>(loc, f64_ty, fp16Values[1]),
            rewriter.create<LLVM::FPExtOp>(loc, f64_ty, fp16Values[2]),
            rewriter.create<LLVM::FPExtOp>(loc, f64_ty, fp16Values[3])};
  }

  static SmallVector<Value>
  convertFp64x4ToFp8E4M3x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    auto c0 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v0);
    auto c1 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v1);
    auto c2 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v2);
    auto c3 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v3);
    return convertFp16x4ToFp8E4M3x4(loc, rewriter, c0, c1, c2, c3);
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

  LogicalResult
  matchAndRewrite(triton::FpToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTensorType = op.getFrom().getType().cast<mlir::RankedTensorType>();
    auto dstTensorType =
        op.getResult().getType().cast<mlir::RankedTensorType>();
    auto srcEltType = srcTensorType.getElementType();
    auto dstEltType = dstTensorType.getElementType();
    auto loc = op->getLoc();
    auto elems = getElemsPerThread(dstTensorType);
    SmallVector<Value> resultVals;
    bool isSrcFP8 =
        srcEltType.isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>();
    bool isDstFP8 =
        dstEltType.isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>();

    // Select convertor
    typedef std::function<SmallVector<Value>(
        Location, ConversionPatternRewriter &, const Value &, const Value &,
        const Value &, const Value &)>
        ConvertorT;

    auto F8E4M3TyID = TypeID::get<mlir::Float8E4M3FNType>();
    auto F8E5M2TyID = TypeID::get<mlir::Float8E5M2Type>();
    auto F16TyID = TypeID::get<mlir::Float16Type>();
    auto BF16TyID = TypeID::get<mlir::BFloat16Type>();
    auto F32TyID = TypeID::get<mlir::Float32Type>();
    auto F64TyID = TypeID::get<mlir::Float64Type>();
    DenseMap<std::pair<TypeID, TypeID>, ConvertorT> convertorMap = {
        // F8 -> F16
        {{F8E4M3TyID, F16TyID}, convertFp8E4M3x4ToFp16x4},
        {{F8E5M2TyID, F16TyID}, convertFp8E5M2x4ToFp16x4},
        // F16 -> F8
        {{F16TyID, F8E4M3TyID}, convertFp16x4ToFp8E4M3x4},
        {{F16TyID, F8E5M2TyID}, convertFp16x4ToFp8E5M2x4},
        // F8 -> BF16
        {{F8E4M3TyID, BF16TyID}, convertFp8E4M3x4ToBf16x4},
        {{F8E5M2TyID, BF16TyID}, convertFp8E5M2x4ToBf16x4},
        // BF16 -> F8
        {{BF16TyID, F8E4M3TyID}, convertBf16x4ToFp8E4M3x4},
        {{BF16TyID, F8E5M2TyID}, convertBf16x4ToFp8E5M2x4},
        // F8 -> F32
        {{F8E4M3TyID, F32TyID}, convertFp8E4M3x4ToFp32x4},
        {{F8E5M2TyID, F32TyID}, convertFp8E5M2x4ToFp32x4},
        // F32 -> F8
        {{F32TyID, F8E4M3TyID}, convertFp32x4ToFp8E4M3x4},
        {{F32TyID, F8E5M2TyID}, convertFp32x4ToFp8E5M2x4},
    };

    std::pair<TypeID, TypeID> key = {srcEltType.getTypeID(),
                                     dstEltType.getTypeID()};
    if (convertorMap.count(key) == 0) {
      llvm::errs() << "Unsupported conversion from " << srcEltType << " to "
                   << dstEltType << "\n";
      llvm_unreachable("");
    }
    auto convertor = convertorMap.lookup(key);

    // Vectorized casting
    assert(elems % 4 == 0 &&
           "FP8 casting only support tensors with 4-aligned sizes");
    auto elements = getTypeConverter()->unpackLLElements(
        loc, adaptor.getFrom(), rewriter, srcTensorType);
    for (size_t i = 0; i < elems; i += 4) {
      auto converted = convertor(loc, rewriter, elements[i], elements[i + 1],
                                 elements[i + 2], elements[i + 3]);
      resultVals.append(converted);
    }

    assert(resultVals.size() == elems);
    auto result = getTypeConverter()->packLLElements(loc, resultVals, rewriter,
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

    unsigned elems = getElemsPerThread(resultTy);
    auto resultElementTy = getElementTypeOrSelf(resultTy);
    Type elemTy = this->getTypeConverter()->convertType(resultElementTy);
    SmallVector<Type> types(elems, elemTy);
    Type structTy = this->getTypeConverter()->convertType(resultTy);

    auto *concreteThis = static_cast<const ConcreteT *>(this);
    auto operands = getOperands(rewriter, adaptor, resultTy, elems, loc);
    SmallVector<Value> resultVals(elems);
    for (unsigned i = 0; i < elems; ++i) {
      resultVals[i] = concreteThis->createDestOp(op, adaptor, rewriter, elemTy,
                                                 operands[i], loc);
      if (!bool(resultVals[i]))
        return failure();
    }
    Value view = this->getTypeConverter()->packLLElements(loc, resultVals,
                                                          rewriter, resultTy);
    rewriter.replaceOp(op, view);

    return success();
  }

protected:
  SmallVector<SmallVector<Value>>
  getOperands(ConversionPatternRewriter &rewriter, OpAdaptor adaptor,
              Type operandTy, const unsigned elems, Location loc) const {
    SmallVector<SmallVector<Value>> operands(elems);
    for (auto operand : adaptor.getOperands()) {
      auto sub_operands = this->getTypeConverter()->unpackLLElements(
          loc, operand, rewriter, operandTy);
      for (size_t i = 0; i < elems; ++i) {
        operands[i].push_back(sub_operands[i]);
      }
    }
    return operands;
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

struct ExtElemwiseOpConversion
    : public ElementwiseOpConversionBase<triton::ExtElemwiseOp,
                                         ExtElemwiseOpConversion> {
  using Base = ElementwiseOpConversionBase<triton::ExtElemwiseOp,
                                           ExtElemwiseOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(triton::ExtElemwiseOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    StringRef funcName = op.getSymbol();
    if (funcName.empty())
      llvm::errs() << "ExtElemwiseOpConversion";

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

  LLVM::LLVMFuncOp appendOrGetFuncOp(ConversionPatternRewriter &rewriter,
                                     triton::ExtElemwiseOp op,
                                     StringRef funcName, Type funcType) const {
    using LLVM::LLVMFuncOp;

    auto funcAttr = StringAttr::get(op->getContext(), funcName);
    Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
    if (funcOp)
      return cast<LLVMFuncOp>(*funcOp);

    mlir::OpBuilder b(op->getParentOfType<LLVMFuncOp>());
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

void populateElementwiseOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, AxisInfoAnalysis &axisInfoAnalysis,
    const Allocation *allocation, Value smem, PatternBenefit benefit) {
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
  POPULATE_BINARY_OP(arith::AndIOp, LLVM::AndOp)   // &
  POPULATE_BINARY_OP(arith::OrIOp, LLVM::OrOp)     // |
  POPULATE_BINARY_OP(arith::XOrIOp, LLVM::XOrOp)   // ^
  POPULATE_BINARY_OP(arith::ShLIOp, LLVM::ShlOp)   // <<
  POPULATE_BINARY_OP(arith::ShRSIOp, LLVM::AShrOp) // >>
  POPULATE_BINARY_OP(arith::ShRUIOp, LLVM::LShrOp) // >>
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

  patterns.add<FpToFpOpConversion>(typeConverter, benefit);

  patterns.add<ExtElemwiseOpConversion>(typeConverter, benefit);
  // ExpOpConversionApprox will try using ex2.approx if the input type is
  // FP32. For other input types, ExpOpConversionApprox will return failure and
  // ElementwiseOpConversion<math::ExpOp, math::ExpOp> defined below will call
  // __nv_expf for higher-precision calculation
  patterns.add<ExpOpConversionApprox>(typeConverter, benefit);
}
