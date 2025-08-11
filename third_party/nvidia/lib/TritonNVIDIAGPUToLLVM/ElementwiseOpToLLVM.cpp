#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir::triton::gpu;

namespace mlir::triton {

namespace gpu {
namespace {

/* ----- FP8E5M2 ------ */
// This data-type is the standard FP8E5M2 format

struct Fp8ConversionDesc {
  std::string ptx;
  int inVecWidthBits;
  int outVecWidthBits;
  size_t numElements;
};

static const Fp8ConversionDesc Fp16_to_Fp8E5M2_RTNE(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    // TODO: nan may become +-inf or +-0
    ret = {"{                            \n"
           ".reg .b32 a<2>, b<2>;        \n"
           "and.b32 b0, $1, 0x01000100;  \n" // RTNE:
           "and.b32 b1, $2, 0x01000100;  \n" // if LSB of fp8 mantissa is 1
           "add.u32 a0, $1, 0x007f007f;  \n" // then add 0x80 to fp16 mantissa
           "add.u32 a1, $2, 0x007f007f;  \n" // else add 0x7f to fp16 mantissa
           "shr.b32 b0, b0, 8;           \n"
           "shr.b32 b1, b1, 8;           \n"
           "add.u32 a0, a0, b0;          \n"
           "add.u32 a1, a1, b1;          \n"
           "prmt.b32 $0, a0, a1, 0x7531; \n" // output = a1a0
           "}",
           32, 32, 4};
  } else {
    ret = {"cvt.rn.satfinite.e5m2x2.f16x2 $0, $1;", 32, 16, 2};
  }
  return ret;
}

const Fp8ConversionDesc Fp16_to_Fp8E5M2_RTZ = {"prmt.b32 $0, $1, $2, 0x7531;",
                                               32, 32, 4};

static const Fp8ConversionDesc Fp8E5M2_to_Fp16(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    ret = {"{                           \n"
           "prmt.b32 $0, 0, $2, 0x5140; \n"
           "prmt.b32 $1, 0, $2, 0x7362; \n"
           "}",
           32, 32, 4};
  } else {
    ret = {"cvt.rn.f16x2.e5m2x2 $0, $1;", 16, 32, 2};
  }
  return ret;
}

static const Fp8ConversionDesc Fp8E5M2_to_Bf16(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    // TODO: +-inf and nan may become +-large finite numbers
    ret = {
        "{                                        \n"
        ".reg .b32 a<2>, b<2>, c<4>, e112;        \n" // if input = 0xf1f2f3f4
        "mov.b32 e112, 0x77800000;                \n"
        "prmt.b32 a0, 0, $2, 0x5140;              \n" // a0 = 0xf300f400
        "prmt.b32 a1, 0, $2, 0x7362;              \n" // a1 = 0xf100f200
        "and.b32 b0, a0, 0x7fff7fff;              \n" // strip sign
        "and.b32 b1, a1, 0x7fff7fff;              \n"
        "shr.b32 b0, b0, 3;                       \n" // shift to bf16 position
        "shr.b32 b1, b1, 3;                       \n"
        "and.b32 c0, b0, 0xffff0000;              \n" // c0 = f3
        "shl.b32 c1, b0, 16;                      \n" // c1 = f4
        "and.b32 c2, b1, 0xffff0000;              \n" // c2 = f1
        "shl.b32 c3, b1, 16;                      \n" // c3 = f2
        "mul.f32 c0, c0, e112;                    \n" // move exponent bias
        "mul.f32 c1, c1, e112;                    \n" // from 15 to 127
        "mul.f32 c2, c2, e112;                    \n" // and handle denormal
        "mul.f32 c3, c3, e112;                    \n"
        "prmt.b32 b0, c0, c1, 0x3276;             \n" // b0 = 0xc0c1
        "prmt.b32 b1, c2, c3, 0x3276;             \n" // b1 = 0xc2c3
        "lop3.b32 $0, b0, 0x80008000, a0, 0xf8;   \n" // out0=b0|(0x80008000&a0)
        "lop3.b32 $1, b1, 0x80008000, a1, 0xf8;   \n" // (restore sign)
        "}",
        32, 32, 4};
  } else {
    ret = {
        "{                                      \n"
        ".reg .b32 a<2>, b<2>;                  \n" // if input = 0xf1f2f3f4
        ".reg .b32 e112;                        \n" // 2**112 represented as
        "mov.u32 e112, 0x77807780;              \n" // bf16x2
        "prmt.b32 a0, 0, $2, 0x5140;            \n" // a0 = 0xf300f400
        "prmt.b32 a1, 0, $2, 0x7362;            \n" // a1 = 0xf100f200
        "lop3.b32 b0, a0, 0x7fff7fff, 0, 0xc0;  \n" // b0 = a0 & 0x7fff7fff
        "lop3.b32 b1, a1, 0x7fff7fff, 0, 0xc0;  \n" // (strip sign)
        "shr.b32 b0, b0, 3;                     \n" // b0 >>= 3
        "shr.b32 b1, b1, 3;                     \n" // shift into bf16 position
        "lop3.b32 b0, b0, 0x80008000, a0, 0xf8; \n" // out0 = b0|(0x80008000&a0)
        "lop3.b32 b1, b1, 0x80008000, a1, 0xf8; \n" // (restore sign)
        "mul.rn.bf16x2 $0, b0, e112;            \n" // b0.exp += 127 - 15
        "mul.rn.bf16x2 $1, b1, e112;            \n" // exponent compensate = 112
        "}",
        32, 32, 4};
  }
  return ret;
}

static const Fp8ConversionDesc Bf16_to_Fp8E5M2(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    // TODO: Large number may become nan when it should become +-inf
    ret = {
        "{                                           \n"
        ".reg .b32 b<2>;                             \n"
        "and.b32 b0, $1, 0x7fff7fff;                 \n" // strip sign
        "and.b32 b1, $2, 0x7fff7fff;                 \n"

        ".reg .b32 c<4>;                             \n" // b0 = 0xf333f444
        "and.b32 c0, b0, 0xffff0000;                 \n" // c0 = 0xf3330000
        "shl.b32 c1, b0, 16;                         \n" // c1 = 0xf4440000
        "and.b32 c2, b1, 0xffff0000;                 \n" // c2 = 0xf1110000
        "shl.b32 c3, b1, 16;                         \n" // c3 = 0xf2220000

        ".reg .b32 e112;                             \n" // move exponent bias
        "mov.b32 e112, 0x07800000;                   \n" // from 127 to 15
        "mul.f32 c0, c0, e112;                       \n" // and handle denormal
        "mul.f32 c1, c1, e112;                       \n"
        "mul.f32 c2, c2, e112;                       \n"
        "mul.f32 c3, c3, e112;                       \n"

        "min.u32 c0, c0, 0x0fefffff;                 \n" // avoid overflow
        "min.u32 c1, c1, 0x0fefffff;                 \n" // when RTNE
        "min.u32 c2, c2, 0x0fefffff;                 \n"
        "min.u32 c3, c3, 0x0fefffff;                 \n"

        ".reg .b32 lsb<4>;                           \n" // RTNE:
        "and.b32 lsb0, c0, 0x00200000;               \n" // if LSB is 1
        "and.b32 lsb1, c1, 0x00200000;               \n" // then add 0x00100000
        "and.b32 lsb2, c2, 0x00200000;               \n" // else add 0x000fffff
        "and.b32 lsb3, c3, 0x00200000;               \n"
        "shr.b32 lsb0, lsb0, 21;                     \n"
        "shr.b32 lsb1, lsb1, 21;                     \n"
        "shr.b32 lsb2, lsb2, 21;                     \n"
        "shr.b32 lsb3, lsb3, 21;                     \n"
        "add.u32 c0, c0, 0x000fffff;                 \n"
        "add.u32 c1, c1, 0x000fffff;                 \n"
        "add.u32 c2, c2, 0x000fffff;                 \n"
        "add.u32 c3, c3, 0x000fffff;                 \n"
        "add.u32 c0, c0, lsb0;                       \n"
        "add.u32 c1, c1, lsb1;                       \n"
        "add.u32 c2, c2, lsb2;                       \n"
        "add.u32 c3, c3, lsb3;                       \n"

        "prmt.b32 b0, c0, c1, 0x3276;                \n" // c0 = 0xf3330000
        "prmt.b32 b1, c2, c3, 0x3276;                \n" // c1 = 0xf4440000
                                                         // b0 = 0xf333f444

        "shl.b32 b0, b0, 3;                          \n" // shift to fp8e5
        "shl.b32 b1, b1, 3;                          \n"
        "lop3.b32 b0, b0, 0x80008000, $1, 0xf8;      \n" // b0=b0|(0x80008000&in0)
        "lop3.b32 b1, b1, 0x80008000, $2, 0xf8;      \n" // (restore sign)
        "prmt.b32 $0, b0, b1, 0x7531;                \n" // b0 = 0xf300f400
                                                         // b1 = 0xf100f200
                                                         // output = 0xf1f2f3f4
        "}",
        32, 32, 4};
  } else {
    ret = {"{                                       \n"
           ".reg .b16 a<2>;                         \n"
           ".reg .f32 b<2>;                         \n"
           "mov.b32 {a0, a1}, $1;                   \n"
           "cvt.f32.bf16 b0, a0;                    \n"
           "cvt.f32.bf16 b1, a1;                    \n"
           "cvt.rn.satfinite.e5m2x2.f32 $0, b1, b0; \n"
           "}",
           32, 16, 2};
  }
  return ret;
}

/* ----- FP8E4M3 ------ */

static const Fp8ConversionDesc Fp8E4M3Nv_to_Fp16(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    // Fp8E4M3 (x4) -> Fp16 (x4) (packed)
    // TODO: nan may become +-480
    ret = {
        "{                                        \n"
        ".reg .b32 a<2>, b<2>, c<4>, e8;          \n" // if input = 0xf1f2f3f4
        "mov.b32 e8, 0x43800000;                  \n"
        "prmt.b32 a0, 0, $2, 0x5140;              \n" // a0 = 0xf300f400
        "prmt.b32 a1, 0, $2, 0x7362;              \n" // a1 = 0xf100f200
        "and.b32 b0, a0, 0x7fff7fff;              \n" // strip sign
        "and.b32 b1, a1, 0x7fff7fff;              \n"
        "shr.b32 b0, b0, 4;                       \n" // shift to bf16 position
        "shr.b32 b1, b1, 4;                       \n"
        "and.b32 c0, b0, 0xffff0000;              \n" // c0 = f3
        "shl.b32 c1, b0, 16;                      \n" // c1 = f4
        "and.b32 c2, b1, 0xffff0000;              \n" // c2 = f1
        "shl.b32 c3, b1, 16;                      \n" // c3 = f2
        "mul.f32 c0, c0, e8;                      \n" // move exponent bias
        "mul.f32 c1, c1, e8;                      \n" // from 7 to 15
        "mul.f32 c2, c2, e8;                      \n" // and handle denormal
        "mul.f32 c3, c3, e8;                      \n"
        "prmt.b32 b0, c0, c1, 0x3276;             \n" // b0 = 0xc0c1
        "prmt.b32 b1, c2, c3, 0x3276;             \n" // b1 = 0xc2c3
        "shl.b32 b0, b0, 3;                       \n" // shift to fp16 position
        "shl.b32 b1, b1, 3;                       \n"
        "lop3.b32 $0, b0, 0x80008000, a0, 0xf8;   \n" // out0=b0|(0x80008000&a0)
        "lop3.b32 $1, b1, 0x80008000, a1, 0xf8;   \n" // (restore sign)
        "}",
        32, 32, 4};
  } else {
    // Fp8E4M3 (x2) -> Fp16 (x2) (packed)
    ret = {"cvt.rn.f16x2.e4m3x2 $0, $1;", 16, 32, 2};
  }
  return ret;
}

static const Fp8ConversionDesc Fp16_to_Fp8E4M3Nv(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    // Fp16 (x4) -> Fp8E4M3 (x4) (packed)
    ret = {
        "{                                           \n"
        ".reg .b32 b<2>;                             \n"
        "and.b32 b0, $1, 0x7fff7fff;                 \n" // strip sign
        "and.b32 b1, $2, 0x7fff7fff;                 \n"

        ".reg .b32 c<4>;                             \n" // b0 = 0xf333f444
        "and.b32 c0, b0, 0xffff0000;                 \n" // c0 = 0xf3330000
        "shl.b32 c1, b0, 16;                         \n" // c1 = 0xf4440000
        "and.b32 c2, b1, 0xffff0000;                 \n" // c2 = 0xf1110000
        "shl.b32 c3, b1, 16;                         \n" // c3 = 0xf2220000

        "shr.b32 c0, c0, 3;                          \n" // shift to fp32
        "shr.b32 c1, c1, 3;                          \n"
        "shr.b32 c2, c2, 3;                          \n"
        "shr.b32 c3, c3, 3;                          \n"

        ".reg .b32 e8;                               \n" // move exponent bias
        "mov.b32 e8, 0x3b800000;                     \n" // from 15 to 7
        "mul.f32 c0, c0, e8;                         \n" // and handle denormal
        "mul.f32 c1, c1, e8;                         \n"
        "mul.f32 c2, c2, e8;                         \n"
        "mul.f32 c3, c3, e8;                         \n"

        "min.u32 c0, c0, 0x07f7ffff;                 \n" // avoid overflow
        "min.u32 c1, c1, 0x07f7ffff;                 \n" // when RTNE
        "min.u32 c2, c2, 0x07f7ffff;                 \n"
        "min.u32 c3, c3, 0x07f7ffff;                 \n"

        ".reg .b32 lsb<4>;                           \n" // RTNE:
        "and.b32 lsb0, c0, 0x00100000;               \n" // if LSB is 1
        "and.b32 lsb1, c1, 0x00100000;               \n" // then add 0x00080000
        "and.b32 lsb2, c2, 0x00100000;               \n" // else add 0x0007ffff
        "and.b32 lsb3, c3, 0x00100000;               \n"
        "shr.b32 lsb0, lsb0, 20;                     \n"
        "shr.b32 lsb1, lsb1, 20;                     \n"
        "shr.b32 lsb2, lsb2, 20;                     \n"
        "shr.b32 lsb3, lsb3, 20;                     \n"
        "add.u32 c0, c0, 0x0007ffff;                 \n"
        "add.u32 c1, c1, 0x0007ffff;                 \n"
        "add.u32 c2, c2, 0x0007ffff;                 \n"
        "add.u32 c3, c3, 0x0007ffff;                 \n"
        "add.u32 c0, c0, lsb0;                       \n"
        "add.u32 c1, c1, lsb1;                       \n"
        "add.u32 c2, c2, lsb2;                       \n"
        "add.u32 c3, c3, lsb3;                       \n"

        "prmt.b32 b0, c0, c1, 0x3276;                \n" // c0 = 0xf3330000
        "prmt.b32 b1, c2, c3, 0x3276;                \n" // c1 = 0xf4440000
                                                         // b0 = 0xf333f444

        "shl.b32 b0, b0, 4;                          \n" // shift to fp8e4
        "shl.b32 b1, b1, 4;                          \n"
        "lop3.b32 b0, b0, 0x80008000, $1, 0xf8;      \n" // b0=b0|(0x80008000&in0)
        "lop3.b32 b1, b1, 0x80008000, $2, 0xf8;      \n" // (restore sign)
        "prmt.b32 $0, b0, b1, 0x7531;                \n" // b0 = 0xf300f400
                                                         // b1 = 0xf100f200
                                                         // output = 0xf1f2f3f4
        "}",
        32, 32, 4};
  } else {
    // Fp16 (x2) -> Fp8E4M3 (x2) (packed)
    ret = {"cvt.rn.satfinite.e4m3x2.f16x2 $0, $1;", 32, 16, 2};
  }
  return ret;
}

static const Fp8ConversionDesc Fp8E4M3Nv_to_Bf16(bool hasNativeFP8,
                                                 bool hasNativeBF16F16) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP8) {
    // Fp8E4M3 (x4) -> Bf16 (x4) (packed)
    // TODO: nan may become +-480
    ret = {
        "{                                        \n"
        ".reg .b32 a<2>, b<2>, c<4>, e120;        \n" // if input = 0xf1f2f3f4
        "mov.b32 e120, 0x7b800000;                \n"
        "prmt.b32 a0, 0, $2, 0x5140;              \n" // a0 = 0xf300f400
        "prmt.b32 a1, 0, $2, 0x7362;              \n" // a1 = 0xf100f200
        "and.b32 b0, a0, 0x7fff7fff;              \n" // strip sign
        "and.b32 b1, a1, 0x7fff7fff;              \n"
        "shr.b32 b0, b0, 4;                       \n" // shift to bf16 position
        "shr.b32 b1, b1, 4;                       \n"
        "and.b32 c0, b0, 0xffff0000;              \n" // c0 = f3
        "shl.b32 c1, b0, 16;                      \n" // c1 = f4
        "and.b32 c2, b1, 0xffff0000;              \n" // c2 = f1
        "shl.b32 c3, b1, 16;                      \n" // c3 = f2
        "mul.f32 c0, c0, e120;                    \n" // move exponent bias
        "mul.f32 c1, c1, e120;                    \n" // from 7 to 127
        "mul.f32 c2, c2, e120;                    \n" // and handle denormal
        "mul.f32 c3, c3, e120;                    \n"
        "prmt.b32 b0, c0, c1, 0x3276;             \n" // b0 = 0xc0c1
        "prmt.b32 b1, c2, c3, 0x3276;             \n" // b1 = 0xc2c3
        "lop3.b32 $0, b0, 0x80008000, a0, 0xf8;   \n" // out0=b0|(0x80008000&a0)
        "lop3.b32 $1, b1, 0x80008000, a1, 0xf8;   \n" // (restore sign)
        "}",
        32, 32, 4};
  } else if (!hasNativeBF16F16) {
    // Fp8E4M3 (x2) -> Bf16 (x2) (packed)
    ret = {"{                                       \n"
           ".reg .b32 a;                            \n"
           ".reg .f16 a<2>;                         \n"
           ".reg .f32 b<2>;                         \n"
           ".reg .b16 c<2>;                         \n"
           "cvt.rn.f16x2.e4m3x2 a, $1;              \n"
           "mov.b32 {a0, a1}, a;                    \n"
           "cvt.f32.f16 b0, a0;                     \n"
           "cvt.f32.f16 b1, a1;                     \n"
           "cvt.rn.bf16.f32 c0, b0;                 \n"
           "cvt.rn.bf16.f32 c1, b1;                 \n"
           "mov.b32 $0, {c0, c1};                   \n"
           "}",
           16, 32, 2};
  } else {
    ret = {"{                                       \n"
           ".reg .b32 a;                            \n"
           ".reg .f16 a<2>;                         \n"
           ".reg .b16 b<2>;                         \n"
           "cvt.rn.f16x2.e4m3x2 a, $1;              \n"
           "mov.b32 {a0, a1}, a;                    \n"
           "cvt.bf16.f16 b0, a0;                    \n"
           "cvt.bf16.f16 b1, a1;                    \n"
           "mov.b32 $0, {b0, b1};                   \n"
           "}",
           16, 32, 2};
  }
  return ret;
}

static const Fp8ConversionDesc Bf16_to_Fp8E4M3Nv(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    // Bf16 (x4) -> Fp8E4M3 (x4) (packed)
    ret = {
        "{                                           \n"
        ".reg .b32 b<2>;                             \n"
        "and.b32 b0, $1, 0x7fff7fff;                 \n" // strip sign
        "and.b32 b1, $2, 0x7fff7fff;                 \n"

        ".reg .b32 c<4>;                             \n" // b0 = 0xf333f444
        "and.b32 c0, b0, 0xffff0000;                 \n" // c0 = 0xf3330000
        "shl.b32 c1, b0, 16;                         \n" // c1 = 0xf4440000
        "and.b32 c2, b1, 0xffff0000;                 \n" // c2 = 0xf1110000
        "shl.b32 c3, b1, 16;                         \n" // c3 = 0xf2220000

        ".reg .b32 e120;                             \n" // move exponent bias
        "mov.b32 e120, 0x03800000;                   \n" // from 127 to 7
        "mul.f32 c0, c0, e120;                       \n" // and handle denormal
        "mul.f32 c1, c1, e120;                       \n"
        "mul.f32 c2, c2, e120;                       \n"
        "mul.f32 c3, c3, e120;                       \n"

        "min.u32 c0, c0, 0x07f7ffff;                 \n" // avoid overflow
        "min.u32 c1, c1, 0x07f7ffff;                 \n" // when RTNE
        "min.u32 c2, c2, 0x07f7ffff;                 \n"
        "min.u32 c3, c3, 0x07f7ffff;                 \n"

        ".reg .b32 lsb<4>;                           \n" // RTNE:
        "and.b32 lsb0, c0, 0x00100000;               \n" // if LSB is 1
        "and.b32 lsb1, c1, 0x00100000;               \n" // then add 0x00080000
        "and.b32 lsb2, c2, 0x00100000;               \n" // else add 0x0007ffff
        "and.b32 lsb3, c3, 0x00100000;               \n"
        "shr.b32 lsb0, lsb0, 20;                     \n"
        "shr.b32 lsb1, lsb1, 20;                     \n"
        "shr.b32 lsb2, lsb2, 20;                     \n"
        "shr.b32 lsb3, lsb3, 20;                     \n"
        "add.u32 c0, c0, 0x0007ffff;                 \n"
        "add.u32 c1, c1, 0x0007ffff;                 \n"
        "add.u32 c2, c2, 0x0007ffff;                 \n"
        "add.u32 c3, c3, 0x0007ffff;                 \n"
        "add.u32 c0, c0, lsb0;                       \n"
        "add.u32 c1, c1, lsb1;                       \n"
        "add.u32 c2, c2, lsb2;                       \n"
        "add.u32 c3, c3, lsb3;                       \n"

        "prmt.b32 b0, c0, c1, 0x3276;                \n" // c0 = 0xf3330000
        "prmt.b32 b1, c2, c3, 0x3276;                \n" // c1 = 0xf4440000
                                                         // b0 = 0xf333f444

        "shl.b32 b0, b0, 4;                          \n" // shift to fp8e4
        "shl.b32 b1, b1, 4;                          \n"
        "lop3.b32 b0, b0, 0x80008000, $1, 0xf8;      \n" // b0=b0|(0x80008000&in0)
        "lop3.b32 b1, b1, 0x80008000, $2, 0xf8;      \n" // (restore sign)
        "prmt.b32 $0, b0, b1, 0x7531;                \n" // b0 = 0xf300f400
                                                         // b1 = 0xf100f200
                                                         // output = 0xf1f2f3f4
        "}",
        32, 32, 4};
  } else {
    // Bf16 (x2) -> Fp8E4M3 (x2) (packed)
    ret = {"{                                       \n"
           ".reg .b16 a<2>;                         \n"
           ".reg .f32 b<2>;                         \n"
           "mov.b32 {a0, a1}, $1;                   \n"
           "cvt.f32.bf16 b0, a0;                    \n"
           "cvt.f32.bf16 b1, a1;                    \n"
           "cvt.rn.satfinite.e4m3x2.f32 $0, b1, b0; \n"
           "}",
           32, 16, 2};
  }
  return ret;
}

static const Fp8ConversionDesc Fp32_to_Fp8E4M3Nv(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    // Fp32 (x4) -> Fp8E4M3 (x4) (packed)
    ret = {
        "{                                           \n"
        ".reg .b32 c<4>, d<4>;                       \n"
        ".reg .pred p<4>;                            \n"
        "and.b32 c0, $1, 0x7fffffff;                 \n" // strip sign
        "and.b32 c1, $2, 0x7fffffff;                 \n"
        "and.b32 c2, $3, 0x7fffffff;                 \n"
        "and.b32 c3, $4, 0x7fffffff;                 \n"

        ".reg .b32 e141;                             \n"
        "mov.b32 e141, 0x46800000;                   \n"
        "setp.lt.u32 p0, c0, 0x3c800000;             \n" // handle fp8 denormal
        "setp.lt.u32 p1, c1, 0x3c800000;             \n"
        "setp.lt.u32 p2, c2, 0x3c800000;             \n"
        "setp.lt.u32 p3, c3, 0x3c800000;             \n"
        "add.f32 d0, c0, e141;                       \n"
        "add.f32 d1, c1, e141;                       \n"
        "add.f32 d2, c2, e141;                       \n"
        "add.f32 d3, c3, e141;                       \n"
        "sub.u32 d0, d0, e141;                       \n"
        "sub.u32 d1, d1, e141;                       \n"
        "sub.u32 d2, d2, e141;                       \n"
        "sub.u32 d3, d3, e141;                       \n"
        "shl.b32 d0, d0, 24;                         \n" // shift to highest
        "shl.b32 d1, d1, 24;                         \n" // 8 bits
        "shl.b32 d2, d2, 24;                         \n"
        "shl.b32 d3, d3, 24;                         \n"

        "min.u32 c0, c0, 0x43f7ffff;                 \n" // not fp8 denormal
        "min.u32 c1, c1, 0x43f7ffff;                 \n" // avoid overflow
        "min.u32 c2, c2, 0x43f7ffff;                 \n" // when RTNE
        "min.u32 c3, c3, 0x43f7ffff;                 \n"

        ".reg .b32 lsb<4>;                           \n" // RTNE:
        "and.b32 lsb0, c0, 0x00100000;               \n" // if LSB is 1
        "and.b32 lsb1, c1, 0x00100000;               \n" // then add 0x00080000
        "and.b32 lsb2, c2, 0x00100000;               \n" // else add 0x0007ffff
        "and.b32 lsb3, c3, 0x00100000;               \n"
        "shr.b32 lsb0, lsb0, 20;                     \n"
        "shr.b32 lsb1, lsb1, 20;                     \n"
        "shr.b32 lsb2, lsb2, 20;                     \n"
        "shr.b32 lsb3, lsb3, 20;                     \n"
        "add.u32 c0, c0, 0xc407ffff;                 \n" // move exponent bias
        "add.u32 c1, c1, 0xc407ffff;                 \n" // from 127 to 7
        "add.u32 c2, c2, 0xc407ffff;                 \n"
        "add.u32 c3, c3, 0xc407ffff;                 \n"
        "add.u32 c0, c0, lsb0;                       \n"
        "add.u32 c1, c1, lsb1;                       \n"
        "add.u32 c2, c2, lsb1;                       \n"
        "add.u32 c3, c3, lsb1;                       \n"

        "shl.b32 c0, c0, 4;                          \n" // shift to fp8e4
        "shl.b32 c1, c1, 4;                          \n"
        "shl.b32 c2, c2, 4;                          \n"
        "shl.b32 c3, c3, 4;                          \n"

        "selp.b32 c0, d0, c0, p0;                    \n" // use result for
        "selp.b32 c1, d1, c1, p1;                    \n" // fp8 denormal
        "selp.b32 c2, d2, c2, p2;                    \n"
        "selp.b32 c3, d3, c3, p3;                    \n"

        "lop3.b32 c0, c0, 0x80008000, $1, 0xf8;      \n" // c0=c0|(0x80008000&in0)
        "lop3.b32 c1, c1, 0x80008000, $2, 0xf8;      \n" // (restore sign)
        "lop3.b32 c2, c2, 0x80008000, $3, 0xf8;      \n"
        "lop3.b32 c3, c3, 0x80008000, $4, 0xf8;      \n"
        "prmt.b32 c0, c0, c1, 0x7430;                \n" // c0 = 0xf300f400
        "prmt.b32 c2, c2, c3, 0x7430;                \n" // c2 = 0xf100f200
        "prmt.b32 $0, c0, c2, 0x7531;                \n" // output = 0xf1f2f3f4
        "}",
        32, 32, 4};
  } else {
    // Fp32 (x2) -> Fp8E4M3 (x2) (packed)
    ret = {"cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;", 32, 16, 2};
  }
  return ret;
}

static const Fp8ConversionDesc Fp32_to_Fp8E5M2(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    // Fp32 (x4) -> Fp8E5M2 (x4) (packed)
    // TODO: Large number may become nan when it should become +-inf
    ret = {
        "{                                           \n"
        ".reg .b32 c<4>, d<4>;                       \n"
        ".reg .pred p<4>;                            \n"
        "and.b32 c0, $1, 0x7fffffff;                 \n" // strip sign
        "and.b32 c1, $2, 0x7fffffff;                 \n"
        "and.b32 c2, $3, 0x7fffffff;                 \n"
        "and.b32 c3, $4, 0x7fffffff;                 \n"

        ".reg .b32 e134;                             \n"
        "mov.b32 e134, 0x43000000;                   \n"
        "setp.lt.u32 p0, c0, 0x38800000;             \n" // handle fp8 denormal
        "setp.lt.u32 p1, c1, 0x38800000;             \n"
        "setp.lt.u32 p2, c2, 0x38800000;             \n"
        "setp.lt.u32 p3, c3, 0x38800000;             \n"
        "add.f32 d0, c0, e134;                       \n"
        "add.f32 d1, c1, e134;                       \n"
        "add.f32 d2, c2, e134;                       \n"
        "add.f32 d3, c3, e134;                       \n"
        "sub.u32 d0, d0, e134;                       \n"
        "sub.u32 d1, d1, e134;                       \n"
        "sub.u32 d2, d2, e134;                       \n"
        "sub.u32 d3, d3, e134;                       \n"
        "shl.b32 d0, d0, 24;                         \n" // shift to highest
        "shl.b32 d1, d1, 24;                         \n" // 8 bits
        "shl.b32 d2, d2, 24;                         \n"
        "shl.b32 d3, d3, 24;                         \n"

        "min.u32 c0, c0, 0x47efffff;                 \n" // not fp8 denormal
        "min.u32 c1, c1, 0x47efffff;                 \n" // avoid overflow
        "min.u32 c2, c2, 0x47efffff;                 \n" // when RTNE
        "min.u32 c3, c3, 0x47efffff;                 \n"

        ".reg .b32 lsb<4>;                           \n" // RTNE:
        "and.b32 lsb0, c0, 0x00200000;               \n" // if LSB is 1
        "and.b32 lsb1, c1, 0x00200000;               \n" // then add 0x00100000
        "and.b32 lsb2, c2, 0x00200000;               \n" // else add 0x000fffff
        "and.b32 lsb3, c3, 0x00200000;               \n"
        "shr.b32 lsb0, lsb0, 21;                     \n"
        "shr.b32 lsb1, lsb1, 21;                     \n"
        "shr.b32 lsb2, lsb2, 21;                     \n"
        "shr.b32 lsb3, lsb3, 21;                     \n"
        "add.u32 c0, c0, 0xc80fffff;                 \n" // move exponent bias
        "add.u32 c1, c1, 0xc80fffff;                 \n" // from 127 to 15
        "add.u32 c2, c2, 0xc80fffff;                 \n"
        "add.u32 c3, c3, 0xc80fffff;                 \n"
        "add.u32 c0, c0, lsb0;                       \n"
        "add.u32 c1, c1, lsb1;                       \n"
        "add.u32 c2, c2, lsb1;                       \n"
        "add.u32 c3, c3, lsb1;                       \n"

        "shl.b32 c0, c0, 3;                          \n" // shift to fp8e5
        "shl.b32 c1, c1, 3;                          \n"
        "shl.b32 c2, c2, 3;                          \n"
        "shl.b32 c3, c3, 3;                          \n"

        "selp.b32 c0, d0, c0, p0;                    \n" // use result for
        "selp.b32 c1, d1, c1, p1;                    \n" // fp8 denormal
        "selp.b32 c2, d2, c2, p2;                    \n"
        "selp.b32 c3, d3, c3, p3;                    \n"

        "lop3.b32 c0, c0, 0x80008000, $1, 0xf8;      \n" // c0=c0|(0x80008000&in0)
        "lop3.b32 c1, c1, 0x80008000, $2, 0xf8;      \n" // (restore sign)
        "lop3.b32 c2, c2, 0x80008000, $3, 0xf8;      \n"
        "lop3.b32 c3, c3, 0x80008000, $4, 0xf8;      \n"
        "prmt.b32 c0, c0, c1, 0x7430;                \n" // c0 = 0xf300f400
        "prmt.b32 c2, c2, c3, 0x7430;                \n" // c2 = 0xf100f200
        "prmt.b32 $0, c0, c2, 0x7531;                \n" // output = 0xf1f2f3f4
        "}",
        32, 32, 4};
  } else {
    // Fp32 (x2) -> Fp8E5M2 (x2) (packed)
    ret = {"cvt.rn.satfinite.e5m2x2.f32 $0, $2, $1;", 32, 16, 2};
  }
  return ret;
}

/* ----- Packed integer to BF16 ------ */
static const std::string S8_to_Bf16 =
    "{                                           \n"
    ".reg .s8 s<4>;                              \n"
    ".reg .f32 f<4>;                             \n"
    "mov.b32 {s0, s1, s2, s3}, $2;               \n" // unpack
    "cvt.rn.f32.s8 f0, s0;                       \n" // no s8->bf16 pre-Hopper
    "cvt.rn.f32.s8 f1, s1;                       \n" // fi[0:15] is always 0
    "cvt.rn.f32.s8 f2, s2;                       \n" //
    "cvt.rn.f32.s8 f3, s3;                       \n" //
    "prmt.b32 $0, f0, f1, 0x7632;                \n" // f32->bf16 + pack
    "prmt.b32 $1, f2, f3, 0x7632;                \n" //
    "}";
// Conversions have low throughput, rely on bit tricks instead of cvt
// instruction on Hopper and later GPUs.
static const std::string S8_to_Bf16_sm90 =
    "{                               \n"
    ".reg .b32 l<3>;                 \n"
    ".reg .b32 h<3>;                 \n"
    "prmt.b32 l0, $2, 0x43, 0x4140;  \n" // Unpack to shifted bf16.
    "prmt.b32 h0, $2, 0x43, 0x4342;  \n"
    "and.b32 l1, l0, 0xff7fff7f;     \n" // Zero the least exp bit.
    "and.b32 h1, h0, 0xff7fff7f;     \n"
    "and.b32 l2, l0, 0xff80ff80;     \n" // Zero the mantissa.
    "and.b32 h2, h0, 0xff80ff80;     \n"
    "sub.bf16x2 $0, l1, l2;          \n" // Subtract the offset.
    "sub.bf16x2 $1, h1, h2;          \n"
    "}";

typedef std::function<SmallVector<Value>(Location, ConversionPatternRewriter &,
                                         const SmallVector<Value> &)>
    ConverterT;

static ConverterT makeConverterFromPtx(const std::string &ptxAsm, Type inType,
                                       Type outType,
                                       const int inVecWidthBits = 32,
                                       const int outVecWidthBits = 32) {
  ConverterT converter =
      [ptxAsm, inType, outType, inVecWidthBits,
       outVecWidthBits](Location loc, ConversionPatternRewriter &rewriter,
                        const SmallVector<Value> &v) -> SmallVector<Value> {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int numElements = v.size();
    assert(numElements == 4 || numElements == 2 && "invalid vector size");

    auto ctx = rewriter.getContext();
    int inBitwidth = inType.getIntOrFloatBitWidth();
    int outBitwidth = outType.getIntOrFloatBitWidth();
    // first, we pack `v` into 32-bit ints
    int inVecWidth = inVecWidthBits / inBitwidth;
    auto inVecTy = vec_ty(inType, inVecWidth);
    SmallVector<Value> inPacked(numElements / inVecWidth, b.undef(inVecTy));
    for (size_t i = 0; i < numElements; i++)
      inPacked[i / inVecWidth] = b.insert_element(
          inVecTy, inPacked[i / inVecWidth], v[i], b.i32_val(i % inVecWidth));
    for (size_t i = 0; i < inPacked.size(); i++)
      inPacked[i] = b.bitcast(inPacked[i], int_ty(inVecWidthBits));

    // then, we run the provided inline PTX
    int outVecWidth = outVecWidthBits / outBitwidth;
    int outNums = numElements / outVecWidth;
    PTXBuilder builder;
    SmallVector<PTXBuilder::Operand *> operands;
    auto outConstraint = outVecWidthBits == 16 ? "=h" : "=r";
    auto inConstraint = inVecWidthBits == 16 ? "h" : "r";
    for (int i = 0; i < outNums; i++) {
      operands.push_back(builder.newOperand(outConstraint));
    }

    for (Value inVal : inPacked) {
      operands.push_back(builder.newOperand(inVal, inConstraint));
    }

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
        outPacked.push_back(b.extract_val(outVecTy, outStruct, i));
    }
    // unpack the output
    SmallVector<Value> ret;
    for (size_t i = 0; i < numElements; i++)
      ret.push_back(b.extract_element(outType, outPacked[i / outVecWidth],
                                      b.i32_val(i % outVecWidth)));
    return ret;
  };
  return converter;
}

// Attempts to use vectorized conversions via inline PTX when possible.
struct FpToFpOpConversion
    : public ElementwiseOpConversionBase<FpToFpOp, FpToFpOpConversion> {
  using ElementwiseOpConversionBase<
      FpToFpOp, FpToFpOpConversion>::ElementwiseOpConversionBase;

  explicit FpToFpOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              int computeCapability,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        computeCapability(computeCapability) {}

  static Value convertFp16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    return rewriter.create<LLVM::FPExtOp>(loc, f32_ty, v);
  }

  static Value convertFp32ToBf16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v, const RoundingMode rounding) {
    StringRef name;
    switch (rounding) {
    case RoundingMode::RTNE:
      name = "llvm.nvvm.f2bf16.rn";
      break;
    case RoundingMode::RTZ:
      name = "llvm.nvvm.f2bf16.rz";
      break;
    default:
      emitError(loc) << "unsupported rounding mode for f32->bf16 conversion: "
                     << stringifyRoundingMode(rounding) << "\n";
      llvm::report_fatal_error(
          "unsupported rounding mode for f32->bf16 conversion: " +
          stringifyRoundingMode(rounding) + "\n");
    }
    return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, name, bf16_ty, {v})
        .getResult(0);
  }

  static Value convertFp32ToFp16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v, const RoundingMode rounding) {
    PTXBuilder builder;
    StringRef ptx;
    switch (rounding) {
    case RoundingMode::RTNE:
      ptx = "cvt.rn.f16.f32";
      break;
    case RoundingMode::RTZ:
      ptx = "cvt.rz.f16.f32";
      break;
    default:
      emitError(loc) << "unsupported rounding mode for f32->f16 conversion: "
                     << stringifyRoundingMode(rounding) << "\n";
      llvm::report_fatal_error(
          "unsupported rounding mode for f32->f16 conversion: " +
          stringifyRoundingMode(rounding) + "\n");
    }
    auto &cvt = *builder.create(ptx.str());
    auto res = builder.newOperand("=h");
    auto operand = builder.newOperand(v, "r");
    cvt(res, operand);
    return builder.launch(rewriter, loc, f16_ty, false);
  }

  std::pair<ConverterT, size_t>
  getConversionFunc(Type srcTy, Type dstTy,
                    std::optional<RoundingMode> roundingMode) const {
    auto F8E4M3TyID = TypeID::get<Float8E4M3FNType>();
    auto F8E5M2TyID = TypeID::get<Float8E5M2Type>();
    auto F16TyID = TypeID::get<Float16Type>();
    auto BF16TyID = TypeID::get<BFloat16Type>();
    auto F32TyID = TypeID::get<Float32Type>();
    auto F64TyID = TypeID::get<Float64Type>();

    auto undefRounding = static_cast<RoundingMode>(-1);

    static DenseMap<std::tuple<TypeID, TypeID, RoundingMode>, Fp8ConversionDesc>
        srcMap = {
            // F8 -> F16
            {{F8E4M3TyID, F16TyID, undefRounding},
             Fp8E4M3Nv_to_Fp16(computeCapability >= 89)},
            {{F8E5M2TyID, F16TyID, undefRounding},
             Fp8E5M2_to_Fp16(computeCapability >= 89)},
            // F16 -> F8
            {{F16TyID, F8E4M3TyID, RoundingMode::RTNE},
             Fp16_to_Fp8E4M3Nv(computeCapability >= 89)},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTNE},
             Fp16_to_Fp8E5M2_RTNE(computeCapability >= 89)},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTZ}, Fp16_to_Fp8E5M2_RTZ},
            // F8 -> BF16
            // mul{.rnd}.bf16 and mul{.rnd}.bf16x2 requires sm_90 or higher.
            {{F8E5M2TyID, BF16TyID, undefRounding},
             Fp8E5M2_to_Bf16(computeCapability >= 90)},
            // cvt with .bf16.f16' requires .target sm_90 or higher
            {{F8E4M3TyID, BF16TyID, undefRounding},
             Fp8E4M3Nv_to_Bf16(computeCapability >= 89,
                               computeCapability >= 90)},
            // BF16 -> F8
            {{BF16TyID, F8E5M2TyID, RoundingMode::RTNE},
             Bf16_to_Fp8E5M2(computeCapability >= 89)},
            {{BF16TyID, F8E4M3TyID, RoundingMode::RTNE},
             Bf16_to_Fp8E4M3Nv(computeCapability >= 89)},
            // F32 -> F8
            {{F32TyID, F8E4M3TyID, RoundingMode::RTNE},
             Fp32_to_Fp8E4M3Nv(computeCapability >= 90)},
            {{F32TyID, F8E5M2TyID, RoundingMode::RTNE},
             Fp32_to_Fp8E5M2(computeCapability >= 90)},
        };
    std::tuple<TypeID, TypeID, RoundingMode> key = {
        srcTy.getTypeID(), dstTy.getTypeID(),
        roundingMode.value_or(undefRounding)};
    if (srcMap.count(key) == 0) {
      llvm::errs() << "Unsupported conversion from " << srcTy << " to "
                   << dstTy;
      if (roundingMode.has_value())
        llvm::errs() << " with rounding mode "
                     << stringifyRoundingMode(roundingMode.value());
      llvm::errs() << "\n";
      llvm::report_fatal_error("Unsupported rounding mode for conversion.");
    }
    auto convDesc = srcMap.lookup(key);
    return {makeConverterFromPtx(
                convDesc.ptx, getTypeConverter()->convertType(srcTy),
                getTypeConverter()->convertType(dstTy), convDesc.inVecWidthBits,
                convDesc.outVecWidthBits),
            convDesc.numElements};
  }

  SmallVector<Value> createDestOps(FpToFpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcElementType = getElementType(op.getSrc());
    auto dstElementType = getElementType(op.getResult());
    auto roundingMode = op.getRounding();

    if (llvm::isa<Float8E5M2Type, Float8E4M3FNType>(dstElementType)) {
      assert(roundingMode.has_value() &&
             "Rounding mode must be specified for convertsions to fp8");

      // For now only RTNE is supported for conversions from fp16 to fp8
      if (!srcElementType.isF32() &&
          roundingMode.value() != RoundingMode::RTNE) {
        llvm::report_fatal_error(
            "Unsupported rounding mode for conversion to fp8: " +
            stringifyRoundingMode(roundingMode.value()) + "\n");
      }
    }

    if (srcElementType.isF16() && dstElementType.isF32()) {
      return llvm::to_vector(llvm::map_range(operands[0], [&](Value v) {
        return convertFp16ToFp32(loc, rewriter, v);
      }));
    }

    if (srcElementType.isF32() && dstElementType.isF16()) {
      assert(roundingMode.has_value() &&
             "rounding mode must be specified for fp32->fp16 conversion");
      SmallVector<Value> outVals;
      for (Value v : operands[0]) {
        outVals.push_back(
            convertFp32ToFp16(loc, rewriter, v, roundingMode.value()));
      }
      return outVals;
    }

    if (srcElementType.isF32() && dstElementType.isBF16()) {
      assert(roundingMode.has_value() &&
             "rounding mode must be specified for fp32->bf16 conversion");
      SmallVector<Value> outVals;
      for (Value v : operands[0]) {
        outVals.push_back(
            convertFp32ToBf16(loc, rewriter, v, roundingMode.value()));
      }
      return outVals;
    }

    bool useFP16IntermediateSrc =
        srcElementType.isF32() && roundingMode.value() == RoundingMode::RTZ;
    bool isDstFP32 = dstElementType.isF32();
    Type srcType = useFP16IntermediateSrc ? f16_ty : srcElementType;
    Type dstType = isDstFP32 ? f16_ty : dstElementType;
    auto [cvtFunc, numElements] =
        getConversionFunc(srcType, dstType, roundingMode);
    SmallVector<Value> inVals;
    for (unsigned i = 0; i < std::min(numElements, operands.size()); i++) {
      inVals.push_back(operands[i][0]);
    }
    if (useFP16IntermediateSrc)
      for (Value &v : inVals)
        v = convertFp32ToFp16(loc, rewriter, v, RoundingMode::RTZ);
    inVals.resize(numElements, b.undef(typeConverter->convertType(srcType)));
    SmallVector<Value> outVals = cvtFunc(loc, rewriter, inVals);
    assert(outVals.size() == inVals.size());
    outVals.resize(std::min(numElements, operands.size()));
    if (isDstFP32)
      for (Value &v : outVals)
        v = convertFp16ToFp32(loc, rewriter, v);
    // Pack values
    return outVals;
  }

private:
  int computeCapability;
};

struct FDivOpConversion
    : ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::DivFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
    StringRef name;
    Type resultTy;
    if (32 == bitwidth) {
      name = "llvm.nvvm.div.full";
      resultTy = f32_ty;
    } else if (64 == bitwidth) {
      name = "llvm.nvvm.div.rn.d";
      resultTy = f64_ty;
    } else {
      llvm::report_fatal_error("Unsupported bitwidth");
    }
    Value args[] = {operands[0][0], operands[0][1]};
    auto callOp =
        LLVM::createLLVMIntrinsicCallOp(rewriter, loc, name, resultTy, args);
    return {callOp.getResult(0)};
  }
};

// Uses inline ptx to convert s8/u8 to bf16, since the
struct SIToFPOpConversion
    : ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion>;
  using Adaptor = typename Base::OpAdaptor;

  explicit SIToFPOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              int computeCapability,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        computeCapability(computeCapability) {}

  SmallVector<Value> createDestOps(arith::SIToFPOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type inElemTy = getElementType(op.getIn());
    Type outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16() && inElemTy.isInteger(8) && operands.size() >= 4) {
      auto cvtFunc = makeConverterFromPtx(
          computeCapability >= 90 ? S8_to_Bf16_sm90 : S8_to_Bf16,
          getTypeConverter()->convertType(inElemTy),
          getTypeConverter()->convertType(outElemTy));
      SmallVector<Value> inVals = {operands[0][0], operands[1][0],
                                   operands[2][0], operands[3][0]};
      auto outVals = cvtFunc(loc, rewriter, inVals);
      assert(outVals.size() == 4);
      return outVals;
    } else {
      return {rewriter.create<LLVM::SIToFPOp>(loc, elemTy, operands[0][0])};
    }
  }

private:
  int computeCapability;
};

struct FPToSIOpConversion
    : ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::FPToSIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    return {rewriter.create<LLVM::FPToSIOp>(loc, elemTy, operands[0][0])};
  }
};

struct ExpOpConversionApprox
    : ElementwiseOpConversionBase<math::ExpOp, ExpOpConversionApprox> {
  using Base = ElementwiseOpConversionBase<math::ExpOp, ExpOpConversionApprox>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(math::ExpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // For non-FP32 input, call __nv_expf for higher-precision calculation
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    const double log2e = 1.4426950408889634;
    Value prod = b.fmul(f32_ty, operands[0][0], b.f32_val(log2e));

    Type resultTy = operands[0][0].getType();
    StringRef name = "llvm.nvvm.ex2.approx.f";
    auto callOp =
        LLVM::createLLVMIntrinsicCallOp(rewriter, loc, name, resultTy, {prod});
    return {callOp.getResult(0)};
  }
};

struct ClampFOpConversion
    : ElementwiseOpConversionBase<ClampFOp, ClampFOpConversion> {
  using Base = ElementwiseOpConversionBase<ClampFOp, ClampFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  explicit ClampFOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              int computeCapability,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        computeCapability(computeCapability) {}

  bool isClipPattern(ClampFOp op) const {
    bool xorsignAbsAvailable = (computeCapability >= 90);
    // Pattern matching the sequence of clamp(x, -limit, limit) to generate
    // more efficient PTX code. NOTE: This pattern matching is not general
    // enough, but it is sufficient. We detect only two cases here:
    // 1. where the "-limit" is computed as 0 - limit:
    //   %cst = arith.constant dense<0.000000e+00>
    //   %8 = tt.load %7, %2
    //   %11 = arith.subf %cst, %8
    //   %12 = tt.clamp %5, %11, %8
    // 2. where "-limit" and "limit" are constants.
    //   %cst_6 = arith.constant dense<-6.0000e+00>
    //   %cst_7 = arith.constant dense<6.0000e+00>
    //   %160 = tt.clamp %158, %cst_6, %cst_7
    bool patternFound = false;

    auto getSplatInitializer = [](Value v) -> std::optional<double> {
      if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto attr = mlir::dyn_cast<DenseIntOrFPElementsAttr>(
                constOp.getValueAttr())) {
          if (attr.isSplat()) {
            return attr.getSplatValue<APFloat>().convertToDouble();
          }
        }
      }
      return std::nullopt;
    };

    if (xorsignAbsAvailable) {
      if (auto subOp = op.getOperand(1).getDefiningOp<arith::SubFOp>()) {
        if (subOp.getOperand(1) == op.getOperand(2)) {
          auto initializer = getSplatInitializer(subOp.getOperand(0));
          if (initializer.has_value() && initializer.value() == 0.0) {
            patternFound = true;
          }
        }
      } else {
        auto initializer1 = getSplatInitializer(op.getOperand(1));
        auto initializer2 = getSplatInitializer(op.getOperand(2));
        if (initializer1.has_value() && initializer2.has_value() &&
            initializer1.value() == -initializer2.value()) {
          patternFound = true;
        }
      }
    }
    return patternFound;
  }

  SmallVector<Value> emitOptimization(ClampFOp op,
                                      ConversionPatternRewriter &rewriter,
                                      Type elemTy,
                                      MultipleOperandsRange operands,
                                      Location loc) const {
    std::string name = "llvm.nvvm.fmin";
    if (op.getPropagateNan() == PropagateNan::ALL) {
      name += ".nan";
    }
    name += ".xorsign.abs";
    if (elemTy.isF32()) {
      name += ".f";
    } else if (elemTy.isF16()) {
      name += ".f16";
    }

    Type resultTy = operands[0][0].getType();
    Value args[] = {operands[0][0], operands[0][2]};
    auto callOp =
        LLVM::createLLVMIntrinsicCallOp(rewriter, loc, name, resultTy, args);
    return {callOp.getResult(0)};
  }

  SmallVector<Value> createDestOps(ClampFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (isClipPattern(op)) {
      return emitOptimization(op, rewriter, elemTy, operands, loc);
    }
    return {};
  }

private:
  int computeCapability;
};

template <typename TritonOp>
struct OpToExternCallConversion
    : public ElementwiseOpConversionBase<TritonOp,
                                         OpToExternCallConversion<TritonOp>> {
  using Base =
      ElementwiseOpConversionBase<TritonOp, OpToExternCallConversion<TritonOp>>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  explicit OpToExternCallConversion(LLVMTypeConverter &typeConverter,
                                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                                    StringRef externFuncName,
                                    PatternBenefit benefit)
      : Base::ElementwiseOpConversionBase(typeConverter, axisAnalysisPass,
                                          benefit),
        funcName(externFuncName) {}

  SmallVector<Value> createDestOps(TritonOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
    return {
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]).getResult()};
  }

private:
  StringRef funcName;
};
} // namespace
} // namespace gpu

} // namespace mlir::triton

void mlir::triton::NVIDIA::populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  using namespace mlir::triton::gpu;

  patterns.add<OpToExternCallConversion<triton::PreciseSqrtOp>>(
      typeConverter, axisInfoAnalysis, "__nv_fsqrt_rn", benefit);
  patterns.add<OpToExternCallConversion<triton::PreciseDivFOp>>(
      typeConverter, axisInfoAnalysis, "__nv_fdiv_rn", benefit);

  mlir::triton::populateElementwiseOpToLLVMPatterns(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);

#define POPULATE_OP(SRC_OP, DST_OP)                                            \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit)

  POPULATE_OP(arith::SubFOp, LLVM::FSubOp);
  POPULATE_OP(arith::AddFOp, LLVM::FAddOp);
  POPULATE_OP(arith::MulFOp, LLVM::FMulOp);

  POPULATE_OP(arith::ExtFOp, LLVM::FPExtOp);
  POPULATE_OP(arith::TruncFOp, LLVM::FPTruncOp);

#undef POPULATE_OP

  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FPToSIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<SIToFPOpConversion>(typeConverter, axisInfoAnalysis,
                                   computeCapability, benefit);
  patterns.add<FpToFpOpConversion>(typeConverter, axisInfoAnalysis,
                                   computeCapability, benefit);

  // ExpOpConversionApprox will try using ex2.approx if the input type is
  // FP32. For other input types, ExpOpConversionApprox will return failure and
  // ElementwiseOpConversion<math::ExpOp, math::ExpOp> defined below will call
  // __nv_expf for higher-precision calculation
  patterns.add<ExpOpConversionApprox>(typeConverter, axisInfoAnalysis, benefit);
  bool hwNanPropagationSupported = computeCapability >= 80;
  mlir::triton::populateMinMaxFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, hwNanPropagationSupported,
      benefit);
  mlir::triton::populateClampFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
}

void mlir::triton::NVIDIA::populateClampFOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    PatternBenefit benefit) {
  using namespace mlir::triton::gpu;

  patterns.add<ClampFOpConversion>(typeConverter, axisInfoAnalysis,
                                   computeCapability, benefit);
}
