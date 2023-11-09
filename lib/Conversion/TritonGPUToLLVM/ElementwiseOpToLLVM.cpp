#include "ElementwiseOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;
using ::mlir::triton::gpu::getTotalElemsPerThread;

/* ----- FP8E5M2 ------ */
// This data-type is the standard FP8E5M2 format
#ifdef USE_ROCM
static SmallVector<Value>
Fp16_to_Fp8E5M2(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec0 = undef(fp16x2VecTy);
  Value fp16x2Vec1 = undef(fp16x2VecTy);
  fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v[0], i32_val(0));
  fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v[1], i32_val(1));
  fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v[2], i32_val(0));
  fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v[3], i32_val(1));

  Value a0 = bitcast(fp16x2Vec0, i32_ty);
  Value a1 = bitcast(fp16x2Vec1, i32_ty);
  
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  a0 = bitcast(a0, fp8x4VecTy); 
  a1 = bitcast(a1, fp8x4VecTy); 

  return {extract_element(i8_ty, a0, i32_val(1)),
	  extract_element(i8_ty, a0, i32_val(3)),
	  extract_element(i8_ty, a1, i32_val(1)),
	  extract_element(i8_ty, a1, i32_val(3))
	  };
}
#else
static const std::string Fp16_to_Fp8E5M2(bool hasNativeFP) {
  std::string ret;
  if (!hasNativeFP) {
    ret = "{                            \n"
          ".reg .b32 a<2>;              \n"
          "and.b32 a0, $1, 0xfffefffe;  \n"   // a0 &= 0xfffefffe
          "and.b32 a1, $2, 0xfffefffe;  \n"   // (strip lowest bit)
          "add.u32 a0, a0, 0x00800080;  \n"   // a0 += 0x00800080
          "add.u32 a1, a1, 0x00800080;  \n"   // (round to nearest)
          "prmt.b32 $0, a0, a1, 0x7531; \n\t" // output = a1a0
          "}";
  } else {
    ret = "cvt.rn.satfinite.e5m2x2.f16x2 $0, $1; \n\t";
  }
  return ret;
}
#endif

#ifdef USE_ROCM
static Value convert_val_Fp16_to_Fp8E5M2FNUZ(
  Location loc, ConversionPatternRewriter &rewriter, Value v) {
  auto vi16 = bitcast(v, i16_ty);
  auto e = and_(i16_ty, vi16, int_val(16, 0x7C00));
  auto sign = and_(i16_ty, vi16, int_val(16, 0x8000));

  // normal value
  auto a = and_(i16_ty, vi16, int_val(16, 0x7FFFF));
  auto a1 = add(i16_ty, a, int_val(16, 0x0400));
  auto o1 = or_(i16_ty, a1, sign);

  // subnormal value, e is 0
  auto m = and_(i16_ty, vi16, int_val(16, 0x03FF));
  auto m2 = shl(m, int_val(16, 1));
  auto o2 = or_(i16_ty, sign, or_(i16_ty, int_val(16, 1), m2));

  auto e_is_zero = icmp_eq(e, int_val(16, 0));
  auto e_is_all1 = icmp_eq(e, int_val(16, 0x7C00));

  auto ot = select(e_is_zero, o2, o1);
  auto o = select(e_is_all1, vi16, ot);
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  auto res = bitcast(o, fp8x2VecTy); 

  return extract_element(i8_ty, res, i32_val(1));
}

static SmallVector<Value>
Fp16_to_Fp8E5M2FNUZ(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  SmallVector<Value> result(4);
  result[0] = convert_val_Fp16_to_Fp8E5M2FNUZ(loc, rewriter, v[0]);
  result[1] = convert_val_Fp16_to_Fp8E5M2FNUZ(loc, rewriter, v[1]);
  result[2] = convert_val_Fp16_to_Fp8E5M2FNUZ(loc, rewriter, v[2]);
  result[3] = convert_val_Fp16_to_Fp8E5M2FNUZ(loc, rewriter, v[3]);

  return result;
}
#else
const std::string Fp16_to_Fp8E5M2FNUZ =
    "{                            \n"
    ".reg .b32 a<2>;              \n"
    "and.b32 a0, $1, 0xfffefffe;  \n"   // a0 &= 0xfffefffe
    "and.b32 a1, $2, 0xfffefffe;  \n"   // (strip lowest bit)
    "add.u32 a0, a0, 0x00800080;  \n"   // a0 += 0x00800080
    "add.u32 a1, a1, 0x00800080;  \n"   // (round to nearest)
    "prmt.b32 $0, a0, a1, 0x7531; \n\t" // output = a1a0
    "}";
#endif

#ifdef USE_ROCM
static SmallVector<Value>
Fp8E5M2_to_Fp16(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = undef(fp8x4VecTy);
  a0 = insert_element(fp8x4VecTy, a0, int_val(8,0), i32_val(0));
  a0 = insert_element(fp8x4VecTy, a0, v[0], i32_val(1));
  a0 = insert_element(fp8x4VecTy, a0, int_val(8,0), i32_val(2));
  a0 = insert_element(fp8x4VecTy, a0, v[1], i32_val(3));
  a0 = bitcast(a0, i32_ty);
  Value a1 = undef(fp8x4VecTy);
  a1 = insert_element(fp8x4VecTy, a1, int_val(8,0), i32_val(0));
  a1 = insert_element(fp8x4VecTy, a1, v[2], i32_val(1));
  a1 = insert_element(fp8x4VecTy, a1, int_val(8,0), i32_val(2));
  a1 = insert_element(fp8x4VecTy, a1, v[3], i32_val(3));
  a1 = bitcast(a1, i32_ty);

  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  auto fp16x2Vec0 = bitcast(a0, fp16x2VecTy);
  auto fp16x2Vec1 = bitcast(a1, fp16x2VecTy);

  return { extract_element(f16_ty, fp16x2Vec0, i32_val(0)),
	   extract_element(f16_ty, fp16x2Vec0, i32_val(1)),
	   extract_element(f16_ty, fp16x2Vec1, i32_val(0)),
	   extract_element(f16_ty, fp16x2Vec1, i32_val(1))
	 };
}
#else
static const std::string Fp8E5M2_to_Fp16(bool hasNativeFP) {
  std::string ret;
  if (!hasNativeFP) {
    ret = "{                           \n"
          "prmt.b32 $0, 0, $2, 0x5140; \n\t"
          "prmt.b32 $1, 0, $2, 0x7362; \n\t"
          "}";
  } else {
    ret = "cvt.rn.f16x2.e5m2x2 $0, $1; \n\t";
  }
  return ret;
}
#endif

#ifdef USE_ROCM

static Value convert_val_Fp8E5M2FNUZ_to_Fp16(
  Location loc, ConversionPatternRewriter &rewriter, Value v) {
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  Value a = undef(fp8x2VecTy);
  a = insert_element(fp8x2VecTy, a, int_val(8, 0), i32_val(0));
  a = insert_element(fp8x2VecTy, a, v, i32_val(1));
  a = bitcast(a, i16_ty);

  auto e = and_(i16_ty, a, int_val(16, 0x7C00));
  auto m = and_(i16_ty, a, int_val(16, 0x0300));
  auto sign = and_(i16_ty, a, int_val(16, 0x8000));

  // check whether all exponents are zeros
  auto e_is_zero = icmp_eq(e, int_val(16, 0x0));

  // case 1, e is zero, need to move m right by 1 bit
  auto m1 = lshr(i16_ty, m, int_val(16, 1));
  auto o0 = or_(i16_ty, sign, m1);

  // case 2, e is nonzero, sub exponent by 1
  auto e1 = sub(i16_ty, e, int_val(16, 0x0400));

  auto e_is_one = icmp_eq(e, int_val(16, 0x0400));
  auto m2 = add(i16_ty, m1, int_val(16, 0x0200));

  auto o1 = or_(i16_ty, sign, or_(i16_ty, m, e1));
  auto o2 = or_(i16_ty, sign, m2);

  auto o12 = select(e_is_one, o2, o1);
  auto o = select(e_is_zero, o0, o12);

  return bitcast(o, f16_ty);
}

static SmallVector<Value>
Fp8E5M2FNUZ_to_Fp16(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {

  SmallVector<Value> result(4);
  result[0] = convert_val_Fp8E5M2FNUZ_to_Fp16(loc, rewriter, v[0]);
  result[1] = convert_val_Fp8E5M2FNUZ_to_Fp16(loc, rewriter, v[1]);
  result[2] = convert_val_Fp8E5M2FNUZ_to_Fp16(loc, rewriter, v[2]);
  result[3] = convert_val_Fp8E5M2FNUZ_to_Fp16(loc, rewriter, v[3]);

  return result;
}
#endif

#ifdef USE_ROCM
static SmallVector<Value>
Fp8E5M2_to_Bf16(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = undef(fp8x4VecTy);
  a0 = insert_element(fp8x4VecTy, a0, int_val(8,0), i32_val(0));
  a0 = insert_element(fp8x4VecTy, a0, v[0], i32_val(1));
  a0 = insert_element(fp8x4VecTy, a0, int_val(8,0), i32_val(2));
  a0 = insert_element(fp8x4VecTy, a0, v[1], i32_val(3));
  a0 = bitcast(a0, i32_ty);

  Value a1 = undef(fp8x4VecTy);
  a1 = insert_element(fp8x4VecTy, a1, int_val(8,0), i32_val(0));
  a1 = insert_element(fp8x4VecTy, a1, v[2], i32_val(1));
  a1 = insert_element(fp8x4VecTy, a1, int_val(8,0), i32_val(2));
  a1 = insert_element(fp8x4VecTy, a1, v[3], i32_val(3));
  a1 = bitcast(a1, i32_ty);

  Value b0 = and_(i32_ty, a0, i32_val(0x7fff7fff));
  Value b1 = and_(i32_ty, a1, i32_val(0x7fff7fff));
  b0 = lshr(i32_ty, b0, i32_val(3));
  b1 = lshr(i32_ty, b1, i32_val(3));

  b0 = add(i32_ty, b0, i32_val(0x38003800));
  b1 = add(i32_ty, b1, i32_val(0x38003800));
  Value sign0 = and_(i32_ty, a0, i32_val(0x80008000));
  Value sign1 = and_(i32_ty, a1, i32_val(0x80008000));


  auto bf16x2VecTy = vec_ty(i16_ty, 2);
  Value bf16x2Vec0 = or_(i32_ty, sign0, b0);
  Value bf16x2Vec1 = or_(i32_ty, sign1, b1);
  bf16x2Vec0 = bitcast(bf16x2Vec0, bf16x2VecTy);
  bf16x2Vec1 = bitcast(bf16x2Vec1, bf16x2VecTy);

  return { extract_element(i16_ty, bf16x2Vec0, i32_val(0)),
	   extract_element(i16_ty, bf16x2Vec0, i32_val(1)),
	   extract_element(i16_ty, bf16x2Vec1, i32_val(0)),
	   extract_element(i16_ty, bf16x2Vec1, i32_val(1))
	 };
}
#else
static const std::string Fp8E5M2_to_Bf16(bool hasNativeFP) {
  std::string ret;
  if (!hasNativeFP) {
        ret = "{                                        \n"
          ".reg .b32 a<2>, b<2>, c<4>, d<4>, e112;  \n" // if input = 0xf1f2f3f4
          "mov.u32 e112, 0x77800000;                \n"
          "prmt.b32 a0, 0, $2, 0x5140;              \n" // a0 = 0xf300f400
          "prmt.b32 a1, 0, $2, 0x7362;              \n" // a1 = 0xf100f200
          "lop3.b32 b0, a0, 0x7fff7fff, 0, 0xc0;    \n" // b0 = a0 & 0x7fff7fff
          "lop3.b32 b1, a1, 0x7fff7fff, 0, 0xc0;    \n" // (strip sign)
          "shr.b32  b0, b0, 3;                      \n" // b0 >>= 3
          "shr.b32  b1, b1, 3;                      \n" // shift into bf16
                                                        // position
          "and.b32 c0, b0, 0xFFFF0000;              \n" // c0 = f3
          "shl.b32 c1, b0, 16;                      \n" // c1 = f4
          "and.b32 c2, b1, 0xFFFF0000;              \n" // c2 = f1
          "shl.b32 c3, b1, 16;                      \n" // c3 = f2
          "mul.f32 d0, c0, e112;                    \n" // d0 = c0 * 0x77800000
          "mul.f32 d1, c1, e112;                    \n" // d1 = c1 * 0x77800000
          "mul.f32 d2, c2, e112;                    \n" // d2 = c2 * 0x77800000
          "mul.f32 d3, c3, e112;                    \n" // d3 = c3 * 0x77800000
          "prmt.b32 b0, d0, d1, 0x3276;             \n" // b0 = 0xd3d4
          "prmt.b32 b1, d2, d3, 0x3276;             \n" // b1 = 0xd1d2
          "lop3.b32 $0, b0, 0x80008000, a0, 0xf8;   \n" // out0 =
                                                        // b0|(0x80008000&a0)
          "lop3.b32 $1, b1, 0x80008000, a1, 0xf8;   \n" // (restore sign)
          "}";
  } else {
    ret = "{                                       \n"
          ".reg .b32 a;                            \n"
          ".reg .f16 a<2>;                         \n"
          ".reg .b16 b<2>;                         \n"
          "cvt.rn.f16x2.e5m2x2 a, $1;              \n"
          "mov.b32 {a0, a1}, a;                    \n"
          "cvt.bf16.f16 b0, a0;                    \n"
          "cvt.bf16.f16 b1, a1;                    \n"
          "mov.b32 $0, {b0, b1};                   \n"
          "}";
  }
  return ret;
}
#endif

#ifdef USE_ROCM
static SmallVector<Value>
Bf16_to_Fp8E5M2(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  auto bf16x2VecTy = vec_ty(i16_ty, 2);
  Value bf16x2Vec0 = undef(bf16x2VecTy);
  Value bf16x2Vec1 = undef(bf16x2VecTy);
  bf16x2Vec0 = insert_element(bf16x2VecTy, bf16x2Vec0, v[0], i32_val(0));
  bf16x2Vec0 = insert_element(bf16x2VecTy, bf16x2Vec0, v[1], i32_val(1));
  bf16x2Vec1 = insert_element(bf16x2VecTy, bf16x2Vec1, v[2], i32_val(0));
  bf16x2Vec1 = insert_element(bf16x2VecTy, bf16x2Vec1, v[3], i32_val(1));
  bf16x2Vec0 = bitcast(bf16x2Vec0, i32_ty);
  bf16x2Vec1 = bitcast(bf16x2Vec1, i32_ty);

  Value sign0 = and_(i32_ty, bf16x2Vec0, i32_val(0x80008000));
  Value sign1 = and_(i32_ty, bf16x2Vec1, i32_val(0x80008000));
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value sign = undef(fp8x4VecTy);
  sign0 = bitcast(sign0, fp8x4VecTy);
  sign1 = bitcast(sign1, fp8x4VecTy);
  sign = insert_element( fp8x4VecTy, sign, extract_element(i8_ty, sign0, i32_val(1)), i32_val(0) );
  sign = insert_element( fp8x4VecTy, sign, extract_element(i8_ty, sign0, i32_val(3)), i32_val(1) );
  sign = insert_element( fp8x4VecTy, sign, extract_element(i8_ty, sign1, i32_val(1)), i32_val(2) );
  sign = insert_element( fp8x4VecTy, sign, extract_element(i8_ty, sign1, i32_val(3)), i32_val(3) );
  sign = bitcast(sign, i32_ty);

  Value nosign0 = and_(i32_ty, bf16x2Vec0, i32_val(0x7fff7fff));
  Value nosign1 = and_(i32_ty, bf16x2Vec1, i32_val(0x7fff7fff));

  Value nosign_0_0 = and_(i32_ty, nosign0, i32_val(0xffff0000));
  nosign_0_0 = umax(i32_ty, nosign_0_0, i32_val(0x38000000));
  nosign_0_0 = umin(i32_ty, nosign_0_0, i32_val(0x57e00000));
  Value nosign_0_1 = and_(i32_ty, nosign0, i32_val(0x0000ffff));
  nosign_0_1 = umax(i32_ty, nosign_0_1, i32_val(0x3800));
  nosign_0_1 = umin(i32_ty, nosign_0_1, i32_val(0x57e0));
  nosign0 = or_(i32_ty, nosign_0_0, nosign_0_1);

  Value nosign_1_0 = and_(i32_ty, nosign1, i32_val(0xffff0000));
  nosign_1_0 = umax(i32_ty, nosign_1_0, i32_val(0x38000000));
  nosign_1_0 = umin(i32_ty, nosign_1_0, i32_val(0x57e00000));
  Value nosign_1_1 = and_(i32_ty, nosign1, i32_val(0x0000ffff));
  nosign_1_1 = umax(i32_ty, nosign_1_1, i32_val(0x3800));
  nosign_1_1 = umin(i32_ty, nosign_1_1, i32_val(0x57e0));
  nosign1 = or_(i32_ty, nosign_1_0, nosign_1_1);

  nosign0 = add(i32_ty, nosign0, i32_val(0x00100010));
  nosign1 = add(i32_ty, nosign1, i32_val(0x00100010));
  nosign0 = sub(i32_ty, nosign0, i32_val(0x38003800));
  nosign1 = sub(i32_ty, nosign1, i32_val(0x38003800));
  nosign0 = shl(i32_ty, nosign0, i32_val(3));
  nosign1 = shl(i32_ty, nosign1, i32_val(3));

  nosign0 = bitcast(nosign0, fp8x4VecTy);
  nosign1 = bitcast(nosign1, fp8x4VecTy);
  Value nosign = undef(fp8x4VecTy);
  nosign = insert_element( fp8x4VecTy, nosign, extract_element(i8_ty, nosign0, i32_val(1)), i32_val(0) );
  nosign = insert_element( fp8x4VecTy, nosign, extract_element(i8_ty, nosign0, i32_val(3)), i32_val(1) );
  nosign = insert_element( fp8x4VecTy, nosign, extract_element(i8_ty, nosign1, i32_val(1)), i32_val(2) );
  nosign = insert_element( fp8x4VecTy, nosign, extract_element(i8_ty, nosign1, i32_val(3)), i32_val(3) );
  nosign = bitcast(nosign, i32_ty);

  Value fp8x4Vec = or_(i32_ty, nosign, sign);
  fp8x4Vec = bitcast(fp8x4Vec, fp8x4VecTy);
  return {extract_element(i8_ty, fp8x4Vec, i32_val(0)),
	  extract_element(i8_ty, fp8x4Vec, i32_val(1)),
	  extract_element(i8_ty, fp8x4Vec, i32_val(2)),
	  extract_element(i8_ty, fp8x4Vec, i32_val(3))};
}
#else
static const std::string Bf16_to_Fp8E5M2(bool hasNativeFP) {
  std::string ret;
  if (!hasNativeFP) {
    ret =
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
  } else {
    ret = "{                                       \n"
          ".reg .b16 a<2>;                         \n"
          ".reg .f32 b<2>;                         \n"
          "mov.b32 {a0, a1}, $1;                   \n"
          "cvt.f32.bf16 b0, a0;                    \n"
          "cvt.f32.bf16 b1, a1;                    \n"
          "cvt.rn.satfinite.e5m2x2.f32 $0, b0, b1; \n"
          "}";
  }
  return ret;
}
#endif
/* ----- FP8E4M3B15 ------ */
// This data-type is a variant of the standard FP8E4M3 format.
// It was designed for fast software conversion to FP16 on
// nvidia GPUs that do not support it natively.
// This is the same format as FP8E4M3Nv, but:
//   - the exponent bias is 15 instead of 7
//   - 0xff and 0x7f are mapped to +-1.750 instead of +-nan
#ifdef USE_ROCM
static SmallVector<Value>
Fp8E4M3B15_to_Fp16(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = undef(fp8x4VecTy);
  a0 = insert_element(fp8x4VecTy, a0, int_val(8,0), i32_val(0));
  a0 = insert_element(fp8x4VecTy, a0, v[0], i32_val(1));
  a0 = insert_element(fp8x4VecTy, a0, int_val(8,0), i32_val(2));
  a0 = insert_element(fp8x4VecTy, a0, v[1], i32_val(3));
  a0 = bitcast(a0, i32_ty);

  Value a1 = undef(fp8x4VecTy);
  a1 = insert_element(fp8x4VecTy, a1, int_val(8,0), i32_val(0));
  a1 = insert_element(fp8x4VecTy, a1, v[2], i32_val(1));
  a1 = insert_element(fp8x4VecTy, a1, int_val(8,0), i32_val(2));
  a1 = insert_element(fp8x4VecTy, a1, v[3], i32_val(3));
  a1 = bitcast(a1, i32_ty);

  Value b0 = and_(i32_ty, a0, i32_val(0x7fff7fff));
  Value b1 = and_(i32_ty, a1, i32_val(0x7fff7fff));

  b0 = lshr(i32_ty, b0, i32_val(1));
  b1 = lshr(i32_ty, b1, i32_val(1));

  b0 = or_( i32_ty, b0, and_(i32_ty, a0, i32_val(0x80008000)) );
  b1 = or_( i32_ty, b1, and_(i32_ty, a1, i32_val(0x80008000)) );

  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  auto fp16x2Vec0 = bitcast(b0, fp16x2VecTy);
  auto fp16x2Vec1 = bitcast(b1, fp16x2VecTy);

  return { extract_element(f16_ty, fp16x2Vec0, i32_val(0)),
	   extract_element(f16_ty, fp16x2Vec0, i32_val(1)),
	   extract_element(f16_ty, fp16x2Vec1, i32_val(0)),
	   extract_element(f16_ty, fp16x2Vec1, i32_val(1))
	 };
}
#else
const std::string Fp8E4M3B15_to_Fp16 =
    "{                                      \n"
    ".reg .b32 a<2>, b<2>;                  \n"
    "prmt.b32 a0, 0, $2, 0x5746;            \n"
    "and.b32 b0, a0, 0x7f007f00;            \n"
    "and.b32 b1, a0, 0x00ff00ff;            \n"
    "and.b32 a1, a0, 0x00800080;            \n"
    "shr.b32  b0, b0, 1;                    \n"
    "add.u32 b1, b1, a1;                    \n"
    "lop3.b32 $0, b0, 0x80008000, a0, 0xf8; \n"
    "shl.b32 $1, b1, 7;                     \n"
    "}                                      \n";
#endif

#ifdef USE_ROCM
static SmallVector<Value>
Fp16_to_Fp8E4M3B15(Location loc, ConversionPatternRewriter &rewriter,
                         const SmallVector<Value> &v) {
  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec0 = undef(fp16x2VecTy);
  Value fp16x2Vec1 = undef(fp16x2VecTy);

  fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v[0], i32_val(0));
  fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v[1], i32_val(1));
  fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v[2], i32_val(0));
  fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v[3], i32_val(1));
  
  Value fp16x2VecMin = i32_val(0xBF80BF80);
  Value fp16x2VecMax = i32_val(0x3F803F80);
  fp16x2VecMin = bitcast(fp16x2VecMin, fp16x2VecTy);
  fp16x2VecMax = bitcast(fp16x2VecMax, fp16x2VecTy);
  fp16x2Vec0 = fmax(fp16x2VecTy, fp16x2Vec0, fp16x2VecMin);
  fp16x2Vec1 = fmax(fp16x2VecTy, fp16x2Vec1, fp16x2VecMin);
  fp16x2Vec0 = fmin(fp16x2VecTy, fp16x2Vec0, fp16x2VecMax);
  fp16x2Vec1 = fmin(fp16x2VecTy, fp16x2Vec1, fp16x2VecMax);

  fp16x2Vec0 = bitcast(fp16x2Vec0, i32_ty);
  fp16x2Vec1 = bitcast(fp16x2Vec1, i32_ty);

  Value a0 = shl(i32_ty, fp16x2Vec0, i32_val(1));
  Value a1 = shl(i32_ty, fp16x2Vec1, i32_val(1));
  a0 = and_(i32_ty, a0, i32_val(0x7fff7fff));
  a1 = and_(i32_ty, a1, i32_val(0x7fff7fff));
  a0 = add(i32_ty, a0, i32_val(0x00800080));
  a1 = add(i32_ty, a1, i32_val(0x00800080));
  Value b0 = or_( i32_ty, and_(i32_ty, fp16x2Vec0, i32_val(0x80008000)), a0 );
  Value b1 = or_( i32_ty, and_(i32_ty, fp16x2Vec1, i32_val(0x80008000)), a1 );

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  b0 = bitcast(b0, fp8x4VecTy); 
  b1 = bitcast(b1, fp8x4VecTy); 

  return {extract_element(i8_ty, b0, i32_val(1)),
	  extract_element(i8_ty, b0, i32_val(3)),
	  extract_element(i8_ty, b1, i32_val(1)),
	  extract_element(i8_ty, b1, i32_val(3))
	  };
}
#else
static const std::string Fp16_to_Fp8E4M3B15(bool has_minx2) {
  std::string ret;
  ret += "{                                      \n"
         ".reg .pred p<4>;                       \n"
         ".reg .b32 a<2>, b<2>;                  \n"
         ".reg .b16 c<4>;                        \n"
         ".reg .b16 max_val_f16;                 \n"
         ".reg .b32 max_val_f16x2;               \n"
         "mov.b16 max_val_f16,   0x3F00;         \n"
         "mov.b32 max_val_f16x2, 0x3F003F00;     \n"
         "and.b32 a0, $1, 0x7fff7fff;            \n"
         "and.b32 a1, $2, 0x7fff7fff;            \n";
  if (has_minx2)
    ret += "min.f16x2 a0, a0, max_val_f16x2;      \n"
           "min.f16x2 a1, a1, max_val_f16x2;      \n";
  else
    ret += "setp.lt.f16x2  p0|p1, a0, max_val_f16x2;   \n"
           "setp.lt.f16x2  p2|p3, a1, max_val_f16x2;   \n"
           "mov.b32 {c0, c1}, a0;                \n"
           "mov.b32 {c2, c3}, a1;                \n"
           "selp.b16  c0, c0, max_val_f16, p0;   \n"
           "selp.b16  c1, c1, max_val_f16, p1;   \n"
           "selp.b16  c2, c2, max_val_f16, p2;   \n"
           "selp.b16  c3, c3, max_val_f16, p3;   \n"
           "mov.b32 a0, {c0, c1};                \n"
           "mov.b32 a1, {c2, c3};                \n";
  ret += "mad.lo.u32 a0, a0, 2, 0x00800080;      \n"
         "mad.lo.u32 a1, a1, 2, 0x00800080;      \n"
         "lop3.b32 b0, $1, 0x80008000, a0, 0xea; \n"
         "lop3.b32 b1, $2, 0x80008000, a1, 0xea; \n"
         "prmt.b32 $0, b0, b1, 0x7531;           \n"
         "}";
  return ret;
}
#endif

/* ----- FP8E4M3B15X4 ------ */
// NOTE: NOT USED RIGHT NOW
// Packed variant of FP8E4M3B15
// A little bit more efficient but elements need are not
// serialized as you expect when 4 are packed into int32.

// fast conversion code provided by Scott Gray @ OpenAI
// $0 = (($2 << 1) & 0x80008000u) | (($2 << 7) & 0x3f803f80u);
// $1 = (($2 << 0) & 0x80008000u) | (($2 << 0) & 0x3f803f80u);
// WARN: subnormal (0bs0000xxx) are not handled
#ifdef USE_ROCM
static SmallVector<Value>
Fp8E4M3B15x4_to_Fp16(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value fp8x4Vec = undef(fp8x4VecTy);
  fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v[0], i32_val(0));
  fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v[1], i32_val(1));
  fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v[2], i32_val(2));
  fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v[3], i32_val(3));
  fp8x4Vec = bitcast(fp8x4Vec, i32_ty);

  Value a0 = add(i32_ty, fp8x4Vec, fp8x4Vec);
  Value a1 = shl(i32_ty, fp8x4Vec, i32_val(7));

  Value fp16x2Vec0 = and_(i32_ty, a0, i32_val(0x80008000)); 
  fp16x2Vec0 = or_(i32_ty, fp16x2Vec0, and_(i32_ty, a1, i32_val(0x3f803f80)) );
  Value fp16x2Vec1 = and_(i32_ty, fp8x4Vec, i32_val(0xbf80bf80));

  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  fp16x2Vec0 = bitcast(fp16x2Vec0, fp16x2VecTy);
  fp16x2Vec1 = bitcast(fp16x2Vec1, fp16x2VecTy);

  return { extract_element(f16_ty, fp16x2Vec0, i32_val(0)),
	   extract_element(f16_ty, fp16x2Vec0, i32_val(1)),
	   extract_element(f16_ty, fp16x2Vec1, i32_val(0)),
	   extract_element(f16_ty, fp16x2Vec1, i32_val(1))
	 };
}
#else
static const std::string Fp8E4M3B15x4_to_Fp16 =
    "{                                      \n"
    ".reg .b32 a<2>;                        \n"
    "add.u32 a0, $2, $2;                    \n"
    "shl.b32 a1, $2, 7;                     \n"
    "and.b32  $0, a0, 0x80008000;           \n"
    "lop3.b32 $0, $0, a1, 0x3f803f80, 0xf8; \n"
    "and.b32  $1, $2, 0xbf80bf80;           \n"
    "}";
#endif

// Fp16 -> Fp8E4M3B15 (packed)
// fast conversion code provided by Scott Gray @ OpenAI
// ret = ((e4.x >> 1) & (0x80008000u >> 1)) |
//       ((e4.x >> 7) & (0x3f803f80u >> 7)) |
//       ((e4.y >> 0) & (0x80008000u >> 0)) |
//       ((e4.y >> 0) & (0x3f803f80u >> 0)) ;
// WARN: subnormal (0bs0000xxx) are not handled
#ifdef USE_ROCM
static SmallVector<Value>
Fp16_to_Fp8E4M3B15x4(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec0 = undef(fp16x2VecTy);
  Value fp16x2Vec1 = undef(fp16x2VecTy);

  fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v[0], i32_val(0));
  fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v[1], i32_val(1));
  fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v[2], i32_val(0));
  fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v[3], i32_val(1));
  
  fp16x2Vec0 = bitcast(fp16x2Vec0, i32_ty);
  fp16x2Vec1 = bitcast(fp16x2Vec1, i32_ty);

  Value a0 = lshr(i32_ty, fp16x2Vec0, i32_val(1));
  Value a1 = lshr(i32_ty, fp16x2Vec0, i32_val(7));

  Value fp8x4Vec = and_(i32_ty, a0, i32_val(0x40004000));
  fp8x4Vec = or_(i32_ty, fp8x4Vec, and_(i32_ty, a1, i32_val(0x007f007f)) );
  fp8x4Vec = or_(i32_ty, fp8x4Vec, and_(i32_ty, fp16x2Vec1, i32_val(0xbf80bf80)) );

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  fp8x4Vec = bitcast(fp8x4Vec, fp8x4VecTy); 

  return {extract_element(i8_ty, fp8x4Vec, i32_val(0)),
	  extract_element(i8_ty, fp8x4Vec, i32_val(1)),
	  extract_element(i8_ty, fp8x4Vec, i32_val(2)),
	  extract_element(i8_ty, fp8x4Vec, i32_val(3))
	  };
}
#else
static const std::string Fp16_to_Fp8E4M3B15x4 =
    "{                                       \n"
    ".reg .b32 a<2>;                         \n"
    "shr.b32  a0, $1, 1;                     \n"
    "shr.b32  a1, $1, 7;                     \n"
    "and.b32  $0,     a0, 0x40004000;        \n"
    "lop3.b32 $0, $0, a1, 0x007f007f, 0xf8;  \n"
    "lop3.b32 $0, $0, $2, 0xbf80bf80, 0xf8;  \n"
    "}";
#endif

/* ----- FP8E4M3 ------ */
// Note: when handled by software, this format
// does not handle denormals and has
// more than a single NaN values.

#ifdef USE_ROCM
static Value convert_val_Fp8E4M3FNUZ_to_Fp16(
  Location loc, ConversionPatternRewriter &rewriter, Value v) {
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  Value a = undef(fp8x2VecTy);
  a = insert_element(fp8x2VecTy, a, int_val(8, 0), i32_val(0));
  a = insert_element(fp8x2VecTy, a, v, i32_val(1));
  a = bitcast(a, i16_ty);

  auto e_mask = int_val(16, 0x7A00);
  auto e = and_(i16_ty, a, e_mask);

  auto m = and_(i16_ty, a, int_val(16, 0x0700));
  auto sign = and_(i16_ty, a, int_val(16, 0x8000));

  // check whether all exponents are zeros
  auto e_is_zero = icmp_eq(e, int_val(16, 0x0));
  auto b = and_(i16_ty, a, int_val(16, 0x7FFF));
  auto b1 = lshr(i16_ty, b, int_val(16, 1));

  // case 1, e is nonzero, add exponent by 6
  auto o0v = add(i16_ty, b1, int_val(16, 0x0C00));
  auto o0 = or_(i16_ty, o0v, sign);

  // case 2, e is nonzero, add exponent by 7
  auto o1v = add(i16_ty, b1, int_val(16, 0x1C00));
  auto o1 = or_(i16_ty, o1v, sign);

  auto io = select(e_is_zero, o0, o1);
  return bitcast(io, f16_ty);
}

// Fp8E4M3FNUZ -> Fp16 (packed)
static SmallVector<Value>
Fp8E4M3FNUZ_to_Fp16(Location loc, ConversionPatternRewriter &rewriter,
		   const SmallVector<Value> &v) {
  SmallVector<Value> result(2);
  result[0] = convert_val_Fp8E4M3FNUZ_to_Fp16(loc, rewriter, v[0]);
  result[1] = convert_val_Fp8E4M3FNUZ_to_Fp16(loc, rewriter, v[1]);

  return result;
}
#else
const std::string Fp8E4M3FNUZ_to_Fp16 =
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
#endif

// Fp16 -> Fp8E4M3 (packed)
#ifdef USE_ROCM
static Value convert_val_Fp16_to_Fp8E4M3FNUZ(
  Location loc, ConversionPatternRewriter &rewriter, Value v) {
  auto vi16 = bitcast(v, i16_ty);
  auto e10 = and_(vi16, int_val(16, 0x7C00));
  auto e = lshr(i16_ty, e10, int_val(16, 10));

  auto s = and_(i16_ty, vi16, int_val(16, 0x8000));

  auto m7 = and_(i16_ty, vi16, int_val(16, 0x0380));
  auto m = shl(i16_ty, m7, int_val(16, 1));

  // three cases: 
  //  1) e > 21 --> e = 1111, 
  //  2) e <= 7 ---> e = 0, 
  //  3) others, normal conversion
  auto e1 = int_val(16, 0x7800);
  auto e2 = int_val(16, 0x0);
  auto e31 = sub(i16_ty, e10, int_val(16, 0x1C00));
  auto e3 = shl(i16_ty, e31, int_val(16, 1));

  auto c13 = icmp_sgt(e, int_val(16, 21));
  auto e13 = select(c13, e1, e3);
  auto c23 = icmp_sle(e, int_val(16, 7));
  auto re = select(c23, e2, e13);

  auto r = or_(i16_ty, s, or_(i16_ty, re, m));
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  auto res = bitcast(r, fp8x2VecTy); 

  return extract_element(i8_ty, res, i32_val(1));
}

static SmallVector<Value>
Fp16_to_Fp8E4M3FNUZ(Location loc, ConversionPatternRewriter &rewriter,
		   const SmallVector<Value> &v) {

  SmallVector<Value> result(2);
  result[0] = convert_val_Fp16_to_Fp8E4M3FNUZ(loc, rewriter, v[0]);
  result[1] = convert_val_Fp16_to_Fp8E4M3FNUZ(loc, rewriter, v[1]);

  return result;
}
#else
const std::string Fp16_to_Fp8E4M3FNUZ =
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
#endif

// WARN: subnormal (0bs0000xxx) are not handled
#ifdef USE_ROCM
static SmallVector<Value>
Fp8E4M3_to_Bf16(Location loc, ConversionPatternRewriter &rewriter,
		   const SmallVector<Value> &v) {
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = undef(fp8x4VecTy);
  a0 = insert_element(fp8x4VecTy, a0, int_val(8,0), i32_val(0));
  a0 = insert_element(fp8x4VecTy, a0, v[0], i32_val(1));
  a0 = insert_element(fp8x4VecTy, a0, int_val(8,0), i32_val(2));
  a0 = insert_element(fp8x4VecTy, a0, v[1], i32_val(3));
  a0 = bitcast(a0, i32_ty);

  Value a1 = undef(fp8x4VecTy);
  a1 = insert_element(fp8x4VecTy, a1, int_val(8,0), i32_val(0));
  a1 = insert_element(fp8x4VecTy, a1, v[2], i32_val(1));
  a1 = insert_element(fp8x4VecTy, a1, int_val(8,0), i32_val(2));
  a1 = insert_element(fp8x4VecTy, a1, v[3], i32_val(3));
  a1 = bitcast(a1, i32_ty);

  Value b0 = and_(i32_ty, a0, i32_val(0x7fff7fff));
  Value b1 = and_(i32_ty, a1, i32_val(0x7fff7fff));
  b0 = lshr(i32_ty, b0, i32_val(4));
  b1 = lshr(i32_ty, b1, i32_val(4));

  b0 = add(i32_ty, b0, i32_val(0x3c003c00));
  b1 = add(i32_ty, b1, i32_val(0x3c003c00));
  Value sign0 = and_(i32_ty, a0, i32_val(0x80008000));
  Value sign1 = and_(i32_ty, a1, i32_val(0x80008000));


  auto bf16x2VecTy = vec_ty(i16_ty, 2);
  Value bf16x2Vec0 = or_(i32_ty, sign0, b0);
  Value bf16x2Vec1 = or_(i32_ty, sign1, b1);
  bf16x2Vec0 = bitcast(bf16x2Vec0, bf16x2VecTy);
  bf16x2Vec1 = bitcast(bf16x2Vec1, bf16x2VecTy);

  return { extract_element(i16_ty, bf16x2Vec0, i32_val(0)),
	   extract_element(i16_ty, bf16x2Vec0, i32_val(1)),
	   extract_element(i16_ty, bf16x2Vec1, i32_val(0)),
	   extract_element(i16_ty, bf16x2Vec1, i32_val(1))
	 };
}
#else
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
#endif

#ifdef USE_ROCM
static SmallVector<Value>
Bf16_to_Fp8E4M3(Location loc, ConversionPatternRewriter &rewriter,
		   const SmallVector<Value> &v) {
  auto bf16x2VecTy = vec_ty(i16_ty, 2);
  Value bf16x2Vec0 = undef(bf16x2VecTy);
  Value bf16x2Vec1 = undef(bf16x2VecTy);
  bf16x2Vec0 = insert_element(bf16x2VecTy, bf16x2Vec0, v[0], i32_val(0));
  bf16x2Vec0 = insert_element(bf16x2VecTy, bf16x2Vec0, v[1], i32_val(1));
  bf16x2Vec1 = insert_element(bf16x2VecTy, bf16x2Vec1, v[2], i32_val(0));
  bf16x2Vec1 = insert_element(bf16x2VecTy, bf16x2Vec1, v[3], i32_val(1));
  bf16x2Vec0 = bitcast(bf16x2Vec0, i32_ty);
  bf16x2Vec1 = bitcast(bf16x2Vec1, i32_ty);

  Value sign0 = and_(i32_ty, bf16x2Vec0, i32_val(0x80008000));
  Value sign1 = and_(i32_ty, bf16x2Vec1, i32_val(0x80008000));
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value sign = undef(fp8x4VecTy);
  sign0 = bitcast(sign0, fp8x4VecTy);
  sign1 = bitcast(sign1, fp8x4VecTy);
  sign = insert_element( fp8x4VecTy, sign, extract_element(i8_ty, sign0, i32_val(1)), i32_val(0) );
  sign = insert_element( fp8x4VecTy, sign, extract_element(i8_ty, sign0, i32_val(3)), i32_val(1) );
  sign = insert_element( fp8x4VecTy, sign, extract_element(i8_ty, sign1, i32_val(1)), i32_val(2) );
  sign = insert_element( fp8x4VecTy, sign, extract_element(i8_ty, sign1, i32_val(3)), i32_val(3) );
  sign = bitcast(sign, i32_ty);

  Value nosign0 = and_(i32_ty, bf16x2Vec0, i32_val(0x7fff7fff));
  Value nosign1 = and_(i32_ty, bf16x2Vec1, i32_val(0x7fff7fff));

  Value nosign_0_0 = and_(i32_ty, nosign0, i32_val(0xffff0000));
  nosign_0_0 = umax(i32_ty, nosign_0_0, i32_val(0x3c000000));
  nosign_0_0 = umin(i32_ty, nosign_0_0, i32_val(0x43f00000));
  Value nosign_0_1 = and_(i32_ty, nosign0, i32_val(0x0000ffff));
  nosign_0_1 = umax(i32_ty, nosign_0_1, i32_val(0x3c00));
  nosign_0_1 = umin(i32_ty, nosign_0_1, i32_val(0x43f0));
  nosign0 = or_(i32_ty, nosign_0_0, nosign_0_1);

  Value nosign_1_0 = and_(i32_ty, nosign1, i32_val(0xffff0000));
  nosign_1_0 = umax(i32_ty, nosign_1_0, i32_val(0x3c000000));
  nosign_1_0 = umin(i32_ty, nosign_1_0, i32_val(0x43f00000));
  Value nosign_1_1 = and_(i32_ty, nosign1, i32_val(0x0000ffff));
  nosign_1_1 = umax(i32_ty, nosign_1_1, i32_val(0x3c00));
  nosign_1_1 = umin(i32_ty, nosign_1_1, i32_val(0x43f0));
  nosign1 = or_(i32_ty, nosign_1_0, nosign_1_1);

  nosign0 = add(i32_ty, nosign0, i32_val(0x80008));
  nosign1 = add(i32_ty, nosign1, i32_val(0x80008));
  nosign0 = sub(i32_ty, nosign0, i32_val(0x3c003c00));
  nosign1 = sub(i32_ty, nosign1, i32_val(0x3c003c00));
  nosign0 = lshr(i32_ty, nosign0, i32_val(4));
  nosign1 = lshr(i32_ty, nosign1, i32_val(4));

  nosign0 = bitcast(nosign0, fp8x4VecTy);
  nosign1 = bitcast(nosign1, fp8x4VecTy);
  Value nosign = undef(fp8x4VecTy);
  nosign = insert_element( fp8x4VecTy, nosign, extract_element(i8_ty, nosign0, i32_val(0)), i32_val(0) );
  nosign = insert_element( fp8x4VecTy, nosign, extract_element(i8_ty, nosign0, i32_val(2)), i32_val(1) );
  nosign = insert_element( fp8x4VecTy, nosign, extract_element(i8_ty, nosign1, i32_val(0)), i32_val(2) );
  nosign = insert_element( fp8x4VecTy, nosign, extract_element(i8_ty, nosign1, i32_val(2)), i32_val(3) );
  nosign = bitcast(nosign, i32_ty);

  Value fp8x4Vec = or_(i32_ty, nosign, sign);
  fp8x4Vec = bitcast(fp8x4Vec, fp8x4VecTy);
  return {extract_element(i8_ty, fp8x4Vec, i32_val(0)),
	  extract_element(i8_ty, fp8x4Vec, i32_val(1)),
	  extract_element(i8_ty, fp8x4Vec, i32_val(2)),
	  extract_element(i8_ty, fp8x4Vec, i32_val(3))};
}
#else
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

// Fp8E4M3 (x2) -> Fp16 (x2) (packed)
static const std::string Fp8E4M3Nv_to_Fp16 = "{ \n"
                                             "cvt.rn.f16x2.e4m3x2 $0, $1; \n"
                                             "}";
// Fp16 (x2) -> Fp8E4M3 (x2) (packed)
static const std::string Fp16_to_Fp8E4M3Nv =
    "{ \n"
    "cvt.rn.satfinite.e4m3x2.f16x2 $0, $1; \n"
    "}";

// Fp8E4M3 (x2) -> Fp16 (x2) (packed)
static const std::string Fp8E4M3Nv_to_Bf16 =
    "{                                       \n"
    ".reg .b32 a;                            \n"
    ".reg .f16 a<2>;                         \n"
    ".reg .b16 b<2>;                         \n"
    "cvt.rn.f16x2.e4m3x2 a, $1;              \n"
    "mov.b32 {a0, a1}, a;                    \n"
    "cvt.bf16.f16 b0, a0;                    \n"
    "cvt.bf16.f16 b1, a1;                    \n"
    "mov.b32 $0, {b0, b1};                   \n"
    "}";

// Bf16 (x2) -> Fp8E4M3 (x2) (packed)
static const std::string Bf16_to_Fp8E4M3Nv =
    "{                                       \n"
    ".reg .b16 a<2>;                         \n"
    ".reg .f32 b<2>;                         \n"
    "mov.b32 {a0, a1}, $1;                   \n"
    "cvt.f32.bf16 b0, a0;                    \n"
    "cvt.f32.bf16 b1, a1;                    \n"
    "cvt.rn.satfinite.e4m3x2.f32 $0, b0, b1; \n"
    "}";

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

// Fp32 (x2) -> Fp8 (x2) (packed)
static const std::string Fp32_to_Fp8E4M3Nv =
    "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \n";
static const std::string Fp32_to_Fp8E5M2 =
    "cvt.rn.satfinite.e5m2x2.f32 $0, $2, $1; \n";
#endif

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
  // If the parent of the dot operand is in block encoding, we don't need to
  // reorder elements
  auto parentEncoding =
      dyn_cast<triton::gpu::MmaEncodingAttr>(ouEncoding.getParent());
  if (!parentEncoding)
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
  }
  llvm_unreachable("unimplemented code path");
}

inline Type getElementType(Value value) {
  auto type = value.getType();
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return tensorType.getElementType();
  return type;
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
  if (!(encoding && (encoding.getParent().isa<MmaEncodingAttr>() or 
        encoding.getParent().isa<MfmaEncodingAttr>()))) {
    return inValues;
  }
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
    int numElements = v.size();
    assert(numElements == 4 || numElements == 2 && "invalid vector size");

    auto ctx = rewriter.getContext();
    int inBitwidth = inType.getIntOrFloatBitWidth();
    int outBitwidth = outType.getIntOrFloatBitWidth();
    // first, we pack `v` into 32-bit ints
    int inVecWidth = inVecWidthBits / inBitwidth;
    auto inVecTy = vec_ty(inType, inVecWidth);
    SmallVector<Value> inPacked(numElements / inVecWidth, undef(inVecTy));
    for (size_t i = 0; i < numElements; i++)
      inPacked[i / inVecWidth] = insert_element(
          inVecTy, inPacked[i / inVecWidth], v[i], i32_val(i % inVecWidth));
    for (size_t i = 0; i < inPacked.size(); i++)
      inPacked[i] = bitcast(inPacked[i], int_ty(inVecWidthBits));

    // then, we run the provided inline PTX
    int outVecWidth = outVecWidthBits / outBitwidth;
    int outNums = numElements / outVecWidth;
    PTXBuilder builder;
    SmallVector<PTXBuilder::Operand *> operands;
    auto outConstriant = outVecWidthBits == 16 ? "=h" : "=r";
    auto inConstraint = inVecWidthBits == 16 ? "h" : "r";
    for (int i = 0; i < outNums; i++) {
      operands.push_back(builder.newOperand(outConstriant));
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
        outPacked.push_back(extract_val(outVecTy, outStruct, i));
    }
    // unpack the output
    SmallVector<Value> ret;
    for (size_t i = 0; i < numElements; i++)
      ret.push_back(extract_element(outType, outPacked[i / outVecWidth],
                                    i32_val(i % outVecWidth)));
    return ret;
  };
  return converter;
}

class MultipleOperandsRange
    : public iterator_range<SmallVector<SmallVector<Value>>::iterator> {
  using ContainerT = SmallVector<SmallVector<Value>>;

public:
  using iterator_range<ContainerT::iterator>::iterator_range;
  ContainerT::reference operator[](ContainerT::size_type idx) {
    return begin()[idx];
  }
  ContainerT::const_reference operator[](ContainerT::size_type idx) const {
    return begin()[idx];
  }
  ContainerT::size_type size() const { return end() - begin(); }
};

// Base pattern for elementwise conversion using ConcreteT. Unpacks individual
// elements from a `!llvm.struct` via `llvm.extactvalue`, calls
// ConcreteT::createDestOps on each element, and packs them back into an
// `!llvm.struct` using `llvm.insertvalue`.
//
// Also supports processing the inputs in a vectorized form by consuming and
// producing multiple operand sets in ConcreteT::createDestOps.
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
    SmallVector<SmallVector<Value>> allOperands;
    for (auto operand : adaptor.getOperands()) {
      auto argTy = op->getOperand(0).getType();
      auto subOperands = this->getTypeConverter()->unpackLLElements(
          loc, operand, rewriter, argTy);
      subOperands = unpackI32(subOperands, argTy, rewriter, loc,
                              this->getTypeConverter());
      allOperands.resize(subOperands.size());
      for (auto v : llvm::enumerate(subOperands))
        allOperands[v.index()].push_back(v.value());
    }
    if (allOperands.size() == 0)
      allOperands.push_back({});

    SmallVector<Value> resultVals;
    for (auto it = allOperands.begin(), end = allOperands.end(); it != end;) {
      auto curr = static_cast<const ConcreteT *>(this)->createDestOps(
          op, adaptor, rewriter, elemTy, MultipleOperandsRange(it, end), loc);
      if (curr.size() == 0)
        return failure();
      for (auto v : curr) {
        if (!static_cast<bool>(v))
          return failure();
        resultVals.push_back(v);
      }
      it += curr.size();
    }

    if (op->getNumOperands() > 0) {
      auto argTy = op->getOperand(0).getType();
      resultVals = reorderValues(resultVals, argTy, resultTy);
    }
    resultVals =
        packI32(resultVals, resultTy, rewriter, loc, this->getTypeConverter());
    resultVals = this->getTypeConverter()->packMfmaOperand(resultVals, resultTy, rewriter, loc);

    Value view = this->getTypeConverter()->packLLElements(loc, resultVals,
                                                          rewriter, resultTy);
    rewriter.replaceOp(op, view);

    return success();
  }

private:
  int computeCapability;
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

  // An interface to support variant DestOp builder.
  SmallVector<DestOp> createDestOps(SourceOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    Type elemTy, MultipleOperandsRange operands,
                                    Location loc) const {
    return {rewriter.create<DestOp>(loc, elemTy, operands[0],
                                    adaptor.getAttributes().getValue())};
  }
};

// Attempts to use vectorized conversions via inline PTX when possible.
struct FpToFpOpConversion
    : public ElementwiseOpConversionBase<triton::FpToFpOp, FpToFpOpConversion> {
  using ElementwiseOpConversionBase<
      triton::FpToFpOp, FpToFpOpConversion>::ElementwiseOpConversionBase;

  explicit FpToFpOpConversion(TritonGPUToLLVMTypeConverter &typeConverter,
                              int computeCapability, PatternBenefit benefit = 1)
      : ElementwiseOpConversionBase(typeConverter, benefit),
        computeCapability(computeCapability) {}

  static Value convertBf16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
#ifdef USE_ROCM
    auto as_int16 = bitcast(v, i16_ty);
    auto as_int32 = zext(i32_ty, as_int16);
    auto shifted = shl(i32_ty, as_int32, i32_val(16));
    return(bitcast(shifted, f32_ty));
#else
    PTXBuilder builder;
    auto &cvt = *builder.create("cvt.f32.bf16");
    auto res = builder.newOperand("=r");
    auto operand = builder.newOperand(v, "h");
    cvt(res, operand);
    return builder.launch(rewriter, loc, f32_ty, false);
#endif
  }

  static Value convertFp16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
#ifdef USE_ROCM
    GCNBuilder builder;
    auto &cvt = *builder.create("v_cvt_f32_f16");
    auto res = builder.newOperand("=v");
    auto operand = builder.newOperand(v, "v");
    cvt(res, operand);
    return builder.launch(rewriter, loc, f32_ty, false);
#else
    PTXBuilder builder;
    auto &cvt = *builder.create("cvt.f32.f16");
    auto res = builder.newOperand("=r");
    auto operand = builder.newOperand(v, "h");
    cvt(res, operand);
    return builder.launch(rewriter, loc, f32_ty, false);
#endif
  }

  static Value convertFp32ToBf16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
#ifdef USE_ROCM
    auto as_uint32 = bitcast(v, i32_ty);
    auto check_exponent = and_(i32_ty, xor_(i32_ty, as_uint32, i32_val(0xffffffff)), i32_val(0x7f800000));
    auto exponent_not_all1s = icmp_ne(check_exponent, i32_val(0)); 
    auto exponent_all1s = icmp_eq(check_exponent, i32_val(0)); 
    auto rounded = add(i32_ty, i32_val(0x7fff),  and_(i32_ty, lshr(i32_ty, as_uint32, i32_val(16)), i32_val(1)) );
    rounded = add(i32_ty, rounded, as_uint32);
    auto res = select(exponent_not_all1s, rounded, as_uint32); 

    auto preserve_nan = and_( i1_ty, exponent_all1s, icmp_ne(and_(i32_ty, as_uint32, i32_val(0xffff)), i32_val(0)) );
    auto nan = or_(i32_ty, as_uint32, i32_val(0x10000));
    res = select(preserve_nan, nan, res); 

    auto shifted = lshr(i32_ty, res, i32_val(16));
    auto truncated = trunc(i16_ty, shifted);
    return truncated;
#else
    PTXBuilder builder;
    auto &cvt = *builder.create("cvt.rn.bf16.f32");
    auto res = builder.newOperand("=h");
    auto operand = builder.newOperand(v, "r");
    cvt(res, operand);
    // TODO: This is a hack to get the right type. We should be able to invoke
    // the type converter
    return builder.launch(rewriter, loc, i16_ty, false);
#endif
  }

  static Value convertFp32ToFp16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
#ifdef USE_ROCM
    GCNBuilder builder;
    auto &cvt = *builder.create("v_cvt_f16_f32");
    auto res = builder.newOperand("=v");
    auto operand = builder.newOperand(v, "v");
    cvt(res, operand);
    return builder.launch(rewriter, loc, f16_ty, false);
#else
    PTXBuilder builder;
    auto &cvt = *builder.create("cvt.rn.f16.f32");
    auto res = builder.newOperand("=h");
    auto operand = builder.newOperand(v, "r");
    cvt(res, operand);
    return builder.launch(rewriter, loc, f16_ty, false);
#endif
  }

  ConverterT getConversionFunc(Type srcTy, Type dstTy) const {
    auto F8E4M3B15TyID = TypeID::get<mlir::Float8E4M3B11FNUZType>();
    auto F8E4M3FNUZTyID = TypeID::get<mlir::Float8E4M3FNUZType>();
    auto F8E4M3FNTyID = TypeID::get<mlir::Float8E4M3FNType>();
    auto F8E5M2TyID = TypeID::get<mlir::Float8E5M2Type>();
    auto F8E5M2FNUZTyID = TypeID::get<mlir::Float8E5M2FNUZType>();
    auto F16TyID = TypeID::get<mlir::Float16Type>();
    auto BF16TyID = TypeID::get<mlir::BFloat16Type>();
    auto F32TyID = TypeID::get<mlir::Float32Type>();
    auto F64TyID = TypeID::get<mlir::Float64Type>();
#ifdef USE_ROCM
    static DenseMap<std::pair<TypeID, TypeID>, ConverterT> srcMap = {
#else
    static DenseMap<std::pair<TypeID, TypeID>, std::string> srcMap = {
#endif
        // F8 -> F16
        {{F8E4M3B15TyID, F16TyID}, Fp8E4M3B15_to_Fp16},
        {{F8E4M3FNTyID, F16TyID}, Fp8E4M3B15x4_to_Fp16},
        {{F8E4M3FNUZTyID, F16TyID}, Fp8E4M3FNUZ_to_Fp16},
#ifdef USE_ROCM
        {{F8E5M2TyID, F16TyID}, Fp8E5M2_to_Fp16},
        {{F8E5M2FNUZTyID, F16TyID}, Fp8E5M2FNUZ_to_Fp16},
#else
        {{F8E4M3TyID, F16TyID}, Fp8E4M3Nv_to_Fp16},
        {{F8E5M2TyID, F16TyID}, Fp8E5M2_to_Fp16(computeCapability >= 90)},
#endif
        // F16 -> F8
#ifdef USE_ROCM
        {{F16TyID, F8E4M3B15TyID}, Fp16_to_Fp8E4M3B15},
#else
        {{F16TyID, F8E4M3B15TyID}, Fp16_to_Fp8E4M3B15(computeCapability >= 80)},
#endif
	{{F16TyID, F8E4M3FNTyID}, Fp16_to_Fp8E4M3B15x4},
        {{F16TyID, F8E4M3FNUZTyID}, Fp16_to_Fp8E4M3FNUZ},
#ifdef USE_ROCM
        {{F16TyID, F8E5M2TyID}, Fp16_to_Fp8E5M2},
        {{F16TyID, F8E5M2FNUZTyID}, Fp16_to_Fp8E5M2FNUZ},
#else
        {{F16TyID, F8E4M3TyID}, Fp16_to_Fp8E4M3Nv},
        {{F16TyID, F8E5M2TyID}, Fp16_to_Fp8E5M2(computeCapability >= 90)},
#endif
	// F8 -> BF16
#ifdef USE_ROCM
	{{F8E5M2TyID, BF16TyID}, Fp8E5M2_to_Bf16},
#else
	{{F8E5M2TyID, BF16TyID}, Fp8E5M2_to_Bf16(computeCapability >= 90)},
        {{F8E4M3TyID, BF16TyID}, Fp8E4M3Nv_to_Bf16},
#endif
	// BF16 -> F8
#ifdef USE_ROCM
        {{BF16TyID, F8E5M2TyID}, Bf16_to_Fp8E5M2},
#else
        {{BF16TyID, F8E5M2TyID}, Bf16_to_Fp8E5M2(computeCapability >= 90)},
        {{BF16TyID, F8E4M3TyID}, Bf16_to_Fp8E4M3Nv},
        // F32 -> F8
        {{F32TyID, F8E4M3TyID}, Fp32_to_Fp8E4M3Nv},
        {{F32TyID, F8E5M2TyID}, Fp32_to_Fp8E5M2},
#endif
    };
    int inVecWidthBits = 32;
    int outVecWidthBits = 32;
    if (srcTy.isFloat8E4M3FNUZ() ||
        (computeCapability >= 90 && srcTy.isFloat8E5M2())) {
      inVecWidthBits = 16;
      outVecWidthBits = 32;
    }
    if (dstTy.isFloat8E4M3FNUZ() ||
        (computeCapability >= 90 && dstTy.isFloat8E5M2())) {
      inVecWidthBits = 32;
      outVecWidthBits = 16;
    }

    std::pair<TypeID, TypeID> key = {srcTy.getTypeID(), dstTy.getTypeID()};
    if (srcMap.count(key) == 0) {
      llvm::errs() << "Unsupported conversion from " << srcTy << " to " << dstTy
                   << "\n";
      llvm_unreachable("");
    }
#ifdef USE_ROCM
    return srcMap.lookup(key);
#else
    if (computeCapability < 90 &&
        (srcTy.isFloat8E4M3FNUZ() || dstTy.isFloat8E4M3FNUZ())) {
      llvm::errs() << "Conversion from/to f8e4m3nv is only supported on "
                      "compute capability >= 90"
                   << "\n";
      llvm_unreachable("");
    }
    return makeConverterFromPtx(srcMap.lookup(key),
                                getTypeConverter()->convertType(srcTy),
                                getTypeConverter()->convertType(dstTy),
                                inVecWidthBits, outVecWidthBits);
#endif
  }

  SmallVector<Value> createDestOps(triton::FpToFpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto srcElementType = getElementType(op.getFrom());
    auto dstElementType = getElementType(op.getResult());

    size_t numElements = 4;
    if (srcElementType.isFloat8E4M3FNUZ() ||
        dstElementType.isFloat8E4M3FNUZ() ||
        (computeCapability >= 90 &&
         (srcElementType.isFloat8E5M2() || dstElementType.isFloat8E5M2()))) {
      numElements = 2;
    }
    bool useFP16IntermediateSrc =
        srcElementType.isF32() &&
        !(computeCapability >= 90 &&
          (dstElementType.isFloat8E4M3FNUZ() || dstElementType.isFloat8E5M2()));
    bool isDstFP32 = dstElementType.isF32();
    auto cvtFunc =
        getConversionFunc(useFP16IntermediateSrc ? f16_ty : srcElementType,
                          isDstFP32 ? f16_ty : dstElementType);
    SmallVector<Value> inVals;
    for (unsigned i = 0; i < std::min(numElements, operands.size()); i++) {
      inVals.push_back(operands[i][0]);
    }
    if (useFP16IntermediateSrc)
      for (Value &v : inVals)
        v = convertFp32ToFp16(loc, rewriter, v);
    inVals.resize(numElements,
                  undef(typeConverter->convertType(srcElementType)));
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

template <typename OP>
Value EmitDualBF16ElementwiseOp(Location loc,
                                ConversionPatternRewriter &rewriter,
                                MultipleOperandsRange operands) {
  auto v0 = FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0][0]);
  auto v1 = FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0][1]);
  auto result = rewriter.create<OP>(loc, f32_ty, v0, v1);
  return FpToFpOpConversion::convertFp32ToBf16(loc, rewriter, result);
}

struct CmpIOpConversion
    : public ElementwiseOpConversionBase<triton::gpu::CmpIOp,
                                         CmpIOpConversion> {
  using Base =
      ElementwiseOpConversionBase<triton::gpu::CmpIOp, CmpIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  SmallVector<LLVM::ICmpOp>
  createDestOps(triton::gpu::CmpIOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter, Type elemTy,
                MultipleOperandsRange operands, Location loc) const {
    return {rewriter.create<LLVM::ICmpOp>(
        loc, elemTy, ArithCmpIPredicateToLLVM(op.getPredicate()),
        operands[0][0], operands[0][1])};
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
  static SmallVector<LLVM::FCmpOp>
  createDestOps(triton::gpu::CmpFOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter, Type elemTy,
                MultipleOperandsRange operands, Location loc) {
    return {rewriter.create<LLVM::FCmpOp>(
        loc, elemTy, ArithCmpFPredicateToLLVM(op.getPredicate()),
        operands[0][0], operands[0][1])};
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

struct ExternElementwiseOpConversion
    : public ElementwiseOpConversionBase<ExternElementwiseOp,
                                         ExternElementwiseOpConversion> {
  using Base = ElementwiseOpConversionBase<ExternElementwiseOp,
                                           ExternElementwiseOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;
  typedef typename Base::OpAdaptor OpAdaptor;

  SmallVector<Value> createDestOps(ExternElementwiseOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    StringRef funcName = op.getSymbol();
    if (funcName.empty())
      llvm::errs() << "ExternElementwiseOpConversion";

    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetFuncOp(rewriter, op, funcName, funcType);
    return {
        rewriter.create<LLVM::CallOp>(loc, funcOp, operands[0]).getResult()};
  }

private:
  Type getFunctionType(Type resultType, ValueRange operands) const {
    SmallVector<Type> operandTypes(operands.getTypes());
    return LLVM::LLVMFunctionType::get(resultType, operandTypes);
  }

  LLVM::LLVMFuncOp appendOrGetFuncOp(ConversionPatternRewriter &rewriter,
                                     ExternElementwiseOp op, StringRef funcName,
                                     Type funcType) const {
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

struct ElementwiseInlineAsmOpConversion
    : public ElementwiseOpConversionBase<ElementwiseInlineAsmOp,
                                         ElementwiseInlineAsmOpConversion> {
  using Base = ElementwiseOpConversionBase<ElementwiseInlineAsmOp,
                                           ElementwiseInlineAsmOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;
  typedef typename Base::OpAdaptor OpAdaptor;

  // If operand size is smaller than 32bits pack by groups of 32bits.
  // Otherwise have separate inputs.
  SmallVector<Value> packOperands(ElementwiseInlineAsmOp op,
                                  MultipleOperandsRange operands,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) const {
    SmallVector<Value> packedOperands;
    unsigned numPackedElements = op.getPackedElement();
    for (int i = 0, e = op.getNumOperands(); i < e; i++) {
      unsigned bitWidth =
          getElementType(op.getOperand(i)).getIntOrFloatBitWidth();
      unsigned numElementPerReg = bitWidth < 32 ? 32 / bitWidth : 1;
      numElementPerReg = std::min(numElementPerReg, numPackedElements);
      for (int j = 0; j < numPackedElements; j += numElementPerReg) {
        if (numElementPerReg == 1) {
          packedOperands.push_back(operands[j][i]);
          continue;
        }
        Type t = vec_ty(
            getTypeConverter()->convertType(getElementType(op.getOperand(i))),
            numElementPerReg);
        Value packed = undef(t);
        for (int k = 0; k < numElementPerReg; k++) {
          packed = insert_element(packed, operands[j + k][i], i32_val(k));
        }
        packedOperands.push_back(packed);
      }
    }
    return packedOperands;
  }

  SmallVector<Value> createDestOps(ElementwiseInlineAsmOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    int numPackedElements = op.getPackedElement();
    if (operands.size() % numPackedElements != 0)
      llvm::report_fatal_error("Inline asm op has more packed elements than "
                               "number of elements per thread.");
    SmallVector<Value> packedOperands =
        packOperands(op, operands, rewriter, loc);
    Type dstType =
        getTypeConverter()->convertType(getElementType(op.getResult()));
    Type retType = dstType;
    if (numPackedElements > 1)
      retType = vec_ty(retType, numPackedElements);
    Value result = rewriter
                       .create<LLVM::InlineAsmOp>(
                           loc, retType,
                           packedOperands,      // operands
                           op.getAsmString(),   // asm_string
                           op.getConstraints(), // constraints
                           !op.getPure(),       // has_side_effects
                           false,               // is_align_stack
                           LLVM::AsmDialectAttr::get(
                               rewriter.getContext(),
                               LLVM::AsmDialect::AD_ATT), // asm_dialect
                           ArrayAttr()                    // operand_attrs
                           )
                       ->getResult(0);
    SmallVector<Value> results;
    if (numPackedElements > 1) {
      for (int i = 0; i < numPackedElements; i++)
        results.push_back(extract_element(result, i32_val(i)));
    } else {
      results = {result};
    }
    return results;
  }
};

struct FDivOpConversion
    : ElementwiseOpConversionBase<mlir::arith::DivFOp, FDivOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::DivFOp, FDivOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::DivFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
#ifdef USE_ROCM
    return {rewriter.create<LLVM::FDivOp>(loc, elemTy, operands[0][0],
                                         operands[0][1])};
#else
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
    auto lhs =
        ptxBuilder.newOperand(operands[0][0], bitwidth == 32 ? "r" : "l");
    auto rhs =
        ptxBuilder.newOperand(operands[0][1], bitwidth == 32 ? "r" : "l");
    fdiv(res, lhs, rhs);

    Value ret = ptxBuilder.launch(rewriter, loc, elemTy, false);
    return {ret};
#endif
  }
};

struct FMulOpConversion
    : ElementwiseOpConversionBase<mlir::arith::MulFOp, FMulOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::MulFOp, FMulOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::MulFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
#ifdef USE_ROCM
      return {EmitDualBF16ElementwiseOp<LLVM::FMulOp>(loc, rewriter, operands)};
#else
      PTXBuilder builder;
      auto ptxAsm = " { .reg .b16 c;        \n"
                    "    mov.b16 c, 0x8000U; \n" // 0.0
                    "    fma.rn.bf16 $0, $1, $2, c; } \n";
      auto &fMul = *builder.create<PTXInstr>(ptxAsm);
      auto res = builder.newOperand("=h");
      auto lhs = builder.newOperand(operands[0][0], "h");
      auto rhs = builder.newOperand(operands[0][1], "h");
      fMul({res, lhs, rhs}, /*onlyAttachMLIRArgs=*/true);
      return {builder.launch(rewriter, loc, i16_ty, false)};
#endif
    } else {
      return {rewriter.create<LLVM::FMulOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};
    }
  }
};

struct FAddOpConversion
    : ElementwiseOpConversionBase<mlir::arith::AddFOp, FAddOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::AddFOp, FAddOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::AddFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
#ifdef USE_ROCM
      return {EmitDualBF16ElementwiseOp<LLVM::FAddOp>(loc, rewriter, operands)};
#else
      PTXBuilder builder;
      auto ptxAsm = "{ .reg .b16 c;         \n"
                    "   mov.b16 c, 0x3f80U; \n" // 1.0
                    "   fma.rn.bf16 $0, $1, c, $2; } \n";
      auto &fAdd = *builder.create<PTXInstr>(ptxAsm);
      auto res = builder.newOperand("=h");
      auto lhs = builder.newOperand(operands[0][0], "h");
      auto rhs = builder.newOperand(operands[0][1], "h");
      fAdd({res, lhs, rhs}, /*onlyAttachMLIRArgs=*/true);
      return {builder.launch(rewriter, loc, i16_ty, false)};
#endif
    } else {
      return {rewriter.create<LLVM::FAddOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};
    }
  }
};

struct FSubOpConversion
    : ElementwiseOpConversionBase<mlir::arith::SubFOp, FSubOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::SubFOp, FSubOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::SubFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
#ifdef USE_ROCM
      return {EmitDualBF16ElementwiseOp<LLVM::FSubOp>(loc, rewriter, operands)};
#else
      PTXBuilder builder;
      auto ptxAsm = " { .reg .b16 c;         \n"
                    "    mov.b16 c, 0xbf80U; \n" // -1.0
                    "    fma.rn.bf16 $0, $2, c, $1;} \n";
      auto &fSub = *builder.create<PTXInstr>(ptxAsm);
      auto res = builder.newOperand("=h");
      auto lhs = builder.newOperand(operands[0][0], "h");
      auto rhs = builder.newOperand(operands[0][1], "h");
      fSub({res, lhs, rhs}, /*onlyAttachMLIRArgs=*/true);
      return {builder.launch(rewriter, loc, i16_ty, false)};
#endif
    } else {
      return {rewriter.create<LLVM::FSubOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};
    }
  }
};

#ifdef USE_ROCM
static SmallVector<Value>
S8_to_Bf16(Location loc, ConversionPatternRewriter &rewriter,
		   const SmallVector<Value> &v) {
  SmallVector<Value> inValues = {v[0], v[1], v[2], v[3]};
  SmallVector<Value> outValues = {};
  for (Value inVal : inValues) {
    Value i32Val = sext(i32_ty, inVal);

    GCNBuilder builder;
    auto &cvt = *builder.create("v_cvt_f32_i32");
    auto res = builder.newOperand("=v");
    auto operand = builder.newOperand(i32Val, "v");
    cvt(res, operand);
    auto f32Val = builder.launch(rewriter, loc, f32_ty, false);

    f32Val = bitcast(f32Val, i32_ty);
    auto shifted = lshr(i32_ty, f32Val, i32_val(16));
    auto truncated = trunc(i16_ty, shifted);
    outValues.push_back(truncated);
  }
  return outValues;
}
#endif

// Uses inline ptx to convert s8/u8 to bf16, since the
struct SIToFPOpConversion
    : ElementwiseOpConversionBase<mlir::arith::SIToFPOp, SIToFPOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::SIToFPOp, SIToFPOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::SIToFPOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type inElemTy = getElementType(op.getIn());
    Type outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16() && inElemTy.isInteger(8) && operands.size() >= 4) {
      #if USE_ROCM
        SmallVector<Value> inVals = {operands[0][0], operands[1][0],
                                    operands[2][0], operands[3][0]};
        auto outVals = S8_to_Bf16(loc, rewriter, inVals);
      #else
        auto cvtFunc = makeConverterFromPtx(
            S8_to_Bf16, getTypeConverter()->convertType(inElemTy),
            getTypeConverter()->convertType(outElemTy));
        SmallVector<Value> inVals = {operands[0][0], operands[1][0],
                                    operands[2][0], operands[3][0]};
        auto outVals = cvtFunc(loc, rewriter, inVals);
      #endif
      assert(outVals.size() == 4);
      return outVals;
    } else if (outElemTy.isBF16()) {
      auto value = rewriter.create<LLVM::SIToFPOp>(loc, f32_ty, operands[0][0]);
      return {FpToFpOpConversion::convertFp32ToBf16(loc, rewriter, value)};
    } else {
      return {rewriter.create<LLVM::SIToFPOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct FPToSIOpConversion
    : ElementwiseOpConversionBase<mlir::arith::FPToSIOp, FPToSIOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::FPToSIOp, FPToSIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::FPToSIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto value =
          FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0][0]);
      return {rewriter.create<LLVM::FPToSIOp>(loc, elemTy, value)};
    } else {
      return {rewriter.create<LLVM::FPToSIOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct ExtFOpConversion
    : ElementwiseOpConversionBase<mlir::arith::ExtFOp, ExtFOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::ExtFOp, ExtFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::ExtFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto outElemTy = getElementType(op.getOut());
      assert(outElemTy.isF32() && "unsupported conversion");
      return {
          FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0][0])};
    } else {
      return {rewriter.create<LLVM::FPExtOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct TruncFOpConversion
    : ElementwiseOpConversionBase<mlir::arith::TruncFOp, TruncFOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::TruncFOp, TruncFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::TruncFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16()) {
      auto inElemTy = getElementType(op.getIn());
      assert(inElemTy.isF32() && "unsupported conversion");
      return {
          FpToFpOpConversion::convertFp32ToBf16(loc, rewriter, operands[0][0])};
    } else {
      return {rewriter.create<LLVM::FPTruncOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct ExpOpConversionApprox
    : ElementwiseOpConversionBase<mlir::math::ExpOp, ExpOpConversionApprox> {
  using Base =
      ElementwiseOpConversionBase<mlir::math::ExpOp, ExpOpConversionApprox>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::math::ExpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // For non-FP32 input, call __nv_expf for higher-precision calculation
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    const double log2e = 1.4426950408889634;
    Value prod = fmul(f32_ty, operands[0][0], f32_val(log2e));

#ifdef USE_ROCM
    return {rewriter.create<math::Exp2Op>(loc, f32_ty, prod,
                                         adaptor.getAttributes().getValue())};
#else
    PTXBuilder ptxBuilder;
    auto &exp2 = ptxBuilder.create<PTXInstr>("ex2")->o("approx").o("f32");
    auto output = ptxBuilder.newOperand("=f");
    auto input = ptxBuilder.newOperand(prod, "f");
    exp2(output, input);
    return {ptxBuilder.launch(rewriter, loc, f32_ty, false)};
#endif
  }
};

struct AbsIOpConversion
    : ElementwiseOpConversionBase<mlir::math::AbsIOp, AbsIOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::math::AbsIOp, AbsIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::math::AbsIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto boolFalse = rewriter.getBoolAttr(false);
    auto constFalse = rewriter.create<LLVM::ConstantOp>(loc, boolFalse);
    return {rewriter.create<LLVM::AbsOp>(loc, elemTy, operands[0][0],
                                         /*is_int_min_poison=*/constFalse)};
  }
};

struct AbsFOpConversion
    : ElementwiseOpConversionBase<mlir::math::AbsFOp, AbsFOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::math::AbsFOp, AbsFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::math::AbsFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (llvm::isa<IntegerType>(elemTy)) {
      // Mask out the sign bit
      auto num_bits =
          getElementTypeOrSelf(op.getType()).getIntOrFloatBitWidth();
      assert(num_bits <= 16);
      auto mask = (1u << (num_bits - 1u)) - 1u;
      auto maskAttr = rewriter.getIntegerAttr(elemTy, mask);
      auto maskConst = rewriter.create<LLVM::ConstantOp>(loc, maskAttr);
      return {and_(operands[0][0], maskConst)};
    }

    return {rewriter.create<LLVM::FAbsOp>(loc, elemTy, operands[0][0])};
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

  SmallVector<Value> createDestOps(arith::IndexCastOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy =
        this->getTypeConverter()->convertType(getElementType(op.getIn()));
    unsigned targetBits = elemTy.getIntOrFloatBitWidth();
    unsigned sourceBits = inElemTy.getIntOrFloatBitWidth();

    if (targetBits == sourceBits)
      return {operands[0][0]};
    if (targetBits < sourceBits)
      return {rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, elemTy,
                                                         operands[0][0])};
    return {
        rewriter.replaceOpWithNewOp<LLVM::SExtOp>(op, elemTy, operands[0][0])};
  }
};

void populateElementwiseOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    int computeCapability, PatternBenefit benefit) {
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
  POPULATE_BINARY_OP(arith::AndIOp, LLVM::AndOp)        // &
  POPULATE_BINARY_OP(arith::OrIOp, LLVM::OrOp)          // |
  POPULATE_BINARY_OP(arith::XOrIOp, LLVM::XOrOp)        // ^
  POPULATE_BINARY_OP(arith::ShLIOp, LLVM::ShlOp)        // <<
  POPULATE_BINARY_OP(arith::ShRSIOp, LLVM::AShrOp)      // >>
  POPULATE_BINARY_OP(arith::ShRUIOp, LLVM::LShrOp)      // >>
  POPULATE_BINARY_OP(arith::MinimumFOp, LLVM::MinNumOp) // fmin
  POPULATE_BINARY_OP(arith::MaximumFOp, LLVM::MaxNumOp) // fmax
  POPULATE_BINARY_OP(arith::MinSIOp, LLVM::SMinOp)      // smin
  POPULATE_BINARY_OP(arith::MaxSIOp, LLVM::SMaxOp)      // smax
  POPULATE_BINARY_OP(arith::MinUIOp, LLVM::UMinOp)      // umin
  POPULATE_BINARY_OP(arith::MaxUIOp, LLVM::UMaxOp)      // umax
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

  patterns.add<FpToFpOpConversion>(typeConverter, computeCapability, benefit);

  patterns.add<ExternElementwiseOpConversion>(typeConverter, benefit);
  patterns.add<ElementwiseInlineAsmOpConversion>(typeConverter, benefit);
  // ExpOpConversionApprox will try using ex2.approx if the input type is
  // FP32. For other input types, ExpOpConversionApprox will return failure and
  // ElementwiseOpConversion<math::ExpOp, math::ExpOp> defined below will call
  // __nv_expf for higher-precision calculation
  patterns.add<ExpOpConversionApprox>(typeConverter, benefit);
}
