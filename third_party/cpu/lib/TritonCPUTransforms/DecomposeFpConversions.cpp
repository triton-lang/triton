#include "cpu/include/TritonCPUTransforms/OptCommon.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_DECOMPOSEFPCONVERSIONS
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

struct Fp32ToBf16Conversion : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getIn();
    if (!isBf16(op.getType()) || !isFp32(src.getType()))
      return failure();

    Location loc = op.getLoc();
    Value i32Src = op_bitcast(toInt32(src.getType()), src);
    Value shiftedSrc = op_lshr(i32Src, cst_like(i32Src, 16));
    Value i16Res = op_trunci(toInt16(src.getType()), shiftedSrc);
    Value res = op_bitcast(op.getType(), i16Res);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct Bf16ToFp32Conversion : public OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtFOp op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getIn();
    if (!isFp32(op.getType()) || !isBf16(src.getType()))
      return failure();

    Location loc = op.getLoc();
    Value i16Src = op_bitcast(toInt16(src.getType()), src);
    Value i32Src = op_zext(toInt32(src.getType()), i16Src);
    Value i32Res = op_shl(i32Src, cst_like(i32Src, 16));
    Value res = op_bitcast(op.getType(), i32Res);
    rewriter.replaceOp(op, res);
    return success();
  }
};

typedef std::function<Value(Location, Value, PatternRewriter &)> FpToFpConvFn;

// Convert FP8 to FP16/FP32.
Value convertFp8(Location loc, Value src, int srcExpBits, int srcExpBias,
                 Type dstFpTy, PatternRewriter &rewriter) {
  assert(srcExpBits >= 4 && srcExpBits <= 5 && "Unexpect FP8 type conversion");
  assert(srcExpBias >= 0 && srcExpBias <= 16 && "Unexpect FP8 type conversion");
  assert((dstFpTy.isF16() || dstFpTy.isF32()) &&
         "Unsupported FP8 type conversion");
  Type srcTy = src.getType();
  Type dstTy = toTyOrVectorOf(srcTy, dstFpTy);
  int dstExpBits = dstFpTy.isF16() ? 5 : 8;
  int dstMantBits = dstFpTy.isF16() ? 10 : 23;
  int dstExpBias = dstFpTy.isF16() ? 15 : 127;
  int srcMantBits = 7 - srcExpBits;
  assert(dstExpBias >= srcExpBias && "Unsupported FP8 type conversion");
  Type dstIntTy =
      dstFpTy.isF16() ? rewriter.getI16Type() : rewriter.getI32Type();
  Value i8Src = op_bitcast(toInt8(srcTy), src);
  Value intSrc = op_zext(toTyOrVectorOf(srcTy, dstIntTy), i8Src);
  Value shiftedVal;
  if (srcExpBits != dstExpBits) {
    Value sign = op_and(intSrc, cst_like(intSrc, 0x80));
    Value nosign = op_and(intSrc, cst_like(intSrc, 0x7f));
    shiftedVal = op_addi(
        op_shl(sign, cst_like(sign, dstFpTy.getIntOrFloatBitWidth() - 8)),
        op_shl(nosign, cst_like(nosign, dstMantBits - srcMantBits)));
  } else {
    shiftedVal =
        op_shl(intSrc, cst_like(intSrc, dstFpTy.getIntOrFloatBitWidth() - 8));
  }
  Value res = op_bitcast(dstTy, shiftedVal);
  if (srcExpBias != dstExpBias) {
    double scale = pow(2, dstExpBias - srcExpBias);
    res = op_mulf(res, cst_like(res, scale));
  }
  return res;
}

Value convertFp8E4M3ToFp16(Location loc, Value src, PatternRewriter &rewriter) {
  return convertFp8(loc, src, 4, 7, rewriter.getF16Type(), rewriter);
}

Value convertFp8E5M2ToFp16(Location loc, Value src, PatternRewriter &rewriter) {
  return convertFp8(loc, src, 5, 15, rewriter.getF16Type(), rewriter);
}

Value convertFp8E5M2B16ToFp16(Location loc, Value src,
                              PatternRewriter &rewriter) {
  Value f32Res = convertFp8(loc, src, 5, 16, rewriter.getF32Type(), rewriter);
  return rewriter.create<arith::TruncFOp>(loc, toFp16(src.getType()), f32Res);
}

Value convertFp8E4M3ToBf16(Location loc, Value src, PatternRewriter &rewriter) {
  Value f32Res = convertFp8(loc, src, 4, 7, rewriter.getF32Type(), rewriter);
  return rewriter.create<arith::TruncFOp>(loc, toBf16(src.getType()), f32Res);
}

Value convertFp8E5M2ToBf16(Location loc, Value src, PatternRewriter &rewriter) {
  Value f32Res = convertFp8(loc, src, 5, 15, rewriter.getF32Type(), rewriter);
  return rewriter.create<arith::TruncFOp>(loc, toBf16(src.getType()), f32Res);
}

Value convertFp8E5M2B16ToBf16(Location loc, Value src,
                              PatternRewriter &rewriter) {
  Value f32Res = convertFp8(loc, src, 5, 16, rewriter.getF32Type(), rewriter);
  return rewriter.create<arith::TruncFOp>(loc, toBf16(src.getType()), f32Res);
}

Value convertFp8E4M3ToFp32(Location loc, Value src, PatternRewriter &rewriter) {
  return convertFp8(loc, src, 4, 7, rewriter.getF32Type(), rewriter);
}

Value convertFp8E5M2ToFp32(Location loc, Value src, PatternRewriter &rewriter) {
  return convertFp8(loc, src, 5, 15, rewriter.getF32Type(), rewriter);
}

Value convertFp8E5M2B16ToFp32(Location loc, Value src,
                              PatternRewriter &rewriter) {
  return convertFp8(loc, src, 5, 16, rewriter.getF32Type(), rewriter);
}

// Convert F16/FP32 to FP8.
Value convertToFp8(Location loc, Value src, Type dstFpTy, int dstExpBits,
                   int dstExpBias, bool rtneRounding, bool unsignedZero,
                   PatternRewriter &rewriter) {
  assert(dstExpBits >= 4 && dstExpBits <= 5 && "Unexpect FP8 type conversion");
  assert(dstExpBias >= 0 && dstExpBias <= 16 && "Unexpect FP8 type conversion");
  Type srcTy = src.getType();
  Type srcFpTy = getElemTyOrTy(srcTy);
  assert((srcFpTy.isF16() || srcFpTy.isF32()) &&
         "Unsupported FP8 type conversion");
  int dstMantBits = 7 - dstExpBits;
  int srcExpBits = srcFpTy.isF16() ? 5 : 8;
  int srcMantBits = srcFpTy.isF16() ? 10 : 23;
  int srcExpBias = srcFpTy.isF16() ? 15 : 127;
  assert(dstExpBias <= srcExpBias && "Unsupported FP8 type conversion");
  Type srcIntTy =
      srcFpTy.isF16() ? rewriter.getI16Type() : rewriter.getI32Type();
  Value intSrc = op_bitcast(toTyOrVectorOf(srcTy, srcIntTy), src);
  // Extract sign and put it to the proper place for FP8.
  Value sign =
      op_lshr(op_and(intSrc, cst_like(intSrc, 1 << (srcExpBits + srcMantBits))),
              cst_like(intSrc, srcFpTy.getIntOrFloatBitWidth() - 8));
  // Extract mantissa and exponent.
  Value mant = op_and(intSrc, cst_like(intSrc, (1 << srcMantBits) - 1));
  Value exp = op_and(op_lshr(intSrc, cst_like(intSrc, srcMantBits)),
                     cst_like(intSrc, (1 << srcExpBits) - 1));
  Value isZeroExp = op_icmp_eq(exp, cst_like(exp, 0));
  mant = op_select(isZeroExp, mant,
                   op_addi(mant, cst_like(mant, 1 << srcMantBits)));
  exp = op_select(isZeroExp, exp, op_subi(exp, cst_like(exp, 1)));
  double adjustment = pow(0.5, srcMantBits - dstMantBits);
  exp = op_subi(exp, cst_like(exp, srcExpBias - dstExpBias));
  mant = op_mulf(op_sitofp(srcTy, mant), cst_like(src, adjustment));
  // Make exponent non-negative.
  if (dstExpBias - srcExpBias <= -8) {
    // In this case we don't have enough mantissa bits, so can round to 0.
    Value mask = op_icmp_sgt(exp, cst_like(exp, -8));
    exp = op_select(mask, exp, cst_like(exp, 0));
    mant = op_select(mask, mant, cst_like(mant, 0.0));
  }
  if (dstExpBias - srcExpBias <= -4) {
    Value mask = op_icmp_sgt(exp, cst_like(exp, -4));
    exp = op_select(mask, exp, op_addi(exp, cst_like(exp, 4)));
    mant = op_select(mask, mant, op_mulf(mant, cst_like(mant, 0.0625)));
  }
  if (dstExpBias - srcExpBias <= -2) {
    Value mask = op_icmp_sgt(exp, cst_like(exp, -2));
    exp = op_select(mask, exp, op_addi(exp, cst_like(exp, 2)));
    mant = op_select(mask, mant, op_mulf(mant, cst_like(mant, 0.25)));
  }
  if (dstExpBias - srcExpBias <= -1) {
    Value mask = op_icmp_sgt(exp, cst_like(exp, -1));
    exp = op_select(mask, exp, op_addi(exp, cst_like(exp, 1)));
    mant = op_select(mask, mant, op_mulf(mant, cst_like(mant, 0.5)));
  }
  if (rtneRounding) {
    // Bring the value to the range [2 ** 10/23, 2 ** 11/24]
    // where the representable fp16/fp32 map exactly to integers.
    // Addition has RTNE semantics.
    Value offs = cst_like(mant, static_cast<double>(1 << srcMantBits));
    mant = op_addf(mant, offs);
    mant = op_subf(mant, offs);
  }
  mant = op_fptosi(toTyOrVectorOf(srcTy, srcIntTy), mant);

  Value res =
      op_addi(sign, op_addi(op_shl(exp, cst_like(exp, 7 - dstExpBits)), mant));
  res = op_trunci(toInt8(srcTy), res);
  if (unsignedZero) {
    Value isNegativeZero = op_icmp_eq(res, cst_like(res, 0x80));
    res = op_select(isNegativeZero, cst_like(res, 0), res);
  }
  res = op_bitcast(toTyOrVectorOf(srcTy, dstFpTy), res);
  return res;
}

Value convertFp16ToFp8E4M3Rtz(Location loc, Value src,
                              PatternRewriter &rewriter) {
  return convertToFp8(loc, src, rewriter.getFloat8E4M3FNType(), 4, 7, false,
                      false, rewriter);
}

Value convertFp16ToFp8E4M3Rtne(Location loc, Value src,
                               PatternRewriter &rewriter) {
  return convertToFp8(loc, src, rewriter.getFloat8E4M3FNType(), 4, 7, true,
                      false, rewriter);
}

Value convertFp16ToFp8E5M2Rtz(Location loc, Value src,
                              PatternRewriter &rewriter) {
  Type srcTy = src.getType();
  Type dstTy = toFp8E5M2(srcTy);
  Value i16Src = op_bitcast(toInt16(srcTy), src);
  Value shiftedSrc = op_lshr(i16Src, cst_like(i16Src, 8));
  Value i8Res = op_trunci(toInt8(srcTy), shiftedSrc);
  Value res = op_bitcast(dstTy, i8Res);
  return res;
}

Value convertFp16ToFp8E5M2Rtne(Location loc, Value src,
                               PatternRewriter &rewriter) {
  Type srcTy = src.getType();
  Type dstTy = toFp8E5M2(srcTy);
  Value i16Src = op_bitcast(toInt16(srcTy), src);
  Value sign = op_and(i16Src, cst_like(i16Src, 0x8000));
  Value truncated = op_and(i16Src, cst_like(i16Src, 0x7f00));
  Value tail = op_and(i16Src, cst_like(i16Src, 0xff));
  Value odd_trunc = op_icmp_ne(op_and(truncated, cst_like(truncated, 0x100)),
                               cst_like(truncated, 0));
  Value round_up =
      op_or(op_icmp_ugt(tail, cst_like(tail, 0x80)),
            op_and(op_icmp_eq(tail, cst_like(tail, 0x80)), odd_trunc));
  // Skip round-up if it leads to inf/nan.
  round_up =
      op_and(round_up, op_icmp_ult(truncated, cst_like(truncated, 0x7b00)));
  truncated = op_select(
      round_up, op_addi(truncated, cst_like(truncated, 0x100)), truncated);

  Value res = op_lshr(op_or(truncated, sign), cst_like(truncated, 8));
  res = op_bitcast(dstTy, op_trunci(toInt8(srcTy), res));
  return res;
}

Value convertFp16ToFp8E5M2B16Rtz(Location loc, Value src,
                                 PatternRewriter &rewriter) {
  Value f32Src =
      rewriter.create<arith::ExtFOp>(loc, toFp32(src.getType()), src);
  return convertToFp8(loc, f32Src, rewriter.getFloat8E5M2FNUZType(), 5, 16,
                      false, true, rewriter);
}

Value convertFp16ToFp8E5M2B16Rtne(Location loc, Value src,
                                  PatternRewriter &rewriter) {
  Value f32Src =
      rewriter.create<arith::ExtFOp>(loc, toFp32(src.getType()), src);
  return convertToFp8(loc, f32Src, rewriter.getFloat8E5M2FNUZType(), 5, 16,
                      true, true, rewriter);
}

Value convertBf16ToFp8E4M3Rtz(Location loc, Value src,
                              PatternRewriter &rewriter) {
  Value f32Src =
      rewriter.create<arith::ExtFOp>(loc, toFp32(src.getType()), src);
  return convertToFp8(loc, f32Src, rewriter.getFloat8E4M3FNType(), 4, 7, false,
                      false, rewriter);
}

Value convertBf16ToFp8E4M3Rtne(Location loc, Value src,
                               PatternRewriter &rewriter) {
  Value f32Src =
      rewriter.create<arith::ExtFOp>(loc, toFp32(src.getType()), src);
  return convertToFp8(loc, f32Src, rewriter.getFloat8E4M3FNType(), 4, 7, true,
                      false, rewriter);
}

Value convertBf16ToFp8E5M2Rtz(Location loc, Value src,
                              PatternRewriter &rewriter) {
  Value f32Src =
      rewriter.create<arith::ExtFOp>(loc, toFp32(src.getType()), src);
  return convertToFp8(loc, f32Src, rewriter.getFloat8E5M2Type(), 5, 15, false,
                      false, rewriter);
}

Value convertBf16ToFp8E5M2Rtne(Location loc, Value src,
                               PatternRewriter &rewriter) {
  Value f32Src =
      rewriter.create<arith::ExtFOp>(loc, toFp32(src.getType()), src);
  return convertToFp8(loc, f32Src, rewriter.getFloat8E5M2Type(), 5, 15, true,
                      false, rewriter);
}

Value convertBf16ToFp8E5M2B16Rtz(Location loc, Value src,
                                 PatternRewriter &rewriter) {
  Value f32Src =
      rewriter.create<arith::ExtFOp>(loc, toFp32(src.getType()), src);
  return convertToFp8(loc, f32Src, rewriter.getFloat8E5M2FNUZType(), 5, 16,
                      false, true, rewriter);
}

Value convertBf16ToFp8E5M2B16Rtne(Location loc, Value src,
                                  PatternRewriter &rewriter) {
  Value f32Src =
      rewriter.create<arith::ExtFOp>(loc, toFp32(src.getType()), src);
  return convertToFp8(loc, f32Src, rewriter.getFloat8E5M2FNUZType(), 5, 16,
                      true, true, rewriter);
}

Value convertFp32ToFp8E4M3Rtz(Location loc, Value src,
                              PatternRewriter &rewriter) {
  return convertToFp8(loc, src, rewriter.getFloat8E4M3FNType(), 4, 7, false,
                      false, rewriter);
}

Value convertFp32ToFp8E4M3Rtne(Location loc, Value src,
                               PatternRewriter &rewriter) {
  return convertToFp8(loc, src, rewriter.getFloat8E4M3FNType(), 4, 7, true,
                      false, rewriter);
}

Value convertFp32ToFp8E5M2Rtz(Location loc, Value src,
                              PatternRewriter &rewriter) {
  return convertToFp8(loc, src, rewriter.getFloat8E5M2Type(), 5, 15, false,
                      false, rewriter);
}

Value convertFp32ToFp8E5M2Rtne(Location loc, Value src,
                               PatternRewriter &rewriter) {
  return convertToFp8(loc, src, rewriter.getFloat8E5M2Type(), 5, 15, true,
                      false, rewriter);
}

Value convertFp32ToFp8E5M2B16Rtz(Location loc, Value src,
                                 PatternRewriter &rewriter) {
  return convertToFp8(loc, src, rewriter.getFloat8E5M2FNUZType(), 5, 16, false,
                      true, rewriter);
}

Value convertFp32ToFp8E5M2B16Rtne(Location loc, Value src,
                                  PatternRewriter &rewriter) {
  return convertToFp8(loc, src, rewriter.getFloat8E5M2FNUZType(), 5, 16, true,
                      true, rewriter);
}

FpToFpConvFn
getFpToFpConversionFn(Type srcTy, Type dstTy,
                      std::optional<arith::RoundingMode> roundMode) {
  auto F8E4M3TyID = TypeID::get<Float8E4M3FNType>();
  auto F8E5M2TyID = TypeID::get<Float8E5M2Type>();
  auto F8E5M2B16TyID = TypeID::get<Float8E5M2FNUZType>();
  auto F16TyID = TypeID::get<Float16Type>();
  auto BF16TyID = TypeID::get<BFloat16Type>();
  auto F32TyID = TypeID::get<Float32Type>();

  static DenseMap<std::tuple<TypeID, TypeID>, FpToFpConvFn> fpExtFnMap = {
      {{F8E4M3TyID, F16TyID}, convertFp8E4M3ToFp16},
      {{F8E5M2TyID, F16TyID}, convertFp8E5M2ToFp16},
      {{F8E5M2B16TyID, F16TyID}, convertFp8E5M2B16ToFp16},
      {{F8E4M3TyID, BF16TyID}, convertFp8E4M3ToBf16},
      {{F8E5M2TyID, BF16TyID}, convertFp8E5M2ToBf16},
      {{F8E5M2B16TyID, BF16TyID}, convertFp8E5M2B16ToBf16},
      {{F8E4M3TyID, F32TyID}, convertFp8E4M3ToFp32},
      {{F8E5M2TyID, F32TyID}, convertFp8E5M2ToFp32},
      {{F8E5M2B16TyID, F32TyID}, convertFp8E5M2B16ToFp32},
  };
  static DenseMap<std::tuple<TypeID, TypeID, arith::RoundingMode>, FpToFpConvFn>
      fpTruncFnMap = {
          {{F16TyID, F8E4M3TyID, arith::RoundingMode::toward_zero},
           convertFp16ToFp8E4M3Rtz},
          {{F16TyID, F8E4M3TyID, arith::RoundingMode::to_nearest_even},
           convertFp16ToFp8E4M3Rtne},
          {{F16TyID, F8E5M2TyID, arith::RoundingMode::toward_zero},
           convertFp16ToFp8E5M2Rtz},
          {{F16TyID, F8E5M2TyID, arith::RoundingMode::to_nearest_even},
           convertFp16ToFp8E5M2Rtne},
          {{F16TyID, F8E5M2B16TyID, arith::RoundingMode::toward_zero},
           convertFp16ToFp8E5M2B16Rtz},
          {{F16TyID, F8E5M2B16TyID, arith::RoundingMode::to_nearest_even},
           convertFp16ToFp8E5M2B16Rtne},
          {{BF16TyID, F8E4M3TyID, arith::RoundingMode::toward_zero},
           convertBf16ToFp8E4M3Rtz},
          {{BF16TyID, F8E4M3TyID, arith::RoundingMode::to_nearest_even},
           convertBf16ToFp8E4M3Rtne},
          {{BF16TyID, F8E5M2TyID, arith::RoundingMode::toward_zero},
           convertBf16ToFp8E5M2Rtz},
          {{BF16TyID, F8E5M2TyID, arith::RoundingMode::to_nearest_even},
           convertBf16ToFp8E5M2Rtne},
          {{BF16TyID, F8E5M2B16TyID, arith::RoundingMode::toward_zero},
           convertBf16ToFp8E5M2B16Rtz},
          {{BF16TyID, F8E5M2B16TyID, arith::RoundingMode::to_nearest_even},
           convertBf16ToFp8E5M2B16Rtne},
          {{F32TyID, F8E4M3TyID, arith::RoundingMode::toward_zero},
           convertFp32ToFp8E4M3Rtz},
          {{F32TyID, F8E4M3TyID, arith::RoundingMode::to_nearest_even},
           convertFp32ToFp8E4M3Rtne},
          {{F32TyID, F8E5M2TyID, arith::RoundingMode::toward_zero},
           convertFp32ToFp8E5M2Rtz},
          {{F32TyID, F8E5M2TyID, arith::RoundingMode::to_nearest_even},
           convertFp32ToFp8E5M2Rtne},
          {{F32TyID, F8E5M2B16TyID, arith::RoundingMode::toward_zero},
           convertFp32ToFp8E5M2B16Rtz},
          {{F32TyID, F8E5M2B16TyID, arith::RoundingMode::to_nearest_even},
           convertFp32ToFp8E5M2B16Rtne},
      };

  if (roundMode) {
    auto key =
        std::make_tuple(srcTy.getTypeID(), dstTy.getTypeID(), *roundMode);
    if (fpTruncFnMap.count(key))
      return fpTruncFnMap.at(key);
  } else {
    auto key = std::make_tuple(srcTy.getTypeID(), dstTy.getTypeID());
    if (fpExtFnMap.count(key))
      return fpExtFnMap.at(key);
  }

  return FpToFpConvFn();
}

Value convertFpToFp(Location loc, Value src, Type dstTy,
                    std::optional<arith::RoundingMode> roundMode,
                    PatternRewriter &rewriter) {
  Type srcTy = src.getType();
  Type srcElemTy = getElemTyOrTy(srcTy);
  Type dstElemTy = getElemTyOrTy(dstTy);
  auto fn = getFpToFpConversionFn(srcElemTy, dstElemTy, roundMode);
  if (!fn) {
    llvm::errs() << "Unsupported conversion from " << srcElemTy << " to "
                 << dstElemTy;
    if (roundMode)
      llvm::errs() << " with rounding mode "
                   << arith::stringifyRoundingMode(*roundMode);
    llvm::errs() << "\n";
    llvm_unreachable("");
  }
  return fn(loc, src, rewriter);
}

struct RewriteTruncFp8 : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src = op.getIn();
    Type srcTy = src.getType();
    Type dstTy = op.getType();
    if (!isFp8(dstTy))
      return failure();
    Value res = convertFpToFp(loc, src, dstTy, op.getRoundingmode(), rewriter);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct RewriteExtFp8 : public OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtFOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src = op.getIn();
    Type srcTy = src.getType();
    if (!isFp8(srcTy))
      return failure();
    Type dstTy = op.getType();
    Value res = convertFpToFp(loc, src, dstTy, std::nullopt, rewriter);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct DecomposeFpConversions
    : public triton::cpu::impl::DecomposeFpConversionsBase<
          DecomposeFpConversions> {
  DecomposeFpConversions() = default;

  DecomposeFpConversions(bool decomposeBf16Conversions,
                         bool decomposeFp8Conversions) {
    this->decomposeBf16Conversions = decomposeBf16Conversions;
    this->decomposeFp8Conversions = decomposeFp8Conversions;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    RewritePatternSet patterns(context);
    if (decomposeBf16Conversions) {
      patterns.add<Fp32ToBf16Conversion>(context);
      patterns.add<Bf16ToFp32Conversion>(context);
    }
    if (decomposeFp8Conversions) {
      patterns.add<RewriteTruncFp8>(context);
      patterns.add<RewriteExtFp8>(context);
    }

    if (failed(mlir::applyPatternsAndFoldGreedily(mod, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createDecomposeFpConversions() {
  return std::make_unique<DecomposeFpConversions>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createDecomposeFpConversions(bool decomposeBf16Conversions,
                             bool decomposeFp8Conversions) {
  return std::make_unique<DecomposeFpConversions>(decomposeBf16Conversions,
                                                  decomposeFp8Conversions);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
