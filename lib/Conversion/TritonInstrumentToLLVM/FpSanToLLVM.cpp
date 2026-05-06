#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/bit.h"
#include <cassert>

using namespace mlir;
using namespace mlir::triton;

namespace {

namespace tti = mlir::triton::instrument;

uint64_t invOddU64(uint64_t a) {
  assert((a & 1) == 1);
  uint64_t x = 2 - a;
  for (unsigned correctBits = 2; correctBits < 64; correctBits *= 2)
    x *= 2 - a * x;
  return x;
}

uint64_t getOneBitPattern(FloatType floatTy) {
  llvm::APFloat one(1.0);
  bool losesInfo = false;
  one.convert(floatTy.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven,
              &losesInfo);
  return one.bitcastToAPInt().getZExtValue();
}

struct PayloadMixConfig {
  unsigned bitWidth;
  unsigned shift;
  uint64_t signMask;
  uint64_t magMask;
  uint64_t mulA;
  uint64_t mulAInv;
  uint64_t mulBPos;
  uint64_t mulBNeg;
  uint64_t mulBPosInv;
  uint64_t mulBNegInv;
};

PayloadMixConfig getPayloadMixConfig(FloatType floatTy) {
  unsigned bitWidth = floatTy.getWidth();
  assert(bitWidth > 1 && bitWidth <= 64);
  uint64_t signMask = uint64_t{1} << (bitWidth - 1);
  uint64_t magMask = signMask - 1;

  uint64_t oneBits = getOneBitPattern(floatTy);
  assert(oneBits != 0 && "expected non-zero 1.0 bit pattern");
  unsigned shift = llvm::countr_zero(oneBits);
  assert(shift != 0 && "expected even 1.0 bit pattern");

  // we firstly multiply by an arbitrary odd constant to mix from low
  // bits to high whilst remaining invertible:
  uint64_t mulA = 922291u & magMask;
  uint64_t oneMixed = (oneBits * mulA) & magMask;
  oneMixed ^= oneMixed >> shift;
  assert((oneMixed & 1) == 1 && "expected odd mixed 1.0");

  // the second multiplier is chosen so that the entire payload mixing
  // operation maps the float 1.0 to the integer 1:
  uint64_t mulBPos = invOddU64(oneMixed) & magMask;
  uint64_t mulBNeg = (mulBPos * magMask) & magMask;
  return PayloadMixConfig{
      bitWidth,
      shift,
      signMask,
      magMask,
      mulA,
      invOddU64(mulA) & magMask,
      mulBPos,
      mulBNeg,
      invOddU64(mulBPos) & magMask,
      invOddU64(mulBNeg) & magMask,
  };
}

Value createUIntConstant(ConversionPatternRewriter &rewriter, Location loc,
                         Type intTy, uint64_t value) {
  auto intType = cast<IntegerType>(intTy);
  auto attr = IntegerAttr::get(intType, llvm::APInt(intType.getWidth(), value));
  return LLVM::ConstantOp::create(rewriter, loc, intType, attr);
}

Value selectUIntConstantOnSign(ConversionPatternRewriter &rewriter,
                               Location loc, Value signSource,
                               uint64_t signMaskValue,
                               uint64_t nonNegativeValue,
                               uint64_t negativeValue) {
  TritonLLVMOpBuilder b(loc, rewriter);
  auto intTy = signSource.getType();
  Value signMask = createUIntConstant(rewriter, loc, intTy, signMaskValue);
  Value zero = createUIntConstant(rewriter, loc, intTy, 0u);
  Value sign = b.and_(signSource, signMask);
  Value isNeg = b.icmp_ne(sign, zero);
  Value nonNeg = createUIntConstant(rewriter, loc, intTy, nonNegativeValue);
  Value neg = createUIntConstant(rewriter, loc, intTy, negativeValue);
  return b.select(isNeg, neg, nonNeg);
}

Value xorShiftRight(ConversionPatternRewriter &rewriter, Location loc, Value v,
                    unsigned shift) {
  TritonLLVMOpBuilder b(loc, rewriter);
  Value shiftValue = createUIntConstant(rewriter, loc, v.getType(), shift);
  Value shifted = b.lshr(v, shiftValue);
  return b.xor_(v, shifted);
}

Value inverseXorShiftRight(ConversionPatternRewriter &rewriter, Location loc,
                           Value v, const PayloadMixConfig &cfg) {
  for (unsigned shift = cfg.shift; shift < cfg.bitWidth; shift *= 2)
    v = xorShiftRight(rewriter, loc, v, shift);
  return v;
}

Value mixFloatToInt(ConversionPatternRewriter &rewriter, Location loc, Value u,
                    FloatType floatTy) {
  TritonLLVMOpBuilder b(loc, rewriter);
  PayloadMixConfig cfg = getPayloadMixConfig(floatTy);
  Value signFlip =
      selectUIntConstantOnSign(rewriter, loc, u, cfg.signMask, 0, cfg.signMask);
  Value x = b.xor_(u, signFlip);
  Value mulA = createUIntConstant(rewriter, loc, u.getType(), cfg.mulA);
  Value magMask = createUIntConstant(rewriter, loc, u.getType(), cfg.magMask);
  Value yMul = b.mul(x, mulA);
  Value y = b.and_(yMul, magMask);
  Value z = xorShiftRight(rewriter, loc, y, cfg.shift);
  Value mulB = selectUIntConstantOnSign(rewriter, loc, u, cfg.signMask,
                                        cfg.mulBPos, cfg.mulBNeg);
  Value wMul = b.mul(z, mulB);
  Value w = b.and_(wMul, magMask);
  return b.xor_(w, signFlip);
}

Value unmixIntToFloat(ConversionPatternRewriter &rewriter, Location loc,
                      Value v, FloatType floatTy) {
  TritonLLVMOpBuilder b(loc, rewriter);
  PayloadMixConfig cfg = getPayloadMixConfig(floatTy);
  Value signFlip =
      selectUIntConstantOnSign(rewriter, loc, v, cfg.signMask, 0, cfg.signMask);
  Value w = b.xor_(v, signFlip);
  Value magMask = createUIntConstant(rewriter, loc, v.getType(), cfg.magMask);
  Value mulBInv = selectUIntConstantOnSign(rewriter, loc, v, cfg.signMask,
                                           cfg.mulBPosInv, cfg.mulBNegInv);
  Value zMul = b.mul(w, mulBInv);
  Value z = b.and_(zMul, magMask);
  Value y = inverseXorShiftRight(rewriter, loc, z, cfg);
  Value mulAInv = createUIntConstant(rewriter, loc, v.getType(), cfg.mulAInv);
  Value xMul = b.mul(y, mulAInv);
  Value x = b.and_(xMul, magMask);
  return b.xor_(x, signFlip);
}

Value bitcastIfNeeded(ConversionPatternRewriter &rewriter, Location loc,
                      Value value, Type dstTy) {
  if (value.getType() == dstTy)
    return value;
  TritonLLVMOpBuilder b(loc, rewriter);
  return b.bitcast(value, dstTy);
}

struct ExperimentalFPSanEmbedOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalFPSanEmbedOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tti::ExperimentalFPSanEmbedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto floatTy = cast<FloatType>(getElementTypeOrSelf(op.getVal().getType()));
    Type intTy = rewriter.getIntegerType(floatTy.getWidth());

    SmallVector<Value> resultVals;
    for (Value elem : unpackLLElements(loc, adaptor.getVal(), rewriter)) {
      Value raw = bitcastIfNeeded(rewriter, loc, elem, intTy);
      resultVals.push_back(mixFloatToInt(rewriter, loc, raw, floatTy));
    }

    Value result = packLLElements(loc, getTypeConverter(), resultVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ExperimentalFPSanUnembedOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalFPSanUnembedOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tti::ExperimentalFPSanUnembedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto floatTy = cast<FloatType>(getElementTypeOrSelf(op.getType()));
    Type resultElemTy = getTypeConverter()->convertType(floatTy);

    SmallVector<Value> resultVals;
    for (Value elem : unpackLLElements(loc, adaptor.getVal(), rewriter)) {
      Value raw = unmixIntToFloat(rewriter, loc, elem, floatTy);
      resultVals.push_back(bitcastIfNeeded(rewriter, loc, raw, resultElemTy));
    }

    Value result = packLLElements(loc, getTypeConverter(), resultVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void mlir::triton::populateFpSanToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                               RewritePatternSet &patterns) {
  patterns.add<ExperimentalFPSanEmbedOpConversion,
               ExperimentalFPSanUnembedOpConversion>(typeConverter);
}
