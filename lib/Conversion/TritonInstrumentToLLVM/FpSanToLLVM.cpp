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
namespace ttg = mlir::triton::gpu;

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

  // This odd multiplier reduces trailing zeros for common model constants
  // while mixing low bits into high bits invertibly:
  uint64_t mulA = 14940041u & magMask;
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
  Value mulA = createUIntConstant(rewriter, loc, u.getType(), cfg.mulA);
  Value magMask = createUIntConstant(rewriter, loc, u.getType(), cfg.magMask);
  // Avoid patterns that InstCombine rewrites to `llvm.fabs`. LLVM specifies
  // that `llvm.fabs` preserves the NaN quiet/signaling bit and payload, but
  // NVPTX lowers it to PTX `abs.f32`, whose NaN result is unspecified. On
  // Blackwell, `abs.f32` is observed to canonicalize signaling NaNs, corrupting
  // FPSan payloads.
  Value yMul = b.mul(u, mulA);
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
  Value magMask = createUIntConstant(rewriter, loc, v.getType(), cfg.magMask);
  Value mulBInv = selectUIntConstantOnSign(rewriter, loc, v, cfg.signMask,
                                           cfg.mulBPosInv, cfg.mulBNegInv);
  Value zMul = b.mul(v, mulBInv);
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
    for (Value elem :
         unpackUniqueTensorElements(loc, adaptor.getVal(), rewriter)) {
      Value raw = bitcastIfNeeded(rewriter, loc, elem, intTy);
      resultVals.push_back(mixFloatToInt(rewriter, loc, raw, floatTy));
    }

    Value result = packUniqueTensorElements(loc, getTypeConverter(), resultVals,
                                            rewriter, op.getType());
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
    for (Value elem :
         unpackUniqueTensorElements(loc, adaptor.getVal(), rewriter)) {
      Value raw = unmixIntToFloat(rewriter, loc, elem, floatTy);
      resultVals.push_back(bitcastIfNeeded(rewriter, loc, raw, resultElemTy));
    }

    Value result = packUniqueTensorElements(loc, getTypeConverter(), resultVals,
                                            rewriter, op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct FpSanAtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp> {
  FpSanAtomicRMWOpConversion(LLVMTypeConverter &converter,
                             const TargetInfoBase &targetInfo,
                             PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op->hasAttr("tti.fpsan"))
      return failure();
    auto loc = op.getLoc();
    insertAtomicOrderingBarriers(op, op.getSem(),
                                 !atomicResultHasOrderingBarrier(op), rewriter,
                                 targetInfo);
    SmallVector<Value> ptrs =
        unpackUniqueTensorElements(loc, adaptor.getPtr(), rewriter);
    SmallVector<Value> values =
        unpackUniqueTensorElements(loc, adaptor.getVal(), rewriter);
    SmallVector<Value> masks;
    if (adaptor.getMask())
      masks = unpackUniqueTensorElements(loc, adaptor.getMask(), rewriter);
    Value threadPred = ttg::emitRedundantThreadPredicate(
        getFreeVariableMasks(op.getPtr().getType()), rewriter, loc, targetInfo);
    auto floatTy = cast<FloatType>(getElementTypeOrSelf(op.getType()));
    StringRef scope = targetInfo.getAtomicSyncScope(op.getScope());
    SmallVector<Value> results;
    for (auto [i, ptr] : llvm::enumerate(ptrs)) {
      Value pred = ttg::maybeAnd(rewriter, loc, threadPred,
                                 masks.empty() ? Value() : masks[i]);
      results.push_back(emitFpSanAtomicRMW(rewriter, loc, ptr, values[i], pred,
                                           floatTy, op.getAtomicRmwOp(),
                                           op.getSem(), scope));
    }
    TritonLLVMOpBuilder b(loc, rewriter);
    finalizeAtomicResults(op, rewriter, results,
                          getTypeConverter()->convertType(floatTy), b,
                          threadPred, targetInfo, getTypeConverter());
    return success();
  }

  const TargetInfoBase &targetInfo;
};

} // namespace

// Commits a payload update to floating carrier storage with a compare-and-swap
// loop. The first attempt guesses zero carrier bits; failed attempts retry with
// the observed bits. Comparing bits also handles NaN carriers and signed zeros.
Value mlir::emitFpSanAtomicRMW(ConversionPatternRewriter &rewriter,
                               Location loc, Value ptr, Value value, Value pred,
                               FloatType floatTy, RMWOp rmwOp, MemSemantic sem,
                               StringRef scope) {
  TritonLLVMOpBuilder b(loc, rewriter);
  Type intTy = rewriter.getIntegerType(floatTy.getWidth());
  Value rhs = mixFloatToInt(
      rewriter, loc, bitcastIfNeeded(rewriter, loc, value, intTy), floatTy);
  Value zero = createUIntConstant(rewriter, loc, intTy, 0);

  Block *entry = rewriter.getInsertionBlock();
  Block *done = rewriter.splitBlock(entry, rewriter.getInsertionPoint());
  Block *loop = rewriter.createBlock(done, TypeRange{intTy}, {loc});
  Value old = done->addArgument(intTy, loc);
  rewriter.setInsertionPointToEnd(entry);
  if (pred)
    LLVM::CondBrOp::create(rewriter, loc, pred, loop, ValueRange{zero}, done,
                           ValueRange{b.undef(intTy)});
  else
    LLVM::BrOp::create(rewriter, loc, ValueRange{zero}, loop);

  rewriter.setInsertionPointToStart(loop);
  Value expected = loop->getArgument(0);
  Value lhs = mixFloatToInt(rewriter, loc, expected, floatTy);
  Value payload;
  switch (rmwOp) {
  case RMWOp::FADD:
    payload = b.add(lhs, rhs);
    break;
  case RMWOp::MIN:
    payload = b.smin(lhs, rhs);
    break;
  case RMWOp::MAX:
    payload = b.smax(lhs, rhs);
    break;
  default:
    llvm_unreachable("unsupported FpSan atomic operation");
  }
  Value desired = unmixIntToFloat(rewriter, loc, payload, floatTy);
  Value cas = LLVM::AtomicCmpXchgOp::create(
      rewriter, loc, ptr, expected, desired, *getMemoryOrdering(sem),
      LLVM::AtomicOrdering::monotonic, scope);
  Value observed = b.extract_val(intTy, cas, 0);
  Value success = b.extract_val(rewriter.getI1Type(), cas, 1);
  LLVM::CondBrOp::create(rewriter, loc, success, done, ValueRange{observed},
                         loop, ValueRange{observed});

  rewriter.setInsertionPointToStart(done);
  return bitcastIfNeeded(rewriter, loc, old, value.getType());
}

void mlir::triton::populateFpSanToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                               RewritePatternSet &patterns,
                                               const TargetInfoBase &targetInfo,
                                               PatternBenefit benefit) {
  patterns.add<ExperimentalFPSanEmbedOpConversion,
               ExperimentalFPSanUnembedOpConversion>(typeConverter);
  patterns.add<FpSanAtomicRMWOpConversion>(typeConverter, targetInfo, benefit);
}
