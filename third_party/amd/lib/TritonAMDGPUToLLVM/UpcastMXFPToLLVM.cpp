#include "PatternTritonGPUOpToLLVM.h"

#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <array>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

SmallVector<Value> convertMxfp4x2ToFp16x2(RewriterBase &rewriter, Location loc,
                                          ArrayRef<Value> values) {
  SmallVector<Value> results;
  for (auto v : values) {
    auto em0 = and_(v, i8_val(0x7));
    auto em1 = and_(v, i8_val(0x70));
    // FP16 bits: sign = 1, exponent = 5, mantissa = 10
    Value v0 = or_(shl(zext(i16_ty, em0), i16_val(10 - 1)),
                   shl(zext(i16_ty, and_(v, i8_val(0x8))), i16_val(12)));
    Value v1 = or_(shl(zext(i16_ty, em1), i16_val(10 - 1 - 4)),
                   shl(zext(i16_ty, and_(v, i8_val(0x80))), i16_val(8)));

    // Three cases:
    // 1) x is normal and non-zero: Correct bias
    v0 = select(icmp_ne(and_(em0, i8_val(0x6)), i8_val(0)),
                add(v0, i16_val((15 - 1) << 10)), v0);
    v1 = select(icmp_ne(and_(em1, i8_val(0x60)), i8_val(0)),
                add(v1, i16_val((15 - 1) << 10)), v1);

    // 2) x is subnormal (x == 0bs001 where s is the sign): Map to fp16 +-0.5
    v0 = bitcast(select(icmp_eq(em0, i8_val(0x1)),
                        or_(i16_val(0x3800), and_(v0, i16_val(0x8000))), v0),
                 f16_ty);
    v1 = bitcast(select(icmp_eq(em1, i8_val(0x10)),
                        or_(i16_val(0x3800), and_(v1, i16_val(0x8000))), v1),
                 f16_ty);
    // 3) x is zero, nothing to do
    results.push_back(v0);
    results.push_back(v1);
  }
  return results;
}

Value mxfpScaleFp16(RewriterBase &rewriter, Location loc, Value v, Value scale,
                    bool fastMath) {
  Value scaleF32 = bitcast(shl(zext(i32_ty, scale), i32_val(23)), f32_ty);
  Value scaleF16 =
      LLVM::AMD::cvtFp32ToFp16(loc, rewriter, scaleF32, RoundingMode::RTNE);
  Value mulF16 = fmul(v, scaleF16);
  if (fastMath)
    return mulF16;
  // Account for NaN in the scale as per the mxfp specification.
  Value scaleIsNan = icmp_eq(scale, i8_val(0xff));
  Value nanF16 = bitcast(i16_val(0x7c01), f16_ty);
  return select(scaleIsNan, nanF16, bitcast(mulF16, f16_ty));
};

// Scales the given bf16 v using the given scale factor without relying on bf16
// multiplication.
//
// In gfx9 architectures, we don't have bf16 VALU ops. So instead this function
// handles v * scale multiplication using fp32 VALU ops. LLVM backend can do it
// for us, just with unnecessary overheads.
Value mxfpScaleBf16ViaF32(RewriterBase &rewriter, Location loc, Value v,
                          Value scale, bool fastMath) {
  Value c16 = i32_val(16);
  Value vF32 = bitcast(shl(zext(i32_ty, bitcast(v, i16_ty)), c16), f32_ty);
  Value scaleF32 = bitcast(shl(zext(i32_ty, scale), i32_val(23)), f32_ty);
  Value mulF32 = fmul(vF32, scaleF32);
  Value mulI16 = trunc(i16_ty, lshr(bitcast(mulF32, i32_ty), c16));
  Value mulBf16 = bitcast(mulI16, bf16_ty);
  if (fastMath)
    return mulBf16;
  // Account for NaN in the scale as per the mxfp specification.
  Value scaleIsNan = icmp_eq(scale, i8_val(0xff));
  Value nanBf16 = bitcast(i16_val(0x7fff), bf16_ty);
  return select(scaleIsNan, nanBf16, mulBf16);
};

class UpcastMXFPOpPattern : public ConvertOpToLLVMPattern<UpcastMXFPOp> {
private:
  const TargetInfoBase &targetInfo;

public:
  UpcastMXFPOpPattern(LLVMTypeConverter &typeConverter,
                      const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(UpcastMXFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fpType = op.getFpType();
    bool isPacked = fpType == ScaleDotElemType::E2M1;
    if (!(isPacked || fpType == ScaleDotElemType::E4M3 ||
          fpType == ScaleDotElemType::E5M2))
      return rewriter.notifyMatchFailure(op, "NYI: non-mxfp4/mxfp8 cases");

    Location loc = op.getLoc();
    auto xVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto scaleVals = unpackLLElements(loc, adaptor.getScale(), rewriter);
    LDBG("x: " << xVals.size() << " x " << xVals.front().getType());
    LDBG("scale: " << scaleVals.size() << " x " << scaleVals.front().getType());

    // When we lower scaled dot op, we made sure to distribute K only on one
    // warp. MXFP spec mandates 1 scale value for every 32 onsecutive values
    // along the K dimension. So in total each thread should read 32x main
    // element values.
    if (xVals.size() != scaleVals.size() * (isPacked ? 16 : 32))
      return rewriter.notifyMatchFailure(op, "unsupported problem size");

    auto dotEncoding =
        cast<DotOperandEncodingAttr>(op.getSrc().getType().getEncoding());
    auto mfmaEncoding = dyn_cast<AMDMfmaEncodingAttr>(dotEncoding.getParent());
    if (!mfmaEncoding)
      return rewriter.notifyMatchFailure(op, "NYI: non-mfma dot operand");
    LDBG("mfma: " << mfmaEncoding);

    int mDim = mfmaEncoding.getMDim();
    if (mDim != 32 && mDim != 16)
      return rewriter.notifyMatchFailure(op, "NYI: non-mfma32/16 intrinsics");

    int numThreads = triton::gpu::TritonGPUDialect::getThreadsPerWarp(
        op->getParentOfType<ModuleOp>());
    Value warpSize = i32_val(numThreads);
    Value tid = tid_val();
    Value warpId = udiv(tid, warpSize);
    Value laneId = urem(tid, warpSize);

    bool useFp16 = op.getType().getElementType().isF16();
    if (isPacked) {
      xVals = useFp16 ? convertMxfp4x2ToFp16x2(rewriter, loc, xVals)
                      : LLVM::convertMxfp4x2ToBf16x2(rewriter, loc, xVals);
    }

    // Given that MFMA layout for the A tensor arranges thread in a column-major
    // manner, for the current tid, it's at row (tid % mDim). When we set up
    // blocked layout for the A scale tensor, we made sure that it has a
    // threadsPerWarp = [M=mDim, K=64/mDim]. So the threads holding scale values
    // for the current thread starts at ((tid % mDim) * (64 / mDim)).
    Value offset = mul(urem(laneId, i32_val(mDim)), i32_val(numThreads / mDim));

    if (mDim == 32) {
      // One mfma32 intrinsic processes a 32x8 A tensor slice. Due to how we
      // tile, the same warp owns the whole K dim. Inside a warp, each thread
      // only holds 4 consecutive elements along K--a 1x4 vector. We need to
      // tile the warp 4 times to cover 32 values along K. So for a thread, the
      // first 4 1x4 vectors it holds shares the first scale value at row (tid %
      // mDim). the second 4 1x4 vectors shares the second scale value at row
      // (tid % mDim); and so forth.
      std::array<Value, 2> scaleThreads = {offset, add(offset, i32_val(1))};

      for (auto [i, scaleVal] : llvm::enumerate(scaleVals)) {
        std::array<Value, 2> si = {
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[0]),
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[1]),
        };

        for (int j = 0; j < 32; ++j) {
          int index = 32 * i + j;
          xVals[index] =
              useFp16 ? mxfpScaleFp16(rewriter, loc, xVals[index], si[j / 16],
                                      op.getFastMath())
                      : mxfpScaleBf16ViaF32(rewriter, loc, xVals[index],
                                            si[j / 16], op.getFastMath());
        }
      }
    } else {
      assert(mDim == 16);
      // One mfma16 intrinsic processes a 16x16 A tensor slice. Similarly, we
      // need to tile the warp 2 times to cover 32 valeus. So for a thread, the
      // first 2 1x4 vectors shares the first scale value at row (tid % mDim).
      std::array<Value, 4> scaleThreads = {offset, add(offset, i32_val(1)),
                                           add(offset, i32_val(2)),
                                           add(offset, i32_val(3))};

      for (auto [i, scaleVal] : llvm::enumerate(scaleVals)) {
        auto si = std::array<Value, 4>{
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[0]),
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[1]),
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[2]),
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[3]),
        };

        for (int j = 0; j < 32; ++j) {
          int index = 32 * i + j;
          xVals[index] = useFp16
                             ? mxfpScaleFp16(rewriter, loc, xVals[index],
                                             si[j / 16], op.getFastMath())
                             : mxfpScaleBf16ViaF32(rewriter, loc, xVals[index],
                                                   si[j / 8], op.getFastMath());
        }
      }
    }

    Value result =
        packLLElements(loc, getTypeConverter(), xVals, rewriter, op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

void mlir::triton::AMD::populateUpcastMXFPToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<UpcastMXFPOpPattern>(typeConverter, targetInfo, benefit);
}
