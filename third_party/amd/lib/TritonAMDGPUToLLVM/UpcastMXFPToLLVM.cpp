#include "PatternTritonGPUOpToLLVM.h"

#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
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

// Returns (EM, S) selectors to the llvm.amdgcn.perm intrinsic for selecting
// resultant bf16/fp16 bytes in the lookup table.
std::pair<Value, Value> composePermuteSelectors(Location loc,
                                                RewriterBase &rewriter,
                                                Value val10, Value val32) {
  // Each input value packs two mxfp4 values. First extract each mxfp4 value's
  // EM and S bits. In order to form the selector for llvm.amdgcn.perm
  // instruction, we need to shuffle them into a 4xu8 manner.

  // 0xX[S.EE.M] -> 0x0000000[0EEM]
  Value v0EM = zext(i32_ty, and_(val10, i8_val(0x07)));
  // 0xX[S.EE.M] -> 0x0000000[000S]
  Value v0S = lshr(zext(i32_ty, and_(val10, i8_val(0x08))), i32_val(3));
  // 0x[S.EE.M]X -> 0x00000[0EEM]00
  Value v1EM = shl(zext(i32_ty, and_(val10, i8_val(0x70))), i32_val(4));
  // 0x[S.EE.M]X -> 0x00000[000S]00
  Value v1S = shl(zext(i32_ty, and_(val10, i8_val(0x80))), i32_val(4 - 3));

  // 0xX[S.EE.M] -> 0x000[0EEM]0000
  Value v2EM = shl(zext(i32_ty, and_(val32, i8_val(0x07))), i32_val(16));
  // 0xX[S.EE.M] -> 0x000[000S]0000
  Value v2S = shl(zext(i32_ty, and_(val32, i8_val(0x08))), i32_val(16 - 3));
  // 0x[S.EE.M]X -> 0x0[0EEM]000000
  Value v3EM = shl(zext(i32_ty, and_(val32, i8_val(0x70))), i32_val(20));
  // 0x[S.EE.M]X -> 0x0[000S]000000
  Value v3S = shl(zext(i32_ty, and_(val32, i8_val(0x80))), i32_val(20 - 3));

  Value selectorEM = or_(v3EM, or_(v2EM, or_(v1EM, v0EM)));
  Value selectorS = or_(v3S, or_(v2S, or_(v1S, v0S)));
  return {selectorEM, selectorS};
}

SmallVector<Value, 2> upcast4xMxfp4(RewriterBase &rewriter,
                                    UpcastMXFPOp upcastOp, bool tofp16,
                                    ArrayRef<Value> inputs) {
  assert(inputs.size() == 2);
  Location loc = upcastOp.getLoc();

  // MXFP4 has 4 bits, S.EE.M, for Sign, Exponent, and Mantissa respectively.
  // For a specific S, we have a total of 8 bit patterns. We can encode all
  // these 8 resultant bf16/fp16 bit patterns in a lookup table (LUT). It
  // happens that llvm.amdgcn.perm supports selecting 4 bytes from 8 input bytes
  // using a 4-byte selector. So the overall idea is to use llvm.amdgcn.perm to
  // implement such a LUT; though we need to select the two bytes for the
  // resultant bf16/fp16 bit patterns separately. For the byte containing S, we
  // also need to handle the S and E bits separately.

  auto [selectorEM, selectorS] =
      composePermuteSelectors(loc, rewriter, inputs[0], inputs[1]);

  // FP4 has 4 bits: S.EE.M. Bf16/fp16 bit patterns for positive values:
  //
  // FP4    | BF16   | FP16   | Value
  // ------ | ------ | ------ | -----
  // 0.00.0 | 0x0000 | 0x0000 | + 0.0
  // 0.00.1 | 0x3f00 | 0x3800 | + 0.5
  // 0.01.0 | 0x3f80 | 0x3c00 | + 1.0
  // 0.01.1 | 0x3fc0 | 0x3e00 | + 1.5
  // 0.10.0 | 0x4000 | 0x4000 | + 2.0
  // 0.10.1 | 0x4040 | 0x4200 | + 3.0
  // 0.11.0 | 0x4080 | 0x4400 | + 4.0
  // 0.11.1 | 0x40c0 | 0x4600 | + 6.0
  //
  // Encode Byte #0 (M) for BF16/FP16 in a LUT.
  Value resB0LutLo = tofp16 ? i32_val(0) : i32_val(0xc0800000);
  Value resB0LutHi = tofp16 ? i32_val(0) : i32_val(0xc0804000);
  // Encode Byte #1 (EM, non-S part) for BF16/FP16 in a LUT.
  Value resB1LutLoNoS = tofp16 ? i32_val(0x3e3c3800) : i32_val(0x3f3f3f00);
  Value resB1LutHiNoS = tofp16 ? i32_val(0x46444240) : i32_val(0x40404040);
  // Encode Byte #1 (S part) for BF16/FP16 in a LUT.
  Value resB1LutLoS = i32_val(0x8000);
  Value resB1LutHiS = i32_val(0);

  Type i32Ty = rewriter.getI32Type();
  auto permU32FnTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty, i32Ty, i32Ty});
  LLVM::LLVMFuncOp funcOp = appendOrGetExternFuncOp(
      rewriter, upcastOp, "llvm.amdgcn.perm", permU32FnTy);

  // Select Byte #0 for all 4 mxfp4 values. It's always 0 if upcasting to fp16.
  Value resB0 = i32_val(0);
  if (!tofp16) {
    resB0 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                   {resB0LutHi, resB0LutLo, selectorEM})
                .getResult();
  }
  // Select Byte #1 for all 4 mxfp4 values.
  auto resB1NoS = LLVM::createLLVMCallOp(
      rewriter, loc, funcOp, {resB1LutHiNoS, resB1LutLoNoS, selectorEM});
  auto resB1S = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                       {resB1LutHiS, resB1LutLoS, selectorS});
  Value restB1 = or_(resB1NoS.getResult(), resB1S.getResult());

  // Extract resultant bf16/fp16 values #0 and #1.
  // #0 would use selector 0x00/0x04 to pick from B0/B1.
  // #1 would use selector 0x01/0x05 to pick from B0/B1.
  auto res10 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                      {restB1, resB0, i32_val(0x05010400)});
  // Extract resultant bf16/fp16 values #2 and #3.
  // #2 would use selector 0x02/0x06 to pick from B0/B1.
  // #3 would use selector 0x03/0x07 to pick from B0/B1.
  auto res32 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                      {restB1, resB0, i32_val(0x07030602)});

  return {res10.getResult(), res32.getResult()};
}

SmallVector<Value> upcastMxfp4(RewriterBase &rewriter, UpcastMXFPOp upcastOp,
                               bool toFp16, ArrayRef<Value> values) {
  assert(values.size() % 2 == 0);
  Location loc = upcastOp.getLoc();

  SmallVector<Value> results;
  results.reserve(values.size() * 2);
  Type elemType = toFp16 ? f16_ty : bf16_ty;
  for (int i = 0; i < values.size(); i += 2) {
    SmallVector<Value, 2> v4i32 =
        upcast4xMxfp4(rewriter, upcastOp, toFp16, values.slice(i, 2));
    for (int j = 0; j < 2; j++) {
      Value elements = bitcast(v4i32[j], vec_ty(elemType, 2));
      results.push_back(extract_element(elements, i32_val(0)));
      results.push_back(extract_element(elements, i32_val(1)));
    }
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
      xVals = upcastMxfp4(rewriter, op, useFp16, xVals);
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
