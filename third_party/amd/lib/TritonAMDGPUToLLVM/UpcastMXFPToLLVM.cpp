#include "PatternTritonGPUOpToLLVM.h"

#include "Dialect/TritonAMDGPU/IR/Dialect.h"
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

template <typename ConvertOp>
SmallVector<Value, 4> mxfp4Scale_HW(RewriterBase &rewriter, Location loc,
                                    const SmallVector<Value> &xVals, int idx,
                                    Value scale) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value v0 = xVals[idx];
  Value v1 = xVals[idx + 1];
  Value v2 = xVals[idx + 2];
  Value v3 = xVals[idx + 3];
  Value packedVec = b.undef(vec_ty(i8_ty, 4));
  packedVec = b.insert_element(packedVec, v0, b.i32_val(0));
  packedVec = b.insert_element(packedVec, v1, b.i32_val(1));
  packedVec = b.insert_element(packedVec, v2, b.i32_val(2));
  packedVec = b.insert_element(packedVec, v3, b.i32_val(3));
  packedVec = b.bitcast(packedVec, i32_ty);
  Type retElemType = bf16_ty;
  if constexpr (std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkF16Fp4Op>)
    retElemType = f16_ty;
  Type resType = vec_ty(retElemType, 2);
  Value scaleF32 =
      b.bitcast(b.shl(b.zext(i32_ty, scale), b.i32_val(23)), f32_ty);
  SmallVector<Value, 4> results;
  results.push_back(rewriter.create<ConvertOp>(loc, resType, packedVec,
                                               scaleF32,
                                               /*srcSelIndex=*/0));
  // Intentionally swap the byte indices 1 and 2 to align with how the LLVM
  // backend accesses them
  results.push_back(rewriter.create<ConvertOp>(loc, resType, packedVec,
                                               scaleF32,
                                               /*srcSelIndex=*/2));
  results.push_back(rewriter.create<ConvertOp>(loc, resType, packedVec,
                                               scaleF32,
                                               /*srcSelIndex=*/1));
  results.push_back(rewriter.create<ConvertOp>(loc, resType, packedVec,
                                               scaleF32,
                                               /*srcSelIndex=*/3));
  return results;
}

template <typename ConvertOp>
SmallVector<Value, 2> mxfp8Scale_HW(RewriterBase &rewriter, Location loc,
                                    const SmallVector<Value> &xVals, int idx,
                                    Value scale) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value v0 = xVals[idx];
  Value v1 = xVals[idx + 1];
  Value v2 = xVals[idx + 2];
  Value v3 = xVals[idx + 3];
  Value packedVec = b.undef(vec_ty(i8_ty, 4));
  packedVec = b.insert_element(packedVec, v0, b.i32_val(0));
  packedVec = b.insert_element(packedVec, v1, b.i32_val(1));
  packedVec = b.insert_element(packedVec, v2, b.i32_val(2));
  packedVec = b.insert_element(packedVec, v3, b.i32_val(3));
  packedVec = b.bitcast(packedVec, i32_ty);
  Type retElemType = bf16_ty;
  if constexpr (std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkF16Fp8Op> ||
                std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkF16Bf8Op>)
    retElemType = f16_ty;
  Type resType = vec_ty(retElemType, 2);
  Value scaleF32 =
      b.bitcast(b.shl(b.zext(i32_ty, scale), b.i32_val(23)), f32_ty);
  SmallVector<Value, 2> results;
  results.push_back(rewriter.create<ConvertOp>(loc, resType, packedVec,
                                               scaleF32,
                                               /*srcLoHiSel=*/false));
  results.push_back(rewriter.create<ConvertOp>(loc, resType, packedVec,
                                               scaleF32,
                                               /*srcLoHiSel=*/true));
  return results;
}

// Upcast 8 mxfp4 values from xVals starting at idx using the given scale
// factor, and store the results into yVals
static void upcast8xMxfp4(RewriterBase &rewriter, Location loc,
                          const SmallVector<Value> &xVals, bool useFp16,
                          int idx, Value scale, SmallVector<Value> &yVals) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Type retElemType = useFp16 ? f16_ty : bf16_ty;
  SmallVector<Value, 4> v4i32 =
      useFp16 ? mxfp4Scale_HW<ROCDL::CvtScaleF32PkF16Fp4Op>(rewriter, loc,
                                                            xVals, idx, scale)
              : mxfp4Scale_HW<ROCDL::CvtScaleF32PkBf16Fp4Op>(rewriter, loc,
                                                             xVals, idx, scale);
  for (int k = 0; k < 4; k++) {
    Value elements = b.bitcast(v4i32[k], vec_ty(retElemType, 2));
    yVals.push_back(b.extract_element(elements, b.i32_val(0)));
    yVals.push_back(b.extract_element(elements, b.i32_val(1)));
  }

  return;
}

// Upcast 4 mxfp8 values from xVals starting at idx using the given scale
// factor, and store the results into yVals
static void upcast4xMxfp8(RewriterBase &rewriter, Location loc,
                          const SmallVector<Value> &xVals, bool useFp16,
                          ScaleDotElemType fpType, int idx, Value scale,
                          SmallVector<Value> &yVals) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Type retElemType = useFp16 ? f16_ty : bf16_ty;
  SmallVector<Value, 2> v2i32 =
      useFp16 ? (fpType == ScaleDotElemType::E4M3
                     ? mxfp8Scale_HW<ROCDL::CvtScaleF32PkF16Fp8Op>(
                           rewriter, loc, xVals, idx, scale)
                     : mxfp8Scale_HW<ROCDL::CvtScaleF32PkF16Bf8Op>(
                           rewriter, loc, xVals, idx, scale))
              : (fpType == ScaleDotElemType::E4M3
                     ? mxfp8Scale_HW<ROCDL::CvtScaleF32PkBf16Fp8Op>(
                           rewriter, loc, xVals, idx, scale)
                     : mxfp8Scale_HW<ROCDL::CvtScaleF32PkBf16Bf8Op>(
                           rewriter, loc, xVals, idx, scale));
  for (int k = 0; k < 2; k++) {
    Value elements = b.bitcast(v2i32[k], vec_ty(retElemType, 2));
    yVals.push_back(b.extract_element(elements, b.i32_val(0)));
    yVals.push_back(b.extract_element(elements, b.i32_val(1)));
  }
  return;
}
class UpcastMXFPOpPattern
    : public ConvertOpToLLVMPattern<amdgpu::UpcastMXFPOp> {
private:
  const TargetInfoBase &targetInfo;

public:
  UpcastMXFPOpPattern(LLVMTypeConverter &typeConverter,
                      const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(amdgpu::UpcastMXFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fpType = op.getFpType();
    bool isPacked = fpType == ScaleDotElemType::E2M1;
    if (!(isPacked || fpType == ScaleDotElemType::E4M3 ||
          fpType == ScaleDotElemType::E5M2))
      return rewriter.notifyMatchFailure(op, "NYI: non-mxfp4/mxfp8 cases");

    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto xVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto scaleVals = unpackLLElements(loc, adaptor.getScale(), rewriter);
    LDBG("x: " << xVals.size() << " x " << xVals.front().getType());
    LDBG("scale: " << scaleVals.size() << " x " << scaleVals.front().getType());
    SmallVector<Value> yVals;
    auto f = isPacked ? 2 : 1;
    yVals.reserve(f * xVals.size());

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

    int numThreads = lookupThreadsPerWarp(rewriter);
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);

    bool useFp16 = op.getType().getElementType().isF16();
    Type retElemType = useFp16 ? f16_ty : bf16_ty;

    // Given that MFMA layout for the A tensor arranges thread in a column-major
    // manner, for the current tid, it's at row (tid % mDim). When we set up
    // blocked layout for the A scale tensor, we made sure that it has a
    // threadsPerWarp = [M=mDim, K=64/mDim]. So the threads holding scale values
    // for the current thread starts at ((tid % mDim) * (64 / mDim)).
    Value offset =
        b.mul(b.urem(laneId, b.i32_val(mDim)), b.i32_val(numThreads / mDim));

    if (mDim == 32) {
      // One mfma32 intrinsic processes a 32x8 A tensor slice. Due to how we
      // tile, the same warp owns the whole K dim. Inside a warp, each thread
      // only holds 4 consecutive elements along K--a 1x4 vector. We need to
      // tile the warp 4 times to cover 32 values along K. So for a thread, the
      // first 4 1x4 vectors it holds shares the first scale value at row (tid %
      // mDim). the second 4 1x4 vectors shares the second scale value at row
      // (tid % mDim); and so forth.
      std::array<Value, 2> scaleThreads = {offset, b.add(offset, b.i32_val(1))};

      for (auto [i, scaleVal] : llvm::enumerate(scaleVals)) {
        std::array<Value, 2> si = {
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[0]),
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[1]),
        };

        if (isPacked) {
          for (int j = 0; j < 16; j += 4) {
            auto idx = 16 * i + j;
            upcast8xMxfp4(rewriter, loc, xVals, useFp16, idx, si[j / 8], yVals);
          }
        } else {
          for (int j = 0; j < 32; j += 4) {
            auto idx = 32 * i + j;
            upcast4xMxfp8(rewriter, loc, xVals, useFp16, fpType, idx,
                          si[j / 16], yVals);
          }
        }
      }
    } else {
      assert(mDim == 16);
      // One mfma16 intrinsic processes a 16x16 A tensor slice. Similarly, we
      // need to tile the warp 2 times to cover 32 values. So for a thread, the
      // first 2 1x4 vectors shares the first scale value at row (tid % mDim).
      std::array<Value, 4> scaleThreads = {offset, b.add(offset, b.i32_val(1)),
                                           b.add(offset, b.i32_val(2)),
                                           b.add(offset, b.i32_val(3))};

      for (auto [i, scaleVal] : llvm::enumerate(scaleVals)) {
        auto si = std::array<Value, 4>{
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[0]),
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[1]),
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[2]),
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[3]),
        };

        if (isPacked) {
          for (int j = 0; j < 16; j += 4) {
            auto idx = 16 * i + j;
            upcast8xMxfp4(rewriter, loc, xVals, useFp16, idx, si[j / 4], yVals);
          }
        } else {
          for (int j = 0; j < 32; j += 4) {
            auto idx = 32 * i + j;
            upcast4xMxfp8(rewriter, loc, xVals, useFp16, fpType, idx, si[j / 8],
                          yVals);
          }
        }
      }
    }

    Value result =
        packLLElements(loc, getTypeConverter(), yVals, rewriter, op.getType());
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
