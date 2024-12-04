#include "PatternTritonGPUOpToLLVM.h"

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
      return rewriter.notifyMatchFailure(op, "NYI: non-mxfp8 cases");

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

    if (isPacked)
      xVals = LLVM::convertMxfp4x2ToBf16x2(rewriter, loc, xVals);

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
              LLVM::mxfpScaleBf16(rewriter, loc, xVals[index], si[j / 16]);
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
          xVals[index] =
              LLVM::mxfpScaleBf16(rewriter, loc, xVals[index], si[j / 8]);
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
