#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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
      : ConvertOpToLLVMPattern<UpcastMXFPOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(UpcastMXFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto tyX = cast<RankedTensorType>(op->getOperandTypes()[0]);
    auto operands = adaptor.getOperands();

    auto xVals = unpackLLElements(loc, operands[0], rewriter);
    auto scaleVals = unpackLLElements(loc, operands[1], rewriter);
    auto fpType = op.getFpType();

    Value tid = tid_val();
    auto mod = op->getParentOfType<ModuleOp>();
    Value warpSize =
        i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
    Value warpId = udiv(tid, warpSize);
    Value laneId = urem(tid, warpSize);

    auto kWidth =
        cast<DotOperandEncodingAttr>(op.getType().getEncoding()).getKWidth();

    if (fpType == ScaleDotElemType::E2M1)
      xVals = LLVM::convertMxfp4x2ToBf16x2(rewriter, loc, xVals);

    // Each thread owns elements of 4 mxfp vectors so we need 4 scales
    // Since we go from a threadShape of 8x4 to 16x2, we let c = tid / 4 * 2
    // Then, we need elements c and c + 16 for the first two mxfp vectors
    // and elements c + 1 and c + 17 for the last two mxfp vectors
    auto c = mul(udiv(laneId, i32_val(4)), i32_val(2));
    std::array<Value, 4> ci = {c, add(c, i32_val(16)), add(c, i32_val(1)),
                               add(c, i32_val(17))};

    // TODO Move this logic to using LinearLayouts
    // Each scale in a warp has to be replicated to cover a tile of shape mxk =
    // 16x64 This 16x64 tile is split into 4 subtiles of shape 8x32, each of
    // which will have to gather a scale and multiply its relevant part of the
    // mxfp vector This tile of 8x32 is split in to 8x4 vectors, leaving each
    // vector with 1x8 mxfp elements as long as kWidth * 4 <= 32
    assert(kWidth <= 8 &&
           "NYI for larger kWidth (but we could do it with less shuffles!)");
    for (auto [i, scaleVal] : llvm::enumerate(scaleVals)) {
      for (int mxfp = 0; mxfp < 2; ++mxfp) {
        auto si = std::array<Value, 2>{
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[mxfp * 2 + 0]),
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[mxfp * 2 + 1])};
        for (int rep = 0; rep < 8 / kWidth; ++rep) {
          for (int subTile = 0; subTile < 2; ++subTile) {
            for (int k = 0; k < kWidth; ++k) {
              auto idx =
                  32 * i + 16 * mxfp + rep * 2 * kWidth + subTile * kWidth + k;
              xVals[idx] =
                  LLVM::mxfpScaleBf16(rewriter, loc, xVals[idx], si[subTile]);
            }
          }
        }
      }
    }

    Value result =
        packLLElements(loc, getTypeConverter(), xVals, rewriter, op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // anonymous namespace

void mlir::triton::NVIDIA::populateUpcastMXFPToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<UpcastMXFPOpPattern>(typeConverter, targetInfo, benefit);
}
