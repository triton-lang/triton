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

    if (fpType == ScaleDotElemType::E2M1)
      xVals = unpackFP4Elements(loc, rewriter, xVals);

    // Each thread owns elements of 4 mxfp vectors so we need 4 scales
    // Letting c = tid / 4 * 2, we need the elements from threads c, c + 1, c +
    // 16, c + 17
    auto c = mul(udiv(laneId, i32_val(4)), i32_val(2));
    std::array<Value, 4> ci = {c, add(c, i32_val(1)), add(c, i32_val(16)),
                               add(c, i32_val(17))};

    for (auto [i, scaleVal] : llvm::enumerate(scaleVals)) {
      // column major as per the DotOperandEncoding(opidx=0) layout
      auto si = std::array<Value, 4>{
          targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[0]),
          targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[2]),
          targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[1]),
          targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[3]),
      };

      for (int j = 0; j < 32; ++j) {
        xVals[32 * i + j] =
            LLVM::mxfpScaleBf16(rewriter, loc, xVals[32 * i + j], si[j / 8]);
      }
    }

    Value result =
        packLLElements(loc, getTypeConverter(), xVals, rewriter, op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  llvm::SmallVector<Value> unpackFP4Elements(Location loc,
                                             RewriterBase &rewriter,
                                             ArrayRef<Value> vals) const {
    llvm::SmallVector<Value> ret;
    ret.reserve(vals.size() * 2);
    for (auto v : vals) {
      auto [e0, e1] = LLVM::convertMxfp4x2ToBf16x2(rewriter, loc, v);
      ret.push_back(e0);
      ret.push_back(e1);
    }

    return ret;
  }
};
} // anonymous namespace

void mlir::triton::NVIDIA::populateUpcastMXFPToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<UpcastMXFPOpPattern>(typeConverter, targetInfo, benefit);
}
