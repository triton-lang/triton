#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
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

    Value tid = tid_val();
    auto mod = op->getParentOfType<ModuleOp>();
    Value warpSize =
        i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
    Value warpId = udiv(tid, warpSize);
    Value laneId = urem(tid, warpSize);

    auto scale = [&loc, &rewriter](Value v, Value s) -> Value {
      // Split bf16x2 into 2 bf16, scale each of them, and pack them back
      // TODO Is it true that the bfloats are always packed as bf16x2?
      auto bf16_0 = bitcast(trunc(i16_ty, v), bf16_ty);
      auto bf16_1 = bitcast(trunc(i16_ty, lshr(v, i32_val(16))), bf16_ty);
      auto scaleIsNan = icmp_eq(s, i8_val(0xff));
      auto scaleBf16 = bitcast(shl(zext(i16_ty, s), i16_val(7)), bf16_ty);
      auto scaledBf16_0 = fmul(bf16_0, scaleBf16);
      auto scaledBf16_1 = fmul(bf16_1, scaleBf16);
      auto i16_0 = bitcast(scaledBf16_0, i16_ty);
      auto i16_1 = bitcast(scaledBf16_1, i16_ty);
      auto packed =
          or_(zext(i32_ty, i16_0), shl(zext(i32_ty, i16_1), i32_val(16)));
      // Account for NaN in the scale as per the mxfp specification
      auto packed_nan = select(scaleIsNan, i32_val(0x7fff7fff), packed);
      return packed_nan;
    };

    // Each thread owns elements of 4 mxfp vectors so we need 4 scales
    // Letting c = tid / 4 * 2, we need the elements from threads c, c + 1, c +
    // 16, c + 17
    auto c = mul(udiv(laneId, i32_val(4)), i32_val(2));
    std::array<Value, 4> ci = {c, add(c, i32_val(1)), add(c, i32_val(16)),
                               add(c, i32_val(17))};
    for (auto [i, scaleVal] : llvm::enumerate(scaleVals)) {
      auto si = std::array<Value, 4>{
          targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[0]),
          targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[1]),
          targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[2]),
          targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[3]),
      };
      // si indices for the 16 elements in x
      std::array<int, 16> siMap = {0, 0, 2, 2, 0, 0, 2, 2,
                                   1, 1, 3, 3, 1, 1, 3, 3};
      for (int j = 0; j < 16; ++j) {
        xVals[16 * i + j] = scale(xVals[16 * i + j], si[siMap[j]]);
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
