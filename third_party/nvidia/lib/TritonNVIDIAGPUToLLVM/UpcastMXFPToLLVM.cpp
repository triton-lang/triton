#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
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

  llvm::SmallVector<Value>
  unpackFP4Elements(Location loc, ConversionPatternRewriter &rewriter,
                    const llvm::SmallVector<Value> &vals, Value laneId) const {
    auto fp4x2ToBf16x2 = [&loc, &rewriter](Value v) -> Value {
      auto em0 = and_(v, i8_val(0x70));
      auto em1 = and_(v, i8_val(0x7));
      Value v0 = or_(shl(zext(i16_ty, em0), i16_val(2)),
                     shl(zext(i16_ty, and_(v, i8_val(0x80))), i16_val(8)));
      Value v1 = or_(shl(zext(i16_ty, em1), i16_val(6)),
                     shl(zext(i16_ty, and_(v, i8_val(0x8))), i16_val(12)));

      // Three cases:
      // 1) x is normal and non-zero: Correct bias
      v0 = select(icmp_ne(and_(em0, i8_val(0x60)), i8_val(0)),
                  add(v0, i16_val((127 - 1) << 7)), v0);
      v1 = select(icmp_ne(and_(em1, i8_val(0x6)), i8_val(0)),
                  add(v1, i16_val((127 - 1) << 7)), v1);

      // 2) x is subnormal (x == 0bs001 where s is the sign): Map to +-0.5 in
      // bf16
      v0 = select(icmp_eq(em0, i8_val(0x10)),
                  or_(i16_val(16128), and_(v0, i16_val(0x8000))), v0);
      v1 = select(icmp_eq(em1, i8_val(0x1)),
                  or_(i16_val(16128), and_(v1, i16_val(0x8000))), v1);
      // 3) x is zero, nothing to do

      // Swap as they come packed in big endian
      return or_(zext(i32_ty, v0), shl(zext(i32_ty, v1), i32_val(16)));
    };

    auto fp4x8ToBf16x2 = [&loc, &rewriter, &fp4x2ToBf16x2](
                             Value v) -> llvm::SmallVector<Value, 4> {
      llvm::SmallVector<Value, 4> results(4);
      for (int i = 0; i < 4; ++i) {
        auto v_i = trunc(i8_ty, lshr(v, i32_val(8 * i)));
        results[i] = fp4x2ToBf16x2(v_i);
      }
      return results;
    };

    // Split fp4x8 into 4 bf16x2
    llvm::SmallVector<Value> ret;
    ret.reserve(vals.size() * 4);
    for (int i = 0; i < vals.size(); ++i) {
      auto vs = fp4x8ToBf16x2(vals[i]);
      assert(vs.size() == 4);
      for (auto v : vs) {
        ret.push_back(v);
      }
    }

    return ret;
  }

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

    if (fpType == ScaleType::E2M1) {
      xVals = unpackFP4Elements(loc, rewriter, xVals, laneId);
    }

    auto scaleBf16x2 = [&loc, &rewriter](Value v, Value s) -> Value {
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
      // column major as per the DotOperandEncoding(opidx=0) layout
      auto si = std::array<Value, 4>{
          targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[0]),
          targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[2]),
          targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[1]),
          targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[3]),
      };

      for (int j = 0; j < 16; ++j) {
        xVals[16 * i + j] = scaleBf16x2(xVals[16 * i + j], si[j / 4]);
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
