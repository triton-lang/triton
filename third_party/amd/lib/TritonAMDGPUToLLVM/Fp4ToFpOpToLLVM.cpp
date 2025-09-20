#include "PatternTritonGPUOpToLLVM.h"

#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using ::mlir::LLVM::AMD::upcast8xMxfp4_SW;

namespace {

class Fp4ToFpOpPattern : public ConvertOpToLLVMPattern<Fp4ToFpOp> {
private:
  const AMD::TargetInfo &targetInfo;

public:
  Fp4ToFpOpPattern(LLVMTypeConverter &typeConverter,
                   const AMD::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<Fp4ToFpOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(Fp4ToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto elemType = op.getType().getElementType();
    assert(elemType == f16_ty || elemType == bf16_ty);
    bool toFp16 = elemType == f16_ty;

    auto xVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);

    SmallVector<Value> results;
    results.reserve(xVals.size() * 2);
    assert(xVals.size() % 4 == 0);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    for (int i = 0; i < xVals.size(); i += 4) {
      Value packedVec = b.undef(vec_ty(i8_ty, 4));
      for (int j : llvm::seq(4)) {
        Value v = xVals[i + j];
        packedVec = b.insert_element(packedVec, v, b.i32_val(j));
      }
      auto isa = targetInfo.getISAFamily();
      SmallVector<Value> upcast =
          upcast8xMxfp4_SW(rewriter, op, toFp16, packedVec, isa);
      results.append(upcast.begin(), upcast.end());
    }

    Value result = packLLElements(loc, getTypeConverter(), results, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // anonymous namespace

void mlir::triton::AMD::populateFp4ToFpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<Fp4ToFpOpPattern>(typeConverter, targetInfo, benefit);
}
