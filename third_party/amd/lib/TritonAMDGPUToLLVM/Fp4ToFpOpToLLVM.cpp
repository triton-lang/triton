#include "PatternTritonGPUOpToLLVM.h"

#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Fp4ToFpOpToLLVMBase.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using ::mlir::LLVM::AMD::upcast8xMxfp4_SW;

namespace {

class Fp4ToFpOpPattern : public Fp4ToFpOpConversionBase {
private:
  const AMD::TargetInfo &targetInfo;

public:
  Fp4ToFpOpPattern(LLVMTypeConverter &typeConverter,
                   const AMD::TargetInfo &targetInfo, PatternBenefit benefit)
      : Fp4ToFpOpConversionBase(typeConverter, benefit),
        targetInfo(targetInfo) {}

protected:
  std::array<Value, 8> upcastPackedFp4(Fp4ToFpOp op,
                                       ConversionPatternRewriter &rewriter,
                                       Value packedVec,
                                       Type elemType) const override {
    auto values = upcast8xMxfp4_SW(rewriter, op, elemType.isF16(), packedVec,
                                   targetInfo.getISAFamily());
    assert(values.size() == 8);
    std::array<Value, 8> results;
    std::copy(values.begin(), values.end(), results.begin());
    return results;
  }
};
} // anonymous namespace

void mlir::triton::AMD::populateFp4ToFpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<Fp4ToFpOpPattern>(typeConverter, targetInfo, benefit);
}
