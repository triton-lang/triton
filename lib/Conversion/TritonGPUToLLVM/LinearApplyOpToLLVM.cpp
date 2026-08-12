#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

using namespace mlir;
using namespace mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

constexpr unsigned kIndexBitWidth = 32;

struct LinearApplyOpConversion
    : public ConvertOpToLLVMPattern<triton::LinearApplyOp> {
  LinearApplyOpConversion(LLVMTypeConverter &typeConverter,
                          const TargetInfoBase &targetInfo,
                          PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::LinearApplyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();
    auto basesType = cast<RankedTensorType>(op.getBases().getType());
    SmallVector<Value> bases =
        unpackUniqueTensorElements(loc, adaptor.getBases(), rewriter);

    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");
    StringAttr kDim = str_attr("dim0");

    // Each lane owns exactly one basis value, and lane i owns bases[i]. A
    // wave64 duplicates lanes 0-31 in lanes 32-63; warps and CTAs duplicate the
    // complete basis. This lets every basis lookup use its bit as the shuffle
    // source lane without inverting or converting the basis layout.
    LinearLayout basisLayout = ttg::toLinearLayout(basesType);
    auto notifyInvalidBasisLayout = [&]() {
      return rewriter.notifyMatchFailure(
          op, "bases must have one value per lane, with basis i owned by lane "
              "i and replicated in every warp and CTA");
    };
    if (bases.size() != 1 ||
        basisLayout.getInDimSize(kLane) < kIndexBitWidth ||
        !basisLayout.sublayoutIsZero({kRegister, kWarp, kBlock}, {kDim})) {
      return notifyInvalidBasisLayout();
    }

    unsigned laneBits = basisLayout.getInDimSizeLog2(kLane);
    for (unsigned laneBit = 0; laneBit < laneBits; ++laneBit) {
      int32_t expected = laneBit < 5 ? int32_t{1} << laneBit : 0;
      if (basisLayout.getBasis(kLane, laneBit, kDim) != expected) {
        return notifyInvalidBasisLayout();
      }
    }

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value zero = b.i32_val(0);
    SmallVector<Value> indices =
        unpackUniqueTensorElements(loc, adaptor.getIndex(), rewriter);
    SmallVector<Value> results(indices.size(), zero);

    // Broadcast each runtime basis once, then reuse it for all
    // output registers owned by this thread. Keeping the bit loop outermost
    // avoids both a [num_elements, bitwidth] temporary and a shuffle per output
    // register.
    for (unsigned bit = 0; bit < kIndexBitWidth; ++bit) {
      Value basis = targetInfo.shuffleIdx(rewriter, loc, bases[0], bit);
      Value mask = b.i32_val(uint32_t{1} << bit);

      for (auto [i, index] : llvm::enumerate(indices)) {
        Value isSet = b.icmp_ne(b.and_(index, mask), zero);
        results[i] = b.xor_(results[i], b.select(isSet, basis, zero));
      }
    }

    rewriter.replaceOp(op, packUniqueTensorElements(loc, getTypeConverter(),
                                                    results, rewriter,
                                                    op.getResult().getType()));
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateLinearApplyOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<LinearApplyOpConversion>(typeConverter, targetInfo, benefit);
}
