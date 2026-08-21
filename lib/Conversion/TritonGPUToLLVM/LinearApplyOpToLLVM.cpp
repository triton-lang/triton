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

    // Packed LLVM tensors omit broadcast register slots. Remove those same
    // slots before mapping logical basis elements back to their owners.
    LinearLayout basisLayout =
        ttg::toLinearLayout(basesType).removeZeroBasesAlongDim(kRegister);
    LinearLayout ctaLayout =
        basisLayout.sublayout({kRegister, kLane, kWarp}, {kDim});
    if (!ctaLayout.isSurjective()) {
      return rewriter.notifyMatchFailure(
          op, "linear_apply bases distributed across CTAs are unsupported");
    }

    LinearLayout warpLayout = basisLayout.sublayout({kRegister, kLane}, {kDim});
    bool isWarpLocal = warpLayout.isSurjective();

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value zero = b.i32_val(0);
    SmallVector<Value> indices =
        unpackUniqueTensorElements(loc, adaptor.getIndex(), rewriter);
    SmallVector<Value> results(indices.size(), zero);

    Value stagedBasis;
    Value ownerRegisterOffset;
    Value ownerLaneOffset;
    std::optional<LinearLayout> inverseWarpLayout;

    if (isWarpLocal) {
      inverseWarpLayout = warpLayout.pseudoinvert();

      // A warp or CTA may redundantly permute its complete copy of the basis.
      // Account for that uniform logical offset when finding source owners.
      if (!basisLayout.sublayoutIsZero({kWarp, kBlock}, {kDim})) {
        auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
        (void)laneId;
        Value blockId = targetInfo.getClusterCTAId(rewriter, loc);
        LinearLayout uniformLayout =
            basisLayout.sublayout({kWarp, kBlock}, {kDim});
        Value logicalOffset =
            applyLinearLayout(loc, rewriter, uniformLayout,
                              {{kWarp, warpId}, {kBlock, blockId}})
                .front()
                .second;
        auto ownerOffsets = applyLinearLayout(loc, rewriter, *inverseWarpLayout,
                                              {{kDim, logicalOffset}});
        for (auto [name, value] : ownerOffsets) {
          if (name == kRegister)
            ownerRegisterOffset = value;
          else if (name == kLane)
            ownerLaneOffset = value;
        }
        assert(ownerRegisterOffset && ownerLaneOffset &&
               "expected register and lane owner offsets");
      }
    } else {
      // The basis spans several warps. Stage one complete CTA-local copy, then
      // restore the canonical one-value-per-lane distribution.
      Value sharedBase =
          LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
      SmallVector<SmallVector<Value>> basisIndices =
          emitIndices(loc, rewriter, targetInfo, basisLayout, basesType,
                      /*withCTAOffset=*/true);
      assert(bases.size() == basisIndices.size());
      for (auto [basis, coordinates] : llvm::zip(bases, basisIndices)) {
        Value ptr =
            b.gep(sharedBase.getType(), i32_ty, sharedBase, coordinates[0]);
        targetInfo.storeShared(rewriter, loc, ptr, basis, b.true_val());
      }
      targetInfo.barrier(loc, rewriter, ttg::AddrSpace::Local);

      Value lane = getLaneId(rewriter, loc);
      Value logicalBasis = b.and_(lane, b.i32_val(kIndexBitWidth - 1));
      Value ptr = b.gep(sharedBase.getType(), i32_ty, sharedBase, logicalBasis);
      stagedBasis =
          targetInfo.loadShared(rewriter, loc, ptr, i32_ty, b.true_val());
    }

    // Broadcast each of the 32 basis values once and reuse it for every output
    // register. This also keeps the cross-warp fallback to one shared load.
    for (unsigned bit = 0; bit < kIndexBitWidth; ++bit) {
      Value basis;
      if (!isWarpLocal) {
        basis = targetInfo.shuffleIdx(rewriter, loc, stagedBasis, bit);
      } else {
        auto owner = inverseWarpLayout->apply({{kDim, static_cast<int>(bit)}});
        unsigned sourceRegister = 0;
        unsigned sourceLane = 0;
        for (auto [name, coordinate] : owner) {
          if (name == kRegister)
            sourceRegister = coordinate;
          else if (name == kLane)
            sourceLane = coordinate;
        }
        assert(sourceRegister < bases.size() &&
               "basis owner register is outside the packed tensor");

        if (!ownerRegisterOffset) {
          basis = targetInfo.shuffleIdx(rewriter, loc, bases[sourceRegister],
                                        static_cast<int>(sourceLane));
        } else {
          Value registerId =
              b.xor_(ownerRegisterOffset, b.i32_val(sourceRegister));
          Value source = bases[0];
          for (unsigned reg = 1; reg < bases.size(); ++reg) {
            source = b.select(b.icmp_eq(registerId, b.i32_val(reg)), bases[reg],
                              source);
          }
          Value laneId = b.xor_(ownerLaneOffset, b.i32_val(sourceLane));
          basis = targetInfo.shuffleIdx(rewriter, loc, source, laneId);
        }
      }
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
