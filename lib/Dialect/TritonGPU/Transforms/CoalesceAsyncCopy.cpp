#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/CoalesceUtils.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUCOALESCEASYNCCOPY
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

static Value convertValueLayout(Value src, Attribute enc,
                                PatternRewriter &rewriter) {
  auto ty = cast<RankedTensorType>(src.getType());
  auto newTy = ty.cloneWithEncoding(enc);
  auto cvt = ConvertLayoutOp::create(rewriter, src.getLoc(), newTy, src);
  return cvt.getResult();
}

static void retargetCopyOperandsToEncoding(
    AsyncCopyGlobalToLocalOp copyOp, Attribute newEncoding,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternRewriter &rewriter) {
  Value src = copyOp.getSrc();
  Value mask = copyOp.getMask();
  Value other = copyOp.getOther();

  // insert cvt's after src, mask, and other
  src = convertValueLayout(src, newEncoding, rewriter);
  if (mask)
    mask = convertValueLayout(mask, newEncoding, rewriter);
  if (other)
    other = convertValueLayout(other, newEncoding, rewriter);

  unsigned contiguity = axisInfoAnalysis.getContiguity(src);
  if (mask)
    contiguity =
        std::min<unsigned>(contiguity, axisInfoAnalysis.getMaskAlignment(mask));

  rewriter.modifyOpInPlace(copyOp, [&]() {
    copyOp.getSrcMutable().assign(src);
    if (mask)
      copyOp.getMaskMutable().assign(mask);
    if (other)
      copyOp.getOtherMutable().assign(other);
    copyOp.setContiguity(contiguity);
  });
}

// cp.async combines a global load and a shared-memory store. When the
// shared layout limits the copy width below the source layout's per-thread
// contiguity, splitting the copy can break coalescing across lanes.
// Clip the source layout's contiguous dimension to the copy width computed
// from the source and destination LinearLayouts.
struct ClipAsyncCopySizePerThread
    : public OpRewritePattern<AsyncCopyGlobalToLocalOp> {
  ModuleAxisInfoAnalysis &axisInfoAnalysis;
  using OpRewritePattern::OpRewritePattern;
  ClipAsyncCopySizePerThread(ModuleAxisInfoAnalysis &axisInfoAnalysis,
                             MLIRContext *context)
      : OpRewritePattern(context), axisInfoAnalysis(axisInfoAnalysis) {}

  LogicalResult matchAndRewrite(AsyncCopyGlobalToLocalOp copyOp,
                                PatternRewriter &rewriter) const override {
    Value src = copyOp.getSrc();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<MemDescType>(copyOp.getResult().getType());
    auto blockedEnc = dyn_cast<BlockedEncodingAttr>(srcTy.getEncoding());
    if (!blockedEnc)
      return rewriter.notifyMatchFailure(copyOp,
                                         "src must be of blocked encoding");
    if (!isa<SwizzledSharedEncodingAttr, NVMMASharedEncodingAttr>(
            dstTy.getEncoding()))
      return failure();
    // Padded/packed NVMMA encodings are outside this per-element clipping
    // rule. In particular, packed FP6 has no per-element LinearLayout.
    if (auto nvmma = dyn_cast<NVMMASharedEncodingAttr>(dstTy.getEncoding())) {
      if (nvmma.getPadding() != NVMmaPadding::None)
        return rewriter.notifyMatchFailure(copyOp,
                                           "padded NVMMA copies are not handled");
    }

    // obtain max contiguous copy size
    // Note this can be further optimized, as copyContigSize can be even
    // smaller when lowering, depending on contiguity and mask alignment
    // (see AsyncCopyGlobalToLocalOpConversion)
    LinearLayout regLayout = triton::gpu::toLinearLayout(srcTy);
    LinearLayout sharedLayout = triton::gpu::toLinearLayout(dstTy);
    auto copyContigSize =
        regLayout.invertAndCompose(sharedLayout).getNumConsecutiveInOut();

    // obtain block sizePerThread along contig dim
    auto contigPerThread = getContigPerThread(srcTy);
    auto blockContigSize = contigPerThread[blockedEnc.getOrder()[0]];

    if (blockContigSize <= copyContigSize)
      return rewriter.notifyMatchFailure(
          copyOp,
          "blocked sizePerThread along contiguous dim must be greater than the "
          "max contiguous copy size ");

    contigPerThread[blockedEnc.getOrder()[0]] = copyContigSize;

    // obtain new blockedEnc based on clipped sizePerThread
    auto mod = copyOp->getParentOfType<ModuleOp>();
    int numWarps = lookupNumWarps(copyOp);
    int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
    auto newBlockEnc = BlockedEncodingAttr::get(
        copyOp.getContext(), srcTy.getShape(), contigPerThread,
        blockedEnc.getOrder(), numWarps, threadsPerWarp,
        blockedEnc.getCGALayout());

    retargetCopyOperandsToEncoding(copyOp, newBlockEnc, axisInfoAnalysis,
                                   rewriter);

    return success();
  }
};

// For cheap loads we usually pick the layout based on users but when converting
// to async_cp the layout of the copy is independent of the layout of the users
// so picking a coalesced layout is better.
struct CoalesceCheapAsyncCopyGlobalToLocal
    : public OpRewritePattern<AsyncCopyGlobalToLocalOp> {
  ModuleAxisInfoAnalysis &axisInfoAnalysis;
  DenseMap<AsyncCopyGlobalToLocalOp, Attribute> &coalescedAsyncCopyMap;
  using OpRewritePattern::OpRewritePattern;
  CoalesceCheapAsyncCopyGlobalToLocal(
      ModuleAxisInfoAnalysis &axisInfoAnalysis,
      DenseMap<AsyncCopyGlobalToLocalOp, Attribute> &coalescedAsyncCopyMap,
      MLIRContext *context)
      : OpRewritePattern(context), axisInfoAnalysis(axisInfoAnalysis),
        coalescedAsyncCopyMap(coalescedAsyncCopyMap) {}

  LogicalResult matchAndRewrite(AsyncCopyGlobalToLocalOp copyOp,
                                PatternRewriter &rewriter) const override {
    Value src = copyOp.getSrc();
    RankedTensorType srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<MemDescType>(copyOp.getResult().getType());
    int numWarps = triton::gpu::lookupNumWarps(copyOp);
    auto mod = copyOp->getParentOfType<ModuleOp>();
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    int64_t size = srcTy.getNumElements();
    // Assume the expensive copies are already coalesced.
    // Skip dtype smaller than 32 bits to avoid problems with contiguity.
    if (size >= numWarps * threadsPerWarp ||
        dstTy.getElementTypeBitWidth() < 32)
      return failure();
    auto shapePerCTA = triton::gpu::getShapePerCTA(dstTy);

    auto newEnc = coalescedAsyncCopyMap[copyOp];
    if (newEnc == nullptr || newEnc == srcTy.getEncoding())
      return failure();

    retargetCopyOperandsToEncoding(copyOp, newEnc, axisInfoAnalysis, rewriter);

    return success();
  }
};

struct CoalesceAsyncCopyPass
    : impl::TritonGPUCoalesceAsyncCopyBase<CoalesceAsyncCopyPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    triton::ModuleAxisInfoAnalysis axisInfoAnalysis(m);
    CoalescedEncodingAnalysis coalescedEncodingAnalysis(axisInfoAnalysis);
    // Collect the coalesced encoding first as changing the IR invalidates the
    // axis analysis.
    DenseMap<AsyncCopyGlobalToLocalOp, Attribute> coalescedAsyncCopyMap;
    m.walk([&](AsyncCopyGlobalToLocalOp copyOp) {
      auto dstTy = cast<MemDescType>(copyOp.getResult().getType());
      int numWarps = triton::gpu::lookupNumWarps(copyOp);
      int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(m);
      auto cgaLayout = triton::gpu::getCGALayout(dstTy.getEncoding());
      auto shapePerCTA = triton::gpu::getShapePerCTA(dstTy);
      coalescedAsyncCopyMap[copyOp] = coalescedEncodingAnalysis.getEncoding(
          copyOp, numWarps, threadsPerWarp, cgaLayout, shapePerCTA);
    });

    MLIRContext *context = &getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<ClipAsyncCopySizePerThread>(axisInfoAnalysis, context);
    patterns.add<CoalesceCheapAsyncCopyGlobalToLocal>(
        axisInfoAnalysis, coalescedAsyncCopyMap, context);

    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
