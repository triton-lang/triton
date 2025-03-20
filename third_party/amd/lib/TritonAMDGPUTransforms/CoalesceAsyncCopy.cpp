#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-coalesce-async-copy"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = triton::gpu;

// On gfx9 global and buffer loads directly to shared memory need to write
// coalesced. This pass converts the layout of the src, mask and other to ensure
// the owned data per thread is contigious and does no exceed the supported
// load vector size to ensure coalesed writes
struct CoalesceAsyncCopyWrites
    : public OpRewritePattern<ttg::AsyncCopyGlobalToLocalOp> {
  CoalesceAsyncCopyWrites(const triton::AMD::TargetInfo &targetInfo,
                          MLIRContext *ctx,
                          triton::ModuleAxisInfoAnalysis &axisAnalysis)
      : OpRewritePattern(ctx), targetInfo{targetInfo},
        axisAnalysis(axisAnalysis) {}
  LogicalResult matchAndRewrite(ttg::AsyncCopyGlobalToLocalOp copyOp,
                                PatternRewriter &rewriter) const override {
    auto src = copyOp.getSrc();
    auto dst = copyOp.getResult();
    Value mask = copyOp.getMask();
    Value other = copyOp.getOther();

    // AxisAnalysis will not have data about the src if we already processed the
    // copyOp
    if (axisAnalysis.getAxisInfo(src) == nullptr) {
      return rewriter.notifyMatchFailure(copyOp, "already adjusted layout");
    }

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<ttg::MemDescType>(dst.getType());

    auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
    if (!blockedEnc)
      return rewriter.notifyMatchFailure(copyOp,
                                         "src encoding must be #blocked");

    auto sharedEnc =
        dyn_cast<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding());
    if (!sharedEnc)
      return rewriter.notifyMatchFailure(
          copyOp, "destination encoding must be #SwizzledShared");
    if (sharedEnc.getMaxPhase() > 1)
      return rewriter.notifyMatchFailure(
          copyOp, "swizzled shared encoding not supported");

    // Get the minimum contiguity based on the src and mask contiguity and
    // alignment
    unsigned loadContig = mlir::LLVM::AMD::getContiguity(src, axisAnalysis);
    if (mask) {
      loadContig =
          std::min<unsigned>(loadContig, axisAnalysis.getMaskAlignment(mask));
    }

    // Select the largest supported load width equal or smaller than loadContig
    auto elemBitWidth = dstTy.getElementTypeBitWidth();
    while (loadContig > 0 && !targetInfo.supportsDirectToLdsLoadBitWidth(
                                 loadContig * elemBitWidth)) {
      loadContig /= 2;
    }

    if (loadContig == 0) {
      return rewriter.notifyMatchFailure(
          copyOp, "could not find layout config to create coalesced writes");
    }

    // Do not rewrite if we already use the correct contiguity
    auto contigPerThread = ttg::getContigPerThread(srcTy);
    auto blockedContig = contigPerThread[blockedEnc.getOrder()[0]];
    if (blockedContig == loadContig) {
      return rewriter.notifyMatchFailure(copyOp,
                                         "already using the correct layout");
    }

    // Get new blocked encoding with loadContig as sizePerThread in the fastest
    // dim
    contigPerThread[blockedEnc.getOrder()[0]] = loadContig;
    int numWarps = triton::gpu::lookupNumWarps(copyOp);
    auto mod = copyOp->getParentOfType<ModuleOp>();
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    auto newBlockEnc = BlockedEncodingAttr::get(
        copyOp.getContext(), srcTy.getShape(), contigPerThread,
        blockedEnc.getOrder(), numWarps, threadsPerWarp,
        blockedEnc.getCTALayout());

    // Convert layout of src, mask and other to new encoding
    auto convertLayout = [&rewriter](auto loc, Value old, auto newEnc) {
      auto oldTy = cast<RankedTensorType>(old.getType());
      RankedTensorType newSrcTy = RankedTensorType::get(
          oldTy.getShape(), oldTy.getElementType(), newEnc);
      return rewriter.create<ttg::ConvertLayoutOp>(loc, newSrcTy, old);
    };

    auto loc = copyOp->getLoc();
    Value cvtSrc = convertLayout(loc, src, newBlockEnc);

    if (mask)
      mask = convertLayout(loc, mask, newBlockEnc);
    if (other)
      other = convertLayout(loc, other, newBlockEnc);

    rewriter.modifyOpInPlace(copyOp, [&]() {
      copyOp.getSrcMutable().assign(cvtSrc);
      if (mask)
        copyOp.getMaskMutable().assign(mask);
      if (other)
        copyOp.getOtherMutable().assign(other);
    });

    return success();
  }

private:
  const triton::AMD::TargetInfo &targetInfo;
  triton::ModuleAxisInfoAnalysis &axisAnalysis;
};

class TritonAMDGPUCoalesceAsyncCopyPass
    : public TritonAMDGPUCoalesceAsyncCopyBase<
          TritonAMDGPUCoalesceAsyncCopyPass> {
public:
  TritonAMDGPUCoalesceAsyncCopyPass(std::string archGenName) {
    this->archGenerationName = std::move(archGenName);
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = &getContext();

    triton::AMD::TargetInfo targetInfo(archGenerationName);
    triton::ModuleAxisInfoAnalysis axisInfoAnalysis(m);

    mlir::RewritePatternSet patterns(context);

    switch (targetInfo.getISAFamily()) {
    case triton::AMD::ISAFamily::CDNA1:
    case triton::AMD::ISAFamily::CDNA2:
    case triton::AMD::ISAFamily::CDNA3:
    case triton::AMD::ISAFamily::CDNA4:
      patterns.add<CoalesceAsyncCopyWrites>(targetInfo, context,
                                            axisInfoAnalysis);
      break;
    default:
      break;
    }

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass>
mlir::createTritonAMDGPUCoalesceAsyncCopyPass(std::string archGenName) {
  return std::make_unique<TritonAMDGPUCoalesceAsyncCopyPass>(
      std::move(archGenName));
}
