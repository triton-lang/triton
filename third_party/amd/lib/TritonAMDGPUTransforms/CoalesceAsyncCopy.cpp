#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/AsyncUtility.h"
#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/LayoutUtils.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-coalesce-async-copy"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUCOALESCEASYNCCOPY
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// On gfx9 global and buffer loads directly to shared memory need to write
// coalesced. This pattern converts the layout of the src, mask and other to
// ensure the owned data per thread is contiguous and does no exceed the
// supported load vector size.
struct CoalesceAsyncCopyWrites
    : public OpRewritePattern<ttg::AsyncCopyGlobalToLocalOp> {
  CoalesceAsyncCopyWrites(const triton::AMD::TargetInfo &targetInfo,
                          const DenseMap<ttg::AsyncCopyGlobalToLocalOp,
                                         unsigned> &asyncCopyContiguity,
                          MLIRContext *ctx)
      : OpRewritePattern(ctx), targetInfo{targetInfo},
        asyncCopyContiguity{std::move(asyncCopyContiguity)} {}

  LogicalResult matchAndRewrite(ttg::AsyncCopyGlobalToLocalOp copyOp,
                                PatternRewriter &rewriter) const override {
    auto src = copyOp.getSrc();
    auto dst = copyOp.getResult();
    Value mask = copyOp.getMask();
    Value other = copyOp.getOther();

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<ttg::MemDescType>(dst.getType());

    auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
    if (!blockedEnc)
      return rewriter.notifyMatchFailure(copyOp,
                                         "src encoding must be #blocked");

    // We start from the precomputed contiguity we got from AxisAnalysis.
    unsigned loadContig = 0;
    if (auto it = asyncCopyContiguity.find(copyOp);
        it != asyncCopyContiguity.end())
      loadContig = it->second;
    else
      return copyOp->emitError()
             << "No contiguity information about the copy op";
    assert(loadContig > 0);

    // Further restrict the contiguity based on the contiguity of the src to dst
    // layout e.g. if the order of the blocked and shared encoding is different
    // we can only load one element at a time or if the shared encoding is
    // swizzled we cannot exceed the vector size of the swizzling pattern
    LinearLayout regLayout = triton::gpu::toLinearLayout(srcTy);
    LinearLayout sharedLayout;
    auto paddedEnc =
        dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(dstTy.getEncoding());
    if (paddedEnc) {
      sharedLayout = paddedEnc.getLinearComponent();
    } else {
      sharedLayout = triton::gpu::toLinearLayout(dstTy);
    }
    auto regToSharedLayout = regLayout.invertAndCompose(sharedLayout);
    loadContig = std::min<unsigned>(loadContig,
                                    regToSharedLayout.getNumConsecutiveInOut());

    // Select the largest supported load width equal or smaller than loadContig
    auto elemBitWidth = dstTy.getElementTypeBitWidth();
    loadContig =
        fitToValidDirectToLdsVecSize(loadContig, elemBitWidth, targetInfo);

    if (loadContig == 0) {
      return rewriter.notifyMatchFailure(
          copyOp, "could not find layout config to create coalesced writes");
    }

    // Do not rewrite if we already use the correct contiguity (could be from a
    // previous rewrite)
    auto contigPerThread = ttg::getContigPerThread(srcTy);
    auto blockedContig = contigPerThread[blockedEnc.getOrder()[0]];
    if (!paddedEnc && blockedContig == loadContig) {
      return rewriter.notifyMatchFailure(copyOp,
                                         "already using the correct layout");
    }

    ttg::DistributedEncodingTrait newDistEnc;
    if (!paddedEnc) {
      // Get new blocked encoding with loadContig as sizePerThread in the
      // fastest dim
      assert(blockedContig >= loadContig);
      contigPerThread[blockedEnc.getOrder()[0]] = loadContig;
      int numWarps = triton::gpu::lookupNumWarps(copyOp);
      auto mod = copyOp->getParentOfType<ModuleOp>();
      int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
      newDistEnc = BlockedEncodingAttr::get(
          copyOp.getContext(), srcTy.getShape(), contigPerThread,
          blockedEnc.getOrder(), numWarps, threadsPerWarp,
          blockedEnc.getCTALayout());
    } else {
      // Adjust blocked to padded
      if (loadContig * elemBitWidth != 128) {
        return copyOp.emitError(
            "NYI: Only 128bit wide padded async loads are implemented!");
      }

      std::vector<std::vector<int>> regBases;
      std::vector<std::vector<int>> laneBases;
      std::vector<std::vector<int>> warpBases;

      // We adjust the global layout so lanes write contiguous into LDS. Note
      // that we do not look for performance here, so if the shared layout is
      // inefficient we will produce ineffect IR/assembly. The padded shared
      // layout composition need to take this into account

      auto *ctx = srcTy.getContext();
      StringAttr kOffset = StringAttr::get(ctx, "offset");

      int logWarpSize = llvm::Log2_32(64);
      auto ll = paddedEnc.getLinearComponent();

      int offsetIndex = 0;
      int logVecSize = 3;
      for (; offsetIndex < logVecSize; offsetIndex++) {
        regBases.push_back(ll.getBasis(kOffset, offsetIndex));
      }
      for (; offsetIndex < logVecSize + logWarpSize; offsetIndex++) {
        laneBases.push_back(ll.getBasis(kOffset, offsetIndex));
      }
      int logNumWarps = 3;
      for (; offsetIndex < logVecSize + logWarpSize + logNumWarps;
           offsetIndex++) {
        warpBases.push_back(ll.getBasis(kOffset, offsetIndex));
      }
      // Remaining basis are added as reg
      for (; offsetIndex < ll.getInDimSizeLog2(kOffset); offsetIndex++) {
        regBases.push_back(ll.getBasis(kOffset, offsetIndex));
      }

      auto transposeBases = [](std::vector<std::vector<int>> &vec) {
        for (auto &p : vec)
          std::swap(p[0], p[1]);
      };

      auto standardOutDims = triton::standardOutDimNames(ctx, srcTy.getRank());
      StringAttr kRegister = StringAttr::get(ctx, "register");
      StringAttr kLane = StringAttr::get(ctx, "lane");
      StringAttr kWarp = StringAttr::get(ctx, "warp");
      StringAttr kBlock = StringAttr::get(ctx, "block");

      triton::LinearLayout paddedLayout(
          {
              {kRegister, regBases},
              {kLane, laneBases},
              {kWarp, warpBases},
          },
          {standardOutDims[0], standardOutDims[1]});

      paddedLayout = triton::gpu::combineCtaCgaWithShape(
          paddedLayout, blockedEnc.getCTALayout(), srcTy.getShape());

      newDistEnc = ttg::LinearEncodingAttr::get(ctx, paddedLayout);
    }

    // Convert layout of src, mask and other to new encoding
    auto convertLayout = [&rewriter](auto loc, Value old, auto newEnc) {
      auto oldTy = cast<RankedTensorType>(old.getType());
      RankedTensorType newSrcTy = oldTy.cloneWithEncoding(newEnc);
      return rewriter.create<ttg::ConvertLayoutOp>(loc, newSrcTy, old);
    };

    auto loc = copyOp->getLoc();
    Value cvtSrc = convertLayout(loc, src, newDistEnc);

    if (mask)
      mask = convertLayout(loc, mask, newDistEnc);
    if (other)
      other = convertLayout(loc, other, newDistEnc);

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
  const DenseMap<ttg::AsyncCopyGlobalToLocalOp, unsigned> &asyncCopyContiguity;
};

} // anonymous namespace

class TritonAMDGPUCoalesceAsyncCopyPass
    : public impl::TritonAMDGPUCoalesceAsyncCopyBase<
          TritonAMDGPUCoalesceAsyncCopyPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = &getContext();

    triton::AMD::TargetInfo targetInfo(archGenerationName);

    mlir::RewritePatternSet patterns(context);

    switch (targetInfo.getISAFamily()) {
    case triton::AMD::ISAFamily::CDNA1:
    case triton::AMD::ISAFamily::CDNA2:
    case triton::AMD::ISAFamily::CDNA3:
    case triton::AMD::ISAFamily::CDNA4: {
      break;
    }
    default:
      return;
    }

    // Precompute the contiguity of all AsyncCopy ops based on the src and
    // mask contiguity/alignment to avoid rebuilding ModuleAxisInfoAnalysis
    // after every IR change.
    AMD::ModuleAxisInfoAnalysis axisAnalysis(m);
    DenseMap<ttg::AsyncCopyGlobalToLocalOp, unsigned> asyncCopyContiguity;
    m->walk([&](ttg::AsyncCopyGlobalToLocalOp copyOp) {
      unsigned contiguity =
          mlir::LLVM::AMD::getContiguity(copyOp.getSrc(), axisAnalysis);
      if (auto mask = copyOp.getMask()) {
        contiguity =
            std::min<unsigned>(contiguity, axisAnalysis.getMaskAlignment(mask));
      }
      asyncCopyContiguity.insert({copyOp, contiguity});
    });
    patterns.add<CoalesceAsyncCopyWrites>(targetInfo, asyncCopyContiguity,
                                          context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
