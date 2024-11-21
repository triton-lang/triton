#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include <memory>

namespace tt = mlir::triton;

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUCOALESCEASYNCCOPY
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

// This pass currently only applies if the following are all true...
//   1) Operand A for WGMMA is to be loaded in registers
//   2) We upcast operand A in registers before the WGMMA
//      (downcasting is not yet supported)
//   3) Pipelining is enabled for loading A
//
// ...then for the AsyncCopyGlobalToLocal op, the SharedEncoding
// vec will be less than BlockedEncoding's sizePerThread for k-dim. E.g. if
// we're upcasting from int8 to bf16, then shared vec is 8 and sizePerThread
// for k is 16. In this case, AsyncCopyGlobalToLocal will generate two
// 8-byte-cp.async's for each contiguous 16B global data owned by each
// thread. This breaks coalescing (i.e. results 2x the minimum required
// transactions).
//
// This issue occurs for cp.async because it combines load and store into one
// instruction. The fix is to clip each dim of sizePerThread by shared vec, so
// that the vectorization of load and store are equal along the contiguous
// dimension. In the above example, each thread will then only own 8B contiguous
// global data.
struct ClipAsyncCopySizePerThread
    : public OpRewritePattern<AsyncCopyGlobalToLocalOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AsyncCopyGlobalToLocalOp copyOp,
                                PatternRewriter &rewriter) const override {
    Value src = copyOp.getSrc();
    Value mask = copyOp.getMask();
    Value other = copyOp.getOther();

    auto inputTy = cast<RankedTensorType>(src.getType());
    auto blockEnc = cast<BlockedEncodingAttr>(inputTy.getEncoding());
    auto resultTy = cast<tt::MemDescType>(copyOp.getResult().getType());
    auto sharedEnc = cast<SharedEncodingAttr>(resultTy.getEncoding());
    auto sharedVec = sharedEnc.getVec();

    // clip each dim of sizePerThread by its respective dim in vec
    SmallVector<unsigned> newSizePerThread;
    llvm::transform(blockEnc.getSizePerThread(),
                    std::back_inserter(newSizePerThread),
                    [&](auto size) { return std::min(size, sharedVec); });

    if (newSizePerThread == blockEnc.getSizePerThread())
      return rewriter.notifyMatchFailure(copyOp,
          "at least one dimension of blocked sizePerThread must be greater than shared vec");

    // obtain new blockedEnc based on clipped sizePerThread
    auto mod = copyOp->getParentOfType<ModuleOp>();
    int numWarps = TritonGPUDialect::getNumWarps(mod);
    int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
    auto newBlockEnc = BlockedEncodingAttr::get(
        copyOp.getContext(), inputTy.getShape(), newSizePerThread,
        blockEnc.getOrder(), numWarps, threadsPerWarp,
        blockEnc.getCTALayout());

    // insert cvt's after src, mask, and other
    auto convertBlockLayout = [&](Value src, BlockedEncodingAttr enc) {
      auto ty = cast<TensorType>(src.getType());
      auto newTy =
          RankedTensorType::get(ty.getShape(), ty.getElementType(), enc);
      auto cvt = rewriter.create<ConvertLayoutOp>(copyOp->getLoc(), newTy, src);
      return cvt.getResult();
    };
    src = convertBlockLayout(src, newBlockEnc);
    if (mask)
      mask = convertBlockLayout(mask, newBlockEnc);
    if (other)
      other = convertBlockLayout(other, newBlockEnc);

    // replace the asyncCopy
    auto newCopyOp = rewriter.create<AsyncCopyGlobalToLocalOp>(
        copyOp.getLoc(), src, copyOp.getResult(), mask, other,
        copyOp.getCache(), copyOp.getEvict(), copyOp.getIsVolatile());
    rewriter.replaceOp(copyOp, newCopyOp);

    return success();
  }
};

class CoalesceAsyncCopyPass
    : public impl::TritonGPUCoalesceAsyncCopyBase<
          CoalesceAsyncCopyPass> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = &getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<ClipAsyncCopySizePerThread>(context);

    if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
