#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace mlir::triton::nvidia_gpu {

namespace {

// Rewrite
//   desc_load -> tt.reshape / tt.trans -> local_alloc -> [memdesc views] -> mma
// into
//   desc_load -> local_alloc -> memdesc reshape / trans -> [memdesc views] ->
//   mma
//
// The MMA operand layout is determined by the sink memdesc already feeding the
// dot-like op. This pattern back-propagates that layout through the tensor
// reshape/transpose chain, hoists local_alloc to the descriptor_load result,
// and then replays the same views as memdesc reshape/transpose ops.
//
// The rewrite only applies when the chain starts at a descriptor_load and all
// tensor views can be re-created as memdesc view ops.
class RewriteMmaOperandViewsToMemDesc
    : public OpInterfaceRewritePattern<triton::DotOpInterface> {
public:
  using OpInterfaceRewritePattern<
      triton::DotOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(triton::DotOpInterface dotOp,
                                PatternRewriter &rewriter) const override {
    // This pass is not specific to dotOps, we could generalise it to any
    // op that has a shared memory operand.
    if (!isa<TCGen5MMAOp, TCGen5MMAScaledOp, WarpGroupDotOp>(dotOp))
      return failure();

    bool changed = false;

    for (OpOperand &operand : dotOp->getOpOperands()) {
      auto memDesc = dyn_cast<TypedValue<gpu::MemDescType>>(operand.get());
      if (!memDesc ||
          !isa<gpu::SharedMemorySpaceAttr>(memDesc.getType().getMemorySpace()))
        continue;
      changed |= rewriteOperand(memDesc, rewriter).succeeded();
    }

    return success(changed);
  }

private:
  LogicalResult rewriteOperand(TypedValue<gpu::MemDescType> memDesc,
                               PatternRewriter &rewriter) const {
    PatternRewriter::InsertionGuard guard(rewriter);
    Value current = memDesc;

    // Consume potential memdesc views.
    while (isa<gpu::MemDescReshapeOp, gpu::MemDescTransOp>(
        current.getDefiningOp())) {
      current = current.getDefiningOp()->getOperand(0);
    }

    // Find the local_alloc that produces the current value.
    auto localAlloc = current.getDefiningOp<gpu::LocalAllocOp>();
    if (!localAlloc || !localAlloc.getSrc())
      return failure();

    current = localAlloc.getSrc();
    Value localAllocSrc = current;

    // Consume all the tensor views
    while (isa<triton::ReshapeOp, triton::TransOp>(current.getDefiningOp())) {
      current = current.getDefiningOp()->getOperand(0);
    }
    // If we didn't see any tensor views, there is nothing to rewrite.
    if (current == localAllocSrc)
      return failure();

    // Get the MemDescType associated with the descriptor load.
    auto descLoad = current.getDefiningOp<DescriptorLoadOp>();
    if (!descLoad)
      return failure();
    RankedTensorType blockTy = descLoad.getDesc().getType().getBlockType();
    auto descriptorSharedEnc =
        cast<gpu::SharedEncodingTrait>(blockTy.getEncoding());

    gpu::MemDescType descMemTy = gpu::MemDescType::get(
        blockTy.getShape(), blockTy.getElementType(), descriptorSharedEnc,
        localAlloc.getType().getMemorySpace(),
        localAlloc.getType().getMutableMemory());

    // Lift the local alloc next to the descriptor load.
    rewriter.setInsertionPoint(localAlloc);
    Value rewritten = gpu::LocalAllocOp::create(rewriter, localAlloc.getLoc(),
                                                descMemTy, current);

    // Using descMemTy instead of currentTy is correct because backpropagating
    // the layout should yield the same result as forward propagating the layout
    // and memdesc_{reshape,trans} propagation rules are the same as those from
    // their tensor counterparts.
    while (current != localAllocSrc) {
      if (!current.hasOneUse())
        return failure();
      Operation *viewOp = *current.getUsers().begin();
      if (auto trans = dyn_cast<triton::TransposeOpInterface>(viewOp)) {
        rewritten = gpu::MemDescTransOp::create(rewriter, viewOp->getLoc(),
                                                rewritten, trans.getOrder());
      } else {
        auto reshape = cast<triton::ReshapeOp>(viewOp);
        rewritten =
            gpu::MemDescReshapeOp::create(rewriter, reshape.getLoc(), rewritten,
                                          reshape.getType().getShape());
      }
      current = viewOp->getResult(0);
    }

    localAlloc.replaceAllUsesWith(rewritten);
    return success();
  }
};

} // namespace

#define GEN_PASS_DEF_TRITONNVIDIAGPUREWRITEMMAOPERANDVIEWSTOMEMDESCPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

class TritonNvidiaGPURewriteMmaOperandViewsToMemDescPass
    : public impl::TritonNvidiaGPURewriteMmaOperandViewsToMemDescPassBase<
          TritonNvidiaGPURewriteMmaOperandViewsToMemDescPass> {
public:
  using BaseT = impl::TritonNvidiaGPURewriteMmaOperandViewsToMemDescPassBase<
      TritonNvidiaGPURewriteMmaOperandViewsToMemDescPass>;
  using BaseT::BaseT;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RewriteMmaOperandViewsToMemDesc>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::triton::nvidia_gpu
