#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
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
// The MMA operand layout is already determined by the shared-memory memdesc
// feeding the dot-like op. This pattern lifts a descriptor-load-backed tensor
// view chain into equivalent memdesc reshape/transpose ops, while keeping the
// chosen MMA sink layout unchanged.
//
// The optimization is intentionally narrow:
// - the chain must start at tt.descriptor_load
// - each tensor view along the lifted path must have one use
// - only shared-memory memdesc operands of dot-like ops are considered
class RewriteMmaOperandViewsToMemDesc
    : public OpInterfaceRewritePattern<triton::DotOpInterface> {
public:
  using OpInterfaceRewritePattern<
      triton::DotOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(triton::DotOpInterface dotOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<TCGen5MMAOp, TCGen5MMAScaledOp, WarpGroupDotOp>(dotOp))
      return failure();

    bool changed = false;
    for (OpOperand &operand : dotOp->getOpOperands()) {
      auto memDesc = dyn_cast<TypedValue<gpu::MemDescType>>(operand.get());
      if (!memDesc ||
          !isa<gpu::SharedMemorySpaceAttr>(memDesc.getType().getMemorySpace()))
        continue;
      changed |= succeeded(rewriteOperand(memDesc, rewriter));
    }

    return success(changed);
  }

private:
  static bool isTensorViewOp(Operation *op) {
    return isa<triton::ReshapeOp, triton::TransOp>(op);
  }

  static bool isMemDescViewOp(Operation *op) {
    return isa<gpu::MemDescReshapeOp, gpu::MemDescTransOp>(op);
  }

  LogicalResult rewriteOperand(TypedValue<gpu::MemDescType> memDesc,
                               PatternRewriter &rewriter) const {
    Value current = memDesc;

    // Strip trailing memdesc views so we can rewrite the producing local_alloc.
    while (Operation *def = current.getDefiningOp()) {
      if (!isMemDescViewOp(def))
        break;
      current = def->getOperand(0);
    }

    auto localAlloc = current.getDefiningOp<gpu::LocalAllocOp>();
    if (!localAlloc || !localAlloc.getSrc())
      return failure();

    Value localAllocSrc = localAlloc.getSrc();
    current = localAllocSrc;

    // Walk back to the base of the tensor view chain.
    while (Operation *def = current.getDefiningOp()) {
      if (!isTensorViewOp(def))
        break;
      current = def->getOperand(0);
    }

    // If there are no tensor views, there is nothing to lift.
    if (current == localAllocSrc)
      return failure();

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

    // Validate that the tensor view chain is a single-use path from the
    // descriptor load to the original local_alloc source.
    SmallVector<Operation *> tensorViewOps;
    Value path = current;
    while (path != localAllocSrc) {
      if (!path.hasOneUse())
        return failure();
      Operation *viewOp = *path.getUsers().begin();
      if (!isTensorViewOp(viewOp))
        return failure();
      tensorViewOps.push_back(viewOp);
      path = viewOp->getResult(0);
    }

    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(localAlloc);

    Value rewritten = gpu::LocalAllocOp::create(rewriter, localAlloc.getLoc(),
                                                descMemTy, current);

    for (Operation *viewOp : tensorViewOps) {
      if (auto trans = dyn_cast<triton::TransposeOpInterface>(viewOp)) {
        rewritten = gpu::MemDescTransOp::create(rewriter, viewOp->getLoc(),
                                                rewritten, trans.getOrder());
      } else {
        auto reshape = cast<triton::ReshapeOp>(viewOp);
        rewritten =
            gpu::MemDescReshapeOp::create(rewriter, reshape.getLoc(), rewritten,
                                          reshape.getType().getShape());
      }
    }

    assert(rewritten.getType() == localAlloc.getType() &&
           "rewrite must preserve local_alloc result type");
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
