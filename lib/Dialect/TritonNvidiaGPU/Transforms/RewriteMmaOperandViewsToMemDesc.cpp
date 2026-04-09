#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include <algorithm>

namespace mlir::triton::nvidia_gpu {

namespace {

// Rewrite
//   tt.reshape / tt.trans -> local_alloc -> [memdesc views] -> mma
// into
//   local_alloc -> memdesc reshape / trans -> [memdesc views] -> mma
//
// The MMA operand layout is determined by the sink memdesc already feeding the
// dot-like op. This pattern back-propagates that layout through the tensor
// reshape/transpose chain, hoists local_alloc to the base tensor feeding that
// view chain, and replays those tensor views as memdesc reshape/transpose
// ops so the original local_alloc type is preserved.
class RewriteMmaOperandViewsToMemDescForDotOp
    : public OpInterfaceRewritePattern<triton::DotOpInterface> {
public:
  using OpInterfaceRewritePattern<
      triton::DotOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(triton::DotOpInterface dotOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<TCGen5MMAOp, TCGen5MMAScaledOp, WarpGroupDotOp>(dotOp))
      return failure();

    bool changed = false;

    if (rewriteOperand(dotOp.getA(), rewriter).succeeded())
      changed = true;

    if (rewriteOperand(dotOp.getB(), rewriter).succeeded())
      changed = true;

    return success(changed);
  }

private:
  static FailureOr<gpu::MemDescType>
  pushLayoutBackward(gpu::MemDescType resultTy, Operation *op) {
    if (auto reshape = dyn_cast<triton::ReshapeOp>(op)) {
      gpu::MemDescType srcTy;
      if (failed(gpu::MemDescReshapeOp::inferReturnTypes(
              resultTy.getContext(), reshape.getLoc(), resultTy,
              reshape.getSrc().getType().getShape(), srcTy)))
        return failure();
      return srcTy;
    }

    auto trans = cast<triton::TransOp>(op);
    Attribute srcEnc = inferSrcEncoding(op, resultTy.getEncoding());
    if (!srcEnc)
      return failure();
    return gpu::MemDescType::get(
        trans.getSrc().getType().getShape(), resultTy.getElementType(), srcEnc,
        resultTy.getMemorySpace(), resultTy.getMutableMemory());
  }

  static Value replayTensorViews(PatternRewriter &rewriter, Value value,
                                 ArrayRef<Operation *> steps) {
    Value rewritten = value;
    for (Operation *op : steps) {
      if (auto reshape = dyn_cast<triton::ReshapeOp>(op)) {
        rewritten = gpu::MemDescReshapeOp::create(
            rewriter, op->getLoc(), rewritten, reshape.getType().getShape());
      } else {
        auto trans = cast<triton::TransOp>(op);
        rewritten = gpu::MemDescTransOp::create(rewriter, op->getLoc(),
                                                rewritten, trans.getOrder());
      }
    }
    return rewritten;
  }

  static Value peelMemDescViews(Value value) {
    Value current = value;
    while (auto view = current.getDefiningOp()) {
      if (auto reshape = dyn_cast<gpu::MemDescReshapeOp>(view)) {
        current = reshape.getSrc();
        continue;
      }
      if (auto trans = dyn_cast<gpu::MemDescTransOp>(view)) {
        current = trans.getSrc();
        continue;
      }
      break;
    }
    return current;
  }

  LogicalResult rewriteOperand(Value operand, PatternRewriter &rewriter) const {
    if (!isa<gpu::MemDescType>(operand.getType()))
      return failure();

    Value beforeTrailing = peelMemDescViews(operand);
    auto localAlloc = beforeTrailing.getDefiningOp<gpu::LocalAllocOp>();
    if (!localAlloc || !localAlloc.getSrc())
      return failure();

    Value baseTensor = localAlloc.getSrc();
    SmallVector<Operation *> tensorReplaySteps;
    gpu::MemDescType baseMemTy = localAlloc.getType();
    while (auto view = baseTensor.getDefiningOp()) {
      if (!isa<triton::ReshapeOp, triton::TransOp>(view))
        break;
      auto srcTy = pushLayoutBackward(baseMemTy, view);
      if (failed(srcTy))
        return failure();
      tensorReplaySteps.push_back(view);
      baseMemTy = *srcTy;
      baseTensor = view->getOperand(0);
    }
    if (tensorReplaySteps.empty())
      return failure();

    std::reverse(tensorReplaySteps.begin(), tensorReplaySteps.end());

    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(localAlloc);

    Value rewritten = gpu::LocalAllocOp::create(rewriter, localAlloc.getLoc(),
                                                baseMemTy, baseTensor);
    rewritten = replayTensorViews(rewriter, rewritten, tensorReplaySteps);

    auto rewrittenSinkTy = cast<gpu::MemDescType>(rewritten.getType());

    rewriter.replaceOp(localAlloc, rewritten);
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
    patterns.add<RewriteMmaOperandViewsToMemDescForDotOp>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::triton::nvidia_gpu
