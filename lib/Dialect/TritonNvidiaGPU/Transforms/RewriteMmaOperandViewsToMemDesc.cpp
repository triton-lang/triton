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
//   tt.reshape / tt.trans -> local_alloc -> [memdesc views] -> mma
// into
//   local_alloc -> memdesc reshape / trans -> [memdesc views] -> mma
//
// The MMA operand layout is determined by the sink memdesc already feeding the
// dot-like op. This pattern back-propagates that layout through the tensor
// reshape/transpose chain, hoists local_alloc to the base tensor feeding that
// view chain, and then replays the same views as memdesc reshape/transpose
// ops.
template <typename DotOpTy>
class RewriteMmaOperandViewsToMemDescForDotOp
    : public OpRewritePattern<DotOpTy> {
public:
  using OpRewritePattern<DotOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(DotOpTy dotOp,
                                PatternRewriter &rewriter) const override {
    Value oldA = dotOp.getA();
    Value oldB = dotOp.getB();
    bool changed = false;

    if (rewriteOperand(dotOp.getAMutable(), rewriter).succeeded()) {
      oldA.replaceAllUsesExcept(dotOp.getA(), dotOp.getOperation());
      changed = true;
    }

    if (rewriteOperand(dotOp.getBMutable(), rewriter).succeeded()) {
      oldB.replaceAllUsesExcept(dotOp.getB(), dotOp.getOperation());
      changed = true;
    }

    return success(changed);
  }

private:
  struct ViewStep {
    enum Kind { Reshape, Transpose } kind;
    SmallVector<int64_t> srcShape;
    SmallVector<int64_t> dstShape;
    SmallVector<int32_t> order;
    Operation *op;
    Location loc;
  };

  template <typename ReshapeOpTy, typename TransOpTy>
  static std::tuple<Value, SmallVector<ViewStep>>
  collectViewSteps(Value value) {
    Value current = value;
    SmallVector<ViewStep> replaySteps;
    while (true) {
      if (auto reshape = current.template getDefiningOp<ReshapeOpTy>()) {
        auto srcTy = reshape.getSrc().getType();
        auto dstTy = reshape.getType();
        replaySteps.push_back(ViewStep{ViewStep::Reshape,
                                       SmallVector<int64_t>(srcTy.getShape()),
                                       SmallVector<int64_t>(dstTy.getShape()),
                                       {},
                                       reshape.getOperation(),
                                       reshape.getLoc()});
        current = reshape.getSrc();
        continue;
      }
      if (auto trans = current.template getDefiningOp<TransOpTy>()) {
        SmallVector<int32_t> order(trans.getOrder().begin(),
                                   trans.getOrder().end());
        auto srcTy = trans.getSrc().getType();
        auto dstTy = trans.getType();
        replaySteps.push_back(ViewStep{
            ViewStep::Transpose, SmallVector<int64_t>(srcTy.getShape()),
            SmallVector<int64_t>(dstTy.getShape()), std::move(order),
            trans.getOperation(), trans.getLoc()});
        current = trans.getSrc();
        continue;
      }
      break;
    }
    return {current, llvm::to_vector(llvm::reverse(replaySteps))};
  }

  static FailureOr<gpu::MemDescType>
  inferViewStepBackward(gpu::MemDescType resultTy, const ViewStep &step) {
    assert(resultTy.getShape() == ArrayRef<int64_t>(step.dstShape) &&
           "backward inference must start from the view step destination "
           "shape");
    if (step.kind == ViewStep::Reshape) {
      gpu::MemDescType srcTy;
      if (failed(gpu::MemDescReshapeOp::inferReturnTypes(
              resultTy.getContext(), step.loc, resultTy, step.srcShape, srcTy)))
        return failure();
      return srcTy;
    }
    Attribute srcEnc = inferSrcEncoding(step.op, resultTy.getEncoding());
    if (!srcEnc)
      return failure();
    return gpu::MemDescType::get(step.srcShape, resultTy.getElementType(),
                                 srcEnc, resultTy.getMemorySpace(),
                                 resultTy.getMutableMemory());
  }

  static Value replayViewSteps(PatternRewriter &rewriter, Value value,
                               ArrayRef<ViewStep> steps) {
    Value rewritten = value;
    for (const ViewStep &step : steps) {
      if (step.kind == ViewStep::Reshape) {
        rewritten = gpu::MemDescReshapeOp::create(rewriter, step.loc, rewritten,
                                                  step.dstShape);
      } else {
        rewritten = gpu::MemDescTransOp::create(rewriter, step.loc, rewritten,
                                                step.order);
      }
    }
    return rewritten;
  }

  static void assertEquivalentMemDescType(gpu::MemDescType actualTy,
                                          gpu::MemDescType expectedTy,
                                          const char *message) {
    assert(actualTy.getShape() == expectedTy.getShape() &&
           actualTy.getElementType() == expectedTy.getElementType() &&
           gpu::areLayoutsEquivalent(
               expectedTy.getShape(),
               cast<gpu::LayoutEncodingTrait>(actualTy.getEncoding()),
               cast<gpu::LayoutEncodingTrait>(expectedTy.getEncoding())) &&
           message);
  }

  LogicalResult rewriteOperand(OpOperand &operand,
                               PatternRewriter &rewriter) const {
    Value orig = operand.get();
    auto origTy = dyn_cast<gpu::MemDescType>(orig.getType());
    if (!origTy)
      return failure();

    auto [beforeTrailing, trailingMemDescReplaySteps] =
        collectViewSteps<gpu::MemDescReshapeOp, gpu::MemDescTransOp>(orig);

    auto localAlloc =
        beforeTrailing.template getDefiningOp<gpu::LocalAllocOp>();
    if (!localAlloc || !localAlloc.getSrc())
      return failure();

    auto [baseTensor, tensorReplaySteps] =
        collectViewSteps<triton::ReshapeOp, triton::TransOp>(
            localAlloc.getSrc());
    if (tensorReplaySteps.empty())
      return failure();

    gpu::MemDescType baseMemTy = localAlloc.getType();
    for (const ViewStep &step : llvm::reverse(tensorReplaySteps)) {
      auto srcTy = inferViewStepBackward(baseMemTy, step);
      if (failed(srcTy))
        return failure();
      baseMemTy = *srcTy;
    }

    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(localAlloc);

    Value rewritten = gpu::LocalAllocOp::create(rewriter, localAlloc.getLoc(),
                                                baseMemTy, baseTensor);
    auto sinkTy = localAlloc.getType();

    rewritten = replayViewSteps(rewriter, rewritten, tensorReplaySteps);

    auto rewrittenSinkTy = cast<gpu::MemDescType>(rewritten.getType());
    assertEquivalentMemDescType(
        rewrittenSinkTy, sinkTy,
        "rewrite must preserve the intermediate sink memdesc");

    rewritten =
        replayViewSteps(rewriter, rewritten, trailingMemDescReplaySteps);

    auto rewrittenTy = cast<gpu::MemDescType>(rewritten.getType());
    assertEquivalentMemDescType(rewrittenTy, origTy,
                                "rewrite must preserve the final memdesc");

    operand.assign(rewritten);
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
    patterns.add<RewriteMmaOperandViewsToMemDescForDotOp<TCGen5MMAOp>,
                 RewriteMmaOperandViewsToMemDescForDotOp<TCGen5MMAScaledOp>,
                 RewriteMmaOperandViewsToMemDescForDotOp<WarpGroupDotOp>>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::triton::nvidia_gpu
