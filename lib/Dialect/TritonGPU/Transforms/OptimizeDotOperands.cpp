#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include <algorithm>
#include <cassert>

namespace mlir::triton::gpu {

namespace {
// Given
//   dot(convert(trans(src)) #dot_operand) ->
//   dot(convert(local_load(trans(alloc(src)))))
// change the encoding of the inner convert to a special, swizzled shared
// encoding.
class SwizzleShmemConvert : public OpRewritePattern<ConvertLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp cvtOp,
                                PatternRewriter &rewriter) const override {
    if (!cvtOp->hasOneUse() ||
        !isa<triton::DotOp>(cvtOp->use_begin()->getOwner()))
      return failure();
    // Match outerCvt(trans(innerCvt(x))).
    auto trans = cvtOp.getSrc().getDefiningOp<TransOp>();
    if (!trans || trans.getOrder() != ArrayRef<int32_t>{1, 0})
      return failure();

    RankedTensorType srcTy = trans.getSrc().getType();

    if (auto srcCvt = trans.getSrc().getDefiningOp<ConvertLayoutOp>()) {
      srcTy = srcCvt.getSrc().getType();
    }
    RankedTensorType sharedLoadTy = cvtOp.getType();
    auto cvtEncoding =
        dyn_cast<DotOperandEncodingAttr>(sharedLoadTy.getEncoding());
    if (!cvtEncoding)
      return failure();

    // Set needTrans to true here. newInnerCvtEnc is computed based on
    // argEncoding which is before the transpose. Without needTrans we will
    // compute vec and maxPhase based on incorrect m, n and k size of mma. The
    // type inference of MemDescTransOp simply swap the order but doesn't fix
    // the vec and maxPhase for the YType, hence it would causing incorrect
    // swizzling code.
    auto ctx = getContext();
    auto oldCGALayout = triton::gpu::getCGALayout(srcTy.getEncoding());
    auto newLl =
        transposeLinearLayout(oldCGALayout.getLinearLayout(), trans.getOrder());
    auto newCGALayout = CGAEncodingAttr::get(ctx, std::move(newLl));
    auto newInnerCvtEnc =
        SwizzledSharedEncodingAttr::get(ctx, cvtEncoding, srcTy.getShape(),
                                        /*order=*/getOrderForMemory(srcTy),
                                        newCGALayout, srcTy.getElementType(),
                                        /*needTrans=*/true);
    if (newInnerCvtEnc == cvtEncoding)
      return failure();
    rewriter.setInsertionPoint(trans);
    auto sharedMemorySpace = SharedMemorySpaceAttr::get(getContext());
    auto alloc = LocalAllocOp::create(
        rewriter, trans.getLoc(),
        MemDescType::get(srcTy.getShape(), srcTy.getElementType(),
                         newInnerCvtEnc, sharedMemorySpace),
        trans.getSrc());
    auto newTrans = MemDescTransOp::create(rewriter, trans.getLoc(), alloc,
                                           ArrayRef<int32_t>({1, 0}));
    auto localLoadOp =
        LocalLoadOp::create(rewriter, trans.getLoc(), sharedLoadTy, newTrans);
    rewriter.modifyOpInPlace(cvtOp, [&]() {
      cvtOp.getSrcMutable().assign(localLoadOp.getResult());
    });
    return success();
  }
};

// Rewrite
//
//   dot(alloc(trans() #shared1) ->
//   dot(trans(alloc() #shared2))
//
// if dot is an MMAv3/v5 (because MMAv3/v5 allows us to fold transposes).
class FuseTransMMAV3Plus : public OpRewritePattern<LocalAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LocalAllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    if (!allocOp.getSrc() || !allocOp->hasOneUse() ||
        !isa<triton::nvidia_gpu::WarpGroupDotOp,
             triton::nvidia_gpu::MMAv5OpInterface>(
            *allocOp->getUsers().begin()))
      return failure();

    // Match outerCvt(trans(innerCvt(x))).
    auto trans = allocOp.getSrc().getDefiningOp<TransOp>();
    if (!trans || trans.getOrder() != ArrayRef<int32_t>({1, 0}))
      return failure();

    MemDescType allocType = allocOp.getType();
    auto allocEncoding =
        dyn_cast<NVMMASharedEncodingAttr>(allocType.getEncoding());
    if (!allocEncoding)
      return failure();
    RankedTensorType srcTy = trans.getSrc().getType();

    Dialect &dialect = allocEncoding.getDialect();
    auto inferLayoutInterface = cast<DialectInferLayoutInterface>(&dialect);
    Attribute newInnerEnc;
    if (failed(inferLayoutInterface->inferTransOpEncoding(
            allocEncoding, srcTy.getShape(), trans.getOrder(), newInnerEnc,
            allocOp.getLoc()))) {
      return failure();
    }

    MemDescType innerTy =
        MemDescType::get(srcTy.getShape(), srcTy.getElementType(), newInnerEnc,
                         allocType.getMemorySpace());
    auto newAlloc = LocalAllocOp::create(rewriter, allocOp.getLoc(), innerTy,
                                         trans.getSrc());
    rewriter.replaceOpWithNewOp<MemDescTransOp>(allocOp, newAlloc,
                                                ArrayRef<int32_t>({1, 0}));
    return success();
  }
};

// Rewrite
//
//   alloc(reshape(), #shared1) ->
//   memdesc_reshape(alloc() #shared2))
//
// if dot is an MMAv3/v5 (because MMAv3/v5 allows us to fold transposes).
class ReshapeMemDesc : public OpRewritePattern<LocalAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LocalAllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    if (!allocOp.getSrc())
      return failure();

    auto reshapeOp = allocOp.getSrc().getDefiningOp<ReshapeOp>();
    if (!reshapeOp)
      return failure();

    MemDescType allocType = allocOp.getType();

    RankedTensorType srcTy = reshapeOp.getSrc().getType();
    auto srcShape = srcTy.getShape();

    // We use the fact that forward and backward inference are the same for
    // MemDescReshapeOp to infer the source MemDescType that would produce
    // `allocType` after a reshape.
    MemDescType innerTy;
    if (failed(MemDescReshapeOp::inferReturnTypes(
            getContext(), allocOp.getLoc(), allocType, srcShape, innerTy)))
      return failure();

    // For now don't apply the transformation if the new encoding is not an
    // MMAv3/v5 encoding as it may not be compatible with the user.
    // The heuristic can be refined once we have more flexible mma ops.
    if (!isa<NVMMASharedEncodingAttr>(innerTy.getEncoding()))
      return failure();

    auto newAlloc = LocalAllocOp::create(rewriter, allocOp.getLoc(), innerTy,
                                         reshapeOp.getSrc());
    rewriter.replaceOpWithNewOp<MemDescReshapeOp>(allocOp, allocOp.getType(),
                                                  newAlloc);
    return success();
  }
};

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
    if (!isa<triton::nvidia_gpu::TCGen5MMAOp,
             triton::nvidia_gpu::TCGen5MMAScaledOp,
             triton::nvidia_gpu::WarpGroupDotOp>(dotOp))
      return failure();

    bool changed = false;

    if (rewriteOperand(dotOp.getA(), rewriter).succeeded())
      changed = true;

    if (rewriteOperand(dotOp.getB(), rewriter).succeeded())
      changed = true;

    if (auto mmaScaledOp = dyn_cast<triton::nvidia_gpu::TCGen5MMAScaledOp>(
            dotOp.getOperation())) {
      if (rewriteOperand(mmaScaledOp.getAScale(), rewriter).succeeded())
        changed = true;
      if (rewriteOperand(mmaScaledOp.getBScale(), rewriter).succeeded())
        changed = true;
    }

    return success(changed);
  }

private:
  LogicalResult rewriteOperand(Value operand, PatternRewriter &rewriter) const {
    auto operandTy = dyn_cast<MemDescType>(operand.getType());
    if (!operandTy)
      return failure();

    // Restrict this rewrite to an operand which already uses a shared-linear
    // encoding. Backward propagation through tensor reshape/trans is not
    // encoding-stable for NVMMAShared.
    if (!isa<SharedLinearEncodingAttr>(operandTy.getEncoding()))
      return failure();

    Value beforeTrailing = operand;
    while (auto view = beforeTrailing.getDefiningOp()) {
      if (auto reshape = dyn_cast<MemDescReshapeOp>(view)) {
        beforeTrailing = reshape.getSrc();
        continue;
      }
      if (auto trans = dyn_cast<MemDescTransOp>(view)) {
        beforeTrailing = trans.getSrc();
        continue;
      }
      break;
    }

    auto localAlloc = beforeTrailing.getDefiningOp<LocalAllocOp>();
    if (!localAlloc || !localAlloc.getSrc())
      return failure();

    Value baseTensor = localAlloc.getSrc();
    SmallVector<Operation *> tensorReplaySteps;
    MemDescType baseMemTy = localAlloc.getType();
    while (auto view = baseTensor.getDefiningOp()) {
      if (auto reshape = dyn_cast<triton::ReshapeOp>(view)) {
        MemDescType srcTy;
        auto inferred = MemDescReshapeOp::inferReturnTypes(
            getContext(), reshape.getLoc(), baseMemTy,
            reshape.getSrc().getType().getShape(), srcTy);
        assert(succeeded(inferred) && "backward memdesc reshape inference "
                                      "must succeed");
        (void)inferred;
        baseMemTy = srcTy;
      } else if (auto trans = dyn_cast<triton::TransOp>(view)) {
        Attribute srcEnc = inferSrcEncoding(view, baseMemTy.getEncoding());
        if (!srcEnc)
          return failure();
        baseMemTy = MemDescType::get(
            trans.getSrc().getType().getShape(), baseMemTy.getElementType(),
            srcEnc, baseMemTy.getMemorySpace(), baseMemTy.getMutableMemory());
      } else {
        break;
      }
      tensorReplaySteps.push_back(view);
      baseTensor = view->getOperand(0);
    }
    if (tensorReplaySteps.empty())
      return failure();

    std::reverse(tensorReplaySteps.begin(), tensorReplaySteps.end());

    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(localAlloc);

    Value rewritten = LocalAllocOp::create(rewriter, localAlloc.getLoc(),
                                           baseMemTy, baseTensor);
    for (Operation *op : tensorReplaySteps) {
      if (auto reshape = dyn_cast<triton::ReshapeOp>(op)) {
        rewritten = MemDescReshapeOp::create(rewriter, op->getLoc(), rewritten,
                                             reshape.getType().getShape());
      } else {
        auto trans = cast<triton::TransOp>(op);
        rewritten = MemDescTransOp::create(rewriter, op->getLoc(), rewritten,
                                           trans.getOrder());
      }
    }
    rewriter.replaceOp(localAlloc, rewritten);
    return success();
  }
};

} // namespace

#define GEN_PASS_DEF_TRITONGPUOPTIMIZEDOTOPERANDS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUOptimizeDotOperandsPass
    : public impl::TritonGPUOptimizeDotOperandsBase<
          TritonGPUOptimizeDotOperandsPass> {
public:
  using impl::TritonGPUOptimizeDotOperandsBase<
      TritonGPUOptimizeDotOperandsPass>::TritonGPUOptimizeDotOperandsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    OpPassManager pm;
    pm.addPass(mlir::createCanonicalizerPass());
    if (failed(runPipeline(pm, m)))
      return signalPassFailure();

    mlir::RewritePatternSet patterns(context);
    patterns.add<SwizzleShmemConvert>(context);
    patterns.add<FuseTransMMAV3Plus, ReshapeMemDesc>(context);
    patterns.add<RewriteMmaOperandViewsToMemDescForDotOp>(context);
    ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::triton::gpu
