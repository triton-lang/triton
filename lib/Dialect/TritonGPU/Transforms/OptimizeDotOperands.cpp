#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include <memory>

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

    auto dot = *allocOp->getUsers().begin();
    // Match outerCvt(trans(innerCvt(x))).
    auto trans = allocOp.getSrc().getDefiningOp<TransOp>();
    if (!trans || trans.getOrder() != ArrayRef<int32_t>({1, 0}))
      return failure();

    MemDescType allocType = allocOp.getType();
    auto allocEncoding = cast<NVMMASharedEncodingAttr>(allocType.getEncoding());
    RankedTensorType srcTy = trans.getSrc().getType();

    auto ctx = getContext();
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
    auto allocEncoding = allocType.getEncoding();

    RankedTensorType srcTy = reshapeOp.getSrc().getType();
    auto srcShape = srcTy.getShape();
    auto dstShape = allocType.getShape();

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

// Inject TMEM copy instructions into IR to efficiently load blocked scales for
// scaled dot
class UseShmemForScales
    : public OpRewritePattern<triton::nvidia_gpu::TCGen5MMAScaledOp> {
public:
  using OpRewritePattern<
      triton::nvidia_gpu::TCGen5MMAScaledOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::nvidia_gpu::TCGen5MMAScaledOp mmaOp,
                                PatternRewriter &rewriter) const override {
    auto aScale = mmaOp.getAScale();
    auto bScale = mmaOp.getBScale();
    LogicalResult ret = failure();
    if (aScale && isa<triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(
                      aScale.getType().getEncoding())) {
      if (rewriteOperand(mmaOp.getAScaleMutable(), rewriter).succeeded())
        ret = success();
    }
    if (bScale && isa<triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(
                      bScale.getType().getEncoding())) {
      if (rewriteOperand(mmaOp.getBScaleMutable(), rewriter).succeeded())
        ret = success();
    }
    return ret;
  }

private:
  LogicalResult rewriteOperand(OpOperand &opOperand,
                               PatternRewriter &rewriter) const {
    auto src = cast<TypedValue<MemDescType>>(opOperand.get());
    auto tmemAlloc = src.getDefiningOp<triton::nvidia_gpu::TMEMAllocOp>();
    if (!tmemAlloc) {
      return failure();
    }
    auto dstType = tmemAlloc.getResult().getType();

    if (!tmemAlloc.getSrc()) {
      return failure();
    }

    // Look for a sequence
    //    local_load
    // -> reshape(..., (BLOCK_MN / 128, BLOCK_K / scale_vec_size / 4, 32, 4,
    // 4)
    // -> transpose(..., (0, 3, 2, 1, 4))
    // -> reshape(..., (BLOCK_MN, BLOCK_K / scale_vec_size)
    // -> tmem_alloc
    // -> tc_gen_mma_scaled
    // and replace it with local_alloc -> tc_gen_mma_scaled
    auto scale2DShape = dstType.getShape();
    auto blockMN = scale2DShape[0];
    auto numScales = scale2DShape[1];
    const SmallVector<int> transposeOrder{0, 3, 2, 1, 4};
    const SmallVector<int64_t> reshape5DShape{blockMN / 128, numScales / 4, 32,
                                              4, 4};

    auto reshapeOp2D = getNextOp<triton::ReshapeOp>(tmemAlloc.getSrc());
    if (!reshapeOp2D ||
        reshapeOp2D.getResult().getType().getShape() != scale2DShape) {
      return failure();
    }

    auto transOp = getNextOp<triton::TransOp>(reshapeOp2D.getSrc());
    if (!transOp || transOp.getOrder() != ArrayRef<int>(transposeOrder)) {
      return failure();
    }

    auto reshapeOp5D = getNextOp<triton::ReshapeOp>(transOp.getSrc());
    if (!reshapeOp5D || reshapeOp5D.getResult().getType().getShape() !=
                            ArrayRef<int64_t>(reshape5DShape)) {
      return failure();
    }

    auto localLoad = getNextOp<triton::gpu::LocalLoadOp>(reshapeOp5D.getSrc());
    if (!localLoad) {
      return failure();
    }
    auto localAlloc = getNextOp<LocalAllocOp>(localLoad.getSrc());
    bool usesTMAload =
        (localAlloc && localAlloc.getSrc() &&
         (getNextOp<DescriptorLoadOp>(localAlloc.getSrc()) != nullptr));
    if (!isTmemCopyCompatible(localLoad.getSrc().getType(), usesTMAload))
      return failure();

    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(tmemAlloc);

    Value shared = localLoad.getSrc();

    Value reshaped5D = MemDescReshapeOp::create(rewriter, reshapeOp5D.getLoc(),
                                                shared, reshape5DShape);
    SmallVector<int32_t> transposeOrder32(transposeOrder.begin(),
                                          transposeOrder.end());
    Value transposed = MemDescTransOp::create(rewriter, transOp.getLoc(),
                                              reshaped5D, transposeOrder32);
    SmallVector<int64_t> scale2DShapeVec(scale2DShape.begin(),
                                         scale2DShape.end());
    Value reshaped2D = MemDescReshapeOp::create(rewriter, reshapeOp2D.getLoc(),
                                                transposed, scale2DShapeVec);

    opOperand.assign(reshaped2D);
    rewriter.eraseOp(tmemAlloc);
    return success();
  }

  template <typename Op> Op getNextOp(Value op) const {
    while (auto cvtOp = op.getDefiningOp<ConvertLayoutOp>()) {
      op = cvtOp.getSrc();
    }
    return op.getDefiningOp<Op>();
  }

  bool isTmemCopyCompatible(triton::gpu::MemDescType scaleType,
                            bool usesTMAload) const {
    // TMEM copy expects that blocked scale "chunks" in SMEM are stored in
    // innermost axes contiguously.
    if (!isInnermostContiguous(scaleType, 512))
      return false;

    if (usesTMAload) {
      return true;
    }

    if (scaleType.getRank() != 2) {
      // TODO: Add support for higher rank when 5D coalesced load is fixed
      return false;
    }

    auto elemBits = scaleType.getElementType().getIntOrFloatBitWidth();

    // We assume that 32x128b chunks are flattened into the inner most axis.
    auto innerMostBits =
        scaleType.getDimSize(scaleType.getRank() - 1) * elemBits;
    return innerMostBits % (32 * 128) == 0;
  }
};

template <typename DotOpTy>
class RewriteSwizzle0OperandViewsToMemDescForDotOp
    : public OpRewritePattern<DotOpTy> {
public:
  using OpRewritePattern<DotOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(DotOpTy dotOp,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    if (succeeded(rewriteOperand(dotOp.getAMutable(), rewriter)))
      changed = true;
    if (succeeded(rewriteOperand(dotOp.getBMutable(), rewriter)))
      changed = true;
    return changed ? success() : failure();
  }

private:
  struct ViewStep {
    enum Kind { Reshape, Transpose } kind;
    SmallVector<int64_t> shape;
    SmallVector<int32_t> order;
    Location loc;
  };

  LogicalResult rewriteOperand(OpOperand &operand,
                               PatternRewriter &rewriter) const {
    Value orig = operand.get();
    auto origTy = dyn_cast<MemDescType>(orig.getType());
    if (!origTy)
      return failure();

    SmallVector<int32_t> trailingMemDescTransOrder;
    Value beforeTrailing = orig;
    if (auto trailing = beforeTrailing.getDefiningOp<MemDescTransOp>()) {
      trailingMemDescTransOrder.assign(trailing.getOrder().begin(),
                                       trailing.getOrder().end());
      beforeTrailing = trailing.getSrc();
    }

    auto localAlloc = beforeTrailing.getDefiningOp<LocalAllocOp>();
    if (!localAlloc || !localAlloc.getSrc())
      return failure();

    auto allocTy = cast<MemDescType>(localAlloc.getType());
    auto allocEnc = dyn_cast<NVMMASharedEncodingAttr>(allocTy.getEncoding());
    if (!allocEnc || allocEnc.getSwizzlingByteWidth() != 0)
      return failure();

    SmallVector<ViewStep> reverseSteps;
    Value baseTensor = localAlloc.getSrc();
    while (true) {
      if (auto cvt = baseTensor.getDefiningOp<ConvertLayoutOp>()) {
        baseTensor = cvt.getSrc();
        continue;
      }
      if (auto reshape = baseTensor.getDefiningOp<triton::ReshapeOp>()) {
        SmallVector<int64_t> shape(reshape.getType().getShape().begin(),
                                   reshape.getType().getShape().end());
        reverseSteps.push_back(ViewStep{
            ViewStep::Reshape, std::move(shape), {}, reshape.getLoc()});
        baseTensor = reshape.getSrc();
        continue;
      }
      if (auto trans = baseTensor.getDefiningOp<triton::TransOp>()) {
        SmallVector<int32_t> order(trans.getOrder().begin(),
                                   trans.getOrder().end());
        reverseSteps.push_back(ViewStep{
            ViewStep::Transpose, {}, std::move(order), trans.getLoc()});
        baseTensor = trans.getSrc();
        continue;
      }
      break;
    }

    if (reverseSteps.empty())
      return failure();

    auto baseTensorTy = dyn_cast<RankedTensorType>(baseTensor.getType());
    if (!baseTensorTy)
      return failure();

    auto cgaLayout = CGAEncodingAttr::get1CTALayout(rewriter.getContext(),
                                                    baseTensorTy.getRank());
    auto baseEnc = NVMMASharedEncodingAttr::get(
        rewriter.getContext(), /*swizzlingByteWidth=*/0,
        /*transposed=*/false, allocEnc.getElementBitWidth(),
        allocEnc.getFp4Padded(), cgaLayout);
    auto baseMemTy = MemDescType::get(
        baseTensorTy.getShape(), baseTensorTy.getElementType(), baseEnc,
        allocTy.getMemorySpace(), allocTy.getMutableMemory());

    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(localAlloc);

    Value rewritten = LocalAllocOp::create(rewriter, localAlloc.getLoc(),
                                           baseMemTy, baseTensor);

    for (ViewStep &step : llvm::reverse(reverseSteps)) {
      if (step.kind == ViewStep::Reshape) {
        MemDescType reshapedTy;
        if (failed(MemDescReshapeOp::inferReturnTypes(
                rewriter.getContext(), step.loc,
                cast<MemDescType>(rewritten.getType()), step.shape,
                reshapedTy)))
          return failure();
        rewritten =
            MemDescReshapeOp::create(rewriter, step.loc, reshapedTy, rewritten);
      } else {
        rewritten =
            MemDescTransOp::create(rewriter, step.loc, rewritten, step.order);
      }
    }

    if (!trailingMemDescTransOrder.empty()) {
      rewritten = MemDescTransOp::create(rewriter, localAlloc.getLoc(),
                                         rewritten, trailingMemDescTransOrder);
    }

    auto rewrittenTy = cast<MemDescType>(rewritten.getType());
    if (rewrittenTy.getShape() != origTy.getShape() ||
        rewrittenTy.getElementType() != origTy.getElementType())
      return failure();

    operand.assign(rewritten);
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
    patterns.add<
        UseShmemForScales,
        RewriteSwizzle0OperandViewsToMemDescForDotOp<triton::nvidia_gpu::TCGen5MMAOp>,
        RewriteSwizzle0OperandViewsToMemDescForDotOp<triton::nvidia_gpu::TCGen5MMAScaledOp>,
        RewriteSwizzle0OperandViewsToMemDescForDotOp<triton::nvidia_gpu::WarpGroupDotOp>>(
        context);
    ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::triton::gpu
