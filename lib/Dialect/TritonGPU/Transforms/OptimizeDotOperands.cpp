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

// Rewrite
//   desc_load<swizzle=0> -> tt.reshape / tt.trans -> local_alloc -> memdesc
//   reshape / trans
// into
//   desc_load<swizzle=0> -> local_alloc<swizzle=0> -> memdesc reshape / trans
//
// swizzle=0 in NVMMASharedEncodingAttr represents a flat, contiguous layout.
// This is valid as the destination encoding for TMA, but unless the operand's
// contiguous dimension is <= 16 bytes, it is not the correct layout for an MMA
// operand which requires the special "core-matrices" layout even with
// swizzle=0. So if the result of swizzle-0 TMA is fed into MMA without smem
// layout conversion between them, the result would be incorrect.
//
// When using swizzle-0 TMA with MMA, it is a user's responsibility to have the
// source of TMA in global memory to be already in the core-matrices format, and
// insert a sequence of tt.reshape / tt.trans transformations between desc_load
// and MMA ops such that the MMA op sees the right core-matrices layout.

// This rewrite pattern ensures that swizzle=0 in TMA and a sequence of
// tt.reshape / tt.trans ops are correctly propagated, via equivalent
// transformations on memdesc, into the right MMA SMEM operand layout with
// swizzle=0.
template <typename DotOpTy>
class RewriteSwizzle0OperandViewsToMemDescForDotOp
    : public OpRewritePattern<DotOpTy> {
public:
  using OpRewritePattern<DotOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(DotOpTy dotOp,
                                PatternRewriter &rewriter) const override {
    Value oldA = dotOp.getA();
    Value oldB = dotOp.getB();
    bool changedA = rewriteOperand(dotOp.getAMutable(), rewriter).succeeded();
    bool changedB = rewriteOperand(dotOp.getBMutable(), rewriter).succeeded();

    if (changedA || changedB) {
      updateDependentOps(dotOp, oldA, oldB, rewriter);
      return success();
    }

    return failure();
  }

private:
  template <typename T>
  static void updateDependentOps(T, Value, Value, PatternRewriter &) {}

  static void updateDependentOps(triton::nvidia_gpu::WarpGroupDotOp dotOp,
                                 Value oldA, Value oldB,
                                 PatternRewriter &rewriter) {
    // Keep warp_group_dot_wait operands consistent with rewritten dot
    // operands. The wait op is variadic and can carry [dot_result, A, B], so
    // after changing dotOp's A/B we need to retarget corresponding wait
    // operands.
    Value newA = dotOp.getA();
    Value newB = dotOp.getB();
    for (Operation *user : dotOp.getResult().getUsers()) {
      auto waitOp = dyn_cast<triton::nvidia_gpu::WarpGroupDotWaitOp>(user);
      if (!waitOp)
        continue;
      rewriter.modifyOpInPlace(waitOp, [&]() {
        for (OpOperand &operand : waitOp->getOpOperands()) {
          Value replacement;
          if (operand.get() == oldA)
            replacement = newA;
          else if (operand.get() == oldB)
            replacement = newB;
          else
            continue;

          operand.assign(replacement);
          if (operand.getOperandNumber() < waitOp->getNumResults())
            waitOp->getResult(operand.getOperandNumber())
                .setType(replacement.getType());
        }
      });
    }
  }

  struct ViewStep {
    enum Kind { Reshape, Transpose } kind;
    SmallVector<int64_t> shape;
    SmallVector<int32_t> order;
    Location loc;
  };

  template <typename ReshapeOpTy, typename TransOpTy>
  static std::tuple<Value, SmallVector<ViewStep>>
  collectViewSteps(Value value) {
    Value current = value;
    SmallVector<ViewStep> replaySteps;
    while (true) {
      if (auto reshape = current.template getDefiningOp<ReshapeOpTy>()) {
        auto ty = reshape.getType();
        SmallVector<int64_t> shape(ty.getShape().begin(), ty.getShape().end());
        replaySteps.push_back(ViewStep{
            ViewStep::Reshape, std::move(shape), {}, reshape.getLoc()});
        current = reshape.getSrc();
        continue;
      }
      if (auto trans = current.template getDefiningOp<TransOpTy>()) {
        SmallVector<int32_t> order(trans.getOrder().begin(),
                                   trans.getOrder().end());
        replaySteps.push_back(ViewStep{
            ViewStep::Transpose, {}, std::move(order), trans.getLoc()});
        current = trans.getSrc();
        continue;
      }
      break;
    }
    return {current, llvm::to_vector(llvm::reverse(replaySteps))};
  }

  static SharedEncodingTrait getSourceSwizzle0SharedEncoding(Value baseTensor) {
    if (auto descLoad = baseTensor.getDefiningOp<DescriptorLoadOp>()) {
      auto descTy = cast<TensorDescType>(descLoad.getDesc().getType());
      auto descBlockTy = descTy.getBlockType();
      auto sourceSharedEnc =
          dyn_cast_or_null<SharedEncodingTrait>(descBlockTy.getEncoding());
      if (auto nvmma = dyn_cast_or_null<NVMMASharedEncodingAttr>(
              cast_or_null<Attribute>(sourceSharedEnc));
          nvmma && nvmma.getSwizzlingByteWidth() == 0)
        return sourceSharedEnc;
    }
    return nullptr;
  }

  // Build a new memdesc type for the rewritten `local_alloc` by taking the
  // original MMA operand memdesc and replacing its shape and shared encoding
  // with those from swizzle-0 `tt.descriptor_load` result
  static FailureOr<MemDescType> getSwizzle0MemDescType(MemDescType refTy,
                                                       Value baseTensor) {
    auto sourceSharedEnc = getSourceSwizzle0SharedEncoding(baseTensor);
    if (!sourceSharedEnc)
      return failure();

    auto baseTensorTy = cast<RankedTensorType>(baseTensor.getType());
    return MemDescType::get(baseTensorTy.getShape(),
                            baseTensorTy.getElementType(),
                            cast<Attribute>(sourceSharedEnc),
                            refTy.getMemorySpace(), refTy.getMutableMemory());
  }

  LogicalResult rewriteOperand(OpOperand &operand,
                               PatternRewriter &rewriter) const {
    Value orig = operand.get();
    auto origTy = dyn_cast<MemDescType>(orig.getType());
    if (!origTy)
      return failure();

    auto [beforeTrailing, trailingMemDescReplaySteps] =
        collectViewSteps<MemDescReshapeOp, MemDescTransOp>(orig);

    auto localAlloc = beforeTrailing.template getDefiningOp<LocalAllocOp>();
    if (!localAlloc || !localAlloc.getSrc())
      return failure();

    auto [baseTensor, tensorReplaySteps] =
        collectViewSteps<triton::ReshapeOp, triton::TransOp>(
            localAlloc.getSrc());

    if (tensorReplaySteps.empty())
      return failure();

    FailureOr<MemDescType> baseMemTy =
        getSwizzle0MemDescType(origTy, baseTensor);
    if (failed(baseMemTy))
      return failure();

    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(localAlloc);

    Value rewritten = LocalAllocOp::create(rewriter, localAlloc.getLoc(),
                                           *baseMemTy, baseTensor);

    for (ViewStep &step : tensorReplaySteps) {
      if (step.kind == ViewStep::Reshape) {
        rewritten =
            MemDescReshapeOp::create(rewriter, step.loc, rewritten, step.shape);
      } else {
        rewritten =
            MemDescTransOp::create(rewriter, step.loc, rewritten, step.order);
      }
    }

    for (ViewStep &step : trailingMemDescReplaySteps) {
      if (step.kind == ViewStep::Reshape) {
        rewritten =
            MemDescReshapeOp::create(rewriter, step.loc, rewritten, step.shape);
      } else {
        rewritten =
            MemDescTransOp::create(rewriter, step.loc, rewritten, step.order);
      }
    }

    auto rewrittenTy = cast<MemDescType>(rewritten.getType());
    assert(rewrittenTy.getShape() == origTy.getShape() &&
           rewrittenTy.getElementType() == origTy.getElementType() &&
           "rewrite must preserve memdesc shape and element type");

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
    patterns.add<UseShmemForScales,
                 RewriteSwizzle0OperandViewsToMemDescForDotOp<
                     triton::nvidia_gpu::TCGen5MMAOp>,
                 RewriteSwizzle0OperandViewsToMemDescForDotOp<
                     triton::nvidia_gpu::TCGen5MMAScaledOp>,
                 RewriteSwizzle0OperandViewsToMemDescForDotOp<
                     triton::nvidia_gpu::WarpGroupDotOp>>(context);
    ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::triton::gpu
