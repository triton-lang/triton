#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUOPTIMIZETMEMLAYOUTSPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// clang-format off
// Converts:
//  %l = ttng.tmem_load %o : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>
//  %r = tt.reshape %l : tensor<128x256xf32, #blocked> -> tensor<128x2x128xf32, #blocked4>
//  %t = tt.trans %r {order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #blocked4> -> tensor<128x128x2xf32, #blocked5>
//  %outLHS, %outRHS = tt.split %t : tensor<128x128x2xf32, #blocked5> -> tensor<128x128xf32, #blocked2>
// To:
//  %o0 = ttng.tmem_subslice %o { N = 0 }: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
//  %outLHS = ttng.tmem_load %o0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
//  %o1 = ttng.tmem_subslice %o { N = 128 }: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
//  %outRHS = ttng.tmem_load %o1 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
// clang-format on
// This will change the layout of the destination tensor to distribute each
// slice across warps. It currently only supports simple cases where tmem can be
// sliced easily. This could be extended if needed with more powerful slicing
// support of tmem.
class TMemSplitLoadPattern : public OpRewritePattern<SplitOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SplitOp splitOp,
                                PatternRewriter &rewriter) const override {
    auto src = splitOp.getSrc();
    // Skip convert layout ops.
    while (auto cvt = src.getDefiningOp<ttg::ConvertLayoutOp>()) {
      src = cvt.getSrc();
    }
    // Only support splitting N dimension on the outer most.
    auto transOp = src.getDefiningOp<TransOp>();
    if (!transOp || transOp.getOrder() != ArrayRef<int>({0, 2, 1}))
      return failure();
    auto reshapeOp = transOp.getSrc().getDefiningOp<ReshapeOp>();
    if (!reshapeOp)
      return failure();
    auto shape = reshapeOp.getResult().getType().getShape();
    if (shape[0] != reshapeOp.getSrc().getType().getShape()[0])
      return failure();
    auto tmemLoad = reshapeOp.getSrc().getDefiningOp<TMEMLoadOp>();
    if (!tmemLoad)
      return failure();
    // We found a tmem_load that is split on the N dimension. We can split it
    // into multiple tmem_loads.
    int mDim = getShapePerCTA(tmemLoad.getSrc().getType())[0];
    // TODO: enable other M cases. (the layout is a bit more complex).
    if (mDim != 128)
      return failure();
    int splitNSize = shape[2];
    if (splitNSize < 8)
      return failure();
    Value tmem = tmemLoad.getSrc();
    int numWarps = ttg::lookupNumWarps(tmemLoad);
    rewriter.setInsertionPoint(tmemLoad);
    // First slice.
    Value subSlice0 =
        rewriter.create<TMEMSubSliceOp>(tmemLoad.getLoc(), tmem, 0, splitNSize);
    Attribute distLayout = getTmemCompatibleLayout(
        mDim, splitNSize, splitOp.getOutLHS().getType(), numWarps);
    RankedTensorType newLoadType = RankedTensorType::get(
        splitOp.getOutLHS().getType().getShape(),
        splitOp.getOutLHS().getType().getElementType(), distLayout);
    auto load0 =
        rewriter.create<TMEMLoadOp>(tmemLoad.getLoc(), newLoadType, subSlice0);
    auto cvt0 = rewriter.create<ttg::ConvertLayoutOp>(
        tmemLoad.getLoc(), splitOp.getOutLHS().getType(), load0);
    // Second slice.
    Value subSlice1 = rewriter.create<TMEMSubSliceOp>(tmemLoad.getLoc(), tmem,
                                                      splitNSize, splitNSize);
    auto load1 =
        rewriter.create<TMEMLoadOp>(tmemLoad.getLoc(), newLoadType, subSlice1);
    auto cvt1 = rewriter.create<ttg::ConvertLayoutOp>(
        tmemLoad.getLoc(), splitOp.getOutRHS().getType(), load1);
    rewriter.replaceOp(splitOp, {cvt0, cvt1});
    return success();
  }
};

class TMemStoreJoinPattern : public OpRewritePattern<TMEMStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TMEMStoreOp storeOp,
                                PatternRewriter &b) const override {
    // Look through layout conversions.
    Value src = storeOp.getSrc();
    while (auto cvt = src.getDefiningOp<ttg::ConvertLayoutOp>()) {
      src = cvt.getSrc();
    }

    // Only support joinin N dimension on the outer most.
    auto reshapeOp = src.getDefiningOp<ReshapeOp>();
    if (!reshapeOp)
      return failure();
    auto shape = reshapeOp.getSrc().getType().getShape();
    if (reshapeOp.getType().getShape().front() != shape[0])
      return failure();
    auto transOp = reshapeOp.getSrc().getDefiningOp<TransOp>();
    if (!transOp || transOp.getOrder() != ArrayRef<int>({0, 2, 1}))
      return failure();
    auto joinOp = transOp.getSrc().getDefiningOp<JoinOp>();
    if (!joinOp)
      return failure();

    // We found a tmem_store that is joined on the N dimension. We can split it
    // into multiple tmem_stores.
    int mDim = getShapePerCTA(storeOp.getDst().getType())[0];
    // TODO: enable other M cases. (the layout is a bit more complex).
    if (mDim != 128)
      return failure();
    int splitNSize = shape[2];
    if (splitNSize < 8)
      return failure();

    Location loc = storeOp.getLoc();
    Value tmem = storeOp.getDst();
    int numWarps = ttg::lookupNumWarps(storeOp);
    Value truePred = b.create<arith::ConstantOp>(loc, b.getBoolAttr(true));

    Attribute distLayout = getTmemCompatibleLayout(
        mDim, splitNSize, joinOp.getLhs().getType(), numWarps);
    auto newStoreType = RankedTensorType::get(
        joinOp.getLhs().getType().getShape(),
        joinOp.getLhs().getType().getElementType(), distLayout);

    // First slice.
    auto subSlice0 = b.create<TMEMSubSliceOp>(loc, tmem, 0, splitNSize);
    auto cvt0 =
        b.create<ttg::ConvertLayoutOp>(loc, newStoreType, joinOp.getLhs());
    auto store0 =
        b.create<TMEMStoreOp>(loc, subSlice0, cvt0.getResult(), truePred);
    // Second slice.
    auto subSlice1 =
        b.create<TMEMSubSliceOp>(loc, tmem, splitNSize, splitNSize);
    auto cvt1 =
        b.create<ttg::ConvertLayoutOp>(loc, newStoreType, joinOp.getRhs());
    auto store1 =
        b.create<TMEMStoreOp>(loc, subSlice1, cvt1.getResult(), truePred);
    b.eraseOp(storeOp);
    return success();
  }
};

// Pick an optimized tmem load layout based on its users. When there are
// multiple warpgroups tmem_load results can be distirbuted along M or N across
// the warpgroups. By default distribute along N but when there is a reduction
// along N dimension we want to distribute along M instead to avoid having to
// reduce across warps.
class TMemLoadReducePattern : public OpRewritePattern<TMEMLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TMEMLoadOp tmemLoadOp,
                                PatternRewriter &rewriter) const override {
    int numWarps = ttg::lookupNumWarps(tmemLoadOp);
    // If there is only 1 warpgroup there is nothing to optimize as the layout
    // is already reduction friendly.
    if (numWarps != 8)
      return failure();
    auto tmemEnc = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
        tmemLoadOp.getSrc().getType().getEncoding());
    if (!tmemEnc)
      return failure();
    int M = tmemEnc.getBlockM();
    int N = tmemEnc.getBlockN();
    if (M != 128)
      return failure();
    bool foundReductionAlongN = false;
    auto filter = [&](Operation *op) {
      if (isa<ttg::ConvertLayoutOp>(op) || op->hasTrait<OpTrait::Elementwise>())
        return true;
      if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
        foundReductionAlongN = reduce.getAxis() == 1;
      }
      return false;
    };
    ForwardSliceOptions fwdOpt;
    fwdOpt.filter = filter;
    SetVector<mlir::Operation *> fwdSlices;
    getForwardSlice(tmemLoadOp.getResult(), &fwdSlices, fwdOpt);
    if (!foundReductionAlongN)
      return failure();
    // Try to split along M dimension but follow the restrictions of TMEM:
    // warp0 get M = 0, warp 1 gets M = 32, warp 2 gets M = 64, warp 3 gets
    // M = 96 warp 4 gets M = 16, warp 5 gets M = 48, warp 6 gets M = 80,
    // warp 7 gets M = 112
    RankedTensorType oldType = tmemLoadOp.getType();
    Attribute newLayout = ttg::LinearEncodingAttr::get(
        tmemLoadOp.getContext(),
        ttg::getTmemLoadLayoutSplitLongM(M, N, oldType, numWarps));
    if (newLayout == oldType.getEncoding())
      return failure();

    auto newType = RankedTensorType::get(oldType.getShape(),
                                         oldType.getElementType(), newLayout);
    tmemLoadOp.getResult().setType(newType);
    OpBuilder builder(tmemLoadOp);
    builder.setInsertionPointAfter(tmemLoadOp);
    auto cvt = builder.create<ttg::ConvertLayoutOp>(
        tmemLoadOp.getLoc(), oldType, tmemLoadOp.getResult());
    tmemLoadOp.getResult().replaceAllUsesExcept(cvt.getResult(), cvt);
    return success();
  }
};

// Optimize local_load -> tmem_store when the layout 16x256b allows better
// code generation for local_load lowering.
class TMemFromSharedMemPattern : public OpRewritePattern<TMEMStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TMEMStoreOp tmemStoreOp,
                                PatternRewriter &rewriter) const override {
    auto tmemEnc = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
        tmemStoreOp.getDst().getType().getEncoding());
    if (!tmemEnc)
      return failure();
    int M = tmemEnc.getBlockM();
    int N = tmemEnc.getBlockN();
    int numWarps = ttg::lookupNumWarps(tmemStoreOp);
    // Compute the alternative layout.
    std::optional<LinearLayout> ll = gpu::getTmemLoadStoreLayout16x256(
        M, N, tmemStoreOp.getSrc().getType(), numWarps);
    if (!ll)
      return failure();
    Attribute newEncoding =
        gpu::LinearEncodingAttr::get(tmemStoreOp.getContext(), *ll);
    auto newType = RankedTensorType::get(
        tmemStoreOp.getSrc().getType().getShape(),
        tmemStoreOp.getSrc().getType().getElementType(), newEncoding);
    if (newType == tmemStoreOp.getSrc().getType())
      return failure();

    SetVector<Value> slice;
    DenseMap<Value, Attribute> layoutMap;
    // Check how it may propagate up the SSA chain.
    LogicalResult result = getConvertBackwardSlice(
        tmemStoreOp.getSrcMutable(), slice, newEncoding, layoutMap);
    if (result.failed())
      return failure();
    bool foundImprovedLoad = false;
    for (Value v : slice) {
      auto localLoad = v.getDefiningOp<gpu::LocalLoadOp>();
      if (!localLoad)
        continue;
      // 16x256b is optimized for 16bits load.
      if (localLoad.getType().getElementType().getIntOrFloatBitWidth() != 16)
        return failure();
      LinearLayout regLayout = gpu::toLinearLayout(
          localLoad.getType().getShape(), localLoad.getType().getEncoding());
      LinearLayout smemLayout =
          gpu::toLinearLayout(localLoad.getSrc().getType().getShape(),
                              localLoad.getSrc().getType().getEncoding());
      int vecDim =
          regLayout.invertAndCompose(smemLayout).getNumConsecutiveInOut();
      // If we find a 16bits load that cannot be vectorized use the alternative
      // layout.
      if (vecDim != 1)
        return failure();
      foundImprovedLoad = true;
    }
    if (!foundImprovedLoad)
      return failure();
    // Use the new layout and rely on RemoveLayoutConversions pass to propagate
    // the convert_layout.
    auto cvt = rewriter.create<ttg::ConvertLayoutOp>(
        tmemStoreOp.getLoc(), newType, tmemStoreOp.getSrc());
    rewriter.modifyOpInPlace(tmemStoreOp, [&]() {
      tmemStoreOp.getSrcMutable().assign(cvt.getResult());
    });
    return success();
  }
};

} // anonymous namespace

class TritonNvidiaGPUOptimizeTMemLayoutsPass
    : public impl::TritonNvidiaGPUOptimizeTMemLayoutsPassBase<
          TritonNvidiaGPUOptimizeTMemLayoutsPass> {
public:
  using BaseT = TritonNvidiaGPUOptimizeTMemLayoutsPassBase<
      TritonNvidiaGPUOptimizeTMemLayoutsPass>;
  using BaseT::BaseT;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<TMemSplitLoadPattern, TMemStoreJoinPattern,
                 TMemLoadReducePattern, TMemFromSharedMemPattern>(context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
