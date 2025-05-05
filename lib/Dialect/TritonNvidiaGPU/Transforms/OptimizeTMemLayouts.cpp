#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"

namespace {

using namespace mlir;

namespace ttng = triton::nvidia_gpu;
namespace ttg = triton::gpu;
namespace tt = triton;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

// If we don't know the effects of the op, we add all possible effects.
static void addAllValuelessEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Read>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Write>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Allocate>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Free>());
}

static bool
collectEffects(Operation *op,
               SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Collect effect instances the operation. Note that the implementation of
  // getEffects erases all effect instances that have the type other than the
  // template parameter so we collect them first in a local buffer and then
  // copy.
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> localEffects;
    iface.getEffects(localEffects);
    llvm::append_range(effects, localEffects);
    return true;
  }
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &innerOp : block)
          if (!collectEffects(&innerOp, effects))
            return false;
      }
    }
    return true;
  }

  // We need to be conservative here in case the op doesn't have the interface
  // and assume it can have any possible effect.
  addAllValuelessEffects(effects);
  return false;
}

// Sink tmem_loads as close to their use as possible to reduce register
// pressure.
static void sinkLoad(ttng::TMEMLoadOp load, Operation *cvt) {
  Operation *insertBefore = nullptr;
  Operation *next = cvt->getNextNode();
  while (next && !next->hasTrait<OpTrait::IsTerminator>()) {
    insertBefore = next;
    bool dep = false;
    for (auto operand : getNestedOperands(next)) {
      if (operand == cvt->getResult(0)) {
        dep = true;
        break;
      }
    }
    if (!isMemoryEffectFree(next)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      collectEffects(next, effects);
      for (auto effect : effects) {
        if (effect.getEffect() ==
                MemoryEffects::Effect::get<MemoryEffects::Write>() ||
            effect.getEffect() ==
                MemoryEffects::Effect::get<MemoryEffects::Allocate>()) {
          if (effect.getResource() ==
                  mlir::SideEffects::DefaultResource::get() ||
              effect.getResource() ==
                  mlir::triton::nvidia_gpu::TensorMemory::get()) {
            dep = true;
            break;
          }
        }
      }
    }
    if (dep)
      break;
    next = next->getNextNode();
  }
  if (insertBefore) {
    load->moveBefore(insertBefore);
    cvt->moveBefore(insertBefore);
  }
}

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
class TMemSplitLoadPattern : public OpRewritePattern<tt::SplitOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::SplitOp splitOp,
                                PatternRewriter &rewriter) const override {
    auto src = splitOp.getSrc();
    // Skip convert layout ops.
    while (auto cvt = src.getDefiningOp<ttg::ConvertLayoutOp>()) {
      src = cvt.getSrc();
    }
    // Only support splitting N dimension on the outer most.
    auto transOp = src.getDefiningOp<tt::TransOp>();
    if (!transOp || transOp.getOrder() != ArrayRef<int>({0, 2, 1}))
      return failure();
    auto reshapeOp = transOp.getSrc().getDefiningOp<tt::ReshapeOp>();
    if (!reshapeOp)
      return failure();
    auto shape = reshapeOp.getResult().getType().getShape();
    if (shape[0] != reshapeOp.getSrc().getType().getShape()[0])
      return failure();
    auto tmemLoad = reshapeOp.getSrc().getDefiningOp<ttng::TMEMLoadOp>();
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
    Value subSlice0 = rewriter.create<ttng::TMEMSubSliceOp>(
        tmemLoad.getLoc(), tmem, 0, splitNSize);
    Attribute distLayout = ttng::getTmemCompatibleLayout(
        mDim, splitNSize, splitOp.getOutLHS().getType(), numWarps);
    RankedTensorType newLoadType = RankedTensorType::get(
        splitOp.getOutLHS().getType().getShape(),
        splitOp.getOutLHS().getType().getElementType(), distLayout);
    auto load0 = rewriter.create<ttng::TMEMLoadOp>(tmemLoad.getLoc(),
                                                   newLoadType, subSlice0);
    auto cvt0 = rewriter.create<ttg::ConvertLayoutOp>(
        tmemLoad.getLoc(), splitOp.getOutLHS().getType(), load0);
    // Second slice.
    Value subSlice1 = rewriter.create<ttng::TMEMSubSliceOp>(
        tmemLoad.getLoc(), tmem, splitNSize, splitNSize);
    auto load1 = rewriter.create<ttng::TMEMLoadOp>(tmemLoad.getLoc(),
                                                   newLoadType, subSlice1);
    auto cvt1 = rewriter.create<ttg::ConvertLayoutOp>(
        tmemLoad.getLoc(), splitOp.getOutRHS().getType(), load1);
    rewriter.replaceOp(splitOp, {cvt0, cvt1});
    sinkLoad(load0, cvt0);
    sinkLoad(load1, cvt1);
    return success();
  }
};

// Pick an optimized tmem load layout based on its users. When there are
// multiple warpgroups tmem_load results can be distirbuted along M or N across
// the warpgroups. By default distribute along N but when there is a reduction
// along N dimension we want to distribute along M instead to avoid having to
// reduce across warps.
class TMemLoadReducePattern : public OpRewritePattern<ttng::TMEMLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttng::TMEMLoadOp tmemLoadOp,
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

class TritonNvidiaGPUOptimizeTMemLayoutsPass
    : public TritonNvidiaGPUOptimizeTMemLayoutsPassBase<
          TritonNvidiaGPUOptimizeTMemLayoutsPass> {
public:
  using BaseT = TritonNvidiaGPUOptimizeTMemLayoutsPassBase<
      TritonNvidiaGPUOptimizeTMemLayoutsPass>;
  using BaseT::BaseT;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<TMemSplitLoadPattern, TMemLoadReducePattern>(context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTritonNvidiaGPUOptimizeTMemLayoutsPass() {
  return std::make_unique<TritonNvidiaGPUOptimizeTMemLayoutsPass>();
}
