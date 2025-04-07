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

class TritonNvidiaGPUOptimizeTMemSubtilingPass
    : public TritonNvidiaGPUOptimizeTMemSubtilingPassBase<
          TritonNvidiaGPUOptimizeTMemSubtilingPass> {
public:
  using BaseT = TritonNvidiaGPUOptimizeTMemSubtilingPassBase<
      TritonNvidiaGPUOptimizeTMemSubtilingPass>;
  using BaseT::BaseT;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<TMemSplitLoadPattern>(context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTritonNvidiaGPUOptimizeTMemSubtilingPass() {
  return std::make_unique<TritonNvidiaGPUOptimizeTMemSubtilingPass>();
}
