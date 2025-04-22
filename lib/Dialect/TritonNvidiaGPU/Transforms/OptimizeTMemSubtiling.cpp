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
//  %l  = ttng.tmem_load  %o : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
//                               -> tensor<128x256xf32, #blocked>
//  %r  = tt.reshape %l  : tensor<128x256xf32, #blocked>
//                               -> tensor<128x2x128xf32, #blocked4>
//  %t  = tt.trans   %r  {order = array<i32: 0, 2, 1>}
//                               -> tensor<128x128x2xf32, #blocked5>
//  %lhs, %rhs = tt.split %t
//
// becomes
//  %o0   = ttng.tmem_subslice %o { N = 0   }
//  %lhs  = ttng.tmem_load     %o0
//  %o1   = ttng.tmem_subslice %o { N = 128 }
//  %rhs  = ttng.tmem_load     %o1
//
// and if %lhs / %rhs are split again through the same reshape->trans->split
// pattern, the transformation is can match again so that each further
// split is materialised as an independent `ttng.tmem_subslice` / `ttng.tmem_load`
// pair.  Consequently, a chain such as
//
//   acc0, acc1  = split(permute(reshape(acc , ...)))
//   acc00, acc01 = split(permute(reshape(acc0, ...)))
//   acc10, acc11 = split(permute(reshape(acc1, ...)))
//
// is lowered to four independent TMEM loads operating on four disjoint
// subslices.
//
// clang-format on
// Strip away all intermediate ttg.convert_layout ops to reach the true
// producer.
static Value stripConvertLayout(Value v) {
  while (auto cvt = v.getDefiningOp<ttg::ConvertLayoutOp>())
    v = cvt.getSrc();
  return v;
}

class TMemSplitLoadPattern : public OpRewritePattern<tt::SplitOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::SplitOp splitOp,
                                PatternRewriter &rewriter) const override {
    // -----------------------------------------------------------------------
    // Match the pattern:
    //      splitOp
    //        ^  |
    //        |  +-- transOp(order = [0, 2, 1])
    //        |       ^  |
    //        |       |  +-- reshapeOp
    //        |       |        ^  |
    //        |       |        |  +-- (maybe convert_layout)
    //        |       |        +-- tmemLoad
    // -----------------------------------------------------------------------

    // Starting from the split source, peel off convert_layouts if any.
    Value src = stripConvertLayout(splitOp.getSrc());
    auto transOp = src.getDefiningOp<tt::TransOp>();
    if (!transOp || transOp.getOrder() != ArrayRef<int>({0, 2, 1}))
      return failure();
    auto reshapeOp = transOp.getSrc().getDefiningOp<tt::ReshapeOp>();
    if (!reshapeOp)
      return failure();

    // Peel off convert_layouts *below* the reshape as well.  This is required
    // for the recursive case where the producer of the reshape is the result
    // of an earlier optimisation pass (i.e. a convert_layout of a previous
    // tmem_load).
    Value reshapeSrc = stripConvertLayout(reshapeOp.getSrc());
    auto tmemLoad = reshapeSrc.getDefiningOp<ttng::TMEMLoadOp>();
    if (!tmemLoad)
      return failure();

    auto shape = reshapeOp.getResult().getType().getShape();
    // Ensure M dimension is preserved by the reshape.
    if (shape[0] != cast<RankedTensorType>(reshapeSrc.getType()).getShape()[0])
      return failure();
    int mDim = getShapePerCTA(tmemLoad.getSrc().getType())[0];
    // TODO: enable other M cases. (the layout is a bit more complex).
    if (mDim != 128)
      return failure();
    int splitNSize = shape[2];
    if (splitNSize < 8)
      return failure();

    // Create the two TMEM subslices and their corresponding loads.
    Value tmem = tmemLoad.getSrc(); // Could itself be a subslice.
    int numWarps = ttg::lookupNumWarps(tmemLoad);
    rewriter.setInsertionPoint(tmemLoad);

    auto createSliceLoad = [&](int64_t nOffset)
        -> std::pair<ttng::TMEMLoadOp, ttg::ConvertLayoutOp> {
      // Generate the subslice op.
      auto subSlice = rewriter.create<ttng::TMEMSubSliceOp>(
          tmemLoad.getLoc(), tmem, nOffset, splitNSize);

      // Choose a layout compatible with the slice size.
      Attribute distLayout = ttng::getTmemCompatibleLayout(
          mDim, splitNSize, splitOp.getOutLHS().getType(), numWarps);

      RankedTensorType newLoadType = RankedTensorType::get(
          splitOp.getOutLHS().getType().getShape(),
          splitOp.getOutLHS().getType().getElementType(), distLayout);

      // Generate the load and convert_layout back to the original layout.
      auto load = rewriter.create<ttng::TMEMLoadOp>(tmemLoad.getLoc(),
                                                    newLoadType, subSlice);
      auto cvt = rewriter.create<ttg::ConvertLayoutOp>(
          tmemLoad.getLoc(), splitOp.getOutLHS().getType(), load);

      return {load, cvt};
    };

    auto [load0, cvt0] = createSliceLoad(/*nOffset=*/0);
    auto [load1, cvt1] = createSliceLoad(/*nOffset=*/splitNSize);
    rewriter.replaceOp(splitOp, {cvt0, cvt1});

    // Sink the loads as close as possible to their use to save registers.
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
