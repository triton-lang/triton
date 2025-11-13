#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LayoutUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include <memory>

#define DEBUG_TYPE "tritonamdgpu-optimize-dot-operands"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::amdgpu {

namespace {

// This pattern creates LocalAllocOp and LocalLoadOp with unswizzled shared
// layout for the scale operand used in ScaledUpcastFp4Op/ScaledUpcastFp8Op.
// StreamPipeliner will respect the layout created here and pipeline ops
// according to the need.
//
// It matches
// tt.load -> ... -> amdg.scaled_upcast_x
//
// And rewrites it to
// tt.load -> ttg.local_alloc -> ttg.local_load -> ... -> amdg.scaled_upcast_x
template <typename OpTy>
class AllocSharedMemForUpcastedScales : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  AllocSharedMemForUpcastedScales(MLIRContext *context,
                                  triton::AMD::ISAFamily isaFamily)
      : OpRewritePattern<OpTy>(context), isaFamily(isaFamily) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (isaFamily != mlir::triton::AMD::ISAFamily::CDNA4)
      return rewriter.notifyMatchFailure(op, "NYI: Only supported on CDNA4");

    auto forOp = op->template getParentOfType<scf::ForOp>();
    if (!forOp)
      return rewriter.notifyMatchFailure(op,
                                         "Don't alloc lds outside for loop");

    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    SetVector<Operation *> slice;
    (void)getBackwardSlice(op.getOperand(1), &slice, options);
    tt::LoadOp loadOp;
    unsigned cnt = 0;
    bool hasAllocatedLDS = false;
    for (auto &op : slice) {
      if (isa<tt::LoadOp>(op)) {
        loadOp = dyn_cast<tt::LoadOp>(op);
        cnt++;
      } else if (isa<ttg::LocalLoadOp>(op)) {
        hasAllocatedLDS = true;
        break;
      }
    }

    if (hasAllocatedLDS)
      return rewriter.notifyMatchFailure(
          op, "There's already lds allocation in the def chain.");

    if (!loadOp || cnt != 1)
      return rewriter.notifyMatchFailure(
          op, "Require exactly 1 load in the def chain.");

    LDBG("Found load of scale: " << loadOp << " for ScaleUpcast: " << op);
    auto srcTy = dyn_cast<RankedTensorType>(loadOp.getType());
    auto sharedOrder = ttg::getOrderForMemory(srcTy);
    auto ctaLayout = ttg::getCTALayout(srcTy.getEncoding());

    auto ctx = loadOp.getContext();
    auto attr = ttg::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, sharedOrder,
                                                     ctaLayout);
    Location loc = loadOp.getLoc();
    auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);
    rewriter.setInsertionPointAfter(loadOp);
    auto alloc = ttg::LocalAllocOp::create(
        rewriter, loc,
        ttg::MemDescType::get(srcTy.getShape(), srcTy.getElementType(), attr,
                              sharedMemorySpace),
        loadOp.getResult());
    LDBG("Create alloc: " << alloc);

    auto localLoad = ttg::LocalLoadOp::create(rewriter, loc, srcTy, alloc);
    LDBG("Create localload: " << localLoad);

    rewriter.replaceAllUsesExcept(loadOp.getResult(), localLoad, alloc);
    return success();
  }

private:
  triton::AMD::ISAFamily isaFamily;
};
} // namespace

#define GEN_PASS_DEF_TRITONAMDGPUOPTIMIZEDOTOPERANDS
#include "TritonAMDGPUTransforms/Passes.h.inc"

class TritonAMDGPUOptimizeDotOperands
    : public impl::TritonAMDGPUOptimizeDotOperandsBase<
          TritonAMDGPUOptimizeDotOperands> {
public:
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    auto isaFamily = triton::AMD::deduceISAFamily(archGenerationName);
    patterns
        .add<AllocSharedMemForUpcastedScales<tt::amdgpu::ScaledUpcastFp8Op>,
             AllocSharedMemForUpcastedScales<tt::amdgpu::ScaledUpcastFp4Op>>(
            context, isaFamily);
    ttg::ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

void registerTritonAMDGPUOptimizeDotOperands() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createTritonAMDGPUOptimizeDotOperands();
  });
}

} // namespace mlir::triton::amdgpu
