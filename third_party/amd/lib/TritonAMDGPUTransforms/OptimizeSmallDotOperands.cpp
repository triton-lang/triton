#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;

namespace {

class OptimizeLoadLayoutPattern : public mlir::RewritePattern {

public:
  explicit OptimizeLoadLayoutPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::DotOp::getOperationName(), 1, context) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dotOp = cast<triton::DotOp>(op);
    auto aCvt = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
        dotOp.getA().getDefiningOp());
    auto bCvt = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
        dotOp.getB().getDefiningOp());
    auto cOp = dotOp.getC();
    if (!aCvt || !bCvt)
      return mlir::failure();
    auto cShape = cOp.getType().getShape();
    auto aShape = aCvt.getType().getShape();
    if (cShape.size() != 2)
      return mlir::failure();
    unsigned m = cShape[0];
    unsigned n = cShape[1];
    unsigned k = aShape[1];
    if (m <= 4 && n >= 32) {
      auto loadOp =
          llvm::dyn_cast_or_null<triton::LoadOp>(bCvt.getSrc().getDefiningOp());
      if (!loadOp)
        return mlir::failure();
      auto loadTy = llvm::cast<RankedTensorType>(loadOp.getResult().getType());
      auto oldBLoadLayout =
          dyn_cast<triton::gpu::BlockedEncodingAttr>(loadTy.getEncoding());
      if (!oldBLoadLayout)
        return mlir::failure();
      auto ctx = dotOp.getContext();
      auto cLayout = llvm::dyn_cast<triton::gpu::BlockedEncodingAttr>(
          cOp.getType().getEncoding());
      if (!cLayout)
        return failure();
      unsigned numWarps = triton::gpu::getNumWarpsPerCTA(cLayout);
      unsigned numThreads = product(triton::gpu::getThreadsPerWarp(cLayout));
      SmallVector<unsigned> cWarpsPerCTA{std::min(m, numWarps),
                                         numWarps / std::min(m, numWarps)};
      SmallVector<unsigned> cThreadsPerWarp{m / cWarpsPerCTA[0],
                                            numThreads / (m / cWarpsPerCTA[0])};
      SmallVector<unsigned> cElemsPerThread{
          std::max(1u, m / (cThreadsPerWarp[0] * cWarpsPerCTA[0])),
          std::max(1u, n / (cThreadsPerWarp[1] * cWarpsPerCTA[1]))};
      auto order = cLayout.getOrder();
      auto ctaLayout = cLayout.getCTALayout();
      auto newCLayout = triton::gpu::BlockedEncodingAttr::get(
          ctx, cElemsPerThread, cThreadsPerWarp, cWarpsPerCTA, order,
          ctaLayout);
      auto elTy = aCvt.getType().getElementType();
      auto newALayout =
          triton::gpu::DotOperandEncodingAttr::get(ctx, 0, newCLayout, elTy);
      auto newBLayout =
          triton::gpu::DotOperandEncodingAttr::get(ctx, 1, newCLayout, elTy);

      auto bElemsPerThread = {k, cElemsPerThread[1]};
      auto bThreadsPerWarp = cThreadsPerWarp;
      auto bWarpsPerCTA = cWarpsPerCTA;
      auto newBLoadLayout = triton::gpu::BlockedEncodingAttr::get(
          ctx, bElemsPerThread, bThreadsPerWarp, bWarpsPerCTA, order,
          ctaLayout);

      if (newCLayout == cLayout && oldBLoadLayout == newBLoadLayout)
        return failure();

      rewriter.setInsertionPoint(loadOp);
      auto loadLoc = loadOp.getLoc();
      auto loadPtrTy = llvm::cast<RankedTensorType>(loadOp.getPtr().getType());
      auto newLoadPtrTy = RankedTensorType::get(
          loadPtrTy.getShape(), loadPtrTy.getElementType(), newBLoadLayout);
      auto newPtr = rewriter.create<triton::gpu::ConvertLayoutOp>(
          loadLoc, newLoadPtrTy, loadOp.getPtr());
      Value newMask;
      if (loadOp.getMask()) {
        auto loadMaskTy =
            llvm::cast<RankedTensorType>(loadOp.getMask().getType());
        auto newLoadMaskTy = RankedTensorType::get(
            loadMaskTy.getShape(), loadMaskTy.getElementType(), newBLoadLayout);
        newMask = rewriter.create<triton::gpu::ConvertLayoutOp>(
            loadLoc, newLoadMaskTy, loadOp.getMask());
      }
      auto newLoadOp = rewriter.replaceOpWithNewOp<triton::LoadOp>(
          loadOp, newPtr, newMask, loadOp.getCache(), loadOp.getEvict(),
          loadOp.getIsVolatile());

      if (dotOp.getA().getDefiningOp())
        rewriter.setInsertionPoint(dotOp.getA().getDefiningOp());
      else
        rewriter.setInsertionPoint(dotOp);
      auto aLoc = aCvt.getLoc();
      auto aCvtTy = dotOp.getA().getType();
      auto newATy = RankedTensorType::get(aCvtTy.getShape(),
                                          aCvtTy.getElementType(), newALayout);
      auto newA = rewriter.create<triton::gpu::ConvertLayoutOp>(aLoc, newATy,
                                                                aCvt.getSrc());

      if (dotOp.getB().getDefiningOp())
        rewriter.setInsertionPoint(dotOp.getB().getDefiningOp());
      else
        rewriter.setInsertionPoint(dotOp);
      auto bLoc = aCvt.getLoc();
      auto bCvtTy = dotOp.getB().getType();
      auto newBTy = RankedTensorType::get(bCvtTy.getShape(),
                                          bCvtTy.getElementType(), newBLayout);
      auto newB = rewriter.create<triton::gpu::ConvertLayoutOp>(bLoc, newBTy,
                                                                newLoadOp);

      if (dotOp.getC().getDefiningOp())
        rewriter.setInsertionPoint(dotOp.getC().getDefiningOp());
      else
        rewriter.setInsertionPoint(dotOp);
      auto cLoc = dotOp.getC().getLoc();
      auto cCvtTy = dotOp.getC().getType();
      auto newCTy = RankedTensorType::get(cCvtTy.getShape(),
                                          cCvtTy.getElementType(), newCLayout);
      auto newC = rewriter.create<triton::gpu::ConvertLayoutOp>(cLoc, newCTy,
                                                                dotOp.getC());

      rewriter.setInsertionPoint(dotOp);
      auto dotLoc = dotOp.getLoc();
      auto newDot = rewriter.create<triton::DotOp>(
          dotLoc, newA, newB, newC, dotOp.getInputPrecision(),
          dotOp.getMaxNumImpreciseAcc());
      auto backCvt = rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(
          dotOp, dotOp.getD().getType(), newDot);
      return success();
    }
    return mlir::failure();
  }
};

} // namespace

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

class TritonAMDGPUOptimizeSmallDotOperandsPass
    : public TritonAMDGPUOptimizeSmallDotOperandsBase<
          TritonAMDGPUOptimizeSmallDotOperandsPass> {

public:
  TritonAMDGPUOptimizeSmallDotOperandsPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);

    patterns.add<OptimizeLoadLayoutPattern>(context);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUOptimizeSmallDotOperandsPass() {
  return std::make_unique<TritonAMDGPUOptimizeSmallDotOperandsPass>();
}
