#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include <memory>

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;

Value allocateBarrier(mlir::MLIRContext *ctx, PatternRewriter &rewriter,
                      Location loc) {
  Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
  auto barrierCTALayout = CTALayoutAttr::get(
      /*context=*/ctx, /*CTAsPerCGA=*/{1},
      /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto barrierEncoding =
        SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, barrierCTALayout);
  MemDescType barrierMemDescType =
      MemDescType::get({1}, rewriter.getI64Type(), barrierEncoding,
                       sharedMemorySpace, /*mutableMemory=*/true);
  Value barrierAlloc =
      rewriter.create<LocalAllocOp>(loc, barrierMemDescType, Value());
  rewriter.create<InitBarrierOp>(loc, barrierAlloc, 1);
  return barrierAlloc;
}

class SyncMMALowering : public OpRewritePattern<TCGen5MMAOp> {
public:
  using OpRewritePattern<TCGen5MMAOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TCGen5MMAOp op,
                                PatternRewriter &rewriter) const override {
    // If the op doesn't have synchronous semantic skip the pattern.
    if (op.getBarrier())
      return failure();
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    auto barrierAlloc = allocateBarrier(ctx, rewriter, loc);
    op.getBarrierMutable().assign(barrierAlloc);

    rewriter.setInsertionPointAfter(op);
    Value phase = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    rewriter.create<WaitBarrierOp>(loc, barrierAlloc, phase);
    rewriter.create<InvalBarrierOp>(loc, barrierAlloc);
    return success();
  }
};

class SyncMMAScaledLowering : public OpRewritePattern<TCGen5MMAScaledOp> {
public:
  using OpRewritePattern<TCGen5MMAScaledOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TCGen5MMAScaledOp op,
                                PatternRewriter &rewriter) const override {
    // If the op doesn't have synchronous semantic skip the pattern.
    if (op.getBarrier())
      return failure();
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    auto barrierAlloc = allocateBarrier(ctx, rewriter, loc);

    op.getBarrierMutable().assign(barrierAlloc);

    rewriter.setInsertionPointAfter(op);
    Value phase = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    rewriter.create<WaitBarrierOp>(loc, barrierAlloc, phase);
    rewriter.create<InvalBarrierOp>(loc, barrierAlloc);

    // attachBarrierToTmemCopy(op.getAScale(), rewriter);
    // attachBarrierToTmemCopy(op.getBScale(), rewriter);

    return success();
  }

private:
  void attachBarrierToTmemCopy(Value scale, PatternRewriter &rewriter) const {
    OpBuilder::InsertionGuard g(rewriter);

    for (auto user : scale.getUsers()) {
      if (auto tmemCopy = dyn_cast<TMEMCopyOp>(user)) {
        rewriter.setInsertionPoint(tmemCopy);
        auto loc = tmemCopy.getLoc();
        auto barrier =
            allocateBarrier(tmemCopy.getContext(), rewriter, tmemCopy.getLoc());
        tmemCopy.getBarrierMutable().assign(barrier);

        rewriter.setInsertionPointAfter(tmemCopy);
        Value phase = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
        rewriter.create<WaitBarrierOp>(loc, barrier, phase);
        rewriter.create<InvalBarrierOp>(loc, barrier);
      }
    }
  }
};

class TritonNvidiaGPUMMALoweringPass
    : public TritonNvidiaGPUMMALoweringPassBase<
          TritonNvidiaGPUMMALoweringPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<SyncMMALowering, SyncMMAScaledLowering>(context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTritonNvidiaGPUMMALoweringPass() {
  return std::make_unique<TritonNvidiaGPUMMALoweringPass>();
}
