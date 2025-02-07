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

template <typename TCGen5MMAOpTy>
class SyncMMALowering : public OpRewritePattern<TCGen5MMAOpTy> {
public:
  using OpRewritePattern<TCGen5MMAOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(TCGen5MMAOpTy op,
                                PatternRewriter &rewriter) const override {
    // If the op doesn't have synchronous semantic skip the pattern.
    if (op.getBarrier())
      return failure();
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
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
    op.getBarrierMutable().assign(barrierAlloc);

    rewriter.setInsertionPointAfter(op);
    Value phase = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    rewriter.create<WaitBarrierOp>(loc, barrierAlloc, phase);
    rewriter.create<InvalBarrierOp>(loc, barrierAlloc);
    return success();
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
    patterns
        .add<SyncMMALowering<TCGen5MMAOp>, SyncMMALowering<TCGen5MMAScaledOp>>(
            context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTritonNvidiaGPUMMALoweringPass() {
  return std::make_unique<TritonNvidiaGPUMMALoweringPass>();
}
