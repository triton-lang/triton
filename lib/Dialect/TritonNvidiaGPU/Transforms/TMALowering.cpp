#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
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

class TMALoadLowering : public OpRewritePattern<ExperimentalDescriptorLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExperimentalDescriptorLoadOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
    auto loc = op.getLoc();
    auto tensorType = op.getResult().getType();
    auto order = getOrder(tensorType.getEncoding());
    auto ctaLayout = getCTALayout(tensorType.getEncoding());
    Attribute encoding = SharedEncodingAttr::get(tensorType.getContext(), 1, 1,
                                                 1, order, ctaLayout);
    if (tensorType.getRank() > 1) {
      encoding = SharedEncodingAttr::get(
          tensorType.getContext(), tensorType.getShape(), order, ctaLayout,
          tensorType.getElementType());
    }
    MemDescType memDescType =
        MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                         encoding, sharedMemorySpace, /*mutableMemory=*/true);
    Value alloc = rewriter.create<LocalAllocOp>(loc, memDescType);
    auto barrierCTALayout = CTALayoutAttr::get(
        /*context=*/tensorType.getContext(), /*CTAsPerCGA=*/{1},
        /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    auto barrierEncoding = SharedEncodingAttr::get(tensorType.getContext(), 1,
                                                   1, 1, {0}, barrierCTALayout);
    MemDescType barrierMemDescType =
        MemDescType::get({1}, rewriter.getI64Type(), barrierEncoding,
                         sharedMemorySpace, /*mutableMemory=*/true);
    Value barrierAlloc = rewriter.create<LocalAllocOp>(loc, barrierMemDescType);
    rewriter.create<InitBarrierOp>(loc, barrierAlloc, 1);
    int sizeInBytes = product(tensorType.getShape()) *
                      tensorType.getElementType().getIntOrFloatBitWidth() / 8;
    Value pred = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
    rewriter.create<triton::nvidia_gpu::BarrierExpectOp>(loc, barrierAlloc,
                                                         sizeInBytes, pred);
    rewriter.create<triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp>(
        loc, op.getDescPtr(), op.getIndices(), barrierAlloc, alloc, pred);
    Value phase = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    rewriter.create<WaitBarrierOp>(loc, barrierAlloc, phase);
    rewriter.create<InvalBarrierOp>(loc, barrierAlloc);
    rewriter.replaceOpWithNewOp<LocalLoadOp>(op, op.getType(), alloc);
    return success();
  }
};

class TMAStoreLowering
    : public OpRewritePattern<ExperimentalDescriptorStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExperimentalDescriptorStoreOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
    auto loc = op.getLoc();
    auto tensorType = op.getSrc().getType();
    auto order = getOrder(tensorType.getEncoding());
    auto ctaLayout = getCTALayout(tensorType.getEncoding());
    Attribute encoding = SharedEncodingAttr::get(tensorType.getContext(), 1, 1,
                                                 1, order, ctaLayout);
    if (tensorType.getRank() > 1) {
      encoding = SharedEncodingAttr::get(
          tensorType.getContext(), tensorType.getShape(), order, ctaLayout,
          tensorType.getElementType());
    }
    MemDescType memDescType =
        MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                         encoding, sharedMemorySpace, /*mutableMemory=*/true);
    Value alloc = rewriter.create<LocalAllocOp>(loc, memDescType, op.getSrc());
    rewriter.create<triton::nvidia_gpu::FenceAsyncSharedOp>(loc, false);
    rewriter.create<triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp>(
        loc, op.getDescPtr(), op.getIndices(), alloc);
    rewriter.create<triton::nvidia_gpu::TMAStoreWait>(loc, 0);
    rewriter.eraseOp(op);
    return success();
  }
};

class TensormapCreateLowering
    : public OpRewritePattern<ExperimentalTensormapCreateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExperimentalTensormapCreateOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();

    constexpr int32_t kTMASize = 128;
    constexpr int32_t kTMAAlign = 128;

    Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
    auto tensorType =
        RankedTensorType::get({kTMASize / 4}, IntegerType::get(ctx, 32));
    SmallVector<unsigned> order{0};
    auto ctaLayout = CTALayoutAttr::get(
        /*context=*/tensorType.getContext(), /*CTAsPerCGA=*/{1},
        /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    Attribute encoding =
        SharedEncodingAttr::get(ctx, /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1,
                                /*order=*/{0}, ctaLayout);
    MemDescType memDescType =
        MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                         encoding, sharedMemorySpace, /*mutableMemory=*/true);

    Value alloc = rewriter.create<LocalAllocOp>(loc, memDescType,
                                                /*src=*/Value{}, kTMAAlign);
    rewriter.create<triton::nvidia_gpu::TensormapLoadDescriptorOp>(
        loc, alloc, op.getTemplatePtr());
    rewriter.create<triton::nvidia_gpu::TensormapReplaceGlobalAddressOp>(
        loc, alloc, op.getGlobalAddress());
    rewriter.create<triton::nvidia_gpu::TensormapReplaceRankOp>(
        loc, alloc, rewriter.getI32IntegerAttr(op.getRank() - 1));
    for (int i = 0; i < op.getRank(); ++i) {
      auto ord = rewriter.getI32IntegerAttr(i);
      rewriter.create<triton::nvidia_gpu::TensormapReplaceBoxDimOp>(
          loc, alloc, ord, op.getBoxDim()[i]);
    }
    for (int i = 0; i < op.getRank(); ++i) {
      auto ord = rewriter.getI32IntegerAttr(i);
      rewriter.create<triton::nvidia_gpu::TensormapReplaceGlobalDimOp>(
          loc, alloc, ord, op.getGlobalDim()[i]);
    }
    for (int i = 0; i + 1 < op.getRank(); ++i) {
      auto ord = rewriter.getI32IntegerAttr(i);
      rewriter.create<triton::nvidia_gpu::TensormapReplaceGlobalStrideOp>(
          loc, alloc, ord, op.getGlobalStride()[i]);
    }
    for (int i = 0; i + 1 < op.getRank(); ++i) {
      auto ord = rewriter.getI32IntegerAttr(i);
      rewriter.create<triton::nvidia_gpu::TensormapReplaceElementStrideOp>(
          loc, alloc, ord, op.getElementStride()[i]);
    }
    rewriter.create<triton::nvidia_gpu::TensormapReplaceElemTypeOp>(
        loc, alloc, op.getElemType());
    rewriter.create<triton::nvidia_gpu::TensormapReplaceInterleaveLayoutOp>(
        loc, alloc, op.getInterleaveLayout());
    rewriter.create<triton::nvidia_gpu::TensormapReplaceSwizzleModeOp>(
        loc, alloc, op.getSwizzleMode());
    rewriter.create<triton::nvidia_gpu::TensormapReplaceFillModeOp>(
        loc, alloc, op.getFillMode());
    rewriter.create<triton::nvidia_gpu::TensormapCpFenceproxyOp>(
        loc, op.getDescPtr(), alloc);
    rewriter.create<LocalDeallocOp>(loc, alloc);
    rewriter.eraseOp(op);
    return success();
  }
};

class TritonNvidiaGPUTMALoweringPass
    : public TritonNvidiaGPUTMALoweringPassBase<
          TritonNvidiaGPUTMALoweringPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<TMALoadLowering, TMAStoreLowering, TensormapCreateLowering>(
        context);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTritonNvidiaGPUTMALoweringPass() {
  return std::make_unique<TritonNvidiaGPUTMALoweringPass>();
}
