#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUTMALOWERINGPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

static void
lowerTMALoad(Operation *op, RankedTensorType tensorType, Value desc,
             function_ref<void(Value, Value, Value, Value)> createLoad,
             PatternRewriter &rewriter) {
  MLIRContext *ctx = op->getContext();
  Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
  auto loc = op->getLoc();
  auto encoding = getEncodingFromDescriptor(op, tensorType, desc);
  gpu::MemDescType memDescType = gpu::MemDescType::get(
      tensorType.getShape(), tensorType.getElementType(), encoding,
      sharedMemorySpace, /*mutableMemory=*/true);
  auto alloc =
      gpu::LocalAllocOp::create(rewriter, loc, memDescType).getResult();
  auto barrierCTALayout = gpu::CTALayoutAttr::get(
      /*context=*/tensorType.getContext(), /*CTAsPerCGA=*/{1},
      /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto barrierEncoding = gpu::SwizzledSharedEncodingAttr::get(
      tensorType.getContext(), 1, 1, 1, {0}, barrierCTALayout);
  gpu::MemDescType barrierMemDescType =
      gpu::MemDescType::get({1}, rewriter.getI64Type(), barrierEncoding,
                            sharedMemorySpace, /*mutableMemory=*/true);
  Value barrierAlloc =
      gpu::LocalAllocOp::create(rewriter, loc, barrierMemDescType);
  InitBarrierOp::create(rewriter, loc, barrierAlloc, 1);
  auto shapePerCTA = getShapePerCTA(encoding, tensorType.getShape());
  int sizeInBytes = product(shapePerCTA) *
                    tensorType.getElementType().getIntOrFloatBitWidth() / 8;
  Value pred = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
  triton::nvidia_gpu::BarrierExpectOp::create(rewriter, loc, barrierAlloc,
                                              sizeInBytes, pred);
  createLoad(desc, barrierAlloc, alloc, pred);
  Value phase = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
  WaitBarrierOp::create(rewriter, loc, barrierAlloc, phase);
  InvalBarrierOp::create(rewriter, loc, barrierAlloc);
  replaceUsesWithLocalLoad(rewriter, op->getResult(0), alloc);
  op->erase();
}

class TMALoadLowering : public OpRewritePattern<DescriptorLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto createLoad = [&](Value tmaPtr, Value barrierAlloc, Value alloc,
                          Value pred) {
      auto indices = translateTMAIndices(
          rewriter, op.getLoc(),
          op.getDesc().getType().getBlockType().getEncoding(), op.getIndices());
      triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp::create(
          rewriter, op.getLoc(), tmaPtr, indices, barrierAlloc, alloc, pred);
    };
    lowerTMALoad(op, op.getType(), op.getDesc(), createLoad, rewriter);
    return success();
  }
};

struct TMAGatherLowering : public OpRewritePattern<DescriptorGatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorGatherOp op,
                                PatternRewriter &rewriter) const override {
    auto createLoad = [&](Value tmaPtr, Value barrierAlloc, Value alloc,
                          Value pred) {
      triton::nvidia_gpu::AsyncTMAGatherOp::create(
          rewriter, op.getLoc(), tmaPtr, op.getXOffsets(), op.getYOffset(),
          barrierAlloc, alloc, pred);
    };
    lowerTMALoad(op, op.getType(), op.getDesc(), createLoad, rewriter);
    return success();
  }
};

static void lowerTMAStore(Operation *op, mlir::TypedValue<RankedTensorType> src,
                          Value desc,
                          function_ref<void(Value, Value)> createStore,
                          PatternRewriter &rewriter) {
  MLIRContext *ctx = op->getContext();
  Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
  auto loc = op->getLoc();
  auto tensorType = src.getType();
  auto encoding = getEncodingFromDescriptor(op, src.getType(), desc);
  assert(isa<gpu::SharedEncodingTrait>(encoding));
  gpu::MemDescType memDescType = gpu::MemDescType::get(
      tensorType.getShape(), tensorType.getElementType(), encoding,
      sharedMemorySpace, /*mutableMemory=*/false);
  Value alloc = gpu::LocalAllocOp::create(rewriter, loc, memDescType, src);
  triton::nvidia_gpu::FenceAsyncSharedOp::create(rewriter, loc, false);
  createStore(desc, alloc);
  triton::nvidia_gpu::TMAStoreWaitOp::create(rewriter, loc, 0);
  rewriter.eraseOp(op);
}

struct TMAStoreLowering : public OpRewritePattern<DescriptorStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto createStore = [&](Value tmaPtr, Value alloc) {
      auto indices = translateTMAIndices(
          rewriter, op.getLoc(),
          op.getDesc().getType().getBlockType().getEncoding(), op.getIndices());
      triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp::create(
          rewriter, op.getLoc(), tmaPtr, indices, alloc);
    };
    lowerTMAStore(op, op.getSrc(), op.getDesc(), createStore, rewriter);
    return success();
  }
};

struct TMAReduceLowering : public OpRewritePattern<DescriptorReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto createStore = [&](Value tmaPtr, Value alloc) {
      auto indices = translateTMAIndices(
          rewriter, op.getLoc(),
          op.getDesc().getType().getBlockType().getEncoding(), op.getIndices());
      triton::nvidia_gpu::AsyncTMAReduceOp::create(
          rewriter, op.getLoc(), op.getKind(), tmaPtr, indices, alloc);
    };
    lowerTMAStore(op, op.getSrc(), op.getDesc(), createStore, rewriter);
    return success();
  }
};

struct TMAScatterLowering : public OpRewritePattern<DescriptorScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorScatterOp op,
                                PatternRewriter &rewriter) const override {
    auto createStore = [&](Value tmaPtr, Value alloc) {
      triton::nvidia_gpu::AsyncTMAScatterOp::create(rewriter, op.getLoc(),
                                                    tmaPtr, op.getXOffsets(),
                                                    op.getYOffset(), alloc);
    };
    lowerTMAStore(op, op.getSrc(), op.getDesc(), createStore, rewriter);
    return success();
  }
};

class TMACreateDescLowering : public OpRewritePattern<MakeTensorDescOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MakeTensorDescOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    auto alloc = triton::gpu::GlobalScratchAllocOp::create(
        rewriter, loc, getPointerType(rewriter.getI8Type()), TMA_SIZE_BYTES,
        TMA_ALIGN);
    if (failed(createTMADesc(alloc, op, rewriter))) {
      return failure();
    }
    TensormapFenceproxyAcquireOp::create(rewriter, loc, alloc.getResult());
    auto newDesc = ReinterpretTensorDescOp::create(rewriter, loc, op.getType(),
                                                   alloc.getResult());
    rewriter.replaceOp(op, newDesc);
    return success();
  }
};

} // anonymous namespace

class TritonNvidiaGPUTMALoweringPass
    : public impl::TritonNvidiaGPUTMALoweringPassBase<
          TritonNvidiaGPUTMALoweringPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<TMALoadLowering, TMAGatherLowering, TMAStoreLowering,
                 TMAScatterLowering, TMAReduceLowering, TMACreateDescLowering>(
        context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
