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

static Attribute getEncoding(Operation *op, RankedTensorType tensorType,
                             Value desc) {
  auto descBlockType = cast<TensorDescType>(desc.getType()).getBlockType();
  Attribute encoding = descBlockType.getEncoding();
  if (!encoding) {
    constexpr auto msg =
        "Internal Error: Tensor descriptor should have encoding set";
    op->emitError() << msg;
    llvm::report_fatal_error(msg);
  }
  assert(isa<gpu::SharedEncodingTrait>(encoding));
  if (descBlockType.getShape() == tensorType.getShape())
    return encoding;

  // Handle rank reducing loads
  auto ctx = encoding.getContext();
  auto rankDiff = descBlockType.getRank() - tensorType.getRank();
  if (auto nvmmaEnc = dyn_cast<gpu::NVMMASharedEncodingAttr>(encoding)) {
    auto existingCta = nvmmaEnc.getCTALayout();
    auto newCtaEnc = gpu::CTALayoutAttr::get(
        ctx, existingCta.getCTAsPerCGA().slice(rankDiff),
        existingCta.getCTASplitNum().slice(rankDiff),
        existingCta.getCTAOrder().slice(rankDiff));

    return gpu::NVMMASharedEncodingAttr::get(
        ctx, nvmmaEnc.getSwizzlingByteWidth(), nvmmaEnc.getTransposed(),
        nvmmaEnc.getElementBitWidth(), nvmmaEnc.getFp4Padded(), newCtaEnc);
  }
  if (auto swizEnc = dyn_cast<gpu::SwizzledSharedEncodingAttr>(encoding)) {
    auto existingCta = swizEnc.getCTALayout();
    auto newCtaEnc = gpu::CTALayoutAttr::get(
        ctx, existingCta.getCTAsPerCGA().slice(rankDiff),
        existingCta.getCTASplitNum().slice(rankDiff),
        existingCta.getCTAOrder().slice(rankDiff));
    return gpu::SwizzledSharedEncodingAttr::get(
        ctx, swizEnc.getVec(), swizEnc.getPerPhase(), swizEnc.getMaxPhase(),
        swizEnc.getOrder().slice(rankDiff), newCtaEnc);
  }

  constexpr auto msg = "Internal Error: Unhandled tensor descriptor encoding";
  op->emitError() << msg;
  llvm::report_fatal_error(msg);
}

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
  auto alloc = rewriter.create<gpu::LocalAllocOp>(loc, memDescType).getResult();
  auto barrierCTALayout = gpu::CTALayoutAttr::get(
      /*context=*/tensorType.getContext(), /*CTAsPerCGA=*/{1},
      /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto barrierEncoding = gpu::SwizzledSharedEncodingAttr::get(
      tensorType.getContext(), 1, 1, 1, {0}, barrierCTALayout);
  gpu::MemDescType barrierMemDescType =
      gpu::MemDescType::get({1}, rewriter.getI64Type(), barrierEncoding,
                            sharedMemorySpace, /*mutableMemory=*/true);
  Value barrierAlloc =
      rewriter.create<gpu::LocalAllocOp>(loc, barrierMemDescType);
  rewriter.create<InitBarrierOp>(loc, barrierAlloc, 1);
  auto shapePerCTA = getShapePerCTA(encoding, tensorType.getShape());
  int sizeInBytes = product(shapePerCTA) *
                    tensorType.getElementType().getIntOrFloatBitWidth() / 8;
  Value pred = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
  rewriter.create<triton::nvidia_gpu::BarrierExpectOp>(loc, barrierAlloc,
                                                       sizeInBytes, pred);
  createLoad(desc, barrierAlloc, alloc, pred);
  Value phase = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  rewriter.create<WaitBarrierOp>(loc, barrierAlloc, phase);
  rewriter.create<InvalBarrierOp>(loc, barrierAlloc);
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
      rewriter.create<triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp>(
          op.getLoc(), tmaPtr, indices, barrierAlloc, alloc, pred);
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
      rewriter.create<triton::nvidia_gpu::AsyncTMAGatherOp>(
          op.getLoc(), tmaPtr, op.getXOffsets(), op.getYOffset(), barrierAlloc,
          alloc, pred);
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
      sharedMemorySpace, /*mutableMemory=*/true);
  Value alloc = rewriter.create<gpu::LocalAllocOp>(loc, memDescType, src);
  rewriter.create<triton::nvidia_gpu::FenceAsyncSharedOp>(loc, false);
  createStore(desc, alloc);
  rewriter.create<triton::nvidia_gpu::TMAStoreWaitOp>(loc, 0);
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
      rewriter.create<triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp>(
          op.getLoc(), tmaPtr, indices, alloc);
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
      rewriter.create<triton::nvidia_gpu::AsyncTMAReduceOp>(
          op.getLoc(), op.getKind(), tmaPtr, indices, alloc);
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
      rewriter.create<triton::nvidia_gpu::AsyncTMAScatterOp>(
          op.getLoc(), tmaPtr, op.getXOffsets(), op.getYOffset(), alloc);
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
    auto alloc = rewriter.create<triton::gpu::GlobalScratchAllocOp>(
        loc, getPointerType(rewriter.getI8Type()), TMA_SIZE_BYTES, TMA_ALIGN);
    if (failed(createTMADesc(alloc, op, rewriter))) {
      return failure();
    }
    rewriter.create<TensormapFenceproxyAcquireOp>(loc, alloc.getResult());
    auto newDesc = rewriter.create<ReinterpretTensorDescOp>(loc, op.getType(),
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
