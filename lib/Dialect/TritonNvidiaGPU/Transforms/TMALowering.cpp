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

#include <memory>

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;

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
  assert(isa<SharedEncodingTrait>(encoding));
  if (descBlockType.getShape() == tensorType.getShape())
    return encoding;

  // Handle rank reducing loads
  auto ctx = encoding.getContext();
  auto rankDiff = descBlockType.getRank() - tensorType.getRank();
  if (auto nvmmaEnc = dyn_cast<NVMMASharedEncodingAttr>(encoding)) {
    auto existingCta = nvmmaEnc.getCTALayout();
    auto newCtaEnc =
        CTALayoutAttr::get(ctx, existingCta.getCTAsPerCGA().slice(rankDiff),
                           existingCta.getCTASplitNum().slice(rankDiff),
                           existingCta.getCTAOrder().slice(rankDiff));

    return NVMMASharedEncodingAttr::get(
        ctx, nvmmaEnc.getSwizzlingByteWidth(), nvmmaEnc.getTransposed(),
        nvmmaEnc.getElementBitWidth(), nvmmaEnc.getFp4Padded(), newCtaEnc);
  }
  if (auto swizEnc = dyn_cast<SwizzledSharedEncodingAttr>(encoding)) {
    auto existingCta = swizEnc.getCTALayout();
    auto newCtaEnc =
        CTALayoutAttr::get(ctx, existingCta.getCTAsPerCGA().slice(rankDiff),
                           existingCta.getCTASplitNum().slice(rankDiff),
                           existingCta.getCTAOrder().slice(rankDiff));
    return SwizzledSharedEncodingAttr::get(
        ctx, swizEnc.getVec(), swizEnc.getPerPhase(), swizEnc.getMaxPhase(),
        swizEnc.getOrder().slice(rankDiff), newCtaEnc);
  }

  constexpr auto msg = "Internal Error: Unhandled tensor descriptor encoding";
  op->emitError() << msg;
  llvm::report_fatal_error(msg);
}

} // namespace

namespace mlir::triton::nvidia_gpu {

void createBarrierExpectOp(Location loc, OpBuilder &rewriter,
                           SmallVector<Operation *> const &ops,
                           Value barrierAlloc) {
  auto getTensorTypeAndDesc =
      [](Operation *op) -> std::pair<RankedTensorType, Value> {
    if (auto loadOp = dyn_cast<DescriptorLoadOp>(op)) {
      return {loadOp.getType(), loadOp.getDesc()};
    } else if (auto gatherOp = dyn_cast<DescriptorGatherOp>(op)) {
      return {gatherOp.getType(), gatherOp.getDesc()};
    } else {
      llvm_unreachable("Unsupported operation type");
    }
  };
  int sizeInBytes = 0;
  for (auto op : ops) {
    auto [tensorType, desc] = getTensorTypeAndDesc(op);
    auto encoding = getEncodingFromDescriptor(op, tensorType, desc);
    auto shapePerCTA = getShapePerCTA(encoding, tensorType.getShape());
    sizeInBytes += product(shapePerCTA) *
                   tensorType.getElementType().getIntOrFloatBitWidth() / 8;
  }
  Value pred = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
  rewriter.create<triton::nvidia_gpu::BarrierExpectOp>(loc, barrierAlloc,
                                                       sizeInBytes, pred);
}

void createTMALoad(DescriptorLoadOp op, OpBuilder &rewriter, Value barrierAlloc,
                   Value alloc, Value pred) {
  Value tmaPtr = rewriter.create<triton::nvidia_gpu::TensorDescToTMAPtrOp>(
      op.getLoc(), op.getDesc());
  auto indices = translateTMAIndices(
      rewriter, op.getLoc(),
      op.getDesc().getType().getBlockType().getEncoding(), op.getIndices());
  rewriter.create<triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp>(
      op.getLoc(), tmaPtr, indices, barrierAlloc, alloc, pred);
};

void createTMAGather(DescriptorGatherOp op, OpBuilder &rewriter,
                     Value barrierAlloc, Value alloc, Value pred) {
  Value tmaPtr = rewriter.create<triton::nvidia_gpu::TensorDescToTMAPtrOp>(
      op.getLoc(), op.getDesc());
  rewriter.create<triton::nvidia_gpu::AsyncTMAGatherOp>(
      op.getLoc(), tmaPtr, op.getXOffsets(), op.getYOffset(), barrierAlloc,
      alloc, pred);
}

} // namespace mlir::triton::nvidia_gpu

namespace {
static void lowerTMALoad(Operation *op, RankedTensorType tensorType, Value desc,
                         function_ref<void(Value, Value, Value)> createLoad,
                         PatternRewriter &rewriter) {
  MLIRContext *ctx = op->getContext();
  Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
  auto loc = op->getLoc();
  auto encoding = getEncodingFromDescriptor(op, tensorType, desc);
  MemDescType memDescType =
      MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                       encoding, sharedMemorySpace, /*mutableMemory=*/true);
  auto alloc = rewriter.create<LocalAllocOp>(loc, memDescType).getResult();
  auto barrierCTALayout = CTALayoutAttr::get(
      /*context=*/tensorType.getContext(), /*CTAsPerCGA=*/{1},
      /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto barrierEncoding = SwizzledSharedEncodingAttr::get(
      tensorType.getContext(), 1, 1, 1, {0}, barrierCTALayout);
  MemDescType barrierMemDescType =
      MemDescType::get({1}, rewriter.getI64Type(), barrierEncoding,
                       sharedMemorySpace, /*mutableMemory=*/true);
  Value barrierAlloc = rewriter.create<LocalAllocOp>(loc, barrierMemDescType);
  rewriter.create<InitBarrierOp>(loc, barrierAlloc, 1);
  createBarrierExpectOp(loc, rewriter, {op}, barrierAlloc);
  auto mod = op->getParentOfType<ModuleOp>();
  if (triton::gpu::TritonGPUDialect::isWarpSpecialized(mod)) {
    insertBarrier(rewriter, loc);
  }
  Value pred = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
  createLoad(barrierAlloc, alloc, pred);
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
    auto createLoad = [&](Value barrierAlloc, Value alloc, Value pred) {
      createTMALoad(op, rewriter, barrierAlloc, alloc, pred);
    };
    lowerTMALoad(op, op.getType(), op.getDesc(), createLoad, rewriter);
    return success();
  }
};

struct TMAGatherLowering : public OpRewritePattern<DescriptorGatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorGatherOp op,
                                PatternRewriter &rewriter) const override {
    auto createLoad = [&](Value barrierAlloc, Value alloc, Value pred) {
      Value tmaPtr = rewriter.create<triton::nvidia_gpu::TensorDescToTMAPtrOp>(
          op.getLoc(), op.getDesc());
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
  assert(isa<SharedEncodingTrait>(encoding));
  MemDescType memDescType =
      MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                       encoding, sharedMemorySpace, /*mutableMemory=*/true);

  bool isWs = triton::gpu::TritonGPUDialect::isWarpSpecialized(
      op->getParentOfType<ModuleOp>());

  Value alloc = rewriter.create<LocalAllocOp>(loc, memDescType, src);
  rewriter.create<triton::nvidia_gpu::FenceAsyncSharedOp>(loc, false);

  if (isWs) {
    // Barrier required for test_tma_persistent_blackwell with subtiling.
    // Manual insertion ensures correctness; membar analysis isn't run with
    // AutoWS For SWP, syncthread must be added by membar here, because it's
    // between STS shared fence and TMA issuing.
    insertBarrier(rewriter, loc);
  }

  Value tmaPtr =
      rewriter.create<triton::nvidia_gpu::TensorDescToTMAPtrOp>(loc, desc);
  createStore(tmaPtr, alloc);
  rewriter.create<triton::nvidia_gpu::TMAStoreWaitOp>(loc, 0);

  if (isWs) {
    // Ensure all threads arrive at this point to avoid race conditions between
    // two TMA stores in Blackwell tests with sub-tiling enabled. Without this
    // barrier, TMAStoreWaitOp might be executed by another warp that is
    // slightly ahead of the warp issuing AsyncTMACopyLocalToGlobal. The barrier
    // ensures that all warps proceed simultaneously after the data is copied.
    insertBarrier(rewriter, loc);
  }

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
    rewriter.create<triton::ExperimentalTensormapFenceproxyAcquireOp>(
        loc, alloc.getResult());
    auto newDesc = rewriter.create<triton::ReinterpretTensorDescOp>(
        loc, op.getType(), alloc.getResult());
    rewriter.replaceOp(op, newDesc);
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
    patterns.add<TMALoadLowering, TMAGatherLowering, TMAStoreLowering,
                 TMAScatterLowering, TMAReduceLowering, TMACreateDescLowering>(
        context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTritonNvidiaGPUTMALoweringPass() {
  return std::make_unique<TritonNvidiaGPUTMALoweringPass>();
}
