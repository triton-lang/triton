#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
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
  auto numCTAs = gpu::lookupNumCTAs(op);
  auto barrierCGALayout =
      gpu::CGAEncodingAttr::get1DLayout(tensorType.getContext(), numCTAs);
  auto barrierEncoding = gpu::SwizzledSharedEncodingAttr::get(
      tensorType.getContext(), 1, 1, 1, {0}, barrierCGALayout);
  gpu::MemDescType barrierMemDescType =
      gpu::MemDescType::get({numCTAs}, rewriter.getI64Type(), barrierEncoding,
                            sharedMemorySpace, /*mutableMemory=*/true);
  Value barrierAlloc =
      gpu::LocalAllocOp::create(rewriter, loc, barrierMemDescType);
  InitBarrierOp::create(rewriter, loc, barrierAlloc, 1);
  auto shapePerCTA = getShapePerCTA(encoding, tensorType.getShape());
  int sizeInBytes = product(shapePerCTA) *
                    getTMATransferBitWidth(tensorType.getElementType()) / 8;
  Value pred = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
  triton::nvidia_gpu::BarrierExpectOp::create(rewriter, loc, barrierAlloc,
                                              sizeInBytes, pred);
  createLoad(desc, barrierAlloc, alloc, pred);
  Value phase = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
  WaitBarrierOp::create(rewriter, loc, barrierAlloc, phase);
  // Conservatively synchronize CTAs for two-CTA MMA, since these completion
  // barriers are per-CTA.
  if (getModuleTwoCTAs(op))
    ClusterBarrierOp::create(rewriter, loc);
  InvalBarrierOp::create(rewriter, loc, barrierAlloc);
  replaceUsesWithLocalLoad(rewriter, op->getResult(0), alloc);
  op->erase();
}

class TMALoadLowering : public OpRewritePattern<DescriptorLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto createLoad = [&](Value desc, Value barrierAlloc, Value alloc,
                          Value pred) {
      triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp::create(
          rewriter, op.getLoc(), desc, op.getIndices(), barrierAlloc, alloc,
          pred, /*multicast=*/false, op.getCachePolicyAttr());
    };
    lowerTMALoad(op, op.getType(), op.getDesc(), createLoad, rewriter);
    return success();
  }
};

struct TMAGatherLowering : public OpRewritePattern<DescriptorGatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorGatherOp op,
                                PatternRewriter &rewriter) const override {
    Value xOffsets =
        sextI16ToI32Indices(op.getXOffsets(), rewriter, op.getLoc());

    auto createLoad = [&](Value desc, Value barrierAlloc, Value alloc,
                          Value pred) {
      triton::nvidia_gpu::AsyncTMAGatherOp::create(rewriter, op.getLoc(), desc,
                                                   xOffsets, op.getYOffset(),
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
  triton::nvidia_gpu::TMAStoreWaitOp::create(rewriter, loc, 0,
                                             /*read_only=*/false);
  rewriter.eraseOp(op);
}

struct TMAStoreLowering : public OpRewritePattern<DescriptorStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto createStore = [&](Value desc, Value alloc) {
      triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp::create(
          rewriter, op.getLoc(), desc, op.getIndices(), alloc);
    };
    lowerTMAStore(op, op.getSrc(), op.getDesc(), createStore, rewriter);
    return success();
  }
};

struct TMAReduceLowering : public OpRewritePattern<DescriptorReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto createStore = [&](Value desc, Value alloc) {
      triton::nvidia_gpu::AsyncTMAReduceOp::create(
          rewriter, op.getLoc(), op.getKind(), desc, op.getIndices(), alloc);
    };
    lowerTMAStore(op, op.getSrc(), op.getDesc(), createStore, rewriter);
    return success();
  }
};

struct TMAScatterLowering : public OpRewritePattern<DescriptorScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorScatterOp op,
                                PatternRewriter &rewriter) const override {
    Value xOffsets =
        sextI16ToI32Indices(op.getXOffsets(), rewriter, op.getLoc());

    auto createStore = [&](Value desc, Value alloc) {
      triton::nvidia_gpu::AsyncTMAScatterOp::create(
          rewriter, op.getLoc(), desc, xOffsets, op.getYOffset(), alloc);
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
    auto loc = op.getLoc();
    auto alloc = triton::gpu::GlobalScratchAllocOp::create(
        rewriter, loc, getPointerType(rewriter.getI8Type()), TMA_SIZE_BYTES,
        TMA_ALIGN, UnitAttr());
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

    // Host arguments are grid-constant. Finish each 64-wide FP6 template
    // once per CTA; an explicit address update can set both fields at once.
    for (auto func : m.getOps<triton::FuncOp>()) {
      if (!triton::isKernel(func) || func.isExternal())
        continue;
      OpBuilder b = OpBuilder::atBlockBegin(&func.getBody().front());
      for (auto arg : func.getArguments()) {
        auto type = dyn_cast<TensorDescType>(arg.getType());
        if (!type || !isNarrowFP6Map(type))
          continue;
        if (auto target = m->getAttrOfType<StringAttr>(gpu::AttrTargetName);
            target && target.getValue() != "cuda:103") {
          func.emitError("64-value FP6 TMA stores require SM103");
          return signalPassFailure();
        }
        auto needsUpdate = [](OpOperand &use) {
          return !isa<TensormapUpdateOp>(use.getOwner());
        };
        if (llvm::any_of(arg.getUses(), needsUpdate)) {
          Value updated = updateTMADesc(b, arg.getLoc(), arg);
          arg.replaceUsesWithIf(updated, needsUpdate);
        }
      }
    }
    mlir::RewritePatternSet patterns(context);
    patterns.add<TMALoadLowering, TMAGatherLowering, TMAStoreLowering,
                 TMAScatterLowering, TMAReduceLowering, TMACreateDescLowering>(
        context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
      return;
    }
    // Descriptor-fed MMA operands reuse the TMA tile. For operands that
    // originate in registers, put a full packing atom in each thread.
    m.walk([](gpu::LocalAllocOp alloc) {
      auto mem = alloc.getType();
      if (!alloc.getSrc() ||
          !gpu::NVMMASharedEncodingAttr::getFp6Type(mem.getEncoding()))
        return;
      OpBuilder b(alloc);
      auto ty = alloc.getSrc().getType();
      auto order = gpu::getOrder(mem);
      SmallVector<unsigned> spt(ty.getRank(), 1);
      spt[order.front()] = 16;
      auto layout = gpu::BlockedEncodingAttr::get(
          ty.getContext(), ty.getShape(), spt, order,
          gpu::lookupNumWarps(alloc), gpu::lookupThreadsPerWarp(b),
          gpu::getCGALayout(mem.getEncoding()));
      auto cvt = gpu::ConvertLayoutOp::create(
          b, alloc.getLoc(), ty.cloneWithEncoding(layout), alloc.getSrc());
      alloc.getSrcMutable().assign(cvt.getResult());
    });
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
