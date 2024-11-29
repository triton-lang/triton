#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
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
    // TOOD
    if (tensorType.getRank() > 1) {
      // The following SharedEncodingAttr constructor creates SMEM encoding with
      // hasLeadingOffset = true, which is not currently supported for
      // higher-rank TMA.
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
    Value tmaPtr = rewriter.create<triton::nvidia_gpu::TensorDescToTMAPtrOp>(
        loc, op.getDesc());
    rewriter.create<triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp>(
        loc, tmaPtr, op.getIndices(), barrierAlloc, alloc, pred);
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
    if (tensorType.getRank() == 2) {
      // The following SharedEncodingAttr constructor creates SMEM encoding with
      // hasLeadingOffset = true, which is not currently supported for
      // higher-rank TMA.
      encoding = SharedEncodingAttr::get(
          tensorType.getContext(), tensorType.getShape(), order, ctaLayout,
          tensorType.getElementType());
    }
    MemDescType memDescType =
        MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                         encoding, sharedMemorySpace, /*mutableMemory=*/true);
    Value alloc = rewriter.create<LocalAllocOp>(loc, memDescType, op.getSrc());
    rewriter.create<triton::nvidia_gpu::FenceAsyncSharedOp>(loc, false);
    Value tmaPtr = rewriter.create<triton::nvidia_gpu::TensorDescToTMAPtrOp>(
        loc, op.getDesc());
    rewriter.create<triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp>(
        loc, tmaPtr, op.getIndices(), alloc);
    rewriter.create<triton::nvidia_gpu::TMAStoreWait>(loc, 0);
    rewriter.eraseOp(op);
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
    constexpr auto kTmaNbytes = 128;
    constexpr auto kTmaAlignment = 128;
    auto alloc = rewriter.create<triton::gpu::GlobalScratchAllocOp>(
        loc, getPointerType(rewriter.getI8Type()), kTmaNbytes, kTmaAlignment);
    auto mkI32Constant = [&](int32_t val) {
      return rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(val));
    };

    auto elemType = op.getBase().getType().getPointeeType();
    auto elemSize = elemType.getIntOrFloatBitWidth() / 8;

    int32_t contig_dim_size = op.getTensorShape().back();
    int32_t contig_dim_size_in_bytes = contig_dim_size * elemSize;
    if (contig_dim_size_in_bytes > 128) {
      contig_dim_size = 128 / elemSize;
    }
    llvm::SmallVector<Value> boxDim;
    boxDim.push_back(mkI32Constant(contig_dim_size));
    for (int k = op.getTensorShape().size() - 2; k >= 0; --k) {
      boxDim.push_back(mkI32Constant(op.getTensorShape()[k]));
    }

    int32_t swizzle_mode;
    if (contig_dim_size_in_bytes >= 128) {
      swizzle_mode = 3;
    } else if (contig_dim_size_in_bytes == 64) {
      swizzle_mode = 2;
    } else if (contig_dim_size_in_bytes == 32) {
      swizzle_mode = 1;
    } else {
      op->emitError()
          << "contiguous box dimension must be at least 32 bytes but got "
          << contig_dim_size_in_bytes;
      return failure();
    }

    Value elemSizeVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(elemSize));
    Value globalStride =
        rewriter.create<arith::MulIOp>(loc, op.getStrides()[0], elemSizeVal);
    // TODO: Workaround for ptxas bug, remove when we update ptxas
    Value four = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(4));
    globalStride = rewriter.create<arith::ShRSIOp>(loc, globalStride, four);

    int elemTypeEnum;
    switch (elemSize) {
    case 1: {
      elemTypeEnum = 0;
      break;
    }
    case 2: {
      elemTypeEnum = 1;
      break;
    }
    case 4: {
      elemTypeEnum = 2;
      break;
    }
    default: {
      op->emitError()
          << "Tensor descriptor element type must have size 1, 2, or 4 but got "
          << elemSize;
      return failure();
    }
    }

    auto one = mkI32Constant(1);
    rewriter.create<triton::ExperimentalTensormapCreateOp>(
        loc,
        /*desc_ptr=*/alloc.getResult(),
        /*global_address=*/op.getBase(),
        /*box_dim=*/boxDim,
        /*global_dim=*/ValueRange{op.getShape()[1], op.getShape()[0]},
        /*global_stride=*/ValueRange{globalStride},
        /*element_strides=*/ValueRange{one, one},
        /*elem_type*/ rewriter.getI32IntegerAttr(elemTypeEnum),
        /*interleave_layout*/ rewriter.getI32IntegerAttr(0),
        /*swizzle_mode=*/rewriter.getI32IntegerAttr(swizzle_mode),
        /*fill_mode=*/rewriter.getI32IntegerAttr(0));
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
    patterns.add<TMALoadLowering, TMAStoreLowering, TMACreateDescLowering>(
        context);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTritonNvidiaGPUTMALoweringPass() {
  return std::make_unique<TritonNvidiaGPUTMALoweringPass>();
}
