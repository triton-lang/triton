#include "TritonAMDGPUTransforms/Passes.h" // IWYU pragma: keep
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUCONVERTTOTENSOROPS
#include "TritonAMDGPUTransforms/Passes.h.inc"
class TensorLoadLowering : public OpRewritePattern<DescriptorLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorLoadOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
    auto loc = op.getLoc();
    auto tensorType = op.getResult().getType();
    auto encoding = getEncodingFromDescriptor(op, tensorType, op.getDesc());
    if (!encoding) {
      op.emitError() << "Could not create encoding for descriptor load";
      return failure();
    }

    // given this descriptor and the encoding, the framework should be able to
    // compute the LDS size.
    MemDescType memDescType =
        MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                         encoding, sharedMemorySpace, /*mutableMemory=*/true);
    Value alloc = LocalAllocOp::create(rewriter, loc, memDescType);
    Value pred = arith::ConstantIntOp::create(rewriter, loc, 1, 32);

    amdgpu::AsyncTDMCopyGlobalToLocalOp::create(rewriter, loc, op.getDesc(),
                                                op.getIndices(), alloc, pred);
    amdgpu::AsyncTDMWait::create(rewriter, loc, ArrayRef<Value>{}, 0);
    rewriter.replaceOpWithNewOp<LocalLoadOp>(op, op.getType(), alloc);
    return success();
  }
};

struct TensorGatherLowering : public OpRewritePattern<DescriptorGatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorGatherOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
    auto loc = op.getLoc();
    auto tensorType = op.getResult().getType();

    auto encoding = getEncodingFromDescriptor(op, tensorType, op.getDesc());
    if (!encoding) {
      op.emitError() << "Could not create encoding for descriptor gather";
      return failure();
    }

    auto indices = op.getXOffsets();
    auto indicesType = cast<RankedTensorType>(indices.getType());
    auto idxEnc = getTDMGatherScatterIndexEncoding(op, indicesType);

    // NOTE: The shared TritonToTritonGPU conversion (GatherScatterOpPattern)
    // unconditionally applies an NVIDIA-oriented index layout. Because of
    // this, the indices arriving here already carry that layout, making default
    // index encoding never matches most desirable AMD index encoding, and
    // therefore an additional ConvertLayoutOp emitted.
    if (indicesType.getEncoding() != idxEnc) {
      auto newIdxType = RankedTensorType::get(
          indicesType.getShape(), indicesType.getElementType(), idxEnc);
      indices = ConvertLayoutOp::create(rewriter, loc, newIdxType, indices);
    }

    MemDescType memDescType =
        MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                         encoding, sharedMemorySpace, /*mutableMemory=*/true);
    Value alloc = LocalAllocOp::create(rewriter, loc, memDescType);
    Value pred = arith::ConstantIntOp::create(rewriter, loc, 1, 32);

    amdgpu::AsyncTDMGatherOp::create(rewriter, loc, op.getDesc(), indices,
                                     op.getYOffset(), alloc, pred);
    amdgpu::AsyncTDMWait::create(rewriter, loc, ArrayRef<Value>{}, 0);
    rewriter.replaceOpWithNewOp<LocalLoadOp>(op, op.getType(), alloc);
    return success();
  }
};

class TensorStoreLowering : public OpRewritePattern<DescriptorStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorStoreOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
    auto loc = op.getLoc();
    Value desc = op.getDesc();
    mlir::TypedValue<RankedTensorType> src = op.getSrc();
    auto tensorType = src.getType();
    auto encoding = getEncodingFromDescriptor(op, tensorType, desc);
    if (!encoding) {
      op.emitError() << "Could not create encoding for descriptor store";
      return failure();
    }

    MemDescType memDescType =
        MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                         encoding, sharedMemorySpace, /*mutableMemory=*/true);
    Value alloc = LocalAllocOp::create(rewriter, loc, memDescType, op.getSrc());
    amdgpu::AsyncTDMCopyLocalToGlobalOp::create(rewriter, loc, op.getDesc(),
                                                op.getIndices(), alloc,
                                                /*barrier=*/Value{});
    amdgpu::AsyncTDMWait::create(rewriter, loc, ArrayRef<Value>{}, 0);
    rewriter.eraseOp(op);
    return success();
  }
};

struct TensorScatterLowering : public OpRewritePattern<DescriptorScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorScatterOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
    auto loc = op.getLoc();
    Value desc = op.getDesc();
    mlir::TypedValue<RankedTensorType> src = op.getSrc();
    auto tensorType = src.getType();

    auto encoding = getEncodingFromDescriptor(op, tensorType, desc);
    if (!encoding) {
      op.emitError() << "Could not create encoding for descriptor scatter";
      return failure();
    }

    auto indices = op.getXOffsets();
    auto indicesType = cast<RankedTensorType>(indices.getType());
    auto idxEnc = getTDMGatherScatterIndexEncoding(op, indicesType);

    // NOTE: The shared TritonToTritonGPU conversion (GatherScatterOpPattern)
    // unconditionally applies an NVIDIA-oriented index layout. Re-layout the
    // indices into the AMD TDM-friendly encoding so the LLVM lowering can use
    // a single TDM instruction per row group.
    if (indicesType.getEncoding() != idxEnc) {
      auto newIdxType = RankedTensorType::get(
          indicesType.getShape(), indicesType.getElementType(), idxEnc);
      indices = ConvertLayoutOp::create(rewriter, loc, newIdxType, indices);
    }

    MemDescType memDescType =
        MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                         encoding, sharedMemorySpace, /*mutableMemory=*/true);
    Value alloc = LocalAllocOp::create(rewriter, loc, memDescType, src);
    amdgpu::AsyncTDMScatterOp::create(rewriter, loc, op.getDesc(), indices,
                                      op.getYOffset(), alloc,
                                      /*barrier=*/Value{});
    amdgpu::AsyncTDMWait::create(rewriter, loc, ArrayRef<Value>{}, 0);
    rewriter.eraseOp(op);
    return success();
  }
};

struct TritonAMDGPUConvertToTensorOps
    : impl::TritonAMDGPUConvertToTensorOpsBase<TritonAMDGPUConvertToTensorOps> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<TensorLoadLowering, TensorGatherLowering, TensorStoreLowering,
                 TensorScatterLowering>(context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
