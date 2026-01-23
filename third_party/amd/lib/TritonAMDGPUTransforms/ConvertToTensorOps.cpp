#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <deque>
#include <optional>

#include <memory>

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

    // Very important. For a shared layout to "work" the main thing we need is
    // the order (all the rest is about swizzling). We need to get the order
    // somewhere.
    SmallVector<unsigned> order = getOrder(tensorType);
    if (auto blockedLayout =
            dyn_cast<BlockedEncodingAttr>(tensorType.getEncoding())) {
      order = llvm::to_vector(blockedLayout.getOrder());
    }

    auto cgaLayout = getCGALayout(tensorType.getEncoding());
    // At this point, we don't have any information about how this load is used.
    // Hence, we cannot set padding information
    Attribute encoding = SwizzledSharedEncodingAttr::get(
        tensorType.getContext(), 1, 1, 1, order, cgaLayout);

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

    SmallVector<unsigned> order = getOrder(tensorType);
    if (auto blockedLayout =
            dyn_cast<BlockedEncodingAttr>(tensorType.getEncoding())) {
      order = llvm::to_vector(blockedLayout.getOrder());
    }

    auto cgaLayout = getCGALayout(tensorType.getEncoding());
    Attribute encoding = SwizzledSharedEncodingAttr::get(
        tensorType.getContext(), 1, 1, 1, order, cgaLayout);

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

struct TritonAMDGPUConvertToTensorOps
    : impl::TritonAMDGPUConvertToTensorOpsBase<TritonAMDGPUConvertToTensorOps> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<TensorLoadLowering, TensorStoreLowering>(context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
