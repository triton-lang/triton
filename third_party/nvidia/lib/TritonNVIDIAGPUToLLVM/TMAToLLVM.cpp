#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/TypeUtilities.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"

#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"

#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::nvidia_gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

void tensormap_cp_fenceproxy(Location loc, MLIRContext *ctx,
                             ConversionPatternRewriter &rewriter, Value outPtr,
                             Value inPtr) {
  PTXBuilder ptxBuilder;
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  // prepare asm operands
  auto *outAddrOpr = ptxBuilder.newAddrOperand(outPtr, "l");
  auto *inAddrOpr = ptxBuilder.newAddrOperand(inPtr, "l");
  auto *sizeOpr = ptxBuilder.newConstantOperand(TMA_SIZE_BYTES);

  // Define the instruction opcode
  auto &cp = *ptxBuilder.create("tensormap.cp_fenceproxy.global.shared::cta."
                                "tensormap::generic.release.gpu.sync.aligned");

  // Execute collectively on first warp in block
  constexpr int kWarpSize = 32;
  Value threadId = getThreadId(rewriter, loc);
  Value pred = b.icmp_slt(threadId, b.i32_val(kWarpSize));
  cp(outAddrOpr, inAddrOpr, sizeOpr).predicate(pred);

  ptxBuilder.launch(rewriter, loc, void_ty(ctx));
};

void tensormap_replace_generic(Location loc, MLIRContext *ctx,
                               ConversionPatternRewriter &rewriter,
                               StringRef fieldName, Value descPtr, Value newVal,
                               std::optional<int32_t> ord = std::nullopt) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (isa<LLVM::LLVMPointerType>(newVal.getType()))
    newVal = b.ptrtoint(IntegerType::get(ctx, 64), newVal);
  SmallVector<Value> args{descPtr};
  if (ord)
    args.push_back(b.i32_val(*ord));
  args.push_back(newVal);
  LLVM::createLLVMIntrinsicCallOp(
      rewriter, loc, ("llvm.nvvm.tensormap.replace." + fieldName).str(),
      TypeRange{}, args);
}

void tensormap_replace_generic(Location loc, MLIRContext *ctx,
                               ConversionPatternRewriter &rewriter,
                               StringRef fieldName, Value descPtr,
                               int32_t newVal) {
  // The descriptor is zero-initialized before its fields are populated.
  if (newVal == 0)
    return;
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  tensormap_replace_generic(loc, ctx, rewriter, fieldName, descPtr,
                            b.i32_val(newVal));
}

struct TensormapFenceproxyAcquireOpConversion
    : public ConvertOpToLLVMPattern<ttng::TensormapFenceproxyAcquireOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ttng::TensormapFenceproxyAcquireOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    PTXBuilder ptxBuilder;
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // prepare asm operands
    auto *descAddrOpr = ptxBuilder.newAddrOperand(adaptor.getDescPtr(), "l");
    auto *sizeOpr = ptxBuilder.newConstantOperand(TMA_SIZE_BYTES);

    // Define the instruction opcode
    constexpr int kWarpSize = 32;
    Value threadId = getThreadId(rewriter, loc);
    Value pred = b.icmp_slt(threadId, b.i32_val(kWarpSize));
    auto &fence =
        *ptxBuilder.create("fence.proxy.tensormap::generic.acquire.gpu");
    fence(descAddrOpr, sizeOpr).predicate(pred);

    // Workaround for a ptxas bug missing a fence after generic.acquire.gpu.
    // TODO: remove the workaround once ptxas is fixed.
    auto &commit = *ptxBuilder.create("cp.async.bulk.commit_group");
    commit().predicate(pred);
    auto &wait = *ptxBuilder.create("cp.async.bulk.wait_group.read 0");
    wait().predicate(pred);

    ptxBuilder.launch(rewriter, loc, getVoidType());

    // We run the fence on a single warp, then use a barrier to synchronize the
    // rest. This ends up being faster than running the fence on each warp.
    // TODO: Ideally we only emit one barrier after all fences are issued
    b.barrier(triton::gpu::AddrSpace::Local);

    rewriter.eraseOp(op);
    return success();
  }
};

void zero_fill_tma(Location loc, MLIRContext *ctx,
                   ConversionPatternRewriter &rewriter,
                   const NVIDIA::TargetInfo &targetInfo, Value descPtr) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // Write out zeros
  constexpr int kWarpSize = 32;
  Value threadId = getThreadId(rewriter, loc);
  Value pred = b.icmp_slt(threadId, b.i32_val(kWarpSize));

  auto fillVal = b.i32_val(0);
  auto writeAddr =
      b.gep(descPtr.getType(), fillVal.getType(), descPtr, threadId);
  targetInfo.storeShared(rewriter, loc, writeAddr, fillVal, pred);
  LLVM::NVIDIA::createSyncWarp(loc, rewriter);
}

struct TensormapCreateOpConversion
    : public ConvertOpToLLVMPattern<ttng::TensormapCreateOp> {
  const NVIDIA::TargetInfo &targetInfo;

  TensormapCreateOpConversion(LLVMTypeConverter &converter,
                              const NVIDIA::TargetInfo &targetInfo,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(ttng::TensormapCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto ctx = getContext();

    bool needsStrideWorkaround = targetInfo.getPtxVersion() <= 85;
    auto smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);

    zero_fill_tma(loc, ctx, rewriter, targetInfo, smemBase);
    Value isLeader = b.icmp_eq(getThreadId(rewriter, loc), b.i32_val(0));
    auto [previousBlock, updateBlock, continuationBlock] =
        createIfBlock(rewriter, loc, isLeader);
    (void)previousBlock;
    rewriter.setInsertionPointToStart(updateBlock);
    tensormap_replace_generic(loc, ctx, rewriter, "global.address", smemBase,
                              adaptor.getGlobalAddress());
    tensormap_replace_generic(loc, ctx, rewriter, "rank", smemBase,
                              op.getRank() - 1);
    for (int i = 0; i < op.getRank(); ++i) {
      tensormap_replace_generic(loc, ctx, rewriter, "box.dim", smemBase,
                                op.getBoxDim()[i], i);
    }
    for (int i = 0; i < op.getRank(); ++i) {
      tensormap_replace_generic(loc, ctx, rewriter, "global.dim", smemBase,
                                op.getGlobalDim()[i], i);
    }
    for (int i = 0; i + 1 < op.getRank(); ++i) {
      auto strideVal = op.getGlobalStride()[i];
      if (needsStrideWorkaround) {
        // Workaround for a ptxas bug
        strideVal = b.ashr(strideVal, b.i64_val(4));
      }
      tensormap_replace_generic(loc, ctx, rewriter, "global.stride", smemBase,
                                strideVal, i);
    }
    for (int i = 0; i < op.getRank(); ++i) {
      tensormap_replace_generic(loc, ctx, rewriter, "element.stride", smemBase,
                                op.getElementStride()[i], i);
    }
    tensormap_replace_generic(loc, ctx, rewriter, "elemtype", smemBase,
                              op.getElemType());
    tensormap_replace_generic(loc, ctx, rewriter, "interleave.layout", smemBase,
                              op.getInterleaveLayout());
    tensormap_replace_generic(loc, ctx, rewriter, "swizzle.mode", smemBase,
                              op.getSwizzleMode());
    tensormap_replace_generic(loc, ctx, rewriter, "fill.mode", smemBase,
                              op.getFillMode());
    rewriter.setInsertionPointToStart(continuationBlock);
    // Make thread zero's field updates visible to the collective copy.
    LLVM::NVIDIA::createSyncWarp(loc, rewriter);
    tensormap_cp_fenceproxy(loc, ctx, rewriter, adaptor.getDescPtr(), smemBase);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReinterpretTensorDescOpConversion
    : public ConvertOpToLLVMPattern<ReinterpretTensorDescOp> {

  ReinterpretTensorDescOpConversion(LLVMTypeConverter &converter,
                                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit) {}

  LogicalResult
  matchAndRewrite(ttng::ReinterpretTensorDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::AddrSpaceCastOp>(op, resultType,
                                                       adaptor.getRawDesc());
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateTMAToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<TensormapCreateOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<TensormapFenceproxyAcquireOpConversion,
               ReinterpretTensorDescOpConversion>(typeConverter, benefit);
}
