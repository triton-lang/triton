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
  auto *inAddrOpr = ptxBuilder.newAddrOperand(inPtr, "r");
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
                               std::string fieldName, Value descPtr,
                               int32_t newVal, bool useSharedMemory = true) {
  PTXBuilder ptxBuilder;
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  // prepare asm operands
  auto *descAddrOpr = ptxBuilder.newAddrOperand(descPtr, useSharedMemory ? "r" : "l");
  auto newValOpr = ptxBuilder.newConstantOperand(newVal);

  // Define the instruction opcode
  auto &replace = ptxBuilder.create("tensormap.replace.tile")
                      ->o(fieldName)
                      .o(useSharedMemory ? "shared::cta" : "global")
                      .o("b1024")
                      .o("b32");

  Value threadId = getThreadId(rewriter, loc);
  Value pred = b.icmp_eq(threadId, b.i32_val(0));
  replace(descAddrOpr, newValOpr).predicate(pred);

  ptxBuilder.launch(rewriter, loc, void_ty(ctx));
}

void tensormap_replace_generic(Location loc, MLIRContext *ctx,
                               ConversionPatternRewriter &rewriter,
                               std::string fieldName, Value descPtr,
                               Value newVal,
                               std::optional<int32_t> ord = std::nullopt,
                               bool useSharedMemory = true) {
  PTXBuilder ptxBuilder;
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  auto newValTy = newVal.getType();
  int width = 0;

  // prepare asm operands
  auto *descAddrOpr = ptxBuilder.newAddrOperand(descPtr, useSharedMemory ? "r" : "l");
  PTXInstr::Operand *ordOpr =
      ord ? ptxBuilder.newConstantOperand(*ord) : nullptr;
  PTXInstr::Operand *newValOpr = nullptr;
  if (mlir::isa<IntegerType>(newValTy)) {
    width = mlir::cast<IntegerType>(newValTy).getWidth();
  } else {
    assert(mlir::isa<mlir::LLVM::LLVMPointerType>(newValTy));
    width = 64;
  }
  const char *constraint = width == 64 ? "l" : "r";
  newValOpr = ptxBuilder.newOperand(newVal, constraint);

  // Define the instruction opcode
  auto &replace = ptxBuilder.create("tensormap.replace.tile")
                      ->o(fieldName)
                      .o(useSharedMemory ? "shared::cta" : "global")
                      .o("b1024")
                      .o("b32", width == 32)
                      .o("b64", width == 64);

  Value threadId = getThreadId(rewriter, loc);
  Value pred = b.icmp_eq(threadId, b.i32_val(0));

  if (ord) {
    replace(descAddrOpr, ordOpr, newValOpr).predicate(pred);
  } else {
    replace(descAddrOpr, newValOpr).predicate(pred);
  }

  ptxBuilder.launch(rewriter, loc, void_ty(ctx));
}

void tensormap_replace_global_address(Location loc, MLIRContext *ctx,
                                      ConversionPatternRewriter &rewriter,
                                      Value descPtr, Value newVal, bool useSharedMemory = true) {
  tensormap_replace_generic(loc, ctx, rewriter, "global_address", descPtr,
                            newVal, std::nullopt, useSharedMemory);
}

void tensormap_replace_rank(Location loc, MLIRContext *ctx,
                            ConversionPatternRewriter &rewriter, Value descPtr,
                            int32_t newVal) {
  tensormap_replace_generic(loc, ctx, rewriter, "rank", descPtr, newVal, true);
}

void tensormap_replace_box_dim(Location loc, MLIRContext *ctx,
                               ConversionPatternRewriter &rewriter,
                               Value descPtr, int32_t ord, Value newVal) {
  tensormap_replace_generic(loc, ctx, rewriter, "box_dim", descPtr, newVal,
                            ord, true);
}

void tensormap_replace_global_dim(Location loc, MLIRContext *ctx,
                                  ConversionPatternRewriter &rewriter,
                                  Value descPtr, int32_t ord, Value newVal, bool useSharedMemory = true) {
  tensormap_replace_generic(loc, ctx, rewriter, "global_dim", descPtr, newVal,
                            ord, useSharedMemory);
}

void tensormap_replace_global_stride(Location loc, MLIRContext *ctx,
                                     ConversionPatternRewriter &rewriter,
                                     Value descPtr, int32_t ord, Value newVal, bool useSharedMemory = true) {
  tensormap_replace_generic(loc, ctx, rewriter, "global_stride", descPtr,
                            newVal, ord, useSharedMemory);
}

void tensormap_replace_element_stride(Location loc, MLIRContext *ctx,
                                      ConversionPatternRewriter &rewriter,
                                      Value descPtr, int32_t ord,
                                      Value newVal) {
  tensormap_replace_generic(loc, ctx, rewriter, "element_stride", descPtr,
                            newVal, ord, true);
}

void tensormap_replace_elemtype(Location loc, MLIRContext *ctx,
                                ConversionPatternRewriter &rewriter,
                                Value descPtr, int32_t newVal) {
  tensormap_replace_generic(loc, ctx, rewriter, "elemtype", descPtr, newVal, true);
}

void tensormap_replace_interleave_layout(Location loc, MLIRContext *ctx,
                                         ConversionPatternRewriter &rewriter,
                                         Value descPtr, int32_t newVal) {
  tensormap_replace_generic(loc, ctx, rewriter, "interleave_layout", descPtr,
                            newVal, true);
}

void tensormap_replace_swizzle_mode(Location loc, MLIRContext *ctx,
                                    ConversionPatternRewriter &rewriter,
                                    Value descPtr, int32_t newVal) {
  tensormap_replace_generic(loc, ctx, rewriter, "swizzle_mode", descPtr,
                            newVal, true);
}

void tensormap_replace_fill_mode(Location loc, MLIRContext *ctx,
                                 ConversionPatternRewriter &rewriter,
                                 Value descPtr, int32_t newVal) {
  tensormap_replace_generic(loc, ctx, rewriter, "fill_mode", descPtr, newVal, true);
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
    b.barrier();

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
    tensormap_replace_global_address(loc, ctx, rewriter, smemBase,
                                     adaptor.getGlobalAddress());
    tensormap_replace_rank(loc, ctx, rewriter, smemBase, op.getRank() - 1);
    for (int i = 0; i < op.getRank(); ++i) {
      tensormap_replace_box_dim(loc, ctx, rewriter, smemBase, i,
                                op.getBoxDim()[i]);
    }
    for (int i = 0; i < op.getRank(); ++i) {
      tensormap_replace_global_dim(loc, ctx, rewriter, smemBase, i,
                                   op.getGlobalDim()[i]);
    }
    for (int i = 0; i + 1 < op.getRank(); ++i) {
      auto strideVal = op.getGlobalStride()[i];
      if (needsStrideWorkaround) {
        // Workaround for a ptxas bug
        strideVal = b.ashr(strideVal, b.i64_val(4));
      }
      tensormap_replace_global_stride(loc, ctx, rewriter, smemBase, i,
                                      strideVal);
    }
    for (int i = 0; i < op.getRank(); ++i) {
      tensormap_replace_element_stride(loc, ctx, rewriter, smemBase, i,
                                       op.getElementStride()[i]);
    }
    tensormap_replace_elemtype(loc, ctx, rewriter, smemBase, op.getElemType());
    tensormap_replace_interleave_layout(loc, ctx, rewriter, smemBase,
                                        op.getInterleaveLayout());
    tensormap_replace_swizzle_mode(loc, ctx, rewriter, smemBase,
                                   op.getSwizzleMode());
    tensormap_replace_fill_mode(loc, ctx, rewriter, smemBase, op.getFillMode());
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

struct GetDescriptorPtrOpConversion
    : public ConvertOpToLLVMPattern<GetDescriptorPtrOp> {

  GetDescriptorPtrOpConversion(LLVMTypeConverter &converter,
                               PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit) {}

  LogicalResult
  matchAndRewrite(ttng::GetDescriptorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::AddrSpaceCastOp>(op, resultType,
                                                       adaptor.getDesc());
    return success();
  }
};

struct TensormapUpdateOpConversion
    : public ConvertOpToLLVMPattern<ttng::TensormapUpdateOp> {
  const NVIDIA::TargetInfo &targetInfo;

  TensormapUpdateOpConversion(LLVMTypeConverter &converter,
                              const NVIDIA::TargetInfo &targetInfo,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(ttng::TensormapUpdateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto ctx = getContext();

    Value descPtr = adaptor.getDescPtr();

    // Detect address space
    auto ptrType = mlir::cast<LLVM::LLVMPointerType>(descPtr.getType());
    auto addrSpace = ptrType.getAddressSpace();

    // For descriptors in global memory, copy to SMEM, update, and copy back
    if (addrSpace != 3) {
      // Get shared memory workspace for updating
      auto smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);

      // Copy descriptor from GMEM to SMEM (128 bytes)
      constexpr int kWarpSize = 32;
      constexpr int kTMASize = 128;
      Value threadId = getThreadId(rewriter, loc);
      Value pred = b.icmp_slt(threadId, b.i32_val(kWarpSize));

      // Cast pointers for 32-bit (4-byte) loads/stores
      auto i32PtrTyGmem = ptr_ty(ctx, 1);
      auto i32PtrTySmem = ptr_ty(ctx, 3);
      Value gmemI32 = b.bitcast(descPtr, i32PtrTyGmem);
      Value smemI32 = b.bitcast(smemBase, i32PtrTySmem);

      // Use threadId clamped to 0 for threads that shouldn't participate
      Value clampedThreadId = b.select(pred, threadId, b.i32_val(0));

      // Each thread copies one i32 (4 bytes)
      Value gmemAddr = b.gep(i32PtrTyGmem, i32_ty, gmemI32, clampedThreadId);
      Value smemAddr = b.gep(i32PtrTySmem, i32_ty, smemI32, clampedThreadId);

      // Load from GMEM and store to SMEM
      Value data = b.load(i32_ty, gmemAddr);
      targetInfo.storeShared(rewriter, loc, smemAddr, data, pred);
      LLVM::NVIDIA::createSyncWarp(loc, rewriter);

      // Perform all updates in SMEM using fast shared::cta
      // instructions
      if (adaptor.getGlobalAddress()) {
        tensormap_replace_global_address(loc, ctx, rewriter, smemBase,
                                         adaptor.getGlobalAddress(), true);
      }
      if (adaptor.getGlobalDim().size() > 0) {
        for (int i = 0; i < adaptor.getGlobalDim().size(); ++i) {
          tensormap_replace_global_dim(loc, ctx, rewriter, smemBase, i,
                                       adaptor.getGlobalDim()[i], true);
        }
      }
      if (adaptor.getGlobalStride().size() > 0) {
        bool needsStrideWorkaround = targetInfo.getPtxVersion() <= 85;
        for (int i = 0; i < adaptor.getGlobalStride().size(); ++i) {
          auto strideVal = adaptor.getGlobalStride()[i];
          if (needsStrideWorkaround) {
            strideVal = b.ashr(strideVal, b.i64_val(4));
          }
          tensormap_replace_global_stride(loc, ctx, rewriter, smemBase, i,
                                          strideVal, true);
        }
      }

      tensormap_cp_fenceproxy(loc, ctx, rewriter, descPtr, smemBase);
    } else {
      // Descriptor already in SMEM, update in place
      if (adaptor.getGlobalAddress()) {
        tensormap_replace_global_address(loc, ctx, rewriter, descPtr,
                                         adaptor.getGlobalAddress(), true);
      }
      if (adaptor.getGlobalDim().size() > 0) {
        for (int i = 0; i < adaptor.getGlobalDim().size(); ++i) {
          tensormap_replace_global_dim(loc, ctx, rewriter, descPtr, i,
                                       adaptor.getGlobalDim()[i], true);
        }
      }
      if (adaptor.getGlobalStride().size() > 0) {
        bool needsStrideWorkaround = targetInfo.getPtxVersion() <= 85;
        for (int i = 0; i < adaptor.getGlobalStride().size(); ++i) {
          auto strideVal = adaptor.getGlobalStride()[i];
          if (needsStrideWorkaround) {
            strideVal = b.ashr(strideVal, b.i64_val(4));
          }
          tensormap_replace_global_stride(loc, ctx, rewriter, descPtr, i,
                                          strideVal, true);
        }
      }

      LLVM::NVIDIA::createSyncWarp(loc, rewriter);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateTMAToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<TensormapFenceproxyAcquireOpConversion,
               ReinterpretTensorDescOpConversion,
               GetDescriptorPtrOpConversion>(typeConverter, benefit);
  patterns.add<TensormapCreateOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<TensormapUpdateOpConversion>(typeConverter, targetInfo, benefit);
}
