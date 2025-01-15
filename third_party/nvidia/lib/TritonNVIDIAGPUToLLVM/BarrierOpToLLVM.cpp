/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "PatternTritonGPUOpToLLVM.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct BarrierOpConversion
    : public ConvertOpToLLVMPattern<mlir::gpu::BarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    if (op->hasAttr("bar_id")) {
      // llvm.nvvm.barrier0 doesn't support bar_id and num_threads attributes,
      // so we have to lower it to ptx manually.
      auto barId = op->getAttrOfType<IntegerAttr>("bar_id").getInt();
      auto numThreads = op->getAttrOfType<IntegerAttr>("num_threads").getInt();
      barSync(rewriter, op, barId, numThreads);
      rewriter.eraseOp(op);
      return success();
    }
    // Otherwise we let the default lowering handle it
    return failure();
  }
};

// --------------------------------------------------------------------------
// -- MBarrier related Ops lowering, to be moved to a separate file ---------
// --------------------------------------------------------------------------
struct MBarrierArriveOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::MBarrierArriveOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::MBarrierArriveOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::MBarrierArriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto mbarrier = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getMbarrier(),
        typeConverter->convertType(op.getMbarrier().getType().getElementType()),
        rewriter);

    bool trackAsyncOp = op.getTrackAsyncOp();
    triton::nvgpu::MBarriveType type = triton::nvgpu::MBarriveType::normal;
    uint32_t txCount = op.getTxCount();
    auto remoteCtaId = adaptor.getRemoteCtaId();
    if (trackAsyncOp) {
      type = triton::nvgpu::MBarriveType::cp_async;
    } else if (remoteCtaId) {
      assert(txCount == 0 &&
             "remote arrive of transaction mbarrier is not implemented yet");
      type = triton::nvgpu::MBarriveType::remote;
    } else if (txCount > 0) {
      type = triton::nvgpu::MBarriveType::expect_tx;
    }
    Value pred = adaptor.getPred();
    if (pred == nullptr) {
      pred = int_val(/*width*/ 1, 1);
    }
    rewriter.replaceOpWithNewOp<triton::nvgpu::MBarrierArriveOp>(
        op, mbarrier.getBase(), pred, remoteCtaId, type, txCount);
    return success();
  }
};

struct NamedBarrierArriveOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::NamedBarrierArriveOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::NamedBarrierArriveOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::NamedBarrierArriveOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    rewriter.replaceOpWithNewOp<triton::nvgpu::NamedBarrierArriveOp>(
        op, adaptor.getBar(), adaptor.getNumThreads());
    return success();
  }
};

struct NamedBarrierWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::NamedBarrierWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::NamedBarrierWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::NamedBarrierWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    rewriter.replaceOpWithNewOp<triton::nvgpu::NamedBarrierWaitOp>(
        op, adaptor.getBar(), adaptor.getNumThreads());
    return success();
  }
};

struct FenceAsyncSharedOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::FenceAsyncSharedOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::FenceAsyncSharedOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::FenceAsyncSharedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    rewriter.replaceOpWithNewOp<triton::nvgpu::FenceAsyncSharedOp>(
        op, adaptor.getBCluster());
    return success();
  }
};

struct InitBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::InitBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::InitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    auto asyncTaskIds = getAsyncTaskIds(op);
    int executingThreadId = 0;
    if (!asyncTaskIds.empty()) {
      assert(asyncTaskIds.size() == 1 && "only support single async task");
      auto mod = op->getParentOfType<ModuleOp>();
      int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
      int warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      executingThreadId = asyncTaskIds[0] * numWarps * warpSize;
    }

    auto id = getThreadId(rewriter, loc);
    auto pred = icmp_eq(id, i32_val(executingThreadId));
    ::mlir::triton::PTXBuilder ptxBuilder;
    const std::string ptx = "@$0 mbarrier.init.shared::cta.b64 [$1], " +
                            std::to_string(op.getCount()) + ";";
    auto &barSyncOp = *ptxBuilder.create<>(ptx);
    barSyncOp({ptxBuilder.newOperand(pred, "b"),
               ptxBuilder.newOperand(smemObj.getBase(), "r")},
              /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);
    rewriter.eraseOp(op);
    return success();
  }
};

struct InvalBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::InvalBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::InvalBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    auto asyncTaskIds = getAsyncTaskIds(op);
    int executingThreadId = 0;
    if (!asyncTaskIds.empty()) {
      assert(asyncTaskIds.size() == 1 && "only support single async task");
      auto mod = op->getParentOfType<ModuleOp>();
      int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
      int warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      executingThreadId = asyncTaskIds[0] * numWarps * warpSize;
    }
    auto id = getThreadId(rewriter, loc);
    Value pred = icmp_eq(id, i32_val(executingThreadId));
    ::mlir::triton::PTXBuilder ptxBuilder;
    const std::string ptx = "@$0 mbarrier.inval.shared::cta.b64 [$1];";
    auto &barSyncOp = *ptxBuilder.create<>(ptx);
    barSyncOp({ptxBuilder.newOperand(pred, "b"),
               ptxBuilder.newOperand(smemObj.getBase(), "r")},
              /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);
    rewriter.eraseOp(op);
    return success();
  }
};

struct BarrierExpectConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::BarrierExpectOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::BarrierExpectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    auto asyncTaskIds = getAsyncTaskIds(op);
    int executingThreadId = 0;
    if (!asyncTaskIds.empty()) {
      assert(asyncTaskIds.size() == 1 && "only support single async task");
      auto mod = op->getParentOfType<ModuleOp>();
      int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
      int warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      executingThreadId = asyncTaskIds[0] * numWarps * warpSize;
    }
    auto id = getThreadId(rewriter, loc);
    Value pred = icmp_eq(id, i32_val(executingThreadId));
    pred = and_(pred, adaptor.getPred());
    ::mlir::triton::PTXBuilder ptxBuilder;
    const std::string ptx =
        "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], " +
        std::to_string(op.getSize()) + ";";
    auto &barSyncOp = *ptxBuilder.create<>(ptx);
    barSyncOp({ptxBuilder.newOperand(pred, "b"),
               ptxBuilder.newOperand(smemObj.getBase(), "r")},
              /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);
    rewriter.eraseOp(op);
    return success();
  }
};

struct WaitBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::WaitBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::WaitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);
    auto loc = op.getLoc();
    const std::string ptx =
        "{                                                           \n\t"
        ".reg .pred P1;                                              \n\t"
        "waitLoop:                                                   \n\t"
        "mbarrier.try_wait.parity.shared.b64 P1, [$0], $1;           \n\t"
        "@!P1 bra.uni waitLoop;                                      \n\t"
        "}                                                           \n\t";
    ::mlir::triton::PTXBuilder ptxBuilder;
    auto &waitLoop = *ptxBuilder.create<>(ptx);
    waitLoop({ptxBuilder.newOperand(smemObj.getBase(), "r"),
              ptxBuilder.newOperand(adaptor.getPhase(), "r")},
             /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, op->getLoc(), voidTy);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void mlir::triton::NVIDIA::populateBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<BarrierOpConversion>(typeConverter, benefit);
  patterns.add<MBarrierArriveOpConversion>(typeConverter, benefit);
  patterns.add<NamedBarrierArriveOpConversion>(typeConverter, benefit);
  patterns.add<NamedBarrierWaitOpConversion>(typeConverter, benefit);
  patterns.add<FenceAsyncSharedOpConversion>(typeConverter, benefit);
  patterns.add<InitBarrierOpConversion, InvalBarrierOpConversion>(typeConverter,
                                                                  benefit);
  patterns.add<WaitBarrierOpConversion>(typeConverter, benefit);
  patterns.add<BarrierExpectConversion>(typeConverter, benefit);
}
