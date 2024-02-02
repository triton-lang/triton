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
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct BarrierOpConversion
    : public ConvertOpToLLVMPattern<mlir::gpu::BarrierOp> {
  using ConvertOpToLLVMPattern<mlir::gpu::BarrierOp>::ConvertOpToLLVMPattern;

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
struct AllocMBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::AllocMBarrierOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::AllocMBarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::AllocMBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, op.getOperation());
    auto resultTy = op.getType();
    auto resultTensorTy = resultTy.dyn_cast<RankedTensorType>();
    Type elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    Type llvmElemTy;
    if (resultTensorTy) {
      llvmElemTy =
          getTypeConverter()->convertType(resultTensorTy.getElementType());
    } else {
      auto resultPtrTy = resultTy.dyn_cast<triton::PointerType>();
      assert(resultPtrTy && "Unknown type for AllocMBarrierOp");
      llvmElemTy =
          getTypeConverter()->convertType(resultPtrTy.getPointeeType());
    }
    smemBase = bitcast(smemBase, elemPtrTy);
    auto threadId = getThreadId(rewriter, loc);
    auto pred = icmp_eq(threadId, i32_val(0));
    int numMBarriers = 1;
    if (resultTensorTy) {
      assert(resultTensorTy.getRank() == 1 &&
             "unexpected rank for AllocMBarrierOp");
      numMBarriers = resultTensorTy.getShape()[0];
    }
    for (int i = 0; i < numMBarriers; ++i) {
      Value smem = smemBase;
      if (i > 0) {
        smem = gep(elemPtrTy, llvmElemTy, smem, i32_val(i));
      }
      rewriter.create<triton::nvgpu::MBarrierInitOp>(loc, smem, pred,
                                                     op.getCount());
    }
    if (resultTensorTy) {
      auto llvmElemTy =
          getTypeConverter()->convertType(resultTensorTy.getElementType());
      auto smemObj = SharedMemoryObject(
          smemBase, llvmElemTy, resultTensorTy.getShape(), {0}, loc, rewriter);
      auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
      rewriter.replaceOp(op, retVal);
    } else {
      rewriter.replaceOp(op, smemBase);
    }
    return success();
  }
};

struct MBarrierArriveOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::MBarrierArriveOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::MBarrierArriveOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::MBarrierArriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto mbarrier = adaptor.getMbarrier();
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
        op, mbarrier, pred, remoteCtaId, type, txCount);
    return success();
  }
};

struct MBarrierWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::MBarrierWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::MBarrierWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::MBarrierWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    rewriter.replaceOpWithNewOp<triton::nvgpu::MBarrierWaitOp>(
        op, adaptor.getMbarrier(), adaptor.getPhase());
    return success();
  }
};

struct ExtractMBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::ExtractMBarrierOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::ExtractMBarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::ExtractMBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto elemTy =
        op.getTensor().getType().cast<RankedTensorType>().getElementType();
    auto tensorStruct = adaptor.getTensor();
    auto index = adaptor.getIndex();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
    auto basePtr =
        extract_val(ptrTy, tensorStruct, rewriter.getDenseI64ArrayAttr(0));
    Value result =
        gep(ptrTy, getTypeConverter()->convertType(elemTy), basePtr, index);
    rewriter.replaceOp(op, result);
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
} // namespace

void mlir::triton::populateBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<BarrierOpConversion>(typeConverter, benefit);
  patterns.add<AllocMBarrierOpConversion>(typeConverter, benefit);
  patterns.add<MBarrierArriveOpConversion>(typeConverter, benefit);
  patterns.add<MBarrierWaitOpConversion>(typeConverter, benefit);
  patterns.add<ExtractMBarrierOpConversion>(typeConverter, benefit);
  patterns.add<NamedBarrierArriveOpConversion>(typeConverter, benefit);
  patterns.add<NamedBarrierWaitOpConversion>(typeConverter, benefit);
  patterns.add<FenceAsyncSharedOpConversion>(typeConverter, benefit);
}
