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

#include "BarrierOpToLLVM.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

using namespace mlir;
using namespace mlir::triton;
using ::AMD::TritonGPUToLLVMTypeConverter;
using ::AMD::ConvertTritonGPUOpToLLVMPatternBase;
using ::AMD::ConvertTritonGPUOpToLLVMPattern;

namespace {
struct BarrierOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<mlir::gpu::BarrierOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      mlir::gpu::BarrierOp>::ConvertTritonGPUOpToLLVMPattern;

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
// -- MBarrier related Ops lowering, to be moved to a seperate file ---------
// --------------------------------------------------------------------------
struct AllocMBarrierOpConversion : public ConvertTritonGPUOpToLLVMPattern<
                                       triton::nvidia_gpu::AllocMBarrierOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::AllocMBarrierOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::AllocMBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getResult());
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

struct MBarrierArriveOpConversion : public ConvertTritonGPUOpToLLVMPattern<
                                        triton::nvidia_gpu::MBarrierArriveOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::MBarrierArriveOp>::ConvertTritonGPUOpToLLVMPattern;

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

struct MBarrierWaitOpConversion : public ConvertTritonGPUOpToLLVMPattern<
                                      triton::nvidia_gpu::MBarrierWaitOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::MBarrierWaitOp>::ConvertTritonGPUOpToLLVMPattern;

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
    : public ConvertTritonGPUOpToLLVMPattern<
          triton::nvidia_gpu::ExtractMBarrierOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::ExtractMBarrierOp>::ConvertTritonGPUOpToLLVMPattern;

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
    : public ConvertTritonGPUOpToLLVMPattern<
          triton::nvidia_gpu::NamedBarrierArriveOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::NamedBarrierArriveOp>::
      ConvertTritonGPUOpToLLVMPattern;

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
    : public ConvertTritonGPUOpToLLVMPattern<
          triton::nvidia_gpu::NamedBarrierWaitOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::NamedBarrierWaitOp>::ConvertTritonGPUOpToLLVMPattern;

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
    : public ConvertTritonGPUOpToLLVMPattern<
          triton::nvidia_gpu::FenceAsyncSharedOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::FenceAsyncSharedOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::FenceAsyncSharedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    rewriter.replaceOpWithNewOp<triton::nvgpu::FenceAsyncSharedOp>(
        op, adaptor.getBCluster());
    return success();
  }
};
}

namespace AMD{
void populateBarrierOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation, PatternBenefit benefit) {
  patterns.add<BarrierOpConversion>(typeConverter, allocation, benefit);
  patterns.add<AllocMBarrierOpConversion>(typeConverter, allocation, benefit);
  patterns.add<MBarrierArriveOpConversion>(typeConverter, allocation, benefit);
  patterns.add<MBarrierWaitOpConversion>(typeConverter, allocation, benefit);
  patterns.add<ExtractMBarrierOpConversion>(typeConverter, allocation, benefit);
  patterns.add<NamedBarrierArriveOpConversion>(typeConverter, allocation,
                                               benefit);
  patterns.add<NamedBarrierWaitOpConversion>(typeConverter, allocation,
                                             benefit);
  patterns.add<FenceAsyncSharedOpConversion>(typeConverter, allocation,
                                             benefit);
}
}
