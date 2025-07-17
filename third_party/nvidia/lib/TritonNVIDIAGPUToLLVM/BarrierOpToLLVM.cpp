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
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct FenceAsyncSharedOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::FenceAsyncSharedOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::FenceAsyncSharedOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::FenceAsyncSharedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto kind = NVVM::ProxyKind::async_shared;
    auto space = op.getBCluster() ? NVVM::SharedSpace::shared_cluster
                                  : NVVM::SharedSpace::shared_cta;
    auto ctx = rewriter.getContext();
    auto spaceAttr = NVVM::SharedSpaceAttr::get(ctx, space);
    rewriter.replaceOpWithNewOp<NVVM::FenceProxyOp>(op, kind, spaceAttr);
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    auto id = getThreadId(rewriter, loc);
    auto pred = b.icmp_eq(id, b.i32_val(0));
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    auto id = getThreadId(rewriter, loc);
    Value pred = b.icmp_eq(id, b.i32_val(0));
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    auto id = getThreadId(rewriter, loc);
    Value pred = b.icmp_eq(id, b.i32_val(0));
    pred = b.and_(pred, adaptor.getPred());
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
  const NVIDIA::TargetInfo *targetInfo;
  WaitBarrierOpConversion(LLVMTypeConverter &typeConverter,
                          PatternBenefit benefit,
                          NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::WaitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);
    auto loc = op.getLoc();
    bool predicated =
        adaptor.getPred() && !matchPattern(op.getPred(), m_NonZero());
    std::string ptx;
    if (targetInfo->getComputeCapability() < 90) {
      if (!predicated) {
        ptx = R"(
{
	.reg .pred complete;
	waitLoop:
	mbarrier.test_wait.parity.shared.b64 complete, [$0], $1;
	@!complete nanosleep.u32 20;
	@!complete bra.uni waitLoop;
}
)";
      } else {
        ptx = R"(
{
	@!$2 bra.uni skipWait;
	.reg .pred complete;
	waitLoop:
	mbarrier.test_wait.parity.shared.b64 complete, [$0], $1;
	@!complete nanosleep.u32 20;
	@!complete bra.uni waitLoop;
	skipWait:
}
)";
      }
    } else {
      if (!predicated) {
        ptx = R"(
{
	.reg .pred complete;
	waitLoop:
	mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;
	@!complete bra.uni waitLoop;
}
)";
      } else {
        ptx = R"(
{
	@!$2 bra.uni skipWait;
	.reg .pred complete;
	waitLoop:
	mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;
	@!complete bra.uni waitLoop;
	skipWait:
}
)";
      }
    }
    ::mlir::triton::PTXBuilder ptxBuilder;
    auto &waitLoop = *ptxBuilder.create<>(ptx);
    SmallVector<::mlir::triton::PTXBuilder::Operand *, 3> operands = {
        ptxBuilder.newOperand(smemObj.getBase(), "r"),
        ptxBuilder.newOperand(adaptor.getPhase(), "r")};
    if (predicated)
      operands.push_back(ptxBuilder.newOperand(adaptor.getPred(), "b"));

    waitLoop(operands, /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, op->getLoc(), voidTy);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ArriveBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::ArriveBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::ArriveBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Add phase result as needed.
    std::stringstream ptxAsm;
    ptxAsm << "@$0 mbarrier.arrive.shared::cta.b64 _, [$1]";
    if (op.getCount() > 1) {
      ptxAsm << ", " << op.getCount();
    }
    ptxAsm << ";";

    TritonLLVMOpBuilder b(op.getLoc(), rewriter);
    Value id = getThreadId(rewriter, op.getLoc());
    Value pred = b.icmp_eq(id, b.i32_val(0));
    if (op.getPred())
      pred = b.and_(pred, adaptor.getPred());

    PTXBuilder ptxBuilder;
    SmallVector<PTXBuilder::Operand *, 2> operands = {
        ptxBuilder.newOperand(pred, "b"),
        ptxBuilder.newOperand(adaptor.getAlloc(), "r")};

    auto arriveOp = *ptxBuilder.create<>(ptxAsm.str());
    arriveOp(operands, /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(getContext());
    ptxBuilder.launch(rewriter, op.getLoc(), voidTy);

    rewriter.eraseOp(op);
    return success();
  }
};
struct AsyncCompleteOpConversion
    : public ConvertOpToLLVMPattern<triton::nvws::AsyncCompleteOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  void emitBarrier(std::string instrStr, LLVM::SharedMemoryObject smemObj,
                   MLIRContext *ctx, ConversionPatternRewriter &rewriter,
                   Value pred, Location loc) const {
    ::mlir::triton::PTXBuilder ptxBuilder;
    auto &barSyncOp = *ptxBuilder.create<>(instrStr);
    auto &instr = barSyncOp({ptxBuilder.newOperand(smemObj.getBase(), "r")},
                            /*onlyAttachMLIRArgs=*/true);
    if (pred) {
      instr.predicate(pred);
    }
    auto voidTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, voidTy);
  }

  LogicalResult
  matchAndRewrite(triton::nvws::AsyncCompleteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);
    auto loc = op.getLoc();

    Value pred = op.getPred() ? adaptor.getPred() : Value();

    auto trackedOp = op.getAsyncOp();
    if (trackedOp == nvws::AsyncOp::TMALoad) {
      // nop, done by HW
    } else if (trackedOp == nvws::AsyncOp::TC5MMA ||
               trackedOp == nvws::AsyncOp::TMEMCopy) {
      ModuleOp m = op->getParentOfType<ModuleOp>();
      Value t0Pred = LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter);
      TritonLLVMOpBuilder b(op.getLoc(), rewriter);
      if (pred) {
        pred = b.and_(pred, t0Pred);
      } else {
        pred = t0Pred;
      }
      LLVM::NVIDIA::createTcgen05Commit(rewriter, loc, smemObj.getBase(), pred);
    } else if (trackedOp == nvws::AsyncOp::CpAsync) {
      llvm_unreachable("cpasync support NYI");
    } else if (trackedOp == nvws::AsyncOp::NONE) {
      const std::string ptx = "mbarrier.arrive.shared.b64 _, [$0];";
      emitBarrier(ptx, smemObj, op->getContext(), rewriter, pred, loc);
    } else {
      llvm_unreachable("unknown tracked op");
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, NVIDIA::TargetInfo &targetInfo) {
  patterns.add<FenceAsyncSharedOpConversion>(typeConverter, benefit);
  patterns.add<InitBarrierOpConversion, InvalBarrierOpConversion>(typeConverter,
                                                                  benefit);
  patterns.add<WaitBarrierOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<BarrierExpectConversion>(typeConverter, benefit);
  patterns.add<ArriveBarrierOpConversion>(typeConverter, benefit);
  patterns.add<AsyncCompleteOpConversion>(typeConverter, benefit);
}
