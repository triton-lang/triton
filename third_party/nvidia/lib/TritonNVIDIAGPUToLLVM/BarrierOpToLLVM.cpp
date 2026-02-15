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
#include "TargetInfo.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace ttg = mlir::triton::gpu;

namespace {
Value getElectWarp0OrThread0(const NVIDIA::TargetInfo &targetInfo,
                             TritonLLVMOpBuilder &b) {
  if (targetInfo.getComputeCapability() >= 90) {
    return LLVM::NVIDIA::createElectPredicateWarp0(b.loc, *b.builder);
  } else {
    auto tid = getThreadId(*b.builder, b.loc);
    return b.icmp_eq(tid, b.i32_val(0));
  }
}

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

struct FenceMBarrierInitReleaseClusterOpConversion
    : public ConvertOpToLLVMPattern<
          triton::nvidia_gpu::FenceMBarrierInitReleaseClusterOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::FenceMBarrierInitReleaseClusterOp>::
      ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::FenceMBarrierInitReleaseClusterOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // Only one thread needs to issue the fence, just like mbarrier.init.
    Value tid = getThreadId(rewriter, loc);
    Value pred = b.icmp_eq(tid, b.i32_val(0));

    PTXBuilder ptxBuilder;
    auto &fence = *ptxBuilder.create("fence.mbarrier_init.release.cluster");
    fence().predicate(pred);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);

    rewriter.eraseOp(op);
    return success();
  }
};

struct InitBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::InitBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  const NVIDIA::TargetInfo *targetInfo;
  InitBarrierOpConversion(LLVMTypeConverter &typeConverter,
                          PatternBenefit benefit,
                          NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::InitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    // We use an elect predicate to tell ptxas that the operation is uniform,
    // which results in better codegen.
    Value pred = getElectWarp0OrThread0(*targetInfo, b);
    ::mlir::triton::PTXBuilder ptxBuilder;
    const std::string ptx = "@$0 mbarrier.init.shared::cta.b64 [$1], " +
                            std::to_string(op.getCount()) + ";";
    auto &barSyncOp = *ptxBuilder.create(ptx);
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
  const NVIDIA::TargetInfo *targetInfo;
  InvalBarrierOpConversion(LLVMTypeConverter &typeConverter,
                           PatternBenefit benefit,
                           NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::InvalBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    // We use an elect predicate to tell ptxas that the operation is uniform,
    // which results in better codegen.
    Value pred = getElectWarp0OrThread0(*targetInfo, b);
    ::mlir::triton::PTXBuilder ptxBuilder;
    const std::string ptx = "@$0 mbarrier.inval.shared::cta.b64 [$1];";
    auto &barSyncOp = *ptxBuilder.create(ptx);
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
    auto barrierTy = op.getAlloc().getType();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(barrierTy.getElementType()), rewriter);
    // If several CTAs cast to the same barrier, that barrier will receive all
    // the bytes from its broadcast group
    auto numCTAs = triton::gpu::lookupNumCTAs(rewriter);
    auto expectedBytes = op.getSize() * (numCTAs / barrierTy.getNumElements());

    auto id = getThreadId(rewriter, loc);
    Value pred = b.icmp_eq(id, b.i32_val(0));
    pred = b.and_(pred, adaptor.getPred());

    auto kBlock = StringAttr::get(op->getContext(), "block");
    auto maskCGABroadcast =
        toLinearLayout(barrierTy).getFreeVariableMasks().lookup(kBlock);
    if (maskCGABroadcast) {
      // If several CTAs cast to the same barrier, as when we do a TMA into a
      // tcgen05.mma 2CTA, we just register the expect in the lead barrier, as
      // it is the only one that will receive the mbarrier signals
      auto ctaId = nvgpu::ClusterCTAIdOp::create(rewriter, loc);
      auto ctaIdInGroup = b.and_(ctaId, b.i32_val(maskCGABroadcast));
      pred = b.and_(pred, b.icmp_eq(ctaIdInGroup, b.i32_val(0)));
    }

    ::mlir::triton::PTXBuilder ptxBuilder;
    const std::string ptx =
        "@$0 mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], " +
        std::to_string(expectedBytes) + ";";
    auto &barSyncOp = *ptxBuilder.create(ptx);
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
    auto barrierTy = op.getAlloc().getType();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getAlloc(),
        typeConverter->convertType(barrierTy.getElementType()), rewriter);
    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto pred = adaptor.getPred();

    auto kBlock = StringAttr::get(ctx, "block");
    auto maskCGABroadcast =
        toLinearLayout(barrierTy).getFreeVariableMasks().lookup(kBlock);
    if (maskCGABroadcast) {
      // If several CTAs cast to the same barrier, as when we do a TMA into a
      // tcgen05.mma 2CTA, we send all the signals to the lead CTA, so even if
      // this barrier is waiting for zero bytes, no one will arrive on it. As
      // such, we predicate it out
      auto ctaId = nvgpu::ClusterCTAIdOp::create(rewriter, loc);
      auto ctaIdInGroup = b.and_(ctaId, b.i32_val(maskCGABroadcast));
      pred = b.and_(pred, b.icmp_eq(ctaIdInGroup, b.i32_val(0)));
    }

    bool predicated = pred && !matchPattern(pred, m_NonZero());
    std::string ptx;
    if (targetInfo->getComputeCapability() < 90) {
      if (!predicated) {
        ptx = R"(
{
	.reg .pred complete;
	waitLoop:
	mbarrier.test_wait.parity.shared::cta.b64 complete, [$0], $1;
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
	mbarrier.test_wait.parity.shared::cta.b64 complete, [$0], $1;
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
	mbarrier.try_wait.parity.shared::cta.b64 complete, [$0], $1;
	@!complete bra.uni waitLoop;
}
)";
      } else {
        ptx = R"(
{
	@!$2 bra.uni skipWait;
	.reg .pred complete;
	waitLoop:
	mbarrier.try_wait.parity.shared::cta.b64 complete, [$0], $1;
	@!complete bra.uni waitLoop;
	skipWait:
}
)";
      }
    }
    ::mlir::triton::PTXBuilder ptxBuilder;
    auto &waitLoop = *ptxBuilder.create(ptx);
    SmallVector<::mlir::triton::PTXBuilder::Operand *, 3> operands = {
        ptxBuilder.newOperand(smemObj.getBase(), "r"),
        ptxBuilder.newOperand(adaptor.getPhase(), "r")};
    if (predicated)
      operands.push_back(ptxBuilder.newOperand(pred, "b"));

    waitLoop(operands, /*onlyAttachMLIRArgs=*/true);
    ptxBuilder.launch(rewriter, loc, void_ty(ctx));
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

    auto arriveOp = *ptxBuilder.create(ptxAsm.str());
    arriveOp(operands, /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(getContext());
    ptxBuilder.launch(rewriter, op.getLoc(), voidTy);

    rewriter.eraseOp(op);
    return success();
  }
};

// CLC (Cluster Launch Control) Ops - Blackwell SM100+
struct CLCTryCancelOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::CLCTryCancelOp> {
  const NVIDIA::TargetInfo *targetInfo;
  CLCTryCancelOpConversion(LLVMTypeConverter &typeConverter,
                           PatternBenefit benefit,
                           NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::CLCTryCancelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (targetInfo->getComputeCapability() < 100) {
      return op.emitError("CLC operations require SM100+ (Blackwell)");
    }

    auto loc = op.getLoc();

    // Use elect predicate - only one thread should issue CLC
    Value pred = LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter);

    auto numCTAs = ttg::lookupNumCTAs(op);
    if (numCTAs > 1) {
      TritonLLVMOpBuilder b(loc, rewriter);
      auto clusterCtaId = targetInfo->getClusterCTAId(rewriter, loc);
      pred = b.and_(pred, b.icmp_eq(clusterCtaId, b.i32_val(0)));
    }

    std::string ptxAsm = "@$2 clusterlaunchcontrol.try_cancel.async.shared::cta"
                         ".mbarrier::complete_tx::bytes";
    if (op.getMulticast())
      ptxAsm += ".multicast::cluster::all";
    ptxAsm += ".b128 [$0], [$1];";

    PTXBuilder ptxBuilder;
    auto &clcOp = *ptxBuilder.create(ptxAsm);
    auto *resultOp = ptxBuilder.newOperand(adaptor.getResult(), "r");
    auto *mbarOp = ptxBuilder.newOperand(adaptor.getMbarrier(), "r");
    auto *predOp = ptxBuilder.newOperand(pred, "b");
    clcOp({resultOp, mbarOp, predOp}, /*onlyAttachMLIRArgs=*/true);

    auto voidTy = void_ty(getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);

    rewriter.eraseOp(op);
    return success();
  }
};

struct CLCLoadResultOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::CLCLoadResultOp> {
  const NVIDIA::TargetInfo *targetInfo;
  CLCLoadResultOpConversion(LLVMTypeConverter &typeConverter,
                            PatternBenefit benefit,
                            NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::CLCLoadResultOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (targetInfo->getComputeCapability() < 100) {
      return op.emitError("CLC operations require SM100+ (Blackwell)");
    }

    auto loc = op.getLoc();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(),
        typeConverter->convertType(op.getSrc().getType().getElementType()),
        rewriter);
    TritonLLVMOpBuilder b(loc, rewriter);
    auto i128Ty = rewriter.getIntegerType(128);
    auto res = b.load(i128Ty, smemObj.getBase());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct CLCIsCanceledOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::CLCIsCanceledOp> {
  const NVIDIA::TargetInfo *targetInfo;
  CLCIsCanceledOpConversion(LLVMTypeConverter &typeConverter,
                            PatternBenefit benefit,
                            NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::CLCIsCanceledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (targetInfo->getComputeCapability() < 100) {
      return op.emitError("CLC operations require SM100+ (Blackwell)");
    }

    auto loc = op.getLoc();
    std::string ptxAsm =
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 $0, $1;";
    PTXBuilder ptxBuilder;
    auto &clcOp = *ptxBuilder.create(ptxAsm);
    auto *resultOp = ptxBuilder.newOperand("=b");
    auto *clcResultOp = ptxBuilder.newOperand(adaptor.getClcResult(), "q");
    clcOp({resultOp, clcResultOp}, /*onlyAttachMLIRArgs=*/true);

    Value result =
        ptxBuilder.launch(rewriter, loc, i1_ty, /*hasSideEffects=*/false);
    rewriter.replaceOp(op, result);

    return success();
  }
};

struct CLCGetProgramIdOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::CLCGetProgramIdOp> {
  const NVIDIA::TargetInfo *targetInfo;
  CLCGetProgramIdOpConversion(LLVMTypeConverter &typeConverter,
                              PatternBenefit benefit,
                              NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::CLCGetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (targetInfo->getComputeCapability() < 100) {
      return op.emitError("CLC operations require SM100+ (Blackwell)");
    }

    auto loc = op.getLoc();

    const char *dimName = [&] {
      switch (op.getDim()) {
      case ProgramIDDim::X:
        return "x";
      case ProgramIDDim::Y:
        return "y";
      case ProgramIDDim::Z:
        return "z";
      }
      llvm::llvm_unreachable_internal("Invalid program id dim");
    }();

    auto ptxAsm = ("clusterlaunchcontrol.query_cancel.get_first_ctaid::" +
                   llvm::Twine(dimName) + ".b32.b128 $0, $1;")
                      .str();

    PTXBuilder ptxBuilder;
    auto &clcOp = *ptxBuilder.create(ptxAsm);
    auto *resultOp = ptxBuilder.newOperand("=r");
    auto *clcResultOp = ptxBuilder.newOperand(adaptor.getClcResult(), "q");
    clcOp({resultOp, clcResultOp}, /*onlyAttachMLIRArgs=*/true);

    Value result =
        ptxBuilder.launch(rewriter, loc, i32_ty, /*hasSideEffects=*/false);

    // Convert ctaid to clusterid, which is the real program id
    // Note that all cluster CTAs are distributed in the X dim
    if (op.getDim() == ProgramIDDim::X) {
      auto numCTAs = ttg::lookupNumCTAs(op);
      if (numCTAs > 1) {
        TritonLLVMOpBuilder b(loc, rewriter);
        result = b.sdiv(result, b.i32_val(numCTAs));
      }
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

void mlir::triton::NVIDIA::populateBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, NVIDIA::TargetInfo &targetInfo) {
  patterns.add<FenceAsyncSharedOpConversion>(typeConverter, benefit);
  patterns.add<FenceMBarrierInitReleaseClusterOpConversion>(typeConverter,
                                                            benefit);
  patterns.add<InitBarrierOpConversion, InvalBarrierOpConversion>(
      typeConverter, benefit, targetInfo);
  patterns.add<WaitBarrierOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<BarrierExpectConversion>(typeConverter, benefit);
  patterns.add<ArriveBarrierOpConversion>(typeConverter, benefit);
  patterns.add<CLCTryCancelOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<CLCLoadResultOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<CLCIsCanceledOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<CLCGetProgramIdOpConversion>(typeConverter, benefit, targetInfo);
}
