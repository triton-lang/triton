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
#include "llvm/Support/MathExtras.h"

#include "Utility.h"
#include <type_traits>

using namespace mlir;
using namespace mlir::triton;

namespace ttg = mlir::triton::gpu;

void mlir::triton::NVIDIA::createFenceMBarrierInitReleaseCluster(
    OpBuilder &builder, Location loc, Value pred) {
  PTXBuilder ptxBuilder;
  auto &fence = *ptxBuilder.create("fence.mbarrier_init.release.cluster");
  fence().predicate(pred);
  ptxBuilder.launch(builder, loc, void_ty(builder.getContext()));
}

namespace {
constexpr int kMaxMbarV1InitCount = (1 << 9) - 1;

bool supportsMbarV1Layout(const NVIDIA::TargetInfo &targetInfo) {
  return targetInfo.getTargetFeatures().supportsMbarV1Layout() &&
         targetInfo.getPtxVersion() >= 93;
}

Value createMbarrierTestWriterPredicate(Operation *op,
                                        ConversionPatternRewriter &rewriter,
                                        const NVIDIA::TargetInfo &targetInfo) {
  if (ttg::lookupNumWarps(op) == 1 && ttg::lookupNumCTAs(op) == 1)
    return {};

  auto loc = op->getLoc();
  auto freeVarMasks = getFreeVariableMasks(op->getResult(0).getType());
  return ttg::emitRedundantThreadPredicate(freeVarMasks, rewriter, loc,
                                           targetInfo);
}

SmallVector<Value> broadcastMbarrierTestResults(
    Operation *op, ArrayRef<Value> results, Value writerPred,
    ConversionPatternRewriter &rewriter, TritonLLVMOpBuilder &b,
    const NVIDIA::TargetInfo &targetInfo) {
  assert(!results.empty() && results.size() <= 8 &&
         llvm::all_of(
             results,
             [](Value result) { return result.getType().isInteger(1); }) &&
         "expected one to eight predicate results");
  if (ttg::lookupNumWarps(op) == 1 && ttg::lookupNumCTAs(op) == 1)
    return SmallVector<Value>(results);

  assert(op->hasAttr("allocation.offset") &&
         "multi-warp or multi-CTA mbarrier test requires shared scratch");
  Value packed = b.zext(i8_ty, results.front());
  for (auto [index, result] : llvm::enumerate(results.drop_front())) {
    Value bit = b.zext(i8_ty, result);
    bit = b.shl(bit, b.i8_val(index + 1));
    packed = b.or_(packed, bit);
  }

  Value scratch =
      LLVM::getSharedMemoryBase(op->getLoc(), rewriter, targetInfo, op);
  targetInfo.storeShared(rewriter, op->getLoc(), scratch, packed, writerPred);
  if (ttg::lookupNumCTAs(op) == 1) {
    targetInfo.barrier(op->getLoc(), rewriter, ttg::AddrSpace::Local);
    packed = targetInfo.loadShared(rewriter, op->getLoc(), scratch, i8_ty,
                                   b.true_val());
  } else {
    targetInfo.clusterBarrier(op->getLoc(), rewriter, op);
    packed = targetInfo.loadDShared(rewriter, op->getLoc(), scratch,
                                    b.i32_val(0), i8_ty, b.true_val());
  }

  SmallVector<Value> broadcast;
  broadcast.reserve(results.size());
  for (unsigned index = 0; index < results.size(); ++index) {
    Value bit = b.and_(packed, b.i8_val(1 << index));
    broadcast.push_back(b.icmp_ne(bit, b.i8_val(0)));
  }
  return broadcast;
}

template <typename OpTy>
struct GridDependencyOpConversion : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PTXBuilder ptxBuilder;
    if constexpr (std::is_same_v<OpTy, triton::GridDependencyWaitOp>)
      (*ptxBuilder.create("griddepcontrol.wait"))();
    else
      (*ptxBuilder.create("griddepcontrol.launch_dependents"))();
    ptxBuilder.launch(rewriter, op.getLoc(), void_ty(rewriter.getContext()));
    rewriter.eraseOp(op);
    return success();
  }
};

Value getElectWarp0OrThread0(const NVIDIA::TargetInfo &targetInfo,
                             TritonLLVMOpBuilder &b) {
  if (targetInfo.getComputeCapability() >= 90) {
    return LLVM::NVIDIA::createElectPredicateWarp0(b.loc, *b.builder);
  } else {
    auto tid = getThreadId(*b.builder, b.loc);
    return b.icmp_eq(tid, b.i32_val(0));
  }
}

struct FromCTALowering {
  Value pred;
  Value barrierPtr;
  Value multicastMask;
};

FromCTALowering getFromCTALowering(Location loc,
                                   ConversionPatternRewriter &rewriter,
                                   Value barrierPtr, uint32_t fromCTA,
                                   bool supportsMBarrierMulticast) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  uint32_t broadcastMask =
      (triton::gpu::lookupNumCTAs(rewriter) - 1) & ~fromCTA;
  Type i32Ty = rewriter.getIntegerType(32);

  Value id = getThreadId(rewriter, loc);
  Value pred =
      supportsMBarrierMulticast
          ? b.icmp_eq(id, b.i32_val(0))
          : b.icmp_ult(id, b.i32_val(1u << llvm::popcount(broadcastMask)));
  Value ctaId = NVVM::ClusterId::create(rewriter, loc, i32Ty);
  Value sourceCTA = b.and_(ctaId, b.i32_val(broadcastMask));
  pred = b.and_(pred, b.icmp_eq(sourceCTA, b.i32_val(0)));

  if (supportsMBarrierMulticast)
    return {pred, barrierPtr,
            LLVM::NVIDIA::createTMAMulticastMask(loc, rewriter, broadcastMask)};

  Value peerOffset = b.i32_val(0);
  unsigned srcBit = 0;
  for (uint32_t mask = broadcastMask; mask; mask &= mask - 1, ++srcBit) {
    unsigned dstBit = llvm::countr_zero(mask);
    Value bit = b.and_(id, b.i32_val(1u << srcBit));
    peerOffset = b.or_(peerOffset, b.shl(bit, b.i32_val(dstBit + 24 - srcBit)));
  }
  Value barrierInt = b.ptrtoint(i32Ty, barrierPtr);
  Value peerBarrierInt = b.xor_(barrierInt, peerOffset);
  Value peerBarrierPtr = b.inttoptr(barrierPtr.getType(), peerBarrierInt);
  return {pred, peerBarrierPtr, {}};
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

    NVIDIA::createFenceMBarrierInitReleaseCluster(rewriter, loc, pred);

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
    auto barrierTy = op.getAlloc().getType();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(barrierTy.getElementType()), rewriter);

    // We use an elect predicate to tell ptxas that the operation is uniform,
    // which results in better codegen.
    Value pred = getElectWarp0OrThread0(*targetInfo, b);

    if (auto leaderPred =
            LLVM::NVIDIA::getLeaderCTAPredicate(loc, rewriter, barrierTy))
      pred = b.and_(pred, *leaderPred);

    auto numCTAs = triton::gpu::lookupNumCTAs(op);
    auto initCount = op.getCount();
    // The lead barrier accounts for all arrives from CTAs that broadcast into
    // the same barrier.
    initCount *= numCTAs / barrierTy.getNumElements();

    ::mlir::triton::PTXBuilder ptxBuilder;
    std::string ptx;
    // The expected arrival count has only 9 bits in the v1 layout. Preserve
    // v0 for barriers whose count cannot be represented by v1.
    // The v1 layout is only needed when the conditonal phase is queried.
    // But for simplicity we always use it when possible.
    if (supportsMbarV1Layout(*targetInfo) && initCount <= kMaxMbarV1InitCount) {
      ptx = "@$0 mbarrier.init.layout::v1.shared::cta.b64 [$1], " +
            std::to_string(initCount) + ";";
    } else {
      ptx = "@$0 mbarrier.init.shared::cta.b64 [$1], " +
            std::to_string(initCount) + ";";
    }
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
    auto barrierTy = op.getAlloc().getType();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(barrierTy.getElementType()), rewriter);

    // We use an elect predicate to tell ptxas that the operation is uniform,
    // which results in better codegen.
    Value pred = getElectWarp0OrThread0(*targetInfo, b);
    if (auto leaderPred =
            LLVM::NVIDIA::getLeaderCTAPredicate(loc, rewriter, barrierTy))
      pred = b.and_(pred, *leaderPred);
    Value barrierPtr = LLVM::NVIDIA::getLeaderAddress(
        loc, rewriter, smemObj.getBase(), barrierTy);
    ::mlir::triton::PTXBuilder ptxBuilder;
    const std::string ptx = "@$0 mbarrier.inval.shared::cta.b64 [$1];";
    auto &barSyncOp = *ptxBuilder.create(ptx);
    barSyncOp({ptxBuilder.newOperand(pred, "b"),
               ptxBuilder.newOperand(barrierPtr, "r")},
              /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);
    rewriter.eraseOp(op);
    return success();
  }
};

struct BarrierExpectConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::BarrierExpectOp> {
  bool supportsMBarrierMulticast;
  BarrierExpectConversion(LLVMTypeConverter &typeConverter,
                          PatternBenefit benefit,
                          bool supportsMBarrierMulticast)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        supportsMBarrierMulticast(supportsMBarrierMulticast) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::BarrierExpectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto barrierTy = op.getAlloc().getType();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(barrierTy.getElementType()), rewriter);
    // Because this operation can signal other partitions we need to synchronize
    // the current partition first.
    ttg::BarrierOp::create(rewriter, loc, ttg::AddrSpace::Local);

    // The partition-relative thread ID lowers the same or marginally better
    // than an elect: LOP3.LUT vs. ELECT + ISETP.EQ.U32.AND.
    Value id = getThreadId(rewriter, loc);
    Value pred = b.icmp_eq(id, b.i32_val(0));
    bool isCrossClusterBarrier =
        LLVM::NVIDIA::getCGABroadcastMask(barrierTy) != 0;
    Value barrierPtr = LLVM::NVIDIA::getLeaderAddress(
        loc, rewriter, smemObj.getBase(), barrierTy);
    Value multicastMask;
    if (std::optional<uint32_t> fromCTA = op.getFromCTA()) {
      FromCTALowering lowering =
          getFromCTALowering(loc, rewriter, smemObj.getBase(), *fromCTA,
                             supportsMBarrierMulticast);
      pred = lowering.pred;
      barrierPtr = lowering.barrierPtr;
      multicastMask = lowering.multicastMask;
      isCrossClusterBarrier = true;
    }
    pred = b.and_(pred, adaptor.getPred());

    ::mlir::triton::PTXBuilder expectPtxBuilder;
    const std::string expectPtx =
        "@$0 mbarrier.arrive.expect_tx." +
        std::string(isCrossClusterBarrier ? "shared::cluster" : "shared::cta") +
        std::string(multicastMask ? ".multicast::cluster::32b" : "") +
        ".b64 _, [$1], " + std::to_string(op.getSize()) +
        std::string(multicastMask ? ", $2" : "") + ";";
    auto &expectOp = *expectPtxBuilder.create(expectPtx);
    SmallVector<PTXBuilder::Operand *, 3> operands = {
        expectPtxBuilder.newOperand(pred, "b"),
        expectPtxBuilder.newOperand(barrierPtr, "r")};
    if (multicastMask)
      operands.push_back(expectPtxBuilder.newOperand(multicastMask, "r"));
    expectOp(operands, /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    expectPtxBuilder.launch(rewriter, loc, voidTy);

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
    if (auto leaderPred =
            LLVM::NVIDIA::getLeaderCTAPredicate(loc, rewriter, barrierTy))
      pred = pred ? b.and_(pred, *leaderPred) : *leaderPred;

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
      std::string phaseType;
      if (supportsMbarV1Layout(*targetInfo) &&
          op.getPhaseType() ==
              triton::nvidia_gpu::MBarrierPhaseType::CONDITIONAL)
        phaseType = ".phase_type::conditional";

      if (!predicated) {
        ptx = R"(
{
	.reg .pred complete;
	waitLoop:
	mbarrier.try_wait.parity)" +
              phaseType + R"(.shared::cta.b64 complete, [$0], $1;
	@!complete bra.uni waitLoop;
}
)";
      } else {
        ptx = R"(
{
	@!$2 bra.uni skipWait;
	.reg .pred complete;
	waitLoop:
	mbarrier.try_wait.parity)" +
              phaseType + R"(.shared::cta.b64 complete, [$0], $1;
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

struct BarrierTestWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::BarrierTestWaitOp> {
  const NVIDIA::TargetInfo *targetInfo;
  BarrierTestWaitOpConversion(LLVMTypeConverter &typeConverter,
                              PatternBenefit benefit,
                              NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::BarrierTestWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool conditional =
        op.getPhaseType() == triton::nvidia_gpu::MBarrierPhaseType::CONDITIONAL;
    if (conditional && !supportsMbarV1Layout(*targetInfo))
      return op.emitError(
          "conditional mbarrier test requires mbarrier v1 layout support");

    auto barrierTy = op.getAlloc().getType();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getAlloc(),
        typeConverter->convertType(barrierTy.getElementType()), rewriter);
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto pred = adaptor.getPred();
    Value writerPred =
        createMbarrierTestWriterPredicate(op, rewriter, *targetInfo);
    if (writerPred)
      pred = pred ? b.and_(pred, writerPred) : writerPred;
    if (auto leaderPred =
            LLVM::NVIDIA::getLeaderCTAPredicate(loc, rewriter, barrierTy))
      pred = pred ? b.and_(pred, *leaderPred) : *leaderPred;
    bool predicated = pred && !matchPattern(pred, m_NonZero());
    std::string phaseType =
        conditional ? ".phase_type::conditional" : std::string();

    ::mlir::triton::PTXBuilder ptxBuilder;
    SmallVector<::mlir::triton::PTXBuilder::Operand *, 4> operands = {
        ptxBuilder.newOperand("=b"),
        ptxBuilder.newOperand(smemObj.getBase(), "r"),
        ptxBuilder.newOperand(adaptor.getPhase(), "r")};

    std::string ptx;
    if (predicated) {
      ptx = R"(
{
	mov.pred $0, 0;
	@!$3 bra.uni skipTest;
	mbarrier.test_wait.parity)" +
            phaseType + R"(.shared::cta.b64 $0, [$1], $2;
	skipTest:
}
)";
      operands.push_back(ptxBuilder.newOperand(pred, "b"));
    } else {
      ptx = R"(
{
	mbarrier.test_wait.parity)" +
            phaseType + R"(.shared::cta.b64 $0, [$1], $2;
}
)";
    }

    auto &test = *ptxBuilder.create(ptx);
    test(operands, /*onlyAttachMLIRArgs=*/true);
    Value complete = ptxBuilder.launch(rewriter, loc, rewriter.getI1Type());
    auto results = broadcastMbarrierTestResults(op, {complete}, writerPred,
                                                rewriter, b, *targetInfo);
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct BarrierTestWaitReportOpConversion
    : public ConvertOpToLLVMPattern<
          triton::nvidia_gpu::BarrierTestWaitReportOp> {
  const NVIDIA::TargetInfo *targetInfo;
  BarrierTestWaitReportOpConversion(LLVMTypeConverter &typeConverter,
                                    PatternBenefit benefit,
                                    NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::BarrierTestWaitReportOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!supportsMbarV1Layout(*targetInfo))
      return op.emitError(
          "primary mbarrier report requires mbarrier v1 layout support");

    auto barrierTy = op.getAlloc().getType();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getAlloc(),
        typeConverter->convertType(barrierTy.getElementType()), rewriter);
    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto pred = adaptor.getPred();
    Value writerPred =
        createMbarrierTestWriterPredicate(op, rewriter, *targetInfo);
    if (writerPred)
      pred = pred ? b.and_(pred, writerPred) : writerPred;
    if (auto leaderPred =
            LLVM::NVIDIA::getLeaderCTAPredicate(loc, rewriter, barrierTy))
      pred = pred ? b.and_(pred, *leaderPred) : *leaderPred;
    bool predicated = pred && !matchPattern(pred, m_NonZero());

    ::mlir::triton::PTXBuilder ptxBuilder;
    SmallVector<::mlir::triton::PTXBuilder::Operand *, 5> operands = {
        ptxBuilder.newOperand("=b"), ptxBuilder.newOperand("=b"),
        ptxBuilder.newOperand(smemObj.getBase(), "r"),
        ptxBuilder.newOperand(adaptor.getPhase(), "r")};

    std::string ptx;
    if (predicated) {
      ptx = R"(
{
	mov.pred $0, 0;
	mov.pred $1, 0;
	@!$4 bra.uni skipTest;
	mbarrier.test_wait.parity.phase_type::primary.shared::cta.b64 $0|$1, [$2], $3;
	skipTest:
}
)";
      operands.push_back(ptxBuilder.newOperand(pred, "b"));
    } else {
      ptx = R"(
{
	mbarrier.test_wait.parity.phase_type::primary.shared::cta.b64 $0|$1, [$2], $3;
}
)";
    }

    auto &test = *ptxBuilder.create(ptx);
    test(operands, /*onlyAttachMLIRArgs=*/true);
    SmallVector<Type> resultTypes(2, rewriter.getI1Type());
    Value packed = ptxBuilder.launch(rewriter, loc, struct_ty(resultTypes));
    SmallVector<Value> results = {
        b.extract_val(rewriter.getI1Type(), packed, 0),
        b.extract_val(rewriter.getI1Type(), packed, 1)};
    results = broadcastMbarrierTestResults(op, results, writerPred, rewriter, b,
                                           *targetInfo);
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct ArriveBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::ArriveBarrierOp> {
  bool supportsMBarrierMulticast;
  ArriveBarrierOpConversion(LLVMTypeConverter &typeConverter,
                            PatternBenefit benefit,
                            bool supportsMBarrierMulticast)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        supportsMBarrierMulticast(supportsMBarrierMulticast) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::ArriveBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto barrierTy = op.getAlloc().getType();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(barrierTy.getElementType()), rewriter);

    // Arrive has block-level semantics, so we must synchronize
    // Technically, this should be MemBar's job but it can include TMEM
    // accesses which doesn't have a MemBar equivalent :/
    ttg::BarrierOp::create(rewriter, loc, ttg::AddrSpace::Local);

    // The partition-relative thread ID lowers the same or marginally better
    // than an elect: LOP3.LUT vs. ELECT + ISETP.EQ.U32.AND.
    Value id = getThreadId(rewriter, loc);
    Value pred = b.icmp_eq(id, b.i32_val(0));

    bool isCrossClusterBarrier =
        op.isMulticast() || LLVM::NVIDIA::getCGABroadcastMask(barrierTy) != 0;
    Value barrierPtr = LLVM::NVIDIA::getLeaderAddress(
        loc, rewriter, smemObj.getBase(), barrierTy);
    Value multicastMask;
    if (std::optional<uint32_t> fromCTA = op.getFromCTA()) {
      FromCTALowering lowering =
          getFromCTALowering(loc, rewriter, smemObj.getBase(), *fromCTA,
                             supportsMBarrierMulticast && !op.isMulticast());
      pred = lowering.pred;
      barrierPtr = lowering.barrierPtr;
      multicastMask = lowering.multicastMask;
      isCrossClusterBarrier = true;
    }
    if (op.getPred())
      pred = b.and_(pred, adaptor.getPred());
    // TODO: Add phase result as needed.
    std::stringstream ptxAsm;
    ptxAsm << "@$0 mbarrier.arrive."
           << (isCrossClusterBarrier ? "shared::cluster" : "shared::cta");
    if (op.isMulticast() || multicastMask)
      ptxAsm << ".multicast::cluster::32b";
    ptxAsm << ".b64 _, [$1]";
    if (op.getCount() > 1) {
      ptxAsm << ", " << op.getCount();
    }
    if (op.isMulticast() || multicastMask)
      ptxAsm << ", $2";
    ptxAsm << ";";

    PTXBuilder ptxBuilder;
    SmallVector<PTXBuilder::Operand *, 3> operands = {
        ptxBuilder.newOperand(pred, "b"),
        ptxBuilder.newOperand(barrierPtr, "r")};
    if (op.isMulticast()) {
      multicastMask = LLVM::NVIDIA::createTMAMulticastMask(
          loc, rewriter, static_cast<uint16_t>(op.getMulticastCTA()));
    }
    if (multicastMask)
      operands.push_back(ptxBuilder.newOperand(multicastMask, "r"));

    auto arriveOp = *ptxBuilder.create(ptxAsm.str());
    arriveOp(operands, /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);

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
    if (numCTAs > 1)
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
  bool supportsMBarrierMulticast =
      targetInfo.getTargetFeatures().supportsMbarMulticast();
  patterns.add<FenceAsyncSharedOpConversion>(typeConverter, benefit);
  patterns.add<
      GridDependencyOpConversion<triton::GridDependencyWaitOp>,
      GridDependencyOpConversion<triton::GridDependencyLaunchDependentsOp>>(
      typeConverter, benefit);
  patterns.add<FenceMBarrierInitReleaseClusterOpConversion>(typeConverter,
                                                            benefit);
  patterns.add<InitBarrierOpConversion, InvalBarrierOpConversion>(
      typeConverter, benefit, targetInfo);
  patterns.add<WaitBarrierOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<BarrierTestWaitOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<BarrierTestWaitReportOpConversion>(typeConverter, benefit,
                                                  targetInfo);
  patterns.add<BarrierExpectConversion>(typeConverter, benefit,
                                        supportsMBarrierMulticast);
  patterns.add<ArriveBarrierOpConversion>(typeConverter, benefit,
                                          supportsMBarrierMulticast);
  patterns.add<CLCTryCancelOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<CLCLoadResultOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<CLCIsCanceledOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<CLCGetProgramIdOpConversion>(typeConverter, benefit, targetInfo);
}
