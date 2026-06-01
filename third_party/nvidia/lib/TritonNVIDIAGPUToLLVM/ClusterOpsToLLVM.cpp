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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
static void createClusterArrive(OpBuilder &b, Location loc, bool relaxed) {
  auto unitAttr = UnitAttr::get(b.getContext());
  if (relaxed)
    NVVM::ClusterArriveRelaxedOp::create(b, loc, unitAttr);
  else
    NVVM::ClusterArriveOp::create(b, loc, unitAttr);
}

static void createClusterWait(OpBuilder &b, Location loc) {
  NVVM::ClusterWaitOp::create(b, loc, UnitAttr::get(b.getContext()));
}

static void createMBarrierInit(OpBuilder &b, Location loc, Value pred,
                               Value barrierPtr, int count) {
  PTXBuilder ptxBuilder;
  auto &init = *ptxBuilder.create("@$0 mbarrier.init.shared::cta.b64 [$1], " +
                                  std::to_string(count) + ";");
  init({ptxBuilder.newOperand(pred, "b"),
        ptxBuilder.newOperand(barrierPtr, "r")},
       /*onlyAttachMLIRArgs=*/true);
  ptxBuilder.launch(b, loc, void_ty(b.getContext()));
}

static void createMBarrierArrive(OpBuilder &b, Location loc, Value pred,
                                 Value barrierPtr) {
  PTXBuilder ptxBuilder;
  auto &arrive = *ptxBuilder.create(
      "@$0 mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$1];");
  arrive({ptxBuilder.newOperand(pred, "b"),
          ptxBuilder.newOperand(barrierPtr, "r")},
         /*onlyAttachMLIRArgs=*/true);
  ptxBuilder.launch(b, loc, void_ty(b.getContext()));
}

static void createMBarrierWait(OpBuilder &b, Location loc, Value barrierPtr,
                               Value parity) {
  PTXBuilder ptxBuilder;
  auto &wait =
      *ptxBuilder.create("{\n"
                         "\t.reg .pred complete;\n"
                         "waitLoop:\n"
                         "\tmbarrier.try_wait.parity.acquire.cluster.shared::"
                         "cta.b64 complete, [$0], $1;\n"
                         "\t@!complete bra.uni waitLoop;\n"
                         "}\n");
  wait({ptxBuilder.newOperand(barrierPtr, "r"),
        ptxBuilder.newOperand(parity, "r")},
       /*onlyAttachMLIRArgs=*/true);
  ptxBuilder.launch(b, loc, void_ty(b.getContext()));
}

template <typename EmitFn>
LogicalResult lowerClusterSyncForAllWarps(Operation *op,
                                          ConversionPatternRewriter &rewriter,
                                          EmitFn emit) {
  auto loc = op->getLoc();
  auto mod = op->getParentOfType<ModuleOp>();
  if (!mod)
    return rewriter.notifyMatchFailure(op, "expected parent module");

  auto defaultNumWarps = triton::gpu::maybeLookupNumWarps(op);
  if (!defaultNumWarps)
    return rewriter.notifyMatchFailure(op, "missing contextual num-warps");
  int totalNumWarps = *defaultNumWarps;
  if (auto totalNumWarpsAttr =
          mod->getAttrOfType<IntegerAttr>("ttg.total-num-warps"))
    totalNumWarps = totalNumWarpsAttr.getInt();
  int workerNumWarps = totalNumWarps - *defaultNumWarps;
  if (workerNumWarps < 0)
    return rewriter.notifyMatchFailure(op, "invalid total/default num-warps");

  rewriter.setInsertionPoint(op);
  if (workerNumWarps == 0) {
    emit(rewriter);
    rewriter.eraseOp(op);
    return success();
  }

  SmallVector<int32_t> partitionNumWarps;
  for (int remainingWarps = workerNumWarps; remainingWarps > 0;) {
    int32_t partitionWarps =
        llvm::bit_floor(static_cast<uint32_t>(remainingWarps));
    partitionNumWarps.push_back(partitionWarps);
    remainingWarps -= partitionWarps;
  }

  auto wsOp = triton::gpu::WarpSpecializeOp::create(rewriter, loc, TypeRange{},
                                                    partitionNumWarps);
  SmallVector<int32_t> startIds;
  int startId = *defaultNumWarps;
  for (int32_t partitionWarps : partitionNumWarps) {
    startIds.push_back(startId);
    startId += partitionWarps;
  }
  wsOp.setWarpGroupStartIds(startIds);

  Block *defaultBlock = rewriter.createBlock(&wsOp.getDefaultRegion());
  rewriter.setInsertionPointToEnd(defaultBlock);
  emit(rewriter);
  triton::gpu::WarpYieldOp::create(rewriter, loc, TypeRange(), ValueRange());

  Block *partitionHolder = rewriter.createBlock(&wsOp.getPartitionOpHolder());
  rewriter.setInsertionPointToStart(partitionHolder);
  auto partitions = triton::gpu::WarpSpecializePartitionsOp::create(
      rewriter, loc, ValueRange(),
      /*numPartitionRegions=*/partitionNumWarps.size());
  for (Region &partitionRegion : partitions.getPartitionRegions()) {
    Block *partitionBlock = rewriter.createBlock(&partitionRegion);
    rewriter.setInsertionPointToEnd(partitionBlock);
    emit(rewriter);
    triton::gpu::WarpReturnOp::create(rewriter, loc);
  }

  rewriter.eraseOp(op);
  return success();
}

struct ClusterArriveOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::ClusterArriveOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::ClusterArriveOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::ClusterArriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerClusterSyncForAllWarps(op, rewriter, [&](OpBuilder &b) {
      createClusterArrive(b, op.getLoc(), op.getRelaxed());
    });
  }
};

struct ClusterWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::ClusterWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::ClusterWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::ClusterWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerClusterSyncForAllWarps(
        op, rewriter, [&](OpBuilder &b) { createClusterWait(b, op.getLoc()); });
  }
};

struct ClusterBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::ClusterBarrierOp> {
  const NVIDIA::TargetInfo *targetInfo;
  mutable bool emittedMBarrierInitSync = false;

  ClusterBarrierOpConversion(LLVMTypeConverter &typeConverter,
                             PatternBenefit benefit,
                             NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {
    setHasBoundedRewriteRecursion();
  }

  void createWSMBarrierPrologue(triton::nvidia_gpu::ClusterBarrierOp op,
                                ConversionPatternRewriter &rewriter) const {
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    assert(func && "expected cluster_barrier inside a converted function");
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&func.getBody().front());

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value barrierPtr =
        LLVM::getSharedMemoryBase(loc, rewriter, *targetInfo, op);
    auto ptrTy = cast<LLVM::LLVMPointerType>(barrierPtr.getType());
    Value parityPtr = b.gep(ptrTy, i8_ty, barrierPtr, LLVM::GEPArg(8));
    Value tid = getThreadId(rewriter, loc);
    Value pred = b.icmp_eq(tid, b.i32_val(0));
    createMBarrierInit(rewriter, loc, pred, barrierPtr,
                       triton::gpu::lookupNumCTAs(op) - 1);
    targetInfo->storeShared(rewriter, loc, parityPtr, b.i32_val(0), pred);

    if (emittedMBarrierInitSync)
      return;
    emittedMBarrierInitSync = true;
    triton::nvidia_gpu::FenceMBarrierInitReleaseClusterOp::create(rewriter,
                                                                  loc);
    triton::nvidia_gpu::ClusterBarrierOp::create(rewriter, loc,
                                                 /*relaxed=*/true);
  }

  LogicalResult
  lowerWarpSpecialized(triton::nvidia_gpu::ClusterBarrierOp op,
                       ConversionPatternRewriter &rewriter) const {
    if (!op->hasAttr("allocation.offset")) {
      ModuleOp mod = op->getParentOfType<ModuleOp>();
      int64_t shared = mod->getAttrOfType<IntegerAttr>("ttg.shared").getInt();
      int64_t count = 0;
      if (auto attr =
              mod->getAttrOfType<IntegerAttr>("ttg.ws_cluster_barrier_count"))
        count = attr.getInt();
      int64_t offset = llvm::alignTo(shared, int64_t{8}) + count * 16;
      op->setAttr("allocation.offset", rewriter.getI32IntegerAttr(offset));
      mod->setAttr("ttg.ws_cluster_barrier_count",
                   rewriter.getI32IntegerAttr(count + 1));
    }
    createWSMBarrierPrologue(op, rewriter);

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value barrierPtr =
        LLVM::getSharedMemoryBase(loc, rewriter, *targetInfo, op);
    auto ptrTy = cast<LLVM::LLVMPointerType>(barrierPtr.getType());
    Value parityPtr = b.gep(ptrTy, i8_ty, barrierPtr, LLVM::GEPArg(8));

    b.barrier(triton::gpu::AddrSpace::Local);
    Value parity = b.load(i32_ty, parityPtr);
    Value tid = getThreadId(rewriter, loc);
    Value pred = b.icmp_eq(tid, b.i32_val(0));
    Value barrierInt = b.ptrtoint(i32_ty, barrierPtr);
    int numCTAs = triton::gpu::lookupNumCTAs(op);
    for (int i = 1; i < numCTAs; ++i) {
      Value peerBarrierInt = b.xor_(barrierInt, b.i32_val(i << 24));
      Value peerBarrierPtr = b.inttoptr(barrierPtr.getType(), peerBarrierInt);
      createMBarrierArrive(rewriter, loc, pred, peerBarrierPtr);
    }
    createMBarrierWait(rewriter, loc, barrierPtr, parity);
    targetInfo->storeShared(rewriter, loc, parityPtr,
                            b.xor_(parity, b.i32_val(1)), pred);
    b.barrier(triton::gpu::AddrSpace::Local);
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::ClusterBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getParentOfType<triton::gpu::WarpSpecializeOp>())
      return lowerWarpSpecialized(op, rewriter);
    return lowerClusterSyncForAllWarps(op, rewriter, [&](OpBuilder &b) {
      createClusterArrive(b, op.getLoc(), op.getRelaxed());
      createClusterWait(b, op.getLoc());
    });
  }
};
} // namespace

void mlir::triton::NVIDIA::populateClusterOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, NVIDIA::TargetInfo &targetInfo) {
  patterns.add<ClusterArriveOpConversion>(typeConverter, benefit);
  patterns.add<ClusterWaitOpConversion>(typeConverter, benefit);
  patterns.add<ClusterBarrierOpConversion>(typeConverter, benefit, targetInfo);
  return;
}
