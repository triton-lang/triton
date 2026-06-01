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
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"

#include <type_traits>

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
                                 Value barrierPtr, bool relaxed) {
  PTXBuilder ptxBuilder;
  auto &arrive = *ptxBuilder.create(
      "@$0 mbarrier.arrive." + std::string(relaxed ? "relaxed" : "release") +
      ".cluster.shared::cluster.b64 _, [$1];");
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

static Value getWSRegionThread0Predicate(OpBuilder &rewriter, Location loc) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // This runs after GPU and arith conversion, so emit only NVVM/LLVM ops.
  Value tid = NVVM::ThreadIdXOp::create(rewriter, loc, i32_ty);
  int startThreadId =
      getWarpGroupStartThreadId(rewriter.getInsertionBlock()).value_or(0);
  return b.icmp_eq(tid, b.i32_val(startThreadId));
}

template <typename EmitFn>
void lowerClusterSyncForAllWarps(Location loc, OpBuilder &rewriter,
                                 int defaultNumWarps, int totalNumWarps,
                                 EmitFn emit) {
  int workerNumWarps = totalNumWarps - defaultNumWarps;
  if (workerNumWarps == 0) {
    emit(rewriter);
    return;
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
  int startId = defaultNumWarps;
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
}

template <typename Op>
struct ClusterSyncOpConversion : public ConvertOpToLLVMPattern<Op> {
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<ModuleOp>();
    int defaultNumWarps = triton::gpu::lookupNumWarps(op);
    int totalNumWarps = defaultNumWarps;
    if (auto attr =
            mod->template getAttrOfType<IntegerAttr>("ttg.total-num-warps"))
      totalNumWarps = attr.getInt();

    rewriter.setInsertionPoint(op);
    lowerClusterSyncForAllWarps(
        op.getLoc(), rewriter, defaultNumWarps, totalNumWarps,
        [&](OpBuilder &b) {
          if constexpr (!std::is_same_v<Op, triton::nvidia_gpu::ClusterWaitOp>)
            createClusterArrive(b, op.getLoc(), op.getRelaxed());
          if constexpr (!std::is_same_v<Op,
                                        triton::nvidia_gpu::ClusterArriveOp>)
            createClusterWait(b, op.getLoc());
        });
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void mlir::triton::NVIDIA::populateClusterOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ClusterSyncOpConversion<triton::nvidia_gpu::ClusterArriveOp>,
               ClusterSyncOpConversion<triton::nvidia_gpu::ClusterWaitOp>,
               ClusterSyncOpConversion<triton::nvidia_gpu::ClusterBarrierOp>>(
      typeConverter, benefit);
  return;
}

LogicalResult mlir::triton::NVIDIA::lowerWarpSpecializedClusterBarriers(
    ModuleOp mod, const NVIDIA::TargetInfo &targetInfo) {
  Builder builder(mod.getContext());
  LLVM::LLVMFuncOp kernel;
  for (LLVM::LLVMFuncOp func : mod.getOps<LLVM::LLVMFuncOp>()) {
    if (triton::isKernel(func)) {
      kernel = func;
      continue;
    }
    WalkResult result = func.walk([&](triton::nvidia_gpu::ClusterBarrierOp op) {
      op.emitError("cluster_barrier inside a non-inline function is not "
                   "supported");
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted())
      return failure();
  }

  if (!kernel)
    return success();

  SmallVector<triton::nvidia_gpu::ClusterBarrierOp> barriers;
  kernel.walk(
      [&](triton::nvidia_gpu::ClusterBarrierOp op) { barriers.push_back(op); });
  if (barriers.empty())
    return success();
  int64_t shared = mod->getAttrOfType<IntegerAttr>("ttg.shared").getInt();

  DenseMap<unsigned, IntegerAttr> regionOffsets;
  SmallVector<triton::nvidia_gpu::ClusterBarrierOp> initBarriers;
  for (triton::nvidia_gpu::ClusterBarrierOp op : barriers) {
    // Regions executed by the same warp group share a barrier.
    unsigned regionId = getWarpGroupStartWarpId(op->getBlock()).value_or(0);

    auto [offsetIt, inserted] = regionOffsets.try_emplace(regionId);
    if (inserted) {
      int64_t offset = llvm::alignTo(shared, int64_t{8}) +
                       int64_t(regionOffsets.size() - 1) * 16;
      offsetIt->second = builder.getI32IntegerAttr(offset);
      initBarriers.push_back(op);
    }
    op->setAttr("allocation.offset", offsetIt->second);
  }
  mod->setAttr("ttg.ws_cluster_barrier_count",
               builder.getI32IntegerAttr(regionOffsets.size()));

  auto loc = initBarriers.front().getLoc();
  TritonLLVMIRRewriter rewriter(loc, mod.getContext());
  rewriter.setInsertionPointToStart(&kernel.getBody().front());
  Value initPred = getWSRegionThread0Predicate(rewriter, loc);
  for (triton::nvidia_gpu::ClusterBarrierOp op : initBarriers) {
    loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value barrierPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
    auto ptrTy = cast<LLVM::LLVMPointerType>(barrierPtr.getType());
    Value parityPtr = b.gep(ptrTy, i8_ty, barrierPtr, LLVM::GEPArg(8));
    createMBarrierInit(rewriter, loc, initPred, barrierPtr,
                       triton::gpu::lookupNumCTAs(op) - 1);
    targetInfo.storeShared(rewriter, loc, parityPtr, b.i32_val(0), initPred);
  }
  createFenceMBarrierInitReleaseCluster(rewriter, loc, initPred);
  int defaultNumWarps = triton::gpu::lookupNumWarps(kernel);
  int totalNumWarps = defaultNumWarps;
  if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.total-num-warps"))
    totalNumWarps = attr.getInt();
  lowerClusterSyncForAllWarps(loc, rewriter, defaultNumWarps, totalNumWarps,
                              [&](OpBuilder &b) {
                                createClusterArrive(b, loc, /*relaxed=*/true);
                                createClusterWait(b, loc);
                              });

  for (triton::nvidia_gpu::ClusterBarrierOp op : barriers) {
    loc = op.getLoc();
    TritonLLVMIRRewriter rewriter(loc, op);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value barrierPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
    auto ptrTy = cast<LLVM::LLVMPointerType>(barrierPtr.getType());
    Value parityPtr = b.gep(ptrTy, i8_ty, barrierPtr, LLVM::GEPArg(8));

    NVVM::Barrier0Op::create(rewriter, loc);
    Value parity = b.load(i32_ty, parityPtr);
    Value pred = getWSRegionThread0Predicate(rewriter, loc);
    Value barrierInt = b.ptrtoint(i32_ty, barrierPtr);
    int numCTAs = triton::gpu::lookupNumCTAs(op);
    bool relaxed = op.getRelaxed() && targetInfo.getPtxVersion() >= 86;
    for (int i = 1; i < numCTAs; ++i) {
      Value peerBarrierInt = b.xor_(barrierInt, b.i32_val(i << 24));
      Value peerBarrierPtr = b.inttoptr(barrierPtr.getType(), peerBarrierInt);
      createMBarrierArrive(rewriter, loc, pred, peerBarrierPtr, relaxed);
    }
    createMBarrierWait(rewriter, loc, barrierPtr, parity);
    targetInfo.storeShared(rewriter, loc, parityPtr,
                           b.xor_(parity, b.i32_val(1)), pred);
    NVVM::Barrier0Op::create(rewriter, loc);
    rewriter.eraseOp(op);
  }
  return success();
}
