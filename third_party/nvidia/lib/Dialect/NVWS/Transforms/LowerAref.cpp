/*
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
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

#include "Utilities.h"
#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::nvws;

#define DEBUG_TYPE "nvws-lower-aref"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSLOWERAREF
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {

// ----------------------------------------------------------------------------

struct PartitionWsTagIds {
  std::optional<int> wsTag;
  SetVector<int> partitionIds;
};
std::optional<PartitionWsTagIds> getPartitionWsTagIds(Operation *op) {
  std::optional<PartitionWsTagIds> partitionWsTagIds;
  if (hasPartition(op)) {
    partitionWsTagIds =
        PartitionWsTagIds{std::nullopt, triton::gpu::getPartitionIds(op)};
    if (auto wsTag = getWarpSpecializeTag(op)) {
      partitionWsTagIds->wsTag = *wsTag;
    }
  }
  return partitionWsTagIds;
}

using PartitionSet = SetVector<int>;
void assignStageCluster(Operation *op,
                        std::optional<PartitionWsTagIds> partitionWsTagIds,
                        StageCluster stageCluster, OpBuilder &builder) {
  if (partitionWsTagIds) {
    setPartition(op, partitionWsTagIds->partitionIds);
    if (auto wsTag = partitionWsTagIds->wsTag) {
      setWarpSpecializeTag(op, *wsTag);
    }
    setStageCluster(builder, op, stageCluster);
  }
}

bool isOperandPipelineable(Value v, scf::ForOp forOp) {
  auto isPipelineable = [](Operation *op) {
    return isa<ArefPutEnterOp, ArefGetEnterOp, ArefBufferOp>(op);
  };

  Operation *foundDef = nullptr;
  return triton::nvidia_gpu::isOperandPipelineableBase(v, forOp, foundDef,
                                                       isPipelineable);
}

void setIsAsync(triton::nvidia_gpu::MMAv5OpInterface mmaOp,
                unsigned defaultNumStages) {
  bool isAsync = true;
  auto forOp = mmaOp->getParentOfType<scf::ForOp>();
  if (!forOp)
    return;

  unsigned numStages = getNumStagesOrDefault(forOp, defaultNumStages);
  if (numStages <= 1)
    return;

  if (auto scaledOp = dyn_cast<triton::nvidia_gpu::TCGen5MMAScaledOp>(
          mmaOp.getOperation())) {
    if (!triton::nvidia_gpu::areScalesPipelineable(scaledOp, forOp)) {
      isAsync = false;
    }
    if (!isOperandPipelineable(scaledOp.getAScale(), forOp) ||
        !isOperandPipelineable(scaledOp.getBScale(), forOp)) {
      isAsync = false;
    }
  }
  mmaOp.setIsAsync(isAsync);
}

struct ArefValue {
  Value emptyMbars;
  Value fullMbars;
  Value tmaFullMbars;
  bool tmaFullMbarCoversFullBarrier;
  SmallVector<bool> twoCTATMABuffers;
  int depth;
  SmallVector<Value> buffers;
};

Value getEmptyBarrier(PatternRewriter &rewriter, Location loc, ArefValue aref,
                      Value stage,
                      std::optional<PartitionWsTagIds> partitionWsTagIds,
                      StageCluster stageCluster) {
  auto barrier = createSingleBufferView(rewriter, aref.emptyMbars, stage);
  assignStageCluster(barrier.getDefiningOp(), partitionWsTagIds, stageCluster,
                     rewriter);
  return barrier;
}

Value getFullBarrier(PatternRewriter &rewriter, Location loc, ArefValue aref,
                     Value stage,
                     std::optional<PartitionWsTagIds> partitionWsTagIds,
                     StageCluster stageCluster) {
  auto barrier = createSingleBufferView(rewriter, aref.fullMbars, stage);
  assignStageCluster(barrier.getDefiningOp(), partitionWsTagIds, stageCluster,
                     rewriter);
  return barrier;
}

Value getTMAFullBarrier(PatternRewriter &rewriter, Location loc, ArefValue aref,
                        Value stage,
                        std::optional<PartitionWsTagIds> partitionWsTagIds,
                        StageCluster stageCluster) {
  if (!aref.tmaFullMbars)
    return {};
  auto barrier =
      createSingleBufferView(rewriter, aref.tmaFullMbars, stage);
  assignStageCluster(barrier.getDefiningOp(), partitionWsTagIds, stageCluster,
                     rewriter);
  return barrier;
}

struct BarrierCount {
  int producerPendingCount{0};
  int consumerPendingCount{0};
};

SmallVector<AsyncOp> castAsyncOpAttrs(ArrayAttr opAttrs) {
  SmallVector<AsyncOp> kinds;
  for (auto asyncKind : opAttrs) {
    kinds.push_back(cast<AsyncOpAttr>(asyncKind).getValue());
  }
  return kinds;
}

BarrierCount getArrivalCount(ArefCreateOp op,
                             const DenseSet<MMAv5OpInterface> &mmav5Ops) {
  SetVector<int> producerGroups, consumerGroups;
  BarrierCount count;

  for (auto user : op->getUsers()) {
    if (!hasPartition(user))
      continue;
    auto partitionIds = getPartitionIds(user);

    assert(partitionIds.size() == 1);

    if (auto putExitOp = dyn_cast<ArefPutExitOp>(user)) {
      if (producerGroups.count(partitionIds.front())) {
        continue;
      }
      producerGroups.insert(partitionIds.front());
      for (auto kind : castAsyncOpAttrs(putExitOp.getAsyncOps())) {
        switch (kind) {
        case AsyncOp::TC5MMA:
        case AsyncOp::TMALoad:
        case AsyncOp::NONE:
          count.producerPendingCount += 1;
          break;
        default:
          llvm_unreachable("unsupported producer kind");
        }
      }
    } else if (auto getExitOp = dyn_cast<ArefGetExitOp>(user)) {
      if (consumerGroups.count(partitionIds.front())) {
        continue;
      }
      consumerGroups.insert(partitionIds.front());
      for (auto kind : castAsyncOpAttrs(getExitOp.getAsyncOps())) {
        switch (kind) {
        case AsyncOp::TC5MMA: {
          int mmaCount = 1;
          for (MMAv5OpInterface mma : mmav5Ops) {
            mmaCount = std::max(
                mmaCount, nvidia_gpu::getMMAv5CompletionBarrierCount(mma));
          }
          count.consumerPendingCount += mmaCount;
          break;
        }
        case AsyncOp::WGMMA:
        case AsyncOp::NONE:
          count.consumerPendingCount += 1;
          break;
        default:
          llvm_unreachable("unsupported consumer kind");
        }
      }
    }
  }
  // If the aref is not used within a warp-specialized loop, the pending counts
  // will be equal 0. Set them to 1.
  if (count.consumerPendingCount == 0)
    count.consumerPendingCount = 1;
  if (count.producerPendingCount == 0)
    count.producerPendingCount = 1;

  return count;
}

Value createArefBarrierAlloc(ImplicitLocOpBuilder &builder, Type type,
                             unsigned numBarriers, bool twoCTAs) {
  if (!twoCTAs)
    return createScalarAlloc(builder, type, numBarriers);

  MLIRContext *ctx = builder.getContext();
  unsigned numCTAs = TritonGPUDialect::getNumCTAs(
      builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>());
  assert(numCTAs % 2 == 0);
  Attribute sharedMemorySpace = SharedMemorySpaceAttr::get(ctx);
  auto kBlock = StringAttr::get(ctx, "block");
  auto dim = standardOutDimNames(ctx, /*rank=*/1)[0];
  auto barrierCGALayout = CGAEncodingAttr::get(
      ctx, LinearLayout::zeros1D(2, kBlock, dim) *
               LinearLayout::identity1D(numCTAs / 2, kBlock, dim));
  auto barrierEncoding =
      SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, barrierCGALayout);
  auto memDescType =
      MemDescType::get({numBarriers, numCTAs / 2}, type, barrierEncoding,
                       sharedMemorySpace, /*mutableMemory=*/true);
  return LocalAllocOp::create(builder, memDescType, Value());
}

Value createBarriers(ImplicitLocOpBuilder &b1, ImplicitLocOpBuilder &b2,
                     int numBarriers, int arrivalCount, bool twoCTAs) {
  Value barrierAlloc =
      createArefBarrierAlloc(b1, b1.getI64Type(), numBarriers, twoCTAs);
  for (unsigned i = 0; i < numBarriers; i++) {
    Value barrierView = createSingleBufferView(b1, barrierAlloc, i);
    InitBarrierOp::create(b1, barrierView, arrivalCount);
  }
  // Invalidate and deallocate the barriers.
  for (unsigned i = 0; i < numBarriers; i++) {
    Value barrierView = createSingleBufferView(b2, barrierAlloc, i);
    InvalBarrierOp::create(b2, barrierView);
  }
  LocalDeallocOp::create(b2, barrierAlloc);
  return barrierAlloc;
}

bool arefBufferFeedsTwoCTAMMA(Value aref, unsigned bufferIdx) {
  for (auto user : aref.getUsers()) {
    auto getEnter = dyn_cast<ArefGetEnterOp>(user);
    if (!getEnter || bufferIdx >= getEnter.getBuffers().size())
      continue;
    if (valueFeedsTwoCTAMMA(getEnter.getBuffers()[bufferIdx]))
      return true;
  }
  return false;
}

struct TMALoadTwoCTAInfo {
  bool hasTwoCTA{false};
  bool allUseTwoCTA{false};
};

TMALoadTwoCTAInfo getTMALoadTwoCTAInfo(ArefCreateOp op,
                                       ArrayRef<bool> twoCTATMABuffers) {
  bool sawTMALoad = false;
  bool hasTwoCTA = false;
  bool allUseTwoCTA = true;
  for (auto user : op.getResult().getUsers()) {
    auto putEnter = dyn_cast<ArefPutEnterOp>(user);
    if (!putEnter)
      continue;
    for (auto [idx, buffer] : llvm::enumerate(putEnter.getBuffers())) {
      bool hasTMALoad = llvm::any_of(buffer.getUsers(), [](Operation *user) {
        return isa<triton::nvws::DescriptorLoadOpInterface>(user);
      });
      if (!hasTMALoad)
        continue;
      sawTMALoad = true;
      bool usesTwoCTA = idx < twoCTATMABuffers.size() && twoCTATMABuffers[idx];
      hasTwoCTA |= usesTwoCTA;
      allUseTwoCTA &= usesTwoCTA;
    }
  }
  return {hasTwoCTA, sawTMALoad && allUseTwoCTA};
}

ArefValue createAndInitMbar(ArefCreateOp op, PatternRewriter &rewriter,
                            const DenseSet<MMAv5OpInterface> &mmav5Ops) {
  BarrierCount count = getArrivalCount(op, mmav5Ops);

  auto arefTy = op.getType();
  auto arefBufTypes = llvm::to_vector(llvm::map_range(
      arefTy.getBaseType(), [](Type type) { return cast<MemDescType>(type); }));
  auto depth = getArefDepth(arefBufTypes[0]);
  SmallVector<bool> twoCTATMABuffers;
  for (unsigned i = 0; i < arefBufTypes.size(); ++i)
    twoCTATMABuffers.push_back(arefBufferFeedsTwoCTAMMA(op.getResult(), i));
  TMALoadTwoCTAInfo tmaInfo = getTMALoadTwoCTAInfo(op, twoCTATMABuffers);
  bool hasTwoCTATMA = tmaInfo.hasTwoCTA;
  bool tmaFullMbarCoversFullBarrier = tmaInfo.allUseTwoCTA;

  SetVector<Operation *> arefUsers;
  for (auto user : op->getUsers())
    arefUsers.insert(user);
  auto sorted = topologicalSort(arefUsers);

  ImplicitLocOpBuilder b1(op->getLoc(), op), b2(op->getLoc(), op);
  auto op1 = op->getBlock()->findAncestorOpInBlock(*sorted.back());
  b2.setInsertionPointAfter(op1);

  auto emptyMbars = createBarriers(b1, b2, depth, count.consumerPendingCount,
                                   /*twoCTAs=*/false);
  Value fullMbars;
  if (!tmaFullMbarCoversFullBarrier) {
    fullMbars = createBarriers(b1, b2, depth, count.producerPendingCount,
                               /*twoCTAs=*/false);
  }
  Value tmaFullMbars;
  if (hasTwoCTATMA) {
    // Full barriers preserve the aref producer/consumer protocol. The extra
    // TMA complete_tx barrier is grouped per producer stage, matching the
    // non-WS TMA wait group while keeping the two-CTA barrier layout local to
    // this aref.
    tmaFullMbars = createBarriers(b1, b2, depth, /*arrivalCount=*/1,
                                  /*twoCTAs=*/true);
  }

  return ArefValue{emptyMbars,
                   fullMbars,
                   tmaFullMbars,
                   tmaFullMbarCoversFullBarrier,
                   std::move(twoCTATMABuffers),
                   static_cast<int>(depth),
                   op.getOperands()};
}

SmallVector<Value>
getSubViews(ArefValue arefVal, Value stage, Location loc, OpBuilder &rewriter,
            std::optional<PartitionWsTagIds> partitionWsTagIds,
            StageCluster stageCluster) {
  SmallVector<Value> views;
  for (auto buffer : arefVal.buffers) {
    auto memDescType = cast<MemDescType>(buffer.getType());
    if (isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(
            memDescType.getEncoding())) {
      // tmem scales encoding doesn't support multi-buffering, use buffer as-is
      views.push_back(buffer);
    } else {
      auto shape = memDescType.getShape();
      SmallVector<int64_t> tensorShape(shape.begin() + 1, shape.end());
      auto memDescTypeNew = MemDescType::get(
          tensorShape, memDescType.getElementType(), memDescType.getEncoding(),
          memDescType.getMemorySpace(), true);
      auto singleBuffer =
          MemDescIndexOp::create(rewriter, loc, memDescTypeNew, buffer, stage);
      assignStageCluster(singleBuffer, partitionWsTagIds, stageCluster,
                         rewriter);
      views.push_back(singleBuffer);
    }
  }

  return views;
}

Operation *createTMALoad(triton::nvws::DescriptorLoadOp op,
                         PatternRewriter &rewriter, Value barrierAlloc,
                         Value pred, bool useTwoCTATMA) {
  auto resultType = cast<MemDescType>(op.getResult().getType());
  bool useMulticast = useTwoCTATMA && hasCGABroadcast(resultType);
  auto newLoadOp = triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp::create(
      rewriter, op.getLoc(), op.getDesc(), op.getIndices(), barrierAlloc,
      op.getResult(), pred, useMulticast);
  assignStageCluster(newLoadOp, getPartitionWsTagIds(op), getStageCluster(op),
                     rewriter);
  return newLoadOp;
};

Operation *createTMAGather(triton::nvws::DescriptorGatherOp op,
                           PatternRewriter &rewriter, Value barrierAlloc,
                           Value pred, bool useTwoCTATMA) {
  auto resultType = cast<MemDescType>(op.getResult().getType());
  bool useMulticast = useTwoCTATMA && hasCGABroadcast(resultType);
  auto newGatherOp = triton::nvidia_gpu::AsyncTMAGatherOp::create(
      rewriter, op.getLoc(), op.getDesc(), op.getXOffsets(), op.getYOffset(),
      barrierAlloc, op.getResult(), pred, useMulticast);
  assignStageCluster(newGatherOp, getPartitionWsTagIds(op), getStageCluster(op),
                     rewriter);
  return newGatherOp;
}

void lowerTMALoad(ArefPutEnterOp op, Value fullBarrier,
                  PatternRewriter &rewriter, ArefValue arefVal) {
  auto loc = op.getLoc();
  // for now handle TMA loads in PutEnterOp
  SmallVector<std::pair<Operation *, unsigned>> loadOps;
  for (auto [idx, buffer] : llvm::enumerate(op.getBuffers())) {
    for (auto user : buffer.getUsers()) {
      if (auto loadOp =
              dyn_cast<triton::nvws::DescriptorLoadOpInterface>(user)) {
        loadOps.push_back({loadOp, idx});
      }
    }
  }
  assert(loadOps.size() <= op.getBuffers().size());
  if (loadOps.empty())
    return;

  auto pred = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
  assignStageCluster(pred, getPartitionWsTagIds(op), getStageCluster(op),
                     rewriter);

  Operation *firstTwoCTALoad = nullptr;
  int twoCTATxCount = 0;
  for (auto [loadOp, bufferIdx] : loadOps) {
    bool useTwoCTATMA = arefVal.tmaFullMbars &&
                        bufferIdx < arefVal.twoCTATMABuffers.size() &&
                        arefVal.twoCTATMABuffers[bufferIdx];
    if (!useTwoCTATMA)
      continue;
    if (!firstTwoCTALoad)
      firstTwoCTALoad = loadOp;
    twoCTATxCount +=
        cast<triton::nvws::DescriptorLoadOpInterface>(loadOp).getTxCount();
  }

  Value twoCTATMAFullBarrier;
  if (firstTwoCTALoad) {
    rewriter.setInsertionPoint(firstTwoCTALoad);
    twoCTATMAFullBarrier =
        getTMAFullBarrier(rewriter, loc, arefVal, op.getStage(),
                          getPartitionWsTagIds(op), getStageCluster(op));
    auto expectOp = triton::nvidia_gpu::BarrierExpectOp::create(
        rewriter, loc, twoCTATMAFullBarrier, twoCTATxCount, pred);
    assignStageCluster(expectOp, getPartitionWsTagIds(op), getStageCluster(op),
                       rewriter);
  }

  Operation *lastTMAOp = nullptr;
  for (auto [loadOp, bufferIdx] : loadOps) {
    rewriter.setInsertionPoint(loadOp);
    bool useTwoCTATMA = arefVal.tmaFullMbars &&
                        bufferIdx < arefVal.twoCTATMABuffers.size() &&
                        arefVal.twoCTATMABuffers[bufferIdx];
    assert((useTwoCTATMA || fullBarrier) &&
           "non-two-CTA TMA loads require the normal full barrier");
    Value tmaBarrier = useTwoCTATMA ? twoCTATMAFullBarrier : fullBarrier;
    int txCount =
        cast<triton::nvws::DescriptorLoadOpInterface>(loadOp).getTxCount();
    if (!useTwoCTATMA) {
      auto expectOp = triton::nvidia_gpu::BarrierExpectOp::create(
          rewriter, loc, tmaBarrier, txCount, pred);
      assignStageCluster(expectOp, getPartitionWsTagIds(op),
                         getStageCluster(op), rewriter);
    }

    if (auto descLoad = dyn_cast<triton::nvws::DescriptorLoadOp>(loadOp)) {
      lastTMAOp =
          createTMALoad(descLoad, rewriter, tmaBarrier, pred, useTwoCTATMA);
    } else if (auto descGather =
                   dyn_cast<triton::nvws::DescriptorGatherOp>(loadOp)) {
      lastTMAOp =
          createTMAGather(descGather, rewriter, tmaBarrier, pred, useTwoCTATMA);
    } else {
      llvm_unreachable("Unknown load op");
    }
    loadOp->erase();
  }

  if (twoCTATMAFullBarrier && !arefVal.tmaFullMbarCoversFullBarrier) {
    assert(lastTMAOp && "expected at least one TMA op");
    rewriter.setInsertionPointAfter(lastTMAOp);
    auto arriveOp =
        nvidia_gpu::ArriveBarrierOp::create(rewriter, loc, fullBarrier, 1);
    assignStageCluster(arriveOp, getPartitionWsTagIds(op), getStageCluster(op),
                       rewriter);
  }
}

void insertWaitOp(PatternRewriter &rewriter, Operation *op, Value barrier,
                  Value phase, Value stage, Value pred = {}) {
  auto waitOp = pred ? WaitBarrierOp::create(rewriter, op->getLoc(), barrier,
                                             phase, pred)
                     : WaitBarrierOp::create(rewriter, op->getLoc(), barrier,
                                             phase);
  assignStageCluster(waitOp, getPartitionWsTagIds(op), getStageCluster(op),
                     rewriter);
}

Value createAfterProloguePredicate(PatternRewriter &rewriter, Operation *op,
                                   int depth) {
  if (depth <= 0)
    return {};
  auto forOp = op->getParentOfType<scf::ForOp>();
  if (!forOp)
    return {};

  Location loc = op->getLoc();
  Type ivType = forOp.getInductionVar().getType();
  Value depthVal;
  if (ivType.isIndex()) {
    depthVal = arith::ConstantIndexOp::create(rewriter, loc, depth);
  } else {
    auto intType = cast<IntegerType>(ivType);
    depthVal = arith::ConstantIntOp::create(rewriter, loc, depth,
                                            intType.getWidth());
  }
  assignStageCluster(depthVal.getDefiningOp(), getPartitionWsTagIds(op),
                     getStageCluster(op), rewriter);
  auto prologueIters =
      arith::MulIOp::create(rewriter, loc, forOp.getStep(), depthVal);
  assignStageCluster(prologueIters, getPartitionWsTagIds(op),
                     getStageCluster(op), rewriter);
  auto firstReuseIter =
      arith::AddIOp::create(rewriter, loc, forOp.getLowerBound(), prologueIters);
  assignStageCluster(firstReuseIter, getPartitionWsTagIds(op),
                     getStageCluster(op), rewriter);
  auto pred = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge,
                                    forOp.getInductionVar(), firstReuseIter);
  assignStageCluster(pred, getPartitionWsTagIds(op), getStageCluster(op),
                     rewriter);
  return pred;
}

void rewritePutEnterOp(ArefPutEnterOp op, PatternRewriter &rewriter,
                       ArefValue arefVal,
                       const DenseSet<MMAv5OpInterface> &mmav5Ops,
                       unsigned defaultNumStages) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);

  // get empty barrier at a given stage
  Value emptyBarrier =
      getEmptyBarrier(rewriter, loc, arefVal, op.getStage(),
                      getPartitionWsTagIds(op), getStageCluster(op));
  auto views = getSubViews(arefVal, op.getStage(), loc, rewriter,
                           getPartitionWsTagIds(op), getStageCluster(op));
  assert(views.size() == op.getBuffers().size());

  // Use the token to find the matching enter / exit pair
  //   %bufs:n, %token = aref_put.enter %aref[%enter_idx]
  //   tma_load %bufs[0]
  //   ..
  //   tma_load %bufs[n-1]
  //   aref_put.exit %aref[%exit_idx], %token
  ArefPutExitOp exitOp;
  for (auto user : op.getToken().getUsers()) {
    if (auto op = dyn_cast<ArefPutExitOp>(user)) {
      exitOp = op;
      break;
    }
  }
  if (!exitOp)
    return;
  assert(exitOp.getAref() == op.getAref() &&
         "Expecting matching Aref on the ArefPutExitOp");

  auto asyncKinds = castAsyncOpAttrs(exitOp.getAsyncOps());
  auto hasAsyncLoad = [](AsyncOp kind) {
    return kind == AsyncOp::TMALoad || kind == AsyncOp::CpAsync;
  };
  auto hasTMA = [](AsyncOp kind) { return kind == AsyncOp::TMALoad; };

  Value emptyWaitPred;
  if (llvm::any_of(asyncKinds, hasTMA))
    emptyWaitPred =
        createAfterProloguePredicate(rewriter, op, arefVal.depth);
  insertWaitOp(rewriter, op, emptyBarrier, op.getPhase(), op.getStage(),
               emptyWaitPred);

  if (llvm::any_of(asyncKinds, hasTMA)) {
    Value fullBarrier;
    if (arefVal.fullMbars) {
      fullBarrier =
          getFullBarrier(rewriter, loc, arefVal, op.getStage(),
                         getPartitionWsTagIds(op), getStageCluster(op));
    }
    lowerTMALoad(op, fullBarrier, rewriter, arefVal);
  }

  if (llvm::any_of(asyncKinds, hasAsyncLoad)) {
    for (auto mmav5 : mmav5Ops) {
      setIsAsync(mmav5, defaultNumStages);
    }
  }

  for (auto [oldBuffer, view] : llvm::zip(op.getBuffers(), views)) {
    oldBuffer.replaceAllUsesWith(view);
  }
}

static MemDescType getAsMutable(MemDescType type) {
  return MemDescType::get(type.getShape(), type.getElementType(),
                          type.getEncoding(), type.getMemorySpace(),
                          /*mutableMemory=*/true);
}

static void propagateMutability(Value value) {
  for (Operation *user : value.getUsers()) {
    if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      user->getResult(0).setType(
          getAsMutable(cast<MemDescType>(user->getResult(0).getType())));
      propagateMutability(user->getResult(0));
    }
  }
}

void rewriteGetEnterOp(ArefGetEnterOp op, PatternRewriter &rewriter,
                       ArefValue arefVal) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);

  if (!arefVal.tmaFullMbarCoversFullBarrier) {
    Value fullBarrier =
        getFullBarrier(rewriter, loc, arefVal, op.getStage(),
                       getPartitionWsTagIds(op), getStageCluster(op));
    insertWaitOp(rewriter, op, fullBarrier, op.getPhase(), op.getStage());
  }
  if (Value tmaBarrier =
          getTMAFullBarrier(rewriter, loc, arefVal, op.getStage(),
                            getPartitionWsTagIds(op), getStageCluster(op))) {
    insertWaitOp(rewriter, op, tmaBarrier, op.getPhase(), op.getStage());
  }
  auto views = getSubViews(arefVal, op.getStage(), loc, rewriter,
                           getPartitionWsTagIds(op), getStageCluster(op));
  assert(views.size() == op.getBuffers().size());

  for (auto [oldBuffer, view] : llvm::zip(op.getBuffers(), views)) {
    oldBuffer.replaceAllUsesWith(view);
    // Before aref lowering, memdesc_trans consumes an immutable buffer from
    // a get enter op. After lowering, all buffers are mutable.
    propagateMutability(view);
  }
}

void rewriteArefBufferOp(ArefBufferOp op, PatternRewriter &rewriter,
                         ArefValue arefVal) {
  auto loc = op->getLoc();
  rewriter.setInsertionPointAfter(op);
  auto views = getSubViews(arefVal, op.getStage(), loc, rewriter,
                           getPartitionWsTagIds(op), getStageCluster(op));
  assert(views.size() == op.getBuffers().size());
  for (int i = 0; i < op.getBuffers().size(); ++i)
    op.getBuffers()[i].replaceAllUsesWith(views[i]);
}

void insertArriveBarrier(Location loc, ArrayRef<AsyncOp> asyncOps,
                         PatternRewriter &rewriter, Value mbar,
                         std::optional<PartitionWsTagIds> partitionWsTagIds,
                         StageCluster stageCluster, ValueRange descs = {}) {
  for (auto asyncOpEnum : asyncOps) {
    Operation *arriveOp = {};
    switch (asyncOpEnum) {
    case AsyncOp::NONE:
    case AsyncOp::WGMMA:
      arriveOp = nvidia_gpu::ArriveBarrierOp::create(rewriter, loc, mbar, 1);
      break;
    case AsyncOp::TC5MMA:
    case AsyncOp::TMEMCopy:
      arriveOp = nvidia_gpu::TCGen5CommitOp::create(rewriter, loc, mbar,
                                                    Value(), descs);
      break;
    case AsyncOp::TMALoad:
      // nothing to do, the arrive is done by HW
      break;
    case AsyncOp::CpAsync:
    default:
      llvm_unreachable("unknown async op");
    }
    if (arriveOp)
      assignStageCluster(arriveOp, partitionWsTagIds, stageCluster, rewriter);
  }
}

void rewritePutExitOp(ArefPutExitOp op, PatternRewriter &rewriter,
                      ArefValue arefVal) {
  auto loc = op->getLoc();
  auto stageCluster = getStageCluster(op);
  auto asyncKinds = castAsyncOpAttrs(op.getAsyncOps());
  rewriter.setInsertionPointAfter(op);

  bool needFence = [&]() {
    bool isGenericProxy = llvm::any_of(
        asyncKinds, [](AsyncOp kind) { return kind == AsyncOp::NONE; });
    if (!isGenericProxy) {
      return false;
    }
    auto tmem = TensorMemorySpaceAttr::get(op.getContext());
    auto arefType = cast<ArefType>(op.getAref().getType());
    // Currently we assume that an aref does not contain both SMEM and TMEM.
    // So checking only the first buffer is fine.
    auto arefBufType = cast<MemDescType>(arefType.getBaseType()[0]);
    if (arefBufType.getMemorySpace() == tmem) {
      return false;
    }
    for (auto arefUser : op.getAref().getUsers()) {
      if (auto getExit = dyn_cast<ArefGetExitOp>(arefUser)) {
        bool isConsumerMMAv5 =
            llvm::any_of(castAsyncOpAttrs(getExit.getAsyncOps()),
                         [](AsyncOp kind) { return kind == AsyncOp::TC5MMA; });
        if (isConsumerMMAv5) {
          return true;
        }
      }
    }
    return false;
  }();

  if (needFence) {
    auto fence = FenceAsyncSharedOp::create(rewriter, loc, /*bCluster=*/false);
    assignStageCluster(fence, getPartitionWsTagIds(op), stageCluster, rewriter);
  }

  if (arefVal.fullMbars) {
    Value fullBarrier =
        getFullBarrier(rewriter, loc, arefVal, op.getStage(),
                       getPartitionWsTagIds(op), getStageCluster(op));
    insertArriveBarrier(loc, castAsyncOpAttrs(op.getAsyncOps()), rewriter,
                        fullBarrier, getPartitionWsTagIds(op),
                        getStageCluster(op));
  } else {
    assert(llvm::all_of(castAsyncOpAttrs(op.getAsyncOps()),
                        [](AsyncOp kind) { return kind == AsyncOp::TMALoad; }));
  }
}

void rewriteGetExitOp(ArefGetExitOp op, PatternRewriter &rewriter,
                      ArefValue arefVal) {
  auto loc = op->getLoc();
  auto stageCluster = getStageCluster(op);
  auto asyncKinds = castAsyncOpAttrs(op.getAsyncOps());
  rewriter.setInsertionPointAfter(op);

  bool needFence = [&]() {
    bool isGenericProxy = llvm::any_of(
        asyncKinds, [](AsyncOp kind) { return kind == AsyncOp::NONE; });
    if (!isGenericProxy) {
      return false;
    }
    for (auto arefUser : op.getAref().getUsers()) {
      if (auto putExit = dyn_cast<ArefPutExitOp>(arefUser)) {
        bool isProducerTMA =
            llvm::any_of(castAsyncOpAttrs(putExit.getAsyncOps()),
                         [](AsyncOp kind) { return kind == AsyncOp::TMALoad; });
        if (isProducerTMA) {
          return true;
        }
      }
    }
    return false;
  }();

  if (needFence) {
    auto fence = FenceAsyncSharedOp::create(rewriter, loc, /*bCluster=*/false);
    assignStageCluster(fence, getPartitionWsTagIds(op), stageCluster, rewriter);
  }

  Value emptyBarrier =
      getEmptyBarrier(rewriter, loc, arefVal, op.getStage(),
                      getPartitionWsTagIds(op), getStageCluster(op));
  SmallVector<Value> completionDescs;
  if (llvm::is_contained(asyncKinds, AsyncOp::TC5MMA)) {
    auto views = getSubViews(arefVal, op.getStage(), loc, rewriter,
                             getPartitionWsTagIds(op), getStageCluster(op));
    for (Value view : views) {
      auto type = dyn_cast<MemDescType>(view.getType());
      if (type && isa<SharedMemorySpaceAttr>(type.getMemorySpace()))
        completionDescs.push_back(view);
    }
  }
  insertArriveBarrier(loc, asyncKinds, rewriter, emptyBarrier,
                      getPartitionWsTagIds(op), stageCluster, completionDescs);
}

DenseSet<MMAv5OpInterface> getAsyncMMAv5Consumers(Value aref) {
  DenseSet<MMAv5OpInterface> mmav5Ops;
  for (auto arefUser : aref.getUsers()) {
    if (auto getEnter = dyn_cast<ArefGetEnterOp>(arefUser)) {
      if (hasPartition(getEnter) && getPartitionIds(getEnter).front() == 0) {
        // Ignore mmav5 ops in the default partition. They are not warp
        // specialized.
        continue;
      }

      for (auto consumer : getEnter->getUsers()) {
        if (auto mmav5 = dyn_cast<MMAv5OpInterface>(consumer)) {
          mmav5Ops.insert(mmav5);
        } else if (auto forOp = consumer->getParentOfType<scf::ForOp>()) {
          auto users =
              getTopLevelUsersInLoop(consumer, forOp, [](Operation *user) {
                return isa<MMAv5OpInterface>(user);
              });
          for (auto user : users) {
            mmav5Ops.insert(cast<MMAv5OpInterface>(user));
          }
        }
      }
    }
  }
  return mmav5Ops;
}

class LowerArefCreate : public OpRewritePattern<ArefCreateOp> {
public:
  LowerArefCreate(MLIRContext *ctx, unsigned defaultNumStages)
      : OpRewritePattern(ctx), defaultNumStages(defaultNumStages) {}

  LogicalResult matchAndRewrite(ArefCreateOp op,
                                PatternRewriter &rewriter) const override {
    SetVector<Operation *> opToDelete;
    opToDelete.insert(op.getOperation());

    // setIsAsync(true) will be invoked on these mmav5 ops during
    // rewritePutEnterOp when the producer is async loads. Since collecting
    // consumer mmav5 ops requires the corresponding get enter op to be still
    // used in the IR, collect them here.
    auto mmav5Ops = getAsyncMMAv5Consumers(op.getResult());
    auto aref = createAndInitMbar(op, rewriter, mmav5Ops);

    for (auto userOp : op->getUsers()) {
      opToDelete.insert(userOp);
      if (auto user = dyn_cast<ArefPutEnterOp>(userOp)) {
        rewritePutEnterOp(user, rewriter, aref, mmav5Ops, defaultNumStages);
      } else if (auto user = dyn_cast<ArefGetEnterOp>(userOp)) {
        rewriteGetEnterOp(user, rewriter, aref);
      } else if (auto user = dyn_cast<ArefPutExitOp>(userOp)) {
        rewritePutExitOp(user, rewriter, aref);
      } else if (auto user = dyn_cast<ArefGetExitOp>(userOp)) {
        rewriteGetExitOp(user, rewriter, aref);
      } else if (auto user = dyn_cast<ArefBufferOp>(userOp)) {
        rewriteArefBufferOp(user, rewriter, aref);
      } else {
        llvm_unreachable("users of aref can only be ArefPut or ArefGet");
      }
    }

    auto sorted = topologicalSort(opToDelete);
    OpBuilder b(op);
    auto replToken =
        ub::PoisonOp::create(b, op.getLoc(), b.getType<AsyncTokenType>());
    for (auto op : sorted) {
      if (auto enterOp = dyn_cast<ArefPutEnterOp>(op))
        enterOp.getToken().replaceAllUsesWith(replToken);
      else if (auto enterOp = dyn_cast<ArefGetEnterOp>(op))
        enterOp.getToken().replaceAllUsesWith(replToken);
    }
    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it)
      rewriter.eraseOp(*it);

    return success();
  }

private:
  unsigned defaultNumStages;
};

bool isProducerLoad(ArefCreateOp arefOp) {
  for (auto user : arefOp.getResult().getUsers()) {
    if (auto putOp = dyn_cast<ArefPutEnterOp>(user)) {
      if (llvm::any_of(putOp->getUsers(), [](auto user) {
            return isa<triton::nvws::DescriptorLoadOpInterface>(user);
          })) {
        return true;
      }
    }
  }
  return false;
}

void multiBufferAref(const SmallVector<ArefCreateOp> &arefOps, int numStages) {
  SmallVector<Operation *> allocsToErase;
  for (auto arefOp : arefOps) {
    SmallVector<Value> allocOps;
    SmallVector<Type> arefTypes;

    bool eligible = true;
    for (auto opnd : arefOp.getOperands()) {
      if (!opnd.getDefiningOp() || isa<TMEMAllocOp>(opnd.getDefiningOp())) {
        eligible = false;
      }
    }

    if (!eligible) {
      continue;
    }

    OpBuilder builder(arefOp);
    for (auto opnd : arefOp.getOperands()) {
      auto oldAlloc = opnd.getDefiningOp();
      auto arefBufType = cast<MemDescType>(opnd.getType());
      arefBufType =
          getMultiBufferedType(getBufferViewType(arefBufType, true), numStages);
      Operation *newAlloc = triton::nvws::createAlloc(
          builder, oldAlloc->getLoc(), arefBufType, Value());
      allocOps.push_back(newAlloc->getResult(0));
      arefTypes.push_back(arefBufType);
      oldAlloc->replaceAllUsesWith(newAlloc);
      allocsToErase.push_back(oldAlloc);
    }

    auto newAref =
        createArefCreateOp(builder, arefTypes, allocOps, arefOp.getLoc());

    arefOp.getResult().replaceAllUsesWith(newAref.getResult());
    arefOp.erase();
  }

  for (auto alloc : allocsToErase) {
    alloc->erase();
  }
}

template <typename EnterOp, typename ExitOp>
ExitOp createCombinedArefOps(SmallVector<EnterOp> &enterOps,
                             SmallVector<ExitOp> &exitOps, ArefCreateOp aref,
                             OpBuilder &builder,
                             Operation *combinedEnterInsertPoint = nullptr) {
  auto firstEnter = *llvm::min_element(enterOps, [](EnterOp a, EnterOp b) {
    assert(a->getBlock() == b->getBlock());
    return a->isBeforeInBlock(b);
  });

  auto lastExit = *llvm::max_element(exitOps, [](ExitOp a, ExitOp b) {
    assert(a->getBlock() == b->getBlock());
    return a->isBeforeInBlock(b);
  });

  SmallVector<Type> arefEnterBuffers;
  for (auto enterOp : enterOps) {
    arefEnterBuffers.push_back(enterOp.getResult(0).getType());
  }

  llvm::SmallSetVector<Attribute, 5> opAttrsSet;
  for (ExitOp exitOp : exitOps) {
    opAttrsSet.insert(exitOp.getAsyncOps().begin(), exitOp.getAsyncOps().end());
  }

  builder.setInsertionPointAfter(aref);
  auto zero = arith::ConstantIntOp::create(builder, aref.getLoc(), 0, 32);
  assignStageCluster(zero, getPartitionWsTagIds(firstEnter),
                     getStageCluster(firstEnter), builder);

  if (combinedEnterInsertPoint) {
    // Combined get enter must be placed after combined put enter
    builder.setInsertionPointAfter(combinedEnterInsertPoint);
  } else {
    builder.setInsertionPoint(firstEnter);
  }
  auto combinedEnter =
      EnterOp::create(builder, firstEnter.getLoc(), arefEnterBuffers,
                      builder.getType<AsyncTokenType>(), aref, zero, zero);
  assignStageCluster(combinedEnter, getPartitionWsTagIds(firstEnter),
                     getStageCluster(firstEnter), builder);

  builder.setInsertionPoint(lastExit);
  llvm::SmallVector<Attribute> AsyncOpAttrs(opAttrsSet.begin(),
                                            opAttrsSet.end());
  auto combinedExit = ExitOp::create(builder, firstEnter.getLoc(), aref,
                                     combinedEnter.getToken(), zero,
                                     builder.getArrayAttr(AsyncOpAttrs));
  assignStageCluster(combinedExit, getPartitionWsTagIds(lastExit),
                     getStageCluster(lastExit), builder);

  std::function<void(Operation *, Operation *)> moveUserAfter =
      [&](Operation *op, Operation *target) {
        auto curBlock = target->getBlock();
        for (auto user : op->getUsers()) {
          auto userOp = curBlock->findAncestorOpInBlock(*user);
          if (userOp->isBeforeInBlock(target)) {
            userOp->moveAfter(target);
            moveUserAfter(userOp, userOp);
          }
        }
      };

  for (auto [idx, enterOp] : llvm::enumerate(enterOps)) {
    moveUserAfter(enterOp, combinedEnter);
    enterOp.getBuffers()[0].replaceAllUsesWith(combinedEnter.getBuffers()[idx]);
  }

  return combinedExit;
}

SmallVector<Operation *> findSharedMemorySinkOps(Value value) {
  SmallVector<Operation *> sinkOps;
  for (Operation *user : value.getUsers()) {
    if (isa<MMAv5OpInterface, LocalLoadOp>(user)) {
      sinkOps.push_back(user);
    } else if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      auto rec = findSharedMemorySinkOps(user->getResult(0));
      sinkOps.insert(sinkOps.end(), rec.begin(), rec.end());
    }
  }
  return sinkOps;
}

Operation *getDominantConsumer(ArefGetEnterOp getEnterOp, Block &container,
                               DominanceInfo &domInfo) {
  assert(getEnterOp->getNumResults() && "Expect a single-result ArefGenterOp");
  auto buf = getEnterOp->getResult(0);
  SmallVector<Operation *> sinkOps = findSharedMemorySinkOps(buf);
  if (sinkOps.empty()) {
    return nullptr;
  }
  Operation *liveBeforeOp = findNearestCommonDominator(sinkOps, domInfo);
  return container.findAncestorOpInBlock(*liveBeforeOp);
}

// This is an optimization to combine arefs for TMA load into one, so that
// barrier arrive and wait are coalesced.
void combineArefs(scf::ForOp loop) {
  // We combine getEnterOps in the same loop body, not across a loop.
  auto getEnterOps = loop.getOps<ArefGetEnterOp>();

  // Arefs whose get-enter ops share the same dominant consumer can be combined
  DominanceInfo domInfo(loop);
  llvm::DenseMap<std::pair<Operation *, int>, SmallVector<ArefGetEnterOp>>
      liveBeforeGroups;
  for (auto getEnterOp : getEnterOps) {
    if (auto liveBeforeOp =
            getDominantConsumer(getEnterOp, *loop.getBody(), domInfo)) {
      assert(hasPartition(getEnterOp));
      auto partitionIds = getPartitionIds(getEnterOp);
      assert(partitionIds.size() == 1);
      liveBeforeGroups[{liveBeforeOp, partitionIds.front()}].push_back(
          getEnterOp);
    }
  }

  for (auto getEnterOps : llvm::make_second_range(liveBeforeGroups)) {
    if (getEnterOps.size() == 1) {
      continue;
    }

    SmallVector<ArefCreateOp> arefs;
    for (auto getEnterOp : getEnterOps) {
      arefs.push_back(cast<ArefCreateOp>(getEnterOp.getAref().getDefiningOp()));
    }

    SmallVector<ArefPutEnterOp> putEnterOps;
    SmallVector<ArefPutExitOp> putExitOps;
    SmallVector<ArefGetExitOp> getExitOps;
    SmallVector<int> producerGroupIds;
    for (auto aref : arefs) {
      for (auto user : aref->getUsers()) {
        if (auto putEnterOp = dyn_cast<ArefPutEnterOp>(user)) {
          putEnterOps.push_back(putEnterOp);
          producerGroupIds.push_back(getPartitionIds(putEnterOp).front());
        } else if (auto putExitOp = dyn_cast<ArefPutExitOp>(user)) {
          putExitOps.push_back(putExitOp);
        } else if (auto getExitOp = dyn_cast<ArefGetExitOp>(user)) {
          getExitOps.push_back(getExitOp);
        }
      }
    }

    // Producer arefs must be in the same partition.
    if (llvm::any_of(producerGroupIds,
                     [&](auto id) { return id != producerGroupIds[0]; })) {
      continue;
    }

    SmallVector<Type> arefBufTypes;
    SmallVector<Value> arefBufs;
    for (auto aref : arefs) {
      arefBufTypes.push_back(aref.getOperands()[0].getType());
      arefBufs.push_back(aref.getOperands()[0]);
    }

    // set insertion point at the last aref_create
    auto lastAref = *llvm::max_element(arefs, [](auto a, auto b) {
      assert(a->getBlock() == b->getBlock());
      return a->isBeforeInBlock(b);
    });

    OpBuilder builder(lastAref);
    auto aref =
        createArefCreateOp(builder, arefBufTypes, arefBufs, lastAref->getLoc());

    auto combinedPutExit =
        createCombinedArefOps(putEnterOps, putExitOps, aref, builder);
    createCombinedArefOps(getEnterOps, getExitOps, aref, builder,
                          combinedPutExit);

    for (auto putExitOp : putExitOps)
      putExitOp->erase();
    for (auto putEnterOp : putEnterOps)
      putEnterOp->erase();
    for (auto getExitOp : getExitOps)
      getExitOp->erase();
    for (auto getEnterOp : getEnterOps)
      getEnterOp->erase();
    for (auto aref : arefs)
      aref->erase();
  }
}

void hoistPoissonOps(triton::FuncOp funcOp) {
  SmallVector<ub::PoisonOp> poisonOps;
  auto block = &funcOp.getBody().front();
  funcOp.walk([&](ub::PoisonOp op) { op->moveBefore(&block->front()); });
}
} // anonymous namespace

class NVWSLowerAref : public impl::NVWSLowerArefBase<NVWSLowerAref> {
  using impl::NVWSLowerArefBase<NVWSLowerAref>::NVWSLowerArefBase;

public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();

    SmallVector<scf::ForOp> loops;
    m.walk([&](scf::ForOp loop) {
      if (loop->hasAttr(triton::kWarpSpecializeAttrName)) {
        loop->walk([&](scf::ForOp op) { loops.push_back(op); });
      }
    });

    for (scf::ForOp loop : loops) {
      combineArefs(loop);
    }

    SmallVector<ArefCreateOp> arefOps;
    m.walk([&](ArefCreateOp arefOp) {
      // Only handles arefs whose producer (a partition with PutEnter / Exit)
      // does load from global to shared memory.
      if (isProducerLoad(arefOp)) {
        arefOps.push_back(arefOp);
      }
    });
    multiBufferAref(arefOps, numStages);

    OpPassManager pm;
    pm.addPass(createNVWSAssignStagePhase());
    if (failed(runPipeline(pm, m)))
      return signalPassFailure();

    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerArefCreate>(context, numStages);
    GreedyRewriteConfig config;
    config.enableConstantCSE(false);
    config.enableFolding(false);
    if (applyPatternsGreedily(m, std::move(patterns), config).failed())
      signalPassFailure();

    // Hoist all poison ops to the top of function from nvws.wg regions.
    // They are unannotated and will trip subsequent passes, same to hoist.
    m.walk([&](triton::FuncOp funcOp) { hoistPoissonOps(funcOp); });
  }
}; // namespace triton

} // namespace triton
} // namespace mlir
