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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/IR/SemaphorePendingCount.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::nvws;

#define DEBUG_TYPE "nvws-lower-semaphore"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSLOWERSEMAPHORE
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
    return isa<SemaphoreAcquireOp, SemaphoreBufferOp>(op);
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

int getPendingCount(SemaphoreCreateOp op) {
  auto analysis = analyzeSemaphorePendingCount(op);
  assert(!analysis.invalidPartitionArity &&
         "partitioned semaphore.release must have exactly one partition id");
  assert(!analysis.unsupportedAsyncOp &&
         "unsupported async kind in semaphore.release pending-count analysis");
  assert(!analysis.inconsistentPartitionId &&
         "inconsistent per-partition pending-count contribution");
  return analysis.pendingCount;
}

Value createAndInitMbar(SemaphoreCreateOp op, PatternRewriter &rewriter) {
  int numStages = op.getType().getNumStages();
  int pendingCount = getPendingCount(op);

  rewriter.setInsertionPoint(op);
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  auto mbars = createScalarAlloc(b, b.getI64Type(), numStages);
  for (int i = 0; i < numStages; ++i) {
    auto view = createSingleBufferView(b, mbars, i);
    InitBarrierOp::create(b, view, pendingCount);
  }

  return mbars;
}

SmallVector<AsyncOp> castAsyncOpAttrs(ArrayAttr opAttrs) {
  SmallVector<AsyncOp> kinds;
  for (auto asyncKind : opAttrs) {
    kinds.push_back(cast<AsyncOpAttr>(asyncKind).getValue());
  }
  return kinds;
}

void createTMALoad(triton::nvws::DescriptorLoadOp op, PatternRewriter &rewriter,
                   Value barrierAlloc, Value pred) {
  auto newLoadOp = triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp::create(
      rewriter, op.getLoc(), op.getDesc(), op.getIndices(), barrierAlloc,
      op.getResult(), pred);
  assignStageCluster(newLoadOp, getPartitionWsTagIds(op), getStageCluster(op),
                     rewriter);
}

void createTMAGather(triton::nvws::DescriptorGatherOp op,
                     PatternRewriter &rewriter, Value barrierAlloc,
                     Value pred) {
  auto newGatherOp = triton::nvidia_gpu::AsyncTMAGatherOp::create(
      rewriter, op.getLoc(), op.getDesc(), op.getXOffsets(), op.getYOffset(),
      barrierAlloc, op.getResult(), pred);
  assignStageCluster(newGatherOp, getPartitionWsTagIds(op), getStageCluster(op),
                     rewriter);
}

void lowerTMALoad(SemaphoreReleaseOp op, PatternRewriter &rewriter,
                  Value mbars) {
  auto kinds = castAsyncOpAttrs(op.getAsyncOps());
  if (!llvm::any_of(kinds,
                    [](AsyncOp kind) { return kind == AsyncOp::TMALoad; }))
    return;

  auto loc = op.getLoc();
  int txCount = 0;
  SmallVector<Operation *> loadOps;
  for (auto tokUser : op.getToken().getUsers()) {
    auto bufOp = dyn_cast<SemaphoreBufferOp>(tokUser);
    if (!bufOp)
      continue;

    for (auto buffer : bufOp.getBuffers()) {
      for (auto user : buffer.getUsers()) {
        if (auto loadOp =
                dyn_cast<triton::nvws::DescriptorLoadOpInterface>(user)) {
          loadOps.push_back(loadOp);
          txCount += loadOp.getTxCount();
        }
      }
    }
  }
  assert(
      loadOps.size() <=
      op.getSemaphore().getDefiningOp<SemaphoreCreateOp>().getBuffers().size());
  if (loadOps.empty())
    return;

  auto topo = topologicalSort({loadOps.begin(), loadOps.end()});
  loadOps.assign(topo.begin(), topo.end());

  auto partitionWsTagIds = getPartitionWsTagIds(op);
  auto stageCluster = getStageCluster(op);

  rewriter.setInsertionPoint(loadOps.front());
  auto fullBarrier = createSingleBufferView(rewriter, mbars, op.getStage());
  assignStageCluster(fullBarrier.getDefiningOp(), partitionWsTagIds,
                     stageCluster, rewriter);

  auto pred = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
  assignStageCluster(pred, partitionWsTagIds, stageCluster, rewriter);
  auto expectOp = triton::nvidia_gpu::BarrierExpectOp::create(
      rewriter, loc, fullBarrier, txCount, pred);
  assignStageCluster(expectOp, partitionWsTagIds, stageCluster, rewriter);

  for (auto loadOp : loadOps) {
    rewriter.setInsertionPoint(loadOp);
    if (auto descLoad = dyn_cast<triton::nvws::DescriptorLoadOp>(loadOp)) {
      createTMALoad(descLoad, rewriter, fullBarrier, pred);
    } else if (auto descGather =
                   dyn_cast<triton::nvws::DescriptorGatherOp>(loadOp)) {
      createTMAGather(descGather, rewriter, fullBarrier, pred);
    } else {
      llvm_unreachable("Unknown load op");
    }
    loadOp->erase();
  }
}

void lowerTMALoads(SemaphoreCreateOp op, PatternRewriter &rewriter,
                   Value mbars) {
  for (auto user : op->getUsers()) {
    auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
    if (!releaseOp)
      continue;
    lowerTMALoad(releaseOp, rewriter, mbars);
  }
}

void rewriteAcquire(SemaphoreAcquireOp op, PatternRewriter &rewriter,
                    Value mbars) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  auto partitionWsTagIds = getPartitionWsTagIds(op);
  auto stageCluster = getStageCluster(op);

  auto mbar = createSingleBufferView(rewriter, mbars, op.getStage());
  assignStageCluster(mbar.getDefiningOp(), partitionWsTagIds, stageCluster,
                     rewriter);

  auto waitOp = WaitBarrierOp::create(rewriter, loc, mbar, op.getPhase());
  assignStageCluster(waitOp, partitionWsTagIds, stageCluster, rewriter);
}

void rewriteRelease(
    SemaphoreCreateOp semaOp, SemaphoreReleaseOp op, PatternRewriter &rewriter,
    Value mbars, const llvm::DenseMap<Operation *, bool> &hasAsyncPeerBySema) {
  auto loc = op.getLoc();
  auto asyncKinds = castAsyncOpAttrs(op.getAsyncOps());
  rewriter.setInsertionPointAfter(op);
  auto partitionWsTagIds = getPartitionWsTagIds(op);
  auto stageCluster = getStageCluster(op);

  bool needFence = [&]() {
    bool isGenericProxy = llvm::any_of(
        asyncKinds, [](AsyncOp kind) { return kind == AsyncOp::NONE; });
    if (!isGenericProxy)
      return false;

    // Currently we assume that an semaphore buffer does not contain both SMEM
    // and TMEM. So checking only the first buffer is fine.
    auto semaType = cast<SemaphoreType>(semaOp.getType());
    auto semaBufType = cast<MemDescType>(semaType.getBaseType()[0]);
    auto tmem = TensorMemorySpaceAttr::get(semaOp.getContext());
    if (semaBufType.getMemorySpace() == tmem)
      return false;

    // Fence decision depends on other semaphores grouped with this semaphore
    // by the first backing buffer. For a generic release (async_ops=[none]),
    // we need a fence if some other semaphore in that group has an async
    // release with either TC5MMA or TMALoad. Those async releases can make a
    // generic release on this semaphore need fence_async_shared before
    // arriving on its mbarrier.
    auto it = hasAsyncPeerBySema.find(semaOp.getOperation());
    return it != hasAsyncPeerBySema.end() && it->second;
  }();

  if (needFence) {
    auto fence = FenceAsyncSharedOp::create(rewriter, loc, /*bCluster=*/false);
    assignStageCluster(fence, partitionWsTagIds, stageCluster, rewriter);
  }

  auto mbar = createSingleBufferView(rewriter, mbars, op.getStage());
  assignStageCluster(mbar.getDefiningOp(), partitionWsTagIds, stageCluster,
                     rewriter);

  for (auto asyncKind : asyncKinds) {
    Operation *arriveOp = nullptr;
    switch (asyncKind) {
    case AsyncOp::NONE:
    case AsyncOp::WGMMA:
      arriveOp = ArriveBarrierOp::create(rewriter, loc, mbar, 1);
      break;
    case AsyncOp::TC5MMA:
    case AsyncOp::TMEMCopy:
      arriveOp =
          TCGen5CommitOp::create(rewriter, loc, mbar, Value(), ValueRange{});
      break;
    case AsyncOp::TMALoad:
      break;
    case AsyncOp::CpAsync:
    default:
      llvm_unreachable("unsupported async op");
    }
    if (arriveOp)
      assignStageCluster(arriveOp, partitionWsTagIds, stageCluster, rewriter);
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

void rewriteBuffer(SemaphoreBufferOp op, PatternRewriter &rewriter,
                   ArrayRef<Value> buffers) {
  auto loc = op.getLoc();
  auto partitionWsTagIds = getPartitionWsTagIds(op);
  auto stageCluster = getStageCluster(op);

  for (auto [i, buffer] : llvm::enumerate(buffers)) {
    // replacement helper may erase ops adjacent to this insertion point,
    // so refresh it for each buffer result before creating new view ops.
    rewriter.setInsertionPointAfter(op);

    auto memDesc = cast<MemDescType>(buffer.getType());
    if (isa<TensorMemoryScalesEncodingAttr>(memDesc.getEncoding())) {
      op.getBuffers()[i].replaceAllUsesWith(buffer);
      continue;
    }

    auto shape = memDesc.getShape();
    assert(shape.size() > 1 && "expected multi-buffered semaphore buffer");
    SmallVector<int64_t> viewShape(shape.begin() + 1, shape.end());
    auto viewType =
        MemDescType::get(viewShape, memDesc.getElementType(),
                         memDesc.getEncoding(), memDesc.getMemorySpace(),
                         /*mutableMemory=*/true);
    auto view =
        MemDescIndexOp::create(rewriter, loc, viewType, buffer, op.getStage());
    assignStageCluster(view, partitionWsTagIds, stageCluster, rewriter);
    op.getBuffers()[i].replaceAllUsesWith(view);
    // Before lowering, memdesc_trans consumes an immutable buffer.
    // After lowering, all buffers are mutable.
    propagateMutability(view);
  }
}

DenseSet<MMAv5OpInterface> getAsyncMMAv5Consumers(Value semaphore) {
  DenseSet<MMAv5OpInterface> mmav5Ops;
  for (auto semaUser : semaphore.getUsers()) {
    auto acquireOp = dyn_cast<SemaphoreAcquireOp>(semaUser);
    if (!acquireOp)
      continue;
    if (hasPartition(acquireOp) && getPartitionIds(acquireOp).front() == 0) {
      // Ignore MMAv5 ops in the default partition. They are not warp
      // specialized.
      continue;
    }

    for (auto tokUser : acquireOp.getToken().getUsers()) {
      auto bufferOp = dyn_cast<SemaphoreBufferOp>(tokUser);
      if (!bufferOp)
        continue;

      for (auto consumer : bufferOp->getUsers()) {
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
class LowerSemaphoreCreate : public OpRewritePattern<SemaphoreCreateOp> {
public:
  LowerSemaphoreCreate(
      MLIRContext *ctx,
      const llvm::DenseMap<Operation *, bool> &hasAsyncPeerBySema,
      unsigned defaultNumStages)
      : OpRewritePattern<SemaphoreCreateOp>(ctx),
        hasAsyncPeerBySema(hasAsyncPeerBySema),
        defaultNumStages(defaultNumStages) {}

  LogicalResult matchAndRewrite(SemaphoreCreateOp op,
                                PatternRewriter &rewriter) const override {
    for (auto user : op->getUsers()) {
      auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
      if (!releaseOp)
        continue;
      auto kinds = castAsyncOpAttrs(releaseOp.getAsyncOps());
      if (llvm::any_of(kinds, [](AsyncOp kind) {
            return kind == AsyncOp::TMALoad || kind == AsyncOp::CpAsync;
          })) {
        // the semaphore release op is async, so we need to setIsAsync(true)
        // if the peer semaphore consumes data via mmav5 ops.
        for (auto mma : getAsyncMMAv5Consumers(op.getResult()))
          setIsAsync(mma, defaultNumStages);
        break;
      }
    }

    auto mbars = createAndInitMbar(op, rewriter);
    SmallVector<Value> buffers(op.getBuffers().begin(), op.getBuffers().end());

    // Load TMA loads before erasing/rewriting semaphore users.
    lowerTMALoads(op, rewriter, mbars);

    SetVector<Operation *> opToDelete;
    opToDelete.insert(op.getOperation());

    SetVector<Operation *> allUsers;
    for (Operation *user : op->getUsers())
      allUsers.insert(user);

    Operation *cleanupAnchor = op.getOperation();
    if (!allUsers.empty()) {
      auto sortedUsers = topologicalSort(allUsers);
      cleanupAnchor =
          op->getBlock()->findAncestorOpInBlock(*sortedUsers.back());
    }

    {
      ImplicitLocOpBuilder b(op.getLoc(), rewriter);
      b.setInsertionPointAfter(cleanupAnchor);
      int numStages = op.getType().getNumStages();
      for (int i = 0; i < numStages; ++i) {
        auto view = createSingleBufferView(b, mbars, i);
        InvalBarrierOp::create(b, view);
      }
      LocalDeallocOp::create(b, mbars);
    }

    SmallVector<Operation *> users(op->getUsers().begin(),
                                   op->getUsers().end());
    for (auto userOp : users) {
      opToDelete.insert(userOp);
      if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(userOp)) {
        rewriteAcquire(acquireOp, rewriter, mbars);
      } else if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(userOp)) {
        rewriteRelease(op, releaseOp, rewriter, mbars, hasAsyncPeerBySema);
      } else if (auto bufferOp = dyn_cast<SemaphoreBufferOp>(userOp)) {
        rewriteBuffer(bufferOp, rewriter, buffers);
      } else {
        llvm_unreachable("unexpected semaphore user");
      }
    }

    auto sorted = topologicalSort(opToDelete);
    OpBuilder b(op);
    auto replToken =
        ub::PoisonOp::create(b, op.getLoc(), b.getType<AsyncTokenType>());
    // Poison tokens may be yielded by ws-loops and PartitionLoops requires
    // all ops to carry partition annotations.  Copy from the semaphore.
    if (hasPartition(op))
      setPartition(replToken, getPartitionIds(op));
    for (auto candidate : sorted) {
      if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(candidate))
        acquireOp.getToken().replaceAllUsesWith(replToken);
    }
    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it)
      rewriter.eraseOp(*it);

    return success();
  }

private:
  const llvm::DenseMap<Operation *, bool> &hasAsyncPeerBySema;
  unsigned defaultNumStages;
};

bool hasProducerLoad(SemaphoreCreateOp semaOp) {
  for (auto user : semaOp->getUsers()) {
    auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
    if (!releaseOp)
      continue;
    auto asyncKinds = castAsyncOpAttrs(releaseOp.getAsyncOps());
    if (llvm::any_of(asyncKinds,
                     [](AsyncOp kind) { return kind == AsyncOp::TMALoad; })) {
      return true;
    }
  }
  return false;
}

void multiBufferSemaphore(
    llvm::DenseMap<Value, SmallVector<SemaphoreCreateOp>> semaGroups,
    int numStages) {
  if (numStages <= 1)
    return;

  SetVector<Operation *> allocsToErase;
  for (auto &[_, semas] : semaGroups) {
    if (!llvm::any_of(semas, hasProducerLoad)) {
      continue;
    }

    bool eligible = true;
    for (auto opnd : semas.front().getBuffers()) {
      if (!opnd.getDefiningOp() || isa<TMEMAllocOp>(opnd.getDefiningOp())) {
        eligible = false;
      }
    }

    if (!eligible) {
      continue;
    }

    OpBuilder builder(semas.front());
    SmallVector<Value> newBuffers;
    SmallVector<Type> newBufferTypes;
    newBuffers.reserve(semas.front().getBuffers().size());
    newBufferTypes.reserve(semas.front().getBuffers().size());

    for (auto opnd : semas.front().getBuffers()) {
      auto oldAlloc = opnd.getDefiningOp();
      auto oldBufType = cast<MemDescType>(opnd.getType());
      auto newBufType =
          getMultiBufferedType(getBufferViewType(oldBufType, true), numStages);
      Operation *newAlloc = triton::nvws::createAlloc(
          builder, oldAlloc->getLoc(), newBufType, Value());
      newBuffers.push_back(newAlloc->getResult(0));
      newBufferTypes.push_back(newBufType);
      oldAlloc->replaceAllUsesWith(newAlloc);
      allocsToErase.insert(oldAlloc);
    }

    for (auto semaOp : semas) {
      OpBuilder semaBuilder(semaOp);
      auto semaTy = SemaphoreType::get(
          semaBuilder.getContext(),
          TypeArrayAttr::get(semaBuilder.getContext(), newBufferTypes));
      auto newSema =
          SemaphoreCreateOp::create(semaBuilder, semaOp.getLoc(), semaTy,
                                    newBuffers, semaOp.getIsReleased());
      newSema->setAttrs(semaOp->getAttrs());
      semaOp.getResult().replaceAllUsesWith(newSema.getResult());
      semaOp.erase();
    }
  }

  for (auto alloc : allocsToErase) {
    alloc->erase();
  }
}

// ---------------------------------------------------------------------------
// combineSemaphores: coalesce multiple semaphore pairs that feed the same
// dominant consumer in a warp-specialize for-loop.
// ---------------------------------------------------------------------------

void createCombinedSemaphoreOps(ArrayRef<SemaphoreAcquireOp> acquireOps,
                                ArrayRef<SemaphoreBufferOp> bufferOps,
                                ArrayRef<SemaphoreReleaseOp> releaseOps,
                                SemaphoreCreateOp acquireSema,
                                SemaphoreCreateOp releaseSema,
                                OpBuilder &builder) {
  assert(!acquireOps.empty() && !bufferOps.empty() && !releaseOps.empty());

  auto firstAcquire = *llvm::min_element(acquireOps, [](auto a, auto b) {
    assert(a->getBlock() == b->getBlock());
    return a->isBeforeInBlock(b);
  });
  auto lastRelease = *llvm::max_element(releaseOps, [](auto a, auto b) {
    assert(a->getBlock() == b->getBlock());
    return a->isBeforeInBlock(b);
  });

  auto partition = getPartitionWsTagIds(firstAcquire);
  auto stage = getStageCluster(firstAcquire);

  builder.setInsertionPoint(firstAcquire);
  auto combinedAcquire =
      SemaphoreAcquireOp::create(builder, firstAcquire.getLoc(), acquireSema,
                                 builder.getType<AsyncTokenType>());
  assignStageCluster(combinedAcquire, partition, stage, builder);

  SmallVector<Type> bufferResultTypes;
  for (auto bufferOp : bufferOps) {
    for (auto res : bufferOp.getBuffers())
      bufferResultTypes.push_back(res.getType());
  }

  builder.setInsertionPointAfter(combinedAcquire);
  auto combinedBuffer = SemaphoreBufferOp::create(
      builder, firstAcquire.getLoc(), acquireSema, TypeRange(bufferResultTypes),
      combinedAcquire.getToken());
  assignStageCluster(combinedBuffer, partition, stage, builder);

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

  int bufOffset = 0;
  for (auto bufferOp : bufferOps) {
    moveUserAfter(bufferOp, combinedBuffer);
    for (auto [j, oldBuf] : llvm::enumerate(bufferOp.getBuffers()))
      oldBuf.replaceAllUsesWith(combinedBuffer.getBuffers()[bufOffset + j]);
    bufOffset += bufferOp.getBuffers().size();
  }

  llvm::SmallSetVector<Attribute, 5> asyncOpsSet;
  for (auto relOp : releaseOps)
    asyncOpsSet.insert(relOp.getAsyncOps().begin(), relOp.getAsyncOps().end());

  builder.setInsertionPoint(lastRelease);
  auto combinedRelease = SemaphoreReleaseOp::create(
      builder, lastRelease.getLoc(), releaseSema, combinedAcquire.getToken(),
      builder.getArrayAttr(
          SmallVector<Attribute>(asyncOpsSet.begin(), asyncOpsSet.end())));
  assignStageCluster(combinedRelease, getPartitionWsTagIds(lastRelease),
                     getStageCluster(lastRelease), builder);
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

// 2-hop traversal: acquireOp → token → SemaphoreBufferOp → buffer results
// → findSharedMemorySinkOps → findNearestCommonDominator.
Operation *getDominantConsumer(SemaphoreAcquireOp acquireOp, Block &container,
                               DominanceInfo &domInfo) {
  SmallVector<Operation *> sinkOps;
  for (auto tokUser : acquireOp.getToken().getUsers()) {
    auto bufferOp = dyn_cast<SemaphoreBufferOp>(tokUser);
    if (!bufferOp)
      continue;
    for (auto buf : bufferOp.getBuffers()) {
      auto ops = findSharedMemorySinkOps(buf);
      sinkOps.insert(sinkOps.end(), ops.begin(), ops.end());
    }
  }
  if (sinkOps.empty()) {
    return nullptr;
  }
  Operation *liveBeforeOp = findNearestCommonDominator(sinkOps, domInfo);
  return container.findAncestorOpInBlock(*liveBeforeOp);
}

struct SemaToCombineInfo {
  SemaphoreCreateOp emptySema;
  SemaphoreCreateOp fullSema;
  SemaphoreBufferOp consBufferOp;
  SemaphoreReleaseOp consReleaseOp;
  SemaphoreAcquireOp prodAcquireOp;
  SemaphoreBufferOp prodBufferOp;
  SemaphoreReleaseOp prodReleaseOp;
};

SmallVector<SemaToCombineInfo>
analyzeCombinedSemaphoreGroup(ArrayRef<SemaphoreAcquireOp> acquireGroup) {
  SmallVector<SemaToCombineInfo> combinedInfos;
  SmallVector<int> producerPartitionIds;

  for (auto consAcquire : acquireGroup) {
    SemaToCombineInfo info;

    // Acquire-token lineage: consumer acquires FULL; consumer release targets
    // EMPTY (cross-release). Follow consumer acquire -> token ->
    // SemaphoreReleaseOp -> getSemaphore() to find that partner EMPTY
    // semaphore.
    for (Operation *tokUser : consAcquire.getToken().getUsers()) {
      if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(tokUser)) {
        if (info.consReleaseOp)
          return {};
        info.consReleaseOp = releaseOp;
      } else if (auto bufferOp = dyn_cast<SemaphoreBufferOp>(tokUser)) {
        if (info.consBufferOp)
          return {};
        info.consBufferOp = bufferOp;
      }
    }

    // Skip groups whose consumer acquire does not have the canonical
    // FULL acquire -> buffer -> cross-release EMPTY protocol shape.
    if (!info.consBufferOp || !info.consReleaseOp)
      return {};

    auto fullSema =
        consAcquire.getSemaphore().getDefiningOp<SemaphoreCreateOp>();
    auto emptySema =
        info.consReleaseOp.getSemaphore().getDefiningOp<SemaphoreCreateOp>();
    info.emptySema = emptySema;
    info.fullSema = fullSema;

    // Collect producer partition IDs from EMPTY semaphore users.
    for (auto user : emptySema->getUsers()) {
      auto prodAcquire = dyn_cast<SemaphoreAcquireOp>(user);
      if (!prodAcquire)
        continue;
      producerPartitionIds.push_back(getPartitionIds(prodAcquire).front());
      if (info.prodAcquireOp)
        return {};
      info.prodAcquireOp = prodAcquire;
      for (Operation *tokUser : prodAcquire.getToken().getUsers()) {
        if (auto bufferOp = dyn_cast<SemaphoreBufferOp>(tokUser)) {
          if (info.prodBufferOp)
            return {};
          info.prodBufferOp = bufferOp;
        }
        if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(tokUser)) {
          if (info.prodReleaseOp)
            return {};
          info.prodReleaseOp = releaseOp;
        }
      }
    }

    if (!info.prodAcquireOp || !info.prodBufferOp || !info.prodReleaseOp)
      return {};
    combinedInfos.push_back(info);
  }

  // All producers must be in the same partition.
  if (!producerPartitionIds.empty() &&
      llvm::any_of(producerPartitionIds,
                   [&](int id) { return id != producerPartitionIds[0]; })) {
    // The combine rewrite assumes one producer partition for the whole group.
    // Mixed producer partitions need a different protocol reconstruction.
    return {};
  }

  return combinedInfos;
}

struct CombinedSemaPair {
  SemaphoreCreateOp empty;
  SemaphoreCreateOp full;
};

CombinedSemaPair createCombinedSemaphores(ArrayRef<SemaToCombineInfo> infos,
                                          scf::ForOp loop) {
  SmallVector<Value> allBufs;
  SmallVector<Type> allBufTypes;
  for (auto info : infos) {
    for (Value buf : info.fullSema.getBuffers()) {
      allBufs.push_back(buf);
      allBufTypes.push_back(buf.getType());
    }
  }

  auto lastInfo = *llvm::max_element(infos, [](auto a, auto b) {
    assert(a.fullSema->getBlock() == b.fullSema->getBlock());
    return a.fullSema->isBeforeInBlock(b.fullSema);
  });
  auto lastCreate = lastInfo.fullSema;

  auto *ctx = loop->getContext();
  auto combinedType =
      SemaphoreType::get(ctx, TypeArrayAttr::get(ctx, allBufTypes));

  OpBuilder builder(lastCreate);
  builder.setInsertionPointAfter(lastCreate);
  // EMPTY must appear before FULL in IR so that the greedy rewriter
  // processes FULL first. lowerTMALoads on the FULL semaphore follows
  // the producer-release token chain through the EMPTY semaphore's
  // acquire/buffer ops; those must still be live at that point.
  auto combinedEmpty =
      SemaphoreCreateOp::create(builder, lastCreate->getLoc(), combinedType,
                                allBufs, /*isReleased=*/true);
  auto combinedFull =
      SemaphoreCreateOp::create(builder, lastCreate->getLoc(), combinedType,
                                allBufs, /*isReleased=*/false);
  return {combinedEmpty, combinedFull};
}

void combineConsumerSide(ArrayRef<SemaphoreAcquireOp> acquireGroup,
                         ArrayRef<SemaToCombineInfo> infos,
                         CombinedSemaPair combinedPair, OpBuilder &builder) {
  // Consumer acquires FULL, buffers from it, then cross-releases EMPTY.
  SmallVector<SemaphoreBufferOp> consBufferOps;
  SmallVector<SemaphoreReleaseOp> consReleaseOps;
  for (auto info : infos) {
    consBufferOps.push_back(info.consBufferOp);
    consReleaseOps.push_back(info.consReleaseOp);
  }
  createCombinedSemaphoreOps(acquireGroup, consBufferOps, consReleaseOps,
                             combinedPair.full, combinedPair.empty, builder);
}

void combineProducerSide(ArrayRef<SemaToCombineInfo> infos,
                         CombinedSemaPair combinedPair, OpBuilder &builder) {
  // Producer acquires EMPTY, buffers from it, then cross-releases FULL.
  SmallVector<SemaphoreAcquireOp> prodAcquireOps;
  SmallVector<SemaphoreBufferOp> prodBufferOps;
  SmallVector<SemaphoreReleaseOp> prodReleaseOps;
  for (auto info : infos) {
    prodAcquireOps.push_back(info.prodAcquireOp);
    prodBufferOps.push_back(info.prodBufferOp);
    prodReleaseOps.push_back(info.prodReleaseOp);
  }
  createCombinedSemaphoreOps(prodAcquireOps, prodBufferOps, prodReleaseOps,
                             combinedPair.empty, combinedPair.full, builder);
}

void eraseSemaToCombineGroup(ArrayRef<SemaphoreAcquireOp> acquireGroup,
                             ArrayRef<SemaToCombineInfo> infos) {
  for (auto info : infos)
    info.consReleaseOp->erase();
  for (auto info : infos)
    info.consBufferOp->erase();
  for (auto acquireOp : acquireGroup)
    acquireOp->erase();
  for (auto info : infos)
    info.prodReleaseOp->erase();
  for (auto info : infos)
    info.prodBufferOp->erase();
  for (auto info : infos)
    info.prodAcquireOp->erase();
  for (auto info : infos)
    info.fullSema->erase();
  for (auto info : infos)
    info.emptySema->erase();
}

void combineSemaphores(scf::ForOp loop) {
  // 1. Find consumer acquire ops (consumer = acquires FULL semaphore,
  //    isReleased == false). Skip TMEM.
  SmallVector<SemaphoreAcquireOp> consumerAcquires;
  auto tmem = TensorMemorySpaceAttr::get(loop.getContext());
  for (auto acquireOp : loop.getOps<SemaphoreAcquireOp>()) {
    auto semaCreate =
        acquireOp.getSemaphore().getDefiningOp<SemaphoreCreateOp>();
    if (semaCreate.getIsReleased())
      continue;
    bool isTMEM = llvm::any_of(semaCreate.getBuffers(), [&](Value buf) {
      return cast<MemDescType>(buf.getType()).getMemorySpace() == tmem;
    });
    if (isTMEM)
      continue;
    consumerAcquires.push_back(acquireOp);
  }

  // 2. Group by (dominant consumer, partition ID).
  DominanceInfo domInfo(loop);
  llvm::DenseMap<std::pair<Operation *, int>, SmallVector<SemaphoreAcquireOp>>
      groups;
  for (auto acquireOp : consumerAcquires) {
    auto liveBeforeOp =
        getDominantConsumer(acquireOp, *loop.getBody(), domInfo);
    if (!liveBeforeOp)
      continue;
    assert(hasPartition(acquireOp));
    auto partitionIds = getPartitionIds(acquireOp);
    assert(partitionIds.size() == 1);
    groups[{liveBeforeOp, partitionIds.front()}].push_back(acquireOp);
  }

  // 3. Combine each group with size > 1.
  for (auto &[key, acquireGroup] : groups) {
    if (acquireGroup.size() <= 1)
      continue;

    auto groupInfo = analyzeCombinedSemaphoreGroup(acquireGroup);
    if (groupInfo.empty())
      continue;
    auto combinedPair = createCombinedSemaphores(groupInfo, loop);
    OpBuilder builder(loop.getContext());
    combineConsumerSide(acquireGroup, groupInfo, combinedPair, builder);
    combineProducerSide(groupInfo, combinedPair, builder);
    eraseSemaToCombineGroup(acquireGroup, groupInfo);
  }
}

// Precompute cross-semaphore async relationships before any rewrite:
//
// hasAsyncPeerBySema[S] == true iff some other semaphore grouped with S by the
// first backing buffer has an async release with either TC5MMA or TMALoad.
// rewriteRelease uses that precomputed fact together with the current release
// kind and memory space to decide whether to insert fence_async_shared before
// arriving on S's mbarrier.
//
// This is required for deterministic/correct fence lowering. Greedy rewrites
// process semaphores independently and may erase one semaphore before
// rewriting its peer; computing this relation ahead of time avoids
// rewrite-order-dependent fence decisions.
llvm::DenseMap<Operation *, bool> computeHasAsyncPeerBySema(
    llvm::DenseMap<Value, SmallVector<SemaphoreCreateOp>> semaGroups) {
  llvm::DenseMap<Operation *, bool> hasAsyncPeerBySema;
  for (auto &[_, semas] : semaGroups) {
    llvm::DenseMap<Operation *, bool> hasAsyncRelease;
    for (auto semaOp : semas) {
      bool hasAsync = false;
      for (Operation *user : semaOp->getUsers()) {
        auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
        if (!releaseOp)
          continue;
        auto kinds = castAsyncOpAttrs(releaseOp.getAsyncOps());
        bool hasAsyncConsumer = llvm::any_of(
            kinds, [](AsyncOp kind) { return kind == AsyncOp::TC5MMA; });
        bool hasAsyncProducer = llvm::any_of(
            kinds, [](AsyncOp kind) { return kind == AsyncOp::TMALoad; });
        if (hasAsyncConsumer || hasAsyncProducer) {
          hasAsync = true;
          break;
        }
      }
      hasAsyncRelease[semaOp.getOperation()] = hasAsync;
    }

    for (auto semaOp : semas) {
      bool hasAsyncPeer = llvm::any_of(semas, [&](SemaphoreCreateOp otherSema) {
        return otherSema != semaOp &&
               hasAsyncRelease.lookup(otherSema.getOperation());
      });
      hasAsyncPeerBySema[semaOp.getOperation()] = hasAsyncPeer;
    }
  }

  return hasAsyncPeerBySema;
}

void hoistPoisonOps(triton::FuncOp funcOp) {
  auto block = &funcOp.getBody().front();
  funcOp.walk([&](ub::PoisonOp op) { op->moveBefore(&block->front()); });
}
} // anonymous namespace

class NVWSLowerSemaphore
    : public impl::NVWSLowerSemaphoreBase<NVWSLowerSemaphore> {
  using impl::NVWSLowerSemaphoreBase<
      NVWSLowerSemaphore>::NVWSLowerSemaphoreBase;

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
      combineSemaphores(loop);
    }

    auto getSemaGroups = [&]() {
      llvm::DenseMap<Value, SmallVector<SemaphoreCreateOp>> semaGroups;
      m.walk([&](SemaphoreCreateOp semaOp) {
        semaGroups[semaOp.getBuffers().front()].push_back(semaOp);
      });
      return semaGroups;
    };

    multiBufferSemaphore(getSemaGroups(), numStages);

    OpPassManager pm;
    pm.addPass(createNVWSAssignStagePhase());
    if (failed(runPipeline(pm, m)))
      return signalPassFailure();

    mlir::RewritePatternSet patterns(context);
    // Pass precomputed peer information into the pattern so each semaphore can
    // make a stable fence decision even after peers are rewritten/erased.
    auto hasAsyncPeerBySema = computeHasAsyncPeerBySema(getSemaGroups());
    patterns.add<LowerSemaphoreCreate>(context, hasAsyncPeerBySema, numStages);
    GreedyRewriteConfig config;
    config.enableConstantCSE(false);
    config.enableFolding(false);
    if (applyPatternsGreedily(m, std::move(patterns), config).failed())
      signalPassFailure();

    // Hoist all poison ops to the top of function from nvws.wg regions.
    // They are unannotated and will trip subsequent passes, same to hoist.
    m.walk([&](triton::FuncOp funcOp) { hoistPoisonOps(funcOp); });
  }
};

} // namespace triton
} // namespace mlir
