#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

namespace {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

static bool isCrossCTAMBarrier(Value barrier, int numCTAs) {
  auto barrierTy = dyn_cast<ttg::MemDescType>(barrier.getType());
  return barrierTy && barrierTy.getShape()[0] != numCTAs;
}

static void getMBarrierUseOperands(Operation *op,
                                   SmallVectorImpl<Value> &barriers) {
  if (auto barrierOp = dyn_cast<ttg::MBarrierOpInterface>(op))
    llvm::append_range(barriers, barrierOp.getBarriers());
  if (auto tma = dyn_cast<ttng::TMALoadLikeOpInterface>(op))
    barriers.push_back(tma.getBarrier());
  if (auto commit = dyn_cast<ttng::TCGen5CommitOp>(op))
    barriers.push_back(commit.getBarrier());
  if (auto clc = dyn_cast<ttng::CLCTryCancelOp>(op))
    barriers.push_back(clc.getMbarrier());
}

static bool isDistributedMultiCTAOp(Operation *op, bool isRead) {
  if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(op)) {
    if (!isRead)
      return false;
    auto srcTy = cvt.getSrc().getType();
    auto dstTy = cvt.getType();
    auto kBlock = StringAttr::get(op->getContext(), "block");
    return !isCvtDimSync(ttg::toLinearLayout(srcTy), ttg::toLinearLayout(dstTy),
                         kBlock);
  }
  if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
    if (!isRead)
      return false;
    auto srcTy = reduce.getInputTypes()[0];
    auto splitNum = ttg::getCTASplitNum(srcTy.getEncoding());
    return splitNum[reduce.getAxis()] > 1;
  }
  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op)) {
    return mma.getTwoCtas() || mma.getMulticast();
  } else if (isa<ttng::TMEMCopyOp>(op)) {
    return ttng::getModuleTwoCTAs(op);
  } else if (auto tma = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
    return tma.getMulticast();
  } else if (auto tma = dyn_cast<ttng::AsyncTMAGatherOp>(op)) {
    return tma.getMulticast();
  }
  return false;
}

static bool isPreAllocAliasSliceFilter(const AllocationSlice &lhsSlice,
                                       const AllocationSlice &rhsSlice,
                                       bool /*lhsIsRead*/, bool /*rhsIsRead*/,
                                       Allocation *allocation) {
  auto bufferId = lhsSlice.getBufferId();
  return bufferId != Allocation::InvalidBufferId &&
         bufferId == rhsSlice.getBufferId() &&
         allocation->isExplicitBuffer(bufferId);
}

static bool hasUnresolvedCrossClusterDependency(const BlockInfo &blockInfo) {
  auto hasDistributedDependency = [](const BlockInfo::SliceMapT &slices,
                                     bool isRead) {
    for (const auto &sliceAndOps : slices)
      for (Operation *depOp : sliceAndOps.second)
        if (isDistributedMultiCTAOp(depOp, isRead))
          return true;
    return false;
  };

  return hasDistributedDependency(blockInfo.syncReadSlices, /*isRead=*/true) ||
         hasDistributedDependency(blockInfo.syncWriteSlices, /*isRead=*/false);
}

static bool valueAliasesTrackedBuffers(Value value,
                                       const Allocation::BufferIdSetT &tracked,
                                       Allocation *allocation) {
  for (auto bufferId : allocation->getAllBufferIdsWithAliases(value)) {
    if (bufferId != Allocation::InvalidBufferId && tracked.contains(bufferId))
      return true;
  }
  return false;
}

static bool
usesTrackedBarrierInCrossCTAConsumerOp(Operation *op,
                                       const Allocation::BufferIdSetT &tracked,
                                       Allocation *allocation) {
  auto aliasesTracked = [&](Value value) {
    return value && valueAliasesTrackedBuffers(value, tracked, allocation);
  };

  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op)) {
    auto barrierOp = cast<ttg::MBarrierOpInterface>(op);
    return (mma.getTwoCtas() || mma.getMulticast()) &&
           llvm::any_of(barrierOp.getBarriers(), aliasesTracked);
  }
  if (auto commit = dyn_cast<ttng::TCGen5CommitOp>(op)) {
    return ttng::getModuleTwoCTAs(op) && aliasesTracked(commit.getBarrier());
  }
  if (auto tma = dyn_cast<ttng::TMALoadLikeOpInterface>(op)) {
    return tma.getMulticast() && aliasesTracked(tma.getBarrier());
  }
  if (auto clc = dyn_cast<ttng::CLCTryCancelOp>(op)) {
    return aliasesTracked(clc.getMbarrier());
  }
  return false;
}

static bool requiresCrossCTAMBarrierInitSync(ttng::InitBarrierOp initBarrierOp,
                                             FunctionOpInterface funcOp,
                                             Allocation *allocation,
                                             int numCTAs) {
  // Barrier init sync is needed for barriers that are themselves cross-CTA,
  // and also for per-CTA barriers consumed by multi-CTA ops that multicast or
  // otherwise fan out barrier state across the cluster.
  if (isCrossCTAMBarrier(initBarrierOp.getBarrier(), numCTAs))
    return true;

  Allocation::BufferIdSetT initBarrierBuffers;
  for (auto bufferId :
       allocation->getAllBufferIdsWithAliases(initBarrierOp.getBarrier())) {
    assert(bufferId != Allocation::InvalidBufferId);
    initBarrierBuffers.insert(bufferId);
  }

  // Or if it's used by a multi-CTA consumer that broadcasts barrier state
  // across CTAs even though the barrier allocation itself looks per-CTA.
  return funcOp
      ->walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (usesTrackedBarrierInCrossCTAConsumerOp(op, initBarrierBuffers,
                                                   allocation)) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      })
      .wasInterrupted();
}

static void addUnpublishedInit(ttng::InitBarrierOp initBarrierOp,
                               BlockInfo *blockInfo,
                               Allocation *allocation) {
  Value barrier = initBarrierOp.getBarrier();
  for (auto bufferId : allocation->getAllBufferIdsWithAliases(barrier)) {
    assert(bufferId != Allocation::InvalidBufferId);
    auto interval = allocation->getAllocatedInterval(bufferId);
    auto slice = AllocationSlice(barrier, interval, bufferId);
    blockInfo->syncWriteSlices[slice].insert(initBarrierOp);
  }
}

static bool hasUnpublishedInit(const BlockInfo &blockInfo) {
  return !blockInfo.syncWriteSlices.empty();
}

static Allocation::BufferIdSetT
getUnpublishedInitBuffers(const BlockInfo &blockInfo) {
  Allocation::BufferIdSetT buffers;
  for (const auto &sliceAndOps : blockInfo.syncWriteSlices) {
    auto bufferId = sliceAndOps.first.getBufferId();
    if (bufferId != Allocation::InvalidBufferId)
      buffers.insert(bufferId);
  }
  return buffers;
}

static bool hasPreviousFenceMBarrierInitReleaseCluster(Operation *op) {
  return isa_and_nonnull<ttng::FenceMBarrierInitReleaseClusterOp>(
      op->getPrevNode());
}

static bool usesTrackedCrossCTAMBarrier(Operation *op,
                                        const Allocation::BufferIdSetT &tracked,
                                        Allocation *allocation, int numCTAs) {
  SmallVector<Value> barriers;
  getMBarrierUseOperands(op, barriers);
  return llvm::any_of(barriers, [&](Value barrier) {
    return valueAliasesTrackedBuffers(barrier, tracked, allocation) &&
           isCrossCTAMBarrier(barrier, numCTAs);
  });
}

static bool
requiresClusterBarrierForTrackedMBarrierUse(
    Operation *op, const Allocation::BufferIdSetT &tracked,
    Allocation *allocation, int numCTAs) {
  return usesTrackedCrossCTAMBarrier(op, tracked, allocation, numCTAs) ||
         usesTrackedBarrierInCrossCTAConsumerOp(op, tracked, allocation);
}

static bool opRequiresClusterBarrierForTrackedMBarrierUse(
    Operation *op, const Allocation::BufferIdSetT &tracked,
    Allocation *allocation, int numCTAs) {
  if (tracked.empty())
    return false;
  return op
      ->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
        if (isa<ttng::InitBarrierOp, ttg::LocalAllocOp>(nestedOp))
          return WalkResult::advance();
        if (requiresClusterBarrierForTrackedMBarrierUse(
                nestedOp, tracked, allocation, numCTAs))
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

class CrossCTAMBarrierInitSyncAnalysis : public MembarOrFenceAnalysis {
public:
  CrossCTAMBarrierInitSyncAnalysis(Allocation *allocation,
                                   MembarFilterFn filter)
      : MembarOrFenceAnalysis(allocation, filter) {
    auto mod = allocation->getOperation()->getParentOfType<ModuleOp>();
    numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
  }

private:
  void update(Operation *op, BlockInfo *blockInfo,
              FuncBlockInfoMapT *funcBlockInfoMap,
              OpBuilder *builder) override;

  int numCTAs;
};

void CrossCTAMBarrierInitSyncAnalysis::update(
    Operation *op, BlockInfo *blockInfo, FuncBlockInfoMapT *funcBlockInfoMap,
    OpBuilder *builder) {
  auto funcOp = op->getParentOfType<FunctionOpInterface>();
  if (!funcOp)
    return;

  if (isa<ttng::ClusterBarrierOp>(op)) {
    if (hasUnpublishedInit(*blockInfo) &&
        !hasPreviousFenceMBarrierInitReleaseCluster(op)) {
      OpBuilder::InsertionGuard guard(*builder);
      builder->setInsertionPoint(op);
      ttng::FenceMBarrierInitReleaseClusterOp::create(*builder, op->getLoc());
    }
    blockInfo->sync();
    return;
  }

  // A non-relaxed arrive/wait cluster sync publishes prior mbarrier.init
  // effects. Treat the wait as the synchronizing point for explicit pairs.
  if (isa<ttng::ClusterWaitOp>(op)) {
    blockInfo->sync();
    return;
  }

  if (auto initBarrierOp = dyn_cast<ttng::InitBarrierOp>(op)) {
    if (requiresCrossCTAMBarrierInitSync(initBarrierOp, funcOp, allocation,
                                         numCTAs))
      addUnpublishedInit(initBarrierOp, blockInfo, allocation);
    return;
  }

  Allocation::BufferIdSetT unpublishedInitBuffers =
      getUnpublishedInitBuffers(*blockInfo);
  if (opRequiresClusterBarrierForTrackedMBarrierUse(
          op, unpublishedInitBuffers, allocation, numCTAs)) {
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPoint(op);
    if (!hasPreviousFenceMBarrierInitReleaseCluster(op))
      ttng::FenceMBarrierInitReleaseClusterOp::create(*builder, op->getLoc());
    ttng::ClusterBarrierOp::create(*builder, op->getLoc());
    blockInfo->sync();
    return;
  }
}

class ClusterBarrierAnalysis : public MembarOrFenceAnalysis {
public:
  explicit ClusterBarrierAnalysis(Allocation *allocation, MembarFilterFn filter)
      : MembarOrFenceAnalysis(allocation, filter) {}

private:
  void update(Operation *op, BlockInfo *blockInfo,
              FuncBlockInfoMapT *funcBlockInfoMap, OpBuilder *builder) override;
};

void ClusterBarrierAnalysis::update(Operation *op, BlockInfo *blockInfo,
                                    FuncBlockInfoMapT *funcBlockInfoMap,
                                    OpBuilder *builder) {
  if (isa<ttng::ClusterBarrierOp, ttng::ClusterWaitOp>(op)) {
    blockInfo->sync();
    return;
  }

  // Any path from distributed shared memory use to kernel exit must include a
  // cluster barrier.
  if (op->hasTrait<OpTrait::ReturnLike>() &&
      isa<FunctionOpInterface>(op->getParentOp())) {
    // During TMEM deallocation lowering we emit a cluster sync for 2CTA
    // kernels, as we need to sync before the TMA deallocation.
    // Note that 2CTA kernels must have a tcgen05_mma instruction and thus must
    // use TensorMemory
    // According to NVIDIA this is enough, so we don't need an extra
    // end-of-kernel barrier
    auto funcOp = cast<FunctionOpInterface>(op->getParentOp());
    if (isKernel(funcOp) && hasUnresolvedCrossClusterDependency(*blockInfo) &&
        !getModuleTwoCTAs(funcOp)) {
      builder->setInsertionPoint(op);
      ttng::ClusterBarrierOp::create(*builder, op->getLoc());
      blockInfo->sync();
    }
    return;
  }

  BlockInfo curBlockInfo;
  auto scratchBufferId = Allocation::InvalidBufferId;
  if (isa<triton::CallOp>(op)) {
    auto callOpInterface = dyn_cast<CallOpInterface>(op);
    if (auto callee =
            dyn_cast<FunctionOpInterface>(callOpInterface.resolveCallable())) {
      auto calleeBlockInfo = funcBlockInfoMap->lookup(callee);
      auto callBufferId = allocation->getBufferId(op);
      size_t callOffset = 0;
      if (callBufferId != Allocation::InvalidBufferId)
        callOffset = allocation->getAllocatedInterval(callBufferId).start();
      curBlockInfo = translateBlockInfoToCallsite(calleeBlockInfo, callOffset);
    }
  } else {
    if (auto memEffects = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>>
          effectInstances;
      memEffects.getEffects(effectInstances);
      for (auto effectInstance : effectInstances) {
        if (auto value = effectInstance.getValue()) {
          for (auto bufferId : allocation->getBufferIds(value)) {
            if (bufferId != Allocation::InvalidBufferId) {
              auto interval = allocation->getAllocatedInterval(bufferId);
              auto slice = AllocationSlice(value, interval, bufferId);
              if (isa<MemoryEffects::Write>(effectInstance.getEffect()))
                curBlockInfo.syncWriteSlices[slice].insert(op);
              else if (isa<MemoryEffects::Read>(effectInstance.getEffect()))
                curBlockInfo.syncReadSlices[slice].insert(op);
            }
          }
        }
      }
    }
    scratchBufferId = allocation->getBufferId(op);
  }

  // Scratch buffer operations consist of a series of shared memory operations
  // starting from a shared memory write, followed by a series of shared memory
  // read/write operations, and ending with a shared memory read, i.e., shared
  // memory write -> ... -> shared memory read.
  if (scratchBufferId != Allocation::InvalidBufferId) {
    if (!curBlockInfo.syncReadSlices.empty() ||
        !curBlockInfo.syncWriteSlices.empty()) {
      llvm::report_fatal_error(
          "scratch buffer operations should not have any shared memory "
          "dependencies");
    }

    auto interval = allocation->getAllocatedInterval(scratchBufferId);
    auto scratchSlice = AllocationSlice(interval);
    curBlockInfo.syncWriteSlices[scratchSlice].insert(op);

    auto insertClusterBarrierNeeded = blockInfo->isIntersected(
        curBlockInfo, filter, allocation, isPreAllocAliasSliceFilter);
    if (insertClusterBarrierNeeded) {
      builder->setInsertionPoint(op);
      ttng::ClusterBarrierOp::create(*builder, op->getLoc());
    }

    // Clear prior distributed dependencies if we have inserted a cluster
    // barrier, or if the scratch op itself performs a cluster-level sync.
    bool hasClusterSync = isDistributedMultiCTAOp(op, /*isRead=*/true);
    if (insertClusterBarrierNeeded || hasClusterSync)
      blockInfo->sync();

    curBlockInfo.syncReadSlices[scratchSlice].insert(op);
  } else if (blockInfo->isIntersected(curBlockInfo, filter, allocation,
                                      isPreAllocAliasSliceFilter)) {
    builder->setInsertionPoint(op);
    ttng::ClusterBarrierOp::create(*builder, op->getLoc());
    blockInfo->sync();
  }

  blockInfo->join(curBlockInfo);
}

} // namespace

void runClusterBarrierInsertion(ModuleAllocation &moduleAllocation,
                                int computeCapability) {
  ModuleOp mod = moduleAllocation.getModuleOp();
  if (computeCapability < 90)
    return;
  if (ttg::TritonGPUDialect::getNumCTAs(mod) == 1)
    return;

  MembarFilterFn filterFn = [](Operation *lhs, Operation *rhs, bool lhsIsRead,
                               bool rhsIsRead, Allocation * /*allocation*/) {
    // Filter ops that do not touch distributed shared memory. Whether the
    // aliasing was already present in TTGIR is handled per-allocation slice.
    bool lhsDist = isDistributedMultiCTAOp(lhs, lhsIsRead);
    bool rhsDist = isDistributedMultiCTAOp(rhs, rhsIsRead);
    return !lhsDist && !rhsDist;
  };

  ModuleMembarOrFenceAnalysis<ClusterBarrierAnalysis> analysis(
      &moduleAllocation, filterFn);
  analysis.run();
}

LogicalResult
runCrossCTAMBarrierInitSyncInsertion(ModuleAllocation &moduleAllocation,
                                     int computeCapability) {
  ModuleOp mod = moduleAllocation.getModuleOp();
  if (computeCapability < 90)
    return success();
  int numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
  if (numCTAs == 1)
    return success();

  ModuleMembarOrFenceAnalysis<CrossCTAMBarrierInitSyncAnalysis> analysis(
      &moduleAllocation);
  analysis.run();
  return success();
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
