#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

namespace {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

static bool isDistributedMultiCTAOp(Operation *op, bool isRead) {
  if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(op)) {
    if (!isRead)
      return false;
    auto srcTy = cvt.getSrc().getType();
    auto dstTy = cvt.getType();
    auto kBlock = StringAttr::get(op->getContext(), "block");
    auto conversion = minimalCvtLayout(srcTy, dstTy);
    return conversion.hasInDim(kBlock);
  }
  if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
    if (!isRead)
      return false;
    auto srcTy = reduce.getInputTypes()[0];
    auto splitNum = ttg::getCTASplitNum(srcTy.getEncoding());
    return splitNum[reduce.getAxis()] > 1;
  }
  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op)) {
    return mma.getTwoCtas();
  } else if (isa<ttng::TMEMCopyOp>(op)) {
    return ttng::getModuleTwoCTAs(op);
  } else if (auto tma = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
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

static bool isCrossCTAMBarrier(ttng::InitBarrierOp initBarrierOp, int numCTAs) {
  auto barrierTy = cast<ttg::MemDescType>(initBarrierOp.getBarrier().getType());
  return barrierTy.getShape()[0] != numCTAs;
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
    return mma.getTwoCtas() &&
           llvm::any_of(barrierOp.getBarriers(), aliasesTracked);
  }
  if (auto commit = dyn_cast<ttng::TCGen5CommitOp>(op)) {
    return ttng::getModuleTwoCTAs(op) && aliasesTracked(commit.getBarrier());
  }
  if (auto tma = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
    return tma.getMulticast() && aliasesTracked(tma.getBarrier());
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
  if (isCrossCTAMBarrier(initBarrierOp, numCTAs))
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

static bool nestedOpUsesTrackedMBarrier(Operation *op,
                                        const Allocation::BufferIdSetT &tracked,
                                        Allocation *allocation) {
  if (isa<ttng::InitBarrierOp, ttg::LocalAllocOp>(op))
    return false;

  if (auto memEffects = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
    memEffects.getEffects(effects);
    for (const auto &effect : effects) {
      Value value = effect.getValue();
      if (value && valueAliasesTrackedBuffers(value, tracked, allocation))
        return true;
    }
  }
  return false;
}

static bool opUsesTrackedMBarrier(Operation *op,
                                  const Allocation::BufferIdSetT &tracked,
                                  Allocation *allocation) {
  return op
      ->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
        if (nestedOpUsesTrackedMBarrier(nestedOp, tracked, allocation))
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

static LogicalResult
insertCrossCTAMBarrierInitSyncForFunction(FunctionOpInterface funcOp,
                                          Allocation *allocation, int numCTAs,
                                          OpBuilder &builder) {
  if (!funcOp || funcOp->getNumRegions() != 1) {
    return funcOp.emitOpError(
        "cross-CTA mbarrier init sync insertion requires a single function "
        "top-level region");
  }
  Region &topLevelRegion = funcOp->getRegion(0);
  llvm::SetVector<Operation *> crossCTAInitAnchors;
  Allocation::BufferIdSetT trackedBarrierBuffers;

  // Find all cross-CTA mbarrier.init ops and map each
  // one to the containing top-level op that bounds the insertion window.
  funcOp.walk([&](ttng::InitBarrierOp initBarrierOp) {
    if (!requiresCrossCTAMBarrierInitSync(initBarrierOp, funcOp, allocation,
                                          numCTAs))
      return;
    Operation *topLevelAnchor =
        topLevelRegion.findAncestorOpInRegion(*initBarrierOp.getOperation());
    assert(topLevelAnchor && "init op must be inside the function region");
    crossCTAInitAnchors.insert(topLevelAnchor);
    for (auto bufferId :
         allocation->getAllBufferIdsWithAliases(initBarrierOp.getBarrier())) {
      assert(bufferId != Allocation::InvalidBufferId);
      trackedBarrierBuffers.insert(bufferId);
    }
  });
  // Nothing to do
  if (crossCTAInitAnchors.empty())
    return success();

  llvm::SetVector<Operation *> trackedUseAnchors;
  for (Block &block : topLevelRegion) {
    for (Operation &op : block) {
      if (opUsesTrackedMBarrier(&op, trackedBarrierBuffers, allocation))
        trackedUseAnchors.insert(&op);
    }
  }
  if (trackedUseAnchors.empty()) {
    return funcOp.emitOpError("found at least one mbarrier.init op but could "
                              "not find any mbarrier use");
  }

  // Find the earliest insertion point that postdominates every tracked init.
  PostDominanceInfo postDomInfo(funcOp);
  llvm::SmallPtrSet<Block *, 8> initBlocks;
  for (Operation *crossCTAInitAnchor : crossCTAInitAnchors)
    initBlocks.insert(crossCTAInitAnchor->getBlock());
  Block *firstInsertionBlock =
      postDomInfo.findNearestCommonDominator(initBlocks);
  if (!firstInsertionBlock) {
    return funcOp.emitOpError(
        "could not find a common post-dominating insertion block for "
        "cross-CTA mbarrier.init");
  }

  Operation *lastInitInInsertionBlock = nullptr;
  for (Operation *crossCTAInitAnchor : crossCTAInitAnchors) {
    if (crossCTAInitAnchor->getBlock() != firstInsertionBlock)
      continue;
    if (!lastInitInInsertionBlock ||
        lastInitInInsertionBlock->isBeforeInBlock(crossCTAInitAnchor)) {
      lastInitInInsertionBlock = crossCTAInitAnchor;
    }
  }
  Operation *firstInsertionAnchor =
      lastInitInInsertionBlock ? lastInitInInsertionBlock->getNextNode()
                               : &firstInsertionBlock->front();

  // Find the latest insertion point that still dominates every tracked use.
  DominanceInfo domInfo(funcOp);
  llvm::SmallPtrSet<Block *, 8> useBlocks;
  for (Operation *trackedUseAnchor : trackedUseAnchors)
    useBlocks.insert(trackedUseAnchor->getBlock());
  Block *lastInsertionBlock = domInfo.findNearestCommonDominator(useBlocks);
  if (!lastInsertionBlock) {
    return funcOp.emitOpError(
        "could not find a common insertion block that dominates all tracked "
        "mbarrier uses");
  }

  Operation *firstTrackedUseInInsertionBlock = nullptr;
  for (Operation *trackedUseAnchor : trackedUseAnchors) {
    if (trackedUseAnchor->getBlock() != lastInsertionBlock)
      continue;
    if (!firstTrackedUseInInsertionBlock ||
        trackedUseAnchor->isBeforeInBlock(firstTrackedUseInInsertionBlock)) {
      firstTrackedUseInInsertionBlock = trackedUseAnchor;
    }
  }
  Operation *lastInsertionAnchor = firstTrackedUseInInsertionBlock
                                       ? firstTrackedUseInInsertionBlock
                                       : lastInsertionBlock->getTerminator();

  if (!domInfo.dominates(firstInsertionAnchor, lastInsertionAnchor)) {
    return funcOp.emitOpError(
        "could not find an insertion point between cross-CTA mbarrier.init "
        "ops and tracked mbarrier uses");
  }

  // Reuse the latest cluster barrier that lies between the init-side and
  // use-side insertion boundaries.
  ttng::ClusterBarrierOp reusedClusterBarrier;
  for (Block &block : topLevelRegion) {
    for (Operation &op : block) {
      auto clusterBarrier = dyn_cast<ttng::ClusterBarrierOp>(&op);
      if (!clusterBarrier)
        continue;
      if (!postDomInfo.postDominates(clusterBarrier.getOperation(),
                                     firstInsertionAnchor))
        continue;
      if (!domInfo.dominates(clusterBarrier.getOperation(),
                             lastInsertionAnchor))
        continue;
      if (!reusedClusterBarrier ||
          domInfo.properlyDominates(reusedClusterBarrier.getOperation(),
                                    clusterBarrier.getOperation())) {
        reusedClusterBarrier = clusterBarrier;
      }
    }
  }

  OpBuilder::InsertionGuard guard(builder);
  Operation *fenceInsertionPoint =
      reusedClusterBarrier && reusedClusterBarrier.getRelaxed()
          ? reusedClusterBarrier.getOperation()
          : lastInsertionAnchor;
  builder.setInsertionPoint(fenceInsertionPoint);
  Location loc = lastInitInInsertionBlock
                     ? lastInitInInsertionBlock->getLoc()
                     : crossCTAInitAnchors.front()->getLoc();
  ttng::FenceMBarrierInitReleaseClusterOp::create(builder, loc);
  if (!reusedClusterBarrier)
    ttng::ClusterBarrierOp::create(builder, loc, /*relaxed=*/true);
  return success();
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

  LogicalResult status = success();
  moduleAllocation.walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
      [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
      [&](FunctionOpInterface funcOp) {
        if (failed(status))
          return;
        auto *allocation = moduleAllocation.getFuncData(funcOp);
        OpBuilder builder(funcOp);
        if (failed(insertCrossCTAMBarrierInitSyncForFunction(
                funcOp, allocation, numCTAs, builder))) {
          status = failure();
        }
      });
  return status;
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
