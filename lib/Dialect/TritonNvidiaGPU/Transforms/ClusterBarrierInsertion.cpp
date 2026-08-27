#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/MBarrierUtilities.h"

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

// Returns whether an operation's tracked access touches distributed shared
// memory across CTAs.
// op: The operation associated with the tracked access.
// isRead: Whether the access is recorded in BlockInfo::syncReadSlices rather
// than BlockInfo::syncWriteSlices.
bool isDistributedMultiCTAOp(Operation *op, bool isRead) {
  // Scratch writes are CTA-local. When the scratch spans CTAs, only its read
  // phase accesses another CTA's shared memory.
  if (hasCrossCTAScratch(op))
    return isRead;
  if (isa<ttng::CLCTryCancelOp, ttng::AsyncSharedStoreOp>(op)) {
    return ttg::lookupNumCTAs(op) > 1;
  } else if (isa<ttng::TMEMCopyOp>(op)) {
    return ttng::getModuleTwoCTAs(op);
  } else if (auto tma = dyn_cast<ttng::TMALoadLikeOpInterface>(op)) {
    return tma.getMulticast();
  } else if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(op)) {
    return arrive.isMulticast();
  }
  return hasTCGen5CommitCrossCTA(op);
}

bool isPreAllocAliasSliceFilter(const AllocationSlice &lhsSlice,
                                const AllocationSlice &rhsSlice,
                                bool /*lhsIsRead*/, bool /*rhsIsRead*/,
                                Allocation *allocation) {
  // Argument effects are checked after binding to the caller's allocation.
  if (lhsSlice.argumentIndex || rhsSlice.argumentIndex)
    return true;
  auto bufferId = lhsSlice.getBufferId();
  return bufferId != Allocation::InvalidBufferId &&
         bufferId == rhsSlice.getBufferId() &&
         allocation->isExplicitBuffer(bufferId);
}

bool hasUnresolvedCrossClusterDependency(const BlockInfo &blockInfo) {
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

bool valueAliasesTrackedBuffers(Value value,
                                const Allocation::BufferIdSetT &tracked,
                                Allocation *allocation) {
  for (auto bufferId : allocation->getAllBufferIdsWithAliases(value)) {
    if (bufferId != Allocation::InvalidBufferId && tracked.contains(bufferId))
      return true;
  }
  return false;
}

bool requiresCrossCTAMBarrierInitSync(ttng::InitBarrierOp initBarrierOp,
                                      FunctionOpInterface funcOp,
                                      Allocation *allocation, int numCTAs) {
  Allocation::BufferIdSetT initBarrierBuffers;
  for (auto bufferId :
       allocation->getAllBufferIdsWithAliases(initBarrierOp.getBarrier())) {
    assert(bufferId != Allocation::InvalidBufferId);
    initBarrierBuffers.insert(bufferId);
  }

  return mlir::triton::nvidia_gpu::requiresCrossCTAMBarrierInitSync(
      funcOp, initBarrierOp.getBarrier(), numCTAs, [&](Value value) {
        return value && valueAliasesTrackedBuffers(value, initBarrierBuffers,
                                                   allocation);
      });
}

bool nestedOpUsesTrackedMBarrier(Operation *op,
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

bool opUsesTrackedMBarrier(Operation *op,
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

LogicalResult
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

class ClusterBarrierAnalysis : public MembarAnalysis {
public:
  ClusterBarrierAnalysis(Allocation &allocation, MembarFilterFn filter,
                         BufferRegionAnalysis &regions)
      : MembarAnalysis(allocation, std::move(filter), regions,
                       isPreAllocAliasSliceFilter,
                       AccessMode::AllocatorAliasesOnly) {}

private:
  llvm::SmallPtrSet<Operation *, 4> returnsWithExitBarrier;

  BarrierStages getBarrierStages(Operation *op) override {
    BarrierStages stages;
    if (auto barrier = dyn_cast<ttng::ClusterBarrierOp>(op))
      stages.beforeMemoryEffects = !barrier.getRelaxed();
    // Distributed scratch synchronizes between its write and read phases.
    stages.betweenMemoryEffects = isDistributedMultiCTAOp(op, /*isRead=*/true);
    return stages;
  }

  void update(Operation *op, MembarInfo *membarInfo, FuncMapT *funcMap,
              OpBuilder *builder) override {
    if (op->hasTrait<OpTrait::ReturnLike>() &&
        isa<FunctionOpInterface>(op->getParentOp())) {
      // Any path from distributed shared memory use to kernel exit must include
      // a cluster barrier. Conservatively insert it because warp-specialized
      // memory effects are not fully modeled.
      if (isKernel(cast<FunctionOpInterface>(op->getParentOp()))) {
        // The solver may revisit this return before convergence.
        if (returnsWithExitBarrier.insert(op).second) {
          builder->setInsertionPoint(op);
          insertBarrier(op, builder, /*cluster=*/true);
        }
        membarInfo->sync();
      }
      return;
    }
    updateMemoryEffects(op, membarInfo, funcMap, builder, /*cluster=*/true);
  }
};

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

  ModuleMembarAnalysis analysis(moduleAllocation, filterFn);
  analysis.run<ClusterBarrierAnalysis>();
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
