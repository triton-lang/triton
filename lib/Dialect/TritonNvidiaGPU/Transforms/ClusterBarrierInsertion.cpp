#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

namespace {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace ttn = mlir::triton::nvgpu;

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
  if (auto mma = dyn_cast<ttng::TCGen5MMAOp>(op)) {
    return mma.getTwoCtas();
  } else if (auto mmaScaled = dyn_cast<ttng::TCGen5MMAScaledOp>(op)) {
    // TODO: Change when we support scaled MMA with 2CTAs
    assert(!ttng::getModuleTwoCTAs(op->getParentOfType<ModuleOp>()) &&
           "Scaled MMA with 2CTAs not supported");
    return false;
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
  auto barrierTy = cast<ttg::MemDescType>(initBarrierOp.getAlloc().getType());
  return barrierTy.getShape()[0] != numCTAs;
}

static Operation *getTopLevelFunctionOp(Operation *op,
                                        FunctionOpInterface funcOp) {
  if (!funcOp || funcOp->getNumRegions() != 1 ||
      !funcOp->getRegion(0).hasOneBlock())
    return nullptr;
  Block *topLevelBlock = &funcOp->getRegion(0).front();

  Operation *topLevelOp = op;
  while (topLevelOp && topLevelOp->getBlock() != topLevelBlock) {
    topLevelOp = topLevelOp->getParentOp();
    if (topLevelOp == funcOp.getOperation())
      return nullptr;
  }
  return topLevelOp;
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

static bool opUsesTrackedMBarrier(Operation *op,
                                  const Allocation::BufferIdSetT &tracked,
                                  Allocation *allocation) {
  if (isa<ttng::InitBarrierOp, ttg::LocalAllocOp>(op))
    return false;

  if (auto memEffects = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
    memEffects.getEffects(effects);
    for (const auto &effect : effects) {
      if (Value value = effect.getValue()) {
        if (valueAliasesTrackedBuffers(value, tracked, allocation))
          return true;
      }
    }
  }

  // Calls do not always model memory effects precisely, but passing a tracked
  // mbarrier into a callee still counts as a use for placement purposes.
  if (isa<CallOpInterface>(op)) {
    for (Value operand : op->getOperands()) {
      if (valueAliasesTrackedBuffers(operand, tracked, allocation))
        return true;
    }
  }

  return false;
}

static LogicalResult
insertCrossCTAMBarrierInitSyncForFunction(FunctionOpInterface funcOp,
                                          Allocation *allocation, int numCTAs,
                                          OpBuilder &builder) {
  Block *topLevelBlock = nullptr;
  if (funcOp->getNumRegions() == 1 && funcOp->getRegion(0).hasOneBlock())
    topLevelBlock = &funcOp->getRegion(0).front();

  SmallVector<ttng::InitBarrierOp> crossCTAInitOps;
  Allocation::BufferIdSetT trackedBarrierBuffers;
  LogicalResult status = success();

  funcOp.walk([&](ttng::InitBarrierOp initBarrierOp) {
    if (failed(status) || !isCrossCTAMBarrier(initBarrierOp, numCTAs))
      return;
    if (!topLevelBlock) {
      initBarrierOp.emitOpError(
          "cross-CTA mbarrier init sync insertion requires a single function "
          "top-level block");
      status = failure();
      return;
    }
    Operation *topLevelOp = getTopLevelFunctionOp(initBarrierOp, funcOp);
    if (topLevelOp != initBarrierOp.getOperation()) {
      initBarrierOp.emitOpError(
          "cross-CTA mbarrier init sync insertion requires the init to be in "
          "the function top-level block");
      status = failure();
      return;
    }
    crossCTAInitOps.push_back(initBarrierOp);
    for (auto bufferId :
         allocation->getAllBufferIdsWithAliases(initBarrierOp.getAlloc())) {
      if (bufferId != Allocation::InvalidBufferId)
        trackedBarrierBuffers.insert(bufferId);
    }
  });
  if (failed(status) || crossCTAInitOps.empty())
    return status;

  SmallVector<Operation *> topLevelOps;
  DenseMap<Operation *, unsigned> topLevelOrder;
  for (Operation &op : *topLevelBlock) {
    topLevelOrder[&op] = topLevelOps.size();
    topLevelOps.push_back(&op);
  }

  Operation *lastCrossCTAInit = crossCTAInitOps.front().getOperation();
  for (ttng::InitBarrierOp initBarrierOp : crossCTAInitOps) {
    if (topLevelOrder.lookup(initBarrierOp.getOperation()) >
        topLevelOrder.lookup(lastCrossCTAInit)) {
      lastCrossCTAInit = initBarrierOp.getOperation();
    }
  }
  unsigned lastCrossCTAInitIdx = topLevelOrder.lookup(lastCrossCTAInit);

  Operation *firstTrackedUseAnchor = nullptr;
  funcOp.walk([&](Operation *op) {
    if (failed(status))
      return;
    if (!opUsesTrackedMBarrier(op, trackedBarrierBuffers, allocation))
      return;

    Operation *topLevelOp = getTopLevelFunctionOp(op, funcOp);
    if (!topLevelOp) {
      op->emitOpError(
          "cross-CTA mbarrier init sync insertion requires uses to be "
          "reachable from the function top-level block");
      status = failure();
      return;
    }
    if (!firstTrackedUseAnchor ||
        topLevelOrder.lookup(topLevelOp) <
            topLevelOrder.lookup(firstTrackedUseAnchor)) {
      firstTrackedUseAnchor = topLevelOp;
    }
  });
  if (failed(status))
    return status;

  unsigned windowEndIdx = firstTrackedUseAnchor
                              ? topLevelOrder.lookup(firstTrackedUseAnchor)
                              : topLevelOps.size();
  if (windowEndIdx <= lastCrossCTAInitIdx) {
    return lastCrossCTAInit->emitOpError()
           << "cannot insert cross-CTA mbarrier init sync ops after all "
              "cross-CTA ttng.init_barrier ops and before the first tracked "
              "mbarrier use";
  }

  auto isInWindow = [&](Operation *op) {
    unsigned idx = topLevelOrder.lookup(op);
    return idx > lastCrossCTAInitIdx && idx < windowEndIdx;
  };

  funcOp.walk([&](ttng::FenceMBarrierInitReleaseClusterOp fenceOp) {
    if (failed(status))
      return;
    Operation *topLevelOp = getTopLevelFunctionOp(fenceOp, funcOp);
    if (topLevelOp != fenceOp.getOperation() || !isInWindow(fenceOp)) {
      fenceOp.emitOpError(
          "must be in the function top-level block between the last cross-CTA "
          "ttng.init_barrier and the first tracked mbarrier use");
      status = failure();
    }
  });
  if (failed(status))
    return status;

  funcOp.walk([&](ttng::ClusterArriveOp arriveOp) {
    if (failed(status) || !arriveOp.getRelaxed())
      return;
    Operation *topLevelOp = getTopLevelFunctionOp(arriveOp, funcOp);
    if (topLevelOp != arriveOp.getOperation() || !isInWindow(arriveOp)) {
      arriveOp.emitOpError(
          "relaxed cluster barrier for cross-CTA mbarrier init sync must be "
          "in the function top-level block between the last cross-CTA "
          "ttng.init_barrier and the first tracked mbarrier use");
      status = failure();
    }
  });
  if (failed(status))
    return status;

  bool hasFenceInWindow = false;
  bool hasClusterBarrierInWindow = false;
  Operation *lastFenceInWindow = nullptr;
  DenseSet<Operation *> pairedClusterWaits;
  for (unsigned idx = lastCrossCTAInitIdx + 1; idx < windowEndIdx; ++idx) {
    Operation *op = topLevelOps[idx];

    if (isa<ttng::FenceMBarrierInitReleaseClusterOp>(op)) {
      hasFenceInWindow = true;
      lastFenceInWindow = op;
    }

    if (auto arriveOp = dyn_cast<ttng::ClusterArriveOp>(op)) {
      if (idx + 1 < windowEndIdx &&
          isa<ttng::ClusterWaitOp>(topLevelOps[idx + 1])) {
        hasClusterBarrierInWindow = true;
        pairedClusterWaits.insert(topLevelOps[idx + 1]);
        ++idx;
        continue;
      }
      return arriveOp.emitOpError(
          "cluster_arrive in cross-CTA mbarrier init sync window must be "
          "immediately followed by ttng.cluster_wait");
    }

    if (isa<ttng::ClusterWaitOp>(op) && !pairedClusterWaits.contains(op)) {
      return op->emitOpError(
          "cluster_wait in cross-CTA mbarrier init sync window must be "
          "immediately preceded by ttng.cluster_arrive");
    }
  }

  if (hasFenceInWindow && hasClusterBarrierInWindow)
    return success();

  OpBuilder::InsertionGuard guard(builder);
  Operation *insertAfter = lastCrossCTAInit;
  if (hasFenceInWindow && !hasClusterBarrierInWindow)
    insertAfter = lastFenceInWindow;
  builder.setInsertionPointAfter(insertAfter);
  Location loc = lastCrossCTAInit->getLoc();
  if (!hasFenceInWindow)
    ttng::FenceMBarrierInitReleaseClusterOp::create(builder, loc);
  if (!hasClusterBarrierInWindow) {
    ttng::ClusterArriveOp::create(builder, loc, /*relaxed=*/true);
    ttng::ClusterWaitOp::create(builder, loc);
  }

  return success();
}

class CrossCTAMBarrierInitSyncAnalysis : public MembarOrFenceAnalysis {
public:
  CrossCTAMBarrierInitSyncAnalysis(Allocation *allocation,
                                   MembarFilterFn filter, int numCTAs)
      : MembarOrFenceAnalysis(allocation, filter), numCTAs(numCTAs) {}

  LogicalResult getStatus() const { return status; }

private:
  void update(Operation *op, BlockInfo *blockInfo,
              FuncBlockInfoMapT *funcBlockInfoMap,
              OpBuilder *builder) override {
    (void)op;
    (void)blockInfo;
    (void)funcBlockInfoMap;
    if (processed || failed(status))
      return;
    processed = true;

    auto funcOp = dyn_cast<FunctionOpInterface>(allocation->getOperation());
    if (!funcOp)
      return;
    status = insertCrossCTAMBarrierInitSyncForFunction(funcOp, allocation,
                                                       numCTAs, *builder);
  }

  int numCTAs;
  bool processed = false;
  LogicalResult status = success();
};

class ClusterBarrierAnalysis : public MembarOrFenceAnalysis {
public:
  ClusterBarrierAnalysis() = default;
  explicit ClusterBarrierAnalysis(Allocation *allocation, MembarFilterFn filter)
      : MembarOrFenceAnalysis(allocation, filter) {}

private:
  void update(Operation *op, BlockInfo *blockInfo,
              FuncBlockInfoMapT *funcBlockInfoMap, OpBuilder *builder) override;
};

void ClusterBarrierAnalysis::update(Operation *op, BlockInfo *blockInfo,
                                    FuncBlockInfoMapT *funcBlockInfoMap,
                                    OpBuilder *builder) {
  if (isa<ttn::ClusterBarrierOp, ttng::ClusterWaitOp>(op)) {
    blockInfo->sync();
    return;
  }

  // Any path from distributed shared memory use to kernel exit must include a
  // cluster barrier.
  if (op->hasTrait<OpTrait::ReturnLike>() &&
      isa<FunctionOpInterface>(op->getParentOp())) {
    // In `freeTMAlloc` we emit a cluster sync during lowering for 2CTA kernels,
    // as we need to sync before the TMA deallocation
    // Note that 2CTA kernels must have a tcgen05_mma instruction and thus must
    // use TensorMemory
    // According to NVIDIA this is enough, so we don't need an extra
    // end-of-kernel barrier
    auto funcOp = cast<FunctionOpInterface>(op->getParentOp());
    if (isKernel(funcOp) && hasUnresolvedCrossClusterDependency(*blockInfo) &&
        !getModuleTwoCTAs(funcOp)) {
      builder->setInsertionPoint(op);
      ttn::ClusterBarrierOp::create(*builder, op->getLoc(), /*relaxed=*/false);
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
      ttn::ClusterBarrierOp::create(*builder, op->getLoc(), /*relaxed=*/false);
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
    ttn::ClusterBarrierOp::create(*builder, op->getLoc(), /*relaxed=*/false);
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

  triton::CallGraph<BlockInfo>::FuncDataMapT funcMap;
  LogicalResult status = success();
  moduleAllocation.walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
      [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
      [&](FunctionOpInterface funcOp) {
        if (failed(status))
          return;
        auto [it, inserted] = funcMap.try_emplace(funcOp, BlockInfo());
        if (!inserted)
          return;

        auto *allocation = moduleAllocation.getFuncData(funcOp);
        CrossCTAMBarrierInitSyncAnalysis analysis(allocation,
                                                  /*filter=*/nullptr, numCTAs);
        analysis.run(funcMap);
        if (failed(analysis.getStatus()))
          status = failure();
      });
  return status;
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
