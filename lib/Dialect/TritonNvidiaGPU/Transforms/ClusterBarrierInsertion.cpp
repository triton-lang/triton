#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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

class ClusterBarrierAnalysis : public MembarOrFenceAnalysis {
public:
  ClusterBarrierAnalysis() = default;
  explicit ClusterBarrierAnalysis(Allocation *allocation, MembarFilterFn filter)
      : MembarOrFenceAnalysis(allocation, filter) {}

private:
  void update(Operation *op, BlockInfo *blockInfo,
              FuncBlockInfoMapT *funcBlockInfoMap, OpBuilder *builder) override;

  void insertClusterBarrier(Operation *op, OpBuilder *builder);
};

void ClusterBarrierAnalysis::insertClusterBarrier(Operation *op,
                                                  OpBuilder *builder) {
  OpBuilder::InsertionGuard guard(*builder);
  ttng::ClusterArriveOp::create(*builder, op->getLoc(), /*relaxed=*/false);
  ttng::ClusterWaitOp::create(*builder, op->getLoc());
}

void ClusterBarrierAnalysis::update(Operation *op, BlockInfo *blockInfo,
                                    FuncBlockInfoMapT *funcBlockInfoMap,
                                    OpBuilder *builder) {
  if (isa<ttng::ClusterWaitOp>(op)) {
    blockInfo->sync();
    return;
  }

  BlockInfo curBlockInfo;
  auto scratchBufferId = Allocation::InvalidBufferId;
  if (isa<triton::CallOp>(op)) {
    auto callOpInterface = dyn_cast<CallOpInterface>(op);
    if (auto callee =
            dyn_cast<FunctionOpInterface>(callOpInterface.resolveCallable()))
      curBlockInfo = funcBlockInfoMap->lookup(callee);
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
      insertClusterBarrier(op, builder);
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
    insertClusterBarrier(op, builder);
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
    if (!lhsDist && !rhsDist)
      return true;
    return false;
  };

  ModuleMembarOrFenceAnalysis<ClusterBarrierAnalysis> analysis(
      &moduleAllocation, filterFn);
  analysis.run();
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
