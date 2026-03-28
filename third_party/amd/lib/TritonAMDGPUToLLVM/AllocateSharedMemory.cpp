#include "Analysis/AMDGPUAllocation.h"
#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::AMD;

namespace mlir::triton {
#define GEN_PASS_DEF_ALLOCATEAMDGPUSHAREDMEMORY
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace {

// Collect all buffer IDs of an op which will access shared memory.
static SmallVector<Allocation::BufferId>
getBufferIds(const Allocation *funcAllocation, Operation *op) {
  DenseSet<Allocation::BufferId> idSet;
  auto scratchId = funcAllocation->getBufferId(op);
  if (scratchId != Allocation::InvalidBufferId)
    idSet.insert(scratchId);
  // For ops that read/write shared memory (local_alloc, local_load,
  // local_store, buffer_load_to_local, etc.), collect buffer IDs from
  // the associated values.
  if (auto memEffects = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
    memEffects.getEffects(effects);
    for (auto &effect : effects) {
      if (effect.getResource() != triton::gpu::SharedMemory::get())
        continue;
      if (auto value = effect.getValue()) {
        for (auto id : funcAllocation->getAllBufferIdsWithAliases(value))
          idSet.insert(id);
      }
    }
  }
  return SmallVector<Allocation::BufferId>(idSet.begin(), idSet.end());
}

// Annotate LDS-accessing ops with alias scope information derived from
// AllocationAnalysis buffer allocated address ranges [offset, offset+size).
// Each buffer gets a unique scope ID, and ops are annotated with the set of
// non-overlapping buffer IDs (noalias).
static void attachAliasScopeInfo(ModuleOp mod, ModuleAllocation &allocation) {
  MLIRContext *ctx = mod.getContext();
  auto i32Ty = IntegerType::get(ctx, 32);

  mod.walk<mlir::WalkOrder::PreOrder>([&](FunctionOpInterface funcOp) {
    auto *funcAllocation = allocation.getFuncData(funcOp);

    // Walk ops once to collect op -> bufferId mapping
    SmallVector<std::pair<Operation *, Allocation::BufferId>> opBufferIds;
    DenseMap<Allocation::BufferId, Interval<size_t>> bufferIntervals;
    funcOp.walk([&](Operation *op) {
      auto ids = getBufferIds(funcAllocation, op);
      // Skip ops with no buffers or partitioned tensors (multiple IDs)
      if (ids.size() != 1)
        return;
      auto id = ids[0];
      opBufferIds.push_back({op, id});
      bufferIntervals.try_emplace(id, funcAllocation->getAllocatedInterval(id));
    });

    // Need at least 2 buffers to have meaningful noalias info
    if (bufferIntervals.size() <= 1)
      return WalkResult::skip();

    // Compute noalias sets: two buffers are non-aliasing iff their allocated
    // address ranges [offset, offset+size) are disjoint
    DenseMap<Allocation::BufferId, SmallVector<Attribute>> noaliasMap;
    for (auto &[id1, interval1] : bufferIntervals) {
      SmallVector<Attribute> noaliasAttrs;
      for (auto &[id2, interval2] : bufferIntervals) {
        if (id1 != id2 && !interval1.intersects(interval2))
          noaliasAttrs.push_back(
              IntegerAttr::get(i32Ty, static_cast<int64_t>(id2)));
      }
      noaliasMap[id1] = std::move(noaliasAttrs);
    }

    // Annotate ops with scopeId and noalias
    for (auto &[op, id] : opBufferIds) {
      op->setAttr("allocation.scope",
                  IntegerAttr::get(i32Ty, static_cast<int64_t>(id)));
      auto &noaliasAttrs = noaliasMap[id];
      if (!noaliasAttrs.empty())
        op->setAttr("allocation.scope.noalias",
                    ArrayAttr::get(ctx, noaliasAttrs));
    }

    return WalkResult::skip();
  });
}

struct AllocateAMDGPUSharedMemory
    : public mlir::triton::impl::AllocateAMDGPUSharedMemoryBase<
          AllocateAMDGPUSharedMemory> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Get partition size from target info
    size_t partitionSize = 0;
    if (auto arch = getAMDArch(mod)) {
      AMD::TargetInfo targetInfo(arch->str());
      partitionSize = targetInfo.getSharedMemoryPartitionSize();
    }

    ModuleAllocation allocation(mod, AMDAllocationAnalysisScratchSizeFn,
                                partitionSize);

    mlir::triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
    attachAliasScopeInfo(mod, allocation);
  }
};

} // namespace
