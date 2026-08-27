#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseSet.h"

#include <limits>

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUTMEMBARRIERINSERTIONPASS
#define GEN_PASS_DEF_TRITONNVIDIAGPUTMEMWAITINSERTIONPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;

enum class TMemAccessKind { None, Load, Store, MMA };

// Keep row groups far apart so per-row column intervals do not alias after
// flattening the physical 2D tensor-memory address space into 1D intervals.
static constexpr size_t kFlattenedRowStride = size_t{1} << 32;
static constexpr int kAllocRowGranularity = 64;
static constexpr int kRowOffsetGranularity = 16;

// Fine grain modeling of TMEM ops as pipelining behavior is not fully
// represented in ops attributes.
static bool isWritingAlloc(Operation *op) {
  auto alloc = dyn_cast<TMEMAllocOp>(op);
  return alloc && alloc.getSrc();
}

static bool isMMALikeOp(Operation *op) {
  return isa<TCGen5MMAOp, TCGen5MMAScaledOp, TMEMCopyOp>(op);
}

static TMemAccessKind getTMemAccessKind(Operation *op) {
  if (isa<TMEMLoadOp>(op))
    return TMemAccessKind::Load;
  if (isa<TMEMStoreOp>(op) || isWritingAlloc(op))
    return TMemAccessKind::Store;
  if (isMMALikeOp(op))
    return TMemAccessKind::MMA;
  return TMemAccessKind::None;
}

static bool filterFn(Operation *lhs, Operation *rhs, bool /*lhsIsRead*/,
                     bool /*rhsIsRead*/, Allocation * /*allocation*/) {
  TMemAccessKind lhsKind = getTMemAccessKind(lhs);
  TMemAccessKind rhsKind = getTMemAccessKind(rhs);

  bool war =
      lhsKind == TMemAccessKind::Load && rhsKind == TMemAccessKind::Store;
  bool raw =
      lhsKind == TMemAccessKind::Store && rhsKind == TMemAccessKind::Load;
  bool waw =
      lhsKind == TMemAccessKind::Store && rhsKind == TMemAccessKind::Store;

  // MMAv5 ops and tmem_copy are special cases, we care about load->mma and
  // store->mma dependencies but mma -> load/store doesn't require a barrier
  // since it would need a mbarrier wait that will ensure the op is finished
  // before any thread can reach the load/store.
  bool loadToMma =
      lhsKind == TMemAccessKind::Load && rhsKind == TMemAccessKind::MMA;
  bool storeToMma =
      lhsKind == TMemAccessKind::Store && rhsKind == TMemAccessKind::MMA;

  bool requiresBarrier = war || raw || waw || loadToMma || storeToMma;
  return !requiresBarrier;
}

static bool isTensorMemory(Value value) {
  auto memDescType = dyn_cast<ttg::MemDescType>(value.getType());
  return memDescType &&
         isa<TensorMemorySpaceAttr>(memDescType.getMemorySpace());
}

static void appendRootAllocs(Value value, SmallVectorImpl<TMEMAllocOp> &allocs,
                             bool &unknown) {
  DenseSet<Value> seen;
  SmallVector<Value> worklist{value};

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!seen.insert(current).second)
      continue;

    if (auto arg = dyn_cast<BlockArgument>(current)) {
      Block *block = arg.getOwner();
      Operation *parentOp = block->getParentOp();

      if (!block->isEntryBlock()) {
        for (Block *pred : block->getPredecessors()) {
          auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
          if (!branch) {
            unknown = true;
            continue;
          }
          auto it = llvm::find(branch->getSuccessors(), block);
          unsigned successorIndex =
              std::distance(branch->getSuccessors().begin(), it);
          SuccessorOperands args = branch.getSuccessorOperands(successorIndex);
          worklist.push_back(
              args.getForwardedOperands()[arg.getArgNumber() -
                                          args.getProducedOperandCount()]);
        }
        continue;
      }

      if (auto ws = dyn_cast<ttg::WarpSpecializePartitionsOp>(parentOp)) {
        worklist.push_back(ws.getExplicitCaptures()[arg.getArgNumber()]);
      } else if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        unsigned idx = arg.getArgNumber() - 1;
        worklist.push_back(forOp.getYieldedValues()[idx]);
        worklist.push_back(forOp.getInits()[idx]);
      } else if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
        unsigned idx = arg.getArgNumber();
        if (arg.getParentRegion() == &whileOp.getAfter()) {
          worklist.push_back(whileOp.getConditionOp().getArgs()[idx]);
        } else {
          worklist.push_back(whileOp.getYieldedValues()[idx]);
          worklist.push_back(whileOp.getInits()[idx]);
        }
      } else {
        unknown = true;
      }
      continue;
    }

    Operation *defOp = current.getDefiningOp();
    if (!defOp) {
      unknown = true;
      continue;
    }

    unsigned resultIndex = cast<OpResult>(current).getResultNumber();
    if (auto alloc = dyn_cast<TMEMAllocOp>(defOp)) {
      allocs.push_back(alloc);
    } else if (defOp->hasTrait<OpTrait::MemDescViewTrait>()) {
      worklist.push_back(defOp->getOperand(0));
    } else if (auto slice = dyn_cast<TMEMSubSliceOp>(defOp)) {
      worklist.push_back(slice.getSrc());
    } else if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
      worklist.push_back(selectOp.getTrueValue());
      worklist.push_back(selectOp.getFalseValue());
    } else if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      worklist.push_back(ifOp.thenYield().getOperand(resultIndex));
      worklist.push_back(ifOp.elseYield().getOperand(resultIndex));
    } else if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      worklist.push_back(forOp.getYieldedValues()[resultIndex]);
      worklist.push_back(forOp.getInits()[resultIndex]);
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(defOp)) {
      worklist.push_back(whileOp.getConditionOp().getArgs()[resultIndex]);
    } else {
      unknown = true;
    }
  }
}

static SmallVector<AllocationSlice> getTMemSlices(Value value) {
  SmallVector<TMEMAllocOp> allocs;
  bool unknown = false;
  appendRootAllocs(value, allocs, unknown);

  SmallVector<AllocationSlice> slices;
  if (unknown || allocs.empty()) {
    slices.emplace_back(
        Interval<size_t>(0, std::numeric_limits<size_t>::max()));
    return slices;
  }

  for (TMEMAllocOp alloc : allocs) {
    auto colAttr =
        alloc->getAttrOfType<IntegerAttr>("tensor_memory_col_offset");
    auto rowAttr =
        alloc->getAttrOfType<IntegerAttr>("tensor_memory_row_offset");
    if (!colAttr || !rowAttr) {
      slices.clear();
      slices.emplace_back(
          Interval<size_t>(0, std::numeric_limits<size_t>::max()));
      return slices;
    }

    int64_t colOffset = colAttr.getInt();
    int64_t rowOffset = rowAttr.getInt();
    TMemAllocation allocSize = getTmemAllocSizes(alloc.getType());
    if (rowOffset % kRowOffsetGranularity != 0 ||
        allocSize.numRows % kAllocRowGranularity != 0) {
      slices.clear();
      slices.emplace_back(
          Interval<size_t>(0, std::numeric_limits<size_t>::max()));
      return slices;
    }

    int64_t rowGroup = rowOffset / kRowOffsetGranularity;
    int64_t numRowGroups = allocSize.numRows / kAllocRowGranularity;
    for (int64_t row = 0; row < numRowGroups; ++row) {
      size_t start = static_cast<size_t>(rowGroup + row) * kFlattenedRowStride +
                     static_cast<size_t>(colOffset);
      slices.emplace_back(Interval<size_t>(start, start + allocSize.numCols));
    }
  }
  return slices;
}

static void appendReadSlices(Value value, Operation *op, BlockInfo *blockInfo) {
  if (!isTensorMemory(value))
    return;
  for (AllocationSlice slice : getTMemSlices(value))
    blockInfo->syncReadSlices[slice].insert(op);
}

static void appendWriteSlices(Value value, Operation *op,
                              BlockInfo *blockInfo) {
  if (!isTensorMemory(value))
    return;
  for (AllocationSlice slice : getTMemSlices(value))
    blockInfo->syncWriteSlices[slice].insert(op);
}

static BlockInfo getTMemAccesses(Operation *op) {
  // Use physical TMEM allocation slices; unknown origins alias all TMEM.
  BlockInfo accesses;
  for (const MemoryAccess &access : getMemoryAccesses(op)) {
    if (access.isRead)
      appendReadSlices(access.value, op, &accesses);
    if (access.isWrite)
      appendWriteSlices(access.value, op, &accesses);
  }
  return accesses;
}

class TMemBarrierAnalysis : public MembarAnalysis {
public:
  using MembarAnalysis::MembarAnalysis;

private:
  void update(Operation *operation, MembarInfo *membarInfo, FuncMapT *funcMap,
              OpBuilder *builder) override;
};

void TMemBarrierAnalysis::update(Operation *op, MembarInfo *membarInfo,
                                 FuncMapT *funcMap, OpBuilder *builder) {
  auto stages = mlir::getLocalBarrierStages(op, &allocation);
  if (stages.beforeMemoryEffects)
    membarInfo->sync();

  if (isa<triton::CallOp>(op)) {
    auto call = cast<CallOpInterface>(op);
    if (auto callee = dyn_cast<FunctionOpInterface>(call.resolveCallable())) {
      // Tensor-memory allocation attributes are physical addresses assigned
      // across the whole module, so callee slices need no call-frame offset.
      MembarInfo calleeMembarInfo = funcMap->lookup(callee);
      if (membarInfo->pending.isIntersected(calleeMembarInfo.entryBlockInfo,
                                            filter, &allocation)) {
        builder->setInsertionPoint(op);
        insertBarrier(op, builder);
        membarInfo->sync();
      }
      membarInfo->applyCallSummary(calleeMembarInfo);
    }
    return;
  }

  BlockInfo curBlockInfo = getTMemAccesses(op);
  if (membarInfo->pending.isIntersected(curBlockInfo, filter, &allocation)) {
    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    membarInfo->sync();
  }

  membarInfo->addBlockInfo(curBlockInfo);
  // A leading or interior rendezvous must preserve the operation's own effects.
  if (stages.afterMemoryEffects ||
      (stages.betweenMemoryEffects && curBlockInfo.syncReadSlices.empty() &&
       curBlockInfo.syncWriteSlices.empty()))
    membarInfo->sync();
}

// Ops before which we 100% need to flush reads and writes
bool isWaitBoundary(Operation *op) {
  // Defer waits at CTA barriers until a hazard or boundary requires them.
  if (isa<gpu::BarrierOp>(op))
    return false;

  // Flush before starting or ending WS
  if (isa<gpu::WarpSpecializeOp, gpu::WarpSpecializePartitionsOp,
          gpu::WarpYieldOp, gpu::WarpReturnOp, CallOpInterface>(op))
    return true;
  // A non-relaxed cluster barrier could be followed by a 2CTA MMA, in which
  // case we need to put the wait before the cluster barrier.
  // We could track barriers and cluster barriers separately
  // and place the wait before the previous relevant op, but
  // I don't think there are many use cases for that.
  if (auto barrier = dyn_cast<ClusterBarrierOp>(op))
    return !barrier.getRelaxed();
  // Atomic ops may synchronise as well
  if (auto atomic = dyn_cast<AtomicOpInterface>(op))
    if (atomic.getMemSemantic() == MemSemantic::RELEASE ||
        atomic.getMemSemantic() == MemSemantic::ACQUIRE_RELEASE)
      return true;
  // Mbarriers may signal other WS partitions.
  if (auto barrier = dyn_cast<gpu::MBarrierOpInterface>(op))
    if (!barrier.getBarriers().empty())
      return true;

  // Carry pending accesses through control-flow joins and backedges.
  if (isa<BranchOpInterface, RegionBranchOpInterface>(op) ||
      (isa<RegionBranchTerminatorOpInterface>(op) &&
       isa<RegionBranchOpInterface>(op->getParentOp())))
    return false;
  if (op->hasTrait<OpTrait::ReturnLike>())
    return true;

  // Volatile loads add an opaque write effect to prevent optimization, but
  // still access only global memory.
  if (isa<triton::LoadOp>(op))
    return false;

  // Wait if we can't prove that it's memory-effect free
  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory)
    return !isMemoryEffectFree(op);
  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);
  return llvm::any_of(effects, [](const auto &effect) {
    Value value = effect.getValue();
    // - An unbound DefaultResource effect may hide synchronization
    //   in impure inline asm or extern calls.
    if (!value)
      return isa<SideEffects::DefaultResource>(effect.getResource());
    return isTensorMemory(value) &&
           isa<MemoryEffects::Free>(effect.getEffect());
  });
}

class TMemWaitAnalysis : public MembarAnalysis {
public:
  using MembarAnalysis::MembarAnalysis;

private:
  static BlockInfo::SliceMapT &accesses(BlockInfo &info, TMEMWaitKind kind) {
    return kind == TMEMWaitKind::LOAD ? info.syncReadSlices
                                      : info.syncWriteSlices;
  }

  bool hasHazard(const BlockInfo &pending, const BlockInfo &effects,
                 TMEMWaitKind kind) {
    return pending.isIntersected(
        effects,
        [kind](Operation *, Operation *, bool isRead, bool, Allocation *) {
          return isRead != (kind == TMEMWaitKind::LOAD);
        },
        &allocation);
  }

  void waitBeforeBarrier(TMEMWaitKind kind, BlockInfo &pending,
                         OpBuilder *builder) {
    if (accesses(beforeBarrier, kind).empty())
      return;
    assert(lastBarrier && "publication needs a remembered rendezvous");
    builder->setInsertionPoint(lastBarrier);
    TMEMWaitOp::create(*builder, lastBarrier->getLoc(), kind);
    // The unconditional wait completes all preceding accesses of this kind.
    // Preserve later accesses, even if the same operation also has pending
    // accesses from an earlier loop iteration.
    accesses(beforeBarrier, kind).clear();
    accesses(pending, kind).clear();
    pending.join(afterBarrier);
  }

  void waitNow(Operation *op, TMEMWaitKind kind, BlockInfo &pending,
               OpBuilder *builder) {
    if (accesses(pending, kind).empty())
      return;
    builder->setInsertionPoint(op);
    TMEMWaitOp::create(*builder, op->getLoc(), kind);
    accesses(pending, kind).clear();
    accesses(afterBarrier, kind).clear();
  }

  void rememberBarrier(Operation *op, const BlockInfo &pending) {
    lastBarrier = op;
    beforeBarrier = pending;
    afterBarrier.sync();
  }

  void finishBlock(BlockInfo &pending, OpBuilder *builder) {
    // Complete accesses preceding lastBarrier before forgetting it.
    // Leave later accesses in pending for joins and backedges.
    for (TMEMWaitKind kind : {TMEMWaitKind::LOAD, TMEMWaitKind::STORE})
      waitBeforeBarrier(kind, pending, builder);
    lastBarrier = nullptr;
    afterBarrier.sync();
  }

  void update(Operation *op, MembarInfo *membarInfo, FuncMapT *,
              OpBuilder *builder) override {
    BlockInfo &pending = membarInfo->pending;
    if (auto wait = dyn_cast<TMEMWaitOp>(op)) {
      accesses(pending, wait.getKind()).clear();
      accesses(afterBarrier, wait.getKind()).clear();
      return;
    }

    BlockInfo effects = getTMemAccesses(op);
    // Include internal scratch barriers even without descriptor memory effects.
    auto stages = getLocalBarrierStages(op, &allocation);
    // Reuse this barrier for incoming hazards only if it precedes the op's TMEM
    // effects, or the op has no TMEM effects.
    bool beforeEffects =
        stages.beforeMemoryEffects ||
        (effects.syncReadSlices.empty() && effects.syncWriteSlices.empty());
    if (stages.hasBarrier() && beforeEffects)
      rememberBarrier(op, pending);

    if (isWaitBoundary(op)) {
      finishBlock(pending, builder);
      for (TMEMWaitKind kind : {TMEMWaitKind::LOAD, TMEMWaitKind::STORE})
        waitNow(op, kind, pending, builder);
    }

    bool loadHazard = hasHazard(pending, effects, TMEMWaitKind::LOAD) ||
                      hasHazard(beforeBarrier, effects, TMEMWaitKind::LOAD);
    bool storeHazard = hasHazard(pending, effects, TMEMWaitKind::STORE) ||
                       hasHazard(beforeBarrier, effects, TMEMWaitKind::STORE);
    if (loadHazard || storeHazard) {
      // Reuse the last barrier if no later access conflicts with this
      // operation. Otherwise, insert a wait and a new barrier before the
      // operation.
      if (lastBarrier &&
          !afterBarrier.isIntersected(effects, nullptr, &allocation)) {
        if (loadHazard)
          waitBeforeBarrier(TMEMWaitKind::LOAD, pending, builder);
        if (storeHazard)
          waitBeforeBarrier(TMEMWaitKind::STORE, pending, builder);
      } else {
        if (loadHazard)
          waitNow(op, TMEMWaitKind::LOAD, pending, builder);
        if (storeHazard)
          waitNow(op, TMEMWaitKind::STORE, pending, builder);
        builder->setInsertionPoint(op);
        insertBarrier(op, builder);
        rememberBarrier(op->getPrevNode(), pending);
      }
    }

    // Handle this operation's hazards before recording its internal barrier.
    // Waits go before the operation, so its own accesses remain in
    // afterBarrier.
    if (stages.hasBarrier() && !beforeEffects)
      rememberBarrier(op, pending);

    // Track loads and stores, including allocation initializers. An allocation
    // without an initializer preserves pending accesses to reused storage.
    auto kind = getTMemAccessKind(op);
    if (kind == TMemAccessKind::Load || kind == TMemAccessKind::Store) {
      pending.join(effects);
      afterBarrier.join(effects);
    }

    if (op->hasTrait<OpTrait::IsTerminator>() ||
        isa<RegionBranchOpInterface>(op))
      finishBlock(pending, builder);
  }

  // These snapshots track accesses before and after lastBarrier within one
  // virtual block. Only MembarInfo::pending flows through joins and backedges.
  Operation *lastBarrier = nullptr;
  BlockInfo beforeBarrier;
  BlockInfo afterBarrier;
};

} // namespace

struct TMemWaitInsertionPass
    : public impl::TritonNvidiaGPUTMemWaitInsertionPassBase<
          TMemWaitInsertionPass> {
  using impl::TritonNvidiaGPUTMemWaitInsertionPassBase<
      TMemWaitInsertionPass>::TritonNvidiaGPUTMemWaitInsertionPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod.walk([](Operation *op) {
              auto kind = getTMemAccessKind(op);
              return kind == TMemAccessKind::Load ||
                             kind == TMemAccessKind::Store
                         ? WalkResult::interrupt()
                         : WalkResult::advance();
            })
             .wasInterrupted())
      return;
    // Run after TMEM and shared-memory barrier insertion. Calls and returns
    // complete pending accesses, so this pass needs no call summaries.
    ModuleAllocation allocation(mod);
    ModuleMembarAnalysis analysis(allocation);
    analysis.runAnalysis<TMemWaitAnalysis>();
  }
};

struct TMemBarrierInsertionPass
    : public impl::TritonNvidiaGPUTMemBarrierInsertionPassBase<
          TMemBarrierInsertionPass> {
  using impl::TritonNvidiaGPUTMemBarrierInsertionPassBase<
      TMemBarrierInsertionPass>::TritonNvidiaGPUTMemBarrierInsertionPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ModuleAllocation allocation(mod);
    ModuleMembarAnalysis analysis(allocation, filterFn);
    analysis.runAnalysis<TMemBarrierAnalysis>();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
