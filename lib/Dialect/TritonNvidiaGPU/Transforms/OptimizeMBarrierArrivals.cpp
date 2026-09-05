#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <numeric>

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUOPTIMIZEMBARRIERARRIVALSPASS
#define GEN_PASS_DEF_TRITONNVIDIAGPUOPTIMIZESYNCHRONIZATIONPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// Follow views and WS captures without accepting descriptor escapes or joins
// with another allocation. Scalar ring-buffer indices may still be dynamic.
static LogicalResult
collectSoftwareBarriers(gpu::LocalAllocOp alloc,
                        SmallVectorImpl<InitBarrierOp> &inits,
                        SmallVectorImpl<ArriveBarrierOp> &arrivals) {
  SmallVector<Value> worklist{alloc.getResult()};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (auto partitions = dyn_cast<gpu::WarpSpecializePartitionsOp>(user)) {
        for (Region &region : partitions.getPartitionRegions())
          worklist.push_back(region.getArgument(use.getOperandNumber()));
        continue;
      }
      if (use.getOperandNumber() != 0)
        return failure();
      if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
        llvm::append_range(worklist, user->getResults());
        continue;
      }
      if (auto init = dyn_cast<InitBarrierOp>(user)) {
        inits.push_back(init);
      } else if (auto arrive = dyn_cast<ArriveBarrierOp>(user)) {
        if (arrive.getArrivalWarps())
          return failure();
        arrivals.push_back(arrive);
      } else if (!isa<WaitBarrierOp, InvalBarrierOp, gpu::LocalDeallocOp>(
                     user)) {
        return failure();
      }
    }
  }
  return success();
}

static void distributeArrivals(gpu::LocalAllocOp alloc) {
  SmallVector<InitBarrierOp> inits;
  SmallVector<ArriveBarrierOp> arrivals;
  if (failed(collectSoftwareBarriers(alloc, inits, arrivals)) ||
      inits.empty() || arrivals.empty())
    return;

  constexpr uint64_t maxCount = (1 << 20) - 1;
  uint64_t scale = 1;
  for (ArriveBarrierOp arrive : arrivals) {
    uint64_t numWarps = gpu::lookupNumWarps(arrive);
    if (numWarps <= 1)
      continue;
    // Choose the smallest common scale making each total count divisible by
    // its partition's warp count. Already-divisible counts need no scaling.
    uint64_t factor =
        numWarps / std::gcd(numWarps, uint64_t(arrive.getCount()));
    scale = std::lcm(scale, factor);
    if (scale > maxCount)
      return;
  }
  for (InitBarrierOp init : inits) {
    // Lowering counts all CTAs that contribute to the same physical barrier.
    uint64_t ctasPerBarrier =
        gpu::lookupNumCTAs(init) / init.getAlloc().getType().getNumElements();
    if (init.getCount() > maxCount / scale / ctasPerBarrier)
      return;
  }
  for (ArriveBarrierOp arrive : arrivals)
    if (arrive.getCount() > maxCount / scale)
      return;

  // Scaling every logical contribution also preserves bootstrap arrivals and
  // conditional paths. Membar still orders accesses to the barrier storage.
  OpBuilder builder(alloc.getContext());
  for (InitBarrierOp init : inits)
    init.setCountAttr(builder.getI32IntegerAttr(init.getCount() * scale));
  for (ArriveBarrierOp arrive : arrivals) {
    arrive.setCountAttr(builder.getI32IntegerAttr(arrive.getCount() * scale));
    int numWarps = gpu::lookupNumWarps(arrive);
    if (numWarps > 1)
      arrive.setArrivalWarpsAttr(builder.getI32IntegerAttr(numWarps));
  }
}

static bool isSynchronizationCandidate(Operation *op) {
  if (auto arrive = dyn_cast<ArriveBarrierOp>(op))
    return arrive.getArrivalWarps().has_value();
  auto barrier = dyn_cast<gpu::BarrierOp>(op);
  return barrier && barrier.isWarp() && barrier.hasLocal();
}

static void foldSynchronizedArrival(ArriveBarrierOp arrive) {
  // Outlined helpers do not carry the caller's partition offset into lowering.
  auto func = arrive->getParentOfType<triton::FuncOp>();
  if (!func || !triton::isKernel(func))
    return;
  auto numWarps = arrive.getArrivalWarps();
  if (!numWarps || *numWarps != gpu::lookupNumWarps(arrive))
    return;
  // Prior effects are synchronized across the region. One thread per routed
  // target can contribute the full count.
  arrive.removeArrivalWarpsAttr();
}

class SynchronizationAnalysis : public MembarAnalysis {
public:
  using MembarAnalysis::MembarAnalysis;

  void run(FunctionOpInterface function, FuncMapT &funcMap) {
    MembarAnalysis::run(function, funcMap);
    // Rewrite only after all predecessors and backedges have been analyzed.
    for (Operation *op : foldableOps) {
      if (auto arrive = dyn_cast<ArriveBarrierOp>(op))
        foldSynchronizedArrival(arrive);
      else
        cast<gpu::BarrierOp>(op).erase();
    }
  }

private:
  void update(Operation *op, MembarInfo *info, FuncMapT *funcMap,
              OpBuilder *) override {
    if (isSynchronizationCandidate(op)) {
      bool canFold = info->warpsSynced;
      if (isa<ArriveBarrierOp>(op))
        canFold &= info->allPathsFromEntrySynced && !info->pending.hasEffects();
      // A later predecessor or backedge can invalidate an earlier decision.
      if (canFold)
        foldableOps.insert(op);
      else
        foldableOps.erase(op);
    }

    if (auto barrier = dyn_cast<gpu::BarrierOp>(op)) {
      if (barrier.isWarp()) {
        // Later warp barriers already fold through warpsSynced.
        if (barrier.hasLocal() && !pendingWarp)
          pendingWarp = op;
      } else {
        if (barrier.hasLocal() && pendingWarp)
          foldableOps.insert(pendingWarp);
        pendingWarp = nullptr;
      }
    } else if (op->getNumRegions() || op->hasTrait<OpTrait::IsTerminator>() ||
               !isMemoryEffectFree(op) ||
               allocation.getBufferId(op) != Allocation::InvalidBufferId) {
      // Pure operations can still use shared scratch during lowering.
      pendingWarp = nullptr;
    }
    MembarAnalysis::update(op, info, funcMap, /*builder=*/nullptr);
  }

  Operation *pendingWarp = nullptr;
  llvm::SmallPtrSet<Operation *, 16> foldableOps;
};

struct OptimizeMBarrierArrivalsPass
    : impl::TritonNvidiaGPUOptimizeMBarrierArrivalsPassBase<
          OptimizeMBarrierArrivalsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    // Completing arrivals with an explicit count require SM90.
    if (computeCapability < 90)
      return;
    mod.walk(distributeArrivals);
  }
};

struct OptimizeSynchronizationPass
    : impl::TritonNvidiaGPUOptimizeSynchronizationPassBase<
          OptimizeSynchronizationPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod.walk([](Operation *op) {
              return isSynchronizationCandidate(op) ? WalkResult::interrupt()
                                                    : WalkResult::advance();
            })
             .wasInterrupted())
      return;
    // Coverage depends on scratch presence, not target-specific scratch sizes.
    ModuleAllocation allocation(mod);
    ModuleMembarAnalysis analysis(allocation);
    analysis.run<SynchronizationAnalysis>();
  }
};

} // namespace
} // namespace mlir::triton::nvidia_gpu
