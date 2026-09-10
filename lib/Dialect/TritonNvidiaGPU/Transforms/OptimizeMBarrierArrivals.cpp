#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <numeric>

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUOPTIMIZEMBARRIERARRIVALSPASS
#define GEN_PASS_DEF_TRITONNVIDIAGPUOPTIMIZESYNCHRONIZATIONPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

using DescriptorForwarding = DenseMap<OpOperand *, SmallVector<Value>>;

// Enumerate forwarded uses, including captures owned by a parent operation.
// BufferRegionAnalysis supplies the converged origin proof for each value.
static DescriptorForwarding collectDescriptorForwarding(ModuleOp mod) {
  DescriptorForwarding forwarding;
  auto addEdge = [&](OpOperand &use, Value value) {
    if (!isa<gpu::MemDescType>(value.getType()))
      return;
    forwarding[&use].push_back(value);
  };
  mod.walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::MemDescViewTrait>()) {
      for (Value result : op->getResults())
        addEdge(op->getOpOperand(0), result);
    } else if (auto select = dyn_cast<arith::SelectOp>(op)) {
      addEdge(op->getOpOperand(1), select.getResult());
      addEdge(op->getOpOperand(2), select.getResult());
    } else if (auto branch = dyn_cast<BranchOpInterface>(op)) {
      for (auto [index, block] : llvm::enumerate(op->getSuccessors())) {
        SuccessorOperands operands = branch.getSuccessorOperands(index);
        for (BlockArgument arg : block->getArguments())
          addEdge(
              op->getOpOperand(operands.getOperandIndex(arg.getArgNumber())),
              arg);
      }
    } else if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
      RegionBranchSuccessorMapping mapping;
      branch.getSuccessorOperandInputMapping(mapping);
      for (auto &[use, values] : mapping)
        for (Value value : values)
          addEdge(*use, value);
    }
  });
  return forwarding;
}

struct BarrierUses {
  SmallVector<InitBarrierOp> inits;
  SmallVector<ArriveBarrierOp> arrivals;
  SmallVector<BarrierExpectOp> expects;
  bool hasTMA = false;
};

// Follow views and forwarded descriptors without accepting escapes or joins
// with another origin. Scalar ring-buffer indices may still be dynamic.
static LogicalResult collectBarrierUses(gpu::LocalAllocOp alloc,
                                        const DescriptorForwarding &forwarding,
                                        BufferRegionAnalysis &regions,
                                        BarrierUses &uses) {
  SmallVector<Value> worklist{alloc.getResult()};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    // Reject the whole allocation if any forwarded use has an unknown or mixed
    // origin, even when other arrivals still have a known footprint.
    const auto *footprint = regions.getFootprint(value);
    if (!footprint ||
        !llvm::all_of(footprint->regionInfo.views, [&](const auto &view) {
          return view.allocation == alloc.getOperation();
        }))
      return failure();
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      auto successors = forwarding.find(&use);
      if (successors != forwarding.end()) {
        auto effects = dyn_cast<MemoryEffectOpInterface>(user);
        assert((effects
                    ? effects.hasNoEffect()
                    : user->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) &&
               "descriptor forwarding operations must not have own effects");
        llvm::append_range(worklist, successors->second);
        continue;
      }
      // Wait dependencies only keep their allocations live.
      if (isa<WaitBarrierOp>(user))
        continue;
      if (auto tma = dyn_cast<TMALoadLikeOpInterface>(user)) {
        // The same allocation used as a TMA payload is not a closed barrier.
        if (&use != &tma.getBarrierMutable())
          return failure();
        uses.hasTMA = true;
        continue;
      }
      if (auto init = dyn_cast<InitBarrierOp>(user)) {
        uses.inits.push_back(init);
      } else if (auto arrive = dyn_cast<ArriveBarrierOp>(user)) {
        uses.arrivals.push_back(arrive);
      } else if (auto expect = dyn_cast<BarrierExpectOp>(user)) {
        uses.expects.push_back(expect);
      } else if (!isa<InvalBarrierOp, gpu::LocalDeallocOp>(user)) {
        return failure();
      }
    }
  }
  return success();
}

static void distributeExpectations(ArrayRef<InitBarrierOp> inits,
                                   ArrayRef<BarrierExpectOp> expects) {
  unsigned numWarps = gpu::lookupNumWarps(expects.front());
  if (numWarps == 1 || gpu::lookupNumCTAs(expects.front()) != 1)
    return;
  constexpr uint64_t maxCount = (1 << 20) - 1;
  uint64_t maxInitCount = 0;
  for (InitBarrierOp init : inits)
    maxInitCount = std::max(maxInitCount, uint64_t(init.getCount()));
  if (maxInitCount > maxCount / numWarps)
    return;
  for (BarrierExpectOp expect : expects) {
    // A phase has at most maxInitCount expectations. Bound the transaction
    // balance even when copies complete before other warps add their shares.
    // An already-distributed expectation must not scale initialization again.
    if (expect.getPerWarp() || gpu::lookupNumWarps(expect) != numWarps ||
        expect.getSize() > maxCount / maxInitCount ||
        expect.getSize() % numWarps != 0)
      return;
  }

  // Pending arrivals stay positive until every warp adds its byte share.
  // After the final arrival, the balance is total bytes minus completed bytes.
  OpBuilder builder(expects.front()->getContext());
  for (InitBarrierOp init : inits)
    init.setCountAttr(builder.getI32IntegerAttr(init.getCount() * numWarps));
  for (BarrierExpectOp expect : expects)
    expect.setPerWarp(true);
}

static void distributeArrivals(gpu::LocalAllocOp alloc,
                               const DescriptorForwarding &forwarding,
                               BufferRegionAnalysis &regions) {
  BarrierUses uses;
  if (failed(collectBarrierUses(alloc, forwarding, regions, uses)) ||
      uses.inits.empty())
    return;
  if (!uses.expects.empty()) {
    if (uses.arrivals.empty())
      distributeExpectations(uses.inits, uses.expects);
    return;
  }
  if (uses.hasTMA || uses.arrivals.empty())
    return;

  auto &inits = uses.inits;
  auto &arrivals = uses.arrivals;

  constexpr uint64_t maxCount = (1 << 20) - 1;
  uint64_t scale = 1;
  for (ArriveBarrierOp arrive : arrivals) {
    uint64_t numWarps = gpu::lookupNumWarps(arrive);
    // Choose the smallest common scale making each total count divisible by
    // its partition's warp count. Power-of-two widths bound the LCM by the
    // largest width, including the no-op factor for one warp.
    uint64_t factor =
        numWarps / std::gcd(numWarps, uint64_t(arrive.getCount()));
    scale = std::lcm(scale, factor);
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
      arrive.setPerWarp(true);
  }
}

static bool isSynchronizationCandidate(Operation *op) {
  if (auto arrive = dyn_cast<ArriveBarrierOp>(op))
    return arrive.getPerWarp();
  auto barrier = dyn_cast<gpu::BarrierOp>(op);
  return barrier && barrier.isWarp() && barrier.hasLocal();
}

static void foldSynchronizedArrival(ArriveBarrierOp arrive) {
  // Prior effects are synchronized across the region. One thread per routed
  // target can contribute the full count.
  arrive.setPerWarp(false);
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
               hasThreadEffects(op) ||
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
    if (computeCapability < 90)
      return;
    // Origin identity does not depend on target-specific scratch sizes.
    ModuleAllocation allocation(mod);
    auto solver = createDataFlowSolver();
    auto *regions = solver->load<BufferRegionAnalysis>(
        BufferRegionAnalysis::Mode::AllMemory, &allocation);
    if (failed(solver->initializeAndRun(mod)))
      return signalPassFailure();
    optimizeMBarrierArrivals(mod, *regions, computeCapability);
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

void optimizeMBarrierArrivals(ModuleOp mod, BufferRegionAnalysis &regions,
                              int computeCapability) {
  // Completing arrivals with an explicit count requires SM90.
  if (computeCapability < 90)
    return;
  auto forwarding = collectDescriptorForwarding(mod);
  mod.walk([&](gpu::LocalAllocOp alloc) {
    distributeArrivals(alloc, forwarding, regions);
  });
}

} // namespace mlir::triton::nvidia_gpu
