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

// Validate the full group before Membar; later subsets need no larger scale.
static uint64_t
getDistributionScale(const BarrierUses &uses,
                     llvm::function_ref<bool(Operation *)> usePerWarp) {
  if (uses.inits.empty() || (!llvm::any_of(uses.arrivals, usePerWarp) &&
                             !llvm::any_of(uses.expects, usePerWarp)))
    return 0;
  constexpr uint64_t maxCount = (1 << 20) - 1;
  uint64_t scale = 1;
  if (!uses.expects.empty()) {
    scale = gpu::lookupNumWarps(uses.expects.front());
    if (!uses.arrivals.empty() || scale == 1 ||
        gpu::lookupNumCTAs(uses.expects.front()) != 1)
      return 0;
    uint64_t maxInitCount = 0;
    for (InitBarrierOp init : uses.inits)
      maxInitCount = std::max(maxInitCount, uint64_t(init.getCount()));
    for (BarrierExpectOp expect : uses.expects) {
      // A phase has at most maxInitCount expectations. Bound the transaction
      // balance even when copies complete before other warps add their shares.
      if (gpu::lookupNumWarps(expect) != scale ||
          expect.getSize() > maxCount / maxInitCount ||
          expect.getSize() % scale != 0)
        return 0;
    }
  } else {
    if (uses.hasTMA)
      return 0;
    for (ArriveBarrierOp arrive : uses.arrivals) {
      if (!usePerWarp(arrive))
        continue;
      uint64_t numWarps = gpu::lookupNumWarps(arrive);
      // The smallest scale making each count divisible by its warp count.
      scale = std::lcm(
          scale, numWarps / std::gcd(numWarps, uint64_t(arrive.getCount())));
    }
    for (ArriveBarrierOp arrive : uses.arrivals)
      if (arrive.getCount() > maxCount / scale)
        return 0;
  }
  for (InitBarrierOp init : uses.inits) {
    // Lowering counts all CTAs that contribute to the same physical barrier.
    uint64_t ctasPerBarrier =
        uses.expects.empty() ? gpu::lookupNumCTAs(init) /
                                   init.getAlloc().getType().getNumElements()
                             : 1;
    if (init.getCount() > maxCount / scale / ctasPerBarrier)
      return 0;
  }
  return scale;
}

static bool isSynchronizationCandidate(Operation *op) {
  auto barrier = dyn_cast<gpu::BarrierOp>(op);
  return barrier && barrier.isWarp();
}

static bool isPerWarp(Operation *op) {
  auto barrier = dyn_cast<gpu::MBarrierOpInterface>(op);
  return barrier && barrier.isPerWarp();
}

// Find scalar arrivals and redundant warp barriers using Membar's state.
//   ttg.barrier local; ttg.barrier warp local -> ttg.barrier local
//   ttg.barrier warp local; ttg.barrier local -> ttg.barrier local
class SynchronizationAnalysis : public MembarAnalysis {
public:
  SynchronizationAnalysis(
      Allocation &allocation, MembarFilterFn filter,
      BufferRegionAnalysis &regions,
      llvm::SmallPtrSetImpl<Operation *> *scalarOps = nullptr)
      : MembarAnalysis(allocation, std::move(filter), regions),
        scalarOps(scalarOps) {}

  void run(FunctionOpInterface function, FuncMapT &funcMap) {
    MembarAnalysis::run(function, funcMap);
    // Rewrite only after all predecessors and backedges have been analyzed.
    for (Operation *op : foldableOps) {
      if (auto barrier = dyn_cast<gpu::BarrierOp>(op))
        barrier.erase();
      else
        scalarOps->insert(op);
    }
  }

private:
  void update(Operation *op, MembarInfo *info, FuncMapT *funcMap,
              OpBuilder *) override {
    bool candidate = scalarOps && isPerWarp(op);
    if (isSynchronizationCandidate(op) || candidate) {
      bool canFold = info->warpsSynced;
      if (candidate)
        canFold &= info->allPathsFromEntrySynced && !info->pending.hasEffects();
      // A later predecessor or backedge can invalidate an earlier decision.
      if (canFold)
        foldableOps.insert(op);
      else
        foldableOps.erase(op);
    }

    if (auto barrier = dyn_cast<gpu::BarrierOp>(op)) {
      if (barrier.isWarp()) {
        if (!firstWarpBarrier)
          firstWarpBarrier = op;
      } else {
        if (barrier.hasLocal() && firstWarpBarrier)
          foldableOps.insert(firstWarpBarrier);
        firstWarpBarrier = nullptr;
      }
    } else if (op->getNumRegions() || op->hasTrait<OpTrait::IsTerminator>() ||
               hasThreadEffects(op) ||
               allocation.getBufferId(op) != Allocation::InvalidBufferId) {
      firstWarpBarrier = nullptr;
    }
    MembarAnalysis::update(op, info, funcMap, /*builder=*/nullptr);
  }

  // First warp barrier that a later CTA barrier can make redundant.
  Operation *firstWarpBarrier = nullptr;
  llvm::SmallPtrSet<Operation *, 16> foldableOps;
  llvm::SmallPtrSetImpl<Operation *> *scalarOps;
};

struct OptimizeMBarrierArrivalsPass
    : impl::TritonNvidiaGPUOptimizeMBarrierArrivalsPassBase<
          OptimizeMBarrierArrivalsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod.walk([](Operation *op) {
              return isPerWarp(op) ? WalkResult::interrupt()
                                   : WalkResult::advance();
            })
             .wasInterrupted())
      return;
    ModuleAllocation allocation(mod);
    auto solver = createDataFlowSolver();
    auto *regions = solver->load<BufferRegionAnalysis>(
        BufferRegionAnalysis::Mode::AllMemory, &allocation);
    if (failed(solver->initializeAndRun(mod)))
      return signalPassFailure();

    // Keep the provisional flags fixed through every function and backedge.
    llvm::SmallPtrSet<Operation *, 16> scalarOps;
    MembarAnalysis::runModule(mod, [&](FunctionOpInterface function) {
      return SynchronizationAnalysis(*allocation.getFuncData(function), nullptr,
                                     *regions, &scalarOps);
    });

    auto forwarding = collectDescriptorForwarding(mod);
    mod.walk([&](gpu::LocalAllocOp alloc) {
      BarrierUses uses;
      if (failed(collectBarrierUses(alloc, forwarding, *regions, uses)) ||
          uses.inits.empty())
        return;
      for (ArriveBarrierOp arrive : uses.arrivals)
        if (scalarOps.contains(arrive))
          arrive.setPerWarp(false);
      for (BarrierExpectOp expect : uses.expects)
        if (scalarOps.contains(expect))
          expect.setPerWarp(false);

      uint64_t scale = getDistributionScale(uses, isPerWarp);
      if (!scale)
        return;
      // Scale every contribution, including bootstrap and conditional arrivals.
      OpBuilder builder(alloc.getContext());
      for (InitBarrierOp init : uses.inits)
        init.setCountAttr(builder.getI32IntegerAttr(init.getCount() * scale));
      for (ArriveBarrierOp arrive : uses.arrivals)
        arrive.setCountAttr(
            builder.getI32IntegerAttr(arrive.getCount() * scale));
      for (BarrierExpectOp expect : uses.expects) {
        if (expect.getPerWarp())
          continue;
        builder.setInsertionPointAfter(expect);
        // Register all bytes before supplying the remaining arrivals.
        auto arrive =
            ArriveBarrierOp::create(builder, expect.getLoc(), expect.getAlloc(),
                                    scale - 1, expect.getPred());
        arrive.setFromCTAAttr(expect.getFromCTAAttr());
      }
    });
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

void prepareMBarrierArrivals(ModuleOp mod, BufferRegionAnalysis &regions,
                             int computeCapability) {
  // Completing arrivals with an explicit count requires SM90.
  if (computeCapability < 90)
    return;
  auto forwarding = collectDescriptorForwarding(mod);
  mod.walk([&](gpu::LocalAllocOp alloc) {
    BarrierUses uses;
    if (failed(collectBarrierUses(alloc, forwarding, regions, uses)) ||
        !getDistributionScale(uses, [](Operation *) { return true; }))
      return;
    for (BarrierExpectOp expect : uses.expects)
      expect.setPerWarp(true);
    for (ArriveBarrierOp arrive : uses.arrivals)
      if (gpu::lookupNumWarps(arrive) > 1)
        arrive.setPerWarp(true);
  });
}

} // namespace mlir::triton::nvidia_gpu
