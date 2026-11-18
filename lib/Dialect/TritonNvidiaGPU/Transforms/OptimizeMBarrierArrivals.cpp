#include "triton/Dialect/TritonNvidiaGPU/Transforms/OptimizeMBarrierArrivals.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "triton/Analysis/Membar/Pass.h"
#include "triton/Analysis/Membar/TmemModel.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"

#include <numeric>

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUOPTIMIZEMBARRIERARRIVALSPASS
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
      // A group with a per-warp contribution already has finalized counts.
      if (auto init = dyn_cast<InitBarrierOp>(user)) {
        uses.inits.push_back(init);
      } else if (auto arrive = dyn_cast<ArriveBarrierOp>(user);
                 arrive && !arrive.getPerWarp()) {
        uses.arrivals.push_back(arrive);
      } else if (auto expect = dyn_cast<BarrierExpectOp>(user);
                 expect && !expect.getPerWarp()) {
        uses.expects.push_back(expect);
      } else if (!isa<InvalBarrierOp, gpu::LocalDeallocOp>(user)) {
        return failure();
      }
    }
  }
  return success();
}

// Validate the full group before Membar; later subsets need no larger scale.
static uint64_t getDistributionScale(const BarrierUses &uses,
                                     bool distributeAll) {
  auto usePerWarp = [=](auto op) { return distributeAll || op.getPerWarp(); };
  if (uses.inits.empty() || (!llvm::any_of(uses.arrivals, usePerWarp) &&
                             !llvm::any_of(uses.expects, usePerWarp)))
    return 0;
  constexpr uint64_t maxCount = (1 << 20) - 1;
  uint64_t maxInitCount = 0;
  for (InitBarrierOp init : uses.inits) {
    // Lowering counts all CTAs that contribute to the same physical barrier.
    uint64_t ctasPerBarrier =
        gpu::lookupNumCTAs(init) / init.getAlloc().getType().getNumElements();
    maxInitCount =
        std::max(maxInitCount, uint64_t(init.getCount()) * ctasPerBarrier);
  }
  uint64_t scale = 1;
  if (!uses.expects.empty()) {
    scale = gpu::lookupNumWarps(uses.expects.front());
    if (!uses.arrivals.empty() || scale == 1)
      return 0;
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
  if (maxInitCount > maxCount / scale)
    return 0;
  return scale;
}

struct OptimizeMBarrierArrivalsPass
    : impl::TritonNvidiaGPUOptimizeMBarrierArrivalsPassBase<
          OptimizeMBarrierArrivalsPass> {
  using Base::Base;

  // Standalone pipelines can supply explicit warp barriers. Membar prunes
  // its own barriers and finalizes arrivals during placement.
  void runOnOperation() override {
    getOperation().walk([](Block *block) {
      gpu::BarrierOp previous;
      for (Operation &op : llvm::make_early_inc_range(*block)) {
        auto barrier = dyn_cast<gpu::BarrierOp>(op);
        if (!barrier || !barrier.hasLocal()) {
          if (previous &&
              (op.getNumRegions() || membar::SMemModel::hasThreadEffects(&op) ||
               defaultAllocationAnalysisScratchSizeFn(&op)))
            previous = {};
          continue;
        }
        if (previous && barrier.isWarp()) {
          barrier.erase();
          continue;
        }
        if (previous && previous.isWarp())
          previous.erase();
        previous = barrier;
      }
    });
  }
};

static SmallVector<BarrierUses> collectGroups(ModuleOp mod,
                                              BufferRegionAnalysis &regions,
                                              int computeCapability) {
  // Completing arrivals with an explicit count requires SM90.
  if (computeCapability < 90)
    return {};
  if (!mod.walk([](Operation *op) {
            return isa<ArriveBarrierOp, BarrierExpectOp>(op)
                       ? WalkResult::interrupt()
                       : WalkResult::advance();
          })
           .wasInterrupted())
    return {};
  auto forwarding = collectDescriptorForwarding(mod);
  SmallVector<BarrierUses> groups;
  mod.walk([&](gpu::LocalAllocOp alloc) {
    BarrierUses uses;
    if (succeeded(collectBarrierUses(alloc, forwarding, regions, uses)) &&
        getDistributionScale(uses, /*distributeAll=*/true))
      groups.push_back(std::move(uses));
  });
  return groups;
}

static void finalizeGroup(const BarrierUses &uses) {
  uint64_t scale = getDistributionScale(uses, /*distributeAll=*/false);
  if (scale <= 1)
    return;
  // Scale every contribution, including bootstrap and conditional arrivals.
  OpBuilder builder(uses.inits.front());
  for (InitBarrierOp init : uses.inits)
    init.setCountAttr(builder.getI32IntegerAttr(init.getCount() * scale));
  for (ArriveBarrierOp arrive : uses.arrivals)
    arrive.setCountAttr(builder.getI32IntegerAttr(arrive.getCount() * scale));
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
}

} // namespace

LogicalResult placeBarriersAndArrivals(ModuleAllocation &allocation,
                                       BufferRegionAnalysis &regions,
                                       MembarFilterFn filter,
                                       int computeCapability) {
  ModuleOp mod = allocation.getModuleOp();
  auto groups = collectGroups(mod, regions, computeCapability);
  DenseSet<Operation *> distributable;
  for (const BarrierUses &uses : groups) {
    for (BarrierExpectOp expect : uses.expects)
      distributable.insert(expect);
    for (ArriveBarrierOp arrive : uses.arrivals)
      if (gpu::lookupNumWarps(arrive) > 1)
        distributable.insert(arrive);
  }
  auto mode = membar::usesTensorMemory(mod)
                  ? membar::PlacementMode::SharedAndTensorMemory
                  : membar::PlacementMode::Shared;
  if (failed(membar::placeBarriers(allocation, regions, std::move(filter), mode,
                                   /*completionScope=*/nullptr,
                                   std::move(distributable))))
    return failure();
  for (const BarrierUses &uses : groups)
    finalizeGroup(uses);
  return success();
}

} // namespace mlir::triton::nvidia_gpu
