#include "triton/Analysis/Membar/MBarrierArrivals.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "triton/Analysis/Membar/Pass.h"
#include "triton/Analysis/Membar/TmemModel.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"

#include <numeric>

namespace mlir::triton::membar {
namespace {

using namespace nvidia_gpu;

struct BarrierUses {
  SmallVector<InitBarrierOp> inits;
  SmallVector<ArriveBarrierOp> arrivals;
  SmallVector<BarrierExpectOp> expects;
  bool hasTMA = false;
  bool closed = true;

  bool add(OpOperand &use) {
    Operation *op = use.getOwner();
    if (auto tma = dyn_cast<TMALoadLikeOpInterface>(op)) {
      // The same allocation used as a TMA payload is not a closed barrier.
      hasTMA = true;
      return &use == &tma.getBarrierMutable();
    }
    // A group with a per-warp contribution already has finalized counts.
    if (auto init = dyn_cast<InitBarrierOp>(op))
      inits.push_back(init);
    else if (auto arrive = dyn_cast<ArriveBarrierOp>(op);
             arrive && !arrive.getPerWarp())
      arrivals.push_back(arrive);
    else if (auto expect = dyn_cast<BarrierExpectOp>(op);
             expect && !expect.getPerWarp())
      expects.push_back(expect);
    else
      // Wait dependencies only keep their allocations live.
      return isa<WaitBarrierOp, InvalBarrierOp, gpu::LocalDeallocOp>(op);
    return true;
  }
};

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

static bool hasSingleOrigin(const RegionInfo &info) {
  return !info.views.empty() && llvm::all_of(info.views, [&](const auto &view) {
    return view.allocation == info.views.begin()->allocation;
  });
}

static SmallVector<BarrierUses> collectGroups(ModuleAllocation &allocation,
                                              BufferRegionAnalysis &regions,
                                              int computeCapability) {
  // Completing arrivals with an explicit count requires SM90.
  if (computeCapability < 90)
    return {};
  SmallVector<BarrierUses> groups;
  for (auto fn : allocation.getModuleOp().getOps<FunctionOpInterface>()) {
    auto &buffers = *allocation.getFuncData(fn);
    DenseMap<Allocation::BufferId, BarrierUses> uses;
    fn.walk([&](Operation *op) {
      // Aliases already follow views, block arguments and warp captures.
      if (op->hasTrait<OpTrait::MemDescViewTrait>() ||
          isa<arith::SelectOp, BranchOpInterface,
              gpu::WarpSpecializePartitionsOp, gpu::WarpYieldOp>(op))
        return;
      for (OpOperand &use : op->getOpOperands()) {
        if (!isa<gpu::MemDescType>(use.get().getType()))
          continue;
        // Alias IDs retain the known inputs to an unknown join. The region
        // also tracks descriptors from other allocation frames.
        auto ids = buffers.getAllBufferIdsWithAliases(use.get());
        for (auto id : ids) {
          auto &group = uses[id];
          if (group.closed)
            group.closed = ids.size() == 1 &&
                           hasSingleOrigin(regions.getRegionInfo(use.get())) &&
                           group.add(use);
        }
      }
    });
    for (auto &[id, group] : uses)
      if (group.closed && getDistributionScale(group, /*distributeAll=*/true))
        groups.push_back(std::move(group));
  }
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
  auto groups = collectGroups(allocation, regions, computeCapability);
  DenseSet<Operation *> distributable;
  for (const BarrierUses &uses : groups) {
    for (BarrierExpectOp expect : uses.expects)
      distributable.insert(expect);
    for (ArriveBarrierOp arrive : uses.arrivals)
      if (gpu::lookupNumWarps(arrive) > 1)
        distributable.insert(arrive);
  }
  auto mode = usesTensorMemory(mod) ? PlacementMode::SharedAndTensorMemory
                                    : PlacementMode::Shared;
  if (failed(placeBarriers(allocation, regions, std::move(filter), mode,
                           /*completionScope=*/nullptr,
                           std::move(distributable))))
    return failure();
  for (const BarrierUses &uses : groups)
    finalizeGroup(uses);
  return success();
}

} // namespace mlir::triton::membar
