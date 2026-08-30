#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "llvm/ADT/SmallSet.h"

#include <numeric>

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUOPTIMIZEMBARRIERARRIVALSPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// Follow views and WS captures without accepting descriptor escapes or joins
// with another allocation. Scalar ring-buffer indices may still be dynamic.
static LogicalResult
collectSoftwareBarriers(gpu::LocalAllocOp alloc,
                        SmallVectorImpl<InitBarrierOp> &inits,
                        SmallVectorImpl<ArriveBarrierOp> &arrivals) {
  SmallVector<Value> worklist{alloc.getResult()};
  llvm::SmallDenseSet<Value, 16> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
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
  bool hasMultipleWarps = false;
  for (ArriveBarrierOp arrive : arrivals) {
    uint64_t numWarps = gpu::lookupNumWarps(arrive);
    if (numWarps <= 1)
      continue;
    hasMultipleWarps = true;
    // Choose the smallest common scale making each total count divisible by
    // its partition's warp count. Already-divisible counts need no scaling.
    uint64_t factor =
        numWarps / std::gcd(numWarps, uint64_t(arrive.getCount()));
    factor /= std::gcd(scale, factor);
    if (scale > maxCount / factor)
      return;
    scale *= factor;
  }
  if (!hasMultipleWarps)
    return;
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

static void foldSynchronizedArrival(ArriveBarrierOp arrive) {
  // Outlined helpers do not carry the caller's partition offset into lowering.
  auto func = arrive->getParentOfType<triton::FuncOp>();
  if (!func || !triton::isKernel(func))
    return;
  auto numWarps = arrive.getArrivalWarps();
  if (!numWarps || *numWarps <= 1 || *numWarps != gpu::lookupNumWarps(arrive))
    return;
  auto barrier = dyn_cast_or_null<gpu::BarrierOp>(arrive->getPrevNode());
  if (!barrier || !barrier.hasLocal())
    return;
  // The preceding rendezvous publishes every warp's effects. Keep the total
  // count, but let one thread per routed target contribute it.
  arrive.removeArrivalWarpsAttr();
}

struct OptimizeMBarrierArrivalsPass
    : impl::TritonNvidiaGPUOptimizeMBarrierArrivalsPassBase<
          OptimizeMBarrierArrivalsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    // Completing arrivals with an explicit count require SM90.
    if (computeCapability < 90)
      return;
    if (foldAfterSync)
      mod.walk(foldSynchronizedArrival);
    else
      mod.walk(distributeArrivals);
  }
};

} // namespace
} // namespace mlir::triton::nvidia_gpu
