#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "TritonNVIDIAGPUToLLVM/Utility.h"

#include "Allocation.h"
#include "TargetInfo.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Membar/MBarrierArrivals.h"
#include "triton/Analysis/Membar/Pass.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONNVIDIAGPUMEMBAR
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace mlir::triton {
namespace {

namespace ttng = mlir::triton::nvidia_gpu;

struct TritonNvidiaGPUMembar
    : public impl::TritonNvidiaGPUMembarBase<TritonNvidiaGPUMembar> {
  using TritonNvidiaGPUMembarBase::TritonNvidiaGPUMembarBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    NVIDIA::TargetInfo targetInfo(computeCapability, ptxVersion);
    ModuleAllocation allocation(
        mod, ttng::getNvidiaAllocationAnalysisScratchSizeFn(targetInfo));
    auto solver = createDataFlowSolver();
    auto *regions = solver->load<triton::BufferRegionAnalysis>(
        triton::BufferRegionAnalysis::Mode::AllMemory, &allocation);
    if (failed(solver->initializeAndRun(mod))) {
      signalPassFailure();
      return;
    }

    // Synchronization insertion and arrival attributes preserve this geometry.
    if (failed(membar::placeClusterBarriers(allocation, *regions)))
      return signalPassFailure();
    NVIDIA::SMemOrdering ordering(allocation);
    if (failed(membar::placeBarriersAndArrivals(allocation, *regions, &ordering,
                                                computeCapability)))
      signalPassFailure();
  }
};

} // namespace

namespace NVIDIA {

using membar::OrderingRule;
using membar::SMemAccess;
using membar::ThreadEffects;

SMemOrdering::SMemOrdering(ModuleAllocation &module) {
  for (auto kernel : module.getRoots()) {
    if (kernel.isExternal())
      continue;
    Block &entry = kernel.getFunctionBody().front();
    for (auto alloc : entry.getOps<gpu::LocalAllocOp>())
      entryKernels[alloc] = kernel;
  }
}

void SMemOrdering::classify(SMemAccess &access, Operation *op,
                            const MemoryAccess &effect, ThreadEffects threads,
                            Allocation &) {
  // Init and inval are generic writes by the same elected lane. They
  // change the counter's lifetime and must also be ordered with waits.
  if (isa<ttng::InitBarrierOp, ttng::InvalBarrierOp>(op)) {
    access.rule = OrderingRule::CounterInit;
    return;
  }
  bool completesTransactions =
      isa<ttng::AsyncLoadOpInterface, ttng::CLCTryCancelOp,
          ttng::AsyncSharedStoreOp>(op);
  if (effect.isShared(gpu::SharedKind::Barrier)) {
    bool wait = isa<ttng::WaitBarrierOp>(op);
    bool release = threads.joins & ThreadEffects::AnyIssuer;
    auto copy = dyn_cast<ttng::AsyncCopyMbarrierArriveOp>(op);
    if (wait)
      access.rule = OrderingRule::CounterWait;
    else if (release || completesTransactions ||
             (copy && copy.getNoIncrement()))
      access.rule = OrderingRule::CounterSignal;
    else
      access.rule = OrderingRule::Counter;

    // A direct arrival or expectation includes a local contribution.
    // Routed issuers and broadcast waits can omit participating CTAs.
    bool direct = release && (threads.pending == ThreadEffects::Thread0 ||
                              threads.pending == ThreadEffects::WarpLeaders);
    if ((wait || direct) &&
        !ttng::hasCGABroadcast(cast<gpu::MemDescType>(effect.value.getType())))
      access.key = counterKeys.try_emplace(effect.value, counterKeys.size() + 1)
                       .first->second;
    return;
  }

  if (completesTransactions)
    access.rule = OrderingRule::AsyncPayload;
  if (auto atomic = dyn_cast<gpu::LocalAtomicScatterRMWOp>(op);
      atomic && atomic.isCommutative()) {
    access.rule = OrderingRule::CommutativeAtomic;
    auto key = std::make_pair(unsigned(atomic.getAtomicRmwOp()),
                              atomic.getDst().getType().getElementType());
    access.key =
        atomicKeys.try_emplace(key, atomicKeys.size() + 1).first->second;
  }
}

// Allocator reuse can give unrelated allocations the same physical address.
bool SMemOrdering::hasOnlySourceAliases(const SMemAccess &source,
                                        const SMemAccess &sink,
                                        Allocation &allocation) const {
  auto *before = source.slice.physicalFootprint;
  auto *after = sink.slice.physicalFootprint;
  if (!before || !after)
    return false;
  bool local = source.slice.getBufferId() != Allocation::InvalidBufferId &&
               sink.slice.getBufferId() != Allocation::InvalidBufferId;
  bool overlaps = false;
  for (const auto &a : before->regionInfo.views) {
    for (const auto &b : after->regionInfo.views) {
      if (a.allocationFrame != b.allocationFrame)
        return false;
      if (!a.region.intersects(b.region))
        continue;
      if (!a.allocation || a.allocation != b.allocation)
        return false;
      if (!local &&
          entryKernels.lookup(a.allocation) != allocation.getOperation())
        return false;
      overlaps = true;
    }
  }
  return overlaps;
}

bool SMemOrdering::canSkip(const SMemAccess &source, const SMemAccess &sink,
                           Allocation &allocation) const {
  switch (source.rule) {
  case OrderingRule::AsyncPayload:
    // Consumers of live destinations must await transaction completion.
    return hasOnlySourceAliases(source, sink, allocation);
  case OrderingRule::CommutativeAtomic:
    return sink.rule == source.rule && sink.key == source.key;
  case OrderingRule::CounterInit:
    return sink.rule == source.rule;
  case OrderingRule::CounterSignal:
    // Contributions to a live counter commute with one another.
    return sink.rule == source.rule || sink.rule == OrderingRule::CounterWait;
  case OrderingRule::Counter:
    return sink.rule == OrderingRule::CounterWait;
  case OrderingRule::CounterWait:
    if (sink.rule == OrderingRule::CounterWait)
      return true;
    // Publication orders the wait before each issuer's contribution.
    // With one issuer per warp, the phase needs every warp to arrive.
    return source.key && sink.rule == OrderingRule::CounterSignal &&
           source.key == sink.key;
  default:
    return false;
  }
}

} // namespace NVIDIA
} // namespace mlir::triton
