#include "triton/Analysis/Membar.h"
#include "MembarCFG.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include <algorithm>

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {

AllocationSlice::AllocationSlice(Value value,
                                 Interval<size_t> allocationInterval,
                                 Allocation::BufferId bufferId)
    : allocationInterval(allocationInterval),
      accessTy(cast<triton::gpu::MemDescType>(value.getType())),
      bufferId(bufferId) {
  // Get the memdesc_subslice information if present. If no subslice is
  // present the whole interval is accessed
  if (auto subslice = value.getDefiningOp<triton::gpu::MemDescSubsliceOp>()) {
    // The source supplies coordinates even if a preceding subslice has not
    // folded, or the descriptor is carried through control flow or a loop.
    subsliceOffsets = subslice.getOffsets();
    subsliceSource = subslice.getSrc();
  }
}

bool AllocationSlice::intersects(const AllocationSlice &other) const {
  // Disjoint intervals don't overlap
  if (!allocationInterval.intersects(other.allocationInterval))
    return false;

  // Physical footprints include every possible origin across loop iterations,
  // retaining CTA and memory-space identity.
  if (!triton::mayOverlap(physicalFootprint, other.physicalFootprint))
    return false;

  // Compare indices of the same source descriptor to prove disjoint slots,
  // including arguments without allocator IDs.
  if (bufferId == other.bufferId &&
      areBufferIndicesProvablyDifferent(*this, other))
    return false;

  // Compare logical offsets only for subslices of the same source descriptor.
  if (!subsliceSource || subsliceSource != other.subsliceSource)
    return true;

  auto shapeA = accessTy.getShape();
  auto shapeB = other.accessTy.getShape();
  // Chek if all subslice region dimensions have some intersection
  // [offsetA, offsetA + shape) and [offsetB, offsetB + other.shape)
  // If any dimension doesn't intersect, we are looking at disjoint subslices
  for (size_t i = 0; i < subsliceOffsets.size(); ++i) {
    int64_t startA = subsliceOffsets[i];
    int64_t endA = startA + shapeA[i];
    int64_t startB = other.subsliceOffsets[i];
    int64_t endB = startB + shapeB[i];

    // Is A completely before B? Is B completely before A? If so, disjoint
    if (endA <= startB || endB <= startA)
      return false;
  }

  // All dimensions of subslices have some intersection
  return true;
}

void AllocationSlice::print(raw_ostream &os) const {
  os << "interval=[" << allocationInterval.start() << ","
     << allocationInterval.end() << ")";

  if (bufferId != Allocation::InvalidBufferId)
    os << " buffer=" << bufferId;
  if (argumentIndex)
    os << " argument=" << *argumentIndex;

  os << " offsets=[";
  if (!subsliceOffsets.empty()) {
    llvm::interleaveComma(subsliceOffsets, os);
  } else {
    os << "unknown";
  }
  os << "]";

  os << " shape=";
  if (accessTy) {
    llvm::interleave(accessTy.getShape(), os, "x");
    os << " layout=" << accessTy.getEncoding();
  } else {
    os << "? layout=unknown";
  }
}

void BlockInfo::invalidateIterationInfo() {
  transformSlices([](AllocationSlice slice) {
    slice.invalidateIterationInfo();
    return SmallVector<AllocationSlice>{std::move(slice)};
  });
}

AllocationSlice AllocationSlice::translateToCallsite(
    CallOpInterface call, FunctionOpInterface callee,
    triton::BufferRegionAnalysis &regions) const {
  assert(!argumentIndex && "argument effects must be bound to call operands");
  AllocationSlice shifted = *this;
  // Unknown accesses span the whole frame; shifting their endpoint overflows.
  if (allocationInterval != Interval<size_t>{}) {
    size_t offset = regions.getCallOffset(call);
    shifted.allocationInterval = Interval<size_t>(
        allocationInterval.start() + offset, allocationInterval.end() + offset);
  }
  shifted.physicalFootprint =
      regions.translateToCallsite(physicalFootprint, call, callee);
  shifted.bufferId = Allocation::InvalidBufferId;
  shifted.invalidateIterationInfo();
  return shifted;
}

static ThreadIssuer joinIssuers(ThreadIssuer a, ThreadIssuer b) {
  if (a == ThreadIssuer::None)
    return b;
  if (b == ThreadIssuer::None || a == b)
    return a;
  return ThreadIssuer::Unknown;
}

void BlockInfo::joinThreadState(const BlockInfo &other) {
  const auto &sync = other.threadSync;
  threadSync.effectIssuer =
      joinIssuers(threadSync.effectIssuer, sync.effectIssuer);
  threadSync.completion = std::max(threadSync.completion, sync.completion);
  threadSync.peerWaitNeedsSync |= sync.peerWaitNeedsSync;
  const auto &demands = other.threadDemands;
  threadDemands.kind = std::max(threadDemands.kind, demands.kind);
  threadDemands.publicationIssuer =
      joinIssuers(threadDemands.publicationIssuer, demands.publicationIssuer);
  threadDemands.mayNotifyPeer |= demands.mayNotifyPeer;
}

bool BlockInfo::requiresThreadSync(const BlockInfo &pending,
                                   const BlockInfo &effects) {
  const auto &sync = pending.threadSync;
  const auto &demands = effects.threadDemands;
  if (sync.effectIssuer == ThreadIssuer::None ||
      demands.kind == ThreadDemandKind::None)
    return false;
  // The maps also include implicit scratch and translated callee accesses.
  bool hasSharedAccesses =
      !effects.syncReadSlices.empty() || !effects.syncWriteSlices.empty();
  return sync.completion == CompletionSync::All ||
         (sync.completion == CompletionSync::Shared &&
          (hasSharedAccesses || demands.kind == ThreadDemandKind::General)) ||
         (demands.publicationIssuer != ThreadIssuer::None &&
          (sync.effectIssuer == ThreadIssuer::Unknown ||
           sync.effectIssuer != demands.publicationIssuer)) ||
         // A notification can let a peer advance a pending wait's phase.
         // Private counter updates are already ordered by the slice hazards.
         (sync.peerWaitNeedsSync && demands.mayNotifyPeer);
}

void MembarAnalysis::syncIfNeeded(Operation *op, const BlockInfo &effects,
                                  MembarInfo *membarInfo, OpBuilder *builder,
                                  bool cluster) {
  if (!builder)
    return;
  auto &pending = membarInfo->pending;
  if (!BlockInfo::requiresThreadSync(pending, effects) &&
      !pending.isIntersected(effects, filter, &allocation, sliceFilter))
    return;
  // The barrier clears incoming state. The operation's own effects still
  // follow it, including a scratch write that conflicts with a pending read.
  builder->setInsertionPoint(op);
  insertBarrier(op, builder, cluster);
  membarInfo->sync();
}

void MembarAnalysis::insertBarrier(Operation *op, OpBuilder *builder,
                                   bool cluster) {
  OpBuilder::InsertionGuard g(*builder);
  if (cluster)
    ttng::ClusterBarrierOp::create(*builder, op->getLoc());
  else
    triton::gpu::BarrierOp::create(*builder, op->getLoc(),
                                   triton::gpu::AddrSpace::Local);
}

void MembarAnalysis::updateSuccessor(Operation *terminator, Block *successor,
                                     MembarInfo *membarInfo) {
  if (shared.getBufferIndexAnalysis().isBackedgeSuccessor(terminator,
                                                          successor)) {
    membarInfo->pending.invalidateIterationInfo();
    membarInfo->entryBlockInfo.invalidateIterationInfo();
  }
}

void MembarAnalysis::updateExitState(MembarInfo *membarInfo) {
  // Function summaries are reused at every call site, so per-function SSA
  // index identity is no longer meaningful.
  membarInfo->pending.invalidateIterationInfo();
  membarInfo->entryBlockInfo.invalidateIterationInfo();
}

void MembarAnalysis::update(Operation *op, MembarInfo *membarInfo,
                            FuncMapT *funcMap, OpBuilder *builder) {
  // Finish completions that block every memory demand once before a loop.
  // Shared-copy completions can still overlap independent global reads.
  if (builder && shared.getBufferIndexAnalysis().entersLoop(op) &&
      membarInfo->pending.threadSync.completion == CompletionSync::All) {
    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    membarInfo->sync();
  }
  updateMemoryEffects(op, membarInfo, funcMap, builder);
}

void MembarAnalysis::updateMemoryEffects(
    Operation *op, MembarInfo *membarInfo, FuncMapT *funcMap,
    OpBuilder *builder, std::optional<triton::BarrierStages> clusterStages) {
  bool cluster = clusterStages.has_value();
  // A call's allocation belongs to the callee and is translated separately.
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    auto callee = cast<FunctionOpInterface>(call.resolveCallable());
    auto summary =
        shared.translateCallSummary(call, callee, funcMap->at(callee));
    syncIfNeeded(op, summary.entryBlockInfo, membarInfo, builder, cluster);
    membarInfo->applyCallSummary(summary);
    return;
  }

  namespace mb = triton::membar;
  auto applyFixed = [&](unsigned mask) {
    if (mask & mb::bit(mb::Need::RegionOrder))
      membarInfo->sync();
    else if (mask & mb::bit(mb::Need::WarpOrder))
      membarInfo->syncWarps();
  };
  mb::OpModel model;
  shared.describe(op, model, triton::getMemoryAccesses(op), clusterStages);
  // The flow visits guarded operations separately. Join pending completions
  // and peer-visible waits before the lanes split.
  if (!cluster && isa<cf::CondBranchOp, scf::IfOp>(op) &&
      op->getOperand(0).getDefiningOp<triton::gpu::ThreadPredicateOp>()) {
    model.sharedIn.threadDemands.kind = ThreadDemandKind::General;
    model.sharedIn.threadDemands.mayNotifyPeer = true;
  }
  // Model a leading barrier before handling the operation's effects.
  applyFixed(model.fixedBefore);
  syncIfNeeded(op, model.sharedIn, membarInfo, builder, cluster);
  if (builder && model.needsWarp && !membarInfo->warpsSynced) {
    builder->setInsertionPoint(op);
    triton::gpu::BarrierOp::create(*builder, op->getLoc(),
                                   triton::gpu::AddrSpace::Local,
                                   triton::gpu::BarrierScope::Warp);
    membarInfo->syncWarps();
  }
  if (model.fixedInside) {
    // Preserve effects reachable from entry before the scratch rendezvous.
    if (!membarInfo->allPathsFromEntrySynced)
      membarInfo->entryBlockInfo.join(model.sharedIn);
    applyFixed(model.fixedInside);
  }
  // Even if a barrier was inserted, retain the operation's outgoing effects.
  membarInfo->addBlockInfo(model.sharedOut);
  // Model a trailing barrier after handling the operation's effects.
  applyFixed(model.fixedAfter);
}

void ModuleMembarAnalysis::run() { run<MembarAnalysis>(); }

} // namespace mlir
