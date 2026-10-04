#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.h"
#include "triton/Analysis/Alias.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/MBarrierUtilities.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

namespace {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

// Returns whether an operation's tracked access touches distributed shared
// memory across CTAs.
// op: The operation associated with the tracked access.
// isRead: Whether the access is recorded in BlockInfo::syncReadSlices rather
// than BlockInfo::syncWriteSlices.
bool isDistributedMultiCTAOp(Operation *op, bool isRead) {
  // Scratch writes are CTA-local. When the scratch spans CTAs, only its read
  // phase accesses another CTA's shared memory.
  if (hasCrossCTAScratch(op) && isRead)
    return true;

  if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
    return isCrossCTALoadStore(load.getSrc().getType(), load.getType());
  } else if (auto store = dyn_cast<ttg::LocalStoreOp>(op)) {
    return isCrossCTALoadStore(store.getDst().getType(),
                               store.getSrc().getType());
  } else if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op)) {
    return alloc.getSrc() &&
           isCrossCTALoadStore(alloc.getType(), alloc.getSrc().getType());
  } else if (auto gather = dyn_cast<ttg::LocalGatherOp>(op)) {
    return isCrossCTAGatherScatter(gather.getSrc().getType(), gather.getType(),
                                   gather.getAxis());
  } else if (auto scatter = dyn_cast<ttg::LocalScatterOp>(op)) {
    return isCrossCTAGatherScatter(scatter.getDst().getType(),
                                   scatter.getValues().getType(),
                                   scatter.getAxis());
  } else if (auto atomic = dyn_cast<ttg::LocalAtomicScatterRMWOp>(op)) {
    return isCrossCTAGatherScatter(atomic.getDst().getType(),
                                   atomic.getValues().getType(),
                                   atomic.getAxis());
  }

  if (isa<ttng::CLCTryCancelOp>(op)) {
    return ttg::lookupNumCTAs(op) > 1;
  } else if (auto store = dyn_cast<ttng::AsyncSharedStoreOp>(op)) {
    return isCrossCTALoadStore(store.getDst().getType(),
                               store.getSrc().getType());
  } else if (isa<ttng::TMEMCopyOp>(op)) {
    return ttng::getModuleTwoCTAs(op);
  } else if (auto tma = dyn_cast<ttng::TMALoadLikeOpInterface>(op)) {
    return tma.getMulticast();
  } else if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(op)) {
    return arrive.isMulticast();
  }
  return hasTCGen5CommitCrossCTA(op);
}

bool isPreAllocAliasSliceFilter(const AllocationSlice &lhsSlice,
                                const AllocationSlice &rhsSlice,
                                bool /*lhsIsRead*/, bool /*rhsIsRead*/,
                                Allocation *allocation) {
  // Argument effects are checked after binding to the caller's allocation.
  if (lhsSlice.argumentIndex || rhsSlice.argumentIndex)
    return true;
  auto bufferId = lhsSlice.getBufferId();
  return bufferId != Allocation::InvalidBufferId &&
         bufferId == rhsSlice.getBufferId() &&
         allocation->isExplicitBuffer(bufferId);
}

bool valueAliasesRoots(Value value, const llvm::DenseSet<Value> &roots,
                       SharedMemoryAliasAnalysis &aliases) {
  if (!value)
    return false;
  auto *lattice = aliases.getLatticeElement(value);
  return lattice &&
         llvm::any_of(lattice->getValue().getAllocs(),
                      [&](Value root) { return roots.contains(root); });
}

bool requiresCrossCTAMBarrierInitSync(ttng::InitBarrierOp initBarrierOp,
                                      FunctionOpInterface funcOp,
                                      SharedMemoryAliasAnalysis &aliases,
                                      int numCTAs) {
  // Module-level aliases retain incoming allocation roots across calls, so a
  // consumer in a nested helper can require publication of a caller's init.
  const auto &roots = aliases.getLatticeElement(initBarrierOp.getBarrier())
                          ->getValue()
                          .getAllocs();
  return mlir::triton::nvidia_gpu::requiresCrossCTAMBarrierInitSync(
      funcOp->getParentOfType<ModuleOp>(), initBarrierOp.getBarrier(), numCTAs,
      [&](Value value) { return valueAliasesRoots(value, roots, aliases); });
}

bool nestedOpUsesTrackedMBarrier(Operation *op,
                                 const llvm::DenseSet<Value> &roots,
                                 SharedMemoryAliasAnalysis &aliases,
                                 SymbolTableCollection &symbols,
                                 llvm::SmallPtrSetImpl<Operation *> &visited) {
  if (isa<ttng::InitBarrierOp, ttg::LocalAllocOp>(op))
    return false;

  // Calls carry their barrier effects through formal arguments. The caller's
  // publication must dominate the call, not just its local mbarrier users.
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    if (!llvm::any_of(call.getArgOperands(), [&](Value value) {
          return valueAliasesRoots(value, roots, aliases);
        }))
      return false;
    auto callee = dyn_cast_or_null<FunctionOpInterface>(
        call.resolveCallableInTable(&symbols));
    if (!callee || callee.isExternal())
      return true;
    if (!visited.insert(callee.getOperation()).second)
      return false;
    // A descriptor-only helper can run before init. Anchor only calls that
    // actually access the barrier, including accesses in transitive callees.
    return callee
        ->walk([&](Operation *nested) {
          return nestedOpUsesTrackedMBarrier(nested, roots, aliases, symbols,
                                             visited)
                     ? WalkResult::interrupt()
                     : WalkResult::advance();
        })
        .wasInterrupted();
  }

  if (auto memEffects = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
    memEffects.getEffects(effects);
    for (const auto &effect : effects) {
      Value value = effect.getValue();
      if (valueAliasesRoots(value, roots, aliases))
        return true;
    }
  }
  return false;
}

bool opUsesTrackedMBarrier(Operation *op, const llvm::DenseSet<Value> &roots,
                           SharedMemoryAliasAnalysis &aliases,
                           SymbolTableCollection &symbols,
                           DominanceInfo &domInfo, Operation *afterInit) {
  llvm::SmallPtrSet<Operation *, 8> visited;
  return op
      ->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
        // Per-init fallback must not include an earlier lifecycle of the same
        // allocation. Test the actual nested use, not its enclosing region op.
        if (afterInit && !domInfo.dominates(afterInit, nestedOp))
          return WalkResult::advance();
        if (nestedOpUsesTrackedMBarrier(nestedOp, roots, aliases, symbols,
                                        visited))
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

LogicalResult insertCrossCTAMBarrierInitSyncForFunction(
    FunctionOpInterface funcOp, SharedMemoryAliasAnalysis &aliases, int numCTAs,
    OpBuilder &builder, ttng::InitBarrierOp onlyInit = {}) {
  if (!funcOp || funcOp->getNumRegions() != 1) {
    return funcOp.emitOpError(
        "cross-CTA mbarrier init sync insertion requires a single function "
        "top-level region");
  }
  Region &topLevelRegion = funcOp->getRegion(0);
  llvm::SetVector<Operation *> crossCTAInitAnchors;
  SmallVector<ttng::InitBarrierOp> crossCTAInits;
  llvm::DenseSet<Value> trackedBarrierRoots;

  // Find all cross-CTA mbarrier.init ops and map each
  // one to the containing top-level op that bounds the insertion window.
  funcOp.walk([&](ttng::InitBarrierOp initBarrierOp) {
    if (onlyInit && initBarrierOp != onlyInit)
      return;
    if (!requiresCrossCTAMBarrierInitSync(initBarrierOp, funcOp, aliases,
                                          numCTAs))
      return;
    Operation *topLevelAnchor =
        topLevelRegion.findAncestorOpInRegion(*initBarrierOp.getOperation());
    assert(topLevelAnchor && "init op must be inside the function region");
    crossCTAInitAnchors.insert(topLevelAnchor);
    crossCTAInits.push_back(initBarrierOp);
    const auto &roots = aliases.getLatticeElement(initBarrierOp.getBarrier())
                            ->getValue()
                            .getAllocs();
    trackedBarrierRoots.insert(roots.begin(), roots.end());
  });
  // Nothing to do
  if (crossCTAInitAnchors.empty())
    return success();

  // Cluster barriers remain unsupported in retained noinline helpers. Diagnose
  // the required synchronization before constructing an illegal barrier op.
  if (!isKernel(funcOp)) {
    auto noinline = funcOp->getAttrOfType<BoolAttr>("noinline");
    if (noinline && noinline.getValue())
      return funcOp.emitOpError(
          "cross-CTA mbarrier initialization in noinline functions is not "
          "supported");
  }

  // Prefer one rendezvous for a group of initializations. Separate lifecycles
  // can have a use before the next init and therefore need separate windows.
  auto failOrSplit = [&](StringRef message) -> LogicalResult {
    if (!onlyInit && crossCTAInits.size() > 1) {
      for (auto init : crossCTAInits)
        if (failed(insertCrossCTAMBarrierInitSyncForFunction(
                funcOp, aliases, numCTAs, builder, init)))
          return failure();
      return success();
    }
    return funcOp.emitOpError(message);
  };

  llvm::SetVector<Operation *> trackedUseAnchors;
  SymbolTableCollection symbols;
  DominanceInfo domInfo(funcOp);
  // A linear entry-block reinit cannot reach uses before itself. For other
  // blocks, keep all uses: a backedge or reconvergent path can reach a use
  // that is not dominated by this init, and still needs publication.
  Operation *afterInit = nullptr;
  if (onlyInit && onlyInit->getBlock() == &topLevelRegion.front() &&
      onlyInit->getBlock()->getPredecessors().empty())
    afterInit = onlyInit.getOperation();
  for (Block &block : topLevelRegion) {
    for (Operation &op : block) {
      if (opUsesTrackedMBarrier(&op, trackedBarrierRoots, aliases, symbols,
                                domInfo, afterInit))
        trackedUseAnchors.insert(&op);
    }
  }
  if (trackedUseAnchors.empty()) {
    return funcOp.emitOpError("found at least one mbarrier.init op but could "
                              "not find any mbarrier use");
  }

  // Find the earliest insertion point that postdominates every tracked init.
  PostDominanceInfo postDomInfo(funcOp);
  llvm::SmallPtrSet<Block *, 8> initBlocks;
  for (Operation *crossCTAInitAnchor : crossCTAInitAnchors)
    initBlocks.insert(crossCTAInitAnchor->getBlock());
  Block *firstInsertionBlock =
      postDomInfo.findNearestCommonDominator(initBlocks);
  if (!firstInsertionBlock) {
    return failOrSplit(
        "could not find a common post-dominating insertion block for "
        "cross-CTA mbarrier.init");
  }

  Operation *lastInitInInsertionBlock = nullptr;
  for (Operation *crossCTAInitAnchor : crossCTAInitAnchors) {
    if (crossCTAInitAnchor->getBlock() != firstInsertionBlock)
      continue;
    if (!lastInitInInsertionBlock ||
        lastInitInInsertionBlock->isBeforeInBlock(crossCTAInitAnchor)) {
      lastInitInInsertionBlock = crossCTAInitAnchor;
    }
  }
  Operation *firstInsertionAnchor =
      lastInitInInsertionBlock ? lastInitInInsertionBlock->getNextNode()
                               : &firstInsertionBlock->front();

  // Find the latest insertion point that still dominates every tracked use.
  llvm::SmallPtrSet<Block *, 8> useBlocks;
  for (Operation *trackedUseAnchor : trackedUseAnchors)
    useBlocks.insert(trackedUseAnchor->getBlock());
  Block *lastInsertionBlock = domInfo.findNearestCommonDominator(useBlocks);
  if (!lastInsertionBlock) {
    return failOrSplit(
        "could not find a common insertion block that dominates all tracked "
        "mbarrier uses");
  }

  Operation *firstTrackedUseInInsertionBlock = nullptr;
  for (Operation *trackedUseAnchor : trackedUseAnchors) {
    if (trackedUseAnchor->getBlock() != lastInsertionBlock)
      continue;
    if (!firstTrackedUseInInsertionBlock ||
        trackedUseAnchor->isBeforeInBlock(firstTrackedUseInInsertionBlock)) {
      firstTrackedUseInInsertionBlock = trackedUseAnchor;
    }
  }
  Operation *lastInsertionAnchor = firstTrackedUseInInsertionBlock
                                       ? firstTrackedUseInInsertionBlock
                                       : lastInsertionBlock->getTerminator();

  if (!domInfo.dominates(firstInsertionAnchor, lastInsertionAnchor)) {
    return failOrSplit(
        "could not find an insertion point between cross-CTA mbarrier.init "
        "ops and tracked mbarrier uses");
  }

  // Reuse the latest cluster barrier that lies between the init-side and
  // use-side insertion boundaries.
  ttng::ClusterBarrierOp reusedClusterBarrier;
  for (Block &block : topLevelRegion) {
    for (Operation &op : block) {
      auto clusterBarrier = dyn_cast<ttng::ClusterBarrierOp>(&op);
      if (!clusterBarrier)
        continue;
      if (!postDomInfo.postDominates(clusterBarrier.getOperation(),
                                     firstInsertionAnchor))
        continue;
      if (!domInfo.dominates(clusterBarrier.getOperation(),
                             lastInsertionAnchor))
        continue;
      if (!reusedClusterBarrier ||
          domInfo.properlyDominates(reusedClusterBarrier.getOperation(),
                                    clusterBarrier.getOperation())) {
        reusedClusterBarrier = clusterBarrier;
      }
    }
  }

  OpBuilder::InsertionGuard guard(builder);
  Operation *fenceInsertionPoint =
      reusedClusterBarrier && reusedClusterBarrier.getRelaxed()
          ? reusedClusterBarrier.getOperation()
          : lastInsertionAnchor;
  builder.setInsertionPoint(fenceInsertionPoint);
  Location loc = lastInitInInsertionBlock
                     ? lastInitInInsertionBlock->getLoc()
                     : crossCTAInitAnchors.front()->getLoc();
  ttng::FenceMBarrierInitReleaseClusterOp::create(builder, loc);
  if (!reusedClusterBarrier)
    ttng::ClusterBarrierOp::create(builder, loc, /*relaxed=*/true);
  return success();
}

class ClusterBarrierAnalysis : public MembarAnalysis {
public:
  ClusterBarrierAnalysis(Allocation &allocation, MembarFilterFn filter,
                         BufferRegionAnalysis &regions)
      : MembarAnalysis(allocation, std::move(filter), regions,
                       isPreAllocAliasSliceFilter,
                       AccessMode::AllocatorAliasesOnly) {}

private:
  llvm::SmallPtrSet<Operation *, 4> returnsWithExitBarrier;

  BarrierStages getBarrierStages(Operation *op) override {
    BarrierStages stages;
    if (auto barrier = dyn_cast<ttng::ClusterBarrierOp>(op))
      stages.beforeMemoryEffects = !barrier.getRelaxed();
    // Distributed scratch synchronizes between its write and read phases.
    stages.betweenMemoryEffects = isDistributedMultiCTAOp(op, /*isRead=*/true);
    return stages;
  }

  void update(Operation *op, MembarInfo *membarInfo, FuncMapT *funcMap,
              OpBuilder *builder) override {
    if (op->hasTrait<OpTrait::ReturnLike>() &&
        isa<FunctionOpInterface>(op->getParentOp())) {
      // Any path from distributed shared memory use to kernel exit must include
      // a cluster barrier. Conservatively insert it because warp-specialized
      // memory effects are not fully modeled.
      if (isKernel(cast<FunctionOpInterface>(op->getParentOp()))) {
        // The solver may revisit this return before convergence.
        if (returnsWithExitBarrier.insert(op).second) {
          builder->setInsertionPoint(op);
          insertBarrier(op, builder, /*cluster=*/true);
        }
        membarInfo->sync();
      }
      return;
    }
    updateMemoryEffects(op, membarInfo, funcMap, builder, /*cluster=*/true);
  }
};

} // namespace

void runClusterBarrierInsertion(ModuleAllocation &moduleAllocation,
                                int computeCapability) {
  ModuleOp mod = moduleAllocation.getModuleOp();
  if (computeCapability < 90)
    return;
  if (ttg::TritonGPUDialect::getNumCTAs(mod) == 1)
    return;

  MembarFilterFn filterFn = [](Operation *lhs, Operation *rhs, bool lhsIsRead,
                               bool rhsIsRead, Allocation * /*allocation*/) {
    // Filter ops that do not touch distributed shared memory. Whether the
    // aliasing was already present in TTGIR is handled per-allocation slice.
    bool lhsDist = isDistributedMultiCTAOp(lhs, lhsIsRead);
    bool rhsDist = isDistributedMultiCTAOp(rhs, rhsIsRead);
    return !lhsDist && !rhsDist;
  };

  ModuleMembarAnalysis analysis(moduleAllocation, filterFn);
  analysis.run<ClusterBarrierAnalysis>();
}

LogicalResult
runCrossCTAMBarrierInitSyncInsertion(ModuleAllocation &moduleAllocation,
                                     int computeCapability) {
  ModuleOp mod = moduleAllocation.getModuleOp();
  if (computeCapability < 90)
    return success();
  int numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
  if (numCTAs == 1)
    return success();

  std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
  auto *aliases = solver->load<SharedMemoryAliasAnalysis>();
  if (failed(solver->initializeAndRun(mod)))
    return mod.emitError("failed to analyze cross-call mbarrier aliases");

  LogicalResult status = success();
  moduleAllocation.walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
      [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
      [&](FunctionOpInterface funcOp) {
        if (failed(status))
          return;
        OpBuilder builder(funcOp);
        if (failed(insertCrossCTAMBarrierInitSyncForFunction(
                funcOp, *aliases, numCTAs, builder))) {
          status = failure();
        }
      });
  return status;
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
