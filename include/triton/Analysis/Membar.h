#ifndef TRITON_ANALYSIS_MEMBAR_H
#define TRITON_ANALYSIS_MEMBAR_H

#include "triton/Analysis/Function.h"
#include "triton/Analysis/Membar/SmemModel.h"
#include <utility>

namespace mlir {

class OpBuilder;

/// Legacy forward-flow state at the current program point and at function
/// boundaries.
struct MembarInfo {
  /// Effects since the most recent whole-region synchronization.
  BlockInfo pending;

  /// Effects reachable from the function entry block before the first
  /// whole-region synchronization.
  /// It keeps incrementing during the iterative algorithm until
  ///  we note all paths to a basic block has synchronized.
  BlockInfo entryBlockInfo;

  /// Whether every path from the function entry block to the current program
  /// point has synchronized the whole execution region.
  bool allPathsFromEntrySynced = false;

  /// Every warp has synchronized since the last memory or thread effect.
  /// This does not discharge dependencies between different warps.
  bool warpsSynced = false;

  MembarInfo &join(const MembarInfo &other) {
    pending.join(other.pending);
    entryBlockInfo.join(other.entryBlockInfo);
    allPathsFromEntrySynced &= other.allPathsFromEntrySynced;
    warpsSynced &= other.warpsSynced;
    return *this;
  }

  void addBlockInfo(const BlockInfo &blockInfo) {
    if (!allPathsFromEntrySynced)
      entryBlockInfo.join(blockInfo);
    pending.join(blockInfo);
    if (blockInfo.hasEffects())
      warpsSynced = false;
  }

  void syncWarps() { warpsSynced = true; }

  void sync() {
    pending.sync();
    allPathsFromEntrySynced = true;
    syncWarps();
  }

  void applyCallSummary(const MembarInfo &callee) {
    if (!allPathsFromEntrySynced)
      entryBlockInfo.join(callee.entryBlockInfo);
    if (callee.allPathsFromEntrySynced)
      sync();
    pending.join(callee.pending);
    warpsSynced &= !callee.pending.hasEffects();
  }

  template <typename Transform> void transformSlices(Transform transform) {
    pending.transformSlices(transform);
    entryBlockInfo.transformSlices(transform);
  }

  bool operator==(const MembarInfo &) const = default;
};

//===----------------------------------------------------------------------===//
// Shared Memory Barrier Analysis
//===----------------------------------------------------------------------===//

/// TODO: Retire this analysis and MembarInfo after mbarrier arrival folding
/// and the AMD conditional-barrier fallback migrate.
class MembarAnalysis : public triton::PostOrderFunctionAnalysis<MembarInfo> {
public:
  /// Creates a new Membar analysis that generates the shared memory barrier
  /// in the following circumstances:
  /// - RAW: If a shared memory write is followed by a shared memory read, and
  /// their addresses are intersected, a barrier is inserted.
  /// - WAR: If a shared memory read is followed by a shared memory write, and
  /// their addresses are intersected, a barrier is inserted.
  /// - WAW: Successive writes to intersecting addresses require ordering.
  /// RAR needs no barrier because no write is performed.
  /// Temporary storage of operations such as Reduce contributes an incoming
  /// scratch write and an outgoing scratch read.
  MembarAnalysis(Allocation &allocation, MembarFilterFn filter,
                 triton::BufferRegionAnalysis &regions,
                 CompletionScopeFn completionScope = nullptr)
      : allocation(allocation), filter(std::move(filter)),
        shared(allocation, regions,
               triton::membar::SMemModel::AccessMode::AllSharedAccesses,
               completionScope) {}

private:
  void updateSuccessor(Operation *terminator, Block *successor,
                       MembarInfo *membarInfo) override;

  void updateExitState(MembarInfo *membarInfo) override;

protected:
  /// A null builder analyzes existing synchronization without inserting any.
  void update(Operation *operation, MembarInfo *membarInfo, FuncMapT *funcMap,
              OpBuilder *builder) override;

  Allocation &allocation;

private:
  void syncIfNeeded(Operation *operation, const BlockInfo &effects,
                    MembarInfo *membarInfo, OpBuilder *builder);

  MembarFilterFn filter;
  triton::membar::SMemModel shared;
};

/// Inserts shared-memory and operation rendezvous barriers across a module,
/// before async-completion analyses consume the synchronization points.
/// Function summaries retain entry-prefix and pending exit states for calls.
/// TODO: Redirect the Python analysis.membar binding and the AMD test pass
/// before removing this driver with MembarAnalysis.
class ModuleMembarAnalysis {
public:
  ModuleMembarAnalysis(ModuleAllocation &moduleAllocation,
                       MembarFilterFn filter = nullptr)
      : moduleAllocation(moduleAllocation), filter(std::move(filter)) {}

  void run();

  template <typename AnalysisT> void run() {
    // Geometry and interval slices use the same completed allocation, including
    // backend scratch sizes and shared-memory partition offsets.
    auto solver = createDataFlowSolver();
    auto *regions = solver->load<triton::BufferRegionAnalysis>(
        triton::BufferRegionAnalysis::Mode::AllMemory, &moduleAllocation);
    if (failed(solver->initializeAndRun(moduleAllocation.getModuleOp())))
      llvm::report_fatal_error("failed to analyze allocated buffer regions");
    runAnalysis<AnalysisT>(*regions);
  }

  template <typename AnalysisT>
  void runAnalysis(triton::BufferRegionAnalysis &regions) {
    AnalysisT::runModule(
        moduleAllocation.getModuleOp(), [&](FunctionOpInterface function) {
          auto &allocation = *moduleAllocation.getFuncData(function);
          return AnalysisT(allocation, filter, regions);
        });
  }

private:
  ModuleAllocation &moduleAllocation;
  MembarFilterFn filter;
};

} // namespace mlir

#endif // TRITON_ANALYSIS_MEMBAR_H
