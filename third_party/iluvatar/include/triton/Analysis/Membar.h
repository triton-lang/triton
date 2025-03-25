#ifndef TRITON_ANALYSIS_MEMBAR_H
#define TRITON_ANALYSIS_MEMBAR_H

#include "Allocation.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <set>

namespace mlir {

class OpBuilder;

struct BlockInfo {
  using BufferIdSetT = Allocation::BufferIdSetT;
  using IntervalSetT = std::set<Interval<size_t>>;

  IntervalSetT syncReadIntervals;
  IntervalSetT syncWriteIntervals;

  BlockInfo() = default;

  /// Unions two BlockInfo objects.
  BlockInfo &join(const BlockInfo &other) {
    syncReadIntervals.insert(other.syncReadIntervals.begin(),
                             other.syncReadIntervals.end());
    syncWriteIntervals.insert(other.syncWriteIntervals.begin(),
                              other.syncWriteIntervals.end());
    return *this;
  }

  /// Returns true if intervals in two BlockInfo objects are intersected.
  bool isIntersected(const BlockInfo &other) const {
    return /*RAW*/ isIntersected(syncWriteIntervals, other.syncReadIntervals) ||
           /*WAR*/
           isIntersected(syncReadIntervals, other.syncWriteIntervals) ||
           /*WAW*/
           isIntersected(syncWriteIntervals, other.syncWriteIntervals);
  }

  /// Clears the intervals because a barrier is inserted.
  void sync() {
    syncReadIntervals.clear();
    syncWriteIntervals.clear();
  }

  /// Compares two BlockInfo objects.
  bool operator==(const BlockInfo &other) const {
    return syncReadIntervals == other.syncReadIntervals &&
           syncWriteIntervals == other.syncWriteIntervals;
  }

  bool operator!=(const BlockInfo &other) const { return !(*this == other); }

private:
  bool isIntersected(const IntervalSetT &lhsIntervalSet,
                     const IntervalSetT &rhsIntervalSet) const {
    for (auto &lhs : lhsIntervalSet)
      for (auto &rhs : rhsIntervalSet)
        if (lhs.intersects(rhs))
          return true;
    return false;
  }
};

//===----------------------------------------------------------------------===//
// Shared Memory Barrier Analysis
//===----------------------------------------------------------------------===//
class MembarAnalysis {
public:
  using FuncBlockInfoMapT = CallGraph<BlockInfo>::FuncDataMapT;
  /// Creates a new Membar analysis that generates the shared memory barrier
  /// in the following circumstances:
  /// - RAW: If a shared memory write is followed by a shared memory read, and
  /// their addresses are intersected, a barrier is inserted.
  /// - WAR: If a shared memory read is followed by a shared memory write, and
  /// their addresses are intersected, a barrier is inserted.
  /// The following circumstances do not require a barrier:
  /// - WAW: not possible because overlapped memory allocation is not allowed.
  /// - RAR: no write is performed.
  /// Temporary storage of operations such as Reduce are considered as both
  /// a shared memory read. If the temporary storage is written but not read,
  /// it is considered as the problem of the operation itself but not the membar
  /// analysis.
  MembarAnalysis() = default;
  explicit MembarAnalysis(Allocation *allocation) : allocation(allocation) {}

  /// Runs the membar analysis to the given operation, inserts a barrier if
  /// necessary.
  void run(FuncBlockInfoMapT &funcBlockInfoMap);

private:
  /// Applies the barrier analysis based on the SCF dialect, in which each
  /// region has a single basic block only.
  /// Example:
  /// region1
  ///   op1
  ///   op2 (scf.if)
  ///      region2
  ///        op3
  ///        op4
  ///      region3
  ///        op5
  ///        op6
  ///   op7
  /// TODO: Explain why we don't use ForwardAnalysis:
  void resolve(FunctionOpInterface funcOp, FuncBlockInfoMapT *funcBlockInfoMap,
               OpBuilder *builder);

  /// Updates the BlockInfo operation based on the operation.
  void update(Operation *operation, BlockInfo *blockInfo,
              FuncBlockInfoMapT *funcBlockInfoMap, OpBuilder *builder);

  /// Collects the successors of the terminator
  void visitTerminator(Operation *operation, SmallVector<Block *> &successors);

  void insertBarrier(Operation *operation, OpBuilder *builder);

private:
  Allocation *allocation = nullptr;
};

/// Postorder traversal on the callgraph to insert membar instructions
/// of each function.
/// Each function maintains a BlockInfo map that includes all potential buffers
/// after returning. This way users do not have to explicitly insert membars
/// before and after function calls, but might be a bit conservative.
class ModuleMembarAnalysis : public CallGraph<BlockInfo> {
public:
  ModuleMembarAnalysis(ModuleAllocation *moduleAllocation)
      : CallGraph<BlockInfo>(moduleAllocation->getModuleOp()),
        moduleAllocation(moduleAllocation) {}

  void run() {
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        // Pre-order walk callback
        [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
        // Post-order walk callback
        [&](FunctionOpInterface funcOp) {
          auto *allocation = moduleAllocation->getFuncData(funcOp);
          auto [it, inserted] = funcMap.try_emplace(funcOp, BlockInfo());
          if (inserted) {
            MembarAnalysis analysis(allocation);
            analysis.run(funcMap);
          }
        });
  }

private:
  ModuleAllocation *moduleAllocation;
};

} // namespace mlir

#endif // TRITON_ANALYSIS_MEMBAR_H
