#ifndef TRITON_ANALYSIS_MEMBAR_H
#define TRITON_ANALYSIS_MEMBAR_H

#include "Allocation.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {

class OpBuilder;

//===----------------------------------------------------------------------===//
// Shared Memory Barrier Analysis
//===----------------------------------------------------------------------===//
class MembarAnalysis {
public:
  /// Creates a new Membar analysis that generates the shared memory barrier
  /// in the following circumstances:
  /// - RAW: If a shared memory write is followed by a shared memory read, and
  /// their addresses are intersected, a barrier is inserted.
  /// - WAR: If a shared memory read is followed by a shared memory read, and
  /// their addresses are intersected, a barrier is inserted.
  /// The following circumstances do not require a barrier:
  /// - WAW: not possible because overlapped memory allocation is not allowed.
  /// - RAR: no write is performed.
  /// Temporary storage of operations such as Reduce are considered as both
  /// a shared memory read. If the temporary storage is written but not read,
  /// it is considered as the problem of the operation itself but not the membar
  /// analysis.
  /// The following circumstances are not considered yet:
  /// - Double buffers
  /// - N buffers
  MembarAnalysis(Allocation *allocation) : allocation(allocation) {}

  /// Runs the membar analysis to the given operation, inserts a barrier if
  /// necessary.
  void run();

private:
  struct RegionInfo {
    using BufferIdSetT = Allocation::BufferIdSetT;

    BufferIdSetT syncReadBuffers;
    BufferIdSetT syncWriteBuffers;

    RegionInfo() = default;
    RegionInfo(const BufferIdSetT &syncReadBuffers,
               const BufferIdSetT &syncWriteBuffers)
        : syncReadBuffers(syncReadBuffers), syncWriteBuffers(syncWriteBuffers) {
    }

    /// Unions two RegionInfo objects.
    void join(const RegionInfo &other) {
      syncReadBuffers.insert(other.syncReadBuffers.begin(),
                             other.syncReadBuffers.end());
      syncWriteBuffers.insert(other.syncWriteBuffers.begin(),
                              other.syncWriteBuffers.end());
    }

    /// Returns true if buffers in two RegionInfo objects are intersected.
    bool isIntersected(const RegionInfo &other, Allocation *allocation) const {
      return /*RAW*/ isIntersected(syncWriteBuffers, other.syncReadBuffers,
                                   allocation) ||
             /*WAR*/
             isIntersected(syncReadBuffers, other.syncWriteBuffers,
                           allocation) ||
             /*WAW*/
             isIntersected(syncWriteBuffers, other.syncWriteBuffers,
                           allocation);
    }

    /// Clears the buffers because a barrier is inserted.
    void sync() {
      syncReadBuffers.clear();
      syncWriteBuffers.clear();
    }

  private:
    /// Returns true if buffers in two sets are intersected.
    bool isIntersected(const BufferIdSetT &lhs, const BufferIdSetT &rhs,
                       Allocation *allocation) const {
      return std::any_of(lhs.begin(), lhs.end(), [&](auto lhsId) {
        return std::any_of(rhs.begin(), rhs.end(), [&](auto rhsId) {
          return allocation->isIntersected(lhsId, rhsId);
        });
      });
    }
  };

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
  /// region2 and region3 started with the information of region1.
  /// Each region is analyzed separately and keeps their own copy of the
  /// information. At op7, we union the information of the region2 and region3
  /// and update the information of region1.
  void dfsOperation(Operation *operation, RegionInfo *blockInfo,
                    OpBuilder *builder);

  /// Updates the RegionInfo operation based on the operation.
  void transfer(Operation *operation, RegionInfo *blockInfo,
                OpBuilder *builder);

private:
  Allocation *allocation;
};

} // namespace mlir

#endif // TRITON_ANALYSIS_MEMBAR_H
