#ifndef TRITON_ANALYSIS_MEMBAR_H
#define TRITON_ANALYSIS_MEMBAR_H

#include "Allocation.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

namespace mlir {

class OpBuilder;

/// Callback to allow backend to provide more information on whether a barrier
/// is needed between two operations. Even though two operations access the same
/// shared memory they may not require a barrier in between them.
using MembarFilterFn = std::function<bool(Operation *, Operation *)>;

// Represents the access to a slice of an allocation
// It contains information both on physical memory (bufferId and interval) and
// a logical view on it (layout, subslice offsets and shape for the access)
struct AllocationSlice {
public:
  // Create allocation slice from a value, collecting subslice offsets
  AllocationSlice(Value value, Allocation::BufferId id,
                  Interval<size_t> allocationInterval)
      : bufferId(id), allocationInterval(allocationInterval) {
    assert(id != Allocation::InvalidBufferId &&
           "fromValue must be called with a valid bufferId");
    auto accessTy = cast<triton::gpu::MemDescType>(value.getType());
    shape = SmallVector<int64_t>(accessTy.getShape());
    layout = accessTy.getEncoding();

    // Get the memdesc_subslice information if present. If no subslice is
    // present then all 0s means we subslice from the origin
    SmallVector<int64_t> offsets(shape.size(), 0);
    if (auto subslice = value.getDefiningOp<triton::gpu::MemDescSubsliceOp>()) {
      // We know there aren't subslices before the one because of subslice::fold
      // Still need to check this for where a fold isn't possible (control flow)
      // and when a subslice is carried in a loop
      if (accessTy.getAllocShape() == subslice.getSrc().getType().getShape()) {
        offsets = SmallVector<int64_t>(subslice.getOffsets());
      } else {
        offsets.clear();
      }
    }
    subsliceOffsets = offsets;
  }

  // Builder for accesses that represent accesses to the whole
  // allocation (scratch buffers, ArriveBarrierOp, layout changes, ..)
  AllocationSlice(Allocation::BufferId id, Interval<size_t> interval,
                  Attribute layout = Attribute())
      : bufferId(id), allocationInterval(interval), layout(layout) {}

  bool operator<(const AllocationSlice &other) const {
    if (bufferId != other.bufferId)
      return bufferId < other.bufferId;
    if (allocationInterval != other.allocationInterval)
      return allocationInterval < other.allocationInterval;
    if (layout != other.layout)
      return layout.getAsOpaquePointer() < other.layout.getAsOpaquePointer();
    if (subsliceOffsets != other.subsliceOffsets) {
      return subsliceOffsets < other.subsliceOffsets;
    }
    return shape < other.shape;
  }

  bool operator==(const AllocationSlice &other) const {
    return subsliceOffsets == other.subsliceOffsets && shape == other.shape &&
           bufferId == other.bufferId &&
           allocationInterval == other.allocationInterval &&
           layout == other.layout;
  }

  bool intersects(const AllocationSlice &other) const;

  void print(raw_ostream &os) const {
    os << "bufferId=";
    if (bufferId != Allocation::InvalidBufferId)
      os << bufferId;
    else
      os << "invalid";

    os << " interval=[" << allocationInterval.start() << ","
       << allocationInterval.end() << ")";

    os << " offsets=[";
    if (!subsliceOffsets.empty()) {
      llvm::interleaveComma(subsliceOffsets, os);
    } else {
      os << "unknown";
    }
    os << "]";

    os << " shape=";
    if (!shape.empty()) {
      llvm::interleave(shape, os, "x");
    } else {
      os << "?";
    }

    if (layout)
      os << " layout=" << layout;
    else
      os << " layout=unknown";
  }

private:
  // Offsets from subslice. Empty when offsets are unknown
  SmallVector<int64_t> subsliceOffsets;
  // Shape of the access (load, store, ..)
  SmallVector<int64_t> shape;
  Allocation::BufferId bufferId = Allocation::InvalidBufferId;
  // The allocated interval for this buffer
  Interval<size_t> allocationInterval;
  // Layout used for the access (load, store, ..)
  Attribute layout;
};

struct BlockInfo {
  using SliceMapT = std::map<Operation *, std::set<AllocationSlice>>;

  SliceMapT syncReadSlices;
  SliceMapT syncWriteSlices;

  BlockInfo() = default;

  /// Unions two BlockInfo objects.
  BlockInfo &join(const BlockInfo &other) {
    for (auto &[op, slices] : other.syncReadSlices)
      syncReadSlices[op].insert(slices.begin(), slices.end());

    for (auto &[op, slices] : other.syncWriteSlices)
      syncWriteSlices[op].insert(slices.begin(), slices.end());
    return *this;
  }

  void dump() {
    auto &os = llvm::errs();
    os << "Block Allocation Slices:\n";
    os << "  Read Slices:\n";
    for (auto &[op, slices] : syncReadSlices) {
      for (auto &slice : slices) {
        os << "    " << op->getName() << ": ";
        slice.print(os);
        os << "\n";
      }
    }
    os << "  Write Slices:\n";
    for (auto &[op, slices] : syncWriteSlices) {
      for (auto &slice : slices) {
        os << "    " << op->getName() << ": ";
        slice.print(os);
        os << "\n";
      }
    }
  }

  /// Returns true if Slices in two BlockInfo objects are intersected.
  bool isIntersected(const BlockInfo &other, MembarFilterFn filter) const {
    return /*RAW*/ isIntersected(syncWriteSlices, other.syncReadSlices,
                                 filter) ||
           /*WAR*/
           isIntersected(syncReadSlices, other.syncWriteSlices, filter) ||
           /*WAW*/
           isIntersected(syncWriteSlices, other.syncWriteSlices, filter);
  }

  /// Clears the slices because a barrier is inserted.
  void sync() {
    syncReadSlices.clear();
    syncWriteSlices.clear();
  }

  /// Compares two BlockInfo objects.
  bool operator==(const BlockInfo &other) const {
    return syncReadSlices == other.syncReadSlices &&
           syncWriteSlices == other.syncWriteSlices;
  }

  bool operator!=(const BlockInfo &other) const { return !(*this == other); }

private:
  bool isIntersected(const SliceMapT &lhsSlices, const SliceMapT &rhsSlices,
                     MembarFilterFn filter) const {
    for (auto &[lhsOp, lhsSliceList] : lhsSlices) {
      for (auto &lhsSlice : lhsSliceList) {
        for (auto &[rhsOp, rhsSliceList] : rhsSlices) {
          for (auto &rhsSlice : rhsSliceList) {
            if (lhsSlice.intersects(rhsSlice)) {
              if (!filter || !filter(lhsOp, rhsOp))
                return true;
            }
          }
        }
      }
    }
    return false;
  }
};

//===----------------------------------------------------------------------===//
// Shared Memory Barrier Analysis
//===----------------------------------------------------------------------===//

// Common class to analyze membar and fence placement.
class MembarOrFenceAnalysis {
  using VirtualBlock = std::pair<Block *, Block::iterator>;

public:
  using FuncBlockInfoMapT = triton::CallGraph<BlockInfo>::FuncDataMapT;
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
  MembarOrFenceAnalysis() = default;
  explicit MembarOrFenceAnalysis(Allocation *allocation, MembarFilterFn filter)
      : allocation(allocation), filter(filter) {}

  virtual ~MembarOrFenceAnalysis() = default;

  /// Runs the membar analysis to the given operation, inserts a barrier if
  /// necessary.
  void run(FuncBlockInfoMapT &funcBlockInfoMap);

protected:
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

  /// Collects the successors of the terminator
  void visitTerminator(Operation *operation,
                       SmallVector<VirtualBlock> &successors);

  /// Updates the BlockInfo operation based on the operation.
  virtual void update(Operation *operation, BlockInfo *blockInfo,
                      FuncBlockInfoMapT *funcBlockInfoMap,
                      OpBuilder *builder) = 0;

  Allocation *allocation = nullptr;
  MembarFilterFn filter = nullptr;
};

class MembarAnalysis : public MembarOrFenceAnalysis {
public:
  MembarAnalysis() = default;
  explicit MembarAnalysis(Allocation *allocation, MembarFilterFn filter)
      : MembarOrFenceAnalysis(allocation, filter) {}

  ~MembarAnalysis() override = default;

private:
  /// Updates the BlockInfo operation based on the operation.
  virtual void update(Operation *operation, BlockInfo *blockInfo,
                      FuncBlockInfoMapT *funcBlockInfoMap,
                      OpBuilder *builder) override;

  void insertBarrier(Operation *operation, OpBuilder *builder);
};

/// Postorder traversal on the callgraph to insert membar instructions
/// of each function.
/// Each function maintains a BlockInfo map that includes all potential buffers
/// after returning. This way users do not have to explicitly insert membars
/// before and after function calls, but might be a bit conservative.
template <typename AnalysisType>
class ModuleMembarOrFenceAnalysis : public triton::CallGraph<BlockInfo> {
public:
  ModuleMembarOrFenceAnalysis(ModuleAllocation *moduleAllocation,
                              MembarFilterFn filter = nullptr)
      : triton::CallGraph<BlockInfo>(moduleAllocation->getModuleOp()),
        moduleAllocation(moduleAllocation), filter(filter) {}

  void run() {
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        // Pre-order walk callback
        [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
        // Post-order walk callback
        [&](FunctionOpInterface funcOp) {
          auto *allocation = moduleAllocation->getFuncData(funcOp);
          auto [it, inserted] = funcMap.try_emplace(funcOp, BlockInfo());
          if (inserted) {
            AnalysisType analysis(allocation, filter);
            analysis.run(funcMap);
          }
        });
  }

private:
  ModuleAllocation *moduleAllocation;
  MembarFilterFn filter;
};

typedef ModuleMembarOrFenceAnalysis<MembarAnalysis> ModuleMembarAnalysis;

} // namespace mlir

#endif // TRITON_ANALYSIS_MEMBAR_H
