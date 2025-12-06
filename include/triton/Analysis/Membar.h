#ifndef TRITON_ANALYSIS_MEMBAR_H
#define TRITON_ANALYSIS_MEMBAR_H

#include "Allocation.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

class OpBuilder;

/// Callback to allow backend to provide more information on whether a barrier
/// is needed between two operations. Even though two operations access the same
/// shared memory they may not require a barrier in between them.
using MembarFilterFn = std::function<bool(Operation *, Operation *)>;

// Represents the complete view chain for an access operation
struct ViewChain {
public:
  // Parse view chain from a value, collecting subslice offsets. This builder
  // is loop-aware and provides fine-grained access information
  static ViewChain getFineGrainAccess(Value value,
                                      Allocation::BufferId bufferId,
                                      Interval<size_t> allocationInterval);

  // Builder for accesses that represent accesses to the whole
  // allocation (scratch buffers, ArriveBarrierOp, layout changes, ..)
  static ViewChain getWholeAllocAccess(Allocation::BufferId id,
                                       Interval<size_t> interval) {
    ViewChain vc;
    vc.bufferId = id;
    vc.allocationInterval = interval;
    return vc;
  }

  bool operator==(const ViewChain &other) const {
    return subsliceOffsets == other.subsliceOffsets &&
           allocationAccessTy == other.allocationAccessTy &&
           bufferId == other.bufferId &&
           allocationInterval == other.allocationInterval;
  }

  bool intersects(const ViewChain &other) const;

  void print(raw_ostream &os) const {
    os << "shape=";
    if (allocationAccessTy) {
      llvm::interleave(allocationAccessTy.getShape(), os, "x");
    } else {
      os << "?";
    }
    os << " offsets=[";
    if (subsliceOffsets) {
      llvm::interleaveComma(*subsliceOffsets, os);
    } else {
      os << "unknown";
    }
    os << "] bufferId=";
    if (bufferId != Allocation::InvalidBufferId)
      os << bufferId;
    else
      os << "invalid";
    os << " interval=[" << allocationInterval.start() << ","
       << allocationInterval.end() << ")";
  }

private:
  // Only allow ViewChain construction through get*AllocAccess
  ViewChain() = default;
  // Offsets from subslice. nullopt if full offsets couldn't be determined
  std::optional<SmallVector<int64_t>> subsliceOffsets;
  // Type at the access point (load, store, ..)
  triton::gpu::MemDescType allocationAccessTy;
  Allocation::BufferId bufferId = Allocation::InvalidBufferId;
  // The allocated interval for this buffer
  Interval<size_t> allocationInterval;
};

struct BlockInfo {
  using ViewChainListT = SmallVector<ViewChain, 2>;
  using ViewChainMapT = std::map<Operation *, ViewChainListT>;
  using BufferLayoutMapT = std::map<Allocation::BufferId, Attribute>;

  ViewChainMapT syncReadViewChains;
  ViewChainMapT syncWriteViewChains;
  BufferLayoutMapT bufferLayouts;

  BlockInfo() = default;

  /// Unions two BlockInfo objects.
  BlockInfo &join(const BlockInfo &other) {
    // Avoid inserting duplicate ViewChains, otherwise Membar
    // cannot to reach a fixed point
    for (auto &[op, chains] : other.syncReadViewChains) {
      for (const auto &chain : chains)
        if (!llvm::is_contained(syncReadViewChains[op], chain))
          syncReadViewChains[op].push_back(chain);
    }
    for (auto &[op, chains] : other.syncWriteViewChains) {
      for (const auto &chain : chains)
        if (!llvm::is_contained(syncWriteViewChains[op], chain))
          syncWriteViewChains[op].push_back(chain);
    }
    for (auto &[bufferId, layout] : other.bufferLayouts)
      bufferLayouts.insert_or_assign(bufferId, layout);
    return *this;
  }

  void dump() {
    auto &os = llvm::errs();
    os << "Block ViewChains:\n";
    os << "  Read ViewChains:\n";
    for (auto &[op, viewChains] : syncReadViewChains) {
      for (auto &viewChain : viewChains) {
        os << "    " << op->getName() << ": ";
        viewChain.print(os);
        os << "\n";
      }
    }
    os << "  Write ViewChains:\n";
    for (auto &[op, viewChains] : syncWriteViewChains) {
      for (auto &viewChain : viewChains) {
        os << "    " << op->getName() << ": ";
        viewChain.print(os);
        os << "\n";
      }
    }
    os << "  Buffer Layouts:\n";
    for (auto &[bufferId, layout] : bufferLayouts) {
      os << "    bufferId=" << bufferId << " layout=";
      layout.print(os);
      os << "\n";
    }
  }

  /// Returns true if ViewChains in two BlockInfo objects are intersected.
  bool isIntersected(const BlockInfo &other, MembarFilterFn filter) const {
    return /*RAW*/ isIntersected(syncWriteViewChains, other.syncReadViewChains,
                                 filter) ||
           /*WAR*/
           isIntersected(syncReadViewChains, other.syncWriteViewChains,
                         filter) ||
           /*WAW*/
           isIntersected(syncWriteViewChains, other.syncWriteViewChains,
                         filter);
  }

  /// Clears the ViewChains and layout tracking because a barrier is inserted.
  void sync() {
    syncReadViewChains.clear();
    syncWriteViewChains.clear();
    bufferLayouts.clear();
  }

  /// Compares two BlockInfo objects.
  bool operator==(const BlockInfo &other) const {
    return syncReadViewChains == other.syncReadViewChains &&
           syncWriteViewChains == other.syncWriteViewChains &&
           bufferLayouts == other.bufferLayouts;
  }

  bool operator!=(const BlockInfo &other) const { return !(*this == other); }

private:
  bool isIntersected(const ViewChainMapT &lhsViewChains,
                     const ViewChainMapT &rhsViewChains,
                     MembarFilterFn filter) const {
    for (auto &[lhsOp, lhsViewChains] : lhsViewChains) {
      for (auto &lhsViewChain : lhsViewChains) {
        for (auto &[rhsOp, rhsViewChains] : rhsViewChains) {
          for (auto &rhsViewChain : rhsViewChains) {
            if (lhsViewChain.intersects(rhsViewChain)) {
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
