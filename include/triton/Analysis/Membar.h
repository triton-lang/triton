#ifndef TRITON_ANALYSIS_MEMBAR_H
#define TRITON_ANALYSIS_MEMBAR_H

#include "Allocation.h"

#include <set>

namespace mlir {

class OpBuilder;

/// Callback to allow backend to provide more information on whether a barrier
/// is needed between two operations. Even though two operations access the same
/// shared memory they may not require a barrier in between them.
using MembarFilterFn = std::function<bool(Operation *, Operation *)>;

struct BlockInfo {
  // Union-Find Disjoint Sets to represent cross-CTA reads/writes
  struct CTA_UFDS {
    SmallVector<unsigned> parent;
    SmallVector<unsigned> rank;
    // Invariant: At the root of a class, minRep[i] is the smallest element in
    // the class
    SmallVector<unsigned> minRep;

    CTA_UFDS() = default;
    explicit CTA_UFDS(unsigned n) : rank(n, 0), minRep(n) {
      assert(llvm::isPowerOf2_32(n) && n != 0);
      parent = llvm::to_vector(llvm::seq(n));
      for (unsigned i = 0; i < n; ++i)
        minRep[i] = i;
    }

    unsigned find(unsigned x) const {
      unsigned p = parent[x];
      while (p != parent[p])
        p = parent[p];
      return p;
    }

    unsigned findMin(unsigned x) const { return minRep[find(x)]; }

    void unite(unsigned x, unsigned y) {
      x = find(x);
      y = find(y);
      if (x == y)
        return;

      if (rank[x] < rank[y])
        std::swap(x, y);

      parent[y] = x;
      minRep[x] = std::min(minRep[x], minRep[y]);

      if (rank[x] == rank[y])
        ++rank[x];
    }

    CTA_UFDS join(const CTA_UFDS &other) const {
      // Transitive closure of two UFDS
      CTA_UFDS result = *this;
      for (unsigned i = 0; i < size(); ++i)
        result.unite(i, other.find(i));
      return result;
    }

    SmallVector<unsigned> canonical() const {
      SmallVector<unsigned> reps(size());
      for (unsigned i = 0; i < size(); ++i)
        reps[i] = findMin(i);
      return reps;
    }

    bool isDistributed() const { return *this != CTA_UFDS(parent.size()); }

    bool operator<(const CTA_UFDS &other) const {
      return canonical() < other.canonical();
    }
    bool operator==(const CTA_UFDS &other) const {
      return canonical() == other.canonical();
    }
    bool operator!=(const CTA_UFDS &other) const { return !(*this == other); }

    void print(raw_ostream &os) const {
      os << "UFDS(";
      llvm::interleaveComma(canonical(), os, [&](unsigned x) { os << x; });
      os << ")";
    }

    size_t size() const { return parent.size(); }
  };

  using IntervalMapT =
      std::map<std::pair<Interval<size_t>, CTA_UFDS>, std::set<Operation *>>;

  IntervalMapT syncReadIntervals;
  IntervalMapT syncWriteIntervals;

  BlockInfo() = default;

  /// Unions two BlockInfo objects.
  BlockInfo &join(const BlockInfo &other) {
    // We don't fold the intervals (we could tho)
    for (auto &[key, ops] : other.syncReadIntervals)
      syncReadIntervals[key].insert(ops.begin(), ops.end());
    for (auto &[key, ops] : other.syncWriteIntervals)
      syncWriteIntervals[key].insert(ops.begin(), ops.end());
    return *this;
  }

  void dump() {
    auto &err = llvm::errs();

    auto printKey = [&](const std::pair<Interval<size_t>, CTA_UFDS> &key) {
      const auto &[interval, ufds] = key;
      err << "    [" << interval.start() << ", " << interval.end() << "] ";
      if (ufds.isDistributed()) {
        ufds.print(err);
        err << " ";
      } else if (ufds.size() == 1) {
        err << " (CTA local) ";
      }
    };
    err << "Block Interval:\n";
    err << "  Read Intervals:\n";
    for (auto &[key, ops] : syncReadIntervals) {
      printKey(key);
      for (auto &op : ops)
        err << op->getName() << " ";
      err << "\n";
    }
    err << "  Write Intervals:\n";
    for (auto &[key, ops] : syncWriteIntervals) {
      printKey(key);
      for (auto &op : ops)
        err << op->getName() << " ";
      err << "\n";
    }
  }

  /// Returns true if intervals in two BlockInfo objects are intersected.
  std::optional<CTA_UFDS> isIntersected(const BlockInfo &other,
                                        MembarFilterFn filter) const {
    auto raw =
        isIntersected(syncWriteIntervals, other.syncReadIntervals, filter);
    auto war =
        isIntersected(syncReadIntervals, other.syncWriteIntervals, filter);
    auto waw =
        isIntersected(syncWriteIntervals, other.syncWriteIntervals, filter);
    auto maybeJoin =
        [](const std::optional<CTA_UFDS> &lhs,
           const std::optional<CTA_UFDS> &rhs) -> std::optional<CTA_UFDS> {
      if (!lhs.has_value())
        return rhs;
      if (!rhs.has_value())
        return lhs;
      return lhs.value().join(rhs.value());
    };
    return maybeJoin(raw, maybeJoin(war, waw));
  }

  /// Clears the intervals because a barrier is inserted.
  /// If `cluster` is true, the barrier synchronizes all CTAs in the cluster and
  /// we can drop every pending dependency. Otherwise only CTA-local
  /// dependencies are cleared; distributed ones remain until a cluster barrier
  /// is observed.
  void sync(bool cluster) {
    if (cluster) {
      syncReadIntervals.clear();
      syncWriteIntervals.clear();
    } else {
      auto eraseNotDistributed = [](auto &map) {
        for (auto &[key, _] : llvm::make_early_inc_range(map)) {
          if (!key.second.isDistributed())
            map.erase(key);
        }
      };
      eraseNotDistributed(syncReadIntervals);
      eraseNotDistributed(syncWriteIntervals);
    }
  }

  /// Compares two BlockInfo objects.
  bool operator==(const BlockInfo &other) const {
    return syncReadIntervals == other.syncReadIntervals &&
           syncWriteIntervals == other.syncWriteIntervals;
  }

  bool operator!=(const BlockInfo &other) const { return !(*this == other); }

private:
  static bool haveSameAlloc(Operation *lhs, Operation *rhs);

  std::optional<CTA_UFDS> isIntersected(const IntervalMapT &lhsIntervalSet,
                                        const IntervalMapT &rhsIntervalSet,
                                        MembarFilterFn filter) const {
    // They intersect whenever the intervals intersect. If they do, collect the
    // union of CTA sets for any op pair that is not filtered out and does not
    // share the exact same explicit shared value.
    std::optional<CTA_UFDS> ret = std::nullopt;
    for (const auto &[lhsKey, lhsOps] : lhsIntervalSet) {
      const auto &[intervalLhs, ctasLhs] = lhsKey;
      for (const auto &[rhsKey, rhsOps] : rhsIntervalSet) {
        const auto &[intervalRhs, ctasRhs] = rhsKey;
        if (!intervalLhs.intersects(intervalRhs))
          continue;

        auto joined = ctasLhs.join(ctasRhs);
        bool skipBarrier =
            llvm::all_of(lhsOps, [&, rhsOpsPtr = &rhsOps](const auto &lhsOp) {
              return llvm::all_of(*rhsOpsPtr, [&](const auto &rhsOp) {
                return (filter && filter(lhsOp, rhsOp)) ||
                       (joined.isDistributed() && haveSameAlloc(lhsOp, rhsOp));
              });
            });
        if (skipBarrier)
          continue;

        if (!ret.has_value()) {
          ret = joined;
        } else {
          ret = ret->join(joined);
        }
        // Single CTA case, we can early exit
        if (ret->size() == 1) {
          return ret;
        }
      }
    }
    return ret;
  }
};

//===----------------------------------------------------------------------===//
// Shared Memory Barrier Analysis
//===----------------------------------------------------------------------===//

// Common class to analyze membar and fence placement.
class MembarOrFenceAnalysis {
  using VirtualBlock = std::pair<Block *, Block::iterator>;

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

  void insertBarrier(Operation *operation, OpBuilder *builder,
                     const BlockInfo::CTA_UFDS &ctaClasses);
};

/// Postorder traversal on the callgraph to insert membar instructions
/// of each function.
/// Each function maintains a BlockInfo map that includes all potential buffers
/// after returning. This way users do not have to explicitly insert membars
/// before and after function calls, but might be a bit conservative.
template <typename AnalysisType>
class ModuleMembarOrFenceAnalysis : public CallGraph<BlockInfo> {
public:
  ModuleMembarOrFenceAnalysis(ModuleAllocation *moduleAllocation,
                              MembarFilterFn filter = nullptr)
      : CallGraph<BlockInfo>(moduleAllocation->getModuleOp()),
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
