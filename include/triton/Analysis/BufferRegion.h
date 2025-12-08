#ifndef TRITON_ANALYSIS_BUFFER_REGION_H
#define TRITON_ANALYSIS_BUFFER_REGION_H

#include <limits>
#include <set>

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/Value.h"

namespace mlir::triton {

//===----------------------------------------------------------------------===//
// BufferRegion: a single logical region derived from an alloc
//===----------------------------------------------------------------------===//
struct BufferRegion {
  uint32_t baseOffset;
  uint32_t length;

  bool operator==(const BufferRegion &other) const {
    return baseOffset == other.baseOffset && length == other.length;
  }

  bool operator<(const BufferRegion &other) const {
    if (baseOffset != other.baseOffset)
      return baseOffset < other.baseOffset;
    return length < other.length;
  }

  template <typename T> void print(T &os) const {
    os << "[" << baseOffset << ", " << length << "]";
  }
};

} // namespace mlir::triton

namespace llvm {

using namespace mlir::triton;

template <> struct DenseMapInfo<BufferRegion> {
  static BufferRegion getEmptyKey() {
    constexpr uint32_t empty = std::numeric_limits<uint32_t>::max();
    return BufferRegion{empty, empty};
  }
  static BufferRegion getTombstoneKey() {
    constexpr uint32_t tombstone = std::numeric_limits<uint32_t>::max() - 1;
    return BufferRegion{tombstone, tombstone};
  }
  static unsigned getHashValue(const BufferRegion &r) {
    return llvm::hash_combine(r.baseOffset, r.length);
  }
  static bool isEqual(const BufferRegion &a, const BufferRegion &b) {
    return a == b;
  }
};

} // namespace llvm

namespace mlir::triton {

//===----------------------------------------------------------------------===//
// RegionInfo lattice
//===----------------------------------------------------------------------===//
//
// This wraps a set of BufferRegions and provides lattice semantics
//
struct RegionInfo {
  using RegionList = llvm::DenseSet<BufferRegion>;
  RegionList regions;

  RegionInfo() = default;
  RegionInfo(const RegionList &r) : regions(r) {}

  // Lattice join: union of regions
  static RegionInfo join(const RegionInfo &lhs, const RegionInfo &rhs) {
    RegionInfo result = lhs;
    for (const auto &reg : rhs.regions)
      if (llvm::find(result.regions, reg) == result.regions.end())
        result.regions.insert(reg);
    return result;
  }

  bool operator==(const RegionInfo &other) const {
    if (regions.size() != other.regions.size())
      return false;
    for (auto &r : regions)
      if (llvm::find(other.regions, r) == other.regions.end())
        return false;
    return true;
  }

  template <typename T> void print(T &os) const {
    llvm::SmallVector<BufferRegion> sortedRegions(regions.begin(),
                                                  regions.end());
    llvm::sort(sortedRegions, [](const BufferRegion &a, const BufferRegion &b) {
      return a < b;
    });
    llvm::interleaveComma(sortedRegions, os,
                          [&](const BufferRegion &r) { r.print(os); });
  }

  static RegionInfo getPessimisticValueState(MLIRContext *context = nullptr) {
    return RegionInfo(); // means "unknown / empty"
  }
  static RegionInfo getPessimisticValueState(Value) { return RegionInfo(); }
};

//===----------------------------------------------------------------------===//
// BufferRegionAnalysis (Sparse Forward Dataflow)
//===----------------------------------------------------------------------===//
//
// Produces a RegionInfo lattice for each MemDesc/ptr-like SSA value,
// and also collects a global list of all discovered BufferRegions.
//
class BufferRegionAnalysis : public dataflow::SparseForwardDataFlowAnalysis<
                                 dataflow::Lattice<RegionInfo>> {

public:
  using Base =
      dataflow::SparseForwardDataFlowAnalysis<dataflow::Lattice<RegionInfo>>;
  using Base::getLatticeElement;
  using Base::SparseForwardDataFlowAnalysis;

  enum RegionType { SHARED_MEMORY, TENSOR_MEMORY, BARRIER, NUM_REGION_TYPES };

  static bool isMemoryAccessOperation(Operation *op);

  // ------------------------------
  // Public API for ConSan
  // ------------------------------

  /// Return the list of all unique (alloc,offset,len) buffer regions
  /// discovered by the analysis.
  llvm::SmallVector<BufferRegion>
  getAllUsedBufferRegions(RegionType type) const {
    return llvm::to_vector(usedBufferRegions[type]);
  }

  void calculateUsedBufferRegions(Operation *op);

  // ------------------------------
  // Required overrides
  // ------------------------------

  void setToEntryState(dataflow::Lattice<RegionInfo> *lat) override {
    propagateIfChanged(
        lat, lat->join(RegionInfo::getPessimisticValueState(lat->getAnchor())));
  }

  LogicalResult visitOperation(
      Operation *op,
      llvm::ArrayRef<const dataflow::Lattice<RegionInfo> *> operands,
      llvm::ArrayRef<dataflow::Lattice<RegionInfo> *> results) override;

  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      llvm::ArrayRef<dataflow::Lattice<RegionInfo> *> argLattices,
      unsigned firstIndex) override;

  LogicalResult initialize(Operation *top) override;

private:
  // Global registry of all regions
  std::set<BufferRegion> usedBufferRegions[NUM_REGION_TYPES];

  static void verifyOpIsSupported(Operation *op);
};

} // namespace mlir::triton

#endif // TRITON_ANALYSIS_BUFFER_REGION_H
