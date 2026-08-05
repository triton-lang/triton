#ifndef TRITON_ANALYSIS_BUFFER_REGION_H
#define TRITON_ANALYSIS_BUFFER_REGION_H

#include <cstdint>
#include <set>
#include <tuple>
#include <utility>

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/UniqueVector.h"

namespace mlir::triton {

//===----------------------------------------------------------------------===//
// Exact physical address sets
//===----------------------------------------------------------------------===//

/// An exact set of physical storage units. Shared-memory addresses are bytes.
/// Tensor-memory addresses are 32-bit words encoded as (row << 16) | column.
class AddressSet {
public:
  AddressSet() = default;

  static AddressSet fromRange(uint32_t begin, uint32_t length);

  void set(uint32_t address);
  void insert(const AddressSet &other);
  AddressSet intersection(const AddressSet &other) const;
  void subtract(const AddressSet &other);

  auto begin() const { return addresses.begin(); }
  auto end() const { return addresses.end(); }
  bool empty() const { return addresses.empty(); }
  bool intersects(const AddressSet &other) const;
  bool contains(const AddressSet &other) const;
  AddressSet translated(uint32_t delta) const;

  bool operator==(const AddressSet &other) const {
    return addresses == other.addresses;
  }
  bool operator<(const AddressSet &other) const {
    auto lhs = begin();
    auto rhs = other.begin();
    while (lhs != end() && rhs != other.end()) {
      if (*lhs != *rhs)
        return *lhs < *rhs;
      ++lhs;
      ++rhs;
    }
    return lhs == end() && rhs != other.end();
  }

private:
  llvm::SparseBitVector<> addresses;
};

//===----------------------------------------------------------------------===//
// BufferRegion: runtime identity plus exact physical geometry
//===----------------------------------------------------------------------===//

struct BufferRegion {
  using CTAAddresses = std::pair<uint32_t, AddressSet>;

  /// Runtime descriptor key. It deliberately does not define geometry:
  /// distinct sparse views may have the same key.
  uint32_t baseOffset = 0;
  uint32_t length = 0;
  llvm::SmallVector<CTAAddresses, 2> ctaAddresses;

  bool intersects(const BufferRegion &other) const {
    return llvm::any_of(ctaAddresses, [&](const CTAAddresses &lhs) {
      return llvm::any_of(other.ctaAddresses, [&](const CTAAddresses &rhs) {
        return lhs.first == rhs.first && lhs.second.intersects(rhs.second);
      });
    });
  }
  bool contains(const BufferRegion &other) const {
    return llvm::all_of(other.ctaAddresses, [&](const CTAAddresses &rhs) {
      return llvm::any_of(ctaAddresses, [&](const CTAAddresses &lhs) {
        return lhs.first == rhs.first && lhs.second.contains(rhs.second);
      });
    });
  }

  bool operator==(const BufferRegion &other) const {
    return std::tie(baseOffset, length, ctaAddresses) ==
           std::tie(other.baseOffset, other.length, other.ctaAddresses);
  }

  bool operator<(const BufferRegion &other) const {
    return std::tie(baseOffset, length, ctaAddresses) <
           std::tie(other.baseOffset, other.length, other.ctaAddresses);
  }

  template <typename T> void print(T &os) const {
    os << "[" << baseOffset << ", " << length << "]";
  }
};

/// A physical region and the provenance required to compose descriptor views.
struct BufferRegionView {
  BufferRegion region;
  uint32_t storageBase = 0;
  uint32_t affineOffset = 0;
  llvm::SmallVector<uint32_t, 2> partitionBases;
  uint32_t affinePartitionOffset = 0;
  uint32_t affineCTAOffset = 0;
  /// Deterministically interned identity of the owning allocation frame.
  uint32_t allocationFrame = 0;

  bool intersects(const BufferRegionView &other) const {
    return allocationFrame == other.allocationFrame &&
           region.intersects(other.region);
  }

  bool contains(const BufferRegionView &other) const {
    return allocationFrame == other.allocationFrame &&
           region.contains(other.region);
  }

private:
  auto key() const {
    return std::tie(allocationFrame, region, storageBase, affineOffset,
                    affinePartitionOffset, affineCTAOffset, partitionBases);
  }

public:
  bool operator==(const BufferRegionView &other) const {
    return key() == other.key();
  }

  bool operator<(const BufferRegionView &other) const {
    return key() < other.key();
  }
};

//===----------------------------------------------------------------------===//
// Buffer state planning
//===----------------------------------------------------------------------===//

/// A compile-time plan for representing mutable ConSan state. Masks are
/// indexed by the input region order and all have numLanes bits.
struct BufferStatePlan {
  unsigned numLanes = 0;
  llvm::SmallVector<llvm::SmallBitVector> regionMasks;
  llvm::SmallBitVector unknownMask;
};

BufferStatePlan createBufferStatePlan(llvm::ArrayRef<BufferRegion> regions,
                                      bool includeUnknown = false);

//===----------------------------------------------------------------------===//
// RegionInfo lattice
//===----------------------------------------------------------------------===//
//
// This wraps a set of descriptor views and provides lattice semantics.
//
struct RegionInfo {
  enum class Kind { Uninitialized, Exact, Unknown };
  using ViewList = std::set<BufferRegionView>;

  Kind kind = Kind::Uninitialized;
  ViewList views;

  RegionInfo() = default;
  RegionInfo(ViewList views) : kind(Kind::Exact), views(std::move(views)) {}

  bool isUnknown() const { return kind == Kind::Unknown; }

  static RegionInfo join(const RegionInfo &lhs, const RegionInfo &rhs) {
    if (lhs.isUnknown() || rhs.isUnknown())
      return getPessimisticValueState();
    if (lhs.kind == Kind::Uninitialized)
      return rhs;
    if (rhs.kind == Kind::Uninitialized)
      return lhs;
    RegionInfo result = lhs;
    result.views.insert(rhs.views.begin(), rhs.views.end());
    return result;
  }

  bool operator==(const RegionInfo &other) const {
    return kind == other.kind && views == other.views;
  }

  template <typename T> void print(T &os) const {
    if (isUnknown()) {
      os << "unknown";
      return;
    }
    llvm::interleaveComma(views, os, [&](const BufferRegionView &view) {
      view.region.print(os);
    });
  }

  static RegionInfo getPessimisticValueState() {
    RegionInfo result;
    result.kind = Kind::Unknown;
    return result;
  }
  static RegionInfo getPessimisticValueState(Value) {
    return getPessimisticValueState();
  }
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

  struct MemoryAccess {
    Value value;
    bool isWrite;
  };

  static llvm::SmallVector<MemoryAccess> getMemoryAccesses(Operation *op);

  uint32_t getOperationId(Operation *operation) const {
    return operationInterner.idFor(operation);
  }

  // ------------------------------
  // Public API for ConSan
  // ------------------------------

  /// Return all unique exact regions discovered by the analysis.
  llvm::SmallVector<BufferRegion>
  getAllUsedBufferRegions(RegionType type) const {
    return llvm::to_vector(usedBufferRegions[type]);
  }

  bool hasUnknownUsedBufferRegions(RegionType type) const {
    return usedUnknownBufferRegions[type];
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

  LogicalResult initialize(Operation *top) override;

private:
  BufferRegionView getAllocView(Value allocation, uint32_t storageBase,
                                llvm::ArrayRef<uint32_t> partitionBases = {});
  BufferRegionView getSubView(Type type, const BufferRegionView &view,
                              uint32_t storageOffset = 0,
                              uint32_t byteOffset = 0,
                              uint32_t partitionOffset = 0,
                              uint32_t ctaOffset = 0);
  // Global registry of all regions
  std::set<BufferRegion> usedBufferRegions[NUM_REGION_TYPES];
  bool usedUnknownBufferRegions[NUM_REGION_TYPES] = {};
  llvm::DenseMap<std::pair<Type, uint32_t>, AddressSet> footprintCache;
  llvm::UniqueVector<Operation *> operationInterner;
};

} // namespace mlir::triton

#endif // TRITON_ANALYSIS_BUFFER_REGION_H
