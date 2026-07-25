#ifndef TRITON_ANALYSIS_BUFFER_REGION_H
#define TRITON_ANALYSIS_BUFFER_REGION_H

#include <algorithm>
#include <cstdint>
#include <set>
#include <tuple>
#include <utility>

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseBitVector.h"

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
  static AddressSet fromAddresses(llvm::ArrayRef<uint32_t> addresses);

  void set(uint32_t address);
  void insert(const AddressSet &other);
  AddressSet intersection(const AddressSet &other) const;
  void subtract(const AddressSet &other);

  auto begin() const { return addresses.begin(); }
  auto end() const { return addresses.end(); }
  bool empty() const { return addresses.empty(); }
  bool contains(uint32_t address) const;
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
  AddressSet addresses;
  llvm::SmallVector<CTAAddresses, 2> ctaAddresses;

  /// Internal view provenance used while composing nested views.
  uint32_t storageBase = 0;
  uint32_t affineOffset = 0;
  llvm::SmallVector<uint32_t, 2> partitionBases;
  uint32_t affinePartitionOffset = 0;
  uint32_t affineCTAOffset = 0;

  BufferRegion() = default;
  BufferRegion(uint32_t baseOffset, uint32_t length)
      : baseOffset(baseOffset), length(length),
        addresses(AddressSet::fromRange(baseOffset, length)),
        ctaAddresses{{0, addresses}}, storageBase(baseOffset) {}
  BufferRegion(uint32_t baseOffset, uint32_t length, AddressSet addresses,
               uint32_t storageBase, uint32_t affineOffset,
               llvm::ArrayRef<uint32_t> partitionBases = {},
               uint32_t affinePartitionOffset = 0,
               llvm::ArrayRef<CTAAddresses> ctaAddresses = {},
               uint32_t affineCTAOffset = 0)
      : baseOffset(baseOffset), length(length), addresses(std::move(addresses)),
        ctaAddresses(ctaAddresses), storageBase(storageBase),
        affineOffset(affineOffset), partitionBases(partitionBases),
        affinePartitionOffset(affinePartitionOffset),
        affineCTAOffset(affineCTAOffset) {
    if (this->ctaAddresses.empty() && !this->addresses.empty())
      this->ctaAddresses.emplace_back(0, this->addresses);
  }

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
    return baseOffset == other.baseOffset && length == other.length &&
           addresses == other.addresses && ctaAddresses == other.ctaAddresses &&
           storageBase == other.storageBase &&
           affineOffset == other.affineOffset &&
           partitionBases == other.partitionBases &&
           affinePartitionOffset == other.affinePartitionOffset &&
           affineCTAOffset == other.affineCTAOffset;
  }

  bool operator<(const BufferRegion &other) const {
    auto lhs = std::tie(baseOffset, length, addresses, storageBase,
                        affineOffset, affinePartitionOffset, affineCTAOffset);
    auto rhs = std::tie(other.baseOffset, other.length, other.addresses,
                        other.storageBase, other.affineOffset,
                        other.affinePartitionOffset, other.affineCTAOffset);
    if (lhs != rhs)
      return lhs < rhs;
    if (partitionBases != other.partitionBases)
      return std::lexicographical_compare(
          partitionBases.begin(), partitionBases.end(),
          other.partitionBases.begin(), other.partitionBases.end());
    return std::lexicographical_compare(
        ctaAddresses.begin(), ctaAddresses.end(), other.ctaAddresses.begin(),
        other.ctaAddresses.end());
  }

  template <typename T> void print(T &os) const {
    os << "[" << baseOffset << ", " << length << "]";
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
// This wraps a set of BufferRegions and provides lattice semantics
//
struct RegionInfo {
  enum class Kind { Uninitialized, Exact, Unknown };
  using RegionList = std::set<BufferRegion>;

  Kind kind = Kind::Uninitialized;
  RegionList regions;

  RegionInfo() = default;
  RegionInfo(const RegionList &r) : kind(Kind::Exact), regions(r) {}

  bool isUnknown() const { return kind == Kind::Unknown; }

  static RegionInfo join(const RegionInfo &lhs, const RegionInfo &rhs) {
    if (lhs.isUnknown() || rhs.isUnknown())
      return getPessimisticValueState();
    if (lhs.kind == Kind::Uninitialized)
      return rhs;
    if (rhs.kind == Kind::Uninitialized)
      return lhs;
    RegionInfo result = lhs;
    result.regions.insert(rhs.regions.begin(), rhs.regions.end());
    return result;
  }

  bool operator==(const RegionInfo &other) const {
    return kind == other.kind && regions == other.regions;
  }

  template <typename T> void print(T &os) const {
    if (isUnknown()) {
      os << "unknown";
      return;
    }
    llvm::interleaveComma(regions, os,
                          [&](const BufferRegion &r) { r.print(os); });
  }

  static RegionInfo getPessimisticValueState(MLIRContext *context = nullptr) {
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
  static bool isMemoryAccessOperation(Operation *op);

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
  // Global registry of all regions
  std::set<BufferRegion> usedBufferRegions[NUM_REGION_TYPES];
  bool usedUnknownBufferRegions[NUM_REGION_TYPES] = {};
  llvm::DenseMap<std::pair<Type, uint32_t>, AddressSet> footprintCache;
};

} // namespace mlir::triton

#endif // TRITON_ANALYSIS_BUFFER_REGION_H
