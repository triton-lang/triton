#ifndef TRITON_ANALYSIS_BUFFER_REGION_H
#define TRITON_ANALYSIS_BUFFER_REGION_H

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/UniqueVector.h"

namespace mlir {
class Allocation;
class ModuleAllocation;
} // namespace mlir

namespace mlir::triton::gpu {
enum class SharedKind : uint32_t;
}

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
  /// Descriptor allocation supplying these views; null for implicit scratch.
  Operation *allocation = nullptr;

  bool contains(const BufferRegionView &other) const {
    return allocationFrame == other.allocationFrame &&
           region.contains(other.region);
  }

  BufferRegionView translated(uint32_t offset,
                              uint32_t newAllocationFrame) const;

private:
  auto key() const {
    return std::tie(allocationFrame, region, storageBase, affineOffset,
                    affinePartitionOffset, affineCTAOffset, partitionBases,
                    allocation);
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

/// Complete MAY-views of a descriptor, shared by all of its memory effects.
/// Runtime indices may select several physical views across loop iterations.
/// Addresses, rather than runtime descriptor keys, define the geometry; the
/// memory space distinguishes shared bytes from TMEM's 32-bit storage words.
struct BufferRegionFootprint {
  Attribute memorySpace;
  RegionInfo regionInfo;
};

/// Prove disjointness only if every candidate pair is disjoint. Missing views
/// and unnormalized allocation frames may overlap, never denote empty storage.
bool mayOverlap(const BufferRegionFootprint *lhs,
                const BufferRegionFootprint *rhs);

enum class RW { Read, Write };

struct MemoryAccess {
  Value value;
  bool isWrite;
  bool isRead;
  std::optional<gpu::SharedKind> sharedKind;

  bool isShared() const { return sharedKind.has_value(); }
  bool isShared(gpu::SharedKind kind) const { return sharedKind == kind; }
};

llvm::SmallVector<MemoryAccess>
getMemoryAccesses(Operation *op,
                  std::optional<gpu::SharedKind> kind = std::nullopt,
                  std::optional<RW> rw = std::nullopt);

bool hasSharedAccess(Operation *op,
                     std::optional<gpu::SharedKind> kind = std::nullopt,
                     std::optional<RW> rw = std::nullopt);

//===----------------------------------------------------------------------===//
// BufferRegionAnalysis (Sparse Forward Dataflow)
//===----------------------------------------------------------------------===//
//
// Produces a RegionInfo lattice for each MemDesc/ptr-like SSA value,
// and also collects a global list of all discovered BufferRegions.
// Requires completed allocation analyses or their materialized offsets.
//
class BufferRegionAnalysis : public dataflow::SparseForwardDataFlowAnalysis<
                                 dataflow::Lattice<RegionInfo>> {

public:
  using Base =
      dataflow::SparseForwardDataFlowAnalysis<dataflow::Lattice<RegionInfo>>;
  using Base::getLatticeElement;
  enum class Mode { AllMemory, TensorMemoryOnly };

  explicit BufferRegionAnalysis(DataFlowSolver &solver,
                                Mode mode = Mode::AllMemory,
                                ModuleAllocation *allocation = nullptr)
      : Base(solver), mode(mode), moduleAllocation(allocation) {}

  enum RegionType { SHARED_MEMORY, TENSOR_MEMORY, BARRIER, NUM_REGION_TYPES };

  const RegionInfo &getRegionInfo(Value value) {
    return getLatticeElement(value)->getValue();
  }

  /// Return an immutable footprint covering every possible view, owned by this
  /// analysis. Call after solver convergence. Return null for unknown geometry
  /// or views outside allocationFrame, when set.
  const BufferRegionFootprint *
  getFootprint(Value value, FunctionOpInterface allocationFrame = {});

  /// Describe an operation's allocated shared scratch, including cross-CTA
  /// accesses.
  const BufferRegionFootprint *getScratchFootprint(Operation *op);

  /// Translate callee-local views into the caller's allocation frame, caching
  /// the complete footprint and preserving views already in other frames.
  const BufferRegionFootprint *
  translateToCallsite(const BufferRegionFootprint *footprint,
                      CallOpInterface call, FunctionOpInterface callee);

  /// Shared-memory offset of the callee frame in the caller's allocation.
  uint32_t getCallOffset(CallOpInterface call) const;

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
  const Mode mode;
  ModuleAllocation *moduleAllocation;

  Allocation *getAllocation(Operation *op) const;

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
  llvm::DenseMap<std::pair<Type, uint32_t>,
                 llvm::SmallVector<BufferRegion::CTAAddresses, 2>>
      footprintCache;
  llvm::DenseMap<Value, std::unique_ptr<BufferRegionFootprint>> valueFootprints;
  llvm::DenseMap<Operation *, std::unique_ptr<BufferRegionFootprint>>
      scratchFootprints;
  llvm::DenseMap<std::pair<const BufferRegionFootprint *, Operation *>,
                 std::unique_ptr<BufferRegionFootprint>>
      callsiteFootprints;
  llvm::UniqueVector<Operation *> operationInterner;
};

} // namespace mlir::triton

#endif // TRITON_ANALYSIS_BUFFER_REGION_H
