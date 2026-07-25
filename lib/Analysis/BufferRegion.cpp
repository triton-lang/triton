#include "triton/Analysis/BufferRegion.h"

#include <limits>
#include <optional>

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/MathExtras.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

using namespace mlir;

namespace {
// TODO: move to Utility.cpp/unify with TritonInstrument/Utility.cpp
FailureOr<SmallVector<uint32_t, 2>> getAllocationOffsets(ttg::LocalAllocOp op) {
  auto offsetAttr = op->getAttr("allocation.offset");
  if (!offsetAttr) {
    op.emitError("ConcurrencySanitizer should run after "
                 "AllocateSharedMemory pass");
    return failure();
  }
  SmallVector<uint32_t, 2> offsets;
  if (auto array = dyn_cast<ArrayAttr>(offsetAttr)) {
    for (Attribute offset : array)
      offsets.push_back(cast<IntegerAttr>(offset).getInt());
  } else {
    offsets.push_back(cast<IntegerAttr>(offsetAttr).getInt());
  }
  return offsets;
}

SmallVector<uint32_t, 2> advancePartitionBases(ArrayRef<uint32_t> bases,
                                               uint32_t offset) {
  SmallVector<uint32_t, 2> advanced;
  advanced.reserve(bases.size());
  for (uint32_t base : bases)
    advanced.push_back(base + offset);
  return advanced;
}

uint64_t getAllocationOffset(ttng::TMEMAllocOp op) {
  auto colOffsetAttr = op->getAttr("tensor_memory_col_offset");
  auto rowOffsetAttr = op->getAttr("tensor_memory_row_offset");
  if (!colOffsetAttr || !rowOffsetAttr) {
    llvm::report_fatal_error(
        "ConcurrencySanitizer should run after AllocateSharedMemory and "
        "TensorMemoryAllocation pass.");
  }
  int colOffset = cast<IntegerAttr>(colOffsetAttr).getInt();
  int rowOffset = cast<IntegerAttr>(rowOffsetAttr).getInt();
  return colOffset | (rowOffset << 16);
}

unsigned getMemDescSize(ttg::MemDescType ty) {
  if (isa<ttng::TensorMemorySpaceAttr>(ty.getMemorySpace()))
    return ttng::getTmemAllocSizes(ty).numCols;
  assert(isa<ttg::SharedMemorySpaceAttr>(ty.getMemorySpace()) &&
         "Unsupported memory space");
  int64_t numElems = ttg::getAllocationElems(ty.getEncoding(), ty.getShape(),
                                             ty.getAllocShape());
  if (auto padded = ttg::getPaddedEncoding(ty.getEncoding()))
    numElems = padded.getPaddedSize({numElems});
  return numElems * ty.getElementType().getIntOrFloatBitWidth() / 8;
}

uint32_t applySharedPadding(uint32_t byteOffset, ttg::MemDescType ty) {
  auto padded = ttg::getPaddedEncoding(ty.getEncoding());
  if (!padded)
    return byteOffset;
  uint64_t elementSize = ty.getElementTypeBitWidth() / 8;
  uint64_t elementOffset = byteOffset / elementSize;
  uint64_t paddedOffset =
      (padded.getPaddedSize({static_cast<int64_t>(elementOffset + 1)}) - 1) *
          elementSize +
      byteOffset % elementSize;
  assert(paddedOffset <= std::numeric_limits<uint32_t>::max());
  return static_cast<uint32_t>(paddedOffset);
}

uint32_t getMemDescStorageOffset(ttg::MemDescType ty, unsigned index);

struct MemDescFootprint {
  triton::AddressSet addresses;
  SmallVector<triton::BufferRegion::CTAAddresses, 2> ctaAddresses;

  void set(uint32_t cta, uint32_t address) {
    addresses.set(address);
    auto it = llvm::find_if(
        ctaAddresses, [=](const auto &entry) { return entry.first == cta; });
    if (it == ctaAddresses.end()) {
      ctaAddresses.emplace_back(cta, triton::AddressSet());
      it = std::prev(ctaAddresses.end());
    }
    it->second.set(address);
  }

  void insert(const MemDescFootprint &other) {
    addresses.insert(other.addresses);
    for (const auto &[cta, otherAddresses] : other.ctaAddresses) {
      auto it = llvm::find_if(
          ctaAddresses, [=](const auto &entry) { return entry.first == cta; });
      if (it == ctaAddresses.end())
        ctaAddresses.emplace_back(cta, otherAddresses);
      else
        it->second.insert(otherAddresses);
    }
  }
};

FailureOr<MemDescFootprint> getMemDescAddresses(
    uint32_t storageBase, uint32_t affineOffset, ttg::MemDescType ty,
    Operation *op,
    llvm::DenseMap<std::pair<Type, uint32_t>, triton::AddressSet> *cache =
        nullptr,
    ArrayRef<uint32_t> partitionBases = {}, uint32_t affinePartitionOffset = 0,
    uint32_t affineCTAOffset = 0) {
  bool isTmem = isa<ttng::TensorMemorySpaceAttr>(ty.getMemorySpace());
  auto collectPages = [&]() -> FailureOr<MemDescFootprint> {
    ttg::MemDescType pageTy =
        ty.cloneWith(ty.getShape().drop_front(), ty.getElementType());
    MemDescFootprint footprint;
    for (int64_t page = 0; page < ty.getDimSize(0); ++page) {
      uint32_t pageOffset = getMemDescStorageOffset(pageTy, page);
      SmallVector<uint32_t, 2> pageBases =
          advancePartitionBases(partitionBases, pageOffset);
      FailureOr<MemDescFootprint> pageAddresses = getMemDescAddresses(
          storageBase + pageOffset, affineOffset, pageTy, op,
          /*cache=*/nullptr, pageBases, affinePartitionOffset, affineCTAOffset);
      if (failed(pageAddresses))
        return failure();
      footprint.insert(*pageAddresses);
    }
    return footprint;
  };
  size_t encodingRank =
      cast<ttg::LayoutEncodingTrait>(ty.getEncoding()).getRank();
  if (encodingRank != ty.getRank()) {
    if (encodingRank + 1 != ty.getRank()) {
      op->emitError("unsupported multibuffer rank in exact buffer region "
                    "analysis");
      return failure();
    }
    return collectPages();
  }
  triton::LinearLayout layout = ttg::isPaddedEncoding(ty.getEncoding())
                                    ? ttg::paddedLinearLayout(ty)
                                    : ttg::toLinearLayout(ty);
  size_t layoutRank = llvm::size(layout.getOutDimNames());
  if (layoutRank != ty.getRank()) {
    if (layoutRank + 1 != ty.getRank()) {
      op->emitError("unsupported multibuffer rank in exact buffer region "
                    "analysis");
      return failure();
    }
    return collectPages();
  }
  triton::LinearLayout inverse = layout.pseudoinvert();
  MLIRContext *ctx = ty.getContext();
  SmallVector<StringAttr> dims = triton::standardOutDimNames(ctx, ty.getRank());
  SmallVector<int64_t> shape(ty.getShape());
  uint64_t numPoints = product(shape);
  uint32_t bitWidth = ty.getElementTypeBitWidth();
  if (!isTmem && bitWidth % 8 != 0) {
    op->emitError("sub-byte shared-memory elements are unsupported by exact "
                  "buffer region analysis");
    return failure();
  }

  StringAttr offsetName = StringAttr::get(ctx, "offset");
  StringAttr blockName = StringAttr::get(ctx, "block");
  StringAttr partitionName = StringAttr::get(ctx, "partition");
  StringAttr rowName = StringAttr::get(ctx, "row");
  StringAttr colName = StringAttr::get(ctx, "col");

  struct PhysicalBasis {
    uint32_t offset = 0;
    uint32_t row = 0;
    uint32_t col = 0;
    uint32_t partition = 0;
    uint32_t block = 0;
  };
  SmallVector<PhysicalBasis> bases;
  bool hasCTAAddressVariation = false;
  for (auto [dim, dimSize] : llvm::zip_equal(dims, shape)) {
    unsigned numBits = llvm::Log2_64(dimSize);
    if (numBits > static_cast<unsigned>(inverse.getInDimSizeLog2(dim))) {
      op->emitError("buffer footprint exceeds its linear-layout domain");
      return failure();
    }
    for (unsigned bit = 0; bit < numBits; ++bit) {
      auto basis = [&](StringAttr name) {
        return inverse.hasOutDim(name)
                   ? static_cast<uint32_t>(inverse.getBasis(dim, bit, name))
                   : 0;
      };
      uint32_t block = basis(blockName);
      hasCTAAddressVariation |= block != 0;
      bases.push_back({basis(offsetName), basis(rowName), basis(colName),
                       basis(partitionName), block});
    }
  }

  if (cache && partitionBases.empty() && !hasCTAAddressVariation) {
    auto key = std::make_pair(Type(ty), affineOffset);
    auto found = cache->find(key);
    if (found == cache->end()) {
      FailureOr<MemDescFootprint> relative =
          getMemDescAddresses(/*storageBase=*/0, affineOffset, ty, op);
      if (failed(relative))
        return failure();
      found = cache->try_emplace(key, std::move(relative->addresses)).first;
    }
    MemDescFootprint footprint;
    footprint.addresses = found->second.translated(storageBase);
    if (!footprint.addresses.empty())
      footprint.ctaAddresses.emplace_back(affineCTAOffset, footprint.addresses);
    return footprint;
  }

  MemDescFootprint footprint;
  auto addPhysicalAddress = [&](const PhysicalBasis &physical) {
    uint32_t cta = physical.block ^ affineCTAOffset;
    if (isTmem) {
      uint64_t bitBegin = static_cast<uint64_t>(physical.col) * bitWidth;
      uint32_t firstWord = bitBegin / 32;
      uint32_t lastWord = llvm::divideCeil(bitBegin + bitWidth, uint64_t{32});
      uint32_t relative = (physical.row << 16) | firstWord;
      uint64_t begin =
          static_cast<uint64_t>(storageBase) + affineOffset + relative;
      for (uint32_t word = firstWord; word < lastWord; ++word) {
        uint64_t address = begin + (word - firstWord);
        assert(address <= std::numeric_limits<uint32_t>::max());
        footprint.set(cta, address);
      }
    } else {
      uint32_t base = storageBase;
      if (!partitionBases.empty())
        base = partitionBases[physical.partition ^ affinePartitionOffset];
      uint32_t relative = physical.offset * (bitWidth / 8);
      uint32_t combined = affineOffset ^ relative;
      uint64_t begin =
          static_cast<uint64_t>(base) + applySharedPadding(combined, ty);
      for (uint32_t byte = 0; byte < bitWidth / 8; ++byte) {
        assert(begin + byte <= std::numeric_limits<uint32_t>::max());
        footprint.set(cta, begin + byte);
      }
    }
  };

  PhysicalBasis physical;
  for (uint64_t index = 0; index < numPoints; ++index) {
    if (index != 0) {
      const PhysicalBasis &basis = bases[llvm::countr_zero(index)];
      physical.offset ^= basis.offset;
      physical.row ^= basis.row;
      physical.col ^= basis.col;
      physical.partition ^= basis.partition;
      physical.block ^= basis.block;
    }
    addPhysicalAddress(physical);
  }
  llvm::sort(footprint.ctaAddresses, llvm::less_first());
  return footprint;
}

FailureOr<triton::BufferRegion> getMemDescRegion(
    uint32_t storageBase, uint32_t affineOffset, ttg::MemDescType ty,
    Operation *op,
    llvm::DenseMap<std::pair<Type, uint32_t>, triton::AddressSet> *cache,
    ArrayRef<uint32_t> partitionBases = {}, uint32_t affinePartitionOffset = 0,
    uint32_t affineCTAOffset = 0) {
  FailureOr<MemDescFootprint> footprint = getMemDescAddresses(
      storageBase, affineOffset, ty, op, cache, partitionBases,
      affinePartitionOffset, affineCTAOffset);
  if (failed(footprint))
    return failure();
  uint32_t runtimeStorageBase = partitionBases.empty()
                                    ? storageBase
                                    : partitionBases[affinePartitionOffset];
  uint32_t baseOffset = runtimeStorageBase +
                        (isa<ttng::TensorMemorySpaceAttr>(ty.getMemorySpace())
                             ? affineOffset
                             : applySharedPadding(affineOffset, ty));
  return triton::BufferRegion(
      baseOffset, getMemDescSize(ty), std::move(footprint->addresses),
      storageBase, affineOffset, partitionBases, affinePartitionOffset,
      footprint->ctaAddresses, affineCTAOffset);
}

uint32_t getMemDescStorageOffset(ttg::MemDescType ty, unsigned index) {
  if (isa<ttng::TensorMemorySpaceAttr>(ty.getMemorySpace()))
    return index * ttng::getTmemAllocSizes(ty).numCols;
  uint64_t elems = ttg::getAllocationElems(ty.getEncoding(), ty.getShape(),
                                           ty.getAllocShape());
  if (auto partitioned =
          dyn_cast<ttg::PartitionedSharedEncodingAttr>(ty.getEncoding()))
    elems /= partitioned.getNumPartitions();
  uint64_t unpadded =
      static_cast<uint64_t>(index) * elems * (ty.getElementTypeBitWidth() / 8);
  assert(unpadded <= std::numeric_limits<uint32_t>::max());
  return applySharedPadding(static_cast<uint32_t>(unpadded), ty);
}

unsigned getNumBuffers(ttg::MemDescIndexOp memdescIndexOp) {
  ttg::MemDescType ty =
      cast<ttg::MemDescType>(memdescIndexOp.getSrc().getType());
  return ty.getShape()[0];
}

llvm::DenseSet<Value> getBarrierOperands(Operation *op) {
  if (auto barrierOp = dyn_cast<ttg::MBarrierOpInterface>(op)) {
    auto barriers = barrierOp.getBarriers();
    return llvm::DenseSet<Value>(barriers.begin(), barriers.end());
  }

  return llvm::DenseSet<Value>{};
}

bool isUsedAsBarrier(Value v) {
  for (auto user : v.getUsers()) {
    if (getBarrierOperands(user).contains(v)) {
      return true;
    }
  }
  return false;
}

bool isUsedAsSharedMemory(Value v) {
  auto type = dyn_cast<ttg::MemDescType>(v.getType());
  return type &&
         isa_and_nonnull<ttg::SharedMemorySpaceAttr>(type.getMemorySpace());
}

bool isUsedAsTensorMemory(Value v) {
  auto type = dyn_cast<ttg::MemDescType>(v.getType());
  return type &&
         isa_and_nonnull<ttng::TensorMemorySpaceAttr>(type.getMemorySpace());
}

struct MemDescSubsliceOffsets {
  uint32_t byteOffset = 0;
  uint32_t partitionOffset = 0;
  uint32_t ctaOffset = 0;
};

MemDescSubsliceOffsets
getMemDescSubsliceUnpaddedOffsets(ttg::MemDescSubsliceOp op) {
  auto srcTy = op.getSrc().getType();
  auto offsets = op.getOffsets();
  if (offsets.empty())
    return MemDescSubsliceOffsets{};

  Attribute encoding = srcTy.getEncoding();
  auto layoutOffsets = ttg::dropPipeliningDim(offsets, encoding);
  auto layoutRank = layoutOffsets.size();
  mlir::triton::LinearLayout layout = ttg::isPaddedEncoding(encoding)
                                          ? ttg::paddedLinearLayout(srcTy)
                                          : ttg::toLinearLayout(srcTy);

  MLIRContext *ctx = op->getContext();
  SmallVector<StringAttr> dimNames =
      mlir::triton::standardOutDimNames(ctx, layoutRank);
  SmallVector<std::pair<StringAttr, int32_t>> logicalOffsets;
  logicalOffsets.reserve(layoutRank);
  for (auto &&[dimName, offset] : llvm::zip_equal(dimNames, layoutOffsets)) {
    logicalOffsets.push_back({dimName, static_cast<int32_t>(offset)});
  }

  StringAttr offsetDim = StringAttr::get(ctx, "offset");
  StringAttr blockDim = StringAttr::get(ctx, "block");
  StringAttr partitionDim = StringAttr::get(ctx, "partition");
  mlir::triton::LinearLayout inverse = layout.pseudoinvert();
  auto mapped = inverse.apply(logicalOffsets);
  uint32_t elementOffset = 0;
  uint32_t blockOffset = 0;
  uint32_t partitionOffset = 0;
  for (auto [dim, offset] : mapped) {
    if (dim == offsetDim)
      elementOffset = static_cast<uint32_t>(offset);
    else if (dim == blockDim)
      blockOffset = static_cast<uint32_t>(offset);
    else if (dim == partitionDim)
      partitionOffset = static_cast<uint32_t>(offset);
  }
  uint64_t totalElementOffset = elementOffset;
  if (offsets.size() != layoutRank) {
    uint64_t stride = ttg::getAllocationElems(
        encoding, ttg::dropPipeliningDim(srcTy.getAllocShape(), encoding));
    if (auto partitioned =
            dyn_cast<ttg::PartitionedSharedEncodingAttr>(encoding))
      stride /= partitioned.getNumPartitions();
    totalElementOffset += static_cast<uint64_t>(offsets.front()) * stride;
  }

  uint64_t elementSizeBytes =
      srcTy.getElementType().getIntOrFloatBitWidth() / 8;
  assert(elementSizeBytes > 0 && "element size must be non-zero");
  uint64_t byteOffset = totalElementOffset * elementSizeBytes;

  assert(byteOffset <= std::numeric_limits<uint32_t>::max() &&
         "memdesc_subslice offset exceeds 32-bit range");
  return MemDescSubsliceOffsets{static_cast<uint32_t>(byteOffset),
                                partitionOffset, blockOffset};
}

std::optional<triton::BufferRegionAnalysis::RegionType> getRegionType(Value v) {
  if (isUsedAsBarrier(v)) {
    return triton::BufferRegionAnalysis::RegionType::BARRIER;
  }
  if (isUsedAsSharedMemory(v)) {
    return triton::BufferRegionAnalysis::RegionType::SHARED_MEMORY;
  }
  if (isUsedAsTensorMemory(v)) {
    return triton::BufferRegionAnalysis::RegionType::TENSOR_MEMORY;
  }
  return std::nullopt;
}

} // namespace

namespace mlir::triton {

AddressSet AddressSet::fromRange(uint32_t begin, uint32_t length) {
  uint64_t end = static_cast<uint64_t>(begin) + length;
  assert(end <= std::numeric_limits<uint32_t>::max() &&
         "address range exceeds 32-bit address space");
  AddressSet result;
  for (uint64_t address = begin; address < end; ++address)
    result.set(address);
  return result;
}

AddressSet AddressSet::fromAddresses(ArrayRef<uint32_t> input) {
  AddressSet result;
  for (uint32_t address : input)
    result.set(address);
  return result;
}

void AddressSet::set(uint32_t address) { addresses.set(address); }

void AddressSet::insert(const AddressSet &other) {
  addresses |= other.addresses;
}

AddressSet AddressSet::intersection(const AddressSet &other) const {
  AddressSet result = *this;
  result.addresses &= other.addresses;
  return result;
}

void AddressSet::subtract(const AddressSet &other) {
  addresses.intersectWithComplement(other.addresses);
}

bool AddressSet::contains(uint32_t address) const {
  return addresses.test(address);
}

bool AddressSet::intersects(const AddressSet &other) const {
  return addresses.intersects(other.addresses);
}

bool AddressSet::contains(const AddressSet &other) const {
  return addresses.contains(other.addresses);
}

AddressSet AddressSet::translated(uint32_t delta) const {
  AddressSet result;
  for (uint32_t address : addresses) {
    uint64_t translated = static_cast<uint64_t>(address) + delta;
    assert(translated <= std::numeric_limits<uint32_t>::max() &&
           "translated address set exceeds 32-bit address space");
    result.set(translated);
  }
  return result;
}

BufferStatePlan createBufferStatePlan(ArrayRef<BufferRegion> regions,
                                      bool includeUnknown) {
  BufferStatePlan plan;
  plan.regionMasks.resize(regions.size());

  llvm::SmallBitVector assigned(regions.size());
  SmallVector<SmallVector<unsigned>> components;
  for (unsigned first = 0; first < regions.size(); ++first) {
    if (assigned.test(first) || regions[first].addresses.empty())
      continue;
    SmallVector<unsigned> component;
    SmallVector<unsigned> worklist = {first};
    assigned.set(first);
    while (!worklist.empty()) {
      unsigned current = worklist.pop_back_val();
      component.push_back(current);
      for (unsigned candidate = 0; candidate < regions.size(); ++candidate) {
        if (assigned.test(candidate) || regions[candidate].addresses.empty())
          continue;
        if (!regions[current].addresses.intersects(
                regions[candidate].addresses))
          continue;
        assigned.set(candidate);
        worklist.push_back(candidate);
      }
    }
    llvm::sort(component);
    components.push_back(std::move(component));
  }

  struct ComponentPlan {
    SmallVector<unsigned> regionIds;
    SmallVector<llvm::SmallBitVector> atomMemberships;
  };
  using Atom = std::pair<AddressSet, llvm::SmallBitVector>;
  SmallVector<ComponentPlan> componentPlans;
  for (const SmallVector<unsigned> &component : components) {
    SmallVector<Atom> atoms;
    for (auto [localId, regionId] : llvm::enumerate(component)) {
      AddressSet uncovered = regions[regionId].addresses;
      for (size_t atomId = 0, atomCount = atoms.size();
           atomId < atomCount && !uncovered.empty(); ++atomId) {
        auto &[addresses, atomMembership] = atoms[atomId];
        AddressSet overlap = addresses.intersection(uncovered);
        if (overlap.empty())
          continue;

        uncovered.subtract(overlap);
        if (addresses == overlap) {
          atomMembership.set(localId);
          continue;
        }

        llvm::SmallBitVector membership = atomMembership;
        membership.set(localId);
        addresses.subtract(overlap);
        atoms.push_back({std::move(overlap), std::move(membership)});
      }

      if (!uncovered.empty()) {
        llvm::SmallBitVector membership(component.size());
        membership.set(localId);
        atoms.push_back({std::move(uncovered), std::move(membership)});
      }
    }

    // Preserve the original lane order by each atom's first address.
    llvm::sort(atoms, llvm::less_first());

    SmallVector<llvm::SmallBitVector> atomMemberships;
    atomMemberships.reserve(atoms.size());
    for (Atom &atom : atoms)
      atomMemberships.push_back(std::move(atom.second));

    plan.numLanes += atomMemberships.size();
    componentPlans.push_back({component, std::move(atomMemberships)});
  }

  if (includeUnknown) {
    ++plan.numLanes;
    plan.unknownMask = llvm::SmallBitVector(plan.numLanes, true);
  }

  for (llvm::SmallBitVector &mask : plan.regionMasks)
    mask.resize(plan.numLanes);

  unsigned laneBegin = 0;
  for (const ComponentPlan &componentPlan : componentPlans) {
    for (auto [atomId, membership] :
         llvm::enumerate(componentPlan.atomMemberships)) {
      unsigned lane = laneBegin + atomId;
      for (auto [localId, regionId] :
           llvm::enumerate(componentPlan.regionIds)) {
        if (!membership.test(localId))
          continue;
        plan.regionMasks[regionId].set(lane);
      }
    }

    laneBegin += componentPlan.atomMemberships.size();
  }
  assert(laneBegin + includeUnknown == plan.numLanes);
  return plan;
}

LogicalResult BufferRegionAnalysis::initialize(Operation *top) {
  // Mark all warp-specialize partitions as live.
  LogicalResult status = Base::initialize(top);
  if (failed(status))
    return failure();

  top->walk([&](ttg::WarpSpecializeOp wsOp) {
    for (Region *region : wsOp.getPartitionRegions()) {
      if (region->empty())
        continue;
      Block &entry = region->front();
      auto *exec =
          getOrCreate<dataflow::Executable>(getProgramPointBefore(&entry));
      propagateIfChanged(exec, exec->setToLive());
    }
  });
  return success();
}

LogicalResult BufferRegionAnalysis::visitOperation(
    Operation *op,
    llvm::ArrayRef<const dataflow::Lattice<RegionInfo> *> operands,
    llvm::ArrayRef<dataflow::Lattice<RegionInfo> *> results) {
  RegionInfo regionInfo(RegionInfo::RegionList{});
  auto propagateRegions = [&]() {
    for (auto *result : results)
      propagateIfChanged(result, result->join(regionInfo));
    return success();
  };
  if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(op)) {
    for (Region *region : wsOp.getPartitionRegions()) {
      if (region->empty())
        continue;

      Block &entry = region->front();
      auto *exec =
          getOrCreate<dataflow::Executable>(getProgramPointBefore(&entry));
      propagateIfChanged(exec, exec->setToLive());
    }
    return success();
  }
  if (auto localAllocOp = dyn_cast<ttg::LocalAllocOp>(op)) {
    FailureOr<SmallVector<uint32_t, 2>> offsets =
        getAllocationOffsets(localAllocOp);
    if (failed(offsets))
      return failure();
    ArrayRef<uint32_t> partitionBases = offsets->size() > 1
                                            ? ArrayRef<uint32_t>(*offsets)
                                            : ArrayRef<uint32_t>();
    FailureOr<BufferRegion> region = getMemDescRegion(
        offsets->front(), /*affineOffset=*/0, localAllocOp.getType(), op,
        &footprintCache, partitionBases);
    if (failed(region))
      return failure();
    regionInfo.regions.insert(std::move(*region));

    return propagateRegions();
  }
  if (auto tmemAllocOp = dyn_cast<ttng::TMEMAllocOp>(op)) {
    uint32_t offset = getAllocationOffset(tmemAllocOp);
    FailureOr<BufferRegion> region = getMemDescRegion(
        offset, /*affineOffset=*/0, tmemAllocOp.getType(), op, &footprintCache);
    if (failed(region))
      return failure();
    regionInfo.regions.insert(std::move(*region));

    return propagateRegions();
  }
  if (auto memdescIndexOp = dyn_cast<ttg::MemDescIndexOp>(op)) {
    RegionInfo in = operands[0]->getValue();
    if (in.isUnknown()) {
      regionInfo = in;
      return propagateRegions();
    }
    int numSubBuffers = getNumBuffers(memdescIndexOp);
    int firstSubBuffer = 0;
    int endSubBuffer = numSubBuffers;
    APInt constantIndex;
    if (matchPattern(memdescIndexOp.getIndex(),
                     m_ConstantInt(&constantIndex))) {
      int64_t index = constantIndex.getSExtValue();
      firstSubBuffer = index;
      endSubBuffer = index + 1;
    }
    for (auto &region : in.regions) {
      for (int i = firstSubBuffer; i < endSubBuffer; ++i) {
        uint32_t stageOffset =
            getMemDescStorageOffset(memdescIndexOp.getType(), i);
        uint32_t storageBase = region.storageBase + stageOffset;
        SmallVector<uint32_t, 2> partitionBases =
            advancePartitionBases(region.partitionBases, stageOffset);
        FailureOr<BufferRegion> subBuffer = getMemDescRegion(
            storageBase, region.affineOffset, memdescIndexOp.getType(), op,
            &footprintCache, partitionBases, region.affinePartitionOffset,
            region.affineCTAOffset);
        if (failed(subBuffer))
          return failure();
        regionInfo.regions.insert(std::move(*subBuffer));
      }
    }

    return propagateRegions();
  }
  if (auto memdescSubsliceOp = dyn_cast<ttg::MemDescSubsliceOp>(op)) {
    RegionInfo in = operands[0]->getValue();
    if (in.isUnknown()) {
      regionInfo = in;
      return propagateRegions();
    }
    MemDescSubsliceOffsets relativeOffset =
        getMemDescSubsliceUnpaddedOffsets(memdescSubsliceOp);
    for (auto &region : in.regions) {
      uint32_t affineOffset = region.affineOffset ^ relativeOffset.byteOffset;
      uint32_t affinePartitionOffset =
          region.affinePartitionOffset ^ relativeOffset.partitionOffset;
      uint32_t affineCTAOffset =
          region.affineCTAOffset ^ relativeOffset.ctaOffset;
      FailureOr<BufferRegion> subBuffer = getMemDescRegion(
          region.storageBase, affineOffset, memdescSubsliceOp.getType(), op,
          &footprintCache, region.partitionBases, affinePartitionOffset,
          affineCTAOffset);
      if (failed(subBuffer))
        return failure();
      regionInfo.regions.insert(std::move(*subBuffer));
    }
    return propagateRegions();
  }
  if (auto tmemSubsliceOp = dyn_cast<ttng::TMEMSubSliceOp>(op)) {
    RegionInfo in = operands[0]->getValue();
    if (in.isUnknown()) {
      regionInfo = in;
      return propagateRegions();
    }
    uint32_t relativeOffset = ttng::getTMemSubSliceOffset(
        tmemSubsliceOp.getSrc().getType(), tmemSubsliceOp.getOffset(),
        tmemSubsliceOp.getDim());
    for (auto &region : in.regions) {
      uint32_t affineOffset = region.affineOffset + relativeOffset;
      FailureOr<BufferRegion> subBuffer = getMemDescRegion(
          region.storageBase, affineOffset, tmemSubsliceOp.getType(), op,
          &footprintCache, region.partitionBases, region.affinePartitionOffset,
          region.affineCTAOffset);
      if (failed(subBuffer))
        return failure();
      regionInfo.regions.insert(std::move(*subBuffer));
    }
    return propagateRegions();
  }
  if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
    if (isa<ttg::MemDescType>(selectOp.getType())) {
      regionInfo =
          RegionInfo::join(operands[1]->getValue(), operands[2]->getValue());
      return propagateRegions();
    }
  }
  if (auto reinterpretOp = dyn_cast<ttg::MemDescReinterpretOp>(op)) {
    RegionInfo in = operands[0]->getValue();
    if (in.isUnknown()) {
      regionInfo = in;
      return propagateRegions();
    }
    for (auto &region : in.regions) {
      FailureOr<BufferRegion> reinterpreted = getMemDescRegion(
          region.storageBase, region.affineOffset, reinterpretOp.getType(), op,
          &footprintCache, region.partitionBases, region.affinePartitionOffset,
          region.affineCTAOffset);
      if (failed(reinterpreted))
        return failure();
      regionInfo.regions.insert(std::move(*reinterpreted));
    }
    return propagateRegions();
  }
  if (isa<ttg::MemDescTransOp, ttg::MemDescReshapeOp>(op)) {
    regionInfo = operands[0]->getValue();
    return propagateRegions();
  }
  if (isa<ttng::WarpGroupDotWaitOp>(op)) {
    for (auto [operand, result] : llvm::zip_equal(operands, results))
      propagateIfChanged(result, result->join(operand->getValue()));
    return success();
  }
  for (auto [result, lattice] : llvm::zip_equal(op->getResults(), results)) {
    if (isa<ttg::MemDescType>(result.getType()))
      propagateIfChanged(
          lattice, lattice->join(RegionInfo::getPessimisticValueState(result)));
  }
  return success();
}

void BufferRegionAnalysis::calculateUsedBufferRegions(Operation *op) {
  op->walk([&](Operation *op) {
    for (const MemoryAccess &access : getMemoryAccesses(op)) {
      std::optional<RegionType> regionType = getRegionType(access.value);
      if (!regionType)
        continue;
      const RegionInfo &regionInfo =
          getLatticeElement(access.value)->getValue();
      if (regionInfo.isUnknown())
        usedUnknownBufferRegions[*regionType] = true;
      usedBufferRegions[*regionType].insert(regionInfo.regions.begin(),
                                            regionInfo.regions.end());
    }
  });
}

SmallVector<BufferRegionAnalysis::MemoryAccess>
BufferRegionAnalysis::getMemoryAccesses(Operation *op) {
  SmallVector<MemoryAccess> accesses;
  auto memoryEffects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memoryEffects)
    return accesses;

  SmallVector<MemoryEffects::EffectInstance> effects;
  memoryEffects.getEffects(effects);
  for (const MemoryEffects::EffectInstance &effect : effects) {
    bool isWrite = isa<MemoryEffects::Write>(effect.getEffect());
    if (!isWrite && !isa<MemoryEffects::Read>(effect.getEffect()))
      continue;
    if (effect.getResource() != ttg::SharedMemory::get() &&
        effect.getResource() != ttng::TensorMemory::get())
      continue;
    Value value = effect.getValue();
    if (!value || !isa<ttg::MemDescType>(value.getType()))
      continue;
    auto existing = llvm::find_if(accesses, [&](const MemoryAccess &access) {
      return access.value == value;
    });
    if (existing == accesses.end())
      accesses.push_back({value, isWrite});
    else
      existing->isWrite |= isWrite;
  }
  return accesses;
}

bool BufferRegionAnalysis::isMemoryAccessOperation(Operation *op) {
  return !getMemoryAccesses(op).empty();
}

} // namespace mlir::triton
