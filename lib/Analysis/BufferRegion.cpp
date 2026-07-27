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
  if (auto array = dyn_cast<ArrayAttr>(offsetAttr))
    for (Attribute offset : array)
      offsets.push_back(cast<IntegerAttr>(offset).getInt());
  else
    offsets.push_back(cast<IntegerAttr>(offsetAttr).getInt());
  return offsets;
}

SmallVector<uint32_t, 2> advancePartitionBases(ArrayRef<uint32_t> bases,
                                               uint32_t offset) {
  return llvm::to_vector<2>(
      llvm::map_range(bases, [=](uint32_t base) { return base + offset; }));
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

using MemDescFootprint = SmallVector<triton::BufferRegion::CTAAddresses, 2>;

triton::AddressSet &getAddressesForCTA(MemDescFootprint &footprint,
                                       uint32_t cta) {
  auto it = llvm::partition_point(
      footprint, [cta](const auto &entry) { return entry.first < cta; });
  if (it == footprint.end() || it->first != cta)
    it = footprint.insert(it, triton::BufferRegion::CTAAddresses{cta, {}});
  return it->second;
}

MemDescFootprint getMemDescAddresses(
    uint32_t storageBase, uint32_t affineOffset, ttg::MemDescType ty,
    llvm::DenseMap<std::pair<Type, uint32_t>, triton::AddressSet> *cache =
        nullptr,
    ArrayRef<uint32_t> partitionBases = {}, uint32_t affinePartitionOffset = 0,
    uint32_t affineCTAOffset = 0) {
  bool isTmem = isa<ttng::TensorMemorySpaceAttr>(ty.getMemorySpace());
  auto collectPages = [&]() {
    ttg::MemDescType pageTy =
        ty.cloneWith(ty.getShape().drop_front(), ty.getElementType());
    MemDescFootprint footprint;
    for (int64_t page = 0; page < ty.getDimSize(0); ++page) {
      uint32_t pageOffset = getMemDescStorageOffset(pageTy, page);
      MemDescFootprint pageFootprint = getMemDescAddresses(
          storageBase + pageOffset, affineOffset, pageTy, /*cache=*/nullptr,
          advancePartitionBases(partitionBases, pageOffset),
          affinePartitionOffset, affineCTAOffset);
      for (const auto &[cta, addresses] : pageFootprint)
        getAddressesForCTA(footprint, cta).insert(addresses);
    }
    return footprint;
  };
  if (cast<ttg::LayoutEncodingTrait>(ty.getEncoding()).getRank() !=
      ty.getRank())
    return collectPages();
  triton::LinearLayout layout = ttg::isPaddedEncoding(ty.getEncoding())
                                    ? ttg::paddedLinearLayout(ty)
                                    : ttg::toLinearLayout(ty);
  if (llvm::size(layout.getOutDimNames()) != ty.getRank())
    return collectPages();
  triton::LinearLayout inverse = layout.pseudoinvert();
  MLIRContext *ctx = ty.getContext();
  SmallVector<StringAttr> dims = triton::standardOutDimNames(ctx, ty.getRank());
  ArrayRef<int64_t> shape = ty.getShape();
  uint64_t numPoints = product(shape);
  uint32_t bitWidth = ty.getElementTypeBitWidth();

  StringAttr offsetName = StringAttr::get(ctx, "offset");
  StringAttr blockName = StringAttr::get(ctx, "block");
  StringAttr partitionName = StringAttr::get(ctx, "partition");
  StringAttr rowName = StringAttr::get(ctx, "row");
  StringAttr colName = StringAttr::get(ctx, "col");

  MemDescFootprint footprint;
  auto addPhysicalAddress = [&](uint32_t offset, uint32_t row, uint32_t col,
                                uint32_t partition, uint32_t block) {
    uint32_t cta = block ^ affineCTAOffset;
    auto &addresses = getAddressesForCTA(footprint, cta);
    if (isTmem) {
      uint64_t bitBegin = static_cast<uint64_t>(col) * bitWidth;
      uint32_t firstWord = bitBegin / 32;
      uint32_t lastWord = llvm::divideCeil(bitBegin + bitWidth, uint64_t{32});
      uint32_t relative = (row << 16) | firstWord;
      uint64_t begin =
          static_cast<uint64_t>(storageBase) + affineOffset + relative;
      for (uint32_t word = firstWord; word < lastWord; ++word) {
        uint64_t address = begin + (word - firstWord);
        assert(address <= std::numeric_limits<uint32_t>::max());
        addresses.set(address);
      }
    } else {
      uint32_t base = storageBase;
      if (!partitionBases.empty())
        base = partitionBases[partition ^ affinePartitionOffset];
      uint32_t relative = offset * (bitWidth / 8);
      uint32_t combined = affineOffset ^ relative;
      uint64_t begin =
          static_cast<uint64_t>(base) + applySharedPadding(combined, ty);
      for (uint32_t byte = 0; byte < bitWidth / 8; ++byte) {
        assert(begin + byte <= std::numeric_limits<uint32_t>::max());
        addresses.set(begin + byte);
      }
    }
  };

  struct PhysicalBasis {
    uint32_t offset = 0;
    uint32_t row = 0;
    uint32_t col = 0;
    uint32_t partition = 0;
    uint32_t block = 0;
  };
  SmallVector<PhysicalBasis> bases;
  bool hasCTAAddressVariation = false;
  for (auto dimAndSize : llvm::zip_equal(dims, shape)) {
    auto dim = std::get<0>(dimAndSize);
    auto dimSize = std::get<1>(dimAndSize);
    unsigned numBits = llvm::Log2_64(dimSize);
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
    auto [found, inserted] =
        cache->try_emplace(std::make_pair(Type(ty), affineOffset));
    if (inserted)
      found->second =
          std::move(getMemDescAddresses(0, affineOffset, ty).front().second);
    footprint.emplace_back(affineCTAOffset,
                           found->second.translated(storageBase));
    return footprint;
  }

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
    addPhysicalAddress(physical.offset, physical.row, physical.col,
                       physical.partition, physical.block);
  }
  return footprint;
}

triton::BufferRegionView getMemDescView(
    uint32_t storageBase, uint32_t affineOffset, ttg::MemDescType ty,
    llvm::DenseMap<std::pair<Type, uint32_t>, triton::AddressSet> *cache,
    ArrayRef<uint32_t> partitionBases = {}, uint32_t affinePartitionOffset = 0,
    uint32_t affineCTAOffset = 0) {
  MemDescFootprint footprint =
      getMemDescAddresses(storageBase, affineOffset, ty, cache, partitionBases,
                          affinePartitionOffset, affineCTAOffset);
  uint32_t runtimeStorageBase = partitionBases.empty()
                                    ? storageBase
                                    : partitionBases[affinePartitionOffset];
  uint32_t baseOffset = runtimeStorageBase +
                        (isa<ttng::TensorMemorySpaceAttr>(ty.getMemorySpace())
                             ? affineOffset
                             : applySharedPadding(affineOffset, ty));
  return {{baseOffset, getMemDescSize(ty), std::move(footprint)},
          storageBase,
          affineOffset,
          llvm::to_vector<2>(partitionBases),
          affinePartitionOffset,
          affineCTAOffset};
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
  uint64_t elementOffset = 0;
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
  if (offsets.size() != layoutRank) {
    uint64_t stride = ttg::getAllocationElems(
        encoding, ttg::dropPipeliningDim(srcTy.getAllocShape(), encoding));
    if (auto partitioned =
            dyn_cast<ttg::PartitionedSharedEncodingAttr>(encoding))
      stride /= partitioned.getNumPartitions();
    elementOffset += static_cast<uint64_t>(offsets.front()) * stride;
  }

  uint64_t elementSizeBytes =
      srcTy.getElementType().getIntOrFloatBitWidth() / 8;
  assert(elementSizeBytes > 0 && "element size must be non-zero");
  uint64_t byteOffset = elementOffset * elementSizeBytes;

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

  SmallVector<AddressSet> projectedAddresses(regions.size());
  for (auto [region, projected] : llvm::zip(regions, projectedAddresses))
    for (const auto &[cta, addresses] : region.ctaAddresses)
      projected.insert(addresses);

  llvm::SmallBitVector assigned(regions.size());
  SmallVector<SmallVector<unsigned>> components;
  for (unsigned first = 0; first < regions.size(); ++first) {
    if (assigned.test(first) || projectedAddresses[first].empty())
      continue;
    SmallVector<unsigned> component;
    SmallVector<unsigned> worklist = {first};
    assigned.set(first);
    while (!worklist.empty()) {
      unsigned current = worklist.pop_back_val();
      component.push_back(current);
      for (unsigned candidate = 0; candidate < regions.size(); ++candidate) {
        if (assigned.test(candidate) || !projectedAddresses[current].intersects(
                                            projectedAddresses[candidate]))
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
      AddressSet uncovered = projectedAddresses[regionId];
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

  if (includeUnknown)
    plan.unknownMask = llvm::SmallBitVector(++plan.numLanes, true);

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
  RegionInfo regionInfo(RegionInfo::ViewList{});
  auto propagateRegions = [&](const RegionInfo &info) {
    for (auto *result : results)
      propagateIfChanged(result, result->join(info));
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
    regionInfo.views.insert(getMemDescView(offsets->front(), /*affineOffset=*/0,
                                           localAllocOp.getType(),
                                           &footprintCache, partitionBases));
    return propagateRegions(regionInfo);
  }
  if (auto tmemAllocOp = dyn_cast<ttng::TMEMAllocOp>(op)) {
    regionInfo.views.insert(
        getMemDescView(getAllocationOffset(tmemAllocOp), /*affineOffset=*/0,
                       tmemAllocOp.getType(), &footprintCache));
    return propagateRegions(regionInfo);
  }
  if (auto memdescIndexOp = dyn_cast<ttg::MemDescIndexOp>(op)) {
    const RegionInfo &in = operands[0]->getValue();
    if (in.isUnknown())
      return propagateRegions(in);
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
    for (const BufferRegionView &view : in.views) {
      for (int i = firstSubBuffer; i < endSubBuffer; ++i) {
        uint32_t stageOffset =
            getMemDescStorageOffset(memdescIndexOp.getType(), i);
        regionInfo.views.insert(getMemDescView(
            view.storageBase + stageOffset, view.affineOffset,
            memdescIndexOp.getType(), &footprintCache,
            advancePartitionBases(view.partitionBases, stageOffset),
            view.affinePartitionOffset, view.affineCTAOffset));
      }
    }

    return propagateRegions(regionInfo);
  }
  if (auto memdescSubsliceOp = dyn_cast<ttg::MemDescSubsliceOp>(op)) {
    const RegionInfo &in = operands[0]->getValue();
    if (in.isUnknown())
      return propagateRegions(in);
    MemDescSubsliceOffsets relativeOffset =
        getMemDescSubsliceUnpaddedOffsets(memdescSubsliceOp);
    for (const BufferRegionView &view : in.views)
      regionInfo.views.insert(getMemDescView(
          view.storageBase, view.affineOffset ^ relativeOffset.byteOffset,
          memdescSubsliceOp.getType(), &footprintCache, view.partitionBases,
          view.affinePartitionOffset ^ relativeOffset.partitionOffset,
          view.affineCTAOffset ^ relativeOffset.ctaOffset));
    return propagateRegions(regionInfo);
  }
  if (auto tmemSubsliceOp = dyn_cast<ttng::TMEMSubSliceOp>(op)) {
    const RegionInfo &in = operands[0]->getValue();
    if (in.isUnknown())
      return propagateRegions(in);
    uint32_t relativeOffset = ttng::getTMemSubSliceOffset(
        tmemSubsliceOp.getSrc().getType(), tmemSubsliceOp.getOffset(),
        tmemSubsliceOp.getDim());
    for (const BufferRegionView &view : in.views)
      regionInfo.views.insert(getMemDescView(
          view.storageBase, view.affineOffset + relativeOffset,
          tmemSubsliceOp.getType(), &footprintCache, view.partitionBases,
          view.affinePartitionOffset, view.affineCTAOffset));
    return propagateRegions(regionInfo);
  }
  if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
    if (isa<ttg::MemDescType>(selectOp.getType())) {
      regionInfo =
          RegionInfo::join(operands[1]->getValue(), operands[2]->getValue());
      return propagateRegions(regionInfo);
    }
  }
  if (auto reinterpretOp = dyn_cast<ttg::MemDescReinterpretOp>(op)) {
    const RegionInfo &in = operands[0]->getValue();
    if (in.isUnknown())
      return propagateRegions(in);
    for (const BufferRegionView &view : in.views)
      regionInfo.views.insert(getMemDescView(
          view.storageBase, view.affineOffset, reinterpretOp.getType(),
          &footprintCache, view.partitionBases, view.affinePartitionOffset,
          view.affineCTAOffset));
    return propagateRegions(regionInfo);
  }
  if (isa<ttg::MemDescTransOp, ttg::MemDescReshapeOp>(op))
    return propagateRegions(operands[0]->getValue());
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
      for (const BufferRegionView &view : regionInfo.views)
        usedBufferRegions[*regionType].insert(view.region);
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

} // namespace mlir::triton
