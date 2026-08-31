#include "triton/Analysis/BufferRegion.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/Allocation.h"
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
triton::LinearLayout getMemDescLinearLayout(ttg::MemDescType ty) {
  if (ttg::isPaddedEncoding(ty.getEncoding()))
    return ttg::paddedLinearLayout(ty);
  auto shape = ttg::dropPipeliningDim(ty.getAllocShape(), ty.getEncoding());
  if (!ttg::isPositivePowerOfTwoShape(shape))
    return ttg::toLinearLayoutWithPow2Shape(ty);
  return ttg::toLinearLayout(ty);
}

FailureOr<SmallVector<uint32_t, 2>>
getAllocationOffsets(ttg::LocalAllocOp op, Allocation *allocation) {
  if (allocation) {
    SmallVector<uint32_t, 2> offsets;
    for (auto id : allocation->getBufferIds(op.getResult()))
      offsets.push_back(allocation->getOffset(id));
    assert(!offsets.empty() && "shared allocation must have allocated buffers");
    return offsets;
  }
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

uint32_t getAllocationOffset(ttng::TMEMAllocOp op) {
  auto colOffsetAttr = op->getAttr("tensor_memory_col_offset");
  auto rowOffsetAttr = op->getAttr("tensor_memory_row_offset");
  if (!colOffsetAttr || !rowOffsetAttr) {
    llvm::report_fatal_error(
        "ConcurrencySanitizer should run after AllocateSharedMemory and "
        "TensorMemoryAllocation pass.");
  }
  uint32_t colOffset = cast<IntegerAttr>(colOffsetAttr).getInt();
  uint32_t rowOffset = cast<IntegerAttr>(rowOffsetAttr).getInt();
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
  return numElems * getIntOrFloatOrPtrBitWidth(ty.getElementType()) / 8;
}

uint32_t applySharedPadding(uint32_t byteOffset, ttg::MemDescType ty) {
  auto padded = ttg::getPaddedEncoding(ty.getEncoding());
  if (!padded)
    return byteOffset;
  uint32_t elementSize = getIntOrFloatOrPtrBitWidth(ty.getElementType()) / 8;
  uint32_t elementOffset = byteOffset / elementSize;
  return (padded.getPaddedSize({elementOffset + 1}) - 1) * elementSize +
         byteOffset % elementSize;
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
    llvm::DenseMap<std::pair<Type, uint32_t>, MemDescFootprint> *cache =
        nullptr,
    ArrayRef<uint32_t> partitionBases = {}, uint32_t affinePartitionOffset = 0,
    uint32_t affineCTAOffset = 0) {
  bool isTmem = isa<ttng::TensorMemorySpaceAttr>(ty.getMemorySpace());
  if (cast<ttg::LayoutEncodingTrait>(ty.getEncoding()).getRank() !=
      ty.getRank()) {
    ttg::MemDescType pageTy =
        ty.cloneWith(ty.getShape().drop_front(), ty.getElementType());
    MemDescFootprint footprint;
    for (int64_t page = 0; page < ty.getDimSize(0); ++page) {
      uint32_t pageOffset = getMemDescStorageOffset(pageTy, page);
      for (const auto &[cta, addresses] : getMemDescAddresses(
               storageBase + pageOffset, affineOffset, pageTy,
               /*cache=*/nullptr,
               advancePartitionBases(partitionBases, pageOffset),
               affinePartitionOffset, affineCTAOffset))
        getAddressesForCTA(footprint, cta).insert(addresses);
    }
    return footprint;
  }
  if (cache && partitionBases.empty()) {
    auto [found, inserted] =
        cache->try_emplace(std::make_pair(Type(ty), affineOffset));
    if (inserted)
      found->second = getMemDescAddresses(0, affineOffset, ty);
    MemDescFootprint footprint;
    for (const auto &[cta, addresses] : found->second)
      getAddressesForCTA(footprint, cta ^ affineCTAOffset) =
          addresses.translated(storageBase);
    return footprint;
  }
  triton::LinearLayout layout = getMemDescLinearLayout(ty);
  triton::LinearLayout inverse = layout.pseudoinvert();
  MLIRContext *ctx = ty.getContext();
  SmallVector<StringAttr> dims = triton::standardOutDimNames(ctx, ty.getRank());
  ArrayRef<int64_t> shape = ty.getShape();
  uint64_t numPoints = product(shape);
  uint32_t bitWidth = getIntOrFloatOrPtrBitWidth(ty.getElementType());

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
      uint32_t bitBegin = col * bitWidth;
      uint32_t firstWord = bitBegin / 32;
      uint32_t lastWord = llvm::divideCeil(bitBegin + bitWidth, uint32_t{32});
      uint32_t relative = (row << 16) | firstWord;
      uint32_t begin = storageBase + affineOffset + relative;
      for (uint32_t word = firstWord; word < lastWord; ++word)
        addresses.set(begin + word - firstWord);
    } else {
      uint32_t base = storageBase;
      if (!partitionBases.empty())
        base = partitionBases[partition ^ affinePartitionOffset];
      uint32_t relative = offset * (bitWidth / 8);
      uint32_t combined = affineOffset ^ relative;
      uint32_t begin = base + applySharedPadding(combined, ty);
      for (uint32_t byte = 0; byte < bitWidth / 8; ++byte)
        addresses.set(begin + byte);
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
  SmallVector<unsigned> basisCounts;
  for (auto [dim, dimSize] : llvm::zip_equal(dims, shape)) {
    unsigned numBits = llvm::Log2_64_Ceil(dimSize);
    basisCounts.push_back(numBits);
    for (unsigned bit = 0; bit < numBits; ++bit) {
      auto basis = [&, dim = dim](StringAttr name) {
        return inverse.hasOutDim(name)
                   ? static_cast<uint32_t>(inverse.getBasis(dim, bit, name))
                   : 0;
      };
      uint32_t block = basis(blockName);
      bases.push_back({basis(offsetName), basis(rowName), basis(colName),
                       basis(partitionName), block});
    }
  }
  unsigned numLogicalBases = bases.size();
  uint64_t numReplicas = 1;
  // The pseudoinverse selects one representative of replicated storage. Zero
  // block bases in the allocation layout denote a local copy in every replica
  // CTA. A nonzero block basis excluded by a subview is not a replica.
  uint32_t broadcastBits = layout.getFreeVariableMasks().lookup(blockName);
  for (unsigned bit = 0; bit < layout.getInDimSizeLog2(blockName); ++bit) {
    if (!(broadcastBits & (uint32_t{1} << bit)))
      continue;
    PhysicalBasis replica;
    replica.block = uint32_t{1} << bit;
    bases.push_back(replica);
    numReplicas *= 2;
  }
  if (isTmem) {
    // Zero row bases at 32 and 64 broadcast across warp-addressable storage;
    // loads/stores still access every replica. The pseudoinverse chooses only
    // one. Other zero row/column bases denote undefined storage, not replicas.
    uint64_t rowBasisMask = triton::getInputBasisMask(layout, rowName, dims);
    for (unsigned bit : {5, 6}) {
      if (!(rowBasisMask & (uint64_t{1} << bit))) {
        PhysicalBasis replica;
        replica.row = uint32_t{1} << bit;
        bases.push_back(replica);
        numReplicas *= 2;
      }
    }
  }

  auto applyBasis = [](PhysicalBasis &physical, const PhysicalBasis &basis) {
    physical.offset ^= basis.offset;
    physical.row ^= basis.row;
    physical.col ^= basis.col;
    physical.partition ^= basis.partition;
    physical.block ^= basis.block;
  };

  if (ttg::isPositivePowerOfTwoShape(shape)) {
    PhysicalBasis physical;
    for (uint64_t index = 0; index < numPoints * numReplicas; ++index) {
      if (index != 0)
        applyBasis(physical, bases[llvm::countr_zero(index)]);
      addPhysicalAddress(physical.offset, physical.row, physical.col,
                         physical.partition, physical.block);
    }
  } else {
    // A flattened Gray-code traversal only describes power-of-two boxes.
    // Enumerate logical coordinates for a non-power-of-two memdesc and apply
    // the corresponding inverse-layout bases explicitly.
    for (uint64_t index = 0; index < numPoints; ++index) {
      uint64_t remaining = index;
      unsigned basisStart = 0;
      PhysicalBasis physical;
      for (auto [dimSize, numBits] : llvm::zip_equal(shape, basisCounts)) {
        uint64_t coordinate = remaining % dimSize;
        remaining /= dimSize;
        for (unsigned bit = 0; bit < numBits; ++bit)
          if (coordinate & (uint64_t{1} << bit))
            applyBasis(physical, bases[basisStart + bit]);
        basisStart += numBits;
      }
      for (uint64_t replica = 0; replica < numReplicas; ++replica) {
        PhysicalBasis replicated = physical;
        for (unsigned bit = 0; bit < llvm::Log2_64(numReplicas); ++bit)
          if (replica & (uint64_t{1} << bit))
            applyBasis(replicated, bases[numLogicalBases + bit]);
        addPhysicalAddress(replicated.offset, replicated.row, replicated.col,
                           replicated.partition, replicated.block);
      }
    }
  }
  return footprint;
}

uint32_t getMemDescStorageOffset(ttg::MemDescType ty, unsigned index) {
  if (isa<ttng::TensorMemorySpaceAttr>(ty.getMemorySpace()))
    return index * ttng::getTmemAllocSizes(ty).numCols;
  uint32_t elems = ttg::getAllocationElems(ty.getEncoding(), ty.getShape(),
                                           ty.getAllocShape());
  if (auto partitioned =
          dyn_cast<ttg::PartitionedSharedEncodingAttr>(ty.getEncoding()))
    elems /= partitioned.getNumPartitions();
  uint32_t elementSize = getIntOrFloatOrPtrBitWidth(ty.getElementType()) / 8;
  return applySharedPadding(index * elems * elementSize, ty);
}

struct MemDescSubsliceOffsets {
  uint32_t storageOffset = 0;
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
  mlir::triton::LinearLayout layout = getMemDescLinearLayout(srcTy);

  MLIRContext *ctx = op->getContext();
  SmallVector<StringAttr> dimNames =
      mlir::triton::standardOutDimNames(ctx, layoutRank);
  SmallVector<std::pair<StringAttr, int32_t>> logicalOffsets;
  logicalOffsets.reserve(layoutRank);
  for (auto &&[dimName, offset] : llvm::zip_equal(dimNames, layoutOffsets))
    logicalOffsets.emplace_back(dimName, static_cast<int32_t>(offset));

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
  uint32_t storageElementOffset = 0;
  if (offsets.size() != layoutRank) {
    uint32_t stride = ttg::getAllocationElems(
        encoding, ttg::dropPipeliningDim(srcTy.getAllocShape(), encoding));
    if (auto partitioned =
            dyn_cast<ttg::PartitionedSharedEncodingAttr>(encoding))
      stride /= partitioned.getNumPartitions();
    // The pipeline prefix advances every base pointer by addition. Only the
    // layout-ranked suffix composes by XOR; nested prefix offsets may carry.
    // Padded pipeline subslices are rejected by the verifier.
    storageElementOffset = offsets.front() * stride;
  }

  uint32_t elementSizeBytes =
      getIntOrFloatOrPtrBitWidth(srcTy.getElementType()) / 8;
  assert(elementSizeBytes > 0 && "element size must be non-zero");
  return MemDescSubsliceOffsets{storageElementOffset * elementSizeBytes,
                                elementOffset * elementSizeBytes,
                                partitionOffset, blockOffset};
}

} // namespace

namespace mlir::triton {

AddressSet AddressSet::fromRange(uint32_t begin, uint32_t length) {
  AddressSet result;
  for (uint32_t offset = 0; offset < length; ++offset)
    result.set(begin + offset);
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
  for (uint32_t address : addresses)
    result.set(address + delta);
  return result;
}

BufferRegionView
BufferRegionView::translated(uint32_t offset,
                             uint32_t newAllocationFrame) const {
  BufferRegionView result = *this;
  result.region.baseOffset += offset;
  for (auto &[cta, addresses] : result.region.ctaAddresses)
    addresses = addresses.translated(offset);
  result.storageBase += offset;
  for (uint32_t &base : result.partitionBases)
    base += offset;
  result.allocationFrame = newAllocationFrame;
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

BufferRegionView
BufferRegionAnalysis::getAllocView(Value allocation, uint32_t storageBase,
                                   ArrayRef<uint32_t> partitionBases) {
  BufferRegionView view;
  view.storageBase = storageBase;
  view.partitionBases = llvm::to_vector<2>(partitionBases);
  Operation *op = allocation.getDefiningOp();
  view.allocation = op;
  // TMEM allocation assigns module-wide addresses; only shared allocations
  // need a per-function frame and translation by a call's shared-memory offset.
  auto memory = cast<ttg::MemDescType>(allocation.getType());
  view.allocationFrame =
      isa<ttng::TensorMemorySpaceAttr>(memory.getMemorySpace())
          ? getOperationId(op->getParentOfType<ModuleOp>())
          : getOperationId(op->getParentOfType<FunctionOpInterface>());
  return getSubView(allocation.getType(), view);
}

BufferRegionView
BufferRegionAnalysis::getSubView(Type type, const BufferRegionView &view,
                                 uint32_t storageOffset, uint32_t byteOffset,
                                 uint32_t partitionOffset, uint32_t ctaOffset) {
  auto ty = cast<ttg::MemDescType>(type);
  uint32_t storageBase = view.storageBase + storageOffset;
  SmallVector<uint32_t, 2> partitionBases =
      advancePartitionBases(view.partitionBases, storageOffset);
  uint32_t affineOffset = isa<ttng::TensorMemorySpaceAttr>(ty.getMemorySpace())
                              ? view.affineOffset + byteOffset
                              : view.affineOffset ^ byteOffset;
  uint32_t affinePartitionOffset = view.affinePartitionOffset ^ partitionOffset;
  uint32_t affineCTAOffset = view.affineCTAOffset ^ ctaOffset;
  MemDescFootprint footprint = getMemDescAddresses(
      storageBase, affineOffset, ty, &footprintCache, partitionBases,
      affinePartitionOffset, affineCTAOffset);
  uint32_t runtimeStorageBase = partitionBases.empty()
                                    ? storageBase
                                    : partitionBases[affinePartitionOffset];
  uint32_t baseOffset = runtimeStorageBase +
                        (isa<ttng::TensorMemorySpaceAttr>(ty.getMemorySpace())
                             ? affineOffset
                             : applySharedPadding(affineOffset, ty));
  BufferRegionView result = view;
  result.region = {baseOffset, getMemDescSize(ty), std::move(footprint)};
  result.storageBase = storageBase;
  result.affineOffset = affineOffset;
  result.partitionBases = std::move(partitionBases);
  result.affinePartitionOffset = affinePartitionOffset;
  result.affineCTAOffset = affineCTAOffset;
  return result;
}

LogicalResult BufferRegionAnalysis::initialize(Operation *top) {
  top->walk([&](Operation *operation) {
    if (isa<ModuleOp, FunctionOpInterface, CallOpInterface>(operation))
      operationInterner.insert(operation);
  });

  // Mark all warp-specialize partitions as live.
  if (failed(Base::initialize(top)))
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

bool mayOverlap(const BufferRegionFootprint *lhs,
                const BufferRegionFootprint *rhs) {
  if (!lhs || !rhs || !lhs->memorySpace || !rhs->memorySpace)
    return true;
  if (lhs->memorySpace != rhs->memorySpace)
    return false;
  const RegionInfo &left = lhs->regionInfo;
  const RegionInfo &right = rhs->regionInfo;
  if (left.kind != RegionInfo::Kind::Exact || left.views.empty() ||
      right.kind != RegionInfo::Kind::Exact || right.views.empty())
    return true;
  return llvm::any_of(left.views, [&](const BufferRegionView &a) {
    return llvm::any_of(right.views, [&](const BufferRegionView &b) {
      bool sameFrame =
          a.allocationFrame && a.allocationFrame == b.allocationFrame;
      return !sameFrame || a.region.intersects(b.region);
    });
  });
}

const BufferRegionFootprint *
BufferRegionAnalysis::getFootprint(Value value,
                                   FunctionOpInterface allocationFrame) {
  auto memory = dyn_cast<ttg::MemDescType>(value.getType());
  if (!memory)
    return nullptr;
  auto [it, inserted] = valueFootprints.try_emplace(value);
  if (inserted) {
    const RegionInfo &info = getRegionInfo(value);
    if (info.kind == RegionInfo::Kind::Exact && !info.views.empty())
      it->second = std::make_unique<BufferRegionFootprint>(
          BufferRegionFootprint{memory.getMemorySpace(), info});
  }
  const auto *footprint = it->second.get();
  if (footprint && allocationFrame &&
      llvm::any_of(footprint->regionInfo.views, [&](const auto &view) {
        return view.allocationFrame != getOperationId(allocationFrame);
      }))
    return nullptr;
  return footprint;
}

const BufferRegionFootprint *
BufferRegionAnalysis::getScratchFootprint(Operation *op) {
  auto [it, inserted] = scratchFootprints.try_emplace(op);
  if (!inserted)
    return it->second.get();

  uint32_t base, length;
  if (auto *allocation = getAllocation(op)) {
    auto id = allocation->getBufferId(op);
    if (id == Allocation::InvalidBufferId)
      return nullptr;
    base = allocation->getOffset(id);
    length = allocation->getAllocatedSize(id);
  } else {
    auto offset = op->getAttrOfType<IntegerAttr>("allocation.offset");
    auto size = op->getAttrOfType<IntegerAttr>("allocation.size");
    if (!offset || !size)
      return nullptr;
    base = offset.getInt();
    length = size.getInt();
  }
  BufferRegionView view{{base, length}, /*storageBase=*/base};
  view.allocationFrame =
      getOperationId(op->getParentOfType<FunctionOpInterface>());
  AddressSet addresses = AddressSet::fromRange(base, length);
  unsigned numCTAs = ttg::lookupNumCTAs(op);
  uint32_t broadcastMask = getAtomicScratchBroadcastMask(op).value_or(0);
  for (unsigned cta = 0; cta < numCTAs; ++cta)
    if (!(cta & broadcastMask))
      view.region.ctaAddresses.emplace_back(cta, addresses);
  it->second = std::make_unique<BufferRegionFootprint>(
      BufferRegionFootprint{ttg::SharedMemorySpaceAttr::get(op->getContext()),
                            RegionInfo({std::move(view)})});
  return it->second.get();
}

const BufferRegionFootprint *BufferRegionAnalysis::translateToCallsite(
    const BufferRegionFootprint *footprint, CallOpInterface call,
    FunctionOpInterface callee) {
  if (!footprint)
    return nullptr;
  uint32_t calleeFrame = getOperationId(callee);
  if (llvm::none_of(footprint->regionInfo.views, [&](const auto &view) {
        return view.allocationFrame == calleeFrame;
      }))
    return footprint;
  auto [it, inserted] =
      callsiteFootprints.try_emplace({footprint, call.getOperation()});
  if (inserted) {
    uint32_t callerFrame =
        getOperationId(call->getParentOfType<FunctionOpInterface>());
    uint32_t offset = getCallOffset(call);
    RegionInfo info(RegionInfo::ViewList{});
    for (const BufferRegionView &view : footprint->regionInfo.views)
      info.views.insert(view.allocationFrame == calleeFrame
                            ? view.translated(offset, callerFrame)
                            : view);
    it->second = std::make_unique<BufferRegionFootprint>(
        BufferRegionFootprint{footprint->memorySpace, std::move(info)});
  }
  return it->second.get();
}

Allocation *BufferRegionAnalysis::getAllocation(Operation *op) const {
  if (!moduleAllocation)
    return nullptr;
  auto *allocation =
      moduleAllocation->getFuncData(op->getParentOfType<FunctionOpInterface>());
  assert(allocation && "function allocation must be available");
  return allocation;
}

uint32_t BufferRegionAnalysis::getCallOffset(CallOpInterface call) const {
  if (auto *allocation = getAllocation(call)) {
    auto id = allocation->getBufferId(call);
    return id == Allocation::InvalidBufferId ? 0 : allocation->getOffset(id);
  }
  auto offset = call->getAttrOfType<IntegerAttr>("allocation.offset");
  return offset ? offset.getInt() : 0;
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
    // Descriptor views preserve memory space, so shared origins cannot
    // contribute to a tensor-memory footprint.
    if (mode == Mode::TensorMemoryOnly)
      return propagateRegions(RegionInfo::getPessimisticValueState());
    FailureOr<SmallVector<uint32_t, 2>> offsets =
        getAllocationOffsets(localAllocOp, getAllocation(op));
    if (failed(offsets))
      return failure();
    ArrayRef<uint32_t> partitionBases = offsets->size() > 1
                                            ? ArrayRef<uint32_t>(*offsets)
                                            : ArrayRef<uint32_t>();
    regionInfo.views.insert(getAllocView(localAllocOp.getResult(),
                                         offsets->front(), partitionBases));
    return propagateRegions(regionInfo);
  }
  if (auto tmemAllocOp = dyn_cast<ttng::TMEMAllocOp>(op)) {
    regionInfo.views.insert(getAllocView(tmemAllocOp.getResult(),
                                         getAllocationOffset(tmemAllocOp)));
    return propagateRegions(regionInfo);
  }
  if (auto memdescIndexOp = dyn_cast<ttg::MemDescIndexOp>(op)) {
    const RegionInfo &in = operands[0]->getValue();
    if (in.isUnknown())
      return propagateRegions(in);
    int numSubBuffers =
        cast<ttg::MemDescType>(memdescIndexOp.getSrc().getType()).getShape()[0];
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
        regionInfo.views.insert(
            getSubView(memdescIndexOp.getType(), view, stageOffset));
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
      regionInfo.views.insert(
          getSubView(memdescSubsliceOp.getType(), view,
                     relativeOffset.storageOffset, relativeOffset.byteOffset,
                     relativeOffset.partitionOffset, relativeOffset.ctaOffset));
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
      regionInfo.views.insert(getSubView(tmemSubsliceOp.getType(), view,
                                         /*storageOffset=*/0, relativeOffset));
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
      regionInfo.views.insert(getSubView(reinterpretOp.getType(), view));
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
      const RegionInfo &regionInfo =
          getLatticeElement(access.value)->getValue();
      auto addRegions = [&](RegionType regionType) {
        if (regionInfo.isUnknown())
          usedUnknownBufferRegions[regionType] = true;
        for (const BufferRegionView &view : regionInfo.views)
          usedBufferRegions[regionType].insert(view.region);
      };

      bool isTensorMemory = isa<ttng::TensorMemorySpaceAttr>(
          cast<ttg::MemDescType>(access.value.getType()).getMemorySpace());
      addRegions(isTensorMemory ? TENSOR_MEMORY : SHARED_MEMORY);
      if (auto barrierOp = dyn_cast<ttg::MBarrierOpInterface>(op))
        if (llvm::is_contained(barrierOp.getBarriers(), access.value))
          addRegions(BARRIER);
    }
  });
}

SmallVector<MemoryAccess> getMemoryAccesses(Operation *op,
                                            std::optional<ttg::SharedKind> kind,
                                            std::optional<RW> rw) {
  SmallVector<MemoryAccess> accesses;
  auto memoryEffects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memoryEffects)
    return accesses;

  SmallVector<MemoryEffects::EffectInstance> effects;
  memoryEffects.getEffects(effects);
  for (const MemoryEffects::EffectInstance &effect : effects) {
    bool isWrite = isa<MemoryEffects::Write>(effect.getEffect());
    bool isRead = isa<MemoryEffects::Read>(effect.getEffect());
    if (!isWrite && !isRead)
      continue;
    if (rw && (*rw == RW::Read ? !isRead : !isWrite))
      continue;
    Value value = effect.getValue();
    if (!value || !isa<ttg::MemDescType>(value.getType()))
      continue;

    std::optional<ttg::SharedKind> sharedKind;
    if (auto shared = dyn_cast<ttg::SharedMemoryEffect>(&effect))
      sharedKind = shared.getKind();
    else if (!isa<ttng::TensorMemory>(effect.getResource()))
      continue;
    if (kind && sharedKind != kind)
      continue;

    auto existing = llvm::find_if(accesses, [&](const MemoryAccess &access) {
      return access.value == value && access.sharedKind == sharedKind;
    });
    if (existing == accesses.end())
      accesses.push_back({value, isWrite, isRead, sharedKind});
    else {
      existing->isWrite |= isWrite;
      existing->isRead |= isRead;
    }
  }
  return accesses;
}

bool hasSharedAccess(Operation *op, std::optional<ttg::SharedKind> kind,
                     std::optional<RW> rw) {
  return llvm::any_of(
      getMemoryAccesses(op, kind, rw),
      [](const MemoryAccess &access) { return access.isShared(); });
}

} // namespace mlir::triton
