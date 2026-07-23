#include "triton/Analysis/BufferRegion.h"

#include <limits>
#include <optional>

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
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
FailureOr<uint32_t> getAllocationOffset(ttg::LocalAllocOp op) {
  auto offsetAttr = op->getAttr("allocation.offset");
  if (!offsetAttr) {
    op.emitError("ConcurrencySanitizer should run after "
                 "AllocateSharedMemory pass");
    return failure();
  }
  auto offset = dyn_cast<IntegerAttr>(offsetAttr);
  if (!offset) {
    op.emitError("exact buffer-region analysis does not support shared-memory "
                 "layouts with multiple physical allocation bases");
    return failure();
  }
  int64_t value = offset.getInt();
  if (value < 0 ||
      static_cast<uint64_t>(value) > std::numeric_limits<uint32_t>::max()) {
    op.emitError("shared-memory allocation offset exceeds 32-bit range");
    return failure();
  }
  return static_cast<uint32_t>(value);
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
  if (isa<ttng::TensorMemorySpaceAttr>(ty.getMemorySpace())) {
    return ttng::getTmemAllocSizes(ty).numCols;
  }
  assert(isa<ttg::SharedMemorySpaceAttr>(ty.getMemorySpace()) &&
         "Unsupported memory space");
  unsigned elSize = ty.getElementType().getIntOrFloatBitWidth() / 8;
  return product(ttg::getShapePerCTA(ty)) * elSize;
}

uint32_t applySharedPadding(uint32_t byteOffset, ttg::MemDescType ty) {
  auto padded = ttg::getPaddedEncoding(ty.getEncoding());
  if (!padded)
    return byteOffset;
  uint64_t paddedOffset = byteOffset;
  uint64_t elementSize = ty.getElementTypeBitWidth() / 8;
  for (auto [interval, padding] :
       llvm::zip_equal(padded.getIntervals(), padded.getPaddings())) {
    uint64_t intervalBytes = static_cast<uint64_t>(interval) * elementSize;
    uint64_t paddingBytes = static_cast<uint64_t>(padding) * elementSize;
    paddedOffset += (byteOffset / intervalBytes) * paddingBytes;
  }
  assert(paddedOffset <= std::numeric_limits<uint32_t>::max());
  return static_cast<uint32_t>(paddedOffset);
}

uint32_t getMemDescStorageStride(ttg::MemDescType ty);

FailureOr<triton::AddressSet> getMemDescAddresses(
    uint32_t storageBase, uint32_t affineOffset, ttg::MemDescType ty,
    Operation *op,
    llvm::DenseMap<std::pair<Type, uint32_t>, triton::AddressSet> *cache =
        nullptr) {
  if (cache) {
    auto key = std::make_pair(Type(ty), affineOffset);
    auto found = cache->find(key);
    if (found != cache->end())
      return found->second.translated(storageBase);
    FailureOr<triton::AddressSet> relative =
        getMemDescAddresses(/*storageBase=*/0, affineOffset, ty, op);
    if (failed(relative))
      return failure();
    cache->try_emplace(key, *relative);
    return relative->translated(storageBase);
  }
  bool isTmem = isa<ttng::TensorMemorySpaceAttr>(ty.getMemorySpace());
  auto collectPages = [&]() -> FailureOr<triton::AddressSet> {
    ttg::MemDescType pageTy =
        ty.cloneWith(ty.getShape().drop_front(), ty.getElementType());
    uint32_t pageStride = getMemDescStorageStride(pageTy);
    triton::AddressSet addresses;
    for (int64_t page = 0; page < ty.getDimSize(0); ++page) {
      FailureOr<triton::AddressSet> pageAddresses = getMemDescAddresses(
          storageBase + page * pageStride, affineOffset, pageTy, op);
      if (failed(pageAddresses))
        return failure();
      addresses.insert(*pageAddresses);
    }
    return addresses;
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
  SmallVector<int64_t> shape = ttg::getShapePerCTA(ty);
  uint64_t numPoints = product(shape);
  uint32_t bitWidth = ty.getElementTypeBitWidth();
  if (!isTmem && bitWidth % 8 != 0) {
    op->emitError("sub-byte shared-memory elements are unsupported by exact "
                  "buffer region analysis");
    return failure();
  }

  StringAttr offsetName = StringAttr::get(ctx, "offset");
  StringAttr blockName = StringAttr::get(ctx, "block");
  StringAttr rowName = StringAttr::get(ctx, "row");
  StringAttr colName = StringAttr::get(ctx, "col");
  triton::AddressSet addresses;
  auto addPhysicalAddress = [&](uint32_t offset, uint32_t row, uint32_t col) {
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
      uint32_t relative = offset * (bitWidth / 8);
      uint32_t combined = affineOffset ^ relative;
      uint64_t begin =
          static_cast<uint64_t>(storageBase) + applySharedPadding(combined, ty);
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
  };
  SmallVector<PhysicalBasis> bases;
  for (auto [dim, dimSize] : llvm::zip_equal(dims, shape)) {
    unsigned numBits = llvm::Log2_64(dimSize);
    if (numBits > static_cast<unsigned>(inverse.getInDimSizeLog2(dim))) {
      op->emitError("buffer footprint exceeds its linear-layout domain");
      return failure();
    }
    for (unsigned bit = 0; bit < numBits; ++bit) {
      if (inverse.hasOutDim(blockName) &&
          inverse.getBasis(dim, bit, blockName) != 0) {
        op->emitError("buffer footprint differs across CTA-local layout "
                      "instances");
        return failure();
      }
      bases.push_back(
          {inverse.hasOutDim(offsetName)
               ? static_cast<uint32_t>(inverse.getBasis(dim, bit, offsetName))
               : 0,
           inverse.hasOutDim(rowName)
               ? static_cast<uint32_t>(inverse.getBasis(dim, bit, rowName))
               : 0,
           inverse.hasOutDim(colName)
               ? static_cast<uint32_t>(inverse.getBasis(dim, bit, colName))
               : 0});
    }
  }

  PhysicalBasis physical;
  for (uint64_t index = 0; index < numPoints; ++index) {
    if (index != 0) {
      const PhysicalBasis &basis = bases[llvm::countr_zero(index)];
      physical.offset ^= basis.offset;
      physical.row ^= basis.row;
      physical.col ^= basis.col;
    }
    addPhysicalAddress(physical.offset, physical.row, physical.col);
  }
  return addresses;
}

FailureOr<triton::BufferRegion> getMemDescRegion(
    uint32_t storageBase, uint32_t affineOffset, ttg::MemDescType ty,
    Operation *op,
    llvm::DenseMap<std::pair<Type, uint32_t>, triton::AddressSet> *cache) {
  FailureOr<triton::AddressSet> addresses =
      getMemDescAddresses(storageBase, affineOffset, ty, op, cache);
  if (failed(addresses))
    return failure();
  uint32_t baseOffset = storageBase + affineOffset;
  return triton::BufferRegion(baseOffset, getMemDescSize(ty),
                              std::move(*addresses), storageBase, affineOffset);
}

uint32_t getMemDescStorageStride(ttg::MemDescType ty) {
  if (isa<ttng::TensorMemorySpaceAttr>(ty.getMemorySpace()))
    return ttng::getTmemAllocSizes(ty).numCols;
  uint32_t elementBytes = ty.getElementTypeBitWidth() / 8;
  uint32_t unpadded = product(ttg::getAllocationShapePerCTA(ty)) * elementBytes;
  return applySharedPadding(unpadded, ty);
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

FailureOr<uint32_t>
getMemDescSubsliceUnpaddedByteOffset(ttg::MemDescSubsliceOp op) {
  auto srcTy = op.getSrc().getType();
  auto offsets = op.getOffsets();
  if (offsets.empty())
    return 0;

  Attribute encoding = srcTy.getEncoding();
  auto layoutOffsets = ttg::dropPipeliningDim(offsets, encoding);
  auto layoutRank = layoutOffsets.size();
  mlir::triton::LinearLayout layout;
  if (auto padded = dyn_cast<ttg::PaddedSharedEncodingAttr>(encoding)) {
    layout = padded.getLinearComponent();
  } else {
    layout = ttg::toLinearLayout(srcTy);
  }

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
  mlir::triton::LinearLayout inverse = layout.pseudoinvert();
  auto mapped = inverse.apply(logicalOffsets);
  if (mapped.size() != 2 || mapped[0].first != offsetDim ||
      mapped[1].first != blockDim) {
    op.emitError("unsupported memdesc subslice layout in buffer region "
                 "analysis");
    return failure();
  }
  if (mapped[1].second != 0) {
    op.emitError("memdesc subslices with cross-CTA affine offsets are "
                 "unsupported by buffer region analysis");
    return failure();
  }
  uint64_t elementOffset = static_cast<uint32_t>(mapped[0].second);
  if (offsets.size() != layoutRank) {
    uint64_t stride = product(ttg::getAllocationShapePerCTA(
        encoding, ttg::dropPipeliningDim(srcTy.getAllocShape(), encoding)));
    elementOffset += static_cast<uint64_t>(offsets.front()) * stride;
  }

  uint64_t elementSizeBytes =
      srcTy.getElementType().getIntOrFloatBitWidth() / 8;
  assert(elementSizeBytes > 0 && "element size must be non-zero");
  uint64_t byteOffset = elementOffset * elementSizeBytes;

  assert(byteOffset <= std::numeric_limits<uint32_t>::max() &&
         "memdesc_subslice offset exceeds 32-bit range");
  return static_cast<uint32_t>(byteOffset);
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

BufferStatePlan createBufferStatePlan(ArrayRef<BufferRegion> regions) {
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
        if (!regions[current].intersects(regions[candidate]))
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
  SmallVector<ComponentPlan> componentPlans;
  for (const SmallVector<unsigned> &component : components) {
    AddressSet componentAddresses;
    for (unsigned regionId : component)
      componentAddresses.insert(regions[regionId].addresses);

    SmallVector<llvm::SmallBitVector> atomMemberships;
    for (uint32_t address : componentAddresses) {
      llvm::SmallBitVector membership(component.size());
      for (auto [localId, regionId] : llvm::enumerate(component))
        if (regions[regionId].addresses.contains(address))
          membership.set(localId);
      if (llvm::is_contained(atomMemberships, membership))
        continue;
      atomMemberships.push_back(std::move(membership));
    }

    plan.numLanes += atomMemberships.size();
    componentPlans.push_back({component, std::move(atomMemberships)});
  }

  for (llvm::SmallBitVector &mask : plan.regionMasks)
    mask.resize(plan.numLanes);

  unsigned laneBegin = 0;
  for (const ComponentPlan &componentPlan : componentPlans) {
    BufferStateComponent component;
    component.regionIds = componentPlan.regionIds;
    component.laneBegin = laneBegin;
    component.laneCount = componentPlan.atomMemberships.size();

    for (auto [atomId, membership] :
         llvm::enumerate(componentPlan.atomMemberships)) {
      unsigned lane = laneBegin + atomId;
      for (auto [localId, regionId] : llvm::enumerate(component.regionIds)) {
        if (!membership.test(localId))
          continue;
        plan.regionMasks[regionId].set(lane);
      }
    }

    plan.components.push_back(std::move(component));
    laneBegin += plan.components.back().laneCount;
  }
  assert(laneBegin == plan.numLanes);
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
  RegionInfo regionInfo;
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
    FailureOr<uint32_t> offset = getAllocationOffset(localAllocOp);
    if (failed(offset))
      return failure();
    FailureOr<BufferRegion> region =
        getMemDescRegion(*offset, /*affineOffset=*/0, localAllocOp.getType(),
                         op, &footprintCache);
    if (failed(region))
      return failure();
    regionInfo.regions.insert(std::move(*region));

    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
    return success();
  }
  if (auto tmemAllocOp = dyn_cast<ttng::TMEMAllocOp>(op)) {
    uint32_t offset = getAllocationOffset(tmemAllocOp);
    FailureOr<BufferRegion> region = getMemDescRegion(
        offset, /*affineOffset=*/0, tmemAllocOp.getType(), op, &footprintCache);
    if (failed(region))
      return failure();
    regionInfo.regions.insert(std::move(*region));

    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
    return success();
  }
  if (auto memdescIndexOp = dyn_cast<ttg::MemDescIndexOp>(op)) {
    RegionInfo in = operands[0]->getValue();
    int numSubBuffers = getNumBuffers(memdescIndexOp);
    int firstSubBuffer = 0;
    int endSubBuffer = numSubBuffers;
    APInt constantIndex;
    if (matchPattern(memdescIndexOp.getIndex(),
                     m_ConstantInt(&constantIndex))) {
      int64_t index = constantIndex.getSExtValue();
      if (index < 0 || index >= numSubBuffers) {
        op->emitError("constant memdesc index is out of bounds");
        return failure();
      }
      firstSubBuffer = index;
      endSubBuffer = index + 1;
    }
    for (auto &region : in.regions) {
      for (int i = firstSubBuffer; i < endSubBuffer; ++i) {
        uint32_t storageBase =
            region.storageBase +
            i * getMemDescStorageStride(memdescIndexOp.getType());
        FailureOr<BufferRegion> subBuffer =
            getMemDescRegion(storageBase, region.affineOffset,
                             memdescIndexOp.getType(), op, &footprintCache);
        if (failed(subBuffer))
          return failure();
        regionInfo.regions.insert(std::move(*subBuffer));
      }
    }

    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
    return success();
  }
  if (auto memdescSubsliceOp = dyn_cast<ttg::MemDescSubsliceOp>(op)) {
    RegionInfo in = operands[0]->getValue();
    FailureOr<uint32_t> relativeOffset =
        getMemDescSubsliceUnpaddedByteOffset(memdescSubsliceOp);
    if (failed(relativeOffset))
      return failure();
    for (auto &region : in.regions) {
      uint32_t affineOffset = region.affineOffset ^ *relativeOffset;
      FailureOr<BufferRegion> subBuffer =
          getMemDescRegion(region.storageBase, affineOffset,
                           memdescSubsliceOp.getType(), op, &footprintCache);
      if (failed(subBuffer))
        return failure();
      regionInfo.regions.insert(std::move(*subBuffer));
    }
    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
    return success();
  }
  if (auto tmemSubsliceOp = dyn_cast<ttng::TMEMSubSliceOp>(op)) {
    RegionInfo in = operands[0]->getValue();
    uint32_t relativeOffset = ttng::getTMemSubSliceOffset(
        tmemSubsliceOp.getSrc().getType(), tmemSubsliceOp.getOffset(),
        tmemSubsliceOp.getDim());
    for (auto &region : in.regions) {
      uint32_t affineOffset = region.affineOffset + relativeOffset;
      FailureOr<BufferRegion> subBuffer =
          getMemDescRegion(region.storageBase, affineOffset,
                           tmemSubsliceOp.getType(), op, &footprintCache);
      if (failed(subBuffer))
        return failure();
      regionInfo.regions.insert(std::move(*subBuffer));
    }
    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
    return success();
  }
  if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
    if (isa<ttg::MemDescType>(selectOp.getType())) {
      regionInfo =
          RegionInfo::join(operands[1]->getValue(), operands[2]->getValue());
      for (auto *r : results) {
        propagateIfChanged(r, r->join(regionInfo));
      }
      return success();
    }
  }
  if (auto reinterpretOp = dyn_cast<ttg::MemDescReinterpretOp>(op)) {
    RegionInfo in = operands[0]->getValue();
    for (auto &region : in.regions) {
      FailureOr<BufferRegion> reinterpreted =
          getMemDescRegion(region.storageBase, region.affineOffset,
                           reinterpretOp.getType(), op, &footprintCache);
      if (failed(reinterpreted))
        return failure();
      regionInfo.regions.insert(std::move(*reinterpreted));
    }
    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
    return success();
  }
  // "Passthrough" ops that don't modify the buffer regions.
  if (isa<ttg::MemDescTransOp, ttg::MemDescReshapeOp>(op)) {
    // Just propagate the regions from the operand.
    RegionInfo in = operands[0]->getValue();
    for (auto &region : in.regions) {
      regionInfo.regions.insert(region);
    }
    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
    return success();
  }
  verifyOpIsSupported(op);
  return success();
}

void BufferRegionAnalysis::calculateUsedBufferRegions(Operation *op) {
  op->walk([&](Operation *op) {
    auto insertRegionForValue = [&](Value v) {
      RegionInfo regionInfo = getLatticeElement(v)->getValue();
      std::optional<RegionType> regionType = getRegionType(v);
      if (!regionType) {
        return;
      }
      for (auto &region : regionInfo.regions) {
        usedBufferRegions[*regionType].insert(region);
      }
    };
    if (BufferRegionAnalysis::isMemoryAccessOperation(op)) {
      // Allocas define their buffers with return value.
      if (isa<ttg::LocalAllocOp, ttng::TMEMAllocOp>(op)) {
        insertRegionForValue(op->getResult(0));
      }
      // All other operations access their operands.
      for (auto operand : op->getOperands()) {
        insertRegionForValue(operand);
      }
    }
  });
}

bool BufferRegionAnalysis::isMemoryAccessOperation(Operation *op) {
  if (isa<ttg::LocalLoadOp, ttg::LocalStoreOp, ttg::LocalGatherOp,
          ttg::LocalScatterOp, ttg::LocalAtomicScatterRMWOp, ttng::TMEMLoadOp,
          ttng::TMEMStoreOp, ttng::TMEMCopyOp, ttg::AsyncCopyGlobalToLocalOp,
          ttng::TMAOpInterface, ttng::CLCLoadResultOp>(op)) {
    return true;
  }
  if (isa<ttg::MBarrierOpInterface>(op)) {
    return true;
  }
  // Allocations with operands write to the memory.
  if (isa<ttg::LocalAllocOp, ttng::TMEMAllocOp>(op) &&
      op->getNumOperands() > 0) {
    return true;
  }
  if (isa<DotOpInterface>(op)) {
    return true;
  }
  return false;
}

void BufferRegionAnalysis::verifyOpIsSupported(Operation *op) {
  bool hasMemoryOperands = llvm::any_of(op->getOperands(), [](Value v) {
    return isUsedAsSharedMemory(v) || isUsedAsTensorMemory(v);
  });
  if (!hasMemoryOperands) {
    return;
  }
  if (isMemoryAccessOperation(op)) {
    return;
  }
  op->emitError(
      "Operation accessing memory unaccounted for in buffer region analysis");
  llvm::report_fatal_error(
      "Operation accessing memory unaccounted for in buffer region analysis");
}

} // namespace mlir::triton
