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

#include <algorithm>

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

using namespace mlir;

namespace {
// TODO: move to Utility.cpp/unify with TritonInstrument/Utility.cpp
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
  triton::LinearLayout layout = ttg::toLinearLayoutIgnoringPadding(
      ttg::dropPipeliningDim(ty.getAllocShape(), ty.getEncoding()),
      ty.getEncoding());
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
  struct PhysicalBasis {
    uint32_t offset = 0;
    uint32_t row = 0;
    uint32_t col = 0;
    uint32_t partition = 0;
    uint32_t block = 0;
  };
  SmallVector<PhysicalBasis> bases;
  for (auto [dim, dimSize] : llvm::zip_equal(dims, shape)) {
    unsigned numBits = llvm::Log2_64(dimSize);
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
  // Include storage reserved by zero allocation bases, but not bases excluded
  // by the subview. The pseudoinverse above selects only one representative.
  for (StringAttr dim : layout.getInDimNames()) {
    uint32_t zeroBits = (layout.getInDimSize(dim) - 1) &
                        ~triton::getInputBasisMask(layout, dim, dims);
    // A zero row-16 basis leaves half of each warp's rows unallocated.
    if (dim == rowName)
      zeroBits &= ~uint32_t{16};
    for (; zeroBits; zeroBits &= zeroBits - 1) {
      uint32_t bit = uint32_t{1} << llvm::countr_zero(zeroBits);
      bases.push_back({dim == offsetName ? bit : 0, dim == rowName ? bit : 0,
                       dim == colName ? bit : 0, 0,
                       dim == blockName ? bit : 0});
      numPoints *= 2;
    }
  }

  assert(llvm::isPowerOf2_64(numPoints) &&
         bases.size() == llvm::Log2_64(numPoints) &&
         "descriptor domain must be an XOR basis image");
  SmallVector<PhysicalBasis> addressBases, selectorBases;
  for (const PhysicalBasis &basis : bases) {
    if (basis.block || (!isTmem && basis.partition))
      selectorBases.push_back(basis);
    else
      addressBases.push_back(basis);
  }
  assert(selectorBases.size() < 64);
  PhysicalBasis selected;
  for (uint64_t i = 0, count = uint64_t{1} << selectorBases.size(); i < count; ++i) {
    if (i) {
      const auto &basis = selectorBases[llvm::countr_zero(i)];
      selected.offset ^= basis.offset;
      selected.row ^= basis.row;
      selected.col ^= basis.col;
      selected.partition ^= basis.partition;
      selected.block ^= basis.block;
    }
    auto &addresses = getAddressesForCTA(footprint, selected.block ^ affineCTAOffset);
    if (isTmem) {
      triton::AddressSet::TensorLayout description{
          storageBase, affineOffset, bitWidth, selected.row, selected.col, {}};
      for (const auto &basis : addressBases)
        description.bases.emplace_back(basis.row, basis.col);
      addresses.insert(triton::AddressSet::fromTensorLayout(std::move(description)));
    } else {
      uint32_t base = partitionBases.empty()
                          ? storageBase
                          : partitionBases[selected.partition ^ affinePartitionOffset];
      triton::AddressSet::SharedLayout description{
          base, affineOffset, bitWidth / 8, selected.offset, {}, {}};
      for (const auto &basis : addressBases)
        description.bases.push_back(basis.offset);
      if (auto padded = ttg::getPaddedEncoding(ty.getEncoding()))
        for (auto [interval, padding] : llvm::zip_equal(padded.getIntervals(), padded.getPaddings()))
          description.padding.emplace_back(interval, padding);
      addresses.insert(triton::AddressSet::fromSharedLayout(std::move(description)));
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
  mlir::triton::LinearLayout layout = ttg::toLinearLayoutIgnoringPadding(srcTy);

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

namespace {
using AddressIntervals = SmallVector<std::pair<uint64_t, uint64_t>, 4>;
constexpr uint64_t addressSpaceSize = uint64_t{1} << 32;

void appendAddressRange(AddressIntervals &ranges, uint32_t begin, uint64_t length) {
  if (!length)
    return;
  uint64_t end = uint64_t(begin) + length;
  ranges.emplace_back(begin, std::min(end, addressSpaceSize));
  if (end > addressSpaceSize)
    ranges.emplace_back(0, end - addressSpaceSize);
}

void coalesceAddressRanges(AddressIntervals &ranges) {
  llvm::sort(ranges);
  size_t size = 0;
  for (auto range : ranges) {
    if (size && range.first <= ranges[size - 1].second)
      ranges[size - 1].second = std::max(ranges[size - 1].second, range.second);
    else
      ranges[size++] = range;
  }
  ranges.resize(size);
}

AddressIntervals intersectAddressRanges(const AddressIntervals &lhs,
                                         const AddressIntervals &rhs) {
  AddressIntervals result;
  size_t i = 0, j = 0;
  while (i < lhs.size() && j < rhs.size()) {
    uint64_t begin = std::max(lhs[i].first, rhs[j].first);
    uint64_t end = std::min(lhs[i].second, rhs[j].second);
    if (begin < end)
      result.emplace_back(begin, end);
    if (lhs[i].second < rhs[j].second)
      ++i;
    else
      ++j;
  }
  return result;
}

AddressIntervals subtractAddressRanges(const AddressIntervals &lhs,
                                        const AddressIntervals &rhs) {
  AddressIntervals result;
  size_t j = 0;
  for (auto [begin, end] : lhs) {
    while (j < rhs.size() && rhs[j].second <= begin)
      ++j;
    size_t k = j;
    while (k < rhs.size() && rhs[k].first < end) {
      if (rhs[k].first > begin)
        result.emplace_back(begin, rhs[k].first);
      begin = std::max(begin, rhs[k].second);
      if (begin >= end)
        break;
      ++k;
    }
    if (begin < end)
      result.emplace_back(begin, end);
  }
  return result;
}
} // namespace

AddressSet AddressSet::fromDescription(Description description) {
  return AddressSet(std::make_shared<Storage>(std::move(description)));
}

AddressSet AddressSet::fromIntervals(Intervals ranges) {
  if (ranges.empty())
    return {};
  if (ranges.size() == 1) {
    auto [begin, end] = ranges.front();
    if (end - begin < addressSpaceSize)
      return fromRange(uint32_t(begin), uint32_t(end - begin));
  }
  return fromDescription(ExplicitIntervals{std::move(ranges)});
}

AddressSet AddressSet::fromXorLayout(uint32_t storageBase, uint32_t xorOffset,
                                      ArrayRef<uint32_t> bases) {
  uint32_t rows[32] = {};
  for (uint32_t value : bases) {
    while (value) {
      unsigned pivot = llvm::Log2_32(value);
      if (!rows[pivot]) {
        rows[pivot] = value;
        break;
      }
      value ^= rows[pivot];
    }
  }
  for (unsigned pivot = 0; pivot < 32; ++pivot)
    if (rows[pivot])
      for (unsigned above = pivot + 1; above < 32; ++above)
        if (rows[above] & (uint32_t{1} << pivot))
          rows[above] ^= rows[pivot];
  for (int pivot = 31; pivot >= 0; --pivot)
    if (xorOffset & (uint32_t{1} << pivot))
      xorOffset ^= rows[pivot];
  XorLayout layout{storageBase, xorOffset, {}};
  for (uint32_t row : rows)
    if (row)
      layout.bases.push_back(row);
  return fromDescription(std::move(layout));
}

AddressSet AddressSet::fromRange(uint32_t begin, uint32_t length) {
  if (!length)
    return {};
  return fromDescription(Range{begin, length});
}

AddressSet AddressSet::fromSharedLayout(SharedLayout layout) {
  if (!layout.elementBytes)
    return {};
  if (layout.padding.empty() && llvm::isPowerOf2_32(layout.elementBytes) &&
      layout.affineOffset % layout.elementBytes == 0) {
    SmallVector<uint32_t> bases;
    for (uint32_t basis : layout.bases)
      bases.push_back(basis * layout.elementBytes);
    for (unsigned bit = 0; bit < llvm::Log2_32(layout.elementBytes); ++bit)
      bases.push_back(uint32_t{1} << bit);
    return fromXorLayout(layout.storageBase,
                         layout.affineOffset ^ (layout.elementOffset * layout.elementBytes),
                         bases);
  }
  return fromDescription(std::move(layout));
}

AddressSet AddressSet::fromTensorLayout(TensorLayout layout) {
  return fromDescription(std::move(layout));
}

bool AddressSet::sameDescription(const AddressSet &other) const {
  if (storage == other.storage)
    return true;
  if (!storage || !other.storage)
    return false;
  auto same = [&](const auto &value) {
    auto *rhs = std::get_if<std::decay_t<decltype(value)>>(&other.storage->description);
    return rhs && value == *rhs;
  };
  return std::visit(llvm::makeVisitor(
      [&](const Range &v) { return same(v); },
      [&](const XorLayout &v) { return same(v); },
      [&](const SharedLayout &v) { return same(v); },
      [&](const TensorLayout &v) { return same(v); },
      [&](const ExplicitIntervals &v) { return same(v); }), storage->description);
}

const AddressSet::Intervals &AddressSet::intervals() const {
  static const Intervals empty;
  if (!storage)
    return empty;
  // Materialized results already own their intervals; do not cache a copy.
  if (auto *explicitSet = std::get_if<ExplicitIntervals>(&storage->description))
    return explicitSet->ranges;
  if (storage->evaluated)
    return *storage->evaluated;
  storage->evaluated = std::visit(llvm::makeVisitor(
      [](const Range &v) {
        Intervals result;
        appendAddressRange(result, v.begin, v.length);
        coalesceAddressRanges(result);
        return result;
      },
      [](const XorLayout &v) {
        // Canonical low unit bases span whole contiguous intervals.
        unsigned lowBits = 0;
        while (lowBits < v.bases.size() && v.bases[lowBits] == (uint64_t{1} << lowBits))
          ++lowBits;
        Intervals result;
        uint32_t value = v.xorOffset;
        uint64_t count = uint64_t{1} << (v.bases.size() - lowBits);
        for (uint64_t i = 0; i < count; ++i) {
          if (i)
            value ^= v.bases[lowBits + llvm::countr_zero(i)];
          appendAddressRange(result, v.storageBase + value, uint64_t{1} << lowBits);
        }
        coalesceAddressRanges(result);
        return result;
      },
      [](const SharedLayout &v) {
        Intervals result;
        uint32_t offset = v.elementOffset;
        assert(v.bases.size() < 64);
        for (uint64_t i = 0, count = uint64_t{1} << v.bases.size(); i < count; ++i) {
          if (i)
            offset ^= v.bases[llvm::countr_zero(i)];
          uint32_t byte = v.affineOffset ^ (offset * v.elementBytes);
          uint32_t element = byte / v.elementBytes;
          uint32_t padded = element;
          for (auto [interval, padding] : v.padding)
            padded += (element / interval) * padding;
          uint32_t begin = v.storageBase + padded * v.elementBytes + byte % v.elementBytes;
          appendAddressRange(result, begin, v.elementBytes);
        }
        coalesceAddressRanges(result);
        return result;
      },
      [](const TensorLayout &v) {
        Intervals result;
        uint32_t row = v.row, column = v.column;
        assert(v.bases.size() < 64);
        for (uint64_t i = 0, count = uint64_t{1} << v.bases.size(); i < count; ++i) {
          if (i) {
            auto basis = v.bases[llvm::countr_zero(i)];
            row ^= basis.first;
            column ^= basis.second;
          }
          uint32_t bitBegin = column * v.bitWidth;
          uint32_t firstWord = bitBegin / 32;
          uint32_t lastWord = llvm::divideCeil(bitBegin + v.bitWidth, uint32_t{32});
          appendAddressRange(result, v.storageBase + v.affineOffset + ((row << 16) | firstWord),
                              lastWord - firstWord);
        }
        coalesceAddressRanges(result);
        return result;
      },
      [](const ExplicitIntervals &v) {
        return v.ranges;
      }), storage->description);
  return *storage->evaluated;
}

AddressSet::const_iterator &AddressSet::const_iterator::operator++() {
  if (++address == (*ranges)[index].second) {
    ++index;
    address = index < ranges->size() ? (*ranges)[index].first : 0;
  }
  return *this;
}

bool AddressSet::empty() const {
  if (!storage)
    return true;
  if (storage->evaluated)
    return storage->evaluated->empty();
  return std::visit(llvm::makeVisitor(
      [](const Range &v) { return v.length == 0; },
      [](const XorLayout &) { return false; },
      [](const SharedLayout &v) { return v.elementBytes == 0; },
      [](const TensorLayout &v) { return v.bitWidth == 0; },
      [](const ExplicitIntervals &v) { return v.ranges.empty(); }), storage->description);
}

void AddressSet::set(uint32_t address) { insert(fromRange(address, 1)); }

void AddressSet::insert(const AddressSet &other) {
  if (other.empty() || sameDescription(other))
    return;
  if (empty()) {
    *this = other;
    return;
  }
  Intervals result = intervals();
  const auto &rhs = other.intervals();
  result.append(rhs.begin(), rhs.end());
  coalesceAddressRanges(result);
  *this = fromIntervals(std::move(result));
}

AddressSet AddressSet::intersection(const AddressSet &other) const {
  if (sameDescription(other))
    return *this;
  if (empty() || other.empty())
    return {};
  return fromIntervals(intersectAddressRanges(intervals(), other.intervals()));
}

void AddressSet::subtract(const AddressSet &other) {
  if (sameDescription(other)) {
    storage.reset();
    return;
  }
  if (!empty() && !other.empty())
    *this = fromIntervals(subtractAddressRanges(intervals(), other.intervals()));
}

bool AddressSet::intersects(const AddressSet &other) const {
  if (sameDescription(other))
    return !empty();
  const auto &lhs = intervals();
  const auto &rhs = other.intervals();
  size_t i = 0, j = 0;
  while (i < lhs.size() && j < rhs.size()) {
    if (lhs[i].first < rhs[j].second && rhs[j].first < lhs[i].second)
      return true;
    if (lhs[i].second < rhs[j].second)
      ++i;
    else
      ++j;
  }
  return false;
}

bool AddressSet::contains(const AddressSet &other) const {
  if (sameDescription(other))
    return true;
  const auto &lhs = intervals();
  size_t i = 0;
  for (auto [begin, end] : other.intervals()) {
    while (i < lhs.size() && lhs[i].second <= begin)
      ++i;
    if (i == lhs.size() || lhs[i].first > begin || lhs[i].second < end)
      return false;
  }
  return true;
}

bool AddressSet::operator==(const AddressSet &other) const {
  return sameDescription(other) || intervals() == other.intervals();
}

bool AddressSet::operator<(const AddressSet &other) const {
  if (sameDescription(other))
    return false;
  const auto &lhs = intervals();
  const auto &rhs = other.intervals();
  auto [l, r] = std::mismatch(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
  if (l == lhs.end() || r == rhs.end())
    return l == lhs.end() && r != rhs.end();
  if (l->first != r->first)
    return l->first < r->first;
  // Equal starts: the shorter interval either ends a prefix or leaves a gap.
  return l->second < r->second ? std::next(l) == lhs.end()
                               : std::next(r) != rhs.end();
}

AddressSet AddressSet::translated(uint32_t delta) const {
  if (!delta || !storage)
    return *this;
  return std::visit(llvm::makeVisitor(
      [=](Range v) { v.begin += delta; return fromDescription(v); },
      [=](XorLayout v) { v.storageBase += delta; return fromDescription(std::move(v)); },
      [=](SharedLayout v) { v.storageBase += delta; return fromDescription(std::move(v)); },
      [=](TensorLayout v) { v.storageBase += delta; return fromDescription(std::move(v)); },
      [=](const ExplicitIntervals &v) {
        Intervals result;
        for (auto [begin, end] : v.ranges)
          appendAddressRange(result, uint32_t(begin) + delta, end - begin);
        coalesceAddressRanges(result);
        return fromIntervals(std::move(result));
      }),
      storage->description);
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
