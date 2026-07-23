#include "triton/Analysis/BufferRegion.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

using namespace mlir;

namespace {
// TODO: move to Utility.cpp/unify with TritonInstrument/Utility.cpp
uint64_t getAllocationOffset(ttg::LocalAllocOp op) {
  auto offsetAttr = op->getAttr("allocation.offset");
  if (!offsetAttr) {
    llvm::report_fatal_error(
        "ConcurrencySanitizer should run after AllocateSharedMemory pass.");
  }
  return cast<IntegerAttr>(offsetAttr).getInt();
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

unsigned getAllocSize(ttg::LocalAllocOp op) {
  auto type = op.getType();
  int64_t numElems =
      ttg::getAllocationElems(type.getEncoding(), type.getAllocShape());
  if (auto padded = ttg::getPaddedEncoding(type.getEncoding()))
    numElems = padded.getPaddedSize({numElems});
  return numElems * type.getElementType().getIntOrFloatBitWidth() / 8;
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

FailureOr<uint32_t> getMemDescSubsliceByteOffset(ttg::MemDescSubsliceOp op) {
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
    uint64_t stride = ttg::getAllocationElems(
        encoding, ttg::dropPipeliningDim(srcTy.getAllocShape(), encoding));
    elementOffset += static_cast<uint64_t>(offsets.front()) * stride;
  }

  if (auto padded = ttg::getPaddedEncoding(encoding))
    elementOffset = padded.getPaddedSize({int64_t(elementOffset + 1)}) - 1;

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
    uint32_t offset = getAllocationOffset(localAllocOp);
    uint32_t size = getAllocSize(localAllocOp);
    regionInfo.regions.insert({offset, size});

    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
    return success();
  }
  if (auto tmemAllocOp = dyn_cast<ttng::TMEMAllocOp>(op)) {
    uint32_t offset = getAllocationOffset(tmemAllocOp);
    uint32_t size = ttng::getTmemAllocSizes(tmemAllocOp.getType()).numCols;
    regionInfo.regions.insert({offset, size});

    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
    return success();
  }
  if (auto memdescIndexOp = dyn_cast<ttg::MemDescIndexOp>(op)) {
    RegionInfo in = operands[0]->getValue();
    int numSubBuffers = getNumBuffers(memdescIndexOp);
    auto type = memdescIndexOp.getType();
    bool isTmem = isa<ttng::TensorMemorySpaceAttr>(type.getMemorySpace());
    uint32_t elementSize =
        isTmem ? 1 : type.getElementType().getIntOrFloatBitWidth() / 8;
    uint64_t stageElems =
        isTmem ? ttng::getTmemAllocSizes(type).numCols
               : ttg::getAllocationElems(type.getEncoding(), type.getShape(),
                                         type.getAllocShape());
    auto padded = ttg::getPaddedEncoding(type.getEncoding());
    uint64_t stageSize =
        padded ? padded.getPaddedSize({int64_t(stageElems)}) : stageElems;
    for (auto &region : in.regions) {
      for (int i = 0; i < numSubBuffers; i++) {
        uint64_t offset = i * stageElems;
        if (padded)
          offset = padded.getPaddedSize({int64_t(offset + 1)}) - 1;
        regionInfo.regions.insert(
            {region.baseOffset + static_cast<uint32_t>(offset * elementSize),
             static_cast<uint32_t>(stageSize * elementSize)});
      }
    }

    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
    return success();
  }
  if (auto memdescSubsliceOp = dyn_cast<ttg::MemDescSubsliceOp>(op)) {
    RegionInfo in = operands[0]->getValue();
    auto type = memdescSubsliceOp.getType();
    uint64_t numElems = ttg::getAllocationElems(
        type.getEncoding(), type.getShape(), type.getAllocShape());
    if (auto padded = ttg::getPaddedEncoding(type.getEncoding()))
      numElems = padded.getPaddedSize({int64_t(numElems)});
    uint32_t subBufferSize =
        numElems * type.getElementType().getIntOrFloatBitWidth() / 8;
    FailureOr<uint32_t> relativeOffset =
        getMemDescSubsliceByteOffset(memdescSubsliceOp);
    if (failed(relativeOffset))
      return failure();
    for (auto &region : in.regions) {
      regionInfo.regions.insert(
          {region.baseOffset + *relativeOffset, subBufferSize});
    }
    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
    return success();
  }
  if (auto tmemSubsliceOp = dyn_cast<ttng::TMEMSubSliceOp>(op)) {
    RegionInfo in = operands[0]->getValue();
    auto type = tmemSubsliceOp.getType();
    uint64_t stages = product(type.getShape().drop_back(
        ttg::dropPipeliningDim(type.getShape(), type.getEncoding()).size()));
    uint32_t subBufferSize = ttng::getTmemAllocSizes(type).numCols;
    if (stages > 1) {
      uint32_t stageCols =
          ttng::getTMemSubSliceOffset(type, /*offset=*/1, /*dim=*/0);
      subBufferSize = (stages - 1) * stageCols + subBufferSize / stages;
    }
    uint32_t relativeOffset = ttng::getTMemSubSliceOffset(
        tmemSubsliceOp.getSrc().getType(), tmemSubsliceOp.getOffset(),
        tmemSubsliceOp.getDim());
    for (auto &region : in.regions) {
      regionInfo.regions.insert(
          {region.baseOffset + relativeOffset, subBufferSize});
    }
    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
    return success();
  }
  if (auto reinterpretOp = dyn_cast<ttg::MemDescReinterpretOp>(op)) {
    auto type = reinterpretOp.getType();
    uint32_t size;
    if (isa<ttng::TensorMemorySpaceAttr>(type.getMemorySpace())) {
      size = ttng::getTmemAllocSizes(type).numCols;
    } else {
      uint64_t numElems = ttg::getAllocationElems(
          type.getEncoding(), type.getShape(), type.getAllocShape());
      if (auto padded = ttg::getPaddedEncoding(type.getEncoding()))
        numElems = padded.getPaddedSize({int64_t(numElems)});
      size = numElems * type.getElementType().getIntOrFloatBitWidth() / 8;
    }
    for (auto &region : operands[0]->getValue().regions)
      regionInfo.regions.insert({region.baseOffset, size});
    for (auto *result : results)
      propagateIfChanged(result, result->join(regionInfo));
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
