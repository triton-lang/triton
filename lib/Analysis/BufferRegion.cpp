#include "triton/Analysis/BufferRegion.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
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

unsigned getMemDescSize(ttg::MemDescType ty) {
  if (isa<ttng::TensorMemorySpaceAttr>(ty.getMemorySpace())) {
    return ttng::getTmemAllocSizes(ty).numCols;
  }
  assert(isa<ttg::SharedMemorySpaceAttr>(ty.getMemorySpace()) &&
         "Unsupported memory space");
  unsigned elSize = ty.getElementType().getIntOrFloatBitWidth() / 8;
  return product(ty.getShape()) * elSize;
}

unsigned getAllocSize(ttg::LocalAllocOp op) {
  return getMemDescSize(op.getType());
}

unsigned getAllocSize(ttng::TMEMAllocOp op) {
  return getMemDescSize(op.getType());
}

unsigned getNumBuffers(ttg::MemDescIndexOp memdescIndexOp) {
  ttg::MemDescType ty =
      cast<ttg::MemDescType>(memdescIndexOp.getSrc().getType());
  return ty.getShape()[0];
}

llvm::DenseSet<Value> getBarrierOperands(Operation *op) {
  if (auto initBarrierOp = dyn_cast<ttng::InitBarrierOp>(op)) {
    return {initBarrierOp.getOperand()};
  }
  if (auto barrierExpectOp = dyn_cast<ttng::BarrierExpectOp>(op)) {
    return {barrierExpectOp.getAlloc()};
  }
  if (auto invalBarrierOp = dyn_cast<ttng::InvalBarrierOp>(op)) {
    return {invalBarrierOp.getAlloc()};
  }
  if (auto asyncOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
    return {asyncOp.getBarrier()};
  }
  if (auto gatherOp = dyn_cast<ttng::AsyncTMAGatherOp>(op)) {
    return {gatherOp.getBarrier()};
  }
  if (auto mmaV5Op = dyn_cast<ttng::MMAv5OpInterface>(op)) {
    return llvm::DenseSet<Value>(mmaV5Op.getCompletionBarriers().begin(),
                                 mmaV5Op.getCompletionBarriers().end());
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

uint32_t getMemDescSubsliceByteOffset(ttg::MemDescSubsliceOp op) {
  auto srcTy = op.getSrc().getType();
  auto offsets = op.getOffsets();
  if (offsets.empty())
    return 0;

  Attribute encoding = srcTy.getEncoding();
  mlir::triton::LinearLayout layout;
  if (auto padded = dyn_cast<ttg::PaddedSharedEncodingAttr>(encoding)) {
    layout = padded.getLinearComponent();
  } else {
    layout = ttg::toLinearLayout(srcTy);
  }

  MLIRContext *ctx = op->getContext();
  SmallVector<StringAttr> dimNames =
      mlir::triton::standardOutDimNames(ctx, srcTy.getRank());
  SmallVector<std::pair<StringAttr, int32_t>> logicalOffsets;
  logicalOffsets.reserve(offsets.size());
  for (auto &&[dimName, offset] : llvm::zip_equal(dimNames, offsets)) {
    logicalOffsets.push_back({dimName, static_cast<int32_t>(offset)});
  }

  StringAttr offsetDim = StringAttr::get(ctx, "offset");
  layout = layout.sublayout({offsetDim}, dimNames);
  mlir::triton::LinearLayout inverse = layout.invert();
  auto mapped = inverse.apply(logicalOffsets);
  assert(mapped.size() == 1 && mapped[0].first == offsetDim &&
         "expected single offset dimension after inversion");
  uint64_t elementOffset = static_cast<uint32_t>(mapped[0].second);

  uint64_t elementSizeBytes =
      srcTy.getElementType().getIntOrFloatBitWidth() / 8;
  assert(elementSizeBytes > 0 && "element size must be non-zero");
  uint64_t byteOffset = elementOffset * elementSizeBytes;

  if (auto padded = dyn_cast<ttg::PaddedSharedEncodingAttr>(encoding)) {
    uint64_t padBytes = 0;
    for (auto &&[interval, padding] :
         llvm::zip_equal(padded.getIntervals(), padded.getPaddings())) {
      if (interval == 0 || padding == 0)
        continue;
      uint64_t intervalScaled =
          static_cast<uint64_t>(interval) * elementSizeBytes;
      uint64_t paddingScaled =
          static_cast<uint64_t>(padding) * elementSizeBytes;
      assert(llvm::isPowerOf2_64(intervalScaled) &&
             llvm::isPowerOf2_64(paddingScaled) &&
             "interval and padding must be powers of two in bytes");
      unsigned intervalLog2 = llvm::Log2_64(intervalScaled);
      unsigned paddingLog2 = llvm::Log2_64(paddingScaled);
      padBytes += (byteOffset >> intervalLog2) << paddingLog2;
    }
    byteOffset += padBytes;
  }

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
    uint32_t size = getAllocSize(tmemAllocOp);
    regionInfo.regions.insert({offset, size});

    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
    return success();
  }
  if (auto memdescIndexOp = dyn_cast<ttg::MemDescIndexOp>(op)) {
    RegionInfo in = operands[0]->getValue();
    int numSubBuffers = getNumBuffers(memdescIndexOp);
    for (auto &region : in.regions) {
      for (int i = 0; i < numSubBuffers; i++) {
        uint32_t subBufferSize = getMemDescSize(memdescIndexOp.getType());
        regionInfo.regions.insert(
            {region.baseOffset + i * subBufferSize, subBufferSize});
      }
    }

    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
    return success();
  }
  if (auto memdescSubsliceOp = dyn_cast<ttg::MemDescSubsliceOp>(op)) {
    RegionInfo in = operands[0]->getValue();
    uint32_t subBufferSize = getMemDescSize(memdescSubsliceOp.getType());
    uint32_t relativeOffset = getMemDescSubsliceByteOffset(memdescSubsliceOp);
    for (auto &region : in.regions) {
      regionInfo.regions.insert(
          {region.baseOffset + relativeOffset, subBufferSize});
    }
    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
    return success();
  }
  if (auto tmemSubsliceOp = dyn_cast<ttng::TMEMSubSliceOp>(op)) {
    RegionInfo in = operands[0]->getValue();
    uint32_t subBufferSize = getMemDescSize(tmemSubsliceOp.getType());
    uint32_t relativeOffset = tmemSubsliceOp.getN();
    for (auto &region : in.regions) {
      regionInfo.regions.insert(
          {region.baseOffset + relativeOffset, subBufferSize});
    }
    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
    return success();
  }
  // "Passthrough" ops that don't modify the buffer regions.
  if (isa<ttg::MemDescTransOp, ttg::MemDescReshapeOp,
          ttg::MemDescReinterpretOp>(op)) {
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

void BufferRegionAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    llvm::ArrayRef<dataflow::Lattice<RegionInfo> *> argLattices,
    unsigned firstIndex) {
  auto wsOp = dyn_cast<triton::gpu::WarpSpecializePartitionsOp>(op);
  if (!wsOp) {
    setAllToEntryStates(argLattices.take_front(firstIndex));
    setAllToEntryStates(argLattices.drop_front(
        firstIndex + successor.getSuccessorInputs().size()));
    return;
  }

  // Propagate aliases from the parent operation's operands to the block
  // arguments.
  assert(!successor.isParent());
  ProgramPoint *point = getProgramPointAfter(wsOp);

  for (auto [capture, argLattice] :
       llvm::zip(wsOp.getParentOp().getExplicitCaptures(), argLattices)) {
    propagateIfChanged(
        argLattice,
        argLattice->join(getLatticeElementFor(point, capture)->getValue()));
  }
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
  if (isa<ttg::LocalLoadOp, ttg::LocalStoreOp, ttng::TMEMLoadOp,
          ttng::TMEMStoreOp, ttg::AsyncCopyGlobalToLocalOp,
          ttng::AsyncTMACopyGlobalToLocalOp, ttng::AsyncTMACopyLocalToGlobalOp,
          ttng::AsyncTMAGatherOp, ttng::AsyncTMAScatterOp, ttng::InitBarrierOp,
          ttng::BarrierExpectOp, ttng::InvalBarrierOp, ttng::WaitBarrierOp>(
          op)) {
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
