#include "triton/Analysis/BufferRegion.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

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
  ttg::MemDescType ty = op.getType();
  unsigned elSize = ty.getElementType().getIntOrFloatBitWidth() / 8;
  return product(ty.getShape()) * elSize;
}

unsigned getAllocSize(ttng::TMEMAllocOp op) {
  return ttng::getTmemAllocSizes(op.getType()).numCols;
}

unsigned getNumBuffers(ttg::MemDescIndexOp memdescIndexOp) {
  ttg::MemDescType ty =
      cast<ttg::MemDescType>(memdescIndexOp.getSrc().getType());
  return ty.getShape()[0];
}

Value getBarrierOperand(Operation *op) {
  if (auto initBarrierOp = dyn_cast<ttng::InitBarrierOp>(op)) {
    return initBarrierOp.getOperand();
  }
  if (auto asyncOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
    return asyncOp.getBarrier();
  }
  if (auto gatherOp = dyn_cast<ttng::AsyncTMAGatherOp>(op)) {
    return gatherOp.getBarrier();
  }
  return nullptr;
}

bool isUsedAsBarrier(Value v) {
  for (auto user : v.getUsers()) {
    if (v == getBarrierOperand(user)) {
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
  }
  if (auto localAllocOp = dyn_cast<ttg::LocalAllocOp>(op)) {
    uint32_t offset = getAllocationOffset(localAllocOp);
    uint32_t size = getAllocSize(localAllocOp);
    regionInfo.regions.insert({offset, size});

    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
  }
  if (auto tmemAllocOp = dyn_cast<ttng::TMEMAllocOp>(op)) {
    uint32_t offset = getAllocationOffset(tmemAllocOp);
    uint32_t size = getAllocSize(tmemAllocOp);
    regionInfo.regions.insert({offset, size});

    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
  }
  if (auto memdescIndexOp = dyn_cast<ttg::MemDescIndexOp>(op)) {
    RegionInfo in = operands[0]->getValue();
    int numSubBuffers = getNumBuffers(memdescIndexOp);
    for (auto &region : in.regions) {
      for (int i = 0; i < numSubBuffers; i++) {
        uint32_t subBufferSize = region.length / numSubBuffers;
        regionInfo.regions.insert(
            {region.baseOffset + i * subBufferSize, subBufferSize});
      }
    }

    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
  }
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
          ttng::AsyncTMAGatherOp, ttng::AsyncTMAScatterOp, ttng::InitBarrierOp>(
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

} // namespace mlir::triton
