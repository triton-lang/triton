#include "triton/Analysis/BufferRegion.h"
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

} // namespace

namespace mlir::triton {

LogicalResult BufferRegionAnalysis::visitOperation(
    Operation *op,
    llvm::ArrayRef<const dataflow::Lattice<RegionInfo> *> operands,
    llvm::ArrayRef<dataflow::Lattice<RegionInfo> *> results) {
  auto propagateToWarpSpecializePartitions = [&](Value capture,
                                                 const RegionInfo &info) {
    for (Operation *user : capture.getUsers()) {
      auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user);
      if (!wsOp)
        continue;
      auto captures = wsOp.getExplicitCaptures();
      auto it = llvm::find(captures, capture);
      if (it == captures.end())
        continue;
      size_t idx = std::distance(captures.begin(), it);
      for (Region *region : wsOp.getPartitionRegions()) {
        if (region->empty())
          continue;
        auto blockArgs = region->front().getArguments();
        if (idx >= blockArgs.size())
          continue;
        auto *argLat = getLatticeElement(blockArgs[idx]);
        propagateIfChanged(argLat, argLat->join(info));
      }
    }
  };
  Value result = nullptr;
  RegionInfo regionInfo;
  if (auto localAllocOp = dyn_cast<ttg::LocalAllocOp>(op)) {
    uint32_t offset = getAllocationOffset(localAllocOp);
    uint32_t size = getAllocSize(localAllocOp);
    regionInfo.regions.insert({offset, size});

    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
    result = localAllocOp.getResult();
  }
  if (auto tmemAllocOp = dyn_cast<ttng::TMEMAllocOp>(op)) {
    uint32_t offset = getAllocationOffset(tmemAllocOp);
    uint32_t size = getAllocSize(tmemAllocOp);
    regionInfo.regions.insert({offset, size});

    for (auto *r : results) {
      propagateIfChanged(r, r->join(regionInfo));
    }
    result = tmemAllocOp.getResult();
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
    result = memdescIndexOp.getResult();
  }
  if (result) {
    propagateToWarpSpecializePartitions(result, regionInfo);
  }
  return success();
}

void BufferRegionAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    llvm::ArrayRef<dataflow::Lattice<RegionInfo> *> argLattices,
    unsigned firstIndex) {
  setAllToEntryStates(argLattices.take_front(firstIndex));
  setAllToEntryStates(argLattices.drop_front(
      firstIndex + successor.getSuccessorInputs().size()));
}

void BufferRegionAnalysis::calculateUsedBufferRegions(Operation *op) {
  op->walk([&](Operation *op) {
    if (usesMemory(op)) {
      for (auto operand : op->getOperands()) {
        RegionInfo regionInfo = getLatticeElement(operand)->getValue();
        std::optional<RegionType> regionType = getRegionType(operand);
        if (!regionType) {
          continue;
        }
        // Note the buffer regions that may be accessed by the operation.
        for (auto &region : regionInfo.regions) {
          insertRegion(*regionType, {region.baseOffset, region.length});
        }
      }
    }
  });
}

bool BufferRegionAnalysis::usesMemory(Operation *op) const {
  return llvm::any_of(op->getOperands(), [](Value v) {
    return isa<ttg::MemDescType>(v.getType());
  });
}

std::optional<BufferRegionAnalysis::RegionType>
BufferRegionAnalysis::getRegionType(Value v) {
  if (isUsedAsBarrier(v)) {
    return RegionType::BARRIER;
  }
  if (isUsedAsSharedMemory(v)) {
    return RegionType::SHARED_MEMORY;
  }
  if (isUsedAsTensorMemory(v)) {
    return RegionType::TENSOR_MEMORY;
  }
  return std::nullopt;
}

} // namespace mlir::triton
