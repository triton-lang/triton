#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseSet.h"

#include <limits>

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUTMEMBARRIERINSERTIONPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;

enum class TMemAccessKind { None, Load, Store, MMA };

// Keep row groups far apart so per-row column intervals do not alias after
// flattening the physical 2D tensor-memory address space into 1D intervals.
static constexpr size_t kFlattenedRowStride = size_t{1} << 32;
static constexpr int kAllocRowGranularity = 64;
static constexpr int kRowOffsetGranularity = 16;

// Fine grain modeling of TMEM ops as pipelining behavior is not fully
// represented in ops attributes.
static bool isWritingAlloc(Operation *op) {
  auto alloc = dyn_cast<TMEMAllocOp>(op);
  return alloc && alloc.getSrc();
}

static bool isMMALikeOp(Operation *op) {
  return isa<TCGen5MMAOp, TCGen5MMAScaledOp, TMEMCopyOp>(op);
}

static TMemAccessKind getTMemAccessKind(Operation *op) {
  if (isa<TMEMLoadOp>(op))
    return TMemAccessKind::Load;
  if (isa<TMEMStoreOp>(op) || isWritingAlloc(op))
    return TMemAccessKind::Store;
  if (isMMALikeOp(op))
    return TMemAccessKind::MMA;
  return TMemAccessKind::None;
}

static bool filterFn(Operation *lhs, Operation *rhs, bool /*lhsIsRead*/,
                     bool /*rhsIsRead*/, Allocation * /*allocation*/) {
  TMemAccessKind lhsKind = getTMemAccessKind(lhs);
  TMemAccessKind rhsKind = getTMemAccessKind(rhs);

  bool war =
      lhsKind == TMemAccessKind::Load && rhsKind == TMemAccessKind::Store;
  bool raw =
      lhsKind == TMemAccessKind::Store && rhsKind == TMemAccessKind::Load;
  bool waw =
      lhsKind == TMemAccessKind::Store && rhsKind == TMemAccessKind::Store;

  // MMAv5 ops and tmem_copy are special cases, we care about load->mma and
  // store->mma dependencies but mma -> load/store doesn't require a barrier
  // since it would need a mbarrier wait that will ensure the op is finished
  // before any thread can reach the load/store.
  bool loadToMma =
      lhsKind == TMemAccessKind::Load && rhsKind == TMemAccessKind::MMA;
  bool storeToMma =
      lhsKind == TMemAccessKind::Store && rhsKind == TMemAccessKind::MMA;

  bool requiresBarrier = war || raw || waw || loadToMma || storeToMma;
  return !requiresBarrier;
}

static bool isTensorMemory(Value value) {
  auto memDescType = dyn_cast<ttg::MemDescType>(value.getType());
  return memDescType &&
         isa<TensorMemorySpaceAttr>(memDescType.getMemorySpace());
}

static void appendRootAllocs(Value value, SmallVectorImpl<TMEMAllocOp> &allocs,
                             bool &unknown) {
  DenseSet<Value> seen;
  SmallVector<Value> worklist{value};

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!seen.insert(current).second)
      continue;

    if (auto arg = dyn_cast<BlockArgument>(current)) {
      Block *block = arg.getOwner();
      Operation *parentOp = block->getParentOp();

      if (!block->isEntryBlock()) {
        for (Block *pred : block->getPredecessors()) {
          auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
          if (!branch) {
            unknown = true;
            continue;
          }
          auto it = llvm::find(branch->getSuccessors(), block);
          unsigned successorIndex =
              std::distance(branch->getSuccessors().begin(), it);
          SuccessorOperands args = branch.getSuccessorOperands(successorIndex);
          worklist.push_back(
              args.getForwardedOperands()[arg.getArgNumber() -
                                          args.getProducedOperandCount()]);
        }
        continue;
      }

      if (auto ws = dyn_cast<ttg::WarpSpecializePartitionsOp>(parentOp)) {
        worklist.push_back(ws.getExplicitCaptures()[arg.getArgNumber()]);
      } else if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        unsigned idx = arg.getArgNumber() - 1;
        worklist.push_back(forOp.getYieldedValues()[idx]);
        worklist.push_back(forOp.getInits()[idx]);
      } else if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
        unsigned idx = arg.getArgNumber();
        if (arg.getParentRegion() == &whileOp.getAfter()) {
          worklist.push_back(whileOp.getConditionOp().getArgs()[idx]);
        } else {
          worklist.push_back(whileOp.getYieldedValues()[idx]);
          worklist.push_back(whileOp.getInits()[idx]);
        }
      } else {
        unknown = true;
      }
      continue;
    }

    Operation *defOp = current.getDefiningOp();
    if (!defOp) {
      unknown = true;
      continue;
    }

    unsigned resultIndex = cast<OpResult>(current).getResultNumber();
    if (auto alloc = dyn_cast<TMEMAllocOp>(defOp)) {
      allocs.push_back(alloc);
    } else if (defOp->hasTrait<OpTrait::MemDescViewTrait>()) {
      worklist.push_back(defOp->getOperand(0));
    } else if (auto slice = dyn_cast<TMEMSubSliceOp>(defOp)) {
      worklist.push_back(slice.getSrc());
    } else if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
      worklist.push_back(selectOp.getTrueValue());
      worklist.push_back(selectOp.getFalseValue());
    } else if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      worklist.push_back(ifOp.thenYield().getOperand(resultIndex));
      worklist.push_back(ifOp.elseYield().getOperand(resultIndex));
    } else if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      worklist.push_back(forOp.getYieldedValues()[resultIndex]);
      worklist.push_back(forOp.getInits()[resultIndex]);
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(defOp)) {
      worklist.push_back(whileOp.getConditionOp().getArgs()[resultIndex]);
    } else {
      unknown = true;
    }
  }
}

static SmallVector<AllocationSlice> getTMemSlices(Value value) {
  SmallVector<TMEMAllocOp> allocs;
  bool unknown = false;
  appendRootAllocs(value, allocs, unknown);

  SmallVector<AllocationSlice> slices;
  if (unknown || allocs.empty()) {
    slices.emplace_back(
        Interval<size_t>(0, std::numeric_limits<size_t>::max()));
    return slices;
  }

  for (TMEMAllocOp alloc : allocs) {
    auto colAttr =
        alloc->getAttrOfType<IntegerAttr>("tensor_memory_col_offset");
    auto rowAttr =
        alloc->getAttrOfType<IntegerAttr>("tensor_memory_row_offset");
    if (!colAttr || !rowAttr) {
      slices.clear();
      slices.emplace_back(
          Interval<size_t>(0, std::numeric_limits<size_t>::max()));
      return slices;
    }

    int64_t colOffset = colAttr.getInt();
    int64_t rowOffset = rowAttr.getInt();
    TMemAllocation allocSize = getTmemAllocSizes(alloc.getType());
    if (rowOffset % kRowOffsetGranularity != 0 ||
        allocSize.numRows % kAllocRowGranularity != 0) {
      slices.clear();
      slices.emplace_back(
          Interval<size_t>(0, std::numeric_limits<size_t>::max()));
      return slices;
    }

    int64_t rowGroup = rowOffset / kRowOffsetGranularity;
    int64_t numRowGroups = allocSize.numRows / kAllocRowGranularity;
    for (int64_t row = 0; row < numRowGroups; ++row) {
      size_t start = static_cast<size_t>(rowGroup + row) * kFlattenedRowStride +
                     static_cast<size_t>(colOffset);
      slices.emplace_back(Interval<size_t>(start, start + allocSize.numCols));
    }
  }
  return slices;
}

static void appendReadSlices(Value value, Operation *op, BlockInfo *blockInfo) {
  if (!isTensorMemory(value))
    return;
  for (AllocationSlice slice : getTMemSlices(value))
    blockInfo->syncReadSlices[slice].insert(op);
}

static void appendWriteSlices(Value value, Operation *op,
                              BlockInfo *blockInfo) {
  if (!isTensorMemory(value))
    return;
  for (AllocationSlice slice : getTMemSlices(value))
    blockInfo->syncWriteSlices[slice].insert(op);
}

class TMemBarrierAnalysis : public MembarOrFenceAnalysis {
public:
  explicit TMemBarrierAnalysis(Allocation *allocation, MembarFilterFn filter)
      : MembarOrFenceAnalysis(allocation, filter) {}

private:
  void update(Operation *operation, BlockInfo *blockInfo,
              FuncBlockInfoMapT *funcBlockInfoMap, OpBuilder *builder) override;

  void insertBarrier(Operation *operation, OpBuilder *builder);
};

void TMemBarrierAnalysis::insertBarrier(Operation *op, OpBuilder *builder) {
  OpBuilder::InsertionGuard g(*builder);
  triton::gpu::BarrierOp::create(*builder, op->getLoc(),
                                 triton::gpu::AddrSpace::Local);
}

void TMemBarrierAnalysis::update(Operation *op, BlockInfo *blockInfo,
                                 FuncBlockInfoMapT *funcBlockInfoMap,
                                 OpBuilder *builder) {
  if (mlir::containsLocalBarrier(op)) {
    blockInfo->sync();
    return;
  }

  BlockInfo curBlockInfo;
  if (isa<triton::CallOp>(op)) {
    auto call = dyn_cast<CallOpInterface>(op);
    if (auto callee = dyn_cast<FunctionOpInterface>(call.resolveCallable()))
      curBlockInfo = funcBlockInfoMap->lookup(callee);
  } else if (auto load = dyn_cast<TMEMLoadOp>(op)) {
    appendReadSlices(load.getSrc(), op, &curBlockInfo);
  } else if (auto store = dyn_cast<TMEMStoreOp>(op)) {
    appendWriteSlices(store.getDst(), op, &curBlockInfo);
  } else if (auto alloc = dyn_cast<TMEMAllocOp>(op)) {
    if (alloc.getSrc())
      appendWriteSlices(alloc.getResult(), op, &curBlockInfo);
  } else if (auto mma = dyn_cast<MMAv5OpInterface>(op)) {
    appendWriteSlices(mma.getAccumulator(), op, &curBlockInfo);
    appendReadSlices(mma.getA(), op, &curBlockInfo);
    if (auto scaledMma = dyn_cast<TCGen5MMAScaledOp>(op)) {
      appendReadSlices(scaledMma.getAScale(), op, &curBlockInfo);
      appendReadSlices(scaledMma.getBScale(), op, &curBlockInfo);
    }
  } else if (auto copy = dyn_cast<TMEMCopyOp>(op)) {
    appendWriteSlices(copy.getDst(), op, &curBlockInfo);
  }

  if (blockInfo->isIntersected(curBlockInfo, filter, allocation)) {
    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    blockInfo->sync();
  }

  blockInfo->join(curBlockInfo);
}

} // namespace

struct TMemBarrierInsertionPass
    : public impl::TritonNvidiaGPUTMemBarrierInsertionPassBase<
          TMemBarrierInsertionPass> {
  using impl::TritonNvidiaGPUTMemBarrierInsertionPassBase<
      TMemBarrierInsertionPass>::TritonNvidiaGPUTMemBarrierInsertionPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ModuleAllocation allocation(mod);
    ModuleMembarOrFenceAnalysis<TMemBarrierAnalysis> analysis(&allocation,
                                                              filterFn);
    analysis.run();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
