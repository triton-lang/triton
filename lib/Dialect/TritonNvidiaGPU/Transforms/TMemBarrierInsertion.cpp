#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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

static SmallVector<AllocationSlice>
getTMemSlices(const MemoryAccess &access, BufferRegionAnalysis &regions) {
  SmallVector<size_t> addresses;
  for (const auto &view : regions.getAccessRegions(access)) {
    if (!view)
      return {AllocationSlice(
          Interval<size_t>(0, std::numeric_limits<size_t>::max()))};
    for (const auto &[cta, words] : view->region.ctaAddresses)
      for (uint32_t word : words) {
        size_t rowGroup = (word >> 16) / kRowOffsetGranularity;
        addresses.push_back(rowGroup * kFlattenedRowStride + (word & 0xffff));
      }
  }
  llvm::sort(addresses);
  addresses.erase(std::unique(addresses.begin(), addresses.end()),
                  addresses.end());
  SmallVector<AllocationSlice> slices;
  for (unsigned i = 0; i < addresses.size();) {
    size_t begin = addresses[i++];
    size_t end = begin + 1;
    while (i < addresses.size() && addresses[i] == end) {
      ++i;
      ++end;
    }
    slices.emplace_back(Interval<size_t>(begin, end));
  }
  return slices;
}

class TMemBarrierAnalysis : public MembarAnalysis {
public:
  TMemBarrierAnalysis(Allocation &allocation, MembarFilterFn filter,
                      BufferRegionAnalysis &regions)
      : MembarAnalysis(allocation, std::move(filter)), regions(regions) {}

private:
  BufferRegionAnalysis &regions;
  void update(Operation *operation, MembarInfo *membarInfo, FuncMapT *funcMap,
              OpBuilder *builder) override;

  void insertBarrier(Operation *operation, OpBuilder *builder);
};

void TMemBarrierAnalysis::insertBarrier(Operation *op, OpBuilder *builder) {
  OpBuilder::InsertionGuard g(*builder);
  triton::gpu::BarrierOp::create(*builder, op->getLoc(),
                                 triton::gpu::AddrSpace::Local);
}

void TMemBarrierAnalysis::update(Operation *op, MembarInfo *membarInfo,
                                 FuncMapT *funcMap, OpBuilder *builder) {
  if (mlir::containsLocalBarrier(op)) {
    membarInfo->sync();
    return;
  }

  BlockInfo curBlockInfo;
  if (isa<triton::CallOp>(op)) {
    auto call = cast<CallOpInterface>(op);
    if (auto callee = dyn_cast<FunctionOpInterface>(call.resolveCallable())) {
      // Tensor-memory allocation attributes are physical addresses assigned
      // across the whole module, so callee slices need no call-frame offset.
      MembarInfo calleeMembarInfo = funcMap->lookup(callee);
      if (membarInfo->pending.isIntersected(calleeMembarInfo.entryBlockInfo,
                                            filter, &allocation)) {
        builder->setInsertionPoint(op);
        insertBarrier(op, builder);
        membarInfo->sync();
      }
      membarInfo->applyCallSummary(calleeMembarInfo);
    }
    return;
  } else if (isa<TMEMLoadOp, TMEMStoreOp, TMEMAllocOp, MMAv5OpInterface,
                 TMEMCopyOp>(op)) {
    for (const auto &access : getMemoryAccesses(op)) {
      if (!isTensorMemory(access.value))
        continue;
      for (AllocationSlice slice : getTMemSlices(access, regions)) {
        if (access.isRead)
          curBlockInfo.syncReadSlices[slice].insert(op);
        if (access.isWrite)
          curBlockInfo.syncWriteSlices[slice].insert(op);
      }
    }
  }

  if (membarInfo->pending.isIntersected(curBlockInfo, filter, &allocation)) {
    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    membarInfo->sync();
  }

  membarInfo->addBlockInfo(curBlockInfo);
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
    auto solver = createDataFlowSolver();
    auto *regions = solver->load<BufferRegionAnalysis>();
    if (failed(solver->initializeAndRun(mod))) {
      signalPassFailure();
      return;
    }
    ModuleMembarAnalysis analysis(allocation, filterFn);
    analysis.runAnalysis<TMemBarrierAnalysis>(*regions);
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
