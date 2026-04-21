#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

//===----------------------------------------------------------------------===//
//
// On Hopper+, async proxy is separate from generic proxy, so when shared memory
// is the generic proxy to the async proxy we need to insert a fence to ensure
// memory consistency.
// This pass analyzes dependencies and will conservatively insert fences to
// avoid race conditions between proxies. Async proxy is defined here:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#async-proxy
//
// This pass runs after shared memory allocation, to make sure we insert fences
// between ops accessing aliasing buffers if needed.
//
// We also run a fence insertion pass during optimization phase as it is easier
// to insert fences at optimial location based on structured control flow.
//
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONGPUPROXYFENCEINSERTION
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

bool isAsyncProxyWrite(Operation *op) {
  return isa<triton::nvidia_gpu::TMALoadLikeOpInterface,
             triton::nvidia_gpu::CLCTryCancelOp>(op);
}

Value getSmemDest(Operation *op) {
  if (auto tmaLoad = dyn_cast<triton::nvidia_gpu::TMALoadLikeOpInterface>(op)) {
    return tmaLoad.getResult();
  }
  if (auto clcTryCancelOp = dyn_cast<triton::nvidia_gpu::CLCTryCancelOp>(op)) {
    return clcTryCancelOp.getResult();
  }
  return Value();
}

bool isAsyncProxyRead(Operation *op) {
  return isa<triton::nvidia_gpu::WarpGroupDotOp,
             triton::nvidia_gpu::MMAv5OpInterface,
             triton::nvidia_gpu::TMEMCopyOp,
             triton::nvidia_gpu::TMAStoreLikeOpInterface>(op);
}

bool isAsyncProxyReadSource(Operation *op, Value value) {
  auto memDescType = dyn_cast<triton::gpu::MemDescType>(value.getType());
  if (!memDescType ||
      !isa<triton::gpu::SharedMemorySpaceAttr>(memDescType.getMemorySpace()))
    return false;
  if (auto tmaStore =
          dyn_cast<triton::nvidia_gpu::TMAStoreLikeOpInterface>(op)) {
    return value == tmaStore.getSrc();
  }
  if (auto warpGroupDotOp = dyn_cast<triton::nvidia_gpu::WarpGroupDotOp>(op)) {
    return value == warpGroupDotOp.getA() || value == warpGroupDotOp.getB();
  }
  if (auto mma = dyn_cast<triton::nvidia_gpu::MMAv5OpInterface>(op)) {
    return value == mma.getA() || value == mma.getB();
  }
  if (auto tmemCopyOp = dyn_cast<triton::nvidia_gpu::TMEMCopyOp>(op)) {
    return value == tmemCopyOp.getSrc();
  }
  return false;
}

bool ignoreOpForProxyFence(Operation *op) {
  return isAsyncProxyRead(op) || isAsyncProxyWrite(op) ||
         isa<triton::nvidia_gpu::ArriveBarrierOp,
             triton::nvidia_gpu::TMEMCopyOp, triton::nvidia_gpu::WaitBarrierOp,
             triton::nvidia_gpu::InitBarrierOp,
             triton::nvidia_gpu::InvalBarrierOp>(op);
}

bool filterFn(Operation *op, Operation *other, bool /*opIsRead*/,
              bool /*otherIsRead*/, Allocation *allocation) {
  return ignoreOpForProxyFence(other);
}

//===----------------------------------------------------------------------===//
// Proxy Fence Analysis
//===----------------------------------------------------------------------===//
class ProxyFenceAnalysis : public MembarOrFenceAnalysis {

public:
  explicit ProxyFenceAnalysis(Allocation *allocation, MembarFilterFn filter)
      : MembarOrFenceAnalysis(allocation, filter) {}

private:
  /// Updates the BlockInfo operation based on the operation.
  virtual void update(Operation *operation, BlockInfo *blockInfo,
                      FuncBlockInfoMapT *funcBlockInfoMap,
                      OpBuilder *builder) override;

  void insertFence(Operation *operation, OpBuilder *builder);
};

void ProxyFenceAnalysis::insertFence(Operation *op, OpBuilder *builder) {
  OpBuilder::InsertionGuard g(*builder);
  triton::nvidia_gpu::FenceAsyncSharedOp::create(*builder, op->getLoc(), false);
}

void ProxyFenceAnalysis::update(Operation *op, BlockInfo *blockInfo,
                                FuncBlockInfoMapT *funcBlockInfoMap,
                                OpBuilder *builder) {
  if (isa<triton::nvidia_gpu::FenceAsyncSharedOp>(op)) {
    // If the current op is a fence, we clear previous reads and writes
    blockInfo->sync();
    return;
  }
  BlockInfo curBlockInfo;
  BlockInfo proxyBlockInfo;

  auto scratchBufferId = Allocation::InvalidBufferId;
  if (isa<triton::CallOp>(op)) {
    // Inter-function dependencies
    auto callOpInterface = dyn_cast<CallOpInterface>(op);
    if (auto callee =
            dyn_cast<FunctionOpInterface>(callOpInterface.resolveCallable()))
      curBlockInfo = funcBlockInfoMap->lookup(callee);
  } else {
    // Intra-function dependencies
    if (auto memoryEffectOpInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Explicit buffer
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>>
          effectInstances;
      memoryEffectOpInterface.getEffects(effectInstances);
      for (auto effectInstance : effectInstances) {
        if (auto value = effectInstance.getValue()) {
          for (auto bufferId : allocation->getAllBufferIdsWithAliases(value)) {
            if (bufferId != Allocation::InvalidBufferId) {
              auto interval = allocation->getAllocatedInterval(bufferId);
              auto slice = AllocationSlice(value, interval, bufferId);

              if (isAsyncProxyWrite(op) && value == getSmemDest(op)) {
                proxyBlockInfo.syncWriteSlices[slice].insert(op);
              } else if (isAsyncProxyRead(op) &&
                         isAsyncProxyReadSource(op, value)) {
                // Safe fallback for async-proxy reads from shared memory when
                // the earlier FenceInsertionPass did not place a fence.
                proxyBlockInfo.syncReadSlices[slice].insert(op);
              } else if (isa<MemoryEffects::Write>(
                             effectInstance.getEffect())) {
                curBlockInfo.syncWriteSlices[slice].insert(op);
              } else if (isa<MemoryEffects::Read>(effectInstance.getEffect())) {
                curBlockInfo.syncReadSlices[slice].insert(op);
              }
            }
          }
        }
      }
    }
    scratchBufferId = allocation->getBufferId(op);
  }

  // Scratch buffer operations consist of a series of shared memory operations
  // starting from a shared memory write, followed by a series of shared memory
  // read/write operations, mark them as a read.
  if (scratchBufferId != Allocation::InvalidBufferId) {
    auto interval = allocation->getAllocatedInterval(scratchBufferId);
    auto scratchSlice = AllocationSlice(interval);
    curBlockInfo.syncReadSlices[scratchSlice].insert(op);
  }
  if (isAsyncProxyWrite(op) || isAsyncProxyRead(op)) {
    if (proxyBlockInfo.isIntersected(*blockInfo, filter, allocation)) {
      builder->setInsertionPoint(op);
      insertFence(op, builder);
      blockInfo->sync();
    }
  }

  // Update the region info, even if barrier is inserted, we have to maintain
  // the current op's read/write buffers.
  blockInfo->join(curBlockInfo);
}
} // namespace

struct ProxyFenceInsertionPass
    : public impl::TritonGPUProxyFenceInsertionBase<ProxyFenceInsertionPass> {

public:
  using impl::TritonGPUProxyFenceInsertionBase<
      ProxyFenceInsertionPass>::TritonGPUProxyFenceInsertionBase;
  void runOnOperation() override {
    // Only insert fences for compute capability 9.0
    if (computeCapability < 90)
      return;
    ModuleOp mod = getOperation();
    // This pass does not depend on the amount of shared memory allocated
    // so we can use the default allocation analysis scratch size function
    ModuleAllocation allocation(mod);
    ModuleMembarOrFenceAnalysis<ProxyFenceAnalysis> analysis(&allocation,
                                                             filterFn);
    analysis.run();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
