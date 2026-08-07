#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/BufferRegion.h"
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

using gpu::SharedKind;

bool ignoreOpForProxyFence(Operation *op) {
  return !hasSharedAccess(op, SharedKind::Generic) && hasSharedAccess(op);
}

bool filterFn(Operation *op, Operation *other, bool /*opIsRead*/,
              bool /*otherIsRead*/, Allocation *allocation) {
  return ignoreOpForProxyFence(other);
}

enum class ProxyFenceScope { CTA, Cluster };

ProxyFenceScope getProxyFenceScope(Operation *op) {
  // Multicast TMA and two-CTA tensor-core operations access peer-CTA shared
  // memory. Multi-CTA CLC multicasts its result to every CTA in the cluster.
  if (auto tma = dyn_cast<triton::nvidia_gpu::TMALoadLikeOpInterface>(op)) {
    if (tma.getMulticast())
      return ProxyFenceScope::Cluster;
  }
  if (auto mma = dyn_cast<triton::nvidia_gpu::MMAv5OpInterface>(op)) {
    if (mma.getTwoCtas())
      return ProxyFenceScope::Cluster;
  }
  if (isa<triton::nvidia_gpu::TMEMCopyOp>(op) &&
      triton::nvidia_gpu::getModuleTwoCTAs(op))
    return ProxyFenceScope::Cluster;
  if (isa<triton::nvidia_gpu::CLCTryCancelOp>(op) &&
      triton::gpu::lookupNumCTAs(op) > 1)
    return ProxyFenceScope::Cluster;
  return ProxyFenceScope::CTA;
}

//===----------------------------------------------------------------------===//
// Proxy Fence Analysis
//===----------------------------------------------------------------------===//
template <ProxyFenceScope scope>
class ProxyFenceAnalysis : public MembarOrFenceAnalysis {

public:
  using MembarOrFenceAnalysis::MembarOrFenceAnalysis;

private:
  void update(Operation *operation, BlockInfo *blockInfo, FuncMapT *funcMap,
              OpBuilder *builder) override;

  void insertFence(Operation *operation, OpBuilder *builder);
};

template <ProxyFenceScope scope>
void ProxyFenceAnalysis<scope>::insertFence(Operation *op, OpBuilder *builder) {
  OpBuilder::InsertionGuard g(*builder);
  triton::nvidia_gpu::FenceAsyncSharedOp::create(
      *builder, op->getLoc(), scope == ProxyFenceScope::Cluster);
}

template <ProxyFenceScope scope>
void ProxyFenceAnalysis<scope>::update(Operation *op, BlockInfo *blockInfo,
                                       FuncMapT *funcMap, OpBuilder *builder) {
  if (auto fence = dyn_cast<triton::nvidia_gpu::FenceAsyncSharedOp>(op)) {
    // A cluster fence covers both frontiers, while a CTA fence only covers the
    // CTA frontier.
    if (scope == ProxyFenceScope::CTA || fence.getBCluster())
      blockInfo->sync();
    return;
  }
  BlockInfo curBlockInfo;
  BlockInfo proxyBlockInfo;
  auto accesses = getMemoryAccesses(op);
  bool isProxyOp =
      hasSharedAccess(op, SharedKind::Async) && getProxyFenceScope(op) == scope;

  auto scratchBufferId = Allocation::InvalidBufferId;
  if (isa<triton::CallOp>(op)) {
    // Inter-function dependencies
    auto callOpInterface = dyn_cast<CallOpInterface>(op);
    if (auto callee =
            dyn_cast<FunctionOpInterface>(callOpInterface.resolveCallable()))
      curBlockInfo = funcMap->lookup(callee);
  } else {
    // Intra-function dependencies
    // Explicit buffers are classified by their memory effects rather than
    // operation-specific proxy lists.
    for (const auto &access : accesses) {
      if (!access.isShared() || access.isShared(SharedKind::Barrier))
        continue;
      for (auto bufferId :
           allocation.getAllBufferIdsWithAliases(access.value)) {
        if (bufferId == Allocation::InvalidBufferId)
          continue;
        auto interval = allocation.getAllocatedInterval(bufferId);
        auto slice = AllocationSlice(access.value, interval, bufferId);
        bool async = access.isShared(SharedKind::Async);
        BlockInfo &effects = async ? proxyBlockInfo : curBlockInfo;
        if (async && !isProxyOp)
          continue;
        if (access.isWrite)
          effects.syncWriteSlices[slice].insert(op);
        if (access.isRead)
          effects.syncReadSlices[slice].insert(op);
      }
    }
    scratchBufferId = allocation.getBufferId(op);
  }

  // Scratch buffer operations consist of a series of shared memory operations
  // starting from a shared memory write, followed by a series of shared memory
  // read/write operations, mark them as a read.
  if (scratchBufferId != Allocation::InvalidBufferId) {
    auto interval = allocation.getAllocatedInterval(scratchBufferId);
    auto scratchSlice = AllocationSlice(interval);
    curBlockInfo.syncReadSlices[scratchSlice].insert(op);
  }
  if (isProxyOp) {
    if (proxyBlockInfo.isIntersected(*blockInfo, filter, &allocation)) {
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
    // Keep independent frontiers for cluster- and CTA-scoped fences. Run the
    // cluster analysis first so the CTA analysis can observe any cluster
    // fences it inserts.
    bool hasClusterProxyOp =
        mod.walk([](Operation *op) {
             return getProxyFenceScope(op) == ProxyFenceScope::Cluster
                        ? WalkResult::interrupt()
                        : WalkResult::advance();
           })
            .wasInterrupted();
    if (hasClusterProxyOp) {
      ModuleMembarOrFenceAnalysis<ProxyFenceAnalysis<ProxyFenceScope::Cluster>>
          clusterAnalysis(allocation, filterFn);
      clusterAnalysis.run();
    }
    ModuleMembarOrFenceAnalysis<ProxyFenceAnalysis<ProxyFenceScope::CTA>>
        ctaAnalysis(allocation, filterFn);
    ctaAnalysis.run();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
