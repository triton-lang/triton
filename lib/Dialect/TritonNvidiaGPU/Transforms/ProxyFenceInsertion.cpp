#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/CallGraph.h"
#include "triton/Analysis/Function.h"
#include "triton/Analysis/MemoryFrontier.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/ConSanConstants.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include <cstdint>

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONGPUPROXYFENCEINSERTION
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

// Track generic-proxy frontiers with exact physical regions. Async-proxy
// accesses are retained only while they can still conflict with generic
// accesses preceding the current function; this lets callers place a fence
// before a call without inventing incoming effects in the callee.
namespace {

enum class ProxyFenceScope : uint8_t { CTA = 1, Cluster = 2 };

uint8_t scopeMask(ProxyFenceScope scope) { return static_cast<uint8_t>(scope); }
uint8_t coveredScopes(ProxyFenceScope scope) {
  return scopeMask(scope) | scopeMask(ProxyFenceScope::CTA);
}

ProxyFenceScope getProxyFenceScope(Operation *op) {
  if (auto load = dyn_cast<TMALoadLikeOpInterface>(op))
    if (load.getMulticast())
      return ProxyFenceScope::Cluster;
  if (auto mma = dyn_cast<MMAv5OpInterface>(op))
    if (mma.getTwoCtas())
      return ProxyFenceScope::Cluster;
  if (isa<TMEMCopyOp>(op) && getModuleTwoCTAs(op))
    return ProxyFenceScope::Cluster;
  if (isa<CLCTryCancelOp>(op) && gpu::lookupNumCTAs(op) > 1)
    return ProxyFenceScope::Cluster;
  return ProxyFenceScope::CTA;
}

using BufferAccess = BufferRegionAccess;
using ProxyBlockInfo = ProxyMemoryState<BufferAccess, uint8_t>;

ScratchBufferInfo getScratchInfo(Operation *op) {
  if (auto cvt = dyn_cast<gpu::ConvertLayoutOp>(op))
    return getConvertLayoutScratchBufferInfo(cvt, /*supportLdMatrix=*/true,
                                             /*supportStMatrix=*/true);
  unsigned size = defaultAllocationAnalysisScratchSizeFn(op);
  if (isa<gpu::WarpSpecializeOp>(op))
    if (auto extra = op->getAttrOfType<IntegerAttr>(
            instrument::kConSanExtraCaptureBytesAttr))
      size += extra.getInt();
  return {size};
}

class ProxyFenceFunctionAnalysis
    : public PostOrderFunctionAnalysis<ProxyBlockInfo> {
  using FuncMapT = PostOrderFunctionAnalysis<ProxyBlockInfo>::FuncMapT;

public:
  ProxyFenceFunctionAnalysis(FunctionOpInterface function,
                             BufferRegionAnalysis &regions, uint8_t scopes)
      : function(function), regions(regions), scopes(scopes) {}

private:
  ProxyBlockInfo getEntryState() const override {
    ProxyBlockInfo state;
    state.entryGenericUnfenced = scopes;
    return state;
  }

  void applyEffects(Operation *op, ProxyBlockInfo &effects,
                    ProxyBlockInfo *state, OpBuilder *builder,
                    bool fromCall = false) {
    for (ProxyFenceScope scope :
         {ProxyFenceScope::Cluster, ProxyFenceScope::CTA}) {
      auto mayAlias = [](const BufferAccess &lhs, const BufferAccess &rhs) {
        return !lhs || !rhs || lhs->intersects(*rhs);
      };
      if ((scopes & scopeMask(scope)) &&
          state->needsFenceBefore(effects, scopeMask(scope), mayAlias)) {
        builder->setInsertionPoint(op);
        FenceAsyncSharedOp::create(*builder, op->getLoc(),
                                   scope == ProxyFenceScope::Cluster);
        state->fenceGeneric(coveredScopes(scope));
        break;
      }
    }

    state->joinAsync(effects, state->entryGenericUnfenced);
    if (fromCall) {
      uint8_t fencedScopes = scopes & ~effects.entryGenericUnfenced;
      if (fencedScopes & scopeMask(ProxyFenceScope::Cluster))
        state->fenceGeneric(coveredScopes(ProxyFenceScope::Cluster));
      else if (fencedScopes & scopeMask(ProxyFenceScope::CTA))
        state->fenceGeneric(coveredScopes(ProxyFenceScope::CTA));
    }
    state->joinGeneric(effects);
  }

  void update(Operation *op, ProxyBlockInfo *state, FuncMapT *funcMap,
              OpBuilder *builder) override {
    if (auto fence = dyn_cast<FenceAsyncSharedOp>(op)) {
      state->fenceGeneric(coveredScopes(fence.getBCluster()
                                            ? ProxyFenceScope::Cluster
                                            : ProxyFenceScope::CTA));
      return;
    }

    if (auto call = dyn_cast<CallOpInterface>(op)) {
      auto callee =
          dyn_cast_or_null<FunctionOpInterface>(call.resolveCallable());
      if (!callee)
        return;
      ProxyBlockInfo effects = funcMap->lookup(callee);
      effects.transformAccesses([&](BufferAccess access) {
        return regions.translateToCallsite(std::move(access), call, function,
                                           callee);
      });
      applyEffects(op, effects, state, builder, /*fromCall=*/true);
      return;
    }

    ProxyBlockInfo effects;
    for (const MemoryAccess &access : getMemoryAccesses(op)) {
      if (!access.isShared() || access.isShared(gpu::SharedKind::Barrier))
        continue;

      bool async = access.isShared(gpu::SharedKind::Async);
      uint8_t accessScopes = async ? scopeMask(getProxyFenceScope(op)) : scopes;
      ProxyBlockInfo::Frontier &frontier =
          async ? effects.async : effects.generic;
      for (BufferAccess region : regions.getAccessRegions(access.value))
        frontier.add(std::move(region), access.isRead, access.isWrite,
                     accessScopes);
    }

    ScratchBufferInfo scratch = getScratchInfo(op);
    if (BufferAccess region = regions.getScratchRegion(
            function, op, scratch.size, scratch.crossCTA))
      effects.generic.add(std::move(region), /*isRead=*/true,
                          /*isWrite=*/true, scopes);
    applyEffects(op, effects, state, builder);
  }

  FunctionOpInterface function;
  BufferRegionAnalysis &regions;
  uint8_t scopes;
};

} // namespace

struct ProxyFenceInsertionPass
    : public impl::TritonGPUProxyFenceInsertionBase<ProxyFenceInsertionPass> {
  using impl::TritonGPUProxyFenceInsertionBase<
      ProxyFenceInsertionPass>::TritonGPUProxyFenceInsertionBase;

  void runOnOperation() override {
    if (computeCapability < 90)
      return;

    ModuleOp module = getOperation();
    uint8_t scopes = 0;
    module.walk([&](Operation *op) {
      if (!hasSharedAccess(op, gpu::SharedKind::Async))
        return WalkResult::advance();
      scopes |=
          scopeMask(getProxyFenceScope(op)) | scopeMask(ProxyFenceScope::CTA);
      return scopes & scopeMask(ProxyFenceScope::Cluster)
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    if (!scopes)
      return;

    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    auto *regions = solver->load<BufferRegionAnalysis>();
    if (failed(solver->initializeAndRun(module)))
      return signalPassFailure();

    CallGraph<ProxyBlockInfo> callGraph(module);
    CallGraph<ProxyBlockInfo>::FuncDataMapT summaries;
    callGraph.walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        [](CallOpInterface, FunctionOpInterface) {},
        [&](FunctionOpInterface function) {
          if (summaries.try_emplace(function).second)
            ProxyFenceFunctionAnalysis(function, *regions, scopes)
                .run(function, summaries);
        });
  }
};

} // namespace mlir::triton::nvidia_gpu
