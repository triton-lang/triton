#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Function.h"
#include "triton/Analysis/MemoryFrontier.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
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
//
// The reverse direction is tracked for async accesses whose completion carries
// no implicit generic-async proxy fence (see needsFenceAfterAsyncAccess). They
// stay pending until an overlapping generic access, which gets a fence directly
// before it and therefore after the program's completion wait. Callee generic
// accesses not directly preceded by a fence are exposed to callers, which fence
// before the call if those accesses conflict with pending async accesses.
namespace {

enum : uint8_t { kCTAScope = 1, kClusterScope = 2 };

uint8_t getProxyFenceScope(Operation *op) {
  auto load = dyn_cast<TMALoadLikeOpInterface>(op);
  auto mma = dyn_cast<MMAv5OpInterface>(op);
  bool cluster = (load && load.getMulticast()) || (mma && mma.getTwoCtas()) ||
                 (isa<TMEMCopyOp>(op) && getModuleTwoCTAs(op)) ||
                 (isa<CLCTryCancelOp>(op) && gpu::lookupNumCTAs(op) > 1);
  return cluster ? kClusterScope : kCTAScope;
}

// Returns true if the op's async-proxy shared-memory accesses carry no
// implicit generic-async proxy fence at completion, so a later generic access
// to the same bytes needs an explicit fence. PTX states such a fence for
// cp.async.bulk and wgmma, but not for tcgen05.mma and tcgen05.cp.
bool needsFenceAfterAsyncAccess(Operation *op) {
  return isa<MMAv5OpInterface, TMEMCopyOp>(op);
}
// A cluster-scope fence also orders CTA-scope accesses.
uint8_t getFencedScopes(uint8_t scope) {
  return scope & kClusterScope ? scope | kCTAScope : scope;
}
// Returns the scopes ordered by a proxy fence directly preceding `op`, or 0.
// Such a fence sits where this pass would insert one, so it orders every async
// access that completed before `op`.
uint8_t getFenceBeforeScopes(Operation *op) {
  auto fence = dyn_cast_or_null<FenceAsyncSharedOp>(op->getPrevNode());
  if (!fence)
    return 0;
  return getFencedScopes(fence.getBCluster() ? kClusterScope : kCTAScope);
}

using BufferAccess = const BufferRegionFootprint *;

struct ProxyBlockInfo {
  using Frontier = ScopedMemoryFrontier<uint8_t>;

  // Generic accesses since the last proxy fence and async accesses before the
  // first proxy fence reachable from function entry.
  Frontier generic;
  Frontier async;
  // Async accesses from ops that lack a completion fence
  // and are not yet fenced before later generic accesses.
  Frontier unfencedAsync;
  // Generic accesses not directly preceded by a proxy fence. Async accesses
  // left unfenced by a caller need a fence before the call if they conflict.
  Frontier exposedGeneric;

  // Scopes fenced on every path from the function entry.
  uint8_t entryGenericFenced = 0;

  ProxyBlockInfo &join(const ProxyBlockInfo &other) {
    generic.join(other.generic);
    async.join(other.async);
    unfencedAsync.join(other.unfencedAsync);
    exposedGeneric.join(other.exposedGeneric);
    entryGenericFenced &= other.entryGenericFenced;
    return *this;
  }

  bool operator==(const ProxyBlockInfo &other) const {
    return std::tie(generic, async, unfencedAsync, exposedGeneric,
                    entryGenericFenced) ==
           std::tie(other.generic, other.async, other.unfencedAsync,
                    other.exposedGeneric, other.entryGenericFenced);
  }

  void fenceGeneric(uint8_t scopes) {
    scopes = getFencedScopes(scopes);
    generic.eraseScopes(scopes);
    entryGenericFenced |= scopes;
  }
};

struct ProxyFenceFunctionAnalysis
    : public PostOrderFunctionAnalysis<ProxyBlockInfo> {
  using FuncMapT = PostOrderFunctionAnalysis<ProxyBlockInfo>::FuncMapT;

  ProxyFenceFunctionAnalysis(BufferRegionAnalysis &regions, uint8_t scopes)
      : regions(regions), scopes(scopes) {}

  void applyEffects(Operation *op, const ProxyBlockInfo &effects,
                    ProxyBlockInfo &state, OpBuilder &builder) {
    for (uint8_t scope : {kClusterScope, kCTAScope}) {
      if (!(scopes & scope))
        continue;
      bool genericToAsync = state.generic.hasHazard(effects.async, scope);
      bool asyncToGeneric =
          state.unfencedAsync.hasHazard(effects.exposedGeneric, scope);
      if (!genericToAsync && !asyncToGeneric)
        continue;
      if (!(getFenceBeforeScopes(op) & scope)) {
        builder.setInsertionPoint(op);
        FenceAsyncSharedOp::create(builder, op->getLoc(),
                                   scope == kClusterScope);
      }
      state.fenceGeneric(scope);
      // A race-free program has waited for the conflicting async accesses
      // before this generic access, so the fence orders them. Other unfenced
      // accesses may still be in flight, so keep them.
      if (asyncToGeneric)
        state.unfencedAsync.eraseConflicts(effects.exposedGeneric,
                                           getFencedScopes(scope));
      break;
    }

    // Generic accesses stay exposed to async accesses pending at function
    // entry unless a fence directly precedes them.
    ProxyBlockInfo::Frontier exposed = effects.exposedGeneric;
    exposed.eraseScopes(getFenceBeforeScopes(op));
    state.exposedGeneric.join(exposed);

    state.async.join(effects.async, scopes & ~state.entryGenericFenced);
    state.unfencedAsync.join(effects.unfencedAsync);
    if (isa<CallOpInterface>(op) && effects.entryGenericFenced)
      state.fenceGeneric(effects.entryGenericFenced);
    state.generic.join(effects.generic);
  }

  void update(Operation *op, ProxyBlockInfo *state, FuncMapT *funcMap,
              OpBuilder *builder) override {
    if (auto fence = dyn_cast<FenceAsyncSharedOp>(op)) {
      state->fenceGeneric(fence.getBCluster() ? kClusterScope : kCTAScope);
      return;
    }

    if (auto call = dyn_cast<CallOpInterface>(op)) {
      auto effects = getCallSummary(
          call, *funcMap,
          [&](ProxyBlockInfo &effects, FunctionOpInterface callee) {
            for (auto *frontier :
                 {&effects.generic, &effects.async, &effects.unfencedAsync,
                  &effects.exposedGeneric})
              frontier->transformAccesses([&](BufferAccess access) {
                return regions.translateToCallsite(access, call, callee);
              });
          });
      // Async accesses pending before the call are fenced before it when they
      // conflict with the callee's exposed generic accesses. This assumes the
      // caller waits for their completion before the call; a wait inside the
      // callee is not modeled.
      if (effects)
        applyEffects(op, *effects, *state, *builder);
      return;
    }

    ProxyBlockInfo effects;
    for (const MemoryAccess &access : getMemoryAccesses(op)) {
      if (!access.isShared() || access.isShared(gpu::SharedKind::Barrier))
        continue;

      bool async = access.isShared(gpu::SharedKind::Async);
      uint8_t accessScopes = async ? getProxyFenceScope(op) : scopes;
      ProxyBlockInfo::Frontier &frontier =
          async ? effects.async : effects.generic;
      BufferAccess footprint = regions.getFootprint(access.value);
      if (access.isRead)
        frontier.addRead(footprint, accessScopes);
      if (access.isWrite)
        frontier.addWrite(footprint, accessScopes);
      if (async && needsFenceAfterAsyncAccess(op)) {
        if (access.isRead)
          effects.unfencedAsync.addRead(footprint, accessScopes);
        if (access.isWrite)
          effects.unfencedAsync.addWrite(footprint, accessScopes);
      }
    }

    if (op->hasAttr("allocation.size")) {
      BufferAccess scratch = regions.getScratchFootprint(op);
      effects.generic.addRead(scratch, scopes);
      effects.generic.addWrite(scratch, scopes);
    }
    effects.exposedGeneric = effects.generic;
    applyEffects(op, effects, *state, *builder);
  }

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
      if (hasSharedAccess(op, gpu::SharedKind::Async))
        scopes |= getProxyFenceScope(op) | kCTAScope;
      return scopes & kClusterScope ? WalkResult::interrupt()
                                    : WalkResult::advance();
    });
    if (!scopes)
      return;

    auto solver = createDataFlowSolver();
    auto *regions = solver->load<BufferRegionAnalysis>();
    if (failed(solver->initializeAndRun(module)))
      return signalPassFailure();

    ProxyFenceFunctionAnalysis::runModule(module, [&](FunctionOpInterface) {
      return ProxyFenceFunctionAnalysis(*regions, scopes);
    });
  }
};

} // namespace mlir::triton::nvidia_gpu
