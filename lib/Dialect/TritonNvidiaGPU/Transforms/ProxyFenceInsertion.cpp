#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/CallGraph.h"
#include "triton/Analysis/Function.h"
#include "triton/Analysis/MemoryFrontier.h"
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

using BufferAccess = BufferRegionAccess;

struct ProxyBlockInfo {
  using Frontier = ScopedMemoryFrontier<uint8_t>;

  // Generic accesses since the last proxy fence and async accesses before the
  // first proxy fence reachable from function entry.
  Frontier generic;
  Frontier async;

  // Scopes fenced on every path from the function entry.
  uint8_t entryGenericFenced = 0;

  ProxyBlockInfo &join(const ProxyBlockInfo &other) {
    generic.join(other.generic);
    async.join(other.async);
    entryGenericFenced &= other.entryGenericFenced;
    return *this;
  }

  bool operator==(const ProxyBlockInfo &other) const {
    return std::tie(generic, async, entryGenericFenced) ==
           std::tie(other.generic, other.async, other.entryGenericFenced);
  }

  void fenceGeneric(uint8_t scopes) {
    if (scopes & kClusterScope)
      scopes |= kCTAScope;
    generic.eraseScopes(scopes);
    entryGenericFenced |= scopes;
  }
};

struct ProxyFenceFunctionAnalysis
    : public PostOrderFunctionAnalysis<ProxyBlockInfo> {
  using FuncMapT = PostOrderFunctionAnalysis<ProxyBlockInfo>::FuncMapT;

  ProxyFenceFunctionAnalysis(FunctionOpInterface function,
                             BufferRegionAnalysis &regions, uint8_t scopes)
      : function(function), regions(regions), scopes(scopes) {}

  void applyEffects(Operation *op, const ProxyBlockInfo &effects,
                    ProxyBlockInfo &state, OpBuilder &builder) {
    for (uint8_t scope : {kClusterScope, kCTAScope}) {
      if ((scopes & scope) && state.generic.hasHazard(effects.async, scope)) {
        builder.setInsertionPoint(op);
        FenceAsyncSharedOp::create(builder, op->getLoc(),
                                   scope == kClusterScope);
        state.fenceGeneric(scope);
        break;
      }
    }

    state.async.join(effects.async, scopes & ~state.entryGenericFenced);
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
      auto callee = cast<FunctionOpInterface>(call.resolveCallable());
      ProxyBlockInfo effects = funcMap->lookup(callee);
      for (auto *frontier : {&effects.generic, &effects.async})
        frontier->transformAccesses([&](BufferAccess access) {
          return regions.translateToCallsite(std::move(access), call, function,
                                             callee);
        });
      applyEffects(op, effects, *state, *builder);
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
      for (BufferAccess region : regions.getAccessRegions(access)) {
        if (access.isRead)
          frontier.addRead(region, accessScopes);
        if (access.isWrite)
          frontier.addWrite(std::move(region), accessScopes);
      }
    }

    if (auto sizeAttr = op->getAttrOfType<IntegerAttr>("allocation.size")) {
      uint32_t base =
          op->getAttrOfType<IntegerAttr>("allocation.offset").getInt();
      uint32_t size = sizeAttr.getInt();
      BufferRegionView scratch{{base, size, {}}, base};
      scratch.allocationFrame = regions.getOperationId(function.getOperation());
      AddressSet addresses = AddressSet::fromRange(base, size);
      unsigned numCTAs = hasCrossCTAScratch(op) ? gpu::lookupNumCTAs(op) : 1;
      for (unsigned cta = 0; cta < numCTAs; ++cta)
        scratch.region.ctaAddresses.emplace_back(cta, addresses);
      effects.generic.addRead(scratch, scopes);
      effects.generic.addWrite(std::move(scratch), scopes);
    }
    applyEffects(op, effects, *state, *builder);
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
