#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/CallGraph.h"
#include "triton/Analysis/Function.h"
#include "triton/Analysis/MemoryFrontier.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonInstrument/IR/ConSanConstants.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"

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
  using Frontier = ScopedMemoryFrontier<BufferAccess, uint8_t>;

  // Generic accesses since the last proxy fence and async accesses before the
  // first proxy fence reachable from function entry.
  Frontier generic;
  Frontier async;

  // Scope bits for which no fence on every path has covered accesses preceding
  // the function.
  uint8_t entryGenericUnfenced = 0;

  ProxyBlockInfo &join(const ProxyBlockInfo &other) {
    generic.join(other.generic);
    async.join(other.async);
    entryGenericUnfenced |= other.entryGenericUnfenced;
    return *this;
  }

  bool operator==(const ProxyBlockInfo &other) const {
    return std::tie(generic, async, entryGenericUnfenced) ==
           std::tie(other.generic, other.async, other.entryGenericUnfenced);
  }

  void fenceGeneric(uint8_t scopes) {
    if (scopes & kClusterScope)
      scopes |= kCTAScope;
    generic.eraseScopes(scopes);
    entryGenericUnfenced &= ~scopes;
  }
};

std::pair<unsigned, bool> getScratchInfo(Operation *op) {
  if (auto cvt = dyn_cast<gpu::ConvertLayoutOp>(op)) {
    if (!cvtNeedsSharedMemory(cvt))
      return {};

    RankedTensorType srcTy = cvt.getSrc().getType();
    LinearLayout src = gpu::toLinearLayout(srcTy);
    LinearLayout dst = gpu::toLinearLayout(cvt.getType());
    src = actionRemoveBroadcastedRegs(src).apply(src);
    dst = actionRemoveBroadcastedRegs(dst).apply(dst);

    MLIRContext *ctx = op->getContext();
    StringAttr block = StringAttr::get(ctx, "block");
    bool crossCTA = !dst.invertAndCompose(src).isTrivialOver({block});
    unsigned bitwidth = getBitwidth(srcTy);
    SmallVector<gpu::LocalMemOpTile> srcTiles{
        {{}, {0, 1, 2}}, {{0, 1}, {2, 3, 4}}, {{2, 3, 4}, {0, 1}}};
    srcTiles.resize(1 + (bitwidth <= 32) + (bitwidth == 16));
    SmallVector<gpu::LocalMemOpTile> dstTiles = srcTiles;
    dstTiles.resize(crossCTA ? 1 : srcTiles.size());

    auto [scratch, _] =
        gpu::optimalSwizzling(src, dst, srcTiles, dstTiles, bitwidth);
    unsigned divisor = scratch.getInDimSize(StringAttr::get(ctx, "reps")) *
                       product(gpu::getCTASplitNum(srcTy.getEncoding()));
    return {scratch.getTotalOutDimSize() / divisor * bitwidth / 8, crossCTA};
  }

  unsigned size = defaultAllocationAnalysisScratchSizeFn(op);
  if (isa<gpu::WarpSpecializeOp>(op))
    if (auto extra = op->getAttrOfType<IntegerAttr>(
            instrument::kConSanExtraCaptureBytesAttr))
      size += extra.getInt();
  return {size, false};
}

struct ProxyFenceFunctionAnalysis
    : public PostOrderFunctionAnalysis<ProxyBlockInfo> {
  using FuncMapT = PostOrderFunctionAnalysis<ProxyBlockInfo>::FuncMapT;

  ProxyFenceFunctionAnalysis(FunctionOpInterface function,
                             BufferRegionAnalysis &regions, uint8_t scopes)
      : function(function), regions(regions), scopes(scopes) {}

  ProxyBlockInfo getEntryState() const override { return {{}, {}, scopes}; }

  void applyEffects(Operation *op, const ProxyBlockInfo &effects,
                    ProxyBlockInfo &state, OpBuilder &builder) {
    auto mayAlias = [](const BufferAccess &lhs, const BufferAccess &rhs) {
      return !lhs || !rhs || lhs->intersects(*rhs);
    };
    for (uint8_t scope : {kClusterScope, kCTAScope}) {
      if ((scopes & scope) &&
          state.generic.hasHazard(effects.async, scope, mayAlias)) {
        builder.setInsertionPoint(op);
        FenceAsyncSharedOp::create(builder, op->getLoc(),
                                   scope == kClusterScope);
        state.fenceGeneric(scope);
        break;
      }
    }

    state.async.join(effects.async, state.entryGenericUnfenced);
    if (uint8_t fenced = scopes & ~effects.entryGenericUnfenced;
        isa<CallOpInterface>(op) && fenced)
      state.fenceGeneric(fenced);
    state.generic.join(effects.generic);
  }

  void update(Operation *op, ProxyBlockInfo *state, FuncMapT *funcMap,
              OpBuilder *builder) override {
    if (auto fence = dyn_cast<FenceAsyncSharedOp>(op)) {
      state->fenceGeneric(fence.getBCluster() ? kClusterScope : kCTAScope);
      return;
    }

    if (auto call = dyn_cast<CallOpInterface>(op)) {
      auto callee =
          dyn_cast_or_null<FunctionOpInterface>(call.resolveCallable());
      if (!callee)
        return;
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
      for (BufferAccess region : regions.getAccessRegions(access.value)) {
        if (access.isRead)
          frontier.addRead(region, accessScopes);
        if (access.isWrite)
          frontier.addWrite(std::move(region), accessScopes);
      }
    }

    if (auto offset = op->getAttrOfType<IntegerAttr>("allocation.offset")) {
      auto [size, crossCTA] = getScratchInfo(op);
      if (size) {
        uint32_t base = offset.getInt();
        BufferRegionView scratch{{base, size, {}}, base};
        scratch.allocationFrame =
            regions.getOperationId(function.getOperation());
        AddressSet addresses = AddressSet::fromRange(base, size);
        unsigned numCTAs = crossCTA ? gpu::lookupNumCTAs(op) : 1;
        for (unsigned cta = 0; cta < numCTAs; ++cta)
          scratch.region.ctaAddresses.emplace_back(cta, addresses);
        effects.generic.addRead(scratch, scopes);
        effects.generic.addWrite(std::move(scratch), scopes);
      }
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
