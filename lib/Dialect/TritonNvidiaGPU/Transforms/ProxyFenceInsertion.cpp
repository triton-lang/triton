#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/CallGraph.h"
#include "triton/Analysis/Function.h"
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
#include <map>
#include <optional>

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

using BufferAccess = std::optional<BufferRegionView>;

struct ProxyBlockInfo {
  using AccessMap = std::map<BufferAccess, uint8_t>;

  // Generic accesses since the last proxy fence.
  AccessMap genericReads;
  AccessMap genericWrites;
  // Async accesses before the first proxy fence reachable from function entry.
  AccessMap asyncReads;
  AccessMap asyncWrites;

  // Scope bits for which no fence on every path has covered accesses preceding
  // the function.
  uint8_t entryGenericUnfenced = 0;

  ProxyBlockInfo &join(const ProxyBlockInfo &other) {
    join(genericReads, other.genericReads);
    join(genericWrites, other.genericWrites);
    join(asyncReads, other.asyncReads);
    join(asyncWrites, other.asyncWrites);
    entryGenericUnfenced |= other.entryGenericUnfenced;
    return *this;
  }

  bool operator==(const ProxyBlockInfo &other) const {
    return std::tie(genericReads, genericWrites, asyncReads, asyncWrites,
                    entryGenericUnfenced) ==
           std::tie(other.genericReads, other.genericWrites, other.asyncReads,
                    other.asyncWrites, other.entryGenericUnfenced);
  }

  void fenceGeneric(ProxyFenceScope scope) {
    uint8_t fencedScopes = scopeMask(scope) | scopeMask(ProxyFenceScope::CTA);
    for (AccessMap *accesses : {&genericReads, &genericWrites})
      for (auto it = accesses->begin(); it != accesses->end();)
        if (it->second &= ~fencedScopes)
          ++it;
        else
          it = accesses->erase(it);
    entryGenericUnfenced &= ~fencedScopes;
  }

  bool needsFenceBefore(const ProxyBlockInfo &async,
                        ProxyFenceScope scope) const {
    return intersects(genericWrites, async.asyncReads, scope) ||
           intersects(genericReads, async.asyncWrites, scope) ||
           intersects(genericWrites, async.asyncWrites, scope);
  }

  void joinGeneric(const ProxyBlockInfo &other) {
    join(genericReads, other.genericReads);
    join(genericWrites, other.genericWrites);
  }

  void joinAsync(const ProxyBlockInfo &other, uint8_t scopes) {
    join(asyncReads, other.asyncReads, scopes);
    join(asyncWrites, other.asyncWrites, scopes);
  }

private:
  static void join(AccessMap &into, const AccessMap &from,
                   uint8_t scopes = 0xff) {
    for (const auto &[access, accessScopes] : from)
      if (uint8_t activeScopes = accessScopes & scopes)
        into[access] |= activeScopes;
  }

  static bool mayAlias(const BufferAccess &lhs, const BufferAccess &rhs) {
    return !lhs || !rhs || lhs->intersects(*rhs);
  }

  static bool intersects(const AccessMap &lhs, const AccessMap &rhs,
                         ProxyFenceScope scope) {
    return llvm::any_of(lhs, [&](const auto &left) {
      return (left.second & scopeMask(scope)) &&
             llvm::any_of(rhs, [&](const auto &right) {
               return (right.second & scopeMask(scope)) &&
                      mayAlias(left.first, right.first);
             });
    });
  }
};

SmallVector<BufferAccess> getBufferAccesses(BufferRegionAnalysis &analysis,
                                            Value value) {
  const RegionInfo &info = analysis.getRegionInfo(value);
  if (info.kind != RegionInfo::Kind::Exact)
    return {std::nullopt};
  SmallVector<BufferAccess> accesses;
  for (const BufferRegionView &view : info.views)
    accesses.push_back(view);
  if (accesses.empty())
    accesses.push_back(std::nullopt);
  return accesses;
}

struct ScratchInfo {
  unsigned size = 0;
  bool crossCTA = false;
};

ScratchInfo getScratchInfo(Operation *op) {
  if (auto cvt = dyn_cast<gpu::ConvertLayoutOp>(op)) {
    RankedTensorType srcTy = cvt.getSrc().getType();
    RankedTensorType dstTy = cvt.getType();
    if (!cvtNeedsSharedMemory(cvt))
      return {};

    LinearLayout src = gpu::toLinearLayout(srcTy);
    LinearLayout dst = gpu::toLinearLayout(dstTy);
    src = actionRemoveBroadcastedRegs(src).apply(src);
    dst = actionRemoveBroadcastedRegs(dst).apply(dst);

    MLIRContext *ctx = op->getContext();
    StringAttr block = StringAttr::get(ctx, "block");
    bool crossCTA = !dst.invertAndCompose(src).isTrivialOver({block});
    unsigned bitwidth = getBitwidth(srcTy);
    SmallVector<gpu::LocalMemOpTile> srcTiles{{{}, {0, 1, 2}}};
    SmallVector<gpu::LocalMemOpTile> dstTiles = srcTiles;
    if (bitwidth <= 32) {
      srcTiles.push_back({{0, 1}, {2, 3, 4}});
      if (!crossCTA)
        dstTiles.push_back(srcTiles.back());
      if (bitwidth == 16) {
        srcTiles.push_back({{2, 3, 4}, {0, 1}});
        if (!crossCTA)
          dstTiles.push_back(srcTiles.back());
      }
    }

    auto [scratch, _] =
        gpu::optimalSwizzling(src, dst, srcTiles, dstTiles, bitwidth);
    unsigned reps = scratch.getInDimSize(StringAttr::get(ctx, "reps"));
    unsigned numCTAs = product(gpu::getCTASplitNum(srcTy.getEncoding()));
    return {scratch.getTotalOutDimSize() / (reps * numCTAs) * bitwidth / 8,
            crossCTA};
  }

  unsigned size = defaultAllocationAnalysisScratchSizeFn(op);
  if (isa<gpu::WarpSpecializeOp>(op))
    if (auto extra = op->getAttrOfType<IntegerAttr>(
            instrument::kConSanExtraCaptureBytesAttr))
      size += extra.getInt();
  return {size};
}

std::optional<BufferRegionView> getScratchAccess(BufferRegionAnalysis &analysis,
                                                 FunctionOpInterface function,
                                                 Operation *op) {
  auto offset = op->getAttrOfType<IntegerAttr>("allocation.offset");
  ScratchInfo scratch = getScratchInfo(op);
  if (!offset || !scratch.size)
    return std::nullopt;

  uint32_t base = offset.getInt();
  AddressSet addresses = AddressSet::fromRange(base, scratch.size);
  SmallVector<BufferRegion::CTAAddresses, 2> ctaAddresses;
  unsigned numCTAs = scratch.crossCTA ? gpu::lookupNumCTAs(op) : 1;
  for (unsigned cta = 0; cta < numCTAs; ++cta)
    ctaAddresses.emplace_back(cta, addresses);
  return BufferRegionView{{base, scratch.size, std::move(ctaAddresses)},
                          base,
                          /*affineOffset=*/0,
                          /*partitionBases=*/{},
                          /*affinePartitionOffset=*/0,
                          /*affineCTAOffset=*/0,
                          analysis.getOperationId(function.getOperation())};
}

void translateAccesses(BufferRegionAnalysis &analysis,
                       ProxyBlockInfo::AccessMap &accesses,
                       FunctionOpInterface caller, FunctionOpInterface callee,
                       uint32_t offset) {
  ProxyBlockInfo::AccessMap translated;
  uint32_t calleeFrame = analysis.getOperationId(callee.getOperation());
  uint32_t callerFrame = analysis.getOperationId(caller.getOperation());
  for (const auto &[original, scopes] : accesses) {
    BufferAccess access = original;
    if (access && access->allocationFrame == calleeFrame) {
      access->region.baseOffset += offset;
      for (auto &[cta, addresses] : access->region.ctaAddresses)
        addresses = addresses.translated(offset);
      access->storageBase += offset;
      for (uint32_t &base : access->partitionBases)
        base += offset;
      access->allocationFrame = callerFrame;
    }
    translated[std::move(access)] |= scopes;
  }
  accesses = std::move(translated);
}

void translateCalleeState(BufferRegionAnalysis &analysis, ProxyBlockInfo &state,
                          CallOpInterface call, FunctionOpInterface caller,
                          FunctionOpInterface callee) {
  auto offset = call->getAttrOfType<IntegerAttr>("allocation.offset");
  uint32_t callOffset = offset ? offset.getInt() : 0;
  translateAccesses(analysis, state.genericReads, caller, callee, callOffset);
  translateAccesses(analysis, state.genericWrites, caller, callee, callOffset);
  translateAccesses(analysis, state.asyncReads, caller, callee, callOffset);
  translateAccesses(analysis, state.asyncWrites, caller, callee, callOffset);
}

class ProxyFenceFunctionAnalysis
    : public PostOrderFunctionAnalysis<ProxyBlockInfo> {
  using Base = PostOrderFunctionAnalysis<ProxyBlockInfo>;
  using FuncMapT = Base::FuncMapT;

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
      if ((scopes & scopeMask(scope)) &&
          state->needsFenceBefore(effects, scope)) {
        builder->setInsertionPoint(op);
        FenceAsyncSharedOp::create(*builder, op->getLoc(),
                                   scope == ProxyFenceScope::Cluster);
        state->fenceGeneric(scope);
        break;
      }
    }

    state->joinAsync(effects, state->entryGenericUnfenced);
    if (fromCall) {
      uint8_t fencedScopes = scopes & ~effects.entryGenericUnfenced;
      if (fencedScopes & scopeMask(ProxyFenceScope::Cluster))
        state->fenceGeneric(ProxyFenceScope::Cluster);
      else if (fencedScopes & scopeMask(ProxyFenceScope::CTA))
        state->fenceGeneric(ProxyFenceScope::CTA);
    }
    state->joinGeneric(effects);
  }

  void update(Operation *op, ProxyBlockInfo *state, FuncMapT *funcMap,
              OpBuilder *builder) override {
    if (auto fence = dyn_cast<FenceAsyncSharedOp>(op)) {
      state->fenceGeneric(fence.getBCluster() ? ProxyFenceScope::Cluster
                                              : ProxyFenceScope::CTA);
      return;
    }

    if (auto call = dyn_cast<CallOpInterface>(op)) {
      auto callee =
          dyn_cast_or_null<FunctionOpInterface>(call.resolveCallable());
      if (!callee)
        return;
      ProxyBlockInfo effects = funcMap->lookup(callee);
      translateCalleeState(regions, effects, call, function, callee);
      applyEffects(op, effects, state, builder, /*fromCall=*/true);
      return;
    }

    ProxyBlockInfo effects;
    for (const MemoryAccess &access : getMemoryAccesses(op)) {
      if (!access.isShared() || access.isShared(gpu::SharedKind::Barrier))
        continue;

      bool async = access.isShared(gpu::SharedKind::Async);
      uint8_t accessScopes = async ? scopeMask(getProxyFenceScope(op)) : scopes;
      for (BufferAccess region : getBufferAccesses(regions, access.value)) {
        if (access.isRead)
          (async ? effects.asyncReads : effects.genericReads)[region] |=
              accessScopes;
        if (access.isWrite)
          (async ? effects.asyncWrites
                 : effects.genericWrites)[std::move(region)] |= accessScopes;
      }
    }

    if (std::optional<BufferRegionView> scratch =
            getScratchAccess(regions, function, op)) {
      effects.genericReads[BufferAccess(*scratch)] |= scopes;
      effects.genericWrites[BufferAccess(std::move(*scratch))] |= scopes;
    }
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
