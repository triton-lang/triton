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

#include <optional>
#include <set>

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONGPUBUFFERREGIONPROXYFENCEINSERTION
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

// Track generic-proxy frontiers with exact physical regions. Async-proxy
// accesses are retained only while they can still conflict with generic
// accesses preceding the current function; this lets callers place a fence
// before a call without inventing incoming effects in the callee.
namespace {

enum class ProxyFenceScope { CTA, Cluster };

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
  using AccessSet = std::set<BufferAccess>;

  // Generic accesses since the last proxy fence.
  AccessSet genericReads;
  AccessSet genericWrites;
  // Async accesses before the first proxy fence reachable from function entry.
  AccessSet asyncReads;
  AccessSet asyncWrites;

  // True until a fence on every path has covered generic accesses preceding
  // the function.
  bool entryGenericUnfenced = false;

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

  void fenceGeneric() {
    genericReads.clear();
    genericWrites.clear();
    entryGenericUnfenced = false;
  }

  bool hasAsyncAccesses() const {
    return !asyncReads.empty() || !asyncWrites.empty();
  }

  bool needsFenceBefore(const ProxyBlockInfo &async) const {
    return intersects(genericWrites, async.asyncReads) ||
           intersects(genericReads, async.asyncWrites) ||
           intersects(genericWrites, async.asyncWrites);
  }

  void joinGeneric(const ProxyBlockInfo &other) {
    join(genericReads, other.genericReads);
    join(genericWrites, other.genericWrites);
  }

  void joinAsync(const ProxyBlockInfo &other) {
    join(asyncReads, other.asyncReads);
    join(asyncWrites, other.asyncWrites);
  }

private:
  static void join(AccessSet &into, const AccessSet &from) {
    into.insert(from.begin(), from.end());
  }

  static bool mayAlias(const BufferAccess &lhs, const BufferAccess &rhs) {
    return !lhs || !rhs || lhs->intersects(*rhs);
  }

  static bool intersects(const AccessSet &lhs, const AccessSet &rhs) {
    return llvm::any_of(lhs, [&](const BufferAccess &left) {
      return llvm::any_of(rhs, [&](const BufferAccess &right) {
        return mayAlias(left, right);
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
    if (!cvtNeedsSharedMemory(srcTy, dstTy))
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
                       ProxyBlockInfo::AccessSet &accesses,
                       FunctionOpInterface caller, FunctionOpInterface callee,
                       uint32_t offset) {
  ProxyBlockInfo::AccessSet translated;
  uint32_t calleeFrame = analysis.getOperationId(callee.getOperation());
  uint32_t callerFrame = analysis.getOperationId(caller.getOperation());
  for (const BufferAccess &original : accesses) {
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
    translated.insert(std::move(access));
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

template <ProxyFenceScope scope>
class ProxyFenceFunctionAnalysis
    : public PostOrderFunctionAnalysis<ProxyBlockInfo> {
  using Base = PostOrderFunctionAnalysis<ProxyBlockInfo>;
  using FuncMapT = Base::FuncMapT;

public:
  ProxyFenceFunctionAnalysis(FunctionOpInterface function,
                             BufferRegionAnalysis &regions)
      : function(function), regions(regions) {}

private:
  ProxyBlockInfo getEntryState() const override {
    ProxyBlockInfo state;
    state.entryGenericUnfenced = true;
    return state;
  }

  void insertFence(Operation *op, OpBuilder *builder) {
    builder->setInsertionPoint(op);
    FenceAsyncSharedOp::create(*builder, op->getLoc(),
                               scope == ProxyFenceScope::Cluster);
  }

  void applyEffects(Operation *op, ProxyBlockInfo &effects,
                    ProxyBlockInfo *state, OpBuilder *builder) {
    if (effects.hasAsyncAccesses()) {
      if (state->needsFenceBefore(effects)) {
        insertFence(op, builder);
        state->fenceGeneric();
      } else if (state->entryGenericUnfenced) {
        state->joinAsync(effects);
      }
    }
    state->joinGeneric(effects);
  }

  void update(Operation *op, ProxyBlockInfo *state, FuncMapT *funcMap,
              OpBuilder *builder) override {
    if (auto fence = dyn_cast<FenceAsyncSharedOp>(op)) {
      if (scope == ProxyFenceScope::CTA || fence.getBCluster())
        state->fenceGeneric();
      return;
    }

    if (auto call = dyn_cast<CallOpInterface>(op)) {
      auto callee =
          dyn_cast_or_null<FunctionOpInterface>(call.resolveCallable());
      if (!callee)
        return;
      ProxyBlockInfo effects = funcMap->lookup(callee);
      translateCalleeState(regions, effects, call, function, callee);
      bool insertedFence =
          effects.hasAsyncAccesses() && state->needsFenceBefore(effects);
      if (insertedFence) {
        insertFence(op, builder);
        state->fenceGeneric();
      } else if (state->entryGenericUnfenced) {
        state->joinAsync(effects);
      }
      if (!effects.entryGenericUnfenced)
        state->fenceGeneric();
      state->joinGeneric(effects);
      return;
    }

    ProxyBlockInfo effects;
    bool matchingProxy = hasSharedAccess(op, gpu::SharedKind::Async) &&
                         getProxyFenceScope(op) == scope;
    for (const MemoryAccess &access : getMemoryAccesses(op)) {
      if (!access.isShared() || access.isShared(gpu::SharedKind::Barrier))
        continue;

      bool async = access.isShared(gpu::SharedKind::Async);
      if (async && !matchingProxy)
        continue;

      for (BufferAccess region : getBufferAccesses(regions, access.value)) {
        if (access.isRead)
          (async ? effects.asyncReads : effects.genericReads).insert(region);
        if (access.isWrite)
          (async ? effects.asyncWrites : effects.genericWrites)
              .insert(std::move(region));
      }
    }

    if (std::optional<BufferRegionView> scratch =
            getScratchAccess(regions, function, op)) {
      effects.genericReads.insert(BufferAccess(*scratch));
      effects.genericWrites.insert(BufferAccess(std::move(*scratch)));
    }
    applyEffects(op, effects, state, builder);
  }

  FunctionOpInterface function;
  BufferRegionAnalysis &regions;
};

template <ProxyFenceScope scope>
class ModuleProxyFenceAnalysis : public CallGraph<ProxyBlockInfo> {
public:
  ModuleProxyFenceAnalysis(ModuleOp module, BufferRegionAnalysis &regions)
      : CallGraph<ProxyBlockInfo>(module), regions(regions) {}

  void run() {
    this->funcMap.clear();
    this->template walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        [](CallOpInterface, FunctionOpInterface) {},
        [&](FunctionOpInterface function) {
          if (!this->funcMap.try_emplace(function).second)
            return;
          ProxyFenceFunctionAnalysis<scope>(function, regions)
              .run(function, this->funcMap);
        });
  }

private:
  BufferRegionAnalysis &regions;
};

} // namespace

struct BufferRegionProxyFenceInsertionPass
    : public impl::TritonGPUBufferRegionProxyFenceInsertionBase<
          BufferRegionProxyFenceInsertionPass> {
  using impl::TritonGPUBufferRegionProxyFenceInsertionBase<
      BufferRegionProxyFenceInsertionPass>::
      TritonGPUBufferRegionProxyFenceInsertionBase;

  void runOnOperation() override {
    if (computeCapability < 90)
      return;

    ModuleOp module = getOperation();
    bool hasProxyOp = false;
    bool hasClusterProxyOp =
        module
            .walk([&](Operation *op) {
              if (!hasSharedAccess(op, gpu::SharedKind::Async))
                return WalkResult::advance();
              hasProxyOp = true;
              return getProxyFenceScope(op) == ProxyFenceScope::Cluster
                         ? WalkResult::interrupt()
                         : WalkResult::advance();
            })
            .wasInterrupted();
    if (!hasProxyOp)
      return;

    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    auto *regions = solver->load<BufferRegionAnalysis>();
    if (failed(solver->initializeAndRun(module)))
      return signalPassFailure();

    if (hasClusterProxyOp)
      ModuleProxyFenceAnalysis<ProxyFenceScope::Cluster>(module, *regions)
          .run();
    ModuleProxyFenceAnalysis<ProxyFenceScope::CTA>(module, *regions).run();
  }
};

} // namespace mlir::triton::nvidia_gpu
