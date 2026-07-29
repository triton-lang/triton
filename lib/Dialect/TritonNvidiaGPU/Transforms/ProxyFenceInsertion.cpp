#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonInstrument/IR/ConSanConstants.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/LayoutUtils.h"

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
    if (value == mma.getA() || value == mma.getB())
      return true;
    if (auto scaled = dyn_cast<triton::nvidia_gpu::TCGen5MMAScaledOp>(op))
      return value == scaled.getAScale() || value == scaled.getBScale();
    return false;
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

bool isBarrierDescriptor(Value value) {
  return llvm::any_of(value.getUsers(), [&](Operation *user) {
    auto barrier = dyn_cast<triton::gpu::MBarrierOpInterface>(user);
    return barrier && llvm::is_contained(barrier.getBarriers(), value);
  });
}

bool isSharedMemoryDescriptor(Value value) {
  auto type = dyn_cast<triton::gpu::MemDescType>(value.getType());
  return type && isa<triton::gpu::SharedMemorySpaceAttr>(type.getMemorySpace());
}

struct ScratchInfo {
  unsigned size;
  bool crossCTA = false;
};

ScratchInfo getScratchInfo(Operation *op) {
  if (auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
    RankedTensorType srcTy = cvt.getSrc().getType();
    RankedTensorType dstTy = cvt.getType();
    if (!cvtNeedsSharedMemory(srcTy, dstTy))
      return {0};

    LinearLayout src = triton::gpu::toLinearLayout(srcTy);
    LinearLayout dst = triton::gpu::toLinearLayout(dstTy);
    src = triton::actionRemoveBroadcastedRegs(src).apply(src);
    dst = triton::actionRemoveBroadcastedRegs(dst).apply(dst);

    MLIRContext *ctx = op->getContext();
    StringAttr block = StringAttr::get(ctx, "block");
    bool crossCTA = !dst.invertAndCompose(src).isTrivialOver({block});
    unsigned bitwidth = getBitwidth(srcTy);
    // Match NVIDIA allocation's instruction-aware swizzling without depending
    // on the LLVM-conversion library. This pass only runs on sm90 and newer.
    SmallVector<triton::gpu::LocalMemOpTile> srcTiles{{{}, {0, 1, 2}}};
    SmallVector<triton::gpu::LocalMemOpTile> dstTiles = srcTiles;
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
        triton::gpu::optimalSwizzling(src, dst, srcTiles, dstTiles, bitwidth);
    unsigned reps = scratch.getInDimSize(StringAttr::get(ctx, "reps"));
    unsigned numCTAs =
        product(triton::gpu::getCTASplitNum(srcTy.getEncoding()));
    return {scratch.getTotalOutDimSize() / (reps * numCTAs) * bitwidth / 8,
            crossCTA};
  }

  unsigned size = defaultAllocationAnalysisScratchSizeFn(op);
  if (isa<triton::gpu::WarpSpecializeOp>(op))
    if (auto extra = op->getAttrOfType<IntegerAttr>(
            triton::instrument::kConSanExtraCaptureBytesAttr))
      size += extra.getInt();
  return {size};
}

template <typename AliasAnalysisT> struct ProxyFenceAliasTraits;

template <> struct ProxyFenceAliasTraits<ModuleAllocation> {
  using BlockInfoT = BlockInfo;
  using AccessT = AllocationSlice;

  static SmallVector<AccessT> getRegions(ModuleAllocation &analysis,
                                         FunctionOpInterface function,
                                         Value value) {
    Allocation *allocation = analysis.getFuncData(function);
    SmallVector<AccessT> regions;
    for (Allocation::BufferId id :
         allocation->getAllBufferIdsWithAliases(value))
      if (id != Allocation::InvalidBufferId)
        regions.emplace_back(value, allocation->getAllocatedInterval(id), id);
    if (regions.empty())
      regions.emplace_back(Interval<size_t>(
          0, static_cast<size_t>(std::numeric_limits<uint32_t>::max())));
    return regions;
  }

  static SmallVector<AccessT> getScratchRegions(ModuleAllocation &analysis,
                                                FunctionOpInterface function,
                                                Operation *op) {
    Allocation *allocation = analysis.getFuncData(function);
    Allocation::BufferId id = allocation->getBufferId(op);
    if (id == Allocation::InvalidBufferId)
      return {};
    return {AccessT(allocation->getAllocatedInterval(id))};
  }

  static void translate(ModuleAllocation &, BlockInfoT &info,
                        CallOpInterface call, FunctionOpInterface,
                        FunctionOpInterface) {
    if (auto offset = call->getAttrOfType<IntegerAttr>("allocation.offset"))
      info.join(translateBlockInfoToCallsite(info, offset.getInt()));
  }
};

template <> struct ProxyFenceAliasTraits<triton::BufferRegionAnalysis> {
  using BlockInfoT = BufferRegionBlockInfo;
  using AccessT = std::optional<triton::BufferRegionView>;

  static SmallVector<AccessT> getRegions(triton::BufferRegionAnalysis &analysis,
                                         FunctionOpInterface, Value value) {
    const triton::RegionInfo &info = analysis.getRegionInfo(value);
    if (info.isUnknown())
      return {std::nullopt};
    SmallVector<AccessT> regions;
    for (const triton::BufferRegionView &view : info.views)
      regions.push_back(view);
    return regions;
  }

  static SmallVector<AccessT>
  getScratchRegions(triton::BufferRegionAnalysis &analysis,
                    FunctionOpInterface function, Operation *op) {
    auto offset = op->getAttrOfType<IntegerAttr>("allocation.offset");
    if (!offset)
      return {};
    ScratchInfo info = getScratchInfo(op);
    if (!info.size)
      return {};
    uint32_t base = offset.getInt();
    triton::AddressSet addresses =
        triton::AddressSet::fromRange(base, info.size);
    SmallVector<triton::BufferRegion::CTAAddresses, 2> ctaAddresses;
    unsigned numCTAs = info.crossCTA ? triton::gpu::lookupNumCTAs(op) : 1;
    for (unsigned cta = 0; cta < numCTAs; ++cta)
      ctaAddresses.emplace_back(cta, addresses);
    return {triton::BufferRegionView{
        {base, info.size, std::move(ctaAddresses)},
        base,
        /*affineOffset=*/0,
        /*partitionBases=*/{},
        /*affinePartitionOffset=*/0,
        /*affineCTAOffset=*/0,
        analysis.getOperationId(function.getOperation())}};
  }

  static void translate(triton::BufferRegionAnalysis &analysis,
                        BlockInfoT &info, CallOpInterface call,
                        FunctionOpInterface caller,
                        FunctionOpInterface callee) {
    auto callOffset = call->getAttrOfType<IntegerAttr>("allocation.offset");
    uint32_t offset = callOffset ? callOffset.getInt() : 0;
    auto translateAccesses = [&](BlockInfoT::SliceMapT &accesses) {
      BlockInfoT::SliceMapT translated;
      for (const auto &[original, operations] : accesses) {
        AccessT view = original;
        if (view && view->allocationFrame ==
                        analysis.getOperationId(callee.getOperation())) {
          view->region.baseOffset += offset;
          for (auto &[cta, addresses] : view->region.ctaAddresses)
            addresses = addresses.translated(offset);
          view->storageBase += offset;
          for (uint32_t &base : view->partitionBases)
            base += offset;
          view->allocationFrame =
              analysis.getOperationId(caller.getOperation());
        }
        auto &destination = translated[std::move(view)];
        destination.insert(operations.begin(), operations.end());
      }
      accesses = std::move(translated);
    };
    translateAccesses(info.syncReadSlices);
    translateAccesses(info.syncWriteSlices);
  }
};

template <ProxyFenceScope scope, typename AliasAnalysisT>
class ProxyFenceAnalysis
    : public MembarOrFenceAnalysisBase<
          AliasAnalysisT,
          typename ProxyFenceAliasTraits<AliasAnalysisT>::BlockInfoT> {
  using Traits = ProxyFenceAliasTraits<AliasAnalysisT>;
  using BlockInfoT = typename Traits::BlockInfoT;
  using Base = MembarOrFenceAnalysisBase<AliasAnalysisT, BlockInfoT>;
  using FuncBlockInfoMapT = typename Base::FuncBlockInfoMapT;

public:
  ProxyFenceAnalysis(FunctionOpInterface function, AliasAnalysisT *analysis,
                     bool assumeArgumentAccesses)
      : Base(function, analysis, filterFn),
        assumeArgumentAccesses(assumeArgumentAccesses) {}

private:
  BlockInfoT getEntryBlockInfo() const override {
    BlockInfoT info;
    // Kernel entrypoints cannot receive memory descriptors. Standalone test
    // functions may, but they have no caller whose accesses need modeling.
    if (!assumeArgumentAccesses)
      return info;
    FunctionOpInterface function = this->function;
    for (Value argument : function.getArguments()) {
      if (!isSharedMemoryDescriptor(argument) || isBarrierDescriptor(argument))
        continue;
      for (typename Traits::AccessT region :
           Traits::getRegions(*this->allocation, function, argument)) {
        info.syncReadSlices[region].insert(function.getOperation());
        info.syncWriteSlices[std::move(region)].insert(function.getOperation());
      }
    }
    return info;
  }

  void update(Operation *op, BlockInfoT *blockInfo,
              FuncBlockInfoMapT *funcBlockInfoMap,
              OpBuilder *builder) override {
    if (auto fence = dyn_cast<triton::nvidia_gpu::FenceAsyncSharedOp>(op)) {
      if (scope == ProxyFenceScope::CTA || fence.getBCluster())
        blockInfo->sync();
      return;
    }

    BlockInfoT generic;
    BlockInfoT proxy;
    bool isProxy = (isAsyncProxyWrite(op) || isAsyncProxyRead(op)) &&
                   getProxyFenceScope(op) == scope;

    if (auto call = dyn_cast<CallOpInterface>(op)) {
      if (auto callee =
              dyn_cast_or_null<FunctionOpInterface>(call.resolveCallable())) {
        generic = funcBlockInfoMap->lookup(callee);
        Traits::translate(*this->allocation, generic, call, this->function,
                          callee);
      }
    } else {
      for (const triton::BufferRegionAnalysis::MemoryAccess &access :
           triton::BufferRegionAnalysis::getMemoryAccesses(op)) {
        if (!isSharedMemoryDescriptor(access.value) ||
            isBarrierDescriptor(access.value))
          continue;

        bool proxyWrite =
            isAsyncProxyWrite(op) && access.value == getSmemDest(op);
        bool proxyRead =
            isAsyncProxyRead(op) && isAsyncProxyReadSource(op, access.value);
        if ((proxyWrite || proxyRead) && !isProxy)
          continue;

        for (typename Traits::AccessT region : Traits::getRegions(
                 *this->allocation, this->function, access.value)) {
          if (proxyWrite)
            proxy.syncWriteSlices[region].insert(op);
          else if (proxyRead)
            proxy.syncReadSlices[region].insert(op);
          else {
            if (access.isRead)
              generic.syncReadSlices[region].insert(op);
            if (access.isWrite)
              generic.syncWriteSlices[std::move(region)].insert(op);
          }
        }
      }

      for (typename Traits::AccessT scratch :
           Traits::getScratchRegions(*this->allocation, this->function, op)) {
        generic.syncReadSlices[scratch].insert(op);
        generic.syncWriteSlices[std::move(scratch)].insert(op);
      }
    }

    if (isProxy && proxy.isIntersected(*blockInfo, this->filter,
                                       /*allocation=*/nullptr)) {
      builder->setInsertionPoint(op);
      triton::nvidia_gpu::FenceAsyncSharedOp::create(
          *builder, op->getLoc(), scope == ProxyFenceScope::Cluster);
      blockInfo->sync();
    }
    blockInfo->join(generic);
  }

  bool assumeArgumentAccesses;
};

template <typename AliasAnalysisT>
class ProxyFenceProvider
    : public triton::CallGraph<
          typename ProxyFenceAliasTraits<AliasAnalysisT>::BlockInfoT> {
  using Traits = ProxyFenceAliasTraits<AliasAnalysisT>;
  using BlockInfoT = typename Traits::BlockInfoT;

public:
  ProxyFenceProvider(ModuleOp module, AliasAnalysisT &analysis)
      : triton::CallGraph<BlockInfoT>(module), analysis(analysis) {}

  template <ProxyFenceScope scope> void run() {
    this->funcMap.clear();
    this->template walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        [](CallOpInterface, FunctionOpInterface) {},
        [&](FunctionOpInterface function) {
          auto [it, inserted] = this->funcMap.try_emplace(function);
          if (!inserted)
            return;
          ProxyFenceAnalysis<scope, AliasAnalysisT>(function, &analysis,
                                                    !this->isRoot(function))
              .run(this->funcMap);
          auto removeAssumedAccesses =
              [&](typename BlockInfoT::SliceMapT &accesses) {
                for (auto region = accesses.begin();
                     region != accesses.end();) {
                  region->second.erase(function.getOperation());
                  if (region->second.empty())
                    region = accesses.erase(region);
                  else
                    ++region;
                }
              };
          removeAssumedAccesses(it->second.syncReadSlices);
          removeAssumedAccesses(it->second.syncWriteSlices);
        });
  }

private:
  AliasAnalysisT &analysis;
};
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
    // Keep independent frontiers for cluster- and CTA-scoped fences. Run the
    // cluster analysis first so the CTA analysis can observe any cluster
    // fences it inserts.
    bool hasProxyOp = false;
    bool hasClusterProxyOp =
        mod.walk([&](Operation *op) {
             if (!isAsyncProxyRead(op) && !isAsyncProxyWrite(op))
               return WalkResult::advance();
             hasProxyOp = true;
             return getProxyFenceScope(op) == ProxyFenceScope::Cluster
                        ? WalkResult::interrupt()
                        : WalkResult::advance();
           })
            .wasInterrupted();
    if (!hasProxyOp)
      return;
    if (!useBufferRegionAliasAnalysis) {
      ModuleAllocation allocation(mod);
      ProxyFenceProvider provider(mod, allocation);
      if (hasClusterProxyOp)
        provider.run<ProxyFenceScope::Cluster>();
      provider.run<ProxyFenceScope::CTA>();
      return;
    }

    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    auto *regions = solver->load<triton::BufferRegionAnalysis>();
    if (failed(solver->initializeAndRun(mod)))
      return signalPassFailure();
    ProxyFenceProvider provider(mod, *regions);
    if (hasClusterProxyOp)
      provider.run<ProxyFenceScope::Cluster>();
    provider.run<ProxyFenceScope::CTA>();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
