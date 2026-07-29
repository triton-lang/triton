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
  explicit ProxyFenceAnalysis(Allocation *allocation, MembarFilterFn filter)
      : MembarOrFenceAnalysis(allocation, filter) {}

private:
  /// Updates the BlockInfo operation based on the operation.
  virtual void update(Operation *operation, BlockInfo *blockInfo,
                      FuncBlockInfoMapT *funcBlockInfoMap,
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
                                       FuncBlockInfoMapT *funcBlockInfoMap,
                                       OpBuilder *builder) {
  if (auto fence = dyn_cast<triton::nvidia_gpu::FenceAsyncSharedOp>(op)) {
    // A cluster fence covers both frontiers, while a CTA fence only covers the
    // CTA frontier.
    if (scope == ProxyFenceScope::CTA || fence.getBCluster())
      blockInfo->sync();
    return;
  }
  BlockInfo curBlockInfo;
  BlockInfo proxyBlockInfo;
  bool isProxyOp = (isAsyncProxyWrite(op) || isAsyncProxyRead(op)) &&
                   getProxyFenceScope(op) == scope;

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
                if (isProxyOp)
                  proxyBlockInfo.syncWriteSlices[slice].insert(op);
              } else if (isAsyncProxyRead(op) &&
                         isAsyncProxyReadSource(op, value)) {
                // Safe fallback for async-proxy reads from shared memory when
                // the earlier FenceInsertionPass did not place a fence.
                if (isProxyOp)
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
  if (isProxyOp) {
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

bool hasRegionIntersection(const BlockInfo::RegionMapT &lhs,
                           const BlockInfo::RegionMapT &rhs) {
  for (const auto &[lhsRegion, lhsOps] : lhs)
    for (const auto &[rhsRegion, rhsOps] : rhs) {
      if (lhsRegion && rhsRegion && !lhsRegion->intersects(*rhsRegion))
        continue;
      for (Operation *lhsOp : lhsOps)
        for (Operation *rhsOp : rhsOps)
          if (!ignoreOpForProxyFence(rhsOp))
            return true;
    }
  return false;
}

bool hasRegionHazard(const BlockInfo &proxy, const BlockInfo &generic) {
  return hasRegionIntersection(proxy.syncWriteRegions,
                               generic.syncReadRegions) ||
         hasRegionIntersection(proxy.syncReadRegions,
                               generic.syncWriteRegions) ||
         hasRegionIntersection(proxy.syncWriteRegions,
                               generic.syncWriteRegions);
}

template <ProxyFenceScope scope>
class BufferRegionProxyFenceAnalysis : public MembarOrFenceAnalysis {
public:
  BufferRegionProxyFenceAnalysis(FunctionOpInterface function,
                                 triton::BufferRegionAnalysis &regions,
                                 bool assumeArgumentAccesses)
      : MembarOrFenceAnalysis(function), regions(regions),
        assumeArgumentAccesses(assumeArgumentAccesses) {}

private:
  BlockInfo getEntryBlockInfo() const override {
    BlockInfo info;
    info.reachesFunctionEntry = true;
    // Kernel entrypoints cannot receive memory descriptors. Standalone test
    // functions may, but they have no caller whose accesses need modeling.
    if (!assumeArgumentAccesses)
      return info;
    FunctionOpInterface currentFunction = function;
    for (Value argument : currentFunction.getArguments()) {
      if (!isSharedMemoryDescriptor(argument) || isBarrierDescriptor(argument))
        continue;
      const triton::RegionInfo &argumentRegions =
          regions.getRegionInfo(argument);
      auto add = [&](std::optional<triton::BufferRegionView> region) {
        info.syncReadRegions[region].insert(currentFunction.getOperation());
        info.syncWriteRegions[std::move(region)].insert(
            currentFunction.getOperation());
      };
      if (argumentRegions.isUnknown())
        add(std::nullopt);
      else
        for (const triton::BufferRegionView &view : argumentRegions.views)
          add(view);
    }
    return info;
  }

  void update(Operation *op, BlockInfo *blockInfo,
              FuncBlockInfoMapT *funcBlockInfoMap,
              OpBuilder *builder) override {
    if (auto fence = dyn_cast<triton::nvidia_gpu::FenceAsyncSharedOp>(op)) {
      if (scope == ProxyFenceScope::CTA || fence.getBCluster())
        blockInfo->sync();
      return;
    }

    BlockInfo generic;
    BlockInfo proxy;
    bool isProxy = (isAsyncProxyWrite(op) || isAsyncProxyRead(op)) &&
                   getProxyFenceScope(op) == scope;

    if (auto call = dyn_cast<CallOpInterface>(op)) {
      if (auto callee =
              dyn_cast_or_null<FunctionOpInterface>(call.resolveCallable())) {
        generic = funcBlockInfoMap->lookup(callee);
        auto callOffset = call->getAttrOfType<IntegerAttr>("allocation.offset");
        uint32_t offset = callOffset ? callOffset.getInt() : 0;
        auto translateCalleeRegions = [&](BlockInfo::RegionMapT &accesses) {
          BlockInfo::RegionMapT translated;
          for (const auto &[original, operations] : accesses) {
            std::optional<triton::BufferRegionView> view = original;
            if (view && view->allocationFrame == callee.getOperation()) {
              view->region.baseOffset += offset;
              for (auto &[cta, addresses] : view->region.ctaAddresses)
                addresses = addresses.translated(offset);
              view->storageBase += offset;
              for (uint32_t &base : view->partitionBases)
                base += offset;
              view->allocationFrame = function.getOperation();
            }
            auto &destination = translated[std::move(view)];
            destination.insert(operations.begin(), operations.end());
          }
          accesses = std::move(translated);
        };
        translateCalleeRegions(generic.syncReadRegions);
        translateCalleeRegions(generic.syncWriteRegions);
        if (!generic.reachesFunctionEntry)
          blockInfo->sync();
        else
          generic.reachesFunctionEntry = blockInfo->reachesFunctionEntry;
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
        // Scaled MMA can consume its scale descriptors directly from shared
        // memory through the same async proxy as its A/B operands.
        if (auto scaled = dyn_cast<triton::nvidia_gpu::TCGen5MMAScaledOp>(op))
          proxyRead |= access.value == scaled.getAScale() ||
                       access.value == scaled.getBScale();
        if ((proxyWrite || proxyRead) && !isProxy)
          continue;

        const triton::RegionInfo &info = regions.getRegionInfo(access.value);
        auto add = [&](BlockInfo::RegionMapT &regions) {
          if (info.isUnknown()) {
            regions[std::nullopt].insert(op);
            return;
          }
          for (const triton::BufferRegionView &view : info.views)
            regions[view].insert(op);
        };

        if (proxyWrite)
          add(proxy.syncWriteRegions);
        else if (proxyRead)
          add(proxy.syncReadRegions);
        else {
          if (access.isRead)
            add(generic.syncReadRegions);
          if (access.isWrite)
            add(generic.syncWriteRegions);
        }
      }

      if (auto offset = op->getAttrOfType<IntegerAttr>("allocation.offset")) {
        ScratchInfo scratchInfo = getScratchInfo(op);
        if (scratchInfo.size) {
          uint32_t base = offset.getInt();
          triton::AddressSet addresses =
              triton::AddressSet::fromRange(base, scratchInfo.size);
          SmallVector<triton::BufferRegion::CTAAddresses, 2> ctaAddresses;
          unsigned numCTAs =
              scratchInfo.crossCTA ? triton::gpu::lookupNumCTAs(op) : 1;
          for (unsigned cta = 0; cta < numCTAs; ++cta)
            ctaAddresses.emplace_back(cta, addresses);
          triton::BufferRegionView scratch{
              {base, scratchInfo.size, std::move(ctaAddresses)},
              base,
              /*affineOffset=*/0,
              /*partitionBases=*/{},
              /*affinePartitionOffset=*/0,
              /*affineCTAOffset=*/0,
              function.getOperation()};
          // Lowered scratch operations both write and read their allocation.
          generic.syncReadRegions[scratch].insert(op);
          generic.syncWriteRegions[std::move(scratch)].insert(op);
        }
      }
    }

    if (isProxy && hasRegionHazard(proxy, *blockInfo)) {
      builder->setInsertionPoint(op);
      triton::nvidia_gpu::FenceAsyncSharedOp::create(
          *builder, op->getLoc(), scope == ProxyFenceScope::Cluster);
      blockInfo->sync();
    }
    blockInfo->join(generic);
  }

  triton::BufferRegionAnalysis &regions;
  bool assumeArgumentAccesses;
};

class BufferRegionProxyFenceProvider : public triton::CallGraph<BlockInfo> {
public:
  BufferRegionProxyFenceProvider(ModuleOp module,
                                 triton::BufferRegionAnalysis &regions)
      : triton::CallGraph<BlockInfo>(module), regions(regions) {}

  template <ProxyFenceScope scope> void run() {
    funcMap.clear();
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        [](CallOpInterface, FunctionOpInterface) {},
        [&](FunctionOpInterface function) {
          auto [it, inserted] = funcMap.try_emplace(function);
          if (!inserted)
            return;
          BufferRegionProxyFenceAnalysis<scope>(function, regions,
                                                !isRoot(function))
              .run(funcMap);
          auto removeAssumedAccesses = [&](BlockInfo::RegionMapT &accesses) {
            for (auto region = accesses.begin(); region != accesses.end();) {
              region->second.erase(function.getOperation());
              if (region->second.empty())
                region = accesses.erase(region);
              else
                ++region;
            }
          };
          removeAssumedAccesses(it->second.syncReadRegions);
          removeAssumedAccesses(it->second.syncWriteRegions);
        });
  }

private:
  triton::BufferRegionAnalysis &regions;
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
      if (hasClusterProxyOp) {
        ModuleMembarOrFenceAnalysis<
            ProxyFenceAnalysis<ProxyFenceScope::Cluster>>
            clusterAnalysis(&allocation, filterFn);
        clusterAnalysis.run();
      }
      ModuleMembarOrFenceAnalysis<ProxyFenceAnalysis<ProxyFenceScope::CTA>>
          ctaAnalysis(&allocation, filterFn);
      ctaAnalysis.run();
      return;
    }

    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    auto *regions = solver->load<triton::BufferRegionAnalysis>();
    if (failed(solver->initializeAndRun(mod)))
      return signalPassFailure();
    BufferRegionProxyFenceProvider provider(mod, *regions);
    if (hasClusterProxyOp)
      provider.run<ProxyFenceScope::Cluster>();
    provider.run<ProxyFenceScope::CTA>();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
