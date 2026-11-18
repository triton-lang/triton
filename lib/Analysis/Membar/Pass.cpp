#include "triton/Analysis/Membar/Pass.h"
#include "triton/Analysis/CallGraph.h"
#include "triton/Analysis/Membar/AsyncFenceModel.h"
#include "triton/Analysis/Membar/CFGBuilder.h"
#include "triton/Analysis/Membar/ClusterBarrierModel.h"
#include "triton/Analysis/Membar/Dependencies.h"
#include "triton/Analysis/Membar/Solver.h"
#include "triton/Analysis/Membar/TmemModel.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#include <map>

#define DEBUG_TYPE "membar-placement"

namespace mlir::triton::membar {
namespace {

template <typename Facts>
void emit(const MembarCFG<Facts> &cfg, ArrayRef<Action> actions,
          const llvm::BitVector &selected) {
  for (unsigned id : selected.set_bits()) {
    const auto &node = cfg.nodes[id / actions.size()];
    Operation *before = node.op;
    Action action = actions[id % actions.size()];
    OpBuilder builder(before);
    auto loc = before->getLoc();
    switch (action) {
    case Action::ClusterBarrier:
    case Action::ClusterInit: {
      // The specialized release must run on the thread that issued the init.
      bool init = action == Action::ClusterInit && node.body == 0;
      if (init)
        nvidia_gpu::FenceMBarrierInitReleaseClusterOp::create(builder, loc);
      // An existing rendezvous at a selected site is a relaxed barrier.
      if (auto barrier = dyn_cast<nvidia_gpu::ClusterBarrierOp>(before))
        barrier.setRelaxed(init);
      else
        nvidia_gpu::ClusterBarrierOp::create(builder, loc, /*relaxed=*/init);
      break;
    }
    case Action::ProxyCTA:
    case Action::ProxyCluster:
      nvidia_gpu::FenceAsyncSharedOp::create(builder, loc,
                                             action == Action::ProxyCluster);
      break;
    case Action::WaitLoad:
    case Action::WaitStore:
      nvidia_gpu::TMEMWaitOp::create(builder, loc,
                                     action == Action::WaitLoad
                                         ? nvidia_gpu::TMEMWaitKind::LOAD
                                         : nvidia_gpu::TMEMWaitKind::STORE);
      break;
    case Action::RegionBarrier:
    case Action::WarpBarrier:
      gpu::BarrierOp::create(builder, loc, gpu::AddrSpace::Local,
                             action == Action::WarpBarrier
                                 ? gpu::BarrierScope::Warp
                                 : gpu::BarrierScope::CTA);
      break;
    }
  }
}

template <typename Model, typename Discover>
LogicalResult placeRegions(FunctionOpInterface function,
                           MembarCFGBuilder<Model> &builder,
                           ArrayRef<Action> actions, Discover discover,
                           bool expandWarpRoles = false) {
  SmallVector<Region *> bodies{&function.getFunctionBody()};
  if (!expandWarpRoles)
    function.walk([&](gpu::WarpSpecializeOp ws) {
      bodies.push_back(&ws.getDefaultRegion());
      llvm::append_range(bodies, ws.getPartitionRegions());
    });
  for (Region *body : bodies) {
    auto cfg = builder.build(*body, expandWarpRoles);
    auto demands = discover(cfg);
    auto cut = solveMembarCut(cfg, demands, actions);
    if (!cut)
      return function.emitError("synchronization demand has no legal cut");
    LLVM_DEBUG({
      llvm::dbgs() << function.getName() << ": " << cfg.nodes.size()
                   << " CFG points, " << demands.size() << " demands, "
                   << cut->sites.count() << " sites, cost " << cut->cost
                   << ", lower bound " << cut->lowerBound << "\n";
    });
    emit(cfg, actions, cut->sites);
  }
  return success();
}

template <typename T, typename Model>
LogicalResult placeAcrossRoles(const CallGraph<T> &calls, Model &model,
                               ArrayRef<Action> actions) {
  MembarCFGBuilder builder(model, calls.getModuleOp());
  for (auto function : calls.getRoots()) {
    auto discover = [&](auto &cfg) { return model.discoverDemands(cfg); };
    if (failed(placeRegions(function, builder, actions, discover,
                            /*expandWarpRoles=*/true)))
      return failure();
  }
  return success();
}

} // namespace

LogicalResult placeProxyFences(ModuleOp module, BufferRegionAnalysis &regions) {
  AsyncFenceModel model(regions, gpu::TritonGPUDialect::getNumCTAs(module));
  return placeAcrossRoles(CallGraph<int>(module), model,
                          {Action::ProxyCluster, Action::ProxyCTA});
}

LogicalResult placeClusterBarriers(ModuleAllocation &allocation,
                                   BufferRegionAnalysis &regions) {
  ModuleOp module = allocation.getModuleOp();
  if (gpu::TritonGPUDialect::getNumCTAs(module) == 1)
    return success();

  // freeTMAlloc relies on a whole-cluster rendezvous at each kernel return
  // before tcgen05.dealloc.cta_group::2. Give the solver credit for it.
  if (nvidia_gpu::getModuleTwoCTAs(module))
    module.walk([](RegionBranchTerminatorOpInterface ret) {
      auto fn = dyn_cast<FunctionOpInterface>(ret->getParentOp());
      if (fn && isKernel(fn)) {
        OpBuilder builder(ret);
        nvidia_gpu::ClusterBarrierOp::create(builder, ret.getLoc());
      }
    });
  ClusterBarrierModel model(regions);
  return placeAcrossRoles(allocation, model,
                          {Action::ClusterBarrier, Action::ClusterInit});
}

LogicalResult placeBarriers(ModuleAllocation &moduleAllocation,
                            BufferRegionAnalysis &regions,
                            MembarFilterFn sharedFilter, PlacementMode mode,
                            CompletionScopeFn completionScope) {
  ModuleOp module = moduleAllocation.getModuleOp();
  std::optional<TMemModel> tmem;
  if (mode != PlacementMode::Shared)
    tmem.emplace(module, regions);
  auto *tmemModel = tmem ? &*tmem : nullptr;

  auto sharedMode = mode == PlacementMode::TensorMemory
                        ? SMemModel::AccessMode::SynchronizationOnly
                        : SMemModel::AccessMode::AllSharedAccesses;
  // Allocation slices borrow index expressions from the issuing function's
  // model.
  std::map<Operation *, SMemModel> models;
  for (auto function : module.getOps<FunctionOpInterface>())
    models.try_emplace(function, *moduleAllocation.getFuncData(function),
                       regions, sharedMode, completionScope,
                       /*followCalls=*/true);

  MembarModel model(models, tmemModel);
  MembarCFGBuilder builder(model, module);
  for (auto function : moduleAllocation.getRoots()) {
    // Callee footprints are in kernel coordinates.
    Allocation &allocation = *moduleAllocation.getFuncData(function);
    auto discover = [&](const auto &cfg) {
      return discoverDemands(cfg, allocation, sharedFilter, tmemModel);
    };
    if (failed(placeRegions(function, builder, ACTIONS, discover)))
      return failure();
  }
  return success();
}

} // namespace mlir::triton::membar
