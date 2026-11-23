#include "triton/Analysis/Membar/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
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

struct Worker {
  SmallVector<nvidia_gpu::ArriveBarrierOp, 2> releases;
  // Reducing reads leave a full tile and its reduction live at the release.
  DenseSet<Operation *> reducing;
  // A conditional payload can compete with the issuer predicate for
  // registers. A blocking acquire limits how early it can be published.
  bool conditionalStore = false;
  bool mayBlock = false;
};

// Look through branches to the next effect on each path. The graph omits
// pure computations, including layout conversions that need no scratch.
static SmallVector<const Node<OpModel> *, 2>
nextEffects(const MembarCFG<OpModel> &cfg, const Node<OpModel> &from) {
  SmallVector<const Node<OpModel> *, 2> effects;
  walkEdges(cfg, from, 0u,
            [&](unsigned i, unsigned, bool) -> std::optional<unsigned> {
              const auto &next = cfg.nodes[i];
              if (isa<BranchOpInterface>(next.op) && next.effects == OpModel{})
                return 0;
              effects.push_back(&next);
              return std::nullopt;
            });
  return effects;
}

// A per-warp release only needs each warp's TMEM wait. When shared
// storage independently needs a region barrier at the same gap, the
// warps can reach that barrier while their loads are still outstanding.
static Worker getIndependentReleases(const MembarCFG<OpModel> &cfg,
                                     ArrayRef<Demand> required,
                                     const llvm::BitVector &sites) {
  // A mandatory cross-warp handoff anywhere on the path can rely on
  // the original wait-before-barrier order.
  if (llvm::any_of(required, [](const Demand &d) {
        return d.first == SyncKind::WaitLoad && d.path == Path::WaitThenOrder;
      }))
    return {};

  Worker worker;
  for (const auto &node : cfg.nodes) {
    if (node.effects.arrivalMode != ArrivalMode::Either) {
      // Promotion adds the other warp leaders as issuers. Reject a fixed
      // consumer that would have to join them but already covers thread zero.
      unsigned joins = node.effects.shared.threads.joins;
      if ((joins & ThreadEffects::WarpLeaders) &&
          !(joins & ThreadEffects::Thread0))
        return {};
      continue;
    }
    // Each scalar handoff we keep must still collect all the warps at
    // its own gap, including any newly participating issuers.
    if (cast<gpu::MBarrierOpInterface>(node.op).isPerWarp())
      continue;
    auto arrive = dyn_cast<nvidia_gpu::ArriveBarrierOp>(node.op);
    if (!arrive || arrive.getFromCTA() || node.body)
      return {};
    unsigned actions = 0;
    for (auto [i, action] : llvm::enumerate(ACTIONS))
      if (sites.test(node.gap * ACTIONS.size() + i))
        actions |= bit(action);
    if (actions != (bit(SyncKind::WaitLoad) | bit(SyncKind::RegionBarrier)))
      return {};
    worker.releases.push_back(arrive);
  }
  if (worker.releases.empty())
    return {};

  // These are scheduling hints only. The checks above make either order
  // legal as long as an overlapped release is per-warp.
  for (const auto &node : cfg.nodes) {
    auto load = dyn_cast<nvidia_gpu::TMEMLoadOp>(node.op);
    if (!load || !load.getRedOp())
      continue;
    for (const auto *release : nextEffects(cfg, node)) {
      if (!llvm::is_contained(worker.releases, release->op))
        continue;
      worker.reducing.insert(release->op);
      for (const auto *next : nextEffects(cfg, *release)) {
        auto store = dyn_cast<gpu::LocalStoreOp>(next->op);
        if (!store) {
          worker.mayBlock = true;
          continue;
        }
        Value value = store.getSrc();
        while (auto reshape = value.getDefiningOp<ReshapeOp>())
          value = reshape.getSrc();
        worker.conditionalStore |=
            store->getBlock() != release->op->getBlock() ||
            value.getDefiningOp<arith::SelectOp>();
      }
    }
  }
  return worker;
}

static void schedulePairedReleases(ArrayRef<Worker> workers, bool twoCTAs) {
  if (workers.size() != 2 ||
      workers[0].releases.size() != workers[1].releases.size())
    return;
  bool conditional = workers[0].conditionalStore && workers[1].conditionalStore;
  bool overlap =
      !twoCTAs || (conditional && !workers[0].mayBlock && !workers[1].mayBlock);
  // Stagger the paired consumers. In a cluster, overlap the second
  // consumer only when it can prepare a conditional payload immediately.
  // Otherwise keep both waits first and distribute only the second release.
  for (auto [i, worker] : llvm::enumerate(workers)) {
    for (auto arrive : worker.releases) {
      if (!overlap && i == 0)
        continue;
      // Within a CTA, stagger reducing reads by keeping a single publisher
      // in the second worker. Both workers can overlap ordinary reads.
      if (!twoCTAs && i == 1 && worker.reducing.contains(arrive))
        continue;
      arrive.setPerWarp(true);
      if (!overlap || (twoCTAs && i == 0))
        continue;
      auto barrier = cast<gpu::BarrierOp>(arrive->getPrevNode());
      auto wait = cast<nvidia_gpu::TMEMWaitOp>(barrier->getPrevNode());
      barrier->moveBefore(wait);
    }
    // Shorten the lifetime of the issuer predicate in register-heavy workers.
    if ((!twoCTAs || conditional) && !worker.reducing.empty()) {
      Region *body = worker.releases.front()->getParentRegion();
      for (auto arrive : body->getOps<nvidia_gpu::ArriveBarrierOp>())
        if (arrive.getPerWarp() && !arrive.getFromCTA())
          arrive->setAttr(nvidia_gpu::AttrRematerializeLaneIdName,
                          UnitAttr::get(arrive.getContext()));
    }
  }
}

template <typename Facts>
FailureOr<llvm::BitVector>
placeRequired(FunctionOpInterface function, const MembarCFG<Facts> &cfg,
              ArrayRef<Demand> required, ArrayRef<SyncKind> actions) {
  auto cut = solveMembarCut(cfg, required, actions);
  if (!cut)
    return function.emitError("synchronization demand has no legal cut");
  LLVM_DEBUG({
    llvm::dbgs() << function.getName() << ": " << cfg.nodes.size()
                 << " CFG points, " << required.size() << " demands, "
                 << cut->sites.count() << " sites, cost " << cut->cost
                 << ", lower bound " << cut->lowerBound << "\n";
  });
  for (unsigned id : cut->sites.set_bits()) {
    const auto &node = cfg.nodes[id / actions.size()];
    Operation *before = node.op;
    SyncKind action = actions[id % actions.size()];
    OpBuilder builder(before);
    auto loc = before->getLoc();
    switch (action) {
    case SyncKind::ClusterBarrier:
    case SyncKind::ClusterFenceMbarInit: {
      // The specialized release must run on the thread that issued the init.
      bool init = action == SyncKind::ClusterFenceMbarInit && node.body == 0;
      if (init)
        nvidia_gpu::FenceMBarrierInitReleaseClusterOp::create(builder, loc);
      // An existing rendezvous at a selected site is a relaxed barrier.
      if (auto barrier = dyn_cast<nvidia_gpu::ClusterBarrierOp>(before))
        barrier.setRelaxed(init);
      else
        nvidia_gpu::ClusterBarrierOp::create(builder, loc, /*relaxed=*/init);
      break;
    }
    case SyncKind::CTAProxyFence:
    case SyncKind::ClusterProxyFence:
      nvidia_gpu::FenceAsyncSharedOp::create(
          builder, loc, action == SyncKind::ClusterProxyFence);
      break;
    case SyncKind::WaitLoad:
    case SyncKind::WaitStore:
      nvidia_gpu::TMEMWaitOp::create(builder, loc,
                                     action == SyncKind::WaitLoad
                                         ? nvidia_gpu::TMEMWaitKind::LOAD
                                         : nvidia_gpu::TMEMWaitKind::STORE);
      break;
    case SyncKind::RegionBarrier:
    case SyncKind::WarpBarrier:
      gpu::BarrierOp::create(builder, loc, gpu::AddrSpace::Local,
                             action == SyncKind::WarpBarrier
                                 ? gpu::BarrierScope::Warp
                                 : gpu::BarrierScope::CTA);
      break;
    }
  }
  return std::move(cut->sites);
}

template <typename T, typename Model>
LogicalResult placeAcrossRoles(const CallGraph<T> &calls, Model &model,
                               ArrayRef<SyncKind> actions) {
  MembarCFGBuilder builder(model, calls.getModuleOp());
  for (auto function : calls.getRoots()) {
    auto cfg =
        builder.build(function.getFunctionBody(), /*expandWarpRoles=*/true);
    if (failed(
            placeRequired(function, cfg, model.discoverDemands(cfg), actions)))
      return failure();
  }
  return success();
}

} // namespace

LogicalResult placeProxyFences(ModuleOp module, BufferRegionAnalysis &regions) {
  AsyncFenceModel model(regions, gpu::TritonGPUDialect::getNumCTAs(module));
  return placeAcrossRoles(
      CallGraph<int>(module), model,
      {SyncKind::ClusterProxyFence, SyncKind::CTAProxyFence});
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
  return placeAcrossRoles(
      allocation, model,
      {SyncKind::ClusterBarrier, SyncKind::ClusterFenceMbarInit});
}

LogicalResult placeBarriers(ModuleAllocation &moduleAllocation,
                            BufferRegionAnalysis &regions,
                            SMemOrdering *ordering, PlacementMode mode,
                            CompletionScopeFn completionScope,
                            DenseSet<Operation *> distributable) {
  ModuleOp module = moduleAllocation.getModuleOp();
  std::optional<TMemModel> tmem;
  if (mode != PlacementMode::Shared)
    tmem.emplace(module, regions);
  auto *tmemModel = tmem ? &*tmem : nullptr;

  auto sharedMode = mode == PlacementMode::TensorMemory
                        ? SMemModel::Mode::SynchronizationOnly
                        : SMemModel::Mode::Role;
  // Graph accesses borrow index expressions from the issuing function.
  std::map<Operation *, SMemModel> models;
  for (auto function : module.getOps<FunctionOpInterface>())
    models.try_emplace(function, *moduleAllocation.getFuncData(function),
                       regions, ordering, sharedMode, completionScope);

  MembarModel model(models, tmemModel);
  MembarCFGBuilder builder(model, module);
  for (auto function : moduleAllocation.getRoots()) {
    // Callee footprints are in kernel coordinates.
    Allocation &allocation = *moduleAllocation.getFuncData(function);
    SmallVector<Region *> bodies{&function.getFunctionBody()};
    function.walk([&](gpu::WarpSpecializeOp ws) {
      bodies.push_back(&ws.getDefaultRegion());
      llvm::append_range(bodies, ws.getPartitionRegions());
    });
    DenseMap<Operation *, SmallVector<Worker, 2>> pairedReleases;
    for (Region *body : bodies) {
      auto cfg = builder.build(*body, /*expandWarpRoles=*/false);
      // A specialized role may inherit transfers from outside its graph.
      if (ordering && bodies.size() == 1 && mode != PlacementMode::TensorMemory)
        recordSharedCompletions(cfg, *ordering);
      // A shared callee occurs once per role graph. The first role chooses
      // its notification modes. Later roles see the emitted modes as fixed.
      for (auto &node : cfg.nodes)
        if (distributable.erase(node.op))
          node.effects.arrivalMode = ArrivalMode::Either;
      auto [required, optional] =
          discoverDemands(cfg, allocation, ordering, tmemModel);
      auto sites = placeRequired(function, cfg, required, ACTIONS);
      if (failed(sites))
        return failure();
      for (unsigned sink : findUncoveredSinks(cfg, optional, ACTIONS, *sites)) {
        Operation *op = cfg.nodes[sink].op;
        if (auto arrive = dyn_cast<nvidia_gpu::ArriveBarrierOp>(op))
          arrive.setPerWarp(true);
        else
          cast<nvidia_gpu::BarrierExpectOp>(op).setPerWarp(true);
      }
      if (tmem && isa<gpu::WarpSpecializePartitionsOp>(body->getParentOp())) {
        auto worker = getIndependentReleases(cfg, required, *sites);
        if (!worker.releases.empty())
          pairedReleases[body->getParentOp()].push_back(std::move(worker));
      }
    }
    for (auto &entry : pairedReleases)
      schedulePairedReleases(entry.second,
                             nvidia_gpu::getModuleTwoCTAs(module));
  }
  return success();
}

} // namespace mlir::triton::membar
