#include "triton/Analysis/Membar/Pass.h"
#include "triton/Analysis/CallGraph.h"
#include "triton/Analysis/Membar/CFGBuilder.h"
#include "triton/Analysis/Membar/Dependencies.h"
#include "triton/Analysis/Membar/Solver.h"
#include "triton/Analysis/Membar/TmemModel.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>
#include <map>

#define DEBUG_TYPE "membar-placement"

namespace mlir::triton::membar {
namespace {

template <typename T, typename Model>
DenseMap<Operation *, unsigned> getCalleeWeights(CallGraph<T> &calls,
                                                 Model &model) {
  DenseMap<Operation *, unsigned> calleeWeights;
  // Each root gets a postorder visit without an incoming call.
  SmallVector<unsigned> pathWeights(calls.getRoots().size(), 1);
  // walk visits each call path. Propagate just this path's contribution.
  calls.template walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
      [&](CallOpInterface call, FunctionOpInterface callee) {
        auto caller = call->getParentOfType<FunctionOpInterface>();
        const auto &indices = model.getIndices(caller);
        unsigned weight =
            getBlockWeight(call->getBlock(), indices, pathWeights.back());
        pathWeights.push_back(weight);
        unsigned &total = calleeWeights[callee];
        bool overflow;
        total = llvm::SaturatingAdd(total, weight, &overflow);
        assert(!overflow && "Membar callee weight overflow");
      },
      [&](FunctionOpInterface) { pathWeights.pop_back(); });
  return calleeWeights;
}

template <typename Facts>
void emit(const MembarCFG<Facts> &cfg, ArrayRef<Action> actions,
          const llvm::BitVector &selected) {
  for (unsigned id : selected.set_bits()) {
    Operation *before = cfg.nodes[id / actions.size()].op;
    Action action = actions[id % actions.size()];
    OpBuilder builder(before);
    auto loc = before->getLoc();
    switch (action) {
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
                           ArrayRef<Action> actions, Discover discover) {
  SmallVector<Region *> bodies{&function.getFunctionBody()};
  function.walk([&](gpu::WarpSpecializeOp ws) {
    bodies.push_back(&ws.getDefaultRegion());
    llvm::append_range(bodies, ws.getPartitionRegions());
  });
  for (Region *body : bodies) {
    auto cfg = builder.build(*body);
    auto demands = discover(*body, cfg);
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

} // namespace

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
  auto callbacks = getTensorCallbackOps(module);
  const auto calleeWeights = getCalleeWeights(moduleAllocation, model);
  MembarCFGBuilder builder(model, callbacks, calleeWeights);
  for (auto function : moduleAllocation.getRoots()) {
    // Callee footprints are in kernel coordinates.
    Allocation &allocation = *moduleAllocation.getFuncData(function);
    auto discover = [&](Region &, const auto &cfg) {
      return discoverDemands(cfg, allocation, sharedFilter, tmemModel);
    };
    if (failed(placeRegions(function, builder, ACTIONS, discover)))
      return failure();
  }
  return success();
}

} // namespace mlir::triton::membar
