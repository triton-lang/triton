#include "triton/Analysis/Membar/Pass.h"
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

DenseMap<Operation *, unsigned>
getCalleeWeights(ModuleAllocation &calls,
                 std::map<Operation *, SMemModel> &models) {
  DenseMap<Operation *, unsigned> calleeWeights;
  // Each root gets a postorder visit without an incoming call.
  SmallVector<unsigned> pathWeights(calls.getRoots().size(), 1);
  // walk visits each call path. Propagate just this path's contribution.
  calls.walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
      [&](CallOpInterface call, FunctionOpInterface callee) {
        auto caller = call->getParentOfType<FunctionOpInterface>();
        const auto &indices = models.at(caller).getBufferIndexAnalysis();
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

void emit(const MembarCFG &cfg, const llvm::BitVector &selected) {
  for (unsigned id : selected.set_bits()) {
    Operation *before = cfg.nodes[id / ACTIONS.size()].op;
    Action action = ACTIONS[id % ACTIONS.size()];
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

  auto callbacks = getTensorCallbackOps(module);
  const auto calleeWeights = getCalleeWeights(moduleAllocation, models);
  MembarCFGBuilder builder(models, tmemModel, callbacks, calleeWeights);
  for (auto function : moduleAllocation.getRoots()) {
    // Callee footprints are in kernel coordinates. Only kernel SSA values
    // have allocator IDs for target-specific alias checks.
    Allocation &allocation = *moduleAllocation.getFuncData(function);
    SmallVector<Region *> bodies{&function.getFunctionBody()};
    function.walk([&](gpu::WarpSpecializeOp ws) {
      bodies.push_back(&ws.getDefaultRegion());
      llvm::append_range(bodies, ws.getPartitionRegions());
    });
    for (Region *body : bodies) {
      auto cfg = builder.build(*body);
      auto demands = discoverDemands(cfg, allocation, sharedFilter, tmemModel);
      auto cut = solveMembarCut(cfg, demands);
      if (!cut)
        return function.emitError("synchronization demand has no legal cut");
      LLVM_DEBUG({
        llvm::dbgs() << function.getName() << ": " << cfg.nodes.size()
                     << " CFG points, " << demands.size() << " demands, "
                     << cut->sites.count() << " sites, cost " << cut->cost
                     << ", lower bound " << cut->lowerBound << "\n";
      });
      emit(cfg, cut->sites);
    }
  }
  return success();
}

} // namespace mlir::triton::membar
