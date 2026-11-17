#include "triton/Analysis/Membar/Pass.h"
#include "triton/Analysis/Membar/CFGBuilder.h"
#include "triton/Analysis/Membar/Dependencies.h"
#include "triton/Analysis/Membar/Solver.h"
#include "triton/Analysis/Membar/TmemModel.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "membar-placement"

namespace mlir::triton::membar {
namespace {

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
  for (auto function : module.getOps<FunctionOpInterface>()) {
    Allocation &allocation = *moduleAllocation.getFuncData(function);
    SMemModel shared(allocation, regions, sharedMode, completionScope);
    MembarCFGBuilder builder(shared, tmemModel,
                             isKernel(function) ? Boundary::KernelReturn
                                                : Boundary::Clean);
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
