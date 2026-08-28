#include "PartitionAttrs.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUAUTOMATICWARPSPECIALIZATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct VerifyWarpSpecializationPartitions
    : PassWrapper<VerifyWarpSpecializationPartitions, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      VerifyWarpSpecializationPartitions)

  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](LoopLikeOpInterface loop) {
      if (!loop->hasAttr(kPartitionStagesAttrName))
        return WalkResult::advance();
      if (failed(verifyPartitionedLoop(loop))) {
        signalPassFailure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    (void)result;
  }
};

struct AutomaticWarpSpecialization
    : triton::gpu::impl::TritonGPUAutomaticWarpSpecializationBase<
          AutomaticWarpSpecialization> {
  using TritonGPUAutomaticWarpSpecializationBase::
      TritonGPUAutomaticWarpSpecializationBase;

  void runOnOperation() override;
};

void multiBufferTMADescriptors(ModuleOp mod, int numStages) {
  SetVector<LoopLikeOpInterface> descUpdateLoops;
  mod.walk([&](LoopLikeOpInterface loop) {
    if (loop->hasAttr(kWarpSpecializeAttrName)) {
      loop.walk([&](triton::MakeTensorDescOp op) {
        if (auto loopOp = op->getParentOfType<LoopLikeOpInterface>()) {
          descUpdateLoops.insert(loopOp);
        }
      });
    }
  });

  // +1 to make sure that overlapping of the next desc update and the oldest
  // inflight TMA load is safe
  const int numDescs = numStages + 1;
  // CoarseSchedule's notion of numStages is the maximuim loop-pipelining
  // stage + 1, see CoarseSchedule::deSerialize(). So if we want n buffers,
  // we need to pass n + 1 as numStages.
  triton::CoarseSchedule schedule(numDescs + 1);

  for (auto loop : descUpdateLoops) {
    triton::lowerTMADescriptors(loop, schedule);
  }
}

void clearInternalWarpSpecializationAttrs(ModuleOp mod) {
  mod.walk([](Operation *op) {
    op->removeAttr(kPartitionAttrName);
    op->removeAttr(kPartitionOutputsAttrName);
    op->removeAttr(kPartitionStagesAttrName);
    op->removeAttr(kWarpSpecializeTagAttrName);
  });
}

// Resolve a response-ring view through the explicit partition capture list.
Value getCLCResponseAllocation(Value value, WarpSpecializeOp ws) {
  while (auto index = value.getDefiningOp<MemDescIndexOp>())
    value = index.getSrc();
  auto arg = dyn_cast<BlockArgument>(value);
  if (arg && arg.getOwner()->getParentOp() == ws.getPartitionOp())
    return ws.getPartitionOp().getExplicitCaptures()[arg.getArgNumber()];
  return value;
}

// Match a worker whose first tile is unconditional and whose subsequent
// iterations are controlled solely by the response from this CLC queue.
Value getCLCContinuationAllocation(scf::WhileOp loop, WarpSpecializeOp ws) {
  if (!loop.getBefore().hasOneBlock() || !loop.getAfter().hasOneBlock() ||
      !llvm::hasSingleElement(loop.getBefore().front()) ||
      !llvm::equal(loop.getConditionOp().getArgs(), loop.getBeforeArguments()))
    return {};
  auto condition =
      dyn_cast<BlockArgument>(loop.getConditionOp().getCondition());
  if (!condition || condition.getOwner() != &loop.getBefore().front() ||
      !matchPattern(loop.getInits()[condition.getArgNumber()], m_One()))
    return {};
  auto canceled = loop.getYieldOp()
                      .getOperand(condition.getArgNumber())
                      .getDefiningOp<ttng::CLCIsCanceledOp>();
  if (!canceled || canceled->getBlock() != &loop.getAfter().front())
    return {};
  auto response =
      canceled.getClcResult().getDefiningOp<ttng::CLCLoadResultOp>();
  if (!response || response->getBlock() != &loop.getAfter().front())
    return {};
  return getCLCResponseAllocation(response.getSrc(), ws);
}

void throttleCLC(WarpSpecializeOp ws) {
  // The automatic CLC pipeline currently supports single-CTA workers. Leave
  // other warp-specialized kernels and unmatched control flow unchanged.
  if (lookupNumCTAs(ws) != 1)
    return;
  SmallVector<ttng::CLCTryCancelOp> requests;
  ws.walk([&](ttng::CLCTryCancelOp op) { requests.push_back(op); });
  if (requests.size() != 1)
    return;
  auto request = requests.front();
  auto scheduler = request->getParentOfType<scf::WhileOp>();
  if (!scheduler ||
      scheduler->getBlock()->getParentOp() != ws.getPartitionOp() ||
      request->getBlock() != &scheduler.getAfter().front())
    return;
  Value response = getCLCContinuationAllocation(scheduler, ws);
  if (!response ||
      response != getCLCResponseAllocation(request.getResult(), ws))
    return;

  scf::WhileOp loaderLoop;
  bool unsupported = false;
  ws.walk([&](ttng::AsyncTMACopyGlobalToLocalOp op) {
    auto loop = op->getParentOfType<scf::WhileOp>();
    if (!loop || loop->getBlock()->getParentOp() != ws.getPartitionOp() ||
        loop == scheduler || (loaderLoop && loaderLoop != loop)) {
      unsupported = true;
      return;
    }
    loaderLoop = loop;
  });
  if (unsupported || !loaderLoop ||
      getCLCContinuationAllocation(loaderLoop, ws) != response)
    return;

  // Authorize one request at the start of each loader tile, before waiting
  // for input buffers. Loader admission allows input prefetch to
  // run ahead of MMA. Keep response and accumulator buffering independent.
  OpBuilder builder(ws);
  Location loc = request.getLoc();
  auto barrierType = cast<MemDescType>(request.getMbarrier().getType());
  Value credit = LocalAllocOp::create(builder, loc, barrierType);
  ttng::InitBarrierOp::create(builder, loc, credit, 1);
  ws.getPartitionOp().getExplicitCapturesMutable().append(credit);
  for (Region *region : ws.getPartitionRegions())
    region->front().addArgument(barrierType, loc);

  builder.setInsertionPointToStart(&loaderLoop.getAfter().front());
  Value loaderCredit = loaderLoop->getBlock()->getArguments().back();
  ttng::ArriveBarrierOp::create(builder, loc, loaderCredit, 1);

  builder.setInsertionPoint(scheduler);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
  Value schedulerCredit = scheduler->getBlock()->getArguments().back();
  scheduler = addIterArgsToLoop(builder, scheduler, ValueRange{zero});
  Value phase = scheduler.getAfterArguments().back();
  builder.setInsertionPointToStart(&scheduler.getAfter().front());
  ttng::WaitBarrierOp::create(builder, loc, schedulerCredit, phase);
  builder.setInsertionPoint(scheduler.getYieldOp());
  Value nextPhase = arith::XOrIOp::create(builder, loc, phase, one);
  scheduler.getYieldOp().getResultsMutable().append(nextPhase);

  // The loader cannot admit its next tile without the current CLC response.
  // This dependency prevents two arrivals before the scheduler consumes a
  // credit, so a single phase-tracked barrier suffices. The failed terminal
  // request consumes the final tile's credit and requires no additional
  // arrival.
  builder.setInsertionPointAfter(ws);
  ttng::InvalBarrierOp::create(builder, loc, credit);
  LocalDeallocOp::create(builder, loc, credit);
}

std::unique_ptr<Pass> createVerifyWarpSpecializationPartitionsPass() {
  return std::make_unique<VerifyWarpSpecializationPartitions>();
}

} // namespace

void AutomaticWarpSpecialization::runOnOperation() {
  if (clcThrottle != "none" && clcThrottle != "load") {
    getOperation().emitError("clc-throttle must be 'none' or 'load'");
    return signalPassFailure();
  }
  OpPassManager pm;
  auto addPassWithPartitionVerifier = [&](std::unique_ptr<Pass> pass) {
    pm.addPass(std::move(pass));
    pm.addPass(createVerifyWarpSpecializationPartitionsPass());
  };

  pm.addPass(createTritonGPUNormalizeWSWhileLoops());
  addPassWithPartitionVerifier(createTritonGPUPartitionScheduling());
  addPassWithPartitionVerifier(createNVWSHoistTmemStore());
  addPassWithPartitionVerifier(createNVWSInsertAref());
  addPassWithPartitionVerifier(createNVWSInsertTmemAref());
  // `int-range-optimizations` and SCCP are good at cleaning up loop arithmetic.
  // FIXME: Re-enable integer range analysis once it is fixed.
  // pm.addPass(arith::createIntRangeOptimizationsPass());
  addPassWithPartitionVerifier(createSCCPPass());
  addPassWithPartitionVerifier(createCSEPass());
  addPassWithPartitionVerifier(createNVWSLowerAref({numStages}));
  pm.addPass(createTritonGPUPartitionLoops());
  pm.addPass(createNVWSLowerWarpGroup());
  pm.addPass(createTritonGPUScheduleLoops());
  if (failed(runPipeline(pm, getOperation())))
    return signalPassFailure();

  // Multi-buffer TMA descriptors. We cannot rely on SWP to do it, to support
  // desc updates in nested loops.
  multiBufferTMADescriptors(getOperation(), numStages);
  clearInternalWarpSpecializationAttrs(getOperation());
  if (clcThrottle != "none")
    getOperation().walk(throttleCLC);
}
