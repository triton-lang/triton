#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-pipeline-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPIPELINESCHEDULER
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

bool hasLatenciesAssigned(scf::ForOp forOp,
                          const DenseMap<Operation *, int> &opLatency) {
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (opLatency.count(&op))
      return true;
  }
  return false;
}

CoarseSchedule scheduleKeyOps(scf::ForOp forOp,
                              const DenseMap<Operation *, int> &opLatency) {
  DenseSet<Operation *> processed;
  std::stack<std::pair<Operation *, int>> stack;
  llvm::MapVector<Operation *, int> opToStage;
  auto terminator = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  for (auto &startOp : forOp.getBody()->without_terminator()) {
    if (!opLatency.count(&startOp) || processed.count(&startOp))
      continue;
    stack.push(std::make_pair(&startOp, 0));
    while (!stack.empty()) {
      auto [op, stage] = stack.top();
      stack.pop();
      if (!processed.insert(op).second)
        continue;
      opToStage[op] = stage;
      if (opLatency.count(op)) {
        stage += opLatency.at(op);
      }
      for (auto user : op->getUsers()) {
        if (user == terminator)
          continue;
        stack.push(std::make_pair(user, stage));
      }
    }
  }
  auto stages = llvm::make_second_range(opToStage);
  int maxStage = *std::max_element(stages.begin(), stages.end());
  CoarseSchedule schedule(maxStage + 1);
  SmallVector<CoarseSchedule::Cluster> clusters(maxStage + 1);
  for (int i = 0; i <= maxStage; i++) {
    clusters[i] = schedule.clusters.newAtBack();
  }
  CoarseSchedule::Cluster epilogue = schedule.clusters.newAtBack();
  // Assign ops to the clusters in reverse-stage order;
  // ops with higher stage numbers are assigned first. This way we will
  // end up with roughly reverse program order in the clusters.
  for (auto [op, stage] : opToStage) {
    if (isa<scf::IfOp>(op)) {
      schedule.insert(op, stage, epilogue);
      continue;
    }
    schedule.insert(op, stage, clusters[maxStage - stage]);
  }

  return schedule;
}

}; // namespace

void scheduleLoop(scf::ForOp forOp,
                  const DenseMap<Operation *, int> &opLatency) {
  if (!preCondition(forOp) || !hasLatenciesAssigned(forOp, opLatency))
    return;
  // 1. Schedule key ops
  //    Based on the latencies, schedule the key ops to the stages.
  CoarseSchedule schedule = scheduleKeyOps(forOp, opLatency);
  // 2. Schedule dependencies
  //    Schedule the dependencies (regular and dist 1)
  // 3. Schedule the rest of the ops to the last stage

  // 4. Write the schedule to the IR
  schedule.serialize(forOp);
}

struct PipelineScheduler
    : public impl::TritonGPUPipelineSchedulerBase<PipelineScheduler> {
  using impl::TritonGPUPipelineSchedulerBase<
      PipelineScheduler>::TritonGPUPipelineSchedulerBase;

  int getNumStagesOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise use the
    // global control.
    if (!forOp->hasAttr(mlir::triton::kNumStagesAttrName))
      return numStages;
    return mlir::cast<IntegerAttr>(
               forOp->getAttr(mlir::triton::kNumStagesAttrName))
        .getInt();
  }

  void runOnOperation() override {
    // 1. Assign latencies
    //    Go over the interesting ops and assign latencies (based on the
    //    numStages) to the them, trying to populate the allowed stages. This
    //    step will be at some point extracted to separate pass that will be run
    //    only for loops missing the latency information.
    DenseMap<Operation *, int> opLatency =
        assignLatencies(getOperation(), numStages);
    // numStages should not be used below this point. We should know everything
    // based on the assigned stages

    // 2. Schedule the loops
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with num_stage <= 1.
      if (preCondition(forOp) && getNumStagesOrDefault(forOp) > 1)
        loops.push_back(forOp);
    });
    if (loops.empty())
      return;

    for (auto forOp : loops) {
      scheduleLoop(forOp, opLatency);
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
