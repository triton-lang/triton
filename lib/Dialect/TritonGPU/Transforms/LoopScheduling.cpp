#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-loop-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPULOOPSCHEDULING
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
  llvm::MapVector<Operation *, int> opToStage;
  // Find terminator for later reference
  auto terminator = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  // Determine all operations that have a non-zero latency
  SmallVector<Operation *> latOps;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (opLatency.count(&op))
      latOps.push_back(&op);
  }
  // If no latency ops, nothing to schedule
  if (latOps.empty())
    return CoarseSchedule(0);

  // Compute the longest path to the yield for each operation reachable
  // from any latency operation.
  DenseMap<Operation *, int> distance;
  std::function<int(Operation *)> computeDistance = [&](Operation *op) -> int {
    auto it = distance.find(op);
    if (it != distance.end())
      return it->second;
    // Compute max distance among all users that are inside the loop body
    int maxDist = -1;
    for (Operation *user : op->getUsers()) {
      // Only consider users inside the same block and not the terminator
      Operation *inBlockUser = forOp.getBody()->findAncestorOpInBlock(*user);
      if (!inBlockUser || inBlockUser == terminator)
        continue;
      int distUser = computeDistance(inBlockUser);
      if (distUser > maxDist)
        maxDist = distUser;
    }
    int lat = 0;
    if (opLatency.count(op))
      lat = opLatency.lookup(op);
    // If an op has no users (maxDist == -1) but has latency, we include its
    // latency otherwise it contributes 0 to the distance.
    int d = lat + (maxDist < 0 ? 0 : maxDist);
    distance[op] = d;
    return d;
  };

  // Compute distances for all latency-starting ops
  int maxDistance = 0;
  for (Operation *latOp : latOps) {
    int d = computeDistance(latOp);
    if (d > maxDistance)
      maxDistance = d;
  }

  // Assign stage to each op reachable from a latency op
  for (auto &kv : distance) {
    Operation *op = kv.first;
    int dist = kv.second;
    // We only schedule ops that are downstream of a latency op
    // (had a non-negative distance due to a latency op).
    if (dist >= 0)
      opToStage[op] = maxDistance - dist;
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

// Find dependencies with distance of 1. They will go to the next stage,
// but in the cluster before the current op.
void scheduleDistanceOneDependencies(scf::ForOp forOp,
                                     CoarseSchedule &schedule) {
  int numStages = schedule.numStages;
  auto getNestedOperands = [](Operation *op) -> SmallVector<Value> {
    SmallVector<Value> operands;
    op->walk([&](Operation *nestedOp) {
      for (Value operand : nestedOp->getOperands()) {
        if (operand.getParentBlock()->getParentOp()->isAncestor(nestedOp))
          operands.push_back(operand);
      }
    });
    return operands;
  };

  // Mapping from the cluster to the cluster before it.
  DenseMap<CoarseSchedule::Cluster *, CoarseSchedule::Cluster> dist1Cluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0)
      continue;
    auto [stage, cluster] = schedule[&op];
    // Can't schedule past the last stage.
    if (stage == numStages - 1)
      continue;
    for (Value operand : getNestedOperands(&op)) {
      if (auto arg = dyn_cast<BlockArgument>(operand)) {
        if (arg.getArgNumber() > 0 && arg.getOwner() == op.getBlock()) {
          auto yieldOp = op.getBlock()->getTerminator();
          Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
          Operation *defOp = v.getDefiningOp();
          if (defOp && schedule.count(defOp) == 0) {
            if (isa<tt::LoadOp>(defOp)) {
              // Exception: Schedule loads with a distance of 1 together
              // with the current op.
              schedule.insertIfAbsent(defOp, stage, cluster);
              schedule.insertDepsOfOp(defOp, stage, cluster, true);
            } else {
              if (dist1Cluster.count(&cluster) == 0) {
                dist1Cluster[&cluster] = schedule.clusters.newBefore(cluster);
              }
              schedule.insertIfAbsent(defOp, stage + 1, dist1Cluster[&cluster]);
              schedule.insertDepsOfOp(defOp, stage + 1, dist1Cluster[&cluster],
                                      true);
            }
          }
        }
      }
    }
  }
}

// Schedule the prologue and epilogue `if` ops in the loop, pushing them as
// close to the loop boundaries as possible. Return the cluster after the
// prologue (or the beginning of the loop if there is no prologue).
CoarseSchedule::Cluster schedulePrologueAndEpilogue(scf::ForOp forOp,
                                                    CoarseSchedule &schedule) {
  int numStages = schedule.numStages;
  CoarseSchedule::Cluster afterPrologue = schedule.clusters.begin();

  // Look for the IfOp that is in the backward slice any of the currently
  // scheduled ops and put it at the beginning of the loop.
  DenseMap<scf::IfOp, int> ifsToStage;
  // Go stage by stage.
  for (int stage = 0; stage < numStages; stage++) {
    for (auto [op, stage_, cluster] : schedule.getOpsInOrder(forOp)) {
      if (stage_ != stage)
        continue;
      SetVector<Operation *> backwardSlice;
      BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      getBackwardSlice((Operation *)op, &backwardSlice, opt);

      for (auto op : backwardSlice) {
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          ifsToStage.insert({ifOp, stage});
        }
      }
    }
  }
  if (!ifsToStage.empty()) {
    CoarseSchedule::Cluster prologueCluster = schedule.clusters.newAtFront();
    for (auto [ifOp, stage] : ifsToStage) {
      schedule.insert(ifOp, stage, prologueCluster);
    }
  }

  // Other IfOps should be pushed to the end.
  CoarseSchedule::Cluster epilogueCluster = schedule.clusters.newAtBack();
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (ifsToStage.count(ifOp) == 0) {
        schedule.insertIfAbsent(ifOp, numStages - 1,
                                epilogueCluster); // after prefetch extracts
      }
    }
  }
  return afterPrologue;
}

void scheduleRemainingToLastStage(scf::ForOp forOp, CoarseSchedule &schedule,
                                  CoarseSchedule::Cluster afterPrologue) {
  int numStages = schedule.numStages;
  // Assign the rest of the ops to the last stage.
  // Take care of the ordering of the ops - uses cannot be scheduled to the
  // cluster before the definition.
  DenseMap<Operation *, CoarseSchedule::Cluster> opToCluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0) {
      opToCluster[&op] = afterPrologue;
    }
  }
  SmallVector<Operation *> queue;
  for (auto [op, stage, cluster] : schedule.getOpsInOrder(forOp)) {
    // We really only care about the producers from the last stage.
    // Others will be scheduled before these ops anyway.
    if (stage == numStages - 1) {
      queue.push_back(op);
    }
  }
  while (!queue.empty()) {
    Operation *op = queue.pop_back_val();
    for (auto user : op->getUsers()) {
      if (opToCluster.count(user)) {
        CoarseSchedule::Cluster userCluster = opToCluster[user];
        CoarseSchedule::Cluster opCluster;
        if (schedule.count(op))
          opCluster = schedule[op].second;
        else
          opCluster = opToCluster[op];
        if (*userCluster < *opCluster) {
          opToCluster[user] = opCluster;
          queue.push_back(user);
        }
      }
    }
  }
  for (auto [op, cluster] : opToCluster) {
    schedule.insert(op, numStages - 1, cluster);
  }
}

}; // namespace

void scheduleLoop(scf::ForOp forOp,
                  const DenseMap<Operation *, int> &opLatency) {
  if (!hasLatenciesAssigned(forOp, opLatency))
    return;
  // Based on the latencies, schedule the key ops to the stages.
  CoarseSchedule schedule = scheduleKeyOps(forOp, opLatency);
  if (schedule.empty())
    return;
  LLVM_DEBUG({
    LDBG("Initial coarse schedule:");
    schedule.dump();
  });
  // Schedule the dependencies
  CoarseSchedule::Cluster afterPrologue =
      schedulePrologueAndEpilogue(forOp, schedule);
  LLVM_DEBUG({
    LDBG("Coarse schedule with prologue and epilogue:");
    schedule.dump();
  });
  scheduleDependencies(forOp, schedule);
  LLVM_DEBUG({
    LDBG("Coarse schedule with dependencies:");
    schedule.dump();
  });
  scheduleDistanceOneDependencies(forOp, schedule);
  LLVM_DEBUG({
    LDBG("Coarse schedule with dist 1:");
    schedule.dump();
  });
  scheduleRemainingToLastStage(forOp, schedule, afterPrologue);
  LLVM_DEBUG({
    LDBG("Final coarse schedule:");
    schedule.dump();
  });

  // Write the schedule to the IR
  schedule.serialize(forOp);
}

class TritonGPULoopSchedulingPass
    : public impl::TritonGPULoopSchedulingBase<TritonGPULoopSchedulingPass> {
public:
  using impl::TritonGPULoopSchedulingBase<
      TritonGPULoopSchedulingPass>::TritonGPULoopSchedulingBase;

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
    // Go over the interesting ops and assign latencies (based on the
    // numStages) to the them, trying to populate the allowed stages. This
    // step will be at some point extracted to separate pass that will be run
    // only for loops missing the latency information.
    DenseMap<Operation *, int> opLatency =
        assignLatencies(getOperation(), numStages);
    // numStages should not be used below this point. We should know everything
    // based on the assigned stages

    // Schedule the loops
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with num_stage <= 1.
      if (getNumStagesOrDefault(forOp) > 1)
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
