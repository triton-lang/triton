#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

bool tt::CoarseSchedule::insertMinimum(Operation *op, int stage,
                                       Cluster cluster) {
  auto res = opToStageAndCluster.insert({op, {stage, cluster}});
  if (res.second) {
    return true;
  }

  auto &[existingStage, existingCluster] = res.first->second;

  // Always insert if the stage is earlier.
  if (stage < existingStage) {
    existingStage = stage;
    existingCluster = cluster;
    return true;
  }

  // If the stage is later, no change.
  if (stage > existingStage) {
    return false;
  }

  // If existingCluster is reachable from cluster,
  // then cluster is earlier in the list
  for (auto it = std::next(cluster); it != clusters.end(); ++it) {
    if (it == existingCluster) {
      if (existingCluster == cluster)
        return false;
      existingCluster = cluster;
      return true;
    }
  }

  // Didn't change the cluster.
  return false;
}

bool tt::CoarseSchedule::insertDepsOfOp(Operation *op, int stage,
                                        tt::CoarseSchedule::Cluster cluster,
                                        bool includeArg, bool insertIfEarlier) {
  auto tryInsert = [&](Operation *op, int stage,
                       tt::CoarseSchedule::Cluster cluster) {
    if (!insertIfEarlier)
      return insertIfAbsent(op, stage, cluster);
    return insertMinimum(op, stage, cluster);
  };

  bool inserted = false;
  for (Value operand : getNestedOperands(op)) {
    Value v = operand;
    llvm::SmallDenseSet<Value> seen;
    while (auto arg = dyn_cast<BlockArgument>(v)) {
      if (!includeArg)
        break;
      if (!seen.insert(v).second)
        break;
      if (arg.getArgNumber() > 0 && arg.getOwner() == op->getBlock()) {
        auto yieldOp = op->getBlock()->getTerminator();
        v = yieldOp->getOperand(arg.getArgNumber() - 1);
        continue;
      }
      break;
    }
    Operation *defOp = v.getDefiningOp();
    if (defOp && defOp->getBlock() == op->getBlock()) {
      if (tryInsert(defOp, stage, cluster)) {
        inserted = true;
        insertDepsOfOp(defOp, stage, cluster, includeArg, insertIfEarlier);
      }
    }
  }
  return inserted;
}

// Split the cluster containing op into two clusters, one containing all
// operations before the op and one containing op and all operations after the
// op. Return the cluster containing op and all operations after the op. Do not
// split if the op is the first operation in the cluster.
tt::CoarseSchedule::Cluster
tt::CoarseSchedule::splitClusterBefore(Operation *op, scf::ForOp forOp) {
  auto cluster = opToStageAndCluster[op].second;
  std::optional<tt::CoarseSchedule::Cluster> newCluster = std::nullopt;
  for (auto &_op : forOp.getBody()->without_terminator()) {
    if (&_op == op) {
      break;
    }
    if (opToStageAndCluster[&_op].second == cluster) {
      if (!newCluster) {
        newCluster = clusters.newBefore(cluster);
      }
      opToStageAndCluster[&_op].second = *newCluster;
    }
  }
  return cluster;
}

// Check if op a will show up before op b in the final unrolled code.
bool tt::CoarseSchedule::isOpBefore(Operation *a, Operation *b) {
  assert(opToStageAndCluster.count(a) && opToStageAndCluster.count(b) &&
         "Operations must be in the schedule");
  auto [aStage, aCluster] = opToStageAndCluster[a];
  auto [bStage, bCluster] = opToStageAndCluster[b];
  if (aStage != bStage) {
    return aStage < bStage;
  }
  if (aCluster != bCluster) {
    return clusters.isBefore(aCluster, bCluster);
  }
  return a->isBeforeInBlock(b);
}

bool tt::CoarseSchedule::isOpInEarlierCluster(Operation *a, Operation *b) {
  assert(opToStageAndCluster.count(a) && opToStageAndCluster.count(b) &&
         "Operations must be in the schedule");
  return clusters.isBefore(opToStageAndCluster[a].second,
                           opToStageAndCluster[b].second);
}

bool tt::CoarseSchedule::isOpInSameCluster(Operation *a, Operation *b) {
  assert(opToStageAndCluster.count(a) && opToStageAndCluster.count(b) &&
         "Operations must be in the schedule");
  return opToStageAndCluster[a].second == opToStageAndCluster[b].second;
}

SmallVector<std::tuple<Operation *, int, tt::CoarseSchedule::Cluster>>
tt::CoarseSchedule::getOpsInOrder(scf::ForOp forOp) {
  SmallVector<SmallVector<std::tuple<Operation *, int, Cluster>>, 8>
      orderClusters(clusters.size());
  for (auto &op : forOp.getBody()->without_terminator()) {
    auto it = opToStageAndCluster.find(&op);
    if (it == opToStageAndCluster.end()) {
      continue;
    }
    auto [stage, cluster] = it->second;
    if (cluster == Cluster{}) {
      continue;
    }
    assert(stage < numStages && "Op with invalid stage!");
    int clusterId = *cluster;
    assert(clusterId == std::distance(clusters.begin(), cluster) &&
           "Cluster ID mismatch!");
    orderClusters[clusterId].push_back(make_tuple(&op, stage, cluster));
  }
  SmallVector<std::tuple<Operation *, int, Cluster>> opsInOrder;
  for (int i = 0; i < orderClusters.size(); i++) {
    for (auto [op, stage, cluster] : orderClusters[i]) {
      opsInOrder.push_back({op, stage, cluster});
    }
  }

  return opsInOrder;
}

std::vector<std::pair<Operation *, unsigned>>
tt::CoarseSchedule::createFinalSchedule(scf::ForOp forOp) {
  SmallVector<std::tuple<Operation *, int, tt::CoarseSchedule::Cluster>>
      opsInOrder = getOpsInOrder(forOp);
  std::vector<std::pair<Operation *, unsigned>> schedule;
  for (auto [op, stage, cluster] : opsInOrder)
    schedule.push_back({op, stage});
  return schedule;
}

void tt::CoarseSchedule::dump() {
  assert(numStages > 0 && "Invalid number of stages");
  for (int i = 0; i < numStages; i++) {
    llvm::dbgs() << "\n---- Ops in stage " << i << "\n";
    for (auto &[op, stageAndCluster] : opToStageAndCluster) {
      if (i == stageAndCluster.first) {
        llvm::dbgs() << "        cluster: " << *stageAndCluster.second
                     << ":\n\t" << *op << "\n";
      }
    }
  }
}

static void setStageCluster(Operation *op, int stage, int cluster) {
  auto ctx = op->getContext();
  op->setAttr(mlir::triton::kLoopStageAttrName,
              IntegerAttr::get(IntegerType::get(ctx, 32), stage));
  op->setAttr(mlir::triton::kLoopClusterAttrName,
              IntegerAttr::get(IntegerType::get(ctx, 32), cluster));
}

static std::pair<int, int> getStageCluster(Operation *op) {
  auto stage = op->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName);
  auto clusterId = op->getAttrOfType<IntegerAttr>(tt::kLoopClusterAttrName);
  assert(stage && clusterId &&
         "Operation is missing stage & cluster attribute");
  return {stage.getValue().getSExtValue(), clusterId.getValue().getSExtValue()};
}

static std::pair<int, int> getMinMaxCluster(scf::ForOp &forOp) {
  int minClusterId = -1, maxClusterId = -1;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (!op.hasAttr(mlir::triton::kLoopStageAttrName) ||
        !op.hasAttr(mlir::triton::kLoopClusterAttrName))
      continue;
    auto [_, cluster] = getStageCluster(&op);
    if (maxClusterId < 0) {
      minClusterId = cluster;
      maxClusterId = cluster;
      continue;
    }
    maxClusterId = cluster > maxClusterId ? cluster : maxClusterId;
    minClusterId = cluster < minClusterId ? cluster : minClusterId;
  }
  return std::make_pair(minClusterId, maxClusterId);
}

static std::optional<int> tryGetMaxStage(scf::ForOp &forOp) {
  std::optional<int> maxStage = std::nullopt;
  if (forOp->hasAttr(mlir::triton::kScheduledMaxStageAttrName)) {
    return forOp
        ->getAttrOfType<IntegerAttr>(mlir::triton::kScheduledMaxStageAttrName)
        .getValue()
        .getSExtValue();
  }
  return maxStage;
}

// Set <stage, cluster> based on CoarseSchedule.
void tt::CoarseSchedule::serialize(scf::ForOp &forOp) {
  for (auto [op, stage, cluster] : getOpsInOrder(forOp)) {
    setStageCluster(op, stage, *cluster);
  }

  Builder b(forOp.getContext());
  int maxStages = numStages - 1;
  if (auto maxStageAttr = tryGetMaxStage(forOp))
    maxStages = std::max(maxStages, *maxStageAttr);
  forOp->setAttr(mlir::triton::kScheduledMaxStageAttrName,
                 b.getI32IntegerAttr(maxStages));
}

// Create a CoarseSchedule based on forOp's <stage, cluster>.
LogicalResult tt::CoarseSchedule::deSerialize(scf::ForOp &forOp) {
  auto [minClusterId, maxClusterId] = getMinMaxCluster(forOp);
  std::optional<int> maxStage = tryGetMaxStage(forOp);
  if (!maxStage) {
    return failure();
  }
  numStages = *maxStage + 1;

  DenseMap<int, tt::CoarseSchedule::Cluster> clustersMap;
  for (int i = minClusterId; i < maxClusterId + 1; i++) {
    clustersMap.insert({i, clusters.newAtBack()});
  }
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!op.hasAttr(mlir::triton::kLoopStageAttrName))
      continue;
    auto [stage, clusterId] = getStageCluster(&op);
    insert(&op, stage, clustersMap[clusterId]);
  }
  return success();
}

// TODO: Should this be moved somewhere else?
// Add dependencies of anchor ops to the coarse schedule. Schedule them to
// the same stage and ordering cluster as the anchor op.
void tt::scheduleDependencies(scf::ForOp forOp, tt::CoarseSchedule &schedule) {
  int numStages = schedule.getNumStages();
  SmallVector<std::tuple<Operation *, int, tt::CoarseSchedule::Cluster>>
      opsInOrder = schedule.getOpsInOrder(forOp);
  // Schedule dependencies stage by stage.
  for (int stage = 0; stage < numStages; stage++) {
    for (auto [op, stage_, cluster] : opsInOrder) {
      if (stage_ != stage)
        continue;
      schedule.insertDepsOfOp(op, stage, cluster, false);
    }
  }
}
