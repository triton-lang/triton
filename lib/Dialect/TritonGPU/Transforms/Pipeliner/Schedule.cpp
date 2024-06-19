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
namespace ttng = mlir::triton::nvidia_gpu;

void tt::CoarseSchedule::insertDepsOfOp(Operation *op, int stage,
                                        tt::CoarseSchedule::Cluster cluster,
                                        bool includeArg) {
  for (Value operand : op->getOperands()) {
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
      if (insertIfAbsent(defOp, stage, cluster)) {
        insertDepsOfOp(defOp, stage, cluster, includeArg);
      }
    }
  }
}

SmallVector<std::tuple<Operation *, int, tt::CoarseSchedule::Cluster>>
tt::CoarseSchedule::getOpsInOrder(scf::ForOp forOp) {
  SmallVector<SmallVector<std::tuple<Operation *, int, Cluster>>, 8>
      orderClusters(clusters.size());
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (opToStageAndCluster.count(&op) == 0) {
      continue;
    }
    assert(opToStageAndCluster[&op].first < numStages &&
           "Op with invalid stage!");
    int clusterId = *opToStageAndCluster[&op].second;
    assert(clusterId == std::distance(clusters.begin(),
                                      opToStageAndCluster[&op].second) &&
           "Cluster ID mismatch!");
    orderClusters[clusterId].push_back(make_tuple(
        &op, opToStageAndCluster[&op].first, opToStageAndCluster[&op].second));
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
