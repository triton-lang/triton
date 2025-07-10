#ifndef TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_
#define TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "llvm/ADT/ArrayRef.h"
#include <list>
#include <vector>

namespace mlir {
namespace triton {

namespace gpu {

/// Lower the loops to prepare them for pipeline expansion.
void lowerLoops(ModuleOp moduleOp);

bool hasGpuBarriers(scf::ForOp forOp);
bool isSafeToPipeline(scf::ForOp forOp);
llvm::MapVector<Operation *, std::pair<int, Operation *>>
loadOpsToIndirectionLevel(scf::ForOp forOp, bool pipelineWithoutDot,
                          triton::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                          int numStages, bool filterSmall = true);

}; // namespace gpu

/// Pipeline the TMA stores in the loop.
bool pipelineTMAStores(scf::ForOp forOp);

/// This does post-processing on the pipelined loop to try to pipeline wgmma
/// ops.
// TODO: this should be included as part of the pipeline but currently the wgmma
// wait modeling is problematic.
void asyncLaunchDots(scf::ForOp forOp);

/// Post process the pipelined loop by updating the wait ops with the right
/// number of groups in flight.
void updateWaits(ModuleOp module);

class CoarseSchedule {
public:
  class ClusterList {
    std::list<int> orderClusters;

  public:
    using iterator = decltype(orderClusters)::iterator;
    using const_iterator = decltype(orderClusters)::const_iterator;
    ClusterList() = default;
    iterator begin() { return orderClusters.begin(); }
    const_iterator begin() const { return orderClusters.begin(); }
    iterator end() { return orderClusters.end(); }
    const_iterator end() const { return orderClusters.end(); }
    size_t size() const { return orderClusters.size(); }
    iterator newAtBack() {
      orderClusters.push_back(orderClusters.size());
      return std::prev(orderClusters.end());
    }
    iterator newAtFront() {
      orderClusters.push_front(-1);
      for (auto &clusterId : orderClusters) {
        clusterId++;
      }
      return orderClusters.begin();
    }
    iterator newBefore(iterator cluster) {
      auto ret = orderClusters.insert(cluster, *cluster);
      for (auto &clusterId : llvm::make_range(cluster, orderClusters.end())) {
        clusterId++;
      }
      return ret;
    }

    bool isBefore(iterator a, iterator b) const {
      if (a == b)
        return false;
      for (auto it = begin(); it != end(); ++it) {
        if (it == a)
          return true;
        if (it == b)
          return false;
      }
      llvm::report_fatal_error(
          "One or both clusters not found in clusters list!");
    }
  };

  CoarseSchedule() = default;
  CoarseSchedule(int numStages) : numStages(numStages) {}
  ClusterList clusters;
  using Cluster = ClusterList::iterator;
  using ClusterHash = size_t;

  llvm::MapVector<Operation *, std::pair<int, Cluster>> opToStageAndCluster;

  void setNumStages(int numStages) { this->numStages = numStages; }
  int getNumStages() const { return numStages; }

  void insert(Operation *op, int stage, Cluster cluster) {
    if (stage >= numStages) {
      numStages = stage + 1;
    }
    opToStageAndCluster[op] = {stage, cluster};
  }

  bool insertIfAbsent(Operation *op, int stage, Cluster cluster) {
    if (opToStageAndCluster.count(op))
      return false;
    insert(op, stage, cluster);
    return true;
  }

  bool insertMinimum(Operation *op, int stage, Cluster cluster);

  bool insertDepsOfOp(Operation *op, int stage, CoarseSchedule::Cluster cluster,
                      bool includeArg, bool insertIfEarlier = false);

  // Remove empty stages and clusters from the schedule, adjusting the maximum
  // number of stages as appropriate.
  void shrinkToFit();

  void erase(Operation *op) { opToStageAndCluster.erase(op); }

  int count(Operation *op) const { return opToStageAndCluster.count(op); }

  std::pair<int, Cluster> operator[](Operation *op) {
    return opToStageAndCluster[op];
  }

  auto find(Operation *op) const { return opToStageAndCluster.find(op); }

  // Split the cluster containing op into two clusters, one containing all
  // operations before the op and one containing op and all operations after the
  // op. Return the cluster containing op and all operations after the op.
  Cluster splitClusterBefore(Operation *op, scf::ForOp forOp);

  // Check if op a will show up before op b in the final unrolled code.
  bool isOpBefore(Operation *a, Operation *b) const;

  // Check if op a is in earlier cluster than op b.
  bool isOpInEarlierCluster(Operation *a, Operation *b) const;

  // Check if op a is in the same cluster as op b.
  bool isOpInSameCluster(Operation *a, Operation *b) const;

  SmallVector<std::tuple<Operation *, int, Cluster>>
  getOpsInOrder(scf::ForOp forOp) const;
  std::vector<std::pair<Operation *, unsigned>>
  createFinalSchedule(scf::ForOp forOp) const;

  bool empty() const { return opToStageAndCluster.size() == 0; }
  auto end() const { return opToStageAndCluster.end(); }
  auto begin() const { return opToStageAndCluster.begin(); }

  // Set <stage, cluster> based on CoarseSchedule.
  void serialize(scf::ForOp &forOp) const;
  // Create a CoarseSchedule based on forOp's <stage, cluster>.
  LogicalResult deSerialize(scf::ForOp &forOp);

  static ClusterHash hashCluster(Cluster cluster) {
    return reinterpret_cast<ClusterHash>(&*cluster);
  }

  LLVM_DUMP_METHOD void dump();

private:
  int numStages = 0;
};

// Add dependencies of anchor ops to the coarse schedule. Schedule them to
// the same stage and ordering cluster as the anchor op.
void scheduleDependencies(scf::ForOp forOp, CoarseSchedule &schedule);

class OpBuilderForStage : public mlir::ImplicitLocOpBuilder,
                          public OpBuilder::Listener {
public:
  explicit OpBuilderForStage(Location loc, Operation *op,
                             CoarseSchedule &schedule)
      : ImplicitLocOpBuilder(loc, op, this), schedule(schedule) {
    if (auto it = schedule.find(op); it != schedule.end())
      std::tie(stage, cluster) = it->second;
  }

  void setStageCluster(std::pair<int, CoarseSchedule::Cluster> stageCluster) {
    stage = stageCluster.first;
    cluster = stageCluster.second;
  }

  void notifyOperationInserted(Operation *op, InsertPoint previous) {
    if (stage && cluster)
      schedule.insert(op, *stage, *cluster);
  }

private:
  std::optional<int> stage;
  std::optional<CoarseSchedule::Cluster> cluster;
  CoarseSchedule &schedule;
};

namespace gpu {
void scheduleDistanceOneDependencies(scf::ForOp forOp,
                                     CoarseSchedule &schedule);
void scheduleRemainingToLastStage(scf::ForOp forOp, CoarseSchedule &schedule,
                                  CoarseSchedule::Cluster afterPrologue);
} // namespace gpu

} // namespace triton
} // namespace mlir
#endif // TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_
