#ifndef TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_
#define TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "llvm/ADT/ArrayRef.h"
#include <list>
#include <vector>

namespace mlir {
namespace triton {

namespace gpu {

/// Discover operations that should become async and assign latencies to them
/// based on the numStages value provided by the user.
DenseMap<Operation *, int> assignLatencies(ModuleOp forOp, int numStages);

/// Schedule the loop based on the latencies assigned to the operations.
void scheduleLoop(scf::ForOp forOp,
                  const DenseMap<Operation *, int> &opLatency);

}; // namespace gpu

/// This fill out the pipelining options including schedule and annotations
/// for wait ops. This also does pre-processing by converting some of the
/// loads into async loads so that the IR is ready to be pipelined.
bool preProcessLoopAndGetSchedule(scf::ForOp &forOp, int numStages,
                                  mlir::triton::PipeliningOption &options);

/// Fills out pipelining options for an outer loop pipelining case. This
/// schedules async copies to overlap with the epilogue of a loop.
bool getOuterLoopSchedule(scf::ForOp &forOp, int numStages,
                          mlir::triton::PipeliningOption &options);

/// Pipeline the Tensor Core Gen 05 MMA ops in `forOps` with `numStages` stages.
/// This will pre-process the loops, lowering the ops related to TG Gen5 MMA,
/// and then pipeline the loops using expander.
void pipelineTC05MMALoops(ModuleOp module,
                          const SmallVector<scf::ForOp> &forOps, int numStages,
                          bool disableExpander = false);

/// Pipeline the TMA stores in the loop.
bool pipelineTMAStores(scf::ForOp forOp);

/// Simple pipelining for the MMA ops which accumulator is modified in the loop.
scf::ForOp pipelineMMAWithScaledAcc(scf::ForOp forOp);

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
    ClusterList() = default;
    iterator begin() { return orderClusters.begin(); }
    iterator end() { return orderClusters.end(); }
    size_t size() { return orderClusters.size(); }
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
  };

  CoarseSchedule(int numStages) : numStages(numStages) {}
  int numStages;
  ClusterList clusters;
  using Cluster = decltype(clusters)::iterator;

  DenseMap<Operation *, std::pair<int, Cluster>> opToStageAndCluster;

  void insert(Operation *op, int stage, Cluster cluster) {
    opToStageAndCluster[op] = {stage, cluster};
  }

  bool insertIfAbsent(Operation *op, int stage, Cluster cluster) {
    if (opToStageAndCluster.count(op))
      return false;
    insert(op, stage, cluster);
    return true;
  }

  void insertMinimum(Operation *op, int stage, Cluster cluster) {
    auto res = opToStageAndCluster.insert({op, {stage, cluster}});
    if (res.second) {
      return;
    }
    auto &[existingStage, existingCluster] = res.first->second;
    existingStage = std::min(stage, existingStage);

    // If existingCluster is reachable from cluster,
    // then cluster is earlier in the list
    auto it = cluster;
    for (auto it = cluster; it != clusters.end(); ++it) {
      if (it == existingCluster) {
        existingCluster = cluster;
        return;
      }
    }
  }

  bool insertDepsOfOp(Operation *op, int stage, CoarseSchedule::Cluster cluster,
                      bool includeArg);

  void erase(Operation *op) { opToStageAndCluster.erase(op); }

  int count(Operation *op) { return opToStageAndCluster.count(op); }

  std::pair<int, Cluster> operator[](Operation *op) {
    return opToStageAndCluster[op];
  }

  auto find(Operation *op) const { return opToStageAndCluster.find(op); }

  SmallVector<std::tuple<Operation *, int, Cluster>>
  getOpsInOrder(scf::ForOp forOp);
  std::vector<std::pair<Operation *, unsigned>>
  createFinalSchedule(scf::ForOp forOp);
  void dump();
  bool empty() { return opToStageAndCluster.size() == 0; }
  void serialize(scf::ForOp &forOp);
  // Create a CoarseSchedule based on forOp's <stage, cluster>.
  void deSerialize(scf::ForOp &forOp);
};

// Add dependencies of anchor ops to the coarse schedule. Schedule them to
// the same stage and ordering cluster as the anchor op.
void scheduleDependencies(scf::ForOp forOp, CoarseSchedule &schedule);

} // namespace triton
} // namespace mlir
#endif // TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_
