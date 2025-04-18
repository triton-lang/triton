#ifndef TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_
#define TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/ArrayRef.h"
#include <list>
#include <vector>

namespace mlir {
namespace triton {

namespace gpu {

/// Discover operations that should become async and assign latencies to them
/// based on the numStages value provided by the user. Return false if no loops
/// are assigned with latencies.
bool assignLatencies(ModuleOp moduleOp, int numStages);

/// Schedule the loops based on the latencies assigned to the operations.
void scheduleLoops(ModuleOp moduleOp);

/// Lower the loops to prepare them for pipeline expansion.
void lowerLoops(ModuleOp moduleOp);

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

  DenseMap<Operation *, std::pair<int, Cluster>> opToStageAndCluster;

  void setNumStages(int numStages) { this->numStages = numStages; }
  int getNumStages() { return numStages; }

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

  bool empty() const { return opToStageAndCluster.size() == 0; }
  auto end() const { return opToStageAndCluster.end(); }
  auto begin() const { return opToStageAndCluster.begin(); }

  // Set <stage, cluster> based on CoarseSchedule.
  void serialize(scf::ForOp &forOp);
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

} // namespace triton
} // namespace mlir
#endif // TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_
