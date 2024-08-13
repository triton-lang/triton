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

/// This fill out the pipelining options including schedule and annotations
/// for wait ops. This also does pre-processing by converting some of the
/// loads into async loads so that the IR is ready to be pipelined.
bool preProcessLoopAndGetSchedule(scf::ForOp &forOp, int numStages,
                                  mlir::triton::PipeliningOption &options);

/// Fills out pipelining options for an outer loop pipelining case. This
/// schedules async copies to overlap with the epilogue of a loop.
bool getOuterLoopSchedule(scf::ForOp &forOp, int numStages,
                          mlir::triton::PipeliningOption &options);

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

  void insertDepsOfOp(Operation *op, int stage, CoarseSchedule::Cluster cluster,
                      bool includeArg);

  void erase(Operation *op) { opToStageAndCluster.erase(op); }

  int count(Operation *op) { return opToStageAndCluster.count(op); }

  std::pair<int, Cluster> operator[](Operation *op) {
    return opToStageAndCluster[op];
  }

  SmallVector<std::tuple<Operation *, int, Cluster>>
  getOpsInOrder(scf::ForOp forOp);
  std::vector<std::pair<Operation *, unsigned>>
  createFinalSchedule(scf::ForOp forOp);
  void dump();
};

} // namespace triton
} // namespace mlir
#endif // TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_
