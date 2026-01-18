#ifndef PROTON_PROFILER_GRAPH_H_
#define PROTON_PROFILER_GRAPH_H_

#include "Context/Context.h"
#include "Data/Metric.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace proton {

class Data;
class Runtime;

struct GraphState {
  using Callpath = std::vector<Context>;

  struct NodeState {
    // Mapping from Data object to captured callpath.
    std::map<Data *, Callpath> captureContexts;
    // A unique id for the graph node
    uint64_t nodeId{};
    // Whether the node is missing name
    bool isMissingName{};
    // Whether the node is a metric kernel node
    bool isMetricNode{};
  };

  // Capture tag to identify captured call paths
  static constexpr const char *captureTag = "<captured_at>";
  using NodeStateRef = std::reference_wrapper<NodeState>;
  // Cached per-Data callpath groups: Data -> (callpath -> [nodeStates...])
  std::map<Data *, std::map<Callpath, std::vector<NodeStateRef>>>
      dataToCallpathToNodeStates;
  // Mapping from node id to node state, has to be ordered based on node id
  // which is the order of node creation
  std::map<uint64_t, NodeState> nodeIdToState;
  // Identify whether a node is a metric kernel node.
  // NOTE: This set has to be ordered to match the node creation order.
  std::set<uint64_t> metricKernelNodeIds;
  // If the graph is launched after profiling started,
  // we need to throw an error and this error is only thrown once
  bool captureStatusChecked{};
  // A unique id for the graph and graphExec instances; they don't overlap
  uint32_t graphId{};
  // Total number of GPU kernels launched by this graph
  size_t numNodes{1};
};

struct PendingGraphQueue {
  struct PendingGraph {
    size_t numNodes;
    std::map<Data *, std::vector<size_t>> dataToEntryIds;
  };

  std::vector<PendingGraph> pendingGraphs;
  // The start buffer offset in the metric buffer for this queue
  size_t startBufferOffset{};
  // Total number of metric nodes in the pending graphs
  size_t numNodes{};
  // Device where the pending graphs are recorded
  void *device{};
  // Phase
  size_t phase{};

  explicit PendingGraphQueue(size_t startBufferOffset, size_t phase,
                             void *device)
      : startBufferOffset(startBufferOffset), phase(phase), device(device) {}

  void push(size_t numNodes,
            const std::map<Data *, std::vector<size_t>> &dataToEntryIds) {
    pendingGraphs.emplace_back(PendingGraph{numNodes, dataToEntryIds});
    this->numNodes += numNodes;
  }
};

class PendingGraphPool {
public:
  explicit PendingGraphPool(MetricBuffer *metricBuffer)
      : metricBuffer(metricBuffer), runtime(metricBuffer->getRuntime()) {}

  void push(size_t phase,
            const std::map<Data *, std::vector<size_t>> &dataToEntryIds,
            size_t numNodes);

  // No GPU synchronization, No CPU locks
  void peek(size_t phase);

  // Synchronize and flush all pending graph
  bool flushAll();

  // Check if we need to flush all before pushing new pending graph
  bool flushIfNeeded(size_t numNodes);

private:
  struct Slot {
    mutable std::mutex mutex;
    std::optional<PendingGraphQueue> queue;
  };

  // The current starting buffer offset in the metric buffer
  // device -> offset
  std::map<void *, size_t> deviceBufferOffset{};
  // How much remaining capacity in the metric buffer we have
  // device -> capacity
  std::map<void *, size_t> deviceRemainingCapacity{};
  MetricBuffer *metricBuffer{};
  Runtime *runtime{};
  mutable std::mutex mutex;
  std::map<void *, std::map<size_t, std::shared_ptr<Slot>>> pool;
};

} // namespace proton

#endif // PROTON_PROFILER_GRAPH_H_
