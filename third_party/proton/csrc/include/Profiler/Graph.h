#ifndef PROTON_PROFILER_GRAPH_H_
#define PROTON_PROFILER_GRAPH_H_

#include "Context/Context.h"
#include "Data/Data.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace proton {

class Data;
class Runtime;

struct NodeStatus {
  using Status = uint8_t;

  static constexpr Status kMissingName = 1u << 0;
  static constexpr Status kMetric = 1u << 1;

  Status status{};

  constexpr NodeStatus() = default;
  constexpr explicit NodeStatus(Status status) : status(status) {}

  constexpr NodeStatus(bool isMissingName, bool isMetricNode)
      : status(static_cast<Status>((isMissingName ? kMissingName : 0) |
                                   (isMetricNode ? kMetric : 0))) {}

  constexpr bool isMissingName() const { return (status & kMissingName) != 0; }
  constexpr bool isMetricNode() const { return (status & kMetric) != 0; }
  void setMissingName() { status |= kMissingName; }
  void setMetricNode() { status |= kMetric; }
};


struct GraphState {
  // Capture tag to identify captured call paths
  static constexpr const char *captureTag = "<captured_at>";
  struct NodeState {
    // The graph node id for this node
    uint64_t nodeId{};
    // The entry id of the static entry associated with this node, which is
    // created at capture time and won't change for the same node id. This is
    // used to link the graph node to the captured call path in Data.
    std::map<Data *, size_t> dataToEntryId;
    // Whether the node has missing name or is a metric node, which is
    // determined at capture time and won't change for the same node id.
    NodeStatus status{};
  };
  using NodeStateRef = std::reference_wrapper<NodeState>;
  // Precomputed per-Data launch links maintained on graph node
  // create/clone/destroy callbacks.
  // data -> (static_entry_id -> graph-node metadata refs)
  std::map<Data *, std::unordered_map<size_t, std::vector<NodeStateRef>>>
      dataToEntryIdToNodeStates;
  // Mapping from node id to node state, has to be ordered based on node id
  // which is the order of node creation.
  std::map<uint64_t, NodeState> nodeIdToState;
  // Metric nodes and their per-node metric words, ordered by node id.
  std::map<uint64_t, size_t> metricNodeIdToNumWords;
  // If the graph is launched after profiling started,
  // we need to throw an error and this error is only thrown once
  bool captureStatusChecked{};
  // A unique id for the graph and graphExec instances; they don't overlap
  uint32_t graphId{};
  // Total number of GPU kernels launched by this graph
  size_t numNodes{1};
  // Total number of uint64 words written by all metric nodes in this graph.
  size_t numMetricWords{};
};

struct PendingGraphQueue {
  struct PendingGraph {
    size_t numNodes;
    size_t numWords;
    std::vector<DataEntry> dataEntries;
  };

  std::vector<PendingGraph> pendingGraphs;
  // The start buffer offset in the metric buffer for this queue
  size_t startBufferOffset{};
  // Total number of metric nodes in the pending graphs
  size_t numNodes{};
  // Total number of uint64 words written by all nodes in this queue
  size_t numWords{};
  // Device where the pending graphs are recorded
  void *device{};
  // Phase
  size_t phase{};

  explicit PendingGraphQueue(size_t startBufferOffset, size_t phase,
                             void *device)
      : startBufferOffset(startBufferOffset), phase(phase), device(device) {}

  void push(size_t numNodes, size_t numWords,
            const std::vector<DataEntry> &dataEntries) {
    pendingGraphs.emplace_back(PendingGraph{numNodes, numWords, dataEntries});
    this->numNodes += numNodes;
    this->numWords += numWords;
  }
};

class PendingGraphPool {
public:
  explicit PendingGraphPool(MetricBuffer *metricBuffer)
      : metricBuffer(metricBuffer), runtime(metricBuffer->getRuntime()) {}

  void push(size_t phase, const std::vector<DataEntry> &dataEntries,
            size_t numNodes, size_t numWords);

  // No GPU synchronization, No CPU locks
  void peek(size_t phase);

  // Synchronize and flush all pending graph
  bool flushAll();

  // Check if we need to flush all before pushing new pending graph
  bool flushIfNeeded(size_t numWords);

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
  // device -> phase -> slot
  std::map<void *, std::map<size_t, std::shared_ptr<Slot>>> pool;
};

} // namespace proton

#endif // PROTON_PROFILER_GRAPH_H_
