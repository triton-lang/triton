#ifndef PROTON_PROFILER_GRAPH_H_
#define PROTON_PROFILER_GRAPH_H_

#include "Context/Context.h"
#include "Data/Data.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <unordered_map>
#include <utility>

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
  static constexpr const char *metricTag = "<metric>";
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

    bool operator<(const NodeState &other) const {
      return nodeId < other.nodeId;
    }
  };
  using NodeIdToStateMap = std::map<uint64_t, NodeState>;
  // Precomputed per-Data launch links maintained on graph node
  // create/clone/destroy callbacks.
  // data -> (static_entry_id -> graph-node metadata pointers)
  std::map<Data *, std::unordered_map<size_t, std::set<NodeState *>>>
      dataToEntryIdToNodeStates;
  // Mapping from node id to node state, has to be ordered based on node id
  // which is the order of node creation.
  NodeIdToStateMap nodeIdToState;
  struct MetricNodeState {
    uint64_t seqId{};
    uint64_t metricId{};
    size_t numWords{};
    // Attributing flexible metrics written by metric kernels.
    std::map<Data *, size_t> dataToEntryId;
  };
  // Metric nodes and their CPU-side metric state, ordered by node id.
  std::map<uint64_t, MetricNodeState> metricNodeIdToState;
  // Maps the sequence id captured in the metric record back to the graph node
  // id.
  std::unordered_map<uint64_t, uint64_t> metricSeqIdToNodeId;
  // If the graph is launched after profiling started,
  // we need to throw an error and this error is only thrown once
  bool captureStatusChecked{};
  // Total number of uint64 words written by all metric nodes in this graph.
  size_t numMetricWords{};
};

struct PendingGraphQueue {
  struct MetricNodeState {
    uint64_t metricId{};
    std::map<Data *, DataEntry> dataToEntry;
  };
  using SeqIdToStateMap = std::map<uint64_t, MetricNodeState>;

  SeqIdToStateMap seqIdToState;
  // The start buffer offset in the metric buffer for this queue
  size_t startBufferOffset{};
  // Total number of uint64 words written by all nodes in this queue
  size_t numWords{};

  explicit PendingGraphQueue(size_t startBufferOffset)
      : startBufferOffset(startBufferOffset) {}

  void push(size_t numWords, SeqIdToStateMap &&seqIdToState) {
    this->seqIdToState.merge(std::move(seqIdToState));
    this->numWords += numWords;
  }
};

class PendingGraphPool {
public:
  explicit PendingGraphPool(MetricBuffer *metricBuffer)
      : metricBuffer(metricBuffer), runtime(metricBuffer->getRuntime()) {}

  void push(size_t phase, size_t numWords,
            PendingGraphQueue::SeqIdToStateMap &&seqIdToState);

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
