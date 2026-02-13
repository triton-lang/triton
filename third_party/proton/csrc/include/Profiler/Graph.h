#ifndef PROTON_PROFILER_GRAPH_H_
#define PROTON_PROFILER_GRAPH_H_

#include "Context/Context.h"
#include "Data/Data.h"
#include "Utility/Table.h"

#include <cstddef>
#include <cstdint>
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
  // Capture tag to identify captured call paths
  static constexpr const char *captureTag = "<captured_at>";
  struct NodeState {
    // The entry id of the static entry associated with this node, which is
    // created at capture time and won't change for the same node id. This is
    // used to link the graph node to the captured call path in Data.
    std::vector<std::pair<Data *, size_t>> dataToEntryId;
    // Whether the node has a missing name, determined at capture time and
    // stable for the same node id.
    bool isMissingName{};
  };
  struct MetricNodeState {
    // The entry id of the static entry associated with this metric node.
    std::map<Data *, size_t> dataToEntryId;
    // Number of uint64 words written by this metric node in the metric buffer.
    size_t numWords{};
  };
  // Data objects that were enabled during graph capture.
  std::set<Data *> dataSet;
  // Mapping from node id to node state.
  // This table only grows capacity and never shrinks.
  using NodeStateTable = RangeTable<NodeState, uint64_t>;
  NodeStateTable nodeIdToState;
  // Mapping from metric node id to metric node state.
  // This table only grows capacity and never shrinks.
  using MetricNodeStateTable = RangeTable<MetricNodeState, uint64_t>;
  MetricNodeStateTable metricNodeIdToState;
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
    // Launch entries to receive graph metric updates.
    std::vector<DataEntry> graphLaunchEntries;
    // Ordered metric-node linked entries reconstructed at launch time.
    std::vector<std::map<Data *, size_t>> metricNodeDataToEntryIdQueue;
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
            const std::vector<DataEntry> &graphLaunchEntries,
            const std::vector<std::map<Data *, size_t>>
                &metricNodeDataToEntryIdQueue) {
    pendingGraphs.emplace_back(PendingGraph{numNodes, numWords,
                                            graphLaunchEntries,
                                            metricNodeDataToEntryIdQueue});
    this->numNodes += numNodes;
    this->numWords += numWords;
  }
};

class PendingGraphPool {
public:
  explicit PendingGraphPool(MetricBuffer *metricBuffer)
      : metricBuffer(metricBuffer), runtime(metricBuffer->getRuntime()) {}

  void push(size_t phase, const std::vector<DataEntry> &graphLaunchEntries,
            const std::vector<std::map<Data *, size_t>>
                &metricNodeDataToEntryIdQueue,
            size_t numNodes,
            size_t numWords);

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
