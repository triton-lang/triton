#ifndef PROTON_DATA_DUMP_TREE_DATA_DUMP_H_
#define PROTON_DATA_DUMP_TREE_DATA_DUMP_H_

#include "Data/Metric.h"
#include "Data/TreeData.h"
#include "DeviceType.h"
#include "Utility/Errors.h"

#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace proton {

namespace tree_data_dump {

constexpr size_t kMaxRegisteredDeviceIds = 32;

struct MetricSummary {
  // Whether we observed at least one kernel metric.
  bool hasKernelMetric = false;
  // Whether we observed at least one PC sampling metric.
  bool hasPCSamplingMetric = false;
  // Whether we observed at least one cycle metric.
  bool hasCycleMetric = false;
  // device_type -> bitmask of observed device ids.
  std::array<uint32_t, static_cast<size_t>(DeviceType::COUNT)> deviceIdMasks{};

  void updateDeviceIdMask(uint64_t deviceType, uint64_t deviceId) {
    if (deviceType >= static_cast<uint64_t>(DeviceType::COUNT)) {
      throw makeOutOfRange("Invalid deviceType " + std::to_string(deviceType));
    }
    if (deviceId >= kMaxRegisteredDeviceIds) {
      throw makeOutOfRange("DeviceId " + std::to_string(deviceId) +
                           " exceeds MaxRegisteredDeviceIds " +
                           std::to_string(kMaxRegisteredDeviceIds) +
                           " for deviceType " + std::to_string(deviceType));
    }
    deviceIdMasks[static_cast<size_t>(deviceType)] |=
        (1u << static_cast<uint32_t>(deviceId));
  }

  void
  observeMetrics(const std::map<MetricKind, std::unique_ptr<Metric>> &metrics) {
    for (const auto &[metricKind, metric] : metrics) {
      if (metricKind == MetricKind::Kernel) {
        hasKernelMetric = true;
        auto *kernelMetric = static_cast<KernelMetric *>(metric.get());
        uint64_t deviceId =
            std::get<uint64_t>(kernelMetric->getValue(KernelMetric::DeviceId));
        uint64_t deviceType = std::get<uint64_t>(
            kernelMetric->getValue(KernelMetric::DeviceType));
        updateDeviceIdMask(deviceType, deviceId);
      } else if (metricKind == MetricKind::PCSampling) {
        hasPCSamplingMetric = true;
      } else if (metricKind == MetricKind::Cycle) {
        hasCycleMetric = true;
        auto *cycleMetric = static_cast<CycleMetric *>(metric.get());
        uint64_t deviceId =
            std::get<uint64_t>(cycleMetric->getValue(CycleMetric::DeviceId));
        uint64_t deviceType =
            std::get<uint64_t>(cycleMetric->getValue(CycleMetric::DeviceType));
        updateDeviceIdMask(deviceType, deviceId);
      } else if (metricKind == MetricKind::Flexible) {
        // Flexible metrics are tracked in a separate map.
      } else {
        throw makeLogicError("MetricKind not supported");
      }
    }
  }
};

} // namespace tree_data_dump

class TreeData::Tree {
public:
  struct TreeNode : public Context {
    inline static const size_t RootId = 0;
    inline static const size_t DummyId = std::numeric_limits<size_t>::max();

    struct ChildEntry {
      std::string_view name;
      size_t id = DummyId;
    };

    TreeNode() = default;
    explicit TreeNode(size_t id, const std::string &name)
        : id(id), Context(name) {}
    TreeNode(size_t id, size_t parentId, const std::string &name)
        : id(id), parentId(parentId), Context(name) {}
    TreeNode(size_t id, size_t parentId, const Context &context)
        : Context(context), parentId(parentId), id(id) {}
    virtual ~TreeNode() = default;

    void addChild(std::string_view childName, size_t id) {
      children.push_back({childName, id});
      childIndex.emplace(childName, id);
    }

    size_t findChild(std::string_view childName) const {
      auto it = childIndex.find(childName);
      return it != childIndex.end() ? it->second : DummyId;
    }

    size_t parentId = DummyId;
    size_t id = DummyId;
    std::vector<ChildEntry> children = {};
    std::unordered_map<std::string_view, size_t> childIndex = {};
    // Direct and linked metrics associated with this tree node.
    DataEntry::MetricSet metricSet{};
    friend class Tree;
  };

  Tree() { treeNodes.emplace_back(TreeNode::RootId, TreeNode::RootId, "ROOT"); }

  size_t addNode(const std::vector<Context> &contexts, size_t parentId) {
    for (const auto &context : contexts) {
      parentId = addNode(context, parentId);
    }
    return parentId;
  }

  size_t addNode(const Context &context, size_t parentId) {
    auto &parent = getNode(parentId);
    std::string_view contextName = context.name;
    auto existingChildId = parent.findChild(contextName);
    if (existingChildId != TreeNode::DummyId)
      return existingChildId;
    auto id = nextContextId++;
    treeNodes.emplace_back(id, parentId, context);
    parent.addChild(treeNodes.back().name, id);
    return id;
  }

  size_t addNode(const std::vector<Context> &indices) {
    auto parentId = TreeNode::RootId;
    for (auto index : indices) {
      parentId = addNode(index, parentId);
    }
    return parentId;
  }

  TreeNode &getNode(size_t id) { return treeNodes.at(id); }

  void upsertFlexibleMetric(size_t contextId,
                            const FlexibleMetric &flexibleMetric) {
    auto &node = getNode(contextId);
    auto &flexibleMetrics = node.metricSet.flexibleMetrics;
    auto valueName = std::string(flexibleMetric.getValueName(0));
    auto it = flexibleMetrics.find(valueName);
    if (it == flexibleMetrics.end()) {
      flexibleMetrics.emplace(std::move(valueName), flexibleMetric);
    } else {
      it->second.updateMetric(flexibleMetric);
    }
  }

  enum class WalkPolicy { PreOrder, PostOrder };

  template <WalkPolicy walkPolicy, typename FnT> void walk(FnT &&fn) {
    if constexpr (walkPolicy == WalkPolicy::PreOrder) {
      walkPreOrder(TreeNode::RootId, fn);
    } else if constexpr (walkPolicy == WalkPolicy::PostOrder) {
      walkPostOrder(TreeNode::RootId, fn);
    }
  }

  template <typename FnT> void walkPreOrder(size_t contextId, FnT &&fn) {
    fn(getNode(contextId));
    for (const auto &child : getNode(contextId).children) {
      walkPreOrder(child.id, fn);
    }
  }

  size_t size() const { return treeNodes.size(); }

private:
  size_t nextContextId = TreeNode::RootId + 1;
  // Node ids are dense and assigned sequentially, so index lookup is enough.
  std::deque<TreeNode> treeNodes;
};

} // namespace proton

#endif // PROTON_DATA_DUMP_TREE_DATA_DUMP_H_
