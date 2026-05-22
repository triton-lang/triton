#include "Data/TreeData.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Device.h"
#include "DeviceType.h"
#include "Profiler/Graph.h"
#include "Utility/Errors.h"
#include "Utility/MsgPackWriter.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace proton {

namespace {

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

} // namespace

class TreeData::Tree {
public:
  struct TreeNode : public Context {
    inline static const size_t RootId = 0;
    inline static const size_t DummyId = std::numeric_limits<size_t>::max();

    struct ChildEntry {
      std::string_view name;
      size_t id = DummyId;
    };

    // Keep a linear child list for small fanouts to avoid hash table overhead.
    // Build the lookup index only once repeated child scans become expensive.
    static constexpr size_t kChildIndexThreshold = 8;

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
      if (childIndex) {
        childIndex->emplace(childName, id);
        return;
      }
      if (children.size() > kChildIndexThreshold) {
        childIndex = std::make_unique<std::unordered_map<std::string_view,
                                                         size_t>>();
        childIndex->reserve(children.size());
        for (const auto &child : children) {
          childIndex->emplace(child.name, child.id);
        }
      }
    }

    size_t findChild(std::string_view childName) const {
      if (childIndex) {
        auto it = childIndex->find(childName);
        return it != childIndex->end() ? it->second : DummyId;
      }
      for (const auto &child : children) {
        if (child.name == childName) {
          return child.id;
        }
      }
      return DummyId;
    }

    size_t parentId = DummyId;
    size_t id = DummyId;
    std::vector<ChildEntry> children = {};
    std::unique_ptr<std::unordered_map<std::string_view, size_t>> childIndex =
        {};
    // Direct and linked metrics associated with this tree node.
    DataEntry::MetricSet metricSet{};
    friend class Tree;
  };

  Tree() {
    treeNodes.emplace_back(TreeNode::RootId, TreeNode::RootId, "ROOT");
  }

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

  Tree structure() const {
    Tree cloned;
    cloned.treeNodes.clear();
    cloned.nextContextId = nextContextId;

    for (const auto &node : treeNodes) {
      cloned.treeNodes.emplace_back(node.id, node.parentId, node);
    }

    for (const auto &node : treeNodes) {
      auto &clonedNode = cloned.getNode(node.id);
      clonedNode.children.reserve(node.children.size());
      for (const auto &child : node.children) {
        clonedNode.addChild(cloned.getNode(child.id).name, child.id);
      }
    }

    return cloned;
  }

private:
  size_t nextContextId = TreeNode::RootId + 1;
  // Node ids are dense and assigned sequentially, so index lookup is enough.
  std::deque<TreeNode> treeNodes;
};

json TreeData::buildHatchetJson(TreeData::Tree *tree,
                                TreeData::Tree *virtualTree) const {
  std::vector<json *> jsonNodes(tree->size(), nullptr);
  json output = json::array();
  output.push_back(json::object());
  jsonNodes[TreeData::Tree::TreeNode::RootId] = &(output.back());
  MetricSummary metricSummary;
  const auto &virtualRootNode = virtualTree->getNode(Tree::TreeNode::RootId);
  auto appendMetrics = [&](json &metricsJson,
                           const std::map<MetricKind, std::unique_ptr<Metric>>
                               &metrics) {
    metricSummary.observeMetrics(metrics);
    for (const auto &[metricKind, metric] : metrics) {
      if (metricKind == MetricKind::Kernel) {
        auto *kernelMetric = static_cast<KernelMetric *>(metric.get());
        uint64_t duration =
            std::get<uint64_t>(kernelMetric->getValue(KernelMetric::Duration));
        uint64_t invocations = std::get<uint64_t>(
            kernelMetric->getValue(KernelMetric::Invocations));
        uint64_t deviceId =
            std::get<uint64_t>(kernelMetric->getValue(KernelMetric::DeviceId));
        uint64_t deviceType = std::get<uint64_t>(
            kernelMetric->getValue(KernelMetric::DeviceType));
        metricSummary.hasKernelMetric = true;
        metricSummary.updateDeviceIdMask(deviceType, deviceId);
        const auto &deviceTypeName =
            getDeviceTypeString(static_cast<DeviceType>(deviceType));
        const auto deviceIdStr = std::to_string(deviceId);

        metricsJson[KernelMetric::getValueName(KernelMetric::Duration)] =
            duration;
        metricsJson[KernelMetric::getValueName(KernelMetric::Invocations)] =
            invocations;
        metricsJson[KernelMetric::getValueName(KernelMetric::DeviceId)] =
            deviceIdStr;
        metricsJson[KernelMetric::getValueName(KernelMetric::DeviceType)] =
            deviceTypeName;
      } else if (metricKind == MetricKind::PCSampling) {
        auto *pcSamplingMetric = static_cast<PCSamplingMetric *>(metric.get());
        for (size_t i = 0; i < PCSamplingMetric::Count; i++) {
          const auto valueName = PCSamplingMetric::getValueName(
              static_cast<PCSamplingMetric::PCSamplingMetricKind>(i));
          std::visit(
              [&](auto &&value) { metricsJson[valueName] = value; },
              pcSamplingMetric->getValues()[i]);
        }
      } else if (metricKind == MetricKind::Cycle) {
        auto *cycleMetric = static_cast<CycleMetric *>(metric.get());
        uint64_t duration =
            std::get<uint64_t>(cycleMetric->getValue(CycleMetric::Duration));
        double normalizedDuration = std::get<double>(
            cycleMetric->getValue(CycleMetric::NormalizedDuration));
        uint64_t deviceId =
            std::get<uint64_t>(cycleMetric->getValue(CycleMetric::DeviceId));
        uint64_t deviceType =
            std::get<uint64_t>(cycleMetric->getValue(CycleMetric::DeviceType));
        const auto deviceIdStr = std::to_string(deviceId);
        const auto deviceTypeStr = std::to_string(deviceType);

        metricsJson[CycleMetric::getValueName(CycleMetric::Duration)] =
            duration;
        metricsJson[CycleMetric::getValueName(
            CycleMetric::NormalizedDuration)] = normalizedDuration;
        metricsJson[CycleMetric::getValueName(CycleMetric::DeviceId)] =
            deviceIdStr;
        metricsJson[CycleMetric::getValueName(CycleMetric::DeviceType)] =
            deviceTypeStr;
      } else if (metricKind == MetricKind::Flexible) {
        // Flexible metrics are handled in a different way
      } else {
        throw makeLogicError("MetricKind not supported");
      }
    }
  };
  auto appendFlexibleMetrics =
      [&](json &metricsJson,
          const std::map<std::string, FlexibleMetric> &flexibleMetrics) {
        for (const auto &[_, flexibleMetric] : flexibleMetrics) {
          const auto valueName = flexibleMetric.getValueName(0);
          std::visit(
              [&](auto &&v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, uint64_t> ||
                              std::is_same_v<T, int64_t> ||
                              std::is_same_v<T, double> ||
                              std::is_same_v<T, std::string>) {
                  metricsJson[valueName] = v;
                } else if constexpr (std::is_same_v<T, std::vector<uint64_t>> ||
                                     std::is_same_v<T, std::vector<int64_t>> ||
                                     std::is_same_v<T, std::vector<double>>) {
                  auto &arr = metricsJson[valueName] = json::array();
                  arr.template get_ref<json::array_t &>().reserve(v.size());
                  for (const auto &value : v) {
                    arr.push_back(value);
                  }
                } else {
                  static_assert(sizeof(T) == 0, "Unsupported MetricValueType");
                }
              },
              flexibleMetric.getValues()[0]);
        }
      };
  auto appendPromotedFlexibleMetrics =
      [&](const auto &children, json &metricsJson,
          const DataEntry::LinkedFlexibleMetricMap &linkedFlexibleMetrics) {
        for (const auto &child : children) {
          auto it = linkedFlexibleMetrics.find(child.id);
          if (it != linkedFlexibleMetrics.end()) {
            appendFlexibleMetrics(metricsJson, it->second);
          }
        }
      };
  tree->template walk<TreeData::Tree::WalkPolicy::PreOrder>(
      [&](TreeData::Tree::TreeNode &treeNode) {
        const auto &contextName = treeNode.name;
        auto contextId = treeNode.id;
        json *jsonNode = jsonNodes[contextId];
        (*jsonNode)["frame"] = {{"name", GraphState::getDisplayName(contextName)},
                                {"type", "function"}};
        (*jsonNode)["metrics"] = json::object();
        auto &metricsJson = (*jsonNode)["metrics"];
        appendMetrics(metricsJson, treeNode.metricSet.metrics);
        appendFlexibleMetrics(metricsJson, treeNode.metricSet.flexibleMetrics);
        auto &childrenArray = (*jsonNode)["children"];
        childrenArray = json::array();
        const bool hasLinkedTargets =
            !treeNode.metricSet.linkedMetrics.empty() ||
            !treeNode.metricSet.linkedFlexibleMetrics.empty();
        childrenArray.get_ref<json::array_t &>().reserve(
            treeNode.children.size() +
            (hasLinkedTargets ? virtualRootNode.children.size() : 0));
        for (const auto &child : treeNode.children) {
          childrenArray.push_back(json::object());
          jsonNodes[child.id] = &childrenArray.back();
        }
        if (!hasLinkedTargets) {
          return;
        }
        std::set<size_t> linkedVirtualNodeIds;
        auto addLinkedVirtualAncestors = [&](size_t virtualNodeId,
                                             bool includeSelf) {
          if (!includeSelf) {
            virtualNodeId = virtualTree->getNode(virtualNodeId).parentId;
          }
          while (virtualNodeId != Tree::TreeNode::RootId) {
            linkedVirtualNodeIds.insert(virtualNodeId);
            virtualNodeId = virtualTree->getNode(virtualNodeId).parentId;
          }
        };
        for (const auto &[virtualNodeId, _] : treeNode.metricSet.linkedMetrics) {
          addLinkedVirtualAncestors(virtualNodeId, /*includeSelf=*/true);
        }
        for (const auto &[virtualNodeId, _] :
             treeNode.metricSet.linkedFlexibleMetrics) {
          addLinkedVirtualAncestors(virtualNodeId, /*includeSelf=*/false);
        }
        std::function<void(size_t, json &, json &)> appendLinkedVirtualNode =
            [&](size_t virtualNodeId, json &outNode,
                json &parentMetricsJson) {
              const auto &virtualNode = virtualTree->getNode(virtualNodeId);
              const auto metricsIt =
                  treeNode.metricSet.linkedMetrics.find(virtualNodeId);
              const auto flexibleIt =
                  treeNode.metricSet.linkedFlexibleMetrics.find(virtualNodeId);
              outNode = json::object();
              outNode["frame"] = {{"name", GraphState::getDisplayName(
                                               virtualNode.name)},
                                  {"type", "function"}};
              outNode["metrics"] = json::object();
              if (metricsIt != treeNode.metricSet.linkedMetrics.end()) {
                appendMetrics(outNode["metrics"], metricsIt->second);
              }
              // Linked flexible metrics are only attached to <metric_node>
              // children, so they always belong on the parent frame.
              if (flexibleIt !=
                  treeNode.metricSet.linkedFlexibleMetrics.end()) {
                appendFlexibleMetrics(parentMetricsJson, flexibleIt->second);
              }
              appendPromotedFlexibleMetrics(
                  virtualNode.children, outNode["metrics"],
                  treeNode.metricSet.linkedFlexibleMetrics);
              outNode["children"] = json::array();
              auto &linkedChildren = outNode["children"];
              linkedChildren.get_ref<json::array_t &>().reserve(
                  virtualNode.children.size());
              for (const auto &child : virtualNode.children) {
                if (linkedVirtualNodeIds.find(child.id) !=
                    linkedVirtualNodeIds.end()) {
                  linkedChildren.push_back(json::object());
                  appendLinkedVirtualNode(child.id, linkedChildren.back(),
                                          outNode["metrics"]);
                }
              }
            };

        for (const auto &virtualChild : virtualRootNode.children) {
          if (linkedVirtualNodeIds.find(virtualChild.id) !=
              linkedVirtualNodeIds.end()) {
            childrenArray.push_back(json::object());
            appendLinkedVirtualNode(virtualChild.id, childrenArray.back(),
                                    metricsJson);
          }
        }
      });

  if (metricSummary.hasKernelMetric) {
    output[TreeData::Tree::TreeNode::RootId]["metrics"]
          [KernelMetric::getValueName(KernelMetric::Invocations)] = 0;
    output[TreeData::Tree::TreeNode::RootId]["metrics"]
          [KernelMetric::getValueName(KernelMetric::Duration)] = 0;
  }
  if (metricSummary.hasCycleMetric) {
    output[TreeData::Tree::TreeNode::RootId]["metrics"]
          [CycleMetric::getValueName(CycleMetric::Duration)] = 0;
    output[TreeData::Tree::TreeNode::RootId]["metrics"]
          [CycleMetric::getValueName(CycleMetric::NormalizedDuration)] = 0;
  }
  if (metricSummary.hasPCSamplingMetric) {
    for (size_t i = 0; i < PCSamplingMetric::Count; i++) {
      const auto valueName = PCSamplingMetric::getValueName(
          static_cast<PCSamplingMetric::PCSamplingMetricKind>(i));
      output[TreeData::Tree::TreeNode::RootId]["metrics"]
            [valueName] = 0;
    }
  }

  output.push_back(json::object());
  auto &deviceJson = output.back();
  for (size_t deviceType = 0;
       deviceType < static_cast<size_t>(DeviceType::COUNT); ++deviceType) {
    auto mask = metricSummary.deviceIdMasks[deviceType];
    if (mask == 0) {
      continue;
    }

    const auto &deviceTypeName =
        getDeviceTypeString(static_cast<DeviceType>(deviceType));
    deviceJson[deviceTypeName] = json::object();

    for (uint64_t deviceId = 0; deviceId < kMaxRegisteredDeviceIds;
         ++deviceId) {
      if ((mask & (1u << static_cast<uint32_t>(deviceId))) == 0) {
        continue;
      }
      Device device = getDevice(static_cast<DeviceType>(deviceType), deviceId);
      deviceJson[deviceTypeName][std::to_string(deviceId)] = {
          {"clock_rate", device.clockRate},
          {"memory_clock_rate", device.memoryClockRate},
          {"bus_width", device.busWidth},
          {"arch", device.arch},
          {"num_sms", device.numSms}};
    }

    if (deviceJson[deviceTypeName].empty()) {
      deviceJson.erase(deviceTypeName);
    }
  }
  return output;
}

std::vector<uint8_t>
TreeData::buildHatchetMsgPack(TreeData::Tree *tree,
                              TreeData::Tree *virtualTree) const {
  MsgPackWriter writer;
  writer.reserve(16 * 1024 * 1024); // 16 MB

  MetricSummary metricSummary;
  metricSummary.hasKernelMetric = true;
  const std::map<MetricKind, std::unique_ptr<Metric>> emptyMetrics;
  const auto &virtualRootNode = virtualTree->getNode(Tree::TreeNode::RootId);

  constexpr uint32_t kernelInclusiveCount = 2;
  constexpr uint32_t kernelTotalCount = 4;
  constexpr uint32_t cycleInclusiveCount = 2;
  constexpr uint32_t cycleTotalCount = 4;

  auto countMetricEntries =
      [&](const std::map<MetricKind, std::unique_ptr<Metric>> &metrics,
          bool isRoot) -> uint32_t {
    uint32_t metricEntries = 0;
    for (const auto &[metricKind, _] : metrics) {
      if (metricKind == MetricKind::Kernel) {
        metricEntries += isRoot ? kernelInclusiveCount : kernelTotalCount;
      } else if (metricKind == MetricKind::PCSampling) {
        metricEntries += PCSamplingMetric::Count;
      } else if (metricKind == MetricKind::Cycle) {
        metricEntries += isRoot ? cycleInclusiveCount : cycleTotalCount;
      } else if (metricKind == MetricKind::Flexible) {
        // Flexible metrics are tracked in a separate map.
      } else {
        throw makeLogicError("MetricKind not supported");
      }
    }
    if (isRoot) {
      if (metricSummary.hasKernelMetric &&
          metrics.find(MetricKind::Kernel) == metrics.end()) {
        metricEntries += kernelInclusiveCount;
      }
      if (metricSummary.hasPCSamplingMetric &&
          metrics.find(MetricKind::PCSampling) == metrics.end()) {
        metricEntries += PCSamplingMetric::Count;
      }
      if (metricSummary.hasCycleMetric &&
          metrics.find(MetricKind::Cycle) == metrics.end()) {
        metricEntries += cycleInclusiveCount;
      }
    }
    return metricEntries;
  };
  auto packKernelMetricValues = [&](const KernelMetric *kernelMetric) {
    uint64_t duration =
        std::get<uint64_t>(kernelMetric->getValue(KernelMetric::Duration));
    uint64_t invocations = std::get<uint64_t>(
        kernelMetric->getValue(KernelMetric::Invocations));
    uint64_t deviceId =
        std::get<uint64_t>(kernelMetric->getValue(KernelMetric::DeviceId));
    uint64_t deviceType =
        std::get<uint64_t>(kernelMetric->getValue(KernelMetric::DeviceType));
    metricSummary.updateDeviceIdMask(deviceType, deviceId);
    const auto &deviceTypeName =
        getDeviceTypeString(static_cast<DeviceType>(deviceType));
    writer.packStr(KernelMetric::getValueName(KernelMetric::Duration));
    writer.packUInt(duration);
    writer.packStr(KernelMetric::getValueName(KernelMetric::Invocations));
    writer.packUInt(invocations);
    writer.packStr(KernelMetric::getValueName(KernelMetric::DeviceId));
    writer.packUIntString(deviceId);
    writer.packStr(KernelMetric::getValueName(KernelMetric::DeviceType));
    writer.packStr(deviceTypeName);
  };

  auto packMetrics = [&](const std::map<MetricKind, std::unique_ptr<Metric>>
                             &metrics,
                         bool isRoot) {
    for (const auto &[metricKind, metric] : metrics) {
      if (metricKind == MetricKind::Kernel) {
        if (isRoot) {
          writer.packStr(KernelMetric::getValueName(KernelMetric::Duration));
          writer.packUInt(0);
          writer.packStr(KernelMetric::getValueName(KernelMetric::Invocations));
          writer.packUInt(0);
          continue;
        }

        packKernelMetricValues(static_cast<KernelMetric *>(metric.get()));
      } else if (metricKind == MetricKind::PCSampling) {
        auto *pcSamplingMetric = static_cast<PCSamplingMetric *>(metric.get());
        for (size_t i = 0; i < PCSamplingMetric::Count; i++) {
          const auto valueName = pcSamplingMetric->getValueName(i);
          writer.packStr(valueName);
          if (isRoot) {
            writer.packUInt(0);
          } else {
            writer.packUInt(
                std::get<uint64_t>(pcSamplingMetric->getValues()[i]));
          }
        }
      } else if (metricKind == MetricKind::Cycle) {
        if (isRoot) {
          writer.packStr(CycleMetric::getValueName(CycleMetric::Duration));
          writer.packUInt(0);
          writer.packStr(
              CycleMetric::getValueName(CycleMetric::NormalizedDuration));
          writer.packUInt(0);
          continue;
        }

        auto *cycleMetric = static_cast<CycleMetric *>(metric.get());
        uint64_t duration =
            std::get<uint64_t>(cycleMetric->getValue(CycleMetric::Duration));
        double normalizedDuration = std::get<double>(
            cycleMetric->getValue(CycleMetric::NormalizedDuration));
        uint64_t deviceId =
            std::get<uint64_t>(cycleMetric->getValue(CycleMetric::DeviceId));
        uint64_t deviceType =
            std::get<uint64_t>(cycleMetric->getValue(CycleMetric::DeviceType));
        metricSummary.updateDeviceIdMask(deviceType, deviceId);

        writer.packStr(CycleMetric::getValueName(CycleMetric::Duration));
        writer.packUInt(duration);
        writer.packStr(
            CycleMetric::getValueName(CycleMetric::NormalizedDuration));
        writer.packDouble(normalizedDuration);
        writer.packStr(CycleMetric::getValueName(CycleMetric::DeviceId));
        writer.packUIntString(deviceId);
        writer.packStr(CycleMetric::getValueName(CycleMetric::DeviceType));
        writer.packUIntString(deviceType);
      } else {
        throw makeLogicError("MetricKind not supported");
      }
    }
    if (isRoot) {
      if (metricSummary.hasKernelMetric &&
          metrics.find(MetricKind::Kernel) == metrics.end()) {
        writer.packStr(KernelMetric::getValueName(KernelMetric::Duration));
        writer.packUInt(0);
        writer.packStr(KernelMetric::getValueName(KernelMetric::Invocations));
        writer.packUInt(0);
      }
      if (metricSummary.hasPCSamplingMetric &&
          metrics.find(MetricKind::PCSampling) == metrics.end()) {
        PCSamplingMetric pcSamplingMetric;
        for (size_t i = 0; i < PCSamplingMetric::Count; i++) {
          const auto valueName = pcSamplingMetric.getValueName(i);
          writer.packStr(valueName);
          writer.packUInt(0);
        }
      }
      if (metricSummary.hasCycleMetric &&
          metrics.find(MetricKind::Cycle) == metrics.end()) {
        writer.packStr(CycleMetric::getValueName(CycleMetric::Duration));
        writer.packUInt(0);
        writer.packStr(
            CycleMetric::getValueName(CycleMetric::NormalizedDuration));
        writer.packUInt(0);
      }
    }
  };
  auto packFlexibleMetrics =
      [&](const std::map<std::string, FlexibleMetric> &flexibleMetrics) {
        for (const auto &[_, flexibleMetric] : flexibleMetrics) {
          const auto valueName = flexibleMetric.getValueName(0);
          writer.packStr(valueName);
          std::visit(
              [&](auto &&v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, uint64_t>) {
                  writer.packUInt(v);
                } else if constexpr (std::is_same_v<T, int64_t>) {
                  writer.packInt(v);
                } else if constexpr (std::is_same_v<T, double>) {
                  writer.packDouble(v);
                } else if constexpr (std::is_same_v<T, std::string>) {
                  writer.packStr(v);
                } else if constexpr (std::is_same_v<T, std::vector<uint64_t>>) {
                  writer.packArray(static_cast<uint32_t>(v.size()));
                  for (auto value : v) {
                    writer.packUInt(value);
                  }
                } else if constexpr (std::is_same_v<T,
                                                    std::vector<int64_t>>) {
                  writer.packArray(static_cast<uint32_t>(v.size()));
                  for (auto value : v) {
                    writer.packInt(value);
                  }
                } else if constexpr (std::is_same_v<T,
                                                    std::vector<double>>) {
                  writer.packArray(static_cast<uint32_t>(v.size()));
                  for (auto value : v) {
                    writer.packDouble(value);
                  }
                } else {
                  static_assert(sizeof(T) == 0, "Unsupported MetricValueType");
                }
              },
              flexibleMetric.getValues()[0]);
        }
      };
  auto countPromotedFlexibleMetricEntries =
      [&](const auto &children,
          const DataEntry::LinkedFlexibleMetricMap &linkedFlexibleMetrics)
      -> uint32_t {
    uint32_t metricEntries = 0;
    for (const auto &child : children) {
      auto it = linkedFlexibleMetrics.find(child.id);
      if (it != linkedFlexibleMetrics.end()) {
        metricEntries += static_cast<uint32_t>(it->second.size());
      }
    }
    return metricEntries;
  };
  auto packPromotedFlexibleMetrics =
      [&](const auto &children,
          const DataEntry::LinkedFlexibleMetricMap &linkedFlexibleMetrics) {
        for (const auto &child : children) {
          auto it = linkedFlexibleMetrics.find(child.id);
          if (it != linkedFlexibleMetrics.end()) {
            packFlexibleMetrics(it->second);
          }
        }
      };
  auto packNode = [&](auto &&packNode, TreeData::Tree::TreeNode &treeNode)
      -> void {
        writer.packMap(3);

        writer.packFixStrLiteral("frame");
        writer.packMap(2);
        writer.packFixStrLiteral("name");
        writer.packStr(GraphState::getDisplayName(treeNode.name));
        writer.packFixStrLiteral("type");
        writer.packFixStrLiteral("function");

        writer.packFixStrLiteral("metrics");
        const bool isRoot = treeNode.id == TreeData::Tree::TreeNode::RootId;
        writer.packMap(
            countMetricEntries(treeNode.metricSet.metrics, isRoot) +
            static_cast<uint32_t>(treeNode.metricSet.flexibleMetrics.size()) +
            countPromotedFlexibleMetricEntries(
                virtualRootNode.children,
                treeNode.metricSet.linkedFlexibleMetrics));
        packMetrics(treeNode.metricSet.metrics, isRoot);
        packFlexibleMetrics(treeNode.metricSet.flexibleMetrics);
        packPromotedFlexibleMetrics(virtualRootNode.children,
                                    treeNode.metricSet.linkedFlexibleMetrics);
        const bool hasLinkedTargets =
            !treeNode.metricSet.linkedMetrics.empty() ||
            !treeNode.metricSet.linkedFlexibleMetrics.empty();
        std::set<size_t> linkedVirtualNodeIds;
        if (hasLinkedTargets) {
          auto addLinkedVirtualAncestors = [&](size_t virtualNodeId,
                                               bool includeSelf) {
            if (!includeSelf) {
              virtualNodeId = virtualTree->getNode(virtualNodeId).parentId;
            }
            while (virtualNodeId != Tree::TreeNode::RootId) {
              linkedVirtualNodeIds.insert(virtualNodeId);
              virtualNodeId = virtualTree->getNode(virtualNodeId).parentId;
            }
          };
          for (const auto &[virtualNodeId, _] :
               treeNode.metricSet.linkedMetrics) {
            addLinkedVirtualAncestors(virtualNodeId, /*includeSelf=*/true);
          }
          for (const auto &[virtualNodeId, _] :
               treeNode.metricSet.linkedFlexibleMetrics) {
            addLinkedVirtualAncestors(virtualNodeId, /*includeSelf=*/false);
          }
        }

        auto countLinkedVirtualChildren =
            [&](const auto &children) -> uint32_t {
          uint32_t childCount = 0;
          for (const auto &child : children) {
            if (linkedVirtualNodeIds.find(child.id) !=
                linkedVirtualNodeIds.end()) {
              ++childCount;
            }
          }
          return childCount;
        };

        auto packLinkedVirtualNode = [&](auto &&packLinkedVirtualNode,
                                         size_t virtualNodeId) -> void {
              const auto &virtualNode = virtualTree->getNode(virtualNodeId);
              const KernelMetric *onlyKernelMetric = nullptr;
              const auto onlyMetricIt =
                  treeNode.metricSet.linkedMetrics.find(virtualNodeId);
              if (onlyMetricIt != treeNode.metricSet.linkedMetrics.end() &&
                  onlyMetricIt->second.size() == 1 &&
                  onlyMetricIt->second.begin()->first == MetricKind::Kernel) {
                onlyKernelMetric = static_cast<KernelMetric *>(
                    onlyMetricIt->second.begin()->second.get());
              }
              if (virtualNode.children.empty() &&
                  onlyKernelMetric != nullptr) {
                writer.packMap(3);

                writer.packFixStrLiteral("frame");
                writer.packMap(2);
                writer.packFixStrLiteral("name");
                writer.packStr(GraphState::getDisplayName(virtualNode.name));
                writer.packFixStrLiteral("type");
                writer.packFixStrLiteral("function");

                writer.packFixStrLiteral("metrics");
                writer.packMap(kernelTotalCount);
                packKernelMetricValues(onlyKernelMetric);

                writer.packFixStrLiteral("children");
                writer.packArray(0);
                return;
              }

              writer.packMap(3);

              writer.packFixStrLiteral("frame");
              writer.packMap(2);
              writer.packFixStrLiteral("name");
              writer.packStr(GraphState::getDisplayName(virtualNode.name));
              writer.packFixStrLiteral("type");
              writer.packFixStrLiteral("function");

              writer.packFixStrLiteral("metrics");
              const auto metricsIt =
                  treeNode.metricSet.linkedMetrics.find(virtualNodeId);
              const auto &linkedMetrics =
                  (metricsIt != treeNode.metricSet.linkedMetrics.end())
                      ? metricsIt->second
                      : emptyMetrics;
              writer.packMap(
                  countMetricEntries(linkedMetrics, /*isRoot=*/false) +
                  countPromotedFlexibleMetricEntries(
                      virtualNode.children,
                      treeNode.metricSet.linkedFlexibleMetrics));
              packMetrics(linkedMetrics, /*isRoot=*/false);
              // Linked flexible metrics are only attached to <metric_node>
              // children, so they are always packed into the parent frame.
              packPromotedFlexibleMetrics(
                  virtualNode.children,
                  treeNode.metricSet.linkedFlexibleMetrics);

              writer.packFixStrLiteral("children");
              writer.packArray(
                  countLinkedVirtualChildren(virtualNode.children));
              for (const auto &child : virtualNode.children) {
                if (linkedVirtualNodeIds.find(child.id) !=
                    linkedVirtualNodeIds.end()) {
                  packLinkedVirtualNode(packLinkedVirtualNode, child.id);
                }
              }
            };

        uint32_t linkedChildCount =
            hasLinkedTargets
                ? countLinkedVirtualChildren(virtualRootNode.children)
                : 0;
        writer.packFixStrLiteral("children");
        writer.packArray(static_cast<uint32_t>(treeNode.children.size()) +
                         linkedChildCount);
        for (const auto &child : treeNode.children) {
          packNode(packNode, tree->getNode(child.id));
        }
        if (hasLinkedTargets) {
          for (const auto &virtualChild : virtualRootNode.children) {
            if (linkedVirtualNodeIds.find(virtualChild.id) !=
                linkedVirtualNodeIds.end()) {
              packLinkedVirtualNode(packLinkedVirtualNode, virtualChild.id);
            }
          }
        }
      };

  // Hatchet format: [tree, device_metadata]. Always emit 2 elements to match
  // the JSON serializer, even if device_metadata is empty.
  writer.packArray(2);
  packNode(packNode, tree->getNode(TreeData::Tree::TreeNode::RootId));

  uint32_t deviceTypeEntries = 0;
  for (size_t deviceType = 0;
       deviceType < static_cast<size_t>(DeviceType::COUNT); ++deviceType) {
    if (metricSummary.deviceIdMasks[deviceType] != 0) {
      ++deviceTypeEntries;
    }
  }

  writer.packMap(deviceTypeEntries);
  for (size_t deviceType = 0;
       deviceType < static_cast<size_t>(DeviceType::COUNT); ++deviceType) {
    auto mask = metricSummary.deviceIdMasks[deviceType];
    if (mask == 0) {
      continue;
    }

    const auto &deviceTypeName =
        getDeviceTypeString(static_cast<DeviceType>(deviceType));
    writer.packStr(deviceTypeName);

    uint32_t deviceIdEntries = 0;
    for (auto remaining = mask; remaining != 0; remaining &= (remaining - 1)) {
      ++deviceIdEntries;
    }
    writer.packMap(deviceIdEntries);
    for (uint64_t deviceId = 0; deviceId < kMaxRegisteredDeviceIds;
         ++deviceId) {
      if ((mask & (1u << static_cast<uint32_t>(deviceId))) == 0) {
        continue;
      }
      Device device = getDevice(static_cast<DeviceType>(deviceType), deviceId);
      writer.packStr(std::to_string(deviceId));
      writer.packMap(5);
      writer.packFixStrLiteral("clock_rate");
      writer.packUInt(device.clockRate);
      writer.packFixStrLiteral("memory_clock_rate");
      writer.packUInt(device.memoryClockRate);
      writer.packFixStrLiteral("bus_width");
      writer.packUInt(device.busWidth);
      writer.packFixStrLiteral("arch");
      writer.packStr(device.arch);
      writer.packFixStrLiteral("num_sms");
      writer.packUInt(device.numSms);
    }
  }

  return std::move(writer).take();
}

void TreeData::enterScope(const Scope &scope) {
  // enterOp and addMetric maybe called from different threads
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto *currentTree = currentPhasePtrAs<Tree>();
  std::vector<Context> contexts;
  if (contextSource != nullptr)
    contexts = contextSource->getContexts();
  else
    contexts.emplace_back(scope.name);
  auto contextId = currentTree->addNode(contexts);
  scopeIdToContextId[scope.scopeId] = contextId;
}

void TreeData::exitScope(const Scope &scope) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  scopeIdToContextId.erase(scope.scopeId);
}

DataEntry TreeData::addOp(size_t phase, size_t contextId,
                          const std::vector<Context> &contexts) {
  auto lock = lockIfCurrentOrVirtualPhase(phase);
  if (contextId == Data::kRootEntryId) {
    contextId = Tree::TreeNode::RootId;
  }
  auto *tree = phasePtrAs<Tree>(phase);
  auto newContextId = tree->addNode(contexts, contextId);
  auto &node = tree->getNode(newContextId);
  return DataEntry(newContextId, phase, node.metricSet);
}

void TreeData::addMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &metrics) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto *currentTree = currentPhasePtrAs<Tree>();
  auto contextId = scopeIdToContextId.at(scopeId);
  for (auto [metricName, metricValue] : metrics) {
    currentTree->upsertFlexibleMetric(contextId,
                                      FlexibleMetric(metricName, metricValue));
  }
}

void TreeData::dumpHatchet(std::ostream &os, size_t phase) const {
  treePhases.withPtr(phase, [&](Tree *tree) {
    treePhases.withPtr(Data::kVirtualPhase, [&](Tree *virtualTree) {
      auto output = buildHatchetJson(tree, virtualTree);
      os << std::endl << output.dump(4) << std::endl;
    });
  });
}

void TreeData::dumpHatchetMsgPack(std::ostream &os, size_t phase) const {
  treePhases.withPtr(phase, [&](Tree *tree) {
    treePhases.withPtr(Data::kVirtualPhase, [&](Tree *virtualTree) {
      auto msgPack = buildHatchetMsgPack(tree, virtualTree);
      os.write(reinterpret_cast<const char *>(msgPack.data()),
               static_cast<std::streamsize>(msgPack.size()));
    });
  });
}

std::string TreeData::toJsonString(size_t phase) const {
  return treePhases.withPtr(phase, [&](Tree *tree) {
    return treePhases.withPtr(Data::kVirtualPhase, [&](Tree *virtualTree) {
      return buildHatchetJson(tree, virtualTree).dump();
    });
  });
}

std::vector<uint8_t> TreeData::toMsgPack(size_t phase) const {
  return treePhases.withPtr(phase, [&](Tree *tree) {
    return treePhases.withPtr(Data::kVirtualPhase, [&](Tree *virtualTree) {
      return buildHatchetMsgPack(tree, virtualTree);
    });
  });
}

void TreeData::doDump(std::ostream &os, OutputFormat outputFormat,
                      size_t phase) const {
  if (outputFormat == OutputFormat::Hatchet) {
    dumpHatchet(os, phase);
  } else if (outputFormat == OutputFormat::HatchetMsgPack) {
    dumpHatchetMsgPack(os, phase);
  } else {
    throw makeInvalidArgument("Output format not supported");
  }
}

TreeData::TreeData(const std::string &path, ContextSource *contextSource)
    : Data(path, contextSource) {
  initPhaseStore(treePhases);
}

TreeData::~TreeData() {}

} // namespace proton
