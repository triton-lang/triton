#include "Data/TreeData.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Device.h"
#include "Utility/MsgPackWriter.h"
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace proton {

namespace {

const std::array<std::string, static_cast<size_t>(DeviceType::COUNT)>
    kDeviceTypeNames = []() {
      std::array<std::string, static_cast<size_t>(DeviceType::COUNT)> names;
      for (size_t i = 0; i < static_cast<size_t>(DeviceType::COUNT); ++i) {
        names[i] = getDeviceTypeString(static_cast<DeviceType>(i));
      }
      return names;
    }();

constexpr size_t kMaxRegisteredDeviceIds = 32;

struct MetricSummary {
  bool hasKernelMetric = false;
  bool hasPCSamplingMetric = false;
  bool hasCycleMetric = false;
  std::array<uint32_t, static_cast<size_t>(DeviceType::COUNT)> deviceIdMasks{};

  void updateDeviceIdMask(uint64_t deviceType, uint64_t deviceId) {
    if (deviceType >= static_cast<uint64_t>(DeviceType::COUNT)) {
      throw std::runtime_error("[PROTON] Invalid deviceType " +
                               std::to_string(deviceType));
    }
    if (deviceId >= kMaxRegisteredDeviceIds) {
      throw std::runtime_error("[PROTON] DeviceId " + std::to_string(deviceId) +
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
        throw std::runtime_error("MetricKind not supported");
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

    TreeNode() = default;
    explicit TreeNode(size_t id, const std::string &name)
        : id(id), Context(name) {}
    TreeNode(size_t id, size_t parentId, const std::string &name)
        : id(id), parentId(parentId), Context(name) {}
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
    std::map<MetricKind, std::unique_ptr<Metric>> metrics = {};
    std::map<std::string, FlexibleMetric> flexibleMetrics = {};
    std::map<size_t, std::map<MetricKind, std::unique_ptr<Metric>>>
        linkedTargetMetrics = {};
    std::map<size_t, std::map<std::string, FlexibleMetric>>
        linkedTargetFlexibleMetrics = {};
    friend class Tree;
  };

  Tree() {
    treeNodeMap.try_emplace(TreeNode::RootId, TreeNode::RootId,
                            TreeNode::RootId, "ROOT");
  }

  size_t addNode(const std::vector<Context> &contexts, size_t parentId) {
    for (const auto &context : contexts) {
      parentId = addNode(context, parentId);
    }
    return parentId;
  }

  size_t addNode(const Context &context, size_t parentId) {
    auto &parent = treeNodeMap.at(parentId);
    std::string_view contextName = context.name;
    auto existingChildId = parent.findChild(contextName);
    if (existingChildId != TreeNode::DummyId)
      return existingChildId;
    auto id = nextContextId++;
    auto [it, inserted] =
        treeNodeMap.try_emplace(id, id, parentId, context.name);
    parent.addChild(it->second.name, id);
    return id;
  }

  size_t addNode(const std::vector<Context> &indices) {
    auto parentId = TreeNode::RootId;
    for (auto index : indices) {
      parentId = addNode(index, parentId);
    }
    return parentId;
  }

  TreeNode &getNode(size_t id) { return treeNodeMap.at(id); }

  void upsertFlexibleMetric(size_t contextId,
                            const FlexibleMetric &flexibleMetric) {
    auto &node = treeNodeMap.at(contextId);
    auto it = node.flexibleMetrics.find(flexibleMetric.getValueName(0));
    if (it == node.flexibleMetrics.end()) {
      node.flexibleMetrics.emplace(flexibleMetric.getValueName(0),
                                   flexibleMetric);
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

  size_t size() const { return nextContextId; }

  Tree structure() const {
    Tree cloned;
    cloned.nextContextId = nextContextId;

    for (const auto &[id, node] : treeNodeMap) {
      cloned.treeNodeMap.try_emplace(id, id, node.parentId, node.name);
    }

    for (const auto &[id, node] : treeNodeMap) {
      auto &clonedNode = cloned.treeNodeMap.at(id);
      clonedNode.children.reserve(node.children.size());
      for (const auto &child : node.children) {
        clonedNode.addChild(cloned.treeNodeMap[child.id].name, child.id);
      }
    }

    return cloned;
  }

private:
  size_t nextContextId = TreeNode::RootId + 1;
  // tree node id -> tree node
  std::unordered_map<size_t, TreeNode> treeNodeMap;
};

json TreeData::buildHatchetJson(TreeData::Tree *tree,
                                TreeData::Tree *staticTree) const {
  std::vector<json *> jsonNodes(tree->size(), nullptr);
  json output = json::array();
  output.push_back(json::object());
  jsonNodes[TreeData::Tree::TreeNode::RootId] = &(output.back());
  MetricSummary metricSummary;
  const std::map<MetricKind, std::unique_ptr<Metric>> emptyMetrics;
  const std::map<std::string, FlexibleMetric> emptyFlexibleMetrics;
  const auto &staticRootNode = staticTree->getNode(Tree::TreeNode::RootId);
  auto appendMetrics =
      [&](json &metricsJson,
          const std::map<MetricKind, std::unique_ptr<Metric>> &metrics,
          const std::map<std::string, FlexibleMetric> &flexibleMetrics) {
        metricSummary.observeMetrics(metrics);
        for (const auto &[metricKind, metric] : metrics) {
          if (metricKind == MetricKind::Kernel) {
            auto *kernelMetric = static_cast<KernelMetric *>(metric.get());
            uint64_t duration = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::Duration));
            uint64_t invocations = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::Invocations));
            uint64_t deviceId = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::DeviceId));
            uint64_t deviceType = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::DeviceType));
            const auto &deviceTypeName =
                kDeviceTypeNames[static_cast<size_t>(deviceType)];
            const auto &durationName =
                kernelMetric->getValueName(KernelMetric::Duration);
            const auto &invocationsName =
                kernelMetric->getValueName(KernelMetric::Invocations);
            const auto &deviceIdName =
                kernelMetric->getValueName(KernelMetric::DeviceId);
            const auto &deviceTypeNameKey =
                kernelMetric->getValueName(KernelMetric::DeviceType);
            const auto deviceIdStr = std::to_string(deviceId);

            metricsJson[durationName] = duration;
            metricsJson[invocationsName] = invocations;
            metricsJson[deviceIdName] = deviceIdStr;
            metricsJson[deviceTypeNameKey] = deviceTypeName;
          } else if (metricKind == MetricKind::PCSampling) {
            auto *pcSamplingMetric =
                static_cast<PCSamplingMetric *>(metric.get());
            for (size_t i = 0; i < PCSamplingMetric::Count; i++) {
              const auto &valueName = pcSamplingMetric->getValueName(i);
              std::visit([&](auto &&value) { metricsJson[valueName] = value; },
                         pcSamplingMetric->getValues()[i]);
            }
          } else if (metricKind == MetricKind::Cycle) {
            auto *cycleMetric = static_cast<CycleMetric *>(metric.get());
            uint64_t duration = std::get<uint64_t>(
                cycleMetric->getValue(CycleMetric::Duration));
            double normalizedDuration = std::get<double>(
                cycleMetric->getValue(CycleMetric::NormalizedDuration));
            uint64_t deviceId = std::get<uint64_t>(
                cycleMetric->getValue(CycleMetric::DeviceId));
            uint64_t deviceType = std::get<uint64_t>(
                cycleMetric->getValue(CycleMetric::DeviceType));
            const auto &durationName =
                cycleMetric->getValueName(CycleMetric::Duration);
            const auto &normalizedDurationName =
                cycleMetric->getValueName(CycleMetric::NormalizedDuration);
            const auto &deviceIdName =
                cycleMetric->getValueName(CycleMetric::DeviceId);
            const auto &deviceTypeName =
                cycleMetric->getValueName(CycleMetric::DeviceType);
            const auto deviceIdStr = std::to_string(deviceId);
            const auto deviceTypeStr = std::to_string(deviceType);

            metricsJson[durationName] = duration;
            metricsJson[normalizedDurationName] = normalizedDuration;
            metricsJson[deviceIdName] = deviceIdStr;
            metricsJson[deviceTypeName] = deviceTypeStr;
          } else if (metricKind == MetricKind::Flexible) {
            // Flexible metrics are handled in a different way
          } else {
            throw std::runtime_error("MetricKind not supported");
          }
        }
        for (const auto &[_, flexibleMetric] : flexibleMetrics) {
          const auto &valueName = flexibleMetric.getValueName(0);
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
                  metricsJson[valueName] = json::array();
                  auto &arr = metricsJson[valueName];
                  arr.get_ref<json::array_t &>().reserve(v.size());
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

  tree->template walk<TreeData::Tree::WalkPolicy::PreOrder>(
      [&](TreeData::Tree::TreeNode &treeNode) {
        const auto contextName = treeNode.name;
        auto contextId = treeNode.id;
        json *jsonNode = jsonNodes[contextId];
        (*jsonNode)["frame"] = {{"name", contextName}, {"type", "function"}};
        (*jsonNode)["metrics"] = json::object();
        auto &metricsJson = (*jsonNode)["metrics"];
        appendMetrics(metricsJson, treeNode.metrics, treeNode.flexibleMetrics);
        auto &childrenArray = (*jsonNode)["children"];
        childrenArray = json::array();
        const bool hasLinkedTargets =
            !treeNode.linkedTargetMetrics.empty() ||
            !treeNode.linkedTargetFlexibleMetrics.empty();
        childrenArray.get_ref<json::array_t &>().reserve(
            treeNode.children.size() +
            (hasLinkedTargets ? staticRootNode.children.size() : 0));
        for (const auto &child : treeNode.children) {
          childrenArray.push_back(json::object());
          jsonNodes[child.id] = &childrenArray.back();
        }
        if (!hasLinkedTargets) {
          return;
        }
        std::function<void(size_t, json &)> appendLinkedStaticNode =
            [&](size_t staticNodeId, json &outNode) {
              const auto &staticNode = staticTree->getNode(staticNodeId);
              const auto metricsIt =
                  treeNode.linkedTargetMetrics.find(staticNodeId);
              const auto flexibleIt =
                  treeNode.linkedTargetFlexibleMetrics.find(staticNodeId);
              outNode = json::object();
              outNode["frame"] = {{"name", staticNode.name},
                                  {"type", "function"}};
              outNode["metrics"] = json::object();
              if (metricsIt != treeNode.linkedTargetMetrics.end() ||
                  flexibleIt != treeNode.linkedTargetFlexibleMetrics.end()) {
                const auto &linkedMetrics =
                    (metricsIt != treeNode.linkedTargetMetrics.end())
                        ? metricsIt->second
                        : emptyMetrics;
                const auto &linkedFlexibleMetrics =
                    (flexibleIt != treeNode.linkedTargetFlexibleMetrics.end())
                        ? flexibleIt->second
                        : emptyFlexibleMetrics;
                appendMetrics(outNode["metrics"], linkedMetrics,
                              linkedFlexibleMetrics);
              }
              outNode["children"] = json::array();
              auto &linkedChildren = outNode["children"];
              linkedChildren.get_ref<json::array_t &>().reserve(
                  staticNode.children.size());
              for (const auto &child : staticNode.children) {
                linkedChildren.push_back(json::object());
                appendLinkedStaticNode(child.id, linkedChildren.back());
              }
            };

        for (const auto &staticChild : staticRootNode.children) {
          json linkedRootChildNode;
          appendLinkedStaticNode(staticChild.id, linkedRootChildNode);
          childrenArray.push_back(std::move(linkedRootChildNode));
        }
      });

  if (metricSummary.hasKernelMetric) {
    KernelMetric kernelMetric;
    output[TreeData::Tree::TreeNode::RootId]["metrics"]
          [kernelMetric.getValueName(KernelMetric::Invocations)] = 0;
    output[TreeData::Tree::TreeNode::RootId]["metrics"]
          [kernelMetric.getValueName(KernelMetric::Duration)] = 0;
  }
  if (metricSummary.hasCycleMetric) {
    CycleMetric cycleMetric;
    output[TreeData::Tree::TreeNode::RootId]["metrics"]
          [cycleMetric.getValueName(CycleMetric::Duration)] = 0;
    output[TreeData::Tree::TreeNode::RootId]["metrics"]
          [cycleMetric.getValueName(CycleMetric::NormalizedDuration)] = 0;
  }
  if (metricSummary.hasPCSamplingMetric) {
    PCSamplingMetric pcSamplingMetric;
    for (size_t i = 0; i < PCSamplingMetric::Count; i++) {
      const auto &valueName = pcSamplingMetric.getValueName(i);
      output[TreeData::Tree::TreeNode::RootId]["metrics"][valueName] = 0;
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

    const auto &deviceTypeName = kDeviceTypeNames[deviceType];
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
                              TreeData::Tree *staticTree) const {
  MsgPackWriter writer;
  writer.reserve(16 * 1024 * 1024); // 16 MB

  MetricSummary metricSummary;
  const std::map<MetricKind, std::unique_ptr<Metric>> emptyMetrics;
  const std::map<std::string, FlexibleMetric> emptyFlexibleMetrics;
  const auto &staticRootNode = staticTree->getNode(Tree::TreeNode::RootId);

  tree->template walk<TreeData::Tree::WalkPolicy::PreOrder>(
      [&](TreeData::Tree::TreeNode &treeNode) {
        metricSummary.observeMetrics(treeNode.metrics);
        for (const auto &[_, linkedMetrics] : treeNode.linkedTargetMetrics) {
          metricSummary.observeMetrics(linkedMetrics);
        }
      });

  // We only need these metrics for tree data
  KernelMetric kernelMetric;
  auto &kernelMetricDurationName =
      kernelMetric.getValueName(KernelMetric::Duration);
  auto &kernelMetricInvocationsName =
      kernelMetric.getValueName(KernelMetric::Invocations);
  auto &kernelMetricDeviceIdName =
      kernelMetric.getValueName(KernelMetric::DeviceId);
  auto &kernelMetricDeviceTypeName =
      kernelMetric.getValueName(KernelMetric::DeviceType);
  CycleMetric cycleMetric;
  auto &cycleMetricDurationName =
      cycleMetric.getValueName(CycleMetric::Duration);
  auto &cycleMetricNormalizedDurationName =
      cycleMetric.getValueName(CycleMetric::NormalizedDuration);
  auto &cycleMetricDeviceIdName =
      cycleMetric.getValueName(CycleMetric::DeviceId);
  auto &cycleMetricDeviceTypeName =
      cycleMetric.getValueName(CycleMetric::DeviceType);
  std::set<std::string> kernelInclusiveValueNames = {
      kernelMetricDurationName, kernelMetricInvocationsName};
  std::set<std::string> kernelExclusiveValueNames = {
      kernelMetricDeviceIdName, kernelMetricDeviceTypeName};
  std::set<std::string> cycleInclusiveValueNames = {
      cycleMetricDurationName, cycleMetricNormalizedDurationName};
  std::set<std::string> cycleExclusiveValueNames = {cycleMetricDeviceIdName,
                                                    cycleMetricDeviceTypeName};
  const auto kernelInclusiveCount =
      static_cast<uint32_t>(kernelInclusiveValueNames.size());
  const auto kernelTotalCount = static_cast<uint32_t>(
      kernelInclusiveValueNames.size() + kernelExclusiveValueNames.size());
  const auto cycleInclusiveCount =
      static_cast<uint32_t>(cycleInclusiveValueNames.size());
  const auto cycleTotalCount = static_cast<uint32_t>(
      cycleInclusiveValueNames.size() + cycleExclusiveValueNames.size());

  auto packFlexibleMetricValue = [&](const MetricValueType &value) {
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
          } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
            writer.packArray(static_cast<uint32_t>(v.size()));
            for (auto value : v) {
              writer.packInt(value);
            }
          } else if constexpr (std::is_same_v<T, std::vector<double>>) {
            writer.packArray(static_cast<uint32_t>(v.size()));
            for (auto value : v) {
              writer.packDouble(value);
            }
          } else {
            static_assert(sizeof(T) == 0, "Unsupported MetricValueType");
          }
        },
        value);
  };

  auto countMetricEntries =
      [&](const std::map<MetricKind, std::unique_ptr<Metric>> &metrics,
          const std::map<std::string, FlexibleMetric> &flexibleMetrics,
          bool isRoot) -> uint32_t {
    uint32_t metricEntries = static_cast<uint32_t>(flexibleMetrics.size());
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
        throw std::runtime_error("MetricKind not supported");
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

  auto packMetrics =
      [&](const std::map<MetricKind, std::unique_ptr<Metric>> &metrics,
          const std::map<std::string, FlexibleMetric> &flexibleMetrics,
          bool isRoot) {
        writer.packMap(countMetricEntries(metrics, flexibleMetrics, isRoot));
        for (const auto &[metricKind, metric] : metrics) {
          if (metricKind == MetricKind::Kernel) {
            if (isRoot) {
              writer.packStr(kernelMetricDurationName);
              writer.packUInt(0);
              writer.packStr(kernelMetricInvocationsName);
              writer.packUInt(0);
              continue;
            }

            auto *kernelMetric = static_cast<KernelMetric *>(metric.get());
            uint64_t duration = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::Duration));
            uint64_t invocations = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::Invocations));
            uint64_t deviceId = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::DeviceId));
            uint64_t deviceType = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::DeviceType));
            const auto &deviceTypeName =
                kDeviceTypeNames[static_cast<size_t>(deviceType)];
            writer.packStr(kernelMetricDurationName);
            writer.packUInt(duration);
            writer.packStr(kernelMetricInvocationsName);
            writer.packUInt(invocations);
            writer.packStr(kernelMetricDeviceIdName);
            writer.packStr(std::to_string(deviceId));
            writer.packStr(kernelMetricDeviceTypeName);
            writer.packStr(deviceTypeName);
          } else if (metricKind == MetricKind::PCSampling) {
            auto *pcSamplingMetric =
                static_cast<PCSamplingMetric *>(metric.get());
            for (size_t i = 0; i < PCSamplingMetric::Count; i++) {
              const auto &valueName = pcSamplingMetric->getValueName(i);
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
              writer.packStr(cycleMetricDurationName);
              writer.packUInt(0);
              writer.packStr(cycleMetricNormalizedDurationName);
              writer.packUInt(0);
              continue;
            }

            auto *cycleMetric = static_cast<CycleMetric *>(metric.get());
            uint64_t duration = std::get<uint64_t>(
                cycleMetric->getValue(CycleMetric::Duration));
            double normalizedDuration = std::get<double>(
                cycleMetric->getValue(CycleMetric::NormalizedDuration));
            uint64_t deviceId = std::get<uint64_t>(
                cycleMetric->getValue(CycleMetric::DeviceId));
            uint64_t deviceType = std::get<uint64_t>(
                cycleMetric->getValue(CycleMetric::DeviceType));

            writer.packStr(cycleMetricDurationName);
            writer.packUInt(duration);
            writer.packStr(cycleMetricNormalizedDurationName);
            writer.packDouble(normalizedDuration);
            writer.packStr(cycleMetricDeviceIdName);
            writer.packStr(std::to_string(deviceId));
            writer.packStr(cycleMetricDeviceTypeName);
            writer.packStr(std::to_string(deviceType));
          } else {
            throw std::runtime_error("MetricKind not supported");
          }
        }
        for (const auto &[_, flexibleMetric] : flexibleMetrics) {
          const auto &valueName = flexibleMetric.getValueName(0);
          writer.packStr(valueName);
          packFlexibleMetricValue(flexibleMetric.getValues()[0]);
        }

        if (isRoot) {
          if (metricSummary.hasKernelMetric &&
              metrics.find(MetricKind::Kernel) == metrics.end()) {
            writer.packStr(kernelMetricDurationName);
            writer.packUInt(0);
            writer.packStr(kernelMetricInvocationsName);
            writer.packUInt(0);
          }
          if (metricSummary.hasPCSamplingMetric &&
              metrics.find(MetricKind::PCSampling) == metrics.end()) {
            PCSamplingMetric pcSamplingMetric;
            for (size_t i = 0; i < PCSamplingMetric::Count; i++) {
              const auto &valueName = pcSamplingMetric.getValueName(i);
              writer.packStr(valueName);
              writer.packUInt(0);
            }
          }
          if (metricSummary.hasCycleMetric &&
              metrics.find(MetricKind::Cycle) == metrics.end()) {
            writer.packStr(cycleMetricDurationName);
            writer.packUInt(0);
            writer.packStr(cycleMetricNormalizedDurationName);
            writer.packUInt(0);
          }
        }
      };
  std::function<void(TreeData::Tree::TreeNode &)> packNode =
      [&](TreeData::Tree::TreeNode &treeNode) {
        writer.packMap(3);

        writer.packStr("frame");
        writer.packMap(2);
        writer.packStr("name");
        writer.packStr(treeNode.name);
        writer.packStr("type");
        writer.packStr("function");

        writer.packStr("metrics");
        packMetrics(treeNode.metrics, treeNode.flexibleMetrics,
                    treeNode.id == TreeData::Tree::TreeNode::RootId);
        const bool hasLinkedTargets =
            !treeNode.linkedTargetMetrics.empty() ||
            !treeNode.linkedTargetFlexibleMetrics.empty();

        std::function<void(size_t)> packLinkedStaticNode =
            [&](size_t staticNodeId) {
              const auto &staticNode = staticTree->getNode(staticNodeId);
              writer.packMap(3);

              writer.packStr("frame");
              writer.packMap(2);
              writer.packStr("name");
              writer.packStr(staticNode.name);
              writer.packStr("type");
              writer.packStr("function");

              writer.packStr("metrics");
              const auto metricsIt =
                  treeNode.linkedTargetMetrics.find(staticNodeId);
              const auto flexibleIt =
                  treeNode.linkedTargetFlexibleMetrics.find(staticNodeId);
              if (metricsIt != treeNode.linkedTargetMetrics.end() ||
                  flexibleIt != treeNode.linkedTargetFlexibleMetrics.end()) {
                const auto &linkedMetrics =
                    (metricsIt != treeNode.linkedTargetMetrics.end())
                        ? metricsIt->second
                        : emptyMetrics;
                const auto &linkedFlexibleMetrics =
                    (flexibleIt != treeNode.linkedTargetFlexibleMetrics.end())
                        ? flexibleIt->second
                        : emptyFlexibleMetrics;
                packMetrics(linkedMetrics, linkedFlexibleMetrics,
                            /*isRoot=*/false);
              } else {
                writer.packMap(0);
              }

              writer.packStr("children");
              writer.packArray(
                  static_cast<uint32_t>(staticNode.children.size()));
              for (const auto &child : staticNode.children) {
                packLinkedStaticNode(child.id);
              }
            };

        uint32_t linkedChildCount =
            hasLinkedTargets
                ? static_cast<uint32_t>(staticRootNode.children.size())
                : 0;
        writer.packStr("children");
        writer.packArray(static_cast<uint32_t>(treeNode.children.size()) +
                         linkedChildCount);
        for (const auto &child : treeNode.children) {
          packNode(tree->getNode(child.id));
        }
        if (hasLinkedTargets) {
          for (const auto &staticChild : staticRootNode.children) {
            packLinkedStaticNode(staticChild.id);
          }
        }
      };

  uint32_t deviceTypeEntries = 0;
  for (size_t deviceType = 0;
       deviceType < static_cast<size_t>(DeviceType::COUNT); ++deviceType) {
    if (metricSummary.deviceIdMasks[deviceType] != 0) {
      ++deviceTypeEntries;
    }
  }
  // Hatchet format: [tree, device_metadata]. Always emit 2 elements to match
  // the JSON serializer, even if device_metadata is empty.
  writer.packArray(2);
  packNode(tree->getNode(TreeData::Tree::TreeNode::RootId));

  auto countSetBits = [](uint32_t mask) -> uint32_t {
    uint32_t count = 0;
    while (mask) {
      mask &= (mask - 1);
      ++count;
    }
    return count;
  };

  writer.packMap(deviceTypeEntries);
  for (size_t deviceType = 0;
       deviceType < static_cast<size_t>(DeviceType::COUNT); ++deviceType) {
    auto mask = metricSummary.deviceIdMasks[deviceType];
    if (mask == 0) {
      continue;
    }

    const auto &deviceTypeName = kDeviceTypeNames[deviceType];
    writer.packStr(deviceTypeName);

    writer.packMap(countSetBits(mask));
    for (uint64_t deviceId = 0; deviceId < kMaxRegisteredDeviceIds;
         ++deviceId) {
      if ((mask & (1u << static_cast<uint32_t>(deviceId))) == 0) {
        continue;
      }
      Device device = getDevice(static_cast<DeviceType>(deviceType), deviceId);
      writer.packStr(std::to_string(deviceId));
      writer.packMap(5);
      writer.packStr("clock_rate");
      writer.packUInt(device.clockRate);
      writer.packStr("memory_clock_rate");
      writer.packUInt(device.memoryClockRate);
      writer.packStr("bus_width");
      writer.packUInt(device.busWidth);
      writer.packStr("arch");
      writer.packStr(device.arch);
      writer.packStr("num_sms");
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
    contexts.push_back(scope.name);
  auto contextId = currentTree->addNode(contexts);
  scopeIdToContextId[scope.scopeId] = contextId;
}

void TreeData::exitScope(const Scope &scope) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  scopeIdToContextId.erase(scope.scopeId);
}

DataEntry TreeData::addOp(size_t phase, size_t contextId,
                          const std::vector<Context> &contexts) {
  auto lock = lockIfCurrentOrStaticPhase(phase);
  if (contextId == Data::kRootEntryId) {
    contextId = Tree::TreeNode::RootId;
  }
  auto *tree = phasePtrAs<Tree>(phase);
  auto newContextId = tree->addNode(contexts, contextId);
  auto &node = tree->getNode(newContextId);
  return DataEntry(newContextId, phase, node.metrics, node.flexibleMetrics);
}

DataEntry TreeData::linkOp(size_t baseEntryId, size_t targetEntryId) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  const auto phase = currentPhase.load(std::memory_order_relaxed);
  auto *tree = currentPhasePtrAs<Tree>();
  auto &baseNode = tree->getNode(baseEntryId);
  auto &linkedMetrics = baseNode.linkedTargetMetrics[targetEntryId];
  auto &linkedFlexibleMetrics =
      baseNode.linkedTargetFlexibleMetrics[targetEntryId];
  return DataEntry(baseEntryId, phase, linkedMetrics, linkedFlexibleMetrics);
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
    Tree staticTreeCloned;
    treePhases.withPtr(Data::kVirtualPhase, [&](Tree *staticTree) {
      staticTreeCloned = staticTree->structure();
    });
    auto output = buildHatchetJson(tree, &staticTreeCloned);
    os << std::endl << output.dump(4) << std::endl;
  });
}

void TreeData::dumpHatchetMsgPack(std::ostream &os, size_t phase) const {
  treePhases.withPtr(phase, [&](Tree *tree) {
    Tree staticTreeCloned;
    treePhases.withPtr(Data::kVirtualPhase, [&](Tree *staticTree) {
      staticTreeCloned = staticTree->structure();
    });
    auto msgPack = buildHatchetMsgPack(tree, &staticTreeCloned);
    os.write(reinterpret_cast<const char *>(msgPack.data()),
             static_cast<std::streamsize>(msgPack.size()));
  });
}

std::string TreeData::toJsonString(size_t phase) const {
  return treePhases.withPtr(phase, [&](Tree *tree) {
    Tree staticTreeCloned;
    treePhases.withPtr(Data::kVirtualPhase, [&](Tree *staticTree) {
      staticTreeCloned = staticTree->structure();
    });
    return buildHatchetJson(tree, &staticTreeCloned).dump();
  });
}

std::vector<uint8_t> TreeData::toMsgPack(size_t phase) const {
  return treePhases.withPtr(phase, [&](Tree *tree) {
    Tree staticTreeCloned;
    treePhases.withPtr(Data::kVirtualPhase, [&](Tree *staticTree) {
      staticTreeCloned = staticTree->structure();
    });
    return buildHatchetMsgPack(tree, &staticTreeCloned);
  });
}

void TreeData::doDump(std::ostream &os, OutputFormat outputFormat,
                      size_t phase) const {
  if (outputFormat == OutputFormat::Hatchet) {
    dumpHatchet(os, phase);
  } else if (outputFormat == OutputFormat::HatchetMsgPack) {
    dumpHatchetMsgPack(os, phase);
  } else {
    throw std::logic_error("Output format not supported");
  }
}

TreeData::TreeData(const std::string &path, ContextSource *contextSource)
    : Data(path, contextSource) {
  initPhaseStore(treePhases);
}

TreeData::~TreeData() {}

} // namespace proton
