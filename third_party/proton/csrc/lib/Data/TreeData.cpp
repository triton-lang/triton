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

} // namespace

class TreeData::Tree {
public:
  struct TreeNode : public Context {
    inline static const size_t RootId = 0;
    inline static const size_t DummyId = std::numeric_limits<size_t>::max();

    TreeNode() = default;
    explicit TreeNode(size_t id, const std::string &name)
        : id(id), Context(name) {}
    TreeNode(size_t id, size_t parentId, const std::string &name)
        : id(id), parentId(parentId), Context(name) {}
    virtual ~TreeNode() = default;

    void addChild(const Context &context, size_t id) {
      children.emplace(context.name, id);
      childIds.push_back(id);
    }

    bool hasChild(const Context &context) const {
      return children.find(context.name) != children.end();
    }

    size_t getChild(const Context &context) const {
      return children.at(context.name);
    }

    size_t parentId = DummyId;
    size_t id = DummyId;
    std::unordered_map<std::string, size_t> children = {};
    std::vector<size_t> childIds = {};
    std::map<MetricKind, std::shared_ptr<Metric>> metrics = {};
    std::map<std::string, FlexibleMetric> flexibleMetrics = {};
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
    if (parent.hasChild(context))
      return parent.getChild(context);
    auto id = nextContextId++;
    treeNodeMap.try_emplace(id, id, parentId, context.name);
    parent.addChild(context, id);
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

  void upsertMetric(size_t contextId, std::shared_ptr<Metric> metric) {
    auto &node = treeNodeMap.at(contextId);
    auto it = node.metrics.find(metric->getKind());
    if (it == node.metrics.end()) {
      node.metrics.emplace(metric->getKind(), metric);
      metricKinds.insert(metric->getKind());
      if (metric->getKind() == MetricKind::Kernel) {
        auto kernelMetric = std::static_pointer_cast<KernelMetric>(metric);
        uint64_t deviceId =
            std::get<uint64_t>(kernelMetric->getValue(KernelMetric::DeviceId));
        uint64_t deviceType = std::get<uint64_t>(
            kernelMetric->getValue(KernelMetric::DeviceType));
        if (deviceType < static_cast<uint64_t>(DeviceType::COUNT) &&
            deviceId < kMaxRegisteredDeviceIds) {
          deviceIdMasks[static_cast<size_t>(deviceType)] |=
              (1u << static_cast<uint32_t>(deviceId));
        }
      } else if (metric->getKind() == MetricKind::Cycle) {
        auto cycleMetric = std::static_pointer_cast<CycleMetric>(metric);
        uint64_t deviceId =
            std::get<uint64_t>(cycleMetric->getValue(CycleMetric::DeviceId));
        uint64_t deviceType =
            std::get<uint64_t>(cycleMetric->getValue(CycleMetric::DeviceType));
        if (deviceType < static_cast<uint64_t>(DeviceType::COUNT) &&
            deviceId < kMaxRegisteredDeviceIds) {
          deviceIdMasks[static_cast<size_t>(deviceType)] |=
              (1u << static_cast<uint32_t>(deviceId));
        }
      }
    } else {
      it->second->updateMetric(*metric);
    }
  }

  void upsertFlexibleMetric(size_t contextId,
                            const FlexibleMetric &flexibleMetric) {
    auto &node = treeNodeMap.at(contextId);
    auto it = node.flexibleMetrics.find(flexibleMetric.getValueName(0));
    if (it == node.flexibleMetrics.end()) {
      node.flexibleMetrics.emplace(flexibleMetric.getValueName(0),
                                   flexibleMetric);
      flexibleMetricNames.insert(flexibleMetric.getValueName(0));
    } else {
      it->second.updateMetric(flexibleMetric);
    }
  }

  bool hasMetricKind(MetricKind kind) const {
    return metricKinds.find(kind) != metricKinds.end();
  }

  const std::array<uint32_t, static_cast<size_t>(DeviceType::COUNT)> &
  getDeviceIdMasks() const {
    return deviceIdMasks;
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
    for (auto childId : getNode(contextId).childIds) {
      walkPreOrder(childId, fn);
    }
  }

  template <typename FnT> void walkPostOrder(size_t contextId, FnT &&fn) {
    for (auto childId : getNode(contextId).childIds) {
      walkPostOrder(childId, fn);
    }
    fn(getNode(contextId));
  }

  size_t size() const { return nextContextId; }

private:
  size_t nextContextId = TreeNode::RootId + 1;
  // tree node id -> tree node
  std::unordered_map<size_t, TreeNode> treeNodeMap;
  // all available metric kinds
  std::set<MetricKind> metricKinds;
  // all available flexible metric names
  std::set<std::string> flexibleMetricNames;
  // device type -> bitmask for active device ids
  std::array<uint32_t, static_cast<size_t>(DeviceType::COUNT)> deviceIdMasks{};
};

json TreeData::buildHatchetJson(TreeData::Tree *tree) const {
  std::vector<json *> jsonNodes(tree->size(), nullptr);
  json output = json::array();
  output.push_back(json::object());
  jsonNodes[TreeData::Tree::TreeNode::RootId] = &(output.back());
  bool hasKernelMetric = false;
  bool hasPCSamplingMetric = false;
  bool hasCycleMetric = false;
  std::array<uint32_t, static_cast<size_t>(DeviceType::COUNT)> deviceIdMasks{};
  tree->template walk<TreeData::Tree::WalkPolicy::PreOrder>(
      [&](TreeData::Tree::TreeNode &treeNode) {
        const auto contextName = treeNode.name;
        auto contextId = treeNode.id;
        json *jsonNode = jsonNodes[contextId];
        (*jsonNode)["frame"] = {{"name", contextName}, {"type", "function"}};
        (*jsonNode)["metrics"] = json::object();
        auto &metricsJson = (*jsonNode)["metrics"];
        for (auto &[metricKind, metric] : treeNode.metrics) {
          if (metricKind == MetricKind::Kernel) {
            hasKernelMetric = true;
            std::shared_ptr<KernelMetric> kernelMetric =
                std::static_pointer_cast<KernelMetric>(metric);
            uint64_t duration = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::Duration));
            uint64_t invocations = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::Invocations));
            uint64_t deviceId = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::DeviceId));
            uint64_t deviceType = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::DeviceType));
            if (deviceId < kMaxRegisteredDeviceIds) {
              deviceIdMasks[static_cast<size_t>(deviceType)] |=
                  (1u << static_cast<uint32_t>(deviceId));
            } else {
              throw std::runtime_error(
                  "[PROTON] DeviceId " + std::to_string(deviceId) +
                  " exceeds MaxRegisteredDeviceIds " +
                  std::to_string(kMaxRegisteredDeviceIds) + " for deviceType " +
                  std::to_string(deviceType));
            }
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
            hasPCSamplingMetric = true;
            auto pcSamplingMetric =
                std::static_pointer_cast<PCSamplingMetric>(metric);
            for (size_t i = 0; i < PCSamplingMetric::Count; i++) {
              const auto &valueName = pcSamplingMetric->getValueName(i);
              std::visit([&](auto &&value) { metricsJson[valueName] = value; },
                         pcSamplingMetric->getValues()[i]);
            }
          } else if (metricKind == MetricKind::Cycle) {
            hasCycleMetric = true;
            auto cycleMetric = std::static_pointer_cast<CycleMetric>(metric);
            uint64_t duration = std::get<uint64_t>(
                cycleMetric->getValue(CycleMetric::Duration));
            double normalizedDuration = std::get<double>(
                cycleMetric->getValue(CycleMetric::NormalizedDuration));
            uint64_t deviceId = std::get<uint64_t>(
                cycleMetric->getValue(CycleMetric::DeviceId));
            uint64_t deviceType = std::get<uint64_t>(
                cycleMetric->getValue(CycleMetric::DeviceType));
            if (deviceId < kMaxRegisteredDeviceIds) {
              deviceIdMasks[static_cast<size_t>(deviceType)] |=
                  (1u << static_cast<uint32_t>(deviceId));
            } else {
              throw std::runtime_error(
                  "[PROTON] DeviceId " + std::to_string(deviceId) +
                  " exceeds MaxRegisteredDeviceIds " +
                  std::to_string(kMaxRegisteredDeviceIds) + " for deviceType " +
                  std::to_string(deviceType));
            }
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
        for (auto &[_, flexibleMetric] : treeNode.flexibleMetrics) {
          const auto &valueName = flexibleMetric.getValueName(0);
          std::visit([&](auto &&value) { metricsJson[valueName] = value; },
                     flexibleMetric.getValues()[0]);
        }
        auto &childrenArray = (*jsonNode)["children"];
        childrenArray = json::array();
        childrenArray.get_ref<json::array_t &>().reserve(
            treeNode.childIds.size());
        for (auto childId : treeNode.childIds) {
          childrenArray.push_back(json::object());
          jsonNodes[childId] = &childrenArray.back();
        }
      });

  if (hasKernelMetric) {
    KernelMetric kernelMetric;
    output[TreeData::Tree::TreeNode::RootId]["metrics"]
          [kernelMetric.getValueName(KernelMetric::Invocations)] = 0;
    output[TreeData::Tree::TreeNode::RootId]["metrics"]
          [kernelMetric.getValueName(KernelMetric::Duration)] = 0;
  }
  if (hasCycleMetric) {
    CycleMetric cycleMetric;
    output[TreeData::Tree::TreeNode::RootId]["metrics"]
          [cycleMetric.getValueName(CycleMetric::Duration)] = 0;
    output[TreeData::Tree::TreeNode::RootId]["metrics"]
          [cycleMetric.getValueName(CycleMetric::NormalizedDuration)] = 0;
  }
  if (hasPCSamplingMetric) {
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
    auto mask = deviceIdMasks[deviceType];
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

std::vector<uint8_t> TreeData::buildHatchetMsgPack(TreeData::Tree *tree) const {
  MsgPackWriter writer;
  writer.reserve(16 * 1024 * 1024); // 16 MB

  bool hasKernelMetric = tree->hasMetricKind(MetricKind::Kernel);
  bool hasPCSamplingMetric = tree->hasMetricKind(MetricKind::PCSampling);
  bool hasCycleMetric = tree->hasMetricKind(MetricKind::Cycle);
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
  const auto &deviceIdMasks = tree->getDeviceIdMasks();

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
        uint32_t metricEntries = 0;
        for (auto &[metricKind, metric] : treeNode.metrics) {
          if (metricKind == MetricKind::Kernel) {
            metricEntries += (treeNode.id == TreeData::Tree::TreeNode::RootId)
                                 ? kernelInclusiveValueNames.size()
                                 : (kernelInclusiveValueNames.size() +
                                    kernelExclusiveValueNames.size());
          } else if (metricKind == MetricKind::PCSampling) {
            metricEntries += PCSamplingMetric::Count;
          } else if (metricKind == MetricKind::Cycle) {
            metricEntries += (treeNode.id == TreeData::Tree::TreeNode::RootId)
                                 ? cycleInclusiveValueNames.size()
                                 : (cycleInclusiveValueNames.size() +
                                    cycleExclusiveValueNames.size());
          }
        }
        if (treeNode.id == TreeData::Tree::TreeNode::RootId) {
          if (hasKernelMetric && treeNode.metrics.find(MetricKind::Kernel) ==
                                     treeNode.metrics.end()) {
            metricEntries +=
                static_cast<uint32_t>(kernelInclusiveValueNames.size());
          }
          if (hasPCSamplingMetric &&
              treeNode.metrics.find(MetricKind::PCSampling) ==
                  treeNode.metrics.end()) {
            metricEntries += PCSamplingMetric::Count;
          }
          if (hasCycleMetric && treeNode.metrics.find(MetricKind::Cycle) ==
                                    treeNode.metrics.end()) {
            metricEntries +=
                static_cast<uint32_t>(cycleInclusiveValueNames.size());
          }
        }
        metricEntries += static_cast<uint32_t>(treeNode.flexibleMetrics.size());
        writer.packMap(metricEntries);

        for (auto &[metricKind, metric] : treeNode.metrics) {
          if (metricKind == MetricKind::Kernel) {
            if (treeNode.id == TreeData::Tree::TreeNode::RootId) {
              writer.packStr(kernelMetricDurationName);
              writer.packUInt(0);
              writer.packStr(kernelMetricInvocationsName);
              writer.packUInt(0);
              continue;
            }

            auto kernelMetric = std::static_pointer_cast<KernelMetric>(metric);
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
            auto pcSamplingMetric =
                std::static_pointer_cast<PCSamplingMetric>(metric);
            for (size_t i = 0; i < PCSamplingMetric::Count; i++) {
              const auto &valueName = pcSamplingMetric->getValueName(i);
              writer.packStr(valueName);
              if (treeNode.id == TreeData::Tree::TreeNode::RootId) {
                writer.packUInt(0);
              } else {
                writer.packUInt(
                    std::get<uint64_t>(pcSamplingMetric->getValues()[i]));
              }
            }
          } else if (metricKind == MetricKind::Cycle) {
            if (treeNode.id == TreeData::Tree::TreeNode::RootId) {
              writer.packStr(cycleMetricDurationName);
              writer.packUInt(0);
              writer.packStr(cycleMetricNormalizedDurationName);
              writer.packUInt(0);
              continue;
            }

            auto cycleMetric = std::static_pointer_cast<CycleMetric>(metric);
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
          }
        }

        for (auto &[_, flexibleMetric] : treeNode.flexibleMetrics) {
          const auto &valueName = flexibleMetric.getValueName(0);
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
                } else {
                  static_assert(sizeof(T) == 0, "Unsupported MetricValueType");
                }
              },
              flexibleMetric.getValues()[0]);
        }

        if (treeNode.id == TreeData::Tree::TreeNode::RootId) {
          if (hasKernelMetric && treeNode.metrics.find(MetricKind::Kernel) ==
                                     treeNode.metrics.end()) {
            writer.packStr(kernelMetricDurationName);
            writer.packUInt(0);
            writer.packStr(kernelMetricInvocationsName);
            writer.packUInt(0);
          }
          if (hasPCSamplingMetric &&
              treeNode.metrics.find(MetricKind::PCSampling) ==
                  treeNode.metrics.end()) {
            PCSamplingMetric pcSamplingMetric;
            for (size_t i = 0; i < PCSamplingMetric::Count; i++) {
              const auto &valueName = pcSamplingMetric.getValueName(i);
              writer.packStr(valueName);
              writer.packUInt(0);
            }
          }
          if (hasCycleMetric && treeNode.metrics.find(MetricKind::Cycle) ==
                                    treeNode.metrics.end()) {
            writer.packStr(cycleMetricDurationName);
            writer.packUInt(0);
            writer.packStr(cycleMetricNormalizedDurationName);
            writer.packUInt(0);
          }
        }

        writer.packStr("children");
        writer.packArray(static_cast<uint32_t>(treeNode.childIds.size()));
        for (auto childId : treeNode.childIds) {
          packNode(tree->getNode(childId));
        }
      };

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

  uint32_t deviceTypeEntries = 0;
  for (size_t deviceType = 0;
       deviceType < static_cast<size_t>(DeviceType::COUNT); ++deviceType) {
    if (deviceIdMasks[deviceType] != 0) {
      ++deviceTypeEntries;
    }
  }
  writer.packMap(deviceTypeEntries);
  for (size_t deviceType = 0;
       deviceType < static_cast<size_t>(DeviceType::COUNT); ++deviceType) {
    auto mask = deviceIdMasks[deviceType];
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
  std::vector<Context> contexts;
  if (contextSource != nullptr)
    contexts = contextSource->getContexts();
  else
    contexts.push_back(scope.name);
  auto contextId = tree->addNode(contexts);
  scopeIdToContextId[scope.scopeId] = contextId;
}

void TreeData::exitScope(const Scope &scope) {}

size_t TreeData::addOp(size_t scopeId, const std::string &name) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto scopeIdIt = scopeIdToContextId.find(scopeId);
  if (scopeIdIt == scopeIdToContextId.end()) {
    // Obtain the current context
    std::vector<Context> contexts;
    if (contextSource != nullptr)
      contexts = contextSource->getContexts();
    // Add an op under the current context
    if (!name.empty())
      contexts.emplace_back(name);
    scopeIdToContextId[scopeId] = tree->addNode(contexts);
  } else {
    // Add a new context under it and update the context
    scopeId = Scope::getNewScopeId();
    scopeIdToContextId[scopeId] =
        tree->addNode(Context(name), scopeIdIt->second);
  }
  return scopeId;
}

size_t TreeData::addOp(size_t scopeId, const std::vector<Context> &contexts) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto scopeIdIt = scopeIdToContextId.find(scopeId);
  if (scopeIdIt == scopeIdToContextId.end()) {
    // Obtain the current context
    std::vector<Context> currentContexts;
    if (contextSource != nullptr)
      currentContexts = contextSource->getContexts();
    // Add an op under the current context
    if (!currentContexts.empty())
      std::merge(currentContexts.begin(), currentContexts.end(),
                 contexts.begin(), contexts.end(), currentContexts.begin());
    scopeIdToContextId[scopeId] = tree->addNode(currentContexts);
  } else {
    // Add a new context under it and update the context
    scopeId = Scope::getNewScopeId();
    scopeIdToContextId[scopeId] = tree->addNode(contexts, scopeIdIt->second);
  }
  return scopeId;
}

void TreeData::addMetric(size_t scopeId, std::shared_ptr<Metric> metric) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto scopeIdIt = scopeIdToContextId.find(scopeId);
  // The profile data is deactivated, ignore the metric
  if (scopeIdIt == scopeIdToContextId.end())
    return;
  auto contextId = scopeIdIt->second;
  tree->upsertMetric(contextId, metric);
}

void TreeData::addOpAndMetric(size_t scopeId, const std::string &opName,
                              std::shared_ptr<Metric> metric) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto scopeIdIt = scopeIdToContextId.find(scopeId);
  // The profile data is deactivated, ignore the metric
  if (scopeIdIt == scopeIdToContextId.end())
    return;

  auto contextId = scopeIdIt->second;
  contextId = tree->addNode(Context(opName), contextId);
  tree->upsertMetric(contextId, metric);
}

void TreeData::addMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &metrics) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto scopeIdIt = scopeIdToContextId.find(scopeId);
  // The profile data is deactivated, ignore the metric
  if (scopeIdIt == scopeIdToContextId.end())
    return;
  auto contextId = scopeIdIt->second;
  auto &node = tree->getNode(contextId);
  for (auto [metricName, metricValue] : metrics) {
    tree->upsertFlexibleMetric(contextId,
                               FlexibleMetric(metricName, metricValue));
  }
}

void TreeData::clear() {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto newTree = std::make_unique<Tree>();
  tree.swap(newTree);
  scopeIdToContextId.clear();
}

void TreeData::clearCache() {
  std::unique_lock<std::shared_mutex> lock(mutex);
  scopeIdToContextId.clear();
}

void TreeData::dumpHatchet(std::ostream &os) const {
  auto output = buildHatchetJson(tree.get());
  os << std::endl << output.dump(4) << std::endl;
}

std::vector<uint8_t> TreeData::toMsgPack() const {
  std::shared_lock<std::shared_mutex> lock(mutex);
  return buildHatchetMsgPack(tree.get());
}

std::string TreeData::toJsonString() const {
  std::shared_lock<std::shared_mutex> lock(mutex);
  return buildHatchetJson(tree.get()).dump();
}

void TreeData::doDump(std::ostream &os, OutputFormat outputFormat) const {
  if (outputFormat == OutputFormat::Hatchet) {
    dumpHatchet(os);
  } else {
    std::logic_error("Output format not supported");
  }
}

TreeData::TreeData(const std::string &path, ContextSource *contextSource)
    : Data(path, contextSource) {
  tree = std::make_unique<Tree>();
}

TreeData::~TreeData() {}

} // namespace proton
