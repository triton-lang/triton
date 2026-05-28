#include "Device.h"
#include "DeviceType.h"
#include "Dump/TreeDataDump.h"
#include "Profiler/Graph.h"
#include "Utility/Errors.h"

#include <functional>
#include <type_traits>

namespace proton {
namespace {

void appendMetricsToJson(
    json &metricsJson,
    const std::map<MetricKind, std::unique_ptr<Metric>> &metrics,
    tree_data_dump::MetricSummary &metricSummary) {
  metricSummary.observeMetrics(metrics);
  for (const auto &[metricKind, metric] : metrics) {
    if (metricKind == MetricKind::Kernel) {
      auto *kernelMetric = static_cast<KernelMetric *>(metric.get());
      uint64_t duration =
          std::get<uint64_t>(kernelMetric->getValue(KernelMetric::Duration));
      uint64_t invocations =
          std::get<uint64_t>(kernelMetric->getValue(KernelMetric::Invocations));
      uint64_t deviceId =
          std::get<uint64_t>(kernelMetric->getValue(KernelMetric::DeviceId));
      uint64_t deviceType =
          std::get<uint64_t>(kernelMetric->getValue(KernelMetric::DeviceType));
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
        std::visit([&](auto &&value) { metricsJson[valueName] = value; },
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

      metricsJson[CycleMetric::getValueName(CycleMetric::Duration)] = duration;
      metricsJson[CycleMetric::getValueName(CycleMetric::NormalizedDuration)] =
          normalizedDuration;
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
}

void appendFlexibleMetricsToJson(
    json &metricsJson,
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
}

} // namespace

json TreeData::buildHatchetJson(TreeData::Tree *tree,
                                TreeData::Tree *virtualTree) const {
  std::vector<json *> jsonNodes(tree->size(), nullptr);
  json output = json::array();
  output.push_back(json::object());
  jsonNodes[TreeData::Tree::TreeNode::RootId] = &(output.back());
  tree_data_dump::MetricSummary metricSummary;
  tree->template walk<TreeData::Tree::WalkPolicy::PreOrder>(
      [&](TreeData::Tree::TreeNode &treeNode) {
        const auto &contextName = treeNode.name;
        auto contextId = treeNode.id;
        json *jsonNode = jsonNodes[contextId];
        (*jsonNode)["frame"] = {{"name", contextName}, {"type", "function"}};
        (*jsonNode)["metrics"] = json::object();
        auto &metricsJson = (*jsonNode)["metrics"];
        appendMetricsToJson(metricsJson, treeNode.metricSet.metrics,
                            metricSummary);
        appendFlexibleMetricsToJson(metricsJson,
                                    treeNode.metricSet.flexibleMetrics);
        auto &childrenArray = (*jsonNode)["children"];
        childrenArray = json::array();
        const auto &virtualRootNode =
            virtualTree->getNode(Tree::TreeNode::RootId);
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
        // JSON dumping is not the performance-critical path, so use a direct
        // recursive copy of the linked virtual tree.
        std::function<void(size_t, json &, json &)> appendLinkedVirtualNode =
            [&](size_t virtualNodeId, json &outNode, json &parentMetricsJson) {
              const auto &virtualNode = virtualTree->getNode(virtualNodeId);
              const auto metricsIt =
                  treeNode.metricSet.linkedMetrics.find(virtualNodeId);
              const auto flexibleIt =
                  treeNode.metricSet.linkedFlexibleMetrics.find(virtualNodeId);
              outNode = json::object();
              outNode["frame"] = {{"name", virtualNode.name},
                                  {"type", "function"}};
              outNode["metrics"] = json::object();
              if (metricsIt != treeNode.metricSet.linkedMetrics.end()) {
                appendMetricsToJson(outNode["metrics"], metricsIt->second,
                                    metricSummary);
              }
              // Linked flexible metrics are attached to generated <metric>
              // helper nodes, but they belong on the helper's parent frame.
              // Other linked virtual nodes should not carry flexible metrics.
              if (flexibleIt !=
                      treeNode.metricSet.linkedFlexibleMetrics.end() &&
                  virtualNode.name == GraphState::metricTag) {
                appendFlexibleMetricsToJson(parentMetricsJson,
                                            flexibleIt->second);
              }
              outNode["children"] = json::array();
              auto &linkedChildren = outNode["children"];
              linkedChildren.get_ref<json::array_t &>().reserve(
                  virtualNode.children.size());
              for (const auto &child : virtualNode.children) {
                linkedChildren.push_back(json::object());
                appendLinkedVirtualNode(child.id, linkedChildren.back(),
                                        outNode["metrics"]);
              }
            };

        for (const auto &child : virtualRootNode.children) {
          json linkedRootChildNode;
          appendLinkedVirtualNode(child.id, linkedRootChildNode, metricsJson);
          childrenArray.push_back(std::move(linkedRootChildNode));
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

    const auto &deviceTypeName =
        getDeviceTypeString(static_cast<DeviceType>(deviceType));
    deviceJson[deviceTypeName] = json::object();

    for (uint64_t deviceId = 0;
         deviceId < tree_data_dump::kMaxRegisteredDeviceIds; ++deviceId) {
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

} // namespace proton
