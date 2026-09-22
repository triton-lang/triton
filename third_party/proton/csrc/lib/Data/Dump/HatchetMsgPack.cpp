#include "Device.h"
#include "DeviceType.h"
#include "Dump/TreeDataDump.h"
#include "Utility/Errors.h"
#include "Utility/MsgPackWriter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace proton {
namespace {

using DeviceIdStrings =
    std::array<std::string, tree_data_dump::kMaxRegisteredDeviceIds>;
using DeviceTypeNames =
    std::array<std::string, static_cast<size_t>(DeviceType::COUNT)>;

// Root metrics only carry inclusive aggregate fields. Non-root metrics also
// include device_id and device_type, so their serialized map entry counts are
// larger.
constexpr uint32_t kKernelInclusiveCount = 2; // duration, count
constexpr uint32_t kKernelTotalCount = 4;     // + device_id, device_type
constexpr uint32_t kCycleInclusiveCount = 2;  // duration, normalized_duration
constexpr uint32_t kCycleTotalCount = 4;      // + device_id, device_type
constexpr size_t kCachedFrameHeaderMinNameBytes = 64;

static constexpr uint8_t kKernelDurationKey[] = {0xa9, 't', 'i', 'm', 'e',
                                                 ' ',  '(', 'n', 's', ')'};
static constexpr uint8_t kKernelInvocationsKey[] = {0xa5, 'c', 'o',
                                                    'u',  'n', 't'};
static constexpr uint8_t kDeviceIdKey[] = {0xa9, 'd', 'e', 'v', 'i',
                                           'c',  'e', '_', 'i', 'd'};
static constexpr uint8_t kDeviceTypeKey[] = {0xab, 'd', 'e', 'v', 'i', 'c',
                                             'e',  '_', 't', 'y', 'p', 'e'};

void packUncachedHatchetFrameHeader(MsgPackWriter &out, std::string_view name) {
  static constexpr uint8_t kHatchetFrameHeaderPrefix[] = {
      0x83, // map(3)
      0xa5, 'f', 'r', 'a', 'm', 'e',
      0x82, // map(2)
      0xa4, 'n', 'a', 'm', 'e'};
  static constexpr uint8_t kHatchetFrameHeaderSuffix[] = {
      0xa4, 't', 'y', 'p',  'e', 0xa8, 'f', 'u', 'n', 'c', 't',
      'i',  'o', 'n', 0xa7, 'm', 'e',  't', 'r', 'i', 'c', 's'};
  out.appendBytes(kHatchetFrameHeaderPrefix);
  out.packStr(name);
  out.appendBytes(kHatchetFrameHeaderSuffix);
}

void packHatchetFrameHeader(
    MsgPackWriter &writer,
    std::unordered_map<std::string_view, std::vector<uint8_t>>
        &frameHeaderCache,
    std::string_view name) {
  // Names that fit in MsgPack fixstr are cheap enough to encode directly.
  // Cache only longer headers so repeated linked virtual frames can skip the
  // larger string copy without adding hash-table overhead to every small name.
  if (name.size() < kCachedFrameHeaderMinNameBytes) {
    packUncachedHatchetFrameHeader(writer, name);
    return;
  }

  auto it = frameHeaderCache.find(name);
  if (it != frameHeaderCache.end()) {
    writer.appendBytes(it->second);
    return;
  }

  const auto offset = writer.size();
  packUncachedHatchetFrameHeader(writer, name);
  std::vector<uint8_t> headerBytes;
  headerBytes.insert(headerBytes.end(), writer.data() + offset,
                     writer.data() + writer.size());
  frameHeaderCache.emplace(name, std::move(headerBytes));
}

uint32_t
countMetricEntries(const std::map<MetricKind, std::unique_ptr<Metric>> &metrics,
                   bool isRoot,
                   const tree_data_dump::MetricSummary &metricSummary) {
  uint32_t metricEntries = 0;
  for (const auto &[metricKind, _] : metrics) {
    if (metricKind == MetricKind::Kernel) {
      metricEntries += isRoot ? kKernelInclusiveCount : kKernelTotalCount;
    } else if (metricKind == MetricKind::PCSampling) {
      metricEntries += PCSamplingMetric::Count;
    } else if (metricKind == MetricKind::Cycle) {
      metricEntries += isRoot ? kCycleInclusiveCount : kCycleTotalCount;
    } else if (metricKind == MetricKind::Flexible) {
      // Flexible metrics are tracked in a separate map.
    } else {
      throw makeLogicError("MetricKind not supported");
    }
  }
  if (isRoot) {
    if (metricSummary.hasKernelMetric &&
        metrics.find(MetricKind::Kernel) == metrics.end()) {
      metricEntries += kKernelInclusiveCount;
    }
    if (metricSummary.hasPCSamplingMetric &&
        metrics.find(MetricKind::PCSampling) == metrics.end()) {
      metricEntries += PCSamplingMetric::Count;
    }
    if (metricSummary.hasCycleMetric &&
        metrics.find(MetricKind::Cycle) == metrics.end()) {
      metricEntries += kCycleInclusiveCount;
    }
  }
  return metricEntries;
}

void packKernelMetricValues(MsgPackWriter &writer,
                            const KernelMetric *kernelMetric,
                            tree_data_dump::MetricSummary &metricSummary,
                            const DeviceIdStrings &deviceIdStrings,
                            const DeviceTypeNames &deviceTypeNames) {
  const auto &values = kernelMetric->getValues();
  uint64_t duration = std::get<uint64_t>(values[KernelMetric::Duration]);
  uint64_t invocations = std::get<uint64_t>(values[KernelMetric::Invocations]);
  uint64_t deviceId = std::get<uint64_t>(values[KernelMetric::DeviceId]);
  uint64_t deviceType = std::get<uint64_t>(values[KernelMetric::DeviceType]);
  metricSummary.updateDeviceIdMask(deviceType, deviceId);
  const auto &deviceTypeName = deviceTypeNames[deviceType];
  writer.appendBytes(kKernelDurationKey);
  writer.packUInt(duration);
  writer.appendBytes(kKernelInvocationsKey);
  writer.packUInt(invocations);
  writer.appendBytes(kDeviceIdKey);
  writer.packFixStr(deviceIdStrings[deviceId]);
  writer.appendBytes(kDeviceTypeKey);
  writer.packFixStr(deviceTypeName);
}

void packMetrics(MsgPackWriter &writer,
                 const std::map<MetricKind, std::unique_ptr<Metric>> &metrics,
                 bool isRoot, tree_data_dump::MetricSummary &metricSummary,
                 const DeviceIdStrings &deviceIdStrings,
                 const DeviceTypeNames &deviceTypeNames) {
  for (const auto &[metricKind, metric] : metrics) {
    if (metricKind == MetricKind::Kernel) {
      if (isRoot) {
        writer.appendBytes(kKernelDurationKey);
        writer.packUInt(0);
        writer.appendBytes(kKernelInvocationsKey);
        writer.packUInt(0);
        continue;
      }

      packKernelMetricValues(writer, static_cast<KernelMetric *>(metric.get()),
                             metricSummary, deviceIdStrings, deviceTypeNames);
    } else if (metricKind == MetricKind::PCSampling) {
      auto *pcSamplingMetric = static_cast<PCSamplingMetric *>(metric.get());
      for (size_t i = 0; i < PCSamplingMetric::Count; i++) {
        const auto valueName = pcSamplingMetric->getValueName(i);
        writer.packStr(valueName);
        if (isRoot) {
          writer.packUInt(0);
        } else {
          writer.packUInt(std::get<uint64_t>(pcSamplingMetric->getValues()[i]));
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
      writer.packStr(std::to_string(deviceId));
      writer.packStr(CycleMetric::getValueName(CycleMetric::DeviceType));
      writer.packStr(std::to_string(deviceType));
    } else {
      throw makeLogicError("MetricKind not supported");
    }
  }
  if (isRoot) {
    if (metricSummary.hasKernelMetric &&
        metrics.find(MetricKind::Kernel) == metrics.end()) {
      writer.appendBytes(kKernelDurationKey);
      writer.packUInt(0);
      writer.appendBytes(kKernelInvocationsKey);
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
}

void packFlexibleMetrics(
    MsgPackWriter &writer,
    const std::map<std::string, FlexibleMetric> &flexibleMetrics) {
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
        flexibleMetric.getValues()[0]);
  }
}

} // namespace

std::vector<uint8_t>
TreeData::buildHatchetMsgPack(TreeData::Tree *tree,
                              TreeData::Tree *virtualTree) const {
  MsgPackWriter writer;
  writer.reserve(16 * 1024 * 1024); // 16 MB

  tree_data_dump::MetricSummary metricSummary;
  // Root metrics are serialized before descendants, so first scan the whole
  // concrete tree for fixed-schema metric kinds. This lets the root emit the
  // zero-valued Hatchet fields required for any metric kind present below it.
  tree->template walk<TreeData::Tree::WalkPolicy::PreOrder>(
      [&](TreeData::Tree::TreeNode &treeNode) {
        metricSummary.observeMetrics(treeNode.metricSet.metrics);
        for (const auto &[_, linkedMetrics] :
             treeNode.metricSet.linkedMetrics) {
          metricSummary.observeMetrics(linkedMetrics);
        }
      });
  const auto &virtualRootNode = virtualTree->getNode(Tree::TreeNode::RootId);
  std::array<std::string, tree_data_dump::kMaxRegisteredDeviceIds>
      deviceIdStrings;
  for (uint64_t deviceId = 0;
       deviceId < tree_data_dump::kMaxRegisteredDeviceIds; ++deviceId) {
    deviceIdStrings[deviceId] = std::to_string(deviceId);
  }
  std::array<std::string, static_cast<size_t>(DeviceType::COUNT)>
      deviceTypeNames;
  for (size_t deviceType = 0;
       deviceType < static_cast<size_t>(DeviceType::COUNT); ++deviceType) {
    deviceTypeNames[deviceType] =
        getDeviceTypeString(static_cast<DeviceType>(deviceType));
  }
  std::unordered_map<std::string_view, std::vector<uint8_t>> frameHeaderCache;
  auto packLinkedVirtualNode = [&](auto &&packLinkedVirtualNode,
                                   const TreeData::Tree::TreeNode &treeNode,
                                   size_t virtualNodeId) -> void {
    const auto &virtualNode = virtualTree->getNode(virtualNodeId);
    const auto &linkedMetrics = treeNode.metricSet.linkedMetrics;
    const auto &linkedFlexibleMetrics =
        treeNode.metricSet.linkedFlexibleMetrics;
    // Write the header
    packHatchetFrameHeader(writer, frameHeaderCache, virtualNode.name);
    // Count linked metrics
    auto metricEntries = 0u;
    const auto metricsIt = linkedMetrics.find(virtualNodeId);
    if (metricsIt != linkedMetrics.end()) {
      metricEntries += countMetricEntries(metricsIt->second, /*isRoot=*/false,
                                          metricSummary);
    }
    // Count linked flexible metrics exist in the child <metric> helpers
    if (!linkedFlexibleMetrics.empty()) {
      for (const auto &child : virtualNode.children) {
        auto it = linkedFlexibleMetrics.find(child.id);
        if (it != linkedFlexibleMetrics.end()) {
          metricEntries += static_cast<uint32_t>(it->second.size());
        }
      }
    }
    // Pack
    writer.packMap(metricEntries);
    if (metricsIt != linkedMetrics.end()) {
      packMetrics(writer, metricsIt->second, /*isRoot=*/false, metricSummary,
                  deviceIdStrings, deviceTypeNames);
    }
    if (!linkedFlexibleMetrics.empty()) {
      for (const auto &child : virtualNode.children) {
        auto it = linkedFlexibleMetrics.find(child.id);
        if (it != linkedFlexibleMetrics.end()) {
          packFlexibleMetrics(writer, it->second);
        }
      }
    }
    // Linked flexible metrics attached to generated helper leaves are promoted
    // into the parent metrics map above. Once promoted, a helper leaf with no
    // linked fixed metrics and no children carries no information in Hatchet.
    std::vector<size_t> linkedChildren;
    linkedChildren.reserve(virtualNode.children.size());
    for (const auto &child : virtualNode.children) {
      const auto &childNode = virtualTree->getNode(child.id);
      if (!childNode.children.empty() ||
          linkedMetrics.find(child.id) != linkedMetrics.end()) {
        linkedChildren.push_back(child.id);
      }
    }
    writer.packFixStr("children");
    writer.packArray(static_cast<uint32_t>(linkedChildren.size()));
    for (auto childId : linkedChildren) {
      packLinkedVirtualNode(packLinkedVirtualNode, treeNode, childId);
    }
  };
  auto packNode = [&](auto &&packNode,
                      TreeData::Tree::TreeNode &treeNode) -> void {
    // Write the header
    packHatchetFrameHeader(writer, frameHeaderCache, treeNode.name);
    const bool isRoot = treeNode.id == TreeData::Tree::TreeNode::RootId;
    // Write the concrete nodes' own metrics and flexible metrics
    writer.packMap(
        countMetricEntries(treeNode.metricSet.metrics, isRoot, metricSummary) +
        static_cast<uint32_t>(treeNode.metricSet.flexibleMetrics.size()));
    packMetrics(writer, treeNode.metricSet.metrics, isRoot, metricSummary,
                deviceIdStrings, deviceTypeNames);
    packFlexibleMetrics(writer, treeNode.metricSet.flexibleMetrics);

    const bool hasLinkedTargets =
        !treeNode.metricSet.linkedMetrics.empty() ||
        !treeNode.metricSet.linkedFlexibleMetrics.empty();
    uint32_t linkedChildCount =
        hasLinkedTargets
            ? static_cast<uint32_t>(virtualRootNode.children.size())
            : 0;
    // CUDA stream capture can create concrete launch-name leaves before the
    // launch callback exits early without correlating metrics. Graph replay
    // metrics are attached through linked virtual nodes instead, so a concrete
    // leaf with no metrics, linked metrics, flexible metrics, or children adds
    // no Hatchet information.
    std::vector<TreeData::Tree::TreeNode *> concreteChildren;
    concreteChildren.reserve(treeNode.children.size());
    for (const auto &child : treeNode.children) {
      auto &childNode = tree->getNode(child.id);
      if (!childNode.children.empty() || !childNode.metricSet.metrics.empty() ||
          !childNode.metricSet.flexibleMetrics.empty() ||
          !childNode.metricSet.linkedMetrics.empty() ||
          !childNode.metricSet.linkedFlexibleMetrics.empty()) {
        concreteChildren.push_back(&childNode);
      }
    }
    writer.packFixStr("children");
    writer.packArray(static_cast<uint32_t>(concreteChildren.size()) +
                     linkedChildCount);
    for (auto *childNode : concreteChildren) {
      packNode(packNode, *childNode);
    }
    if (hasLinkedTargets) {
      for (const auto &virtualChild : virtualRootNode.children) {
        packLinkedVirtualNode(packLinkedVirtualNode, treeNode, virtualChild.id);
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
    for (uint64_t deviceId = 0;
         deviceId < tree_data_dump::kMaxRegisteredDeviceIds; ++deviceId) {
      if ((mask & (1u << static_cast<uint32_t>(deviceId))) == 0) {
        continue;
      }
      Device device = getDevice(static_cast<DeviceType>(deviceType), deviceId);
      writer.packStr(std::to_string(deviceId));
      writer.packMap(5);
      writer.packFixStr("clock_rate");
      writer.packUInt(device.clockRate);
      writer.packFixStr("memory_clock_rate");
      writer.packUInt(device.memoryClockRate);
      writer.packFixStr("bus_width");
      writer.packUInt(device.busWidth);
      writer.packFixStr("arch");
      writer.packStr(device.arch);
      writer.packFixStr("num_sms");
      writer.packUInt(device.numSms);
    }
  }

  return std::move(writer).take();
}

} // namespace proton
