#ifndef PROTON_DATA_DUMP_TREE_DATA_DUMP_H_
#define PROTON_DATA_DUMP_TREE_DATA_DUMP_H_

#include "Data/Metric.h"
#include "Data/TreeData.h"
#include "DeviceType.h"
#include "Utility/Errors.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

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

} // namespace proton

#endif // PROTON_DATA_DUMP_TREE_DATA_DUMP_H_
