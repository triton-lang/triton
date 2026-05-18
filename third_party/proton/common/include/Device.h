#ifndef PROTON_COMMON_DEVICE_H_
#define PROTON_COMMON_DEVICE_H_

#include "DeviceType.h"
#include <cstdint>
#include <string>

namespace proton {

struct Device {
  DeviceType type;
  uint64_t id;
  uint64_t clockRate;       // khz
  uint64_t memoryClockRate; // khz
  uint64_t busWidth;
  uint64_t numSms;
  std::string arch;

  Device() = default;

  Device(DeviceType type, uint64_t id, uint64_t clockRate,
         uint64_t memoryClockRate, uint64_t busWidth, uint64_t numSms,
         std::string arch)
      : type(type), id(id), clockRate(clockRate),
        memoryClockRate(memoryClockRate), busWidth(busWidth), numSms(numSms),
        arch(arch) {}
};

Device getDevice(DeviceType type, uint64_t index);

const std::string getDeviceTypeString(DeviceType type);

}; // namespace proton

#endif // PROTON_COMMON_DEVICE_H_
