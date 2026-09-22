#include "Device.h"
#include "Backend/Backend.h"
#include "DeviceType.h"

#include "Utility/Errors.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace proton {
namespace {

const DeviceRegistration getDeviceEntry(DeviceType type) {
  const auto devices = getDeviceRegistrations();
  auto itr = std::find_if(devices.begin(), devices.end(),
                          [&](const DeviceRegistration &entry) {
                            return type == entry.getDeviceType();
                          });
  if (itr == devices.end()) {
    throw makeInvalidArgument("DeviceType not supported");
  }
  return *itr;
}

} // namespace

Device getDevice(DeviceType type, uint64_t index) {
  return getDeviceEntry(type).getDevice()(index);
}

const std::string getDeviceTypeString(DeviceType type) {
  return getDeviceEntry(type).getName();
}

} // namespace proton
