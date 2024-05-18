#include "Driver/Device.h"
#include "Driver/GPU/Cuda.h"

#include "Utility/Errors.h"
#include "Utility/String.h"

namespace proton {

Device getDevice(DeviceType type, uint64_t index) {
  if (type == DeviceType::CUDA) {
    return cuda::getDevice(index);
  } else if (type == DeviceType::HIP) {
    throw NotImplemented();
  }
  throw std::runtime_error("DeviceType not supported");
}

const std::string getDeviceTypeString(DeviceType type) {
  if (type == DeviceType::CUDA) {
    return DeviceTraits<DeviceType::CUDA>::name;
  } else if (type == DeviceType::HIP) {
    return DeviceTraits<DeviceType::HIP>::name;
  }
  throw std::runtime_error("DeviceType not supported");
}

} // namespace proton
