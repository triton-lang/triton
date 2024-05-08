#include "Driver/Device.h"
#include "Driver/GPU/Cuda.h"

#include "Utility/Errors.h"
#include "Utility/String.h"

#include <iostream>

namespace proton {

namespace {

Device getCudaDevice(uint64_t index) {
  CUcontext context;
  CUdevice device;
  cuda::ctxGetCurrent<true>(&context);
  cuda::deviceGet<true>(&device, index);
  if (!context) {
    cuda::devicePrimaryCtxRetain<true>(&context, device);
    cuda::ctxSetCurrent<true>(context);
  }
  int clockRate;
  cuda::deviceGetAttribute<true>(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                 device);
  int memoryClockRate;
  cuda::deviceGetAttribute<true>(&memoryClockRate,
                                 CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device);
  int busWidth;
  cuda::deviceGetAttribute<true>(
      &busWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device);
  return Device(DeviceType::CUDA, index, clockRate, memoryClockRate, busWidth);
}

} // namespace

Device getDevice(DeviceType type, uint64_t index) {
  if (type == DeviceType::CUDA) {
    return getCudaDevice(index);
  } else if (type == DeviceType::ROCM) {
    throw NotImplemented();
  }
  throw std::runtime_error("DeviceType not supported");
}

const std::string getDeviceTypeString(DeviceType type) {
  if (type == DeviceType::CUDA) {
    return DeviceTraits<DeviceType::CUDA>::name;
  } else if (type == DeviceType::ROCM) {
    return DeviceTraits<DeviceType::ROCM>::name;
  }
  throw std::runtime_error("DeviceType not supported");
}

} // namespace proton
