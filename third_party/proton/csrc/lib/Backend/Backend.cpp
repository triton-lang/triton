#include "Backend/Backend.h"
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/HipApi.h"
#include "Profiler/Cupti/CuptiProfiler.h"
#include "Profiler/Instrumentation/InstrumentationProfiler.h"
#include "Profiler/Roctracer/RoctracerProfiler.h"
#include "Runtime/CudaRuntime.h"
#include "Runtime/HipRuntime.h"
#include <vector>

namespace proton {

const std::vector<ProfilerRegistration> &getProtonProfilers() {
  static const std::vector<ProfilerRegistration> registeredProfilers = []() {
    std::vector<ProfilerRegistration> entries{
        {"cupti", "cuda", []() { return &CuptiProfiler::instance(); }},
        {"roctracer", "hip", []() { return &RoctracerProfiler::instance(); }},
        {"instrumentation",
         {},
         []() { return &InstrumentationProfiler::instance(); }},
    };
    for (const auto &backend : getBackendRegistrations()) {
      const auto &profiler = backend.getProfiler();
      if (profiler)
        entries.push_back(*profiler);
    }
    return entries;
  }();
  return registeredProfilers;
}

const std::vector<DeviceRegistration> &getProtonDevices() {
  static const std::vector<DeviceRegistration> registeredDevices = []() {
    std::vector<DeviceRegistration> entries{
        {"CUDA", DeviceType::CUDA,
         [](uint64_t index) { return cuda::getDevice(index); }},
        {"HIP", DeviceType::HIP,
         [](uint64_t index) { return hip::getDevice(index); }},
    };
    for (const auto &backend : getBackendRegistrations()) {
      const auto &device = backend.getDevice();
      if (device)
        entries.push_back(*device);
    }
    return entries;
  }();
  return registeredDevices;
}

const std::vector<RuntimeRegistration> &getProtonRuntimes() {
  static const std::vector<RuntimeRegistration> registeredRuntimes = []() {
    std::vector<RuntimeRegistration> entries{
        {"CUDA", []() { return &CudaRuntime::instance(); }},
        {"HIP", []() { return &HipRuntime::instance(); }},
    };
    for (const auto &backend : getBackendRegistrations()) {
      const auto &runtime = backend.getRuntime();
      if (runtime) {
        entries.push_back(*runtime);
      }
    }
    return entries;
  }();
  return registeredRuntimes;
}

} // namespace proton
