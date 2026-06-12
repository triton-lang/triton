#ifndef PROTON_BACKEND_BACKEND_H_
#define PROTON_BACKEND_BACKEND_H_

#include "Device.h"
#include "DeviceType.h"
#include "Runtime/Runtime.h"
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace proton {

class Profiler;

// A Proton backend registers itself by exposing a registerProtonBackend()
// function that returns a BackendRegistration. The generated
// RegisteredBackends.cpp file calls those hooks for all linked backends, and
// Backend.cpp folds the optional profiler/device/runtime records into the
// public getProton* registries alongside Proton's built-in entries.
// Backends may provide any subset of these records.
//
// Profiler: an implemention of proton::Profiler, for example
// proton::CuptiProfiler. Profilers are responsible for collecting profiling
// information and making it available to Proton.
//
// Device: a physical device which is present in the DeviceType enum and
// provides a getDevice function to retrieve information about the device. For
// example DeviceType::Cuda. To extend the DeviceType enum with a new device,
// use the cmake add_proton_device_type function.
//
// Runtime: an implementaiton of proton::Runtime, for example
// proton::CudaRuntime. A runtime is responsible for allocating buffers and
// interacting with the device when using the intrumentation backend.
//
// Backend: a collection of optional profiler/device/runtime implementations
// which are associated together.
struct ProfilerRegistration {
public:
  ProfilerRegistration(std::string name,
                       std::optional<std::string> tritonBackend,
                       std::function<Profiler *()> getInstance)
      : name(std::move(name)), tritonBackend(std::move(tritonBackend)),
        getProfilerInstance(std::move(getInstance)) {}

  const std::string &getName() const { return name; }

  const std::optional<std::string> &getTritonBackend() const {
    return tritonBackend;
  }

  const std::function<Profiler *()> &getInstance() const {
    return getProfilerInstance;
  }

private:
  std::string name;
  std::optional<std::string> tritonBackend;
  std::function<Profiler *()> getProfilerInstance;
};

struct DeviceRegistration {
public:
  DeviceRegistration(std::string name, DeviceType deviceType,
                     std::function<Device(uint64_t)> getDevice)
      : name(std::move(name)), deviceType(deviceType),
        getDeviceFn(std::move(getDevice)) {}

  const std::string &getName() const { return name; }

  const DeviceType &getDeviceType() const { return deviceType; }

  const std::function<Device(uint64_t)> &getDevice() const {
    return getDeviceFn;
  }

private:
  std::string name;
  DeviceType deviceType;
  std::function<Device(uint64_t)> getDeviceFn;
};

struct RuntimeRegistration {
public:
  RuntimeRegistration(std::string deviceName,
                      std::function<Runtime *()> getInstance)
      : deviceName(std::move(deviceName)),
        getRuntimeInstance(std::move(getInstance)) {}

  const std::string &getDeviceName() const { return deviceName; }

  const std::function<Runtime *()> &getInstance() const {
    return getRuntimeInstance;
  }

private:
  std::string deviceName;
  std::function<Runtime *()> getRuntimeInstance;
};

struct BackendRegistration {
public:
  BackendRegistration(std::optional<ProfilerRegistration> profiler = {},
                      std::optional<DeviceRegistration> device = {},
                      std::optional<RuntimeRegistration> runtime = {})
      : profiler(std::move(profiler)), device(std::move(device)),
        runtime(std::move(runtime)) {}

  const std::optional<ProfilerRegistration> &getProfiler() const {
    return profiler;
  }

  const std::optional<DeviceRegistration> &getDevice() const { return device; }

  const std::optional<RuntimeRegistration> &getRuntime() const {
    return runtime;
  }

private:
  std::optional<ProfilerRegistration> profiler;
  std::optional<DeviceRegistration> device;
  std::optional<RuntimeRegistration> runtime;
};

const std::vector<BackendRegistration> &getBackendRegistrations();
const std::vector<ProfilerRegistration> getProfilerRegistrations();
const std::vector<DeviceRegistration> getDeviceRegistrations();
const std::vector<RuntimeRegistration> getRuntimeRegistrations();
const std::vector<std::string> getRegisteredProfilerNames();
const std::optional<std::string>
getProfilerForTritonBackend(const std::string &);

} // namespace proton

#endif // PROTON_BACKEND_BACKEND_H_
