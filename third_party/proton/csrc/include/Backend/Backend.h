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
struct ProfilerRegistration {
public:
  ProfilerRegistration(
      std::string name,
      std::optional<std::string> correspondingTritonDriverBackend,
      std::function<Profiler *()> getInstance)
      : name_(std::move(name)), correspondingTritonDriverBackend_(std::move(
                                    correspondingTritonDriverBackend)),
        getInstance_(std::move(getInstance)) {}

  const std::string &getName() const { return name_; }

  const std::optional<std::string> &
  getCorrespondingTritonDriverBackend() const {
    return correspondingTritonDriverBackend_;
  }

  const std::function<Profiler *()> &getInstance() const {
    return getInstance_;
  }

private:
  std::string name_;
  std::optional<std::string> correspondingTritonDriverBackend_;
  std::function<Profiler *()> getInstance_;
};

struct DeviceRegistration {
public:
  DeviceRegistration(std::string name, DeviceType deviceType,
                     std::function<Device(uint64_t)> getDevice)
      : name_(std::move(name)), deviceType_(deviceType),
        getDevice_(std::move(getDevice)) {}

  const std::string &getName() const { return name_; }

  const DeviceType &getDeviceType() const { return deviceType_; }

  const std::function<Device(uint64_t)> &getDevice() const {
    return getDevice_;
  }

private:
  std::string name_;
  DeviceType deviceType_;
  std::function<Device(uint64_t)> getDevice_;
};

struct RuntimeRegistration {
public:
  RuntimeRegistration(std::string deviceName,
                      std::function<Runtime *()> getInstance)
      : deviceName_(std::move(deviceName)),
        getInstance_(std::move(getInstance)) {}

  const std::string &getDeviceName() const { return deviceName_; }

  const std::function<Runtime *()> &getInstance() const { return getInstance_; }

private:
  std::string deviceName_;
  std::function<Runtime *()> getInstance_;
};

struct BackendRegistration {
public:
  BackendRegistration(std::optional<ProfilerRegistration> profiler = {},
                      std::optional<DeviceRegistration> device = {},
                      std::optional<RuntimeRegistration> runtime = {})
      : profiler_(std::move(profiler)), device_(std::move(device)),
        runtime_(std::move(runtime)) {}

  const std::optional<ProfilerRegistration> &getProfiler() const {
    return profiler_;
  }

  const std::optional<DeviceRegistration> &getDevice() const { return device_; }

  const std::optional<RuntimeRegistration> &getRuntime() const {
    return runtime_;
  }

private:
  std::optional<ProfilerRegistration> profiler_;
  std::optional<DeviceRegistration> device_;
  std::optional<RuntimeRegistration> runtime_;
};

const std::vector<BackendRegistration> &getBackendRegistrations();

const std::vector<ProfilerRegistration> &getProtonProfilers();
const std::vector<DeviceRegistration> &getProtonDevices();
const std::vector<RuntimeRegistration> &getProtonRuntimes();

} // namespace proton

#endif // PROTON_BACKEND_BACKEND_H_
