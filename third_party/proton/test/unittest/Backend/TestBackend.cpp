#include "Backend/Backend.h"

namespace TestBackend {

proton::Device getTestDevice(uint64_t index) { return {}; }

proton::BackendRegistration registerProtonBackend() {
  return {
      proton::ProfilerRegistration{
          "test_backend", "test_triton_backend",
          []() -> proton::Profiler * { return nullptr; }},
      proton::DeviceRegistration{
          "TEST_DEVICE", proton::DeviceType::CUDA,
          [](uint64_t index) { return getTestDevice(index); }},
      proton::RuntimeRegistration{
          "TEST_DEVICE", []() -> proton::Runtime * { return nullptr; }},
  };
}

} // namespace TestBackend
