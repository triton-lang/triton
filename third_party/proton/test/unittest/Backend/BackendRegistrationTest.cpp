#include "Backend/Backend.h"
#include "DeviceType.h"
#include <gtest/gtest.h>

TEST(BackendRegistrationTest, testBackendRegistration) {
  const auto &registrations = proton::getBackendRegistrations();
  ASSERT_EQ(registrations.size(), 1u);
  const auto &registration = registrations.front();

  ASSERT_TRUE(registration.getDevice());
  EXPECT_EQ(registration.getDevice()->getName(), "TEST_DEVICE");
  EXPECT_EQ(registration.getDevice()->getDeviceType(),
            proton::DeviceType::CUDA);

  ASSERT_TRUE(registration.getProfiler());
  EXPECT_EQ(registration.getProfiler()->getName(), "test_backend");
  EXPECT_EQ(registration.getProfiler()->getTritonBackend(),
            "test_triton_backend");

  ASSERT_TRUE(registration.getRuntime());
  EXPECT_EQ(registration.getRuntime()->getDeviceName(), "TEST_DEVICE");
}

TEST(BackendRegistrationTest, testRuntimeRegistration) {
  const auto numBuiltIn = proton::getBackendRegistrations().size();
  proton::registerBackend({
      proton::ProfilerRegistration{
          "runtime_backend", "runtime_triton_backend",
          []() -> proton::Profiler * { return nullptr; }},
      proton::DeviceRegistration{
          "RUNTIME_DEVICE", proton::DeviceType::EXTERNAL,
          [](uint64_t index) { return proton::Device{}; }},
  });

  const auto &registrations = proton::getBackendRegistrations();
  ASSERT_EQ(registrations.size(), numBuiltIn + 1);
  // Backends built into Proton come first.
  EXPECT_EQ(registrations.front().getProfiler()->getName(), "test_backend");
  const auto &registration = registrations.back();
  ASSERT_TRUE(registration.getProfiler());
  EXPECT_EQ(registration.getProfiler()->getName(), "runtime_backend");
  ASSERT_TRUE(registration.getDevice());
  EXPECT_EQ(registration.getDevice()->getDeviceType(),
            proton::DeviceType::EXTERNAL);
  EXPECT_FALSE(registration.getRuntime());
}
