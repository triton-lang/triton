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
