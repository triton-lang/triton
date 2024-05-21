#include "Driver/GPU/Hip.h"
#include "Driver/Dispatch.h"
#include <hip/hip_runtime_api.h>
#include <string>

namespace proton {

namespace hip {

struct ExternLibHip : public ExternLibBase {
  using RetType = hipError_t;
  static constexpr const char *name = "libamdhip64.so";
  static constexpr RetType success = hipSuccess;
  static void *lib;
};

void *ExternLibHip::lib = nullptr;

DEFINE_DISPATCH(ExternLibHip, deviceSynchronize, hipDeviceSynchronize)

DEFINE_DISPATCH(ExternLibHip, deviceGetAttribute, hipDeviceGetAttribute, int *,
                hipDeviceAttribute_t, int);

Device getDevice(uint64_t index) {
  int clockRate;
  (void)hip::deviceGetAttribute<true>(&clockRate, hipDeviceAttributeClockRate,
                                      index);
  int memoryClockRate;
  (void)hip::deviceGetAttribute<true>(&memoryClockRate,
                                      hipDeviceAttributeMemoryClockRate, index);
  int busWidth;
  (void)hip::deviceGetAttribute<true>(&busWidth,
                                      hipDeviceAttributeMemoryBusWidth, index);
  int smCount;
  (void)hip::deviceGetAttribute<true>(
      &smCount, hipDeviceAttributeMultiprocessorCount, index);

  // TODO: Compute capability is a NVIDIA concept. It doesn't map naturally to
  // AMD GPUs. Figure out a better way to support this.
  uint64_t arch = 0;
  return Device(DeviceType::HIP, index, clockRate, memoryClockRate, busWidth,
                smCount, arch);
}

} // namespace hip

} // namespace proton
