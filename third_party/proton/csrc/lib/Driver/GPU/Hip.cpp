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

DEFINE_DISPATCH(ExternLibHip, getDeviceProperties, hipGetDeviceProperties,
                hipDeviceProp_tR0000 *, int)

Device getDevice(uint64_t index) {
  hipDeviceProp_tR0000 props;
  (void)hip::getDeviceProperties<true>(&props, index);

  // Parse the gfxNNN arch name to drop the prefix and potential suffix.
  const char *archName = props.gcnArchName;
  uint64_t arch = std::stoi(archName + /*gfx*/ 3);
  return Device(DeviceType::HIP, index, props.clockRate, props.memoryClockRate,
                props.memoryBusWidth, props.multiProcessorCount, arch);
}

} // namespace hip

} // namespace proton
