#include "Driver/GPU/CudaApi.h"
#include "Driver/Dispatch.h"

namespace proton {

namespace cuda {

struct ExternLibCuda : public ExternLibBase {
  using RetType = CUresult;
  // https://forums.developer.nvidia.com/t/wsl2-libcuda-so-and-libcuda-so-1-should-be-symlink/236301
  // On WSL, "libcuda.so" and "libcuda.so.1" may not be linked, so we use
  // "libcuda.so.1" instead.
  static constexpr const char *name = "libcuda.so.1";
  static constexpr RetType success = CUDA_SUCCESS;
  static void *lib;
};

void *ExternLibCuda::lib = nullptr;

DEFINE_DISPATCH(ExternLibCuda, init, cuInit, int)

DEFINE_DISPATCH(ExternLibCuda, ctxSynchronize, cuCtxSynchronize)

DEFINE_DISPATCH(ExternLibCuda, ctxGetCurrent, cuCtxGetCurrent, CUcontext *)

DEFINE_DISPATCH(ExternLibCuda, deviceGet, cuDeviceGet, CUdevice *, int)

DEFINE_DISPATCH(ExternLibCuda, deviceGetAttribute, cuDeviceGetAttribute, int *,
                CUdevice_attribute, CUdevice)

Device getDevice(uint64_t index) {
  CUdevice device;
  cuda::deviceGet<true>(&device, index);
  int clockRate;
  cuda::deviceGetAttribute<true>(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                 device);
  int memoryClockRate;
  cuda::deviceGetAttribute<true>(&memoryClockRate,
                                 CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device);
  int busWidth;
  cuda::deviceGetAttribute<true>(
      &busWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device);
  int numSms;
  cuda::deviceGetAttribute<true>(
      &numSms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
  int major;
  cuda::deviceGetAttribute<true>(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  int minor;
  cuda::deviceGetAttribute<true>(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  auto arch = major * 10 + minor;

  std::string archName = getCudaArchName(arch);
  return Device(DeviceType::CUDA, index, clockRate, memoryClockRate, busWidth,
                numSms, arch, archName);
}

const std::string getCudaArchName(int arch) {
  if (arch == 80)
    return "Ampere";
  else if (arch == 89)
    return "Ada";
  else if (arch == 90)
    return "Hopper";
  else
    return "Unknown";
}

} // namespace cuda

} // namespace proton
