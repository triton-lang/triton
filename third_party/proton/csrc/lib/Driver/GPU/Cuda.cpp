#include "Driver/GPU/Cuda.h"
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

} // namespace cuda

} // namespace proton
