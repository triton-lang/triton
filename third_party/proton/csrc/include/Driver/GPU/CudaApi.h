#ifndef PROTON_DRIVER_GPU_CUDA_H_
#define PROTON_DRIVER_GPU_CUDA_H_

#include "Device.h"
#include "cuda.h"

namespace proton {

namespace cuda {

template <bool CheckSuccess> CUresult init(int flags);

template <bool CheckSuccess> CUresult ctxSynchronize();

template <bool CheckSuccess> CUresult ctxGetCurrent(CUcontext *pctx);

template <bool CheckSuccess> CUresult ctxGetDevice(CUdevice *device);

template <bool CheckSuccess>
CUresult ctxGetStreamPriorityRange(int *leastPriority, int *greatestPriority);

template <bool CheckSuccess>
CUresult deviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);

template <bool CheckSuccess> CUresult deviceGet(CUdevice *device, int ordinal);

template <bool CheckSuccess>
CUresult streamCreateWithPriority(CUstream *pStream, unsigned int flags,
                                  int priority);

template <bool CheckSuccess> CUresult streamSynchronize(CUstream stream);

template <bool CheckSuccess>
CUresult memcpyDToHAsync(void *dst, CUdeviceptr src, size_t count,
                         CUstream stream);

template <bool CheckSuccess> CUresult memAllocHost(void **pp, size_t bytesize);

template <bool CheckSuccess> CUresult memFreeHost(void *p);

Device getDevice(uint64_t index);

} // namespace cuda

} // namespace proton

#endif // PROTON_DRIVER_GPU_CUDA_H_
