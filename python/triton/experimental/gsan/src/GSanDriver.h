#pragma once

// Thin driver shim so GSanAllocator.cc can target either the CUDA driver API
// or HIP. Addresses are uintptr_t on both sides; HIP's void* addresses are
// produced only at the call boundary. Allocation properties are always the
// same (pinned, on `device`, exportable as `handleType`), so they are built
// here rather than at every call site.

#include <cstddef>
#include <cstdint>

#if defined(GSAN_BACKEND_HIP)
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime_api.h>
#else
#include <cuda.h>
#endif

namespace gsan::drv {

#if defined(GSAN_BACKEND_HIP)

using Result = hipError_t;
using Handle = hipMemGenericAllocationHandle_t;
using Stream = hipStream_t;

inline constexpr Result kSuccess = hipSuccess;
inline constexpr Result kErrorNotInitialized = hipErrorNotInitialized;
inline constexpr Result kErrorInvalidDevice = hipErrorInvalidDevice;
inline constexpr Result kErrorInvalidValue = hipErrorInvalidValue;
inline constexpr Result kErrorNoDevice = hipErrorNoDevice;
inline constexpr Result kErrorNotSupported = hipErrorNotSupported;

#else

using Result = CUresult;
using Handle = CUmemGenericAllocationHandle;
using Stream = CUstream;

inline constexpr Result kSuccess = CUDA_SUCCESS;
inline constexpr Result kErrorNotInitialized = CUDA_ERROR_NOT_INITIALIZED;
inline constexpr Result kErrorInvalidDevice = CUDA_ERROR_INVALID_DEVICE;
inline constexpr Result kErrorInvalidValue = CUDA_ERROR_INVALID_VALUE;
inline constexpr Result kErrorNoDevice = CUDA_ERROR_NO_DEVICE;
inline constexpr Result kErrorNotSupported = CUDA_ERROR_NOT_SUPPORTED;

#endif

// Values match ShareableHandleType in _allocator.py and the CUDA enum.
enum class HandleType : int {
  PosixFileDescriptor = 0x1,
  Fabric = 0x8,
};

#if defined(GSAN_BACKEND_HIP)
// HIP has no fabric handles. Keep the union in GSanAllocator.cc the same size.
struct FabricHandle {
  unsigned char data[64];
};
inline constexpr bool kHasFabricHandles = false;
#else
using FabricHandle = CUmemFabricHandle;
inline constexpr bool kHasFabricHandles = true;
#endif

inline bool isSupportedHandleType(HandleType type) {
  return type == HandleType::PosixFileDescriptor ||
         (kHasFabricHandles && type == HandleType::Fabric);
}

#if defined(GSAN_BACKEND_HIP)

inline void *ptr(uintptr_t addr) { return reinterpret_cast<void *>(addr); }

inline hipMemAllocationProp makeProp(int device, HandleType type) {
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  prop.requestedHandleType = static_cast<hipMemAllocationHandleType>(type);
  return prop;
}

inline const char *errorString(Result err) { return hipGetErrorString(err); }

inline Result deviceCount(int *count) { return hipGetDeviceCount(count); }

inline Result multiprocessorCount(int device, int *count) {
  return hipDeviceGetAttribute(count, hipDeviceAttributeMultiprocessorCount,
                               device);
}

inline Result supportsFabricHandles(int /*device*/, bool *supported) {
  *supported = false;
  return kSuccess;
}

inline Result allocationGranularity(int device, HandleType type,
                                    size_t *granularity) {
  auto prop = makeProp(device, type);
  return hipMemGetAllocationGranularity(granularity, &prop,
                                        hipMemAllocationGranularityMinimum);
}

// ROCm accepts but ignores the alignment argument, while GSan's address math
// relies on size-aligned reservations. Over-reserve and return the aligned
// subrange; reservations are never freed, so the slack is harmless.
inline Result addressReserve(uintptr_t *addr, size_t size, size_t alignment) {
  void *base = nullptr;
  Result err =
      hipMemAddressReserve(&base, size + alignment, alignment, nullptr, 0);
  auto raw = reinterpret_cast<uintptr_t>(base);
  *addr = alignment ? (raw + alignment - 1) & ~(uintptr_t(alignment) - 1) : raw;
  return err;
}

inline Result memCreate(Handle *handle, size_t size, int device,
                        HandleType type) {
  auto prop = makeProp(device, type);
  return hipMemCreate(handle, size, &prop, 0);
}

inline Result memMap(uintptr_t addr, size_t size, Handle handle) {
  return hipMemMap(ptr(addr), size, /*offset*/ 0, handle, /*flags*/ 0);
}

inline Result memSetAccess(uintptr_t addr, size_t size, int device) {
  hipMemAccessDesc desc = {};
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = device;
  desc.flags = hipMemAccessFlagsProtReadWrite;
  return hipMemSetAccess(ptr(addr), size, &desc, 1);
}

inline Result memUnmap(uintptr_t addr, size_t size) {
  return hipMemUnmap(ptr(addr), size);
}

inline Result memRelease(Handle handle) { return hipMemRelease(handle); }

inline Result exportHandle(void *out, Handle handle, HandleType type) {
  return hipMemExportToShareableHandle(
      out, handle, static_cast<hipMemAllocationHandleType>(type), 0);
}

inline Result importHandle(Handle *handle, void *osHandle, HandleType type) {
  return hipMemImportFromShareableHandle(
      handle, osHandle, static_cast<hipMemAllocationHandleType>(type));
}

inline Result memsetD8(uintptr_t addr, size_t size) {
  return hipMemsetD8(ptr(addr), 0, size);
}

inline Result memsetD8Async(uintptr_t addr, size_t size, Stream stream) {
  return hipMemsetD8Async(ptr(addr), 0, size, stream);
}

inline Result memcpyHtoD(uintptr_t dst, const void *src, size_t size) {
  return hipMemcpyHtoD(ptr(dst), src, size);
}

inline Result streamSynchronize(Stream stream) {
  return hipStreamSynchronize(stream);
}

// Makes `device` current for the guard's lifetime and restores the previous
// device afterwards. HIP's context API is deprecated, so use the device API.
class DeviceGuard {
public:
  Result enter(int device) {
    Result err = hipGetDevice(&previous);
    if (err != kSuccess)
      return err;
    entered = true;
    return hipSetDevice(device);
  }
  Result synchronize() { return hipDeviceSynchronize(); }
  Result exit() {
    if (!entered)
      return kSuccess;
    entered = false;
    return hipSetDevice(previous);
  }
  ~DeviceGuard() { (void)exit(); }

private:
  int previous = 0;
  bool entered = false;
};

// HIP has no "no current context" state and every VMM call above names its
// device explicitly, so only make sure the runtime is initialized. Unlike
// hipSetDevice, this leaves the caller's current device alone.
inline Result ensureCurrentDevice(int /*device*/) { return hipInit(0); }

#else // CUDA

inline CUmemAllocationProp makeProp(int device, HandleType type) {
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.requestedHandleTypes = static_cast<CUmemAllocationHandleType>(type);
  return prop;
}

inline const char *errorString(Result err) {
  const char *msg = "<unknown error>";
  cuGetErrorString(err, &msg);
  return msg;
}

inline Result deviceCount(int *count) { return cuDeviceGetCount(count); }

inline Result multiprocessorCount(int device, int *count) {
  CUdevice cuDevice = 0;
  Result err = cuDeviceGet(&cuDevice, device);
  if (err != kSuccess)
    return err;
  return cuDeviceGetAttribute(count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                              cuDevice);
}

inline Result supportsFabricHandles(int device, bool *supported) {
  CUdevice cuDevice = 0;
  Result err = cuDeviceGet(&cuDevice, device);
  if (err != kSuccess)
    return err;
  int value = 0;
  err = cuDeviceGetAttribute(
      &value, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, cuDevice);
  *supported = value != 0;
  return err;
}

inline Result allocationGranularity(int device, HandleType type,
                                    size_t *granularity) {
  auto prop = makeProp(device, type);
  return cuMemGetAllocationGranularity(granularity, &prop,
                                       CU_MEM_ALLOC_GRANULARITY_MINIMUM);
}

inline Result addressReserve(uintptr_t *addr, size_t size, size_t alignment) {
  CUdeviceptr base = 0;
  Result err = cuMemAddressReserve(&base, size, alignment, /*addr*/ 0, 0);
  *addr = static_cast<uintptr_t>(base);
  return err;
}

inline Result memCreate(Handle *handle, size_t size, int device,
                        HandleType type) {
  auto prop = makeProp(device, type);
  return cuMemCreate(handle, size, &prop, 0);
}

inline Result memMap(uintptr_t addr, size_t size, Handle handle) {
  return cuMemMap(addr, size, /*offset*/ 0, handle, /*flags*/ 0);
}

inline Result memSetAccess(uintptr_t addr, size_t size, int device) {
  CUmemAccessDesc desc = {};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = device;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  return cuMemSetAccess(addr, size, &desc, 1);
}

inline Result memUnmap(uintptr_t addr, size_t size) {
  return cuMemUnmap(addr, size);
}

inline Result memRelease(Handle handle) { return cuMemRelease(handle); }

inline Result exportHandle(void *out, Handle handle, HandleType type) {
  return cuMemExportToShareableHandle(
      out, handle, static_cast<CUmemAllocationHandleType>(type), 0);
}

inline Result importHandle(Handle *handle, void *osHandle, HandleType type) {
  return cuMemImportFromShareableHandle(
      handle, osHandle, static_cast<CUmemAllocationHandleType>(type));
}

inline Result memsetD8(uintptr_t addr, size_t size) {
  return cuMemsetD8(addr, 0, size);
}

inline Result memsetD8Async(uintptr_t addr, size_t size, Stream stream) {
  return cuMemsetD8Async(addr, 0, size, stream);
}

inline Result memcpyHtoD(uintptr_t dst, const void *src, size_t size) {
  return cuMemcpyHtoD(dst, src, size);
}

inline Result streamSynchronize(Stream stream) {
  return cuStreamSynchronize(stream);
}

// Retains `device`'s primary context and makes it current for the guard's
// lifetime, then restores the previous context and releases the retain.
class DeviceGuard {
public:
  Result enter(int device) {
    Result err = cuCtxGetCurrent(&previous);
    if (err != kSuccess)
      return err;
    err = cuDevicePrimaryCtxRetain(&context, device);
    if (err != kSuccess)
      return err;
    retainedDevice = device;
    return cuCtxSetCurrent(context);
  }
  Result synchronize() { return cuCtxSynchronize(); }
  Result exit() {
    if (retainedDevice < 0)
      return kSuccess;
    Result err = cuCtxSetCurrent(previous);
    Result releaseErr = cuDevicePrimaryCtxRelease(retainedDevice);
    retainedDevice = -1;
    return err != kSuccess ? err : releaseErr;
  }
  ~DeviceGuard() { (void)exit(); }

private:
  CUcontext previous = nullptr;
  CUcontext context = nullptr;
  int retainedDevice = -1;
};

// Malloc may be called before anything has made a context current; adopt the
// primary context in that case and leave an existing context alone.
inline Result ensureCurrentDevice(int device) {
  CUcontext ctx = nullptr;
  Result err = cuCtxGetCurrent(&ctx);
  if (err != kSuccess || ctx)
    return err;
  err = cuDevicePrimaryCtxRetain(&ctx, device);
  if (err != kSuccess)
    return err;
  return cuCtxSetCurrent(ctx);
}

#endif

} // namespace gsan::drv
