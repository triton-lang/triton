// GSanAllocatorHIP.cc — HIP port of GSanAllocator.cc
//
// Strategy: include the original GSanAllocator.cc wholesale, but first
// #define all CUDA driver API names to their HIP equivalents so the
// compiler sees valid HIP calls.  This avoids forking the 1800-line file.
//
// Differences from CUDA that need special handling here (not 1:1 renames):
//   1. cuGetErrorString(err, &ptr) -> ptr = hipGetErrorString(err)
//   2. CUmemFabricHandle / CU_MEM_HANDLE_TYPE_FABRIC: not in ROCm 7.1 ->
//      stub them out so code that queries/uses fabric always gets POSIX FD.
//   3. CUDA_SUCCESS -> hipSuccess (covered by #define)
//   4. cuMemsetD8 / cuMemsetD8Async: same name in HIP, no rename needed
//      but needs hipDeviceptr_t cast from void*.
//
// Built by _allocator.py (_load_gsan_module_hip); hip_shim/ provides an
// empty cuda.h so the #include in GSanAllocator.cc resolves.

#include <hip/hip_runtime_api.h>

// ── Error code aliases ───────────────────────────────────────────────────────
#define CUDA_ERROR_NO_DEVICE hipErrorNoDevice
#define CUDA_ERROR_NOT_INITIALIZED hipErrorNotInitialized
#define CUDA_ERROR_INVALID_DEVICE hipErrorInvalidDevice
#define CUDA_ERROR_INVALID_VALUE hipErrorInvalidValue
#define CUDA_ERROR_NOT_SUPPORTED hipErrorNotSupported

// ── Type aliases ────────────────────────────────────────────────────────────
#define CUresult hipError_t
#define CUDA_SUCCESS hipSuccess
#define CUdevice hipDevice_t
// CUdeviceptr on NVIDIA is unsigned long long, which allows integer arithmetic.
// hipDeviceptr_t is void*, which does not.  Map CUdeviceptr to uintptr_t so
// all the arithmetic in GSanAllocator.cc compiles unchanged.
#define CUdeviceptr uintptr_t
#define CUmemGenericAllocationHandle hipMemGenericAllocationHandle_t
#define CUmemAllocationProp hipMemAllocationProp
#define CUmemAllocationHandleType hipMemAllocationHandleType
#define CUmemAccessDesc hipMemAccessDesc
#define CUctx_st ihipCtx_t
#define CUctx_t hipCtx_t
#define CUcontext hipCtx_t
#define CUstream hipStream_t
#define cuStream hipStream

// CUmemFabricHandle is not in ROCm 7.1 — stub it as an empty struct.
// Code that tests for FABRIC handle type will never reach it because
// getRequestedShareableHandleType() always returns POSIX_FILE_DESCRIPTOR.
struct CUmemFabricHandle_st {
  char data[64];
};
typedef CUmemFabricHandle_st CUmemFabricHandle;

// ── Enum constant aliases ────────────────────────────────────────────────────
// Default (coarse-grained) VMM memory is cached in L2, so sys-scope polls of a
// peer's counter can spin on a stale value; uncached memory stays coherent.
#define CU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypeUncached
#define CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#define CU_MEM_ALLOC_GRANULARITY_MINIMUM hipMemAllocationGranularityMinimum
#define CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR                               \
  hipMemHandleTypePosixFileDescriptor
// Fabric is not in ROCm 7.1: alias to an unused value so comparisons compile
// but can never match PYTORCH_CUDA_ALLOC_CONF=fabric_handles:True at runtime.
#define CU_MEM_HANDLE_TYPE_FABRIC static_cast<hipMemAllocationHandleType>(0x8)

#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT                               \
  hipDeviceAttributeMultiprocessorCount
// Fabric attribute not in ROCm 7.1: use 0 so the attribute query returns 0
// (unsupported).
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED                       \
  static_cast<hipDeviceAttribute_t>(0)

// ── Function aliases ─────────────────────────────────────────────────────────
#define cuDeviceGet hipDeviceGet
#define cuDeviceGetCount hipGetDeviceCount
#define cuDeviceGetAttribute hipDeviceGetAttribute
#define cuDevicePrimaryCtxRetain hipDevicePrimaryCtxRetain
#define cuDevicePrimaryCtxRelease hipDevicePrimaryCtxRelease
#define cuCtxGetCurrent hipCtxGetCurrent
#define cuCtxSetCurrent hipCtxSetCurrent
#define cuCtxSynchronize hipCtxSynchronize
#define cuStreamSynchronize hipStreamSynchronize
// All HIP VMM functions take void* where CUDA uses CUdeviceptr (uintptr_t
// here). Cast explicitly to keep -fpermissive errors away. ROCm 7.1 quirks
// worked around here:
//  * hipMemAddressReserve ignores `alignment`, but GSan derives shadow and
//    per-device state addresses by masking pointers.
//  * hipMemSetAccess validates `size` against the reservation's sub-buffers
//    starting from the first one, so it fails once a reservation holds more
//    than one mapping.
// So cuMemAddressReserve only picks an aligned free VA range (then releases
// it), and each cuMemMap reserves exactly its own range at that address.
// Another VA allocation could land in the released range in the meantime;
// in that case cuMemMap fails with hipErrorOutOfMemory.
static inline hipError_t cuMemAddressReserve(uintptr_t *ptr, size_t size,
                                             size_t align, uintptr_t addr,
                                             unsigned long long flags) {
  void *base = nullptr;
  hipError_t err = hipMemAddressReserve(&base, size + align, 0,
                                        reinterpret_cast<void *>(addr), flags);
  if (err != hipSuccess)
    return err;
  uintptr_t b = reinterpret_cast<uintptr_t>(base);
  *ptr = align ? (b + align - 1) & ~(uintptr_t)(align - 1) : b;
  return hipMemAddressFree(base, size + align);
}

static inline hipError_t cuMemMap(uintptr_t ptr, size_t size, size_t offset,
                                  hipMemGenericAllocationHandle_t handle,
                                  unsigned long long flags) {
  void *want = reinterpret_cast<void *>(ptr);
  void *got = nullptr;
  hipError_t err = hipMemAddressReserve(&got, size, 0, want, 0);
  if (err != hipSuccess)
    return err;
  if (got != want) {
    (void)hipMemAddressFree(got, size);
    return hipErrorOutOfMemory;
  }
  err = hipMemMap(want, size, offset, handle, flags);
  if (err != hipSuccess)
    (void)hipMemAddressFree(want, size);
  return err;
}

static inline hipError_t cuMemUnmap(uintptr_t ptr, size_t size) {
  void *p = reinterpret_cast<void *>(ptr);
  hipError_t err = hipMemUnmap(p, size);
  hipError_t freeErr = hipMemAddressFree(p, size);
  return err != hipSuccess ? err : freeErr;
}

#define cuMemRelease hipMemRelease
#define cuMemCreate hipMemCreate
#define cuMemSetAccess(ptr, size, desc, n)                                     \
  hipMemSetAccess(reinterpret_cast<void *>(static_cast<uintptr_t>(ptr)), size, \
                  desc, n)
#define cuMemGetAllocationGranularity hipMemGetAllocationGranularity
#define cuMemExportToShareableHandle hipMemExportToShareableHandle
#define cuMemImportFromShareableHandle hipMemImportFromShareableHandle
#define cuMemcpyHtoD(dst, src, n)                                              \
  hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(static_cast<uintptr_t>(dst)), \
                src, n)
#define cuMemsetD8(ptr, val, n)                                                \
  hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(static_cast<uintptr_t>(ptr)),   \
              val, n)
#define cuMemsetD8Async(ptr, val, n, s)                                        \
  hipMemsetD8Async(                                                            \
      reinterpret_cast<hipDeviceptr_t>(static_cast<uintptr_t>(ptr)), val, n,   \
      s)

// ── cuGetErrorString: different signature in HIP ─────────────────────────────
// CUDA: CUresult cuGetErrorString(CUresult err, const char **pStr)
// HIP:  const char* hipGetErrorString(hipError_t err)
// Wrap it so the call site `cuGetErrorString(err, &ptr)` compiles.
static inline hipError_t cuGetErrorString(hipError_t err, const char **pStr) {
  *pStr = hipGetErrorString(err);
  return hipSuccess;
}

// ── AMD-specific constant override ───────────────────────────────────────────
// gfx950 has 256 CUs; with 8 GPUs numThreads = 2048, requiring ~1025 MiB
// per device — exceeding the default kPerDeviceStateStride = 1 GiB in GSan.h.
// GSan.h checks #ifdef GSAN_HIP_LARGE_STRIDE and uses 2 GiB instead.
#define GSAN_HIP_LARGE_STRIDE

// ── Pull in the original implementation ─────────────────────────────────────
// Now include the implementation.  Everything cuda-named will resolve through
// our #defines above.
#include "GSanAllocator.cc"
