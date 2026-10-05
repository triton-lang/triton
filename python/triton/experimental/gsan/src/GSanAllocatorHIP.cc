// GSanAllocatorHIP.cc — HIP port of GSanAllocator.cc
//
// Strategy: include the original GSanAllocator.cc wholesale, but first
// #define all CUDA driver API names to their HIP equivalents so the
// compiler sees valid HIP calls.  This avoids forking the 1800-line file.
// HIP-specific behavior that is not a rename lives in GSanAllocator.cc under
// #ifdef GSAN_HIP.
//
// Built by _allocator.py (_load_gsan_module_hip), which passes GSAN_HIP,
// GSAN_LIBHIP_PATH and GSAN_HIP_LARGE_STRIDE; hip_shim/ provides an empty
// cuda.h so the #include in GSanAllocator.cc resolves.

#include <dlfcn.h>
#include <hip/hip_runtime_api.h>

#include <cstdio>

#ifndef GSAN_LIBHIP_PATH
#error "GSAN_LIBHIP_PATH must point to libamdhip64.so"
#endif

// ── Runtime-loaded HIP entry points ─────────────────────────────────────────
// Like Triton's AMD driver, load the HIP runtime the process already uses
// rather than linking against one at build time.
static void *gsanLoadHipSymbol(const char *name) {
  static void *lib = [] {
    void *handle = dlopen(GSAN_LIBHIP_PATH, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr)
      fprintf(stderr, "GSan: failed to load %s: %s\n", GSAN_LIBHIP_PATH,
              dlerror());
    return handle;
  }();
  if (lib == nullptr)
    return nullptr;
  void *sym = dlsym(lib, name);
  if (sym == nullptr)
    fprintf(stderr, "GSan: %s not found in %s\n", name, GSAN_LIBHIP_PATH);
  return sym;
}

#define GSAN_HIP_FN(name)                                                      \
  template <typename... Args> static inline hipError_t gsan_##name(Args... args) { \
    using Fn = decltype(&::name);                                              \
    static Fn fn = reinterpret_cast<Fn>(gsanLoadHipSymbol(#name));             \
    return fn ? fn(args...) : hipErrorSharedObjectInitFailed;                  \
  }

GSAN_HIP_FN(hipDeviceGet)
GSAN_HIP_FN(hipGetDeviceCount)
GSAN_HIP_FN(hipDeviceGetAttribute)
GSAN_HIP_FN(hipDevicePrimaryCtxRetain)
GSAN_HIP_FN(hipDevicePrimaryCtxRelease)
GSAN_HIP_FN(hipCtxGetCurrent)
GSAN_HIP_FN(hipCtxSetCurrent)
GSAN_HIP_FN(hipCtxSynchronize)
GSAN_HIP_FN(hipStreamSynchronize)
GSAN_HIP_FN(hipMemAddressReserve)
GSAN_HIP_FN(hipMemAddressFree)
GSAN_HIP_FN(hipMemMap)
GSAN_HIP_FN(hipMemUnmap)
GSAN_HIP_FN(hipMemRelease)
GSAN_HIP_FN(hipMemCreate)
GSAN_HIP_FN(hipMemSetAccess)
GSAN_HIP_FN(hipMemGetAllocationGranularity)
GSAN_HIP_FN(hipMemExportToShareableHandle)
GSAN_HIP_FN(hipMemImportFromShareableHandle)
GSAN_HIP_FN(hipMemcpyHtoD)
GSAN_HIP_FN(hipMemsetD8)
GSAN_HIP_FN(hipMemsetD8Async)

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

// ROCm 7.1 has no fabric handles. These stand-ins only let the shared code
// compile; GSanAllocator.cc rejects fabric requests on HIP before they reach
// the HIP runtime.
struct CUmemFabricHandle_st {
  char data[64];
};
typedef CUmemFabricHandle_st CUmemFabricHandle;
#define CU_MEM_HANDLE_TYPE_FABRIC static_cast<hipMemAllocationHandleType>(0x8)

// ── Enum constant aliases ────────────────────────────────────────────────────
#define CU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned
#define CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#define CU_MEM_ALLOC_GRANULARITY_MINIMUM hipMemAllocationGranularityMinimum
#define CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR                               \
  hipMemHandleTypePosixFileDescriptor
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT                               \
  hipDeviceAttributeMultiprocessorCount

// ── Function aliases ─────────────────────────────────────────────────────────
#define cuDeviceGet gsan_hipDeviceGet
#define cuDeviceGetCount gsan_hipGetDeviceCount
#define cuDeviceGetAttribute gsan_hipDeviceGetAttribute
#define cuDevicePrimaryCtxRetain gsan_hipDevicePrimaryCtxRetain
#define cuDevicePrimaryCtxRelease gsan_hipDevicePrimaryCtxRelease
#define cuCtxGetCurrent gsan_hipCtxGetCurrent
#define cuCtxSetCurrent gsan_hipCtxSetCurrent
#define cuCtxSynchronize gsan_hipCtxSynchronize
#define cuStreamSynchronize gsan_hipStreamSynchronize
// All HIP VMM functions take void* where CUDA uses CUdeviceptr (uintptr_t
// here). Cast explicitly to keep -fpermissive errors away. ROCm 7.1 quirks
// worked around here:
//  * hipMemAddressReserve ignores `alignment`, but GSan derives shadow and
//    per-device state addresses by masking pointers.
//  * hipMemSetAccess validates `size` against the reservation's sub-buffers
//    starting from the first one, so it fails once a reservation holds a
//    mapping larger than its first one.
// So cuMemAddressReserve only picks an aligned free VA range (then releases
// it), and each cuMemMap reserves exactly its own range at that address.
// Another VA allocation could land in the released range in the meantime;
// in that case cuMemMap fails with hipErrorOutOfMemory.
static inline hipError_t cuMemAddressReserve(uintptr_t *ptr, size_t size,
                                             size_t align, uintptr_t addr,
                                             unsigned long long flags) {
  void *base = nullptr;
  hipError_t err = gsan_hipMemAddressReserve(
      &base, size + align, 0, reinterpret_cast<void *>(addr), flags);
  if (err != hipSuccess)
    return err;
  uintptr_t b = reinterpret_cast<uintptr_t>(base);
  *ptr = align ? (b + align - 1) & ~(uintptr_t)(align - 1) : b;
  return gsan_hipMemAddressFree(base, size + align);
}

static inline hipError_t cuMemMap(uintptr_t ptr, size_t size, size_t offset,
                                  hipMemGenericAllocationHandle_t handle,
                                  unsigned long long flags) {
  void *want = reinterpret_cast<void *>(ptr);
  void *got = nullptr;
  hipError_t err = gsan_hipMemAddressReserve(&got, size, 0, want, 0);
  if (err != hipSuccess)
    return err;
  if (got != want) {
    (void)gsan_hipMemAddressFree(got, size);
    return hipErrorOutOfMemory;
  }
  err = gsan_hipMemMap(want, size, offset, handle, flags);
  if (err != hipSuccess)
    (void)gsan_hipMemAddressFree(want, size);
  return err;
}

static inline hipError_t cuMemUnmap(uintptr_t ptr, size_t size) {
  void *p = reinterpret_cast<void *>(ptr);
  hipError_t err = gsan_hipMemUnmap(p, size);
  hipError_t freeErr = gsan_hipMemAddressFree(p, size);
  return err != hipSuccess ? err : freeErr;
}

#define cuMemRelease gsan_hipMemRelease
#define cuMemCreate gsan_hipMemCreate
#define cuMemSetAccess(ptr, size, desc, n)                                     \
  gsan_hipMemSetAccess(reinterpret_cast<void *>(static_cast<uintptr_t>(ptr)), \
                       size, desc, n)
#define cuMemGetAllocationGranularity gsan_hipMemGetAllocationGranularity
#define cuMemExportToShareableHandle gsan_hipMemExportToShareableHandle
#define cuMemImportFromShareableHandle gsan_hipMemImportFromShareableHandle
#define cuMemcpyHtoD(dst, src, n)                                              \
  gsan_hipMemcpyHtoD(                                                          \
      reinterpret_cast<hipDeviceptr_t>(static_cast<uintptr_t>(dst)), src, n)
#define cuMemsetD8(ptr, val, n)                                                \
  gsan_hipMemsetD8(                                                            \
      reinterpret_cast<hipDeviceptr_t>(static_cast<uintptr_t>(ptr)), val, n)
#define cuMemsetD8Async(ptr, val, n, s)                                        \
  gsan_hipMemsetD8Async(                                                       \
      reinterpret_cast<hipDeviceptr_t>(static_cast<uintptr_t>(ptr)), val, n,   \
      s)

// ── cuGetErrorString: different signature in HIP ─────────────────────────────
// CUDA: CUresult cuGetErrorString(CUresult err, const char **pStr)
// HIP:  const char* hipGetErrorString(hipError_t err)
// Wrap it so the call site `cuGetErrorString(err, &ptr)` compiles.
static inline hipError_t cuGetErrorString(hipError_t err, const char **pStr) {
  using Fn = decltype(&::hipGetErrorString);
  static Fn fn = reinterpret_cast<Fn>(gsanLoadHipSymbol("hipGetErrorString"));
  *pStr = fn ? fn(err) : "HIP runtime not loaded";
  return hipSuccess;
}

// ── Pull in the original implementation ─────────────────────────────────────
// Now include the implementation.  Everything cuda-named will resolve through
// our #defines above.
#include "GSanAllocator.cc"
