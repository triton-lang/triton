#ifndef TRITON_THIRD_PARTY_NVIDIA_BACKEND_CUDA_DECLARATIONS_H
#define TRITON_THIRD_PARTY_NVIDIA_BACKEND_CUDA_DECLARATIONS_H

#include <cstdint>
#include <dlfcn.h>
#include <pybind11/pybind11.h>
#include <string>

// Used to load cuda functions dynamically.
#define INITIALIZE_FUNCTION_POINTER_IF_NULL(funcPointer, initializerFunction)  \
  do {                                                                         \
    if ((funcPointer) == nullptr) {                                            \
      (funcPointer) = (initializerFunction)();                                 \
    }                                                                          \
  } while (0)

#define GET_FUNCTION_HANDLE(name, symbolName)                                  \
  static symbolName##_t name() {                                               \
    /* Open the shared library */                                              \
    void *libHandle = dlopen("libcuda.so.1", RTLD_LAZY);                       \
    /* Check for errors */                                                     \
    if (dlerror()) {                                                           \
      throw pybind11::value_error("Failed to open libcuda.so.1");              \
      return nullptr;                                                          \
    }                                                                          \
    symbolName##_t funcHandle =                                                \
        reinterpret_cast<symbolName##_t>(dlsym(libHandle, #symbolName));       \
    /* Check for errors */                                                     \
    if (dlerror()) {                                                           \
      dlclose(libHandle);                                                      \
      throw pybind11::value_error("Failed to retrieve " #symbolName            \
                                  " from libcuda.so.1");                       \
      return nullptr;                                                          \
    }                                                                          \
    return funcHandle;                                                         \
  }

typedef int CUresult;
CUresult CUDA_SUCCESS = 0;
CUresult CUDA_ERROR_INVALID_VALUE = 1;
typedef CUresult (*cuGetErrorString_t)(CUresult error, const char **errorPtr);
GET_FUNCTION_HANDLE(getGetErrorStringHandle, cuGetErrorString);

CUresult getErrorString(CUresult error, const char **errorPtr) {
  static cuGetErrorString_t cuGetErrorString = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuGetErrorString,
                                      getGetErrorStringHandle);
  return cuGetErrorString(error, errorPtr);
}

// Raise a python exception if the CUDA result code is not CUDA_SUCCESS.
inline void gpuAssert(CUresult code, const char *file, int line) {
  if (code == CUDA_SUCCESS)
    return;
  const char *errorPtr = nullptr;
  getErrorString(code, &errorPtr);
  std::string errorStr = errorPtr ? errorPtr : "Unknown CUDA error";
  std::string errorMessage = "Triton Error [CUDA]: " + errorStr;
  throw pybind11::cast_error(errorMessage);
}

#define CUDA_CHECK(ans) gpuAssert((ans), __FILE__, __LINE__);

/*
cuTensorMapEncodeTiled
*/
// Forward-declare all things we are using before linking.
#define CU_TENSOR_MAP_SIZE 16
typedef uint64_t cuuint64_t;
typedef uint32_t cuuint32_t;
typedef struct CUtensorMap_st {
  alignas(64) cuuint64_t opaque[CU_TENSOR_MAP_SIZE];
} CUtensorMap;
typedef enum CUtensorMapDataType_enum {
  CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,
  CU_TENSOR_MAP_DATA_TYPE_UINT16,
  CU_TENSOR_MAP_DATA_TYPE_UINT32,
  CU_TENSOR_MAP_DATA_TYPE_INT32,
  CU_TENSOR_MAP_DATA_TYPE_UINT64,
  CU_TENSOR_MAP_DATA_TYPE_INT64,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ,
  CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
  CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ,
  CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
  CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
  CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B
} CUtensorMapDataType;
typedef enum CUtensorMapInterleave_enum {
  CU_TENSOR_MAP_INTERLEAVE_NONE = 0,
  CU_TENSOR_MAP_INTERLEAVE_16B,
  CU_TENSOR_MAP_INTERLEAVE_32B
} CUtensorMapInterleave;
typedef enum CUtensorMapSwizzle_enum {
  CU_TENSOR_MAP_SWIZZLE_NONE = 0,
  CU_TENSOR_MAP_SWIZZLE_32B,
  CU_TENSOR_MAP_SWIZZLE_64B,
  CU_TENSOR_MAP_SWIZZLE_128B,
  CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,
  CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B,
  CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B
} CUtensorMapSwizzle;
typedef enum CUtensorMapL2promotion_enum {
  CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
  CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
  CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
  CU_TENSOR_MAP_L2_PROMOTION_L2_256B
} CUtensorMapL2promotion;
typedef enum CUtensorMapFloatOOBfill_enum {
  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0,
  CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
} CUtensorMapFloatOOBfill;

typedef CUresult (*cuTensorMapEncodeTiled_t)(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const cuuint32_t *boxDim,
    const cuuint32_t *elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill);
GET_FUNCTION_HANDLE(getCuTensorMapEncodeTiledHandle, cuTensorMapEncodeTiled);

CUresult tensorMapEncodeTiled(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const cuuint32_t *boxDim,
    const cuuint32_t *elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
  static cuTensorMapEncodeTiled_t cuTensorMapEncodeTiled = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuTensorMapEncodeTiled,
                                      getCuTensorMapEncodeTiledHandle);
  return cuTensorMapEncodeTiled(tensorMap, tensorDataType, tensorRank,
                                globalAddress, globalDim, globalStrides, boxDim,
                                elementStrides, interleave, swizzle,
                                l2Promotion, oobFill);
}

/*
cuLaunchKernelEx
*/
struct CUevent_st;
typedef struct CUevent_st *CUevent;
struct CUgraphDeviceUpdatableNode_st;
typedef struct CUgraphDeviceUpdatableNode_st *CUgraphDeviceNode;
struct CUgraphDeviceUpdatableNode_st;
typedef struct CUgraphDeviceUpdatableNode_st *CUgraphDeviceNode;
typedef enum CUlaunchAttributeID_enum {
  CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
  CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
  CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
  CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
} CUlaunchAttributeID;
typedef enum CUclusterSchedulingPolicy_enum {
  CU_CLUSTER_SCHEDULING_POLICY_DEFAULT = 0,
  CU_CLUSTER_SCHEDULING_POLICY_SPREAD = 1,
  CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING = 2
} CUclusterSchedulingPolicy;
typedef enum CUaccessProperty_enum {
  CU_ACCESS_PROPERTY_NORMAL = 0,
  CU_ACCESS_PROPERTY_STREAMING = 1,
  CU_ACCESS_PROPERTY_PERSISTING = 2
} CUaccessProperty;
typedef struct CUlaunchMemSyncDomainMap_st {
  unsigned char default_;
  unsigned char remote;
} CUlaunchMemSyncDomainMap;
typedef enum CUlaunchMemSyncDomain_enum {
  CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT = 0,
  CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE = 1
} CUlaunchMemSyncDomain;
typedef struct CUaccessPolicyWindow_st {
  void *base_ptr;
  size_t num_bytes;
  float hitRatio;
  CUaccessProperty hitProp;
  CUaccessProperty missProp;
} CUaccessPolicyWindow;
typedef enum CUsynchronizationPolicy_enum {
  CU_SYNC_POLICY_AUTO = 1,
  CU_SYNC_POLICY_SPIN = 2,
  CU_SYNC_POLICY_YIELD = 3,
  CU_SYNC_POLICY_BLOCKING_SYNC = 4
} CUsynchronizationPolicy;
typedef union CUlaunchAttributeValue_union {
  char pad[64];
  CUaccessPolicyWindow accessPolicyWindow;
  int cooperative;
  CUsynchronizationPolicy syncPolicy;
  struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
  } clusterDim;
  CUclusterSchedulingPolicy clusterSchedulingPolicyPreference;
  int programmaticStreamSerializationAllowed;
  struct {
    CUevent event;
    int flags;
    int triggerAtBlockStart;
  } programmaticEvent;
  struct {
    CUevent event;
    int flags;
  } launchCompletionEvent;
  int priority;
  CUlaunchMemSyncDomainMap memSyncDomainMap;
  CUlaunchMemSyncDomain memSyncDomain;
  struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
  } preferredClusterDim;
  struct {
    int deviceUpdatable;
    CUgraphDeviceNode devNode;
  } deviceUpdatableKernelNode;
  unsigned int sharedMemCarveout;
} CUlaunchAttributeValue;

struct CUfunc_st;
typedef CUfunc_st *CUfunction;
typedef struct CUlaunchAttribute_st {
  CUlaunchAttributeID id;
  CUlaunchAttributeValue value;
} CUlaunchAttribute;
typedef void *CUstream;
typedef struct CUlaunchConfig_st {
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;
  CUstream hStream;
  CUlaunchAttribute *attrs;
  unsigned int numAttrs;
} CUlaunchConfig;

typedef CUresult (*cuLaunchKernelEx_t)(const CUlaunchConfig *config,
                                       CUfunction f, void **kernelParams,
                                       void **extra);
GET_FUNCTION_HANDLE(getLaunchKernelExHandle, cuLaunchKernelEx);

CUresult launchKernelEx(const CUlaunchConfig *config, CUfunction f,
                        void **kernelParams, void **extra) {
  static cuLaunchKernelEx_t cuLaunchKernelEx = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuLaunchKernelEx,
                                      getLaunchKernelExHandle);
  return cuLaunchKernelEx(config, f, kernelParams, extra);
}

/*
cuPointerGetAttribute
*/
typedef unsigned long long CUdeviceptr;
typedef int CUpointer_attribute;
CUpointer_attribute CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3;
typedef CUresult (*cuPointerGetAttribute_t)(void *data,
                                            CUpointer_attribute attribute,
                                            CUdeviceptr ptr);
GET_FUNCTION_HANDLE(getPointerGetAttributeHandle, cuPointerGetAttribute);

CUresult pointerGetAttribute(void *data, CUpointer_attribute attribute,
                             CUdeviceptr ptr) {
  static cuPointerGetAttribute_t cuPointerGetAttribute = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuPointerGetAttribute,
                                      getPointerGetAttributeHandle);
  return cuPointerGetAttribute(data, attribute, ptr);
}

/*
cuOccupancyMaxActiveClusters
*/
typedef CUresult (*cuOccupancyMaxActiveClusters_t)(
    int *numClusters, CUfunction func, const CUlaunchConfig *config);
GET_FUNCTION_HANDLE(getCuOccupancyMaxActiveClustersHandle,
                    cuOccupancyMaxActiveClusters);

CUresult occupancyMaxActiveClusters(int *numClusters, CUfunction func,
                                    const CUlaunchConfig *config) {
  static cuOccupancyMaxActiveClusters_t cuOccupancyMaxActiveClusters = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuOccupancyMaxActiveClusters,
                                      getCuOccupancyMaxActiveClustersHandle);
  return cuOccupancyMaxActiveClusters(numClusters, func, config);
}

/*
cuFuncSetAttribute
*/
typedef int CUfunction_attribute;
CUfunction_attribute CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0;
CUfunction_attribute CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1;
CUfunction_attribute CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3;
CUfunction_attribute CU_FUNC_ATTRIBUTE_NUM_REGS = 4;
CUfunction_attribute CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8;
CUfunction_attribute CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14;

typedef CUresult (*cuFuncSetAttribute_t)(CUfunction hfunc,
                                         CUfunction_attribute attrib,
                                         int value);
GET_FUNCTION_HANDLE(getFuncSetAttributeHandle, cuFuncSetAttribute);
CUresult funcSetAttribute(CUfunction hfunc, CUfunction_attribute attrib,
                          int value) {
  static cuFuncSetAttribute_t cuFuncSetAttribute = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuFuncSetAttribute,
                                      getFuncSetAttributeHandle);
  return cuFuncSetAttribute(hfunc, attrib, value);
}

/*
cuFuncGetAttribute
*/
typedef CUresult (*cuFuncGetAttribute_t)(int *pi, CUfunction_attribute attrib,
                                         CUfunction hfunc);
GET_FUNCTION_HANDLE(getFuncGetAttributeHandle, cuFuncGetAttribute);

CUresult funcGetAttribute(int *pi, CUfunction_attribute attrib,
                          CUfunction hfunc) {
  static cuFuncGetAttribute_t cuFuncGetAttribute = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuFuncGetAttribute,
                                      getFuncGetAttributeHandle);
  return cuFuncGetAttribute(pi, attrib, hfunc);
}

/*
cuDeviceGet
*/
typedef int CUdevice;
typedef CUresult (*cuDeviceGet_t)(CUdevice *device, int ordinal);
GET_FUNCTION_HANDLE(getDeviceGetHandle, cuDeviceGet);

CUresult deviceGet(CUdevice *device, int ordinal) {
  static cuDeviceGet_t cuDeviceGet = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuDeviceGet, getDeviceGetHandle);
  return cuDeviceGet(device, ordinal);
}

/*
cuCtxGetCurrent
*/
struct CUctx_st;
typedef CUctx_st *CUcontext;
typedef CUresult (*cuCtxGetCurrent_t)(CUcontext *pctx);
GET_FUNCTION_HANDLE(getCtxGetCurrentHandle, cuCtxGetCurrent);

CUresult ctxGetCurrent(CUcontext *pctx) {
  static cuCtxGetCurrent_t cuCtxGetCurrent = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuCtxGetCurrent, getCtxGetCurrentHandle);
  return cuCtxGetCurrent(pctx);
}

/*
cuDevicePrimaryCtxRetain
*/
typedef CUresult (*cuDevicePrimaryCtxRetain_t)(CUcontext *pctx, CUdevice dev);
GET_FUNCTION_HANDLE(getDevicePrimaryCtxRetainHandle, cuDevicePrimaryCtxRetain);

CUresult devicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
  static cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuDevicePrimaryCtxRetain,
                                      getDevicePrimaryCtxRetainHandle);
  return cuDevicePrimaryCtxRetain(pctx, dev);
}

/*
cuCtxSetCurrent
*/
typedef CUresult (*cuCtxSetCurrent_t)(CUcontext ctx);
GET_FUNCTION_HANDLE(getCtxSetCurrentHandle, cuCtxSetCurrent);

CUresult ctxSetCurrent(CUcontext ctx) {
  static cuCtxSetCurrent_t cuCtxSetCurrent = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuCtxSetCurrent, getCtxSetCurrentHandle);
  return cuCtxSetCurrent(ctx);
}

/*
cuDeviceGetAttribute
*/
typedef int CUdevice_attribute;
CUdevice_attribute CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10;
CUdevice_attribute CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12;
CUdevice_attribute CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13;
CUdevice_attribute CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
CUdevice_attribute CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36;
CUdevice_attribute CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37;
CUdevice_attribute CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97;
typedef CUresult (*cuDeviceGetAttribute_t)(int *pi, CUdevice_attribute attrib,
                                           CUdevice dev);
GET_FUNCTION_HANDLE(getDeviceGetAttributeHandle, cuDeviceGetAttribute);

CUresult deviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
  static cuDeviceGetAttribute_t cuDeviceGetAttribute = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuDeviceGetAttribute,
                                      getDeviceGetAttributeHandle);
  return cuDeviceGetAttribute(pi, attrib, dev);
}

/*
cuModuleLoadData
*/
struct CUmod_st;
typedef CUmod_st *CUmodule;
typedef CUresult (*cuModuleLoadData_t)(CUmodule *module, const void *image);
GET_FUNCTION_HANDLE(getModuleLoadDataHandle, cuModuleLoadData);

CUresult moduleLoadData(CUmodule *module, const void *image) {
  static cuModuleLoadData_t cuModuleLoadData = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuModuleLoadData,
                                      getModuleLoadDataHandle);
  return cuModuleLoadData(module, image);
}

/*
cuModuleGetFunction
*/
typedef CUresult (*cuModuleGetFunction_t)(CUfunction *hfunc, CUmodule hmod,
                                          const char *name);
GET_FUNCTION_HANDLE(getModuleGetFunctionHandle, cuModuleGetFunction);

CUresult moduleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
  static cuModuleGetFunction_t cuModuleGetFunction = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuModuleGetFunction,
                                      getModuleGetFunctionHandle);
  return cuModuleGetFunction(hfunc, hmod, name);
}

/*
cuFuncSetCacheConfig
*/
typedef int CUfunc_cache;
CUfunc_cache CU_FUNC_CACHE_PREFER_SHARED = 0x01;
typedef CUresult (*cuFuncSetCacheConfig_t)(CUfunction hfunc,
                                           CUfunc_cache config);
GET_FUNCTION_HANDLE(getFuncSetCacheConfigHandle, cuFuncSetCacheConfig);

CUresult funcSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
  static cuFuncSetCacheConfig_t cuFuncSetCacheConfig = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuFuncSetCacheConfig,
                                      getFuncSetCacheConfigHandle);
  return cuFuncSetCacheConfig(hfunc, config);
}

/*
cuCtxGetLimit
*/
typedef int CUlimit;
CUlimit CU_LIMIT_PRINTF_FIFO_SIZE = 0x01;
typedef CUresult (*cuCtxGetLimit_t)(size_t *pvalue, CUlimit limit);
GET_FUNCTION_HANDLE(getCtxGetLimitHandle, cuCtxGetLimit);

CUresult ctxGetLimit(size_t *pvalue, CUlimit limit) {
  static cuCtxGetLimit_t cuCtxGetLimit = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuCtxGetLimit, getCtxGetLimitHandle);
  return cuCtxGetLimit(pvalue, limit);
}

/*
cuCtxSetLimit
*/
typedef CUresult (*cuCtxSetLimit_t)(CUlimit limit, size_t value);
GET_FUNCTION_HANDLE(getCtxSetLimitHandle, cuCtxSetLimit);

CUresult ctxSetLimit(CUlimit limit, size_t value) {
  static cuCtxSetLimit_t cuCtxSetLimit = nullptr;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuCtxSetLimit, getCtxSetLimitHandle);
  return cuCtxSetLimit(limit, value);
}

#endif // TRITON_THIRD_PARTY_NVIDIA_BACKEND_CUDA_DECLARATIONS_H
