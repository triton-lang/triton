#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// Include shared TDM utilities
#include "TDMCommon.h"

typedef struct {
  uint32_t group0_0;
  uint32_t group0_1;
  uint32_t group0_2;
  uint32_t group0_3;
  uint32_t group1_0;
  uint32_t group1_1;
  uint32_t group1_2;
  uint32_t group1_3;
  uint32_t group1_4;
  uint32_t group1_5;
  uint32_t group1_6;
  uint32_t group1_7;
  uint32_t group2_0;
  uint32_t group2_1;
  uint32_t group2_2;
  uint32_t group2_3;
  uint32_t group3_0;
  uint32_t group3_1;
  uint32_t group3_2;
  uint32_t group3_3;
} TDMDescriptor;

typedef struct {
  PyObject_HEAD;
  TDMDescriptor desc;
} PyTDMDescriptorObject;

static PyObject *PyTDMDescriptor_new(PyTypeObject *type, PyObject *args,
                                     PyObject *kw) {
  PyTDMDescriptorObject *self =
      (PyTDMDescriptorObject *)type->tp_alloc(type, 0);
  if (!self)
    return NULL;

  memset(&self->desc, 0, sizeof(self->desc));
  return (PyObject *)self;
}

static void PyTDMDescriptor_dealloc(PyTDMDescriptorObject *self) {
  Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyTypeObject PyTDMDescriptorType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name =
        "triton.backends.amd.PyTDMDescriptor",
    .tp_basicsize = sizeof(PyTDMDescriptorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "PyObject for TDMDescriptor",
    .tp_new = PyTDMDescriptor_new,
    .tp_dealloc = (destructor)PyTDMDescriptor_dealloc,
};

typedef enum { ARG_CONSTEXPR = 0, ARG_KERNEL = 1, ARG_TUPLE = 2 } ArgType;

// Annotation struct to know how the argument should be handled.
typedef struct {
  PyObject_HEAD;
  PyObject *nested_tuple; // Can be a List of PyKernelArgObjects or None
  ArgType type;
} PyKernelArgObject;

// Deallocator
static void PyKernelArg_dealloc(PyKernelArgObject *self) {
  Py_XDECREF(self->nested_tuple);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

// Constructor
static int PyKernelArg_init(PyKernelArgObject *self, PyObject *args,
                            PyObject *kwds) {
  static char *kwlist[] = {"nested_tuple", "type", NULL};
  PyObject *tup = NULL;
  int type_val = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", kwlist, &tup,
                                   &type_val)) {
    return -1;
  }
  Py_XINCREF(tup);
  self->nested_tuple = tup;
  self->type = (ArgType)type_val;
  return 0;
}

static void PyKernelArg_free(void *ptr) { free(ptr); }

static PyTypeObject PyKernelArgType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name =
        "triton.backends.nvidia.PyKernelArg",
    .tp_basicsize = sizeof(PyKernelArgObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Kernel Argument Metadata",
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)PyKernelArg_init,
    .tp_dealloc = (destructor)PyKernelArg_dealloc,
};

// Encodes a TDM descriptor. Supports 1D-5D tensors.
// Uses the same encoding format as createTDMDescriptor in TDMUtility.cpp.
static bool encodeTDMDescriptor(TDMDescriptor *desc, int elementBitWidth,
                                uint32_t *blockSize, int numWarps,
                                int padInterval, int padAmount, uint32_t *shape,
                                uint32_t *strides, uint64_t globalAddress,
                                int rank) {
  if (rank < 1 || rank > 5)
    return false;

  memset(desc, 0, sizeof(TDMDescriptor));

  // Convert to int64_t for shared function and get adjusted block sizes
  int64_t blockShape64[5], adjustedBlockSize64[5];
  for (int i = 0; i < rank; ++i)
    blockShape64[i] = blockSize[i];
  tdmGetAdjustedBlockShape(blockShape64, rank, numWarps, adjustedBlockSize64);

  // Convert back to uint32_t
  uint32_t adjustedBlockSize[5];
  for (int i = 0; i < rank; ++i)
    adjustedBlockSize[i] = (uint32_t)adjustedBlockSize64[i];

  // group0 (128 bits / 4 dwords) effective bit encoding:
  // [1:0]:     pred (to be filled later)
  // [63:32]:   lds address (to be filled later)
  // [120:64]:  global address
  // [127:126]: type - currently always set to 0x2
  desc->group0_2 = (uint32_t)(globalAddress & 0xFFFFFFFF);
  desc->group0_3 = (uint32_t)((globalAddress >> 32) & 0x7FFFFFFF) | (0x1 << 31);

  // group1 (256 bits / 8 dwords) effective bit encoding:
  // [15:0]:    multicast mask
  // [17:16]:   data size - log2(element size in bytes)
  // [20]:      enable padding
  // [24:22]:   pad interval - log2(pad interval in dwords) - 1
  // [31:25]:   pad amount - pad amount in dwords - 1
  // [79:48]:   tensor shape dim inner
  // [111:80]:  tensor shape dim outer
  // [127:112]: block shape dim inner
  // [143:128]: block shape dim outer
  // [159:144]: tile_dim2
  // [207:160]: tensor stride dim outer (we only use 32 bits)
  // [255:208]: tensor stride dim 2 (48 bits)
  int elementSizeInBytes = elementBitWidth / 8;
  int dataSize = (int)log2(elementSizeInBytes);
  int dwordSize = 32;
  int padIntervalInDwords = padInterval * elementBitWidth / dwordSize;
  int padAmountInDwords = padAmount * elementBitWidth / dwordSize;

  desc->group1_0 = (dataSize << 16);
  if (padIntervalInDwords > 0 && padAmountInDwords > 0) {
    int log2PadInterval = (int)log2(padIntervalInDwords);
    desc->group1_0 |= (1 << 20);
    desc->group1_0 |= ((log2PadInterval - 1) << 22);
    desc->group1_0 |= ((padAmountInDwords - 1) << 25);
  }

  // Encode tensor shapes (48-bit encoding, indices from end: rank-1 is inner)
  desc->group1_1 = (shape[rank - 1] << 16);
  desc->group1_2 = (shape[rank - 1] >> 16);

  if (rank >= 2) {
    desc->group1_2 |= (shape[rank - 2] << 16);
    desc->group1_3 = (shape[rank - 2] >> 16);
  }

  // Block shapes
  desc->group1_3 |= (adjustedBlockSize[rank - 1] << 16);
  if (rank >= 2)
    desc->group1_4 = (adjustedBlockSize[rank - 2] & 0xFFFF);
  if (rank >= 3)
    desc->group1_4 |= (adjustedBlockSize[rank - 3] << 16);

  // Strides
  if (rank >= 2)
    desc->group1_5 = strides[rank - 2];
  if (rank >= 3) {
    desc->group1_6 = (strides[rank - 3] << 16);
    desc->group1_7 = (strides[rank - 3] >> 16);
  }

  // group2 (128 bits / 4 dwords) for 3D-5D tensors:
  // [31:0]:    tensor_dim2 (3rd dimension from end)
  // [63:32]:   tensor_dim3 (4th dimension from end)
  // [111:64]:  tensor_dim2_stride (48 bits, we use 32 bits)
  // [127:112]: tile_dim3
  if (rank >= 3) {
    desc->group2_0 = shape[rank - 3];
    if (rank >= 4) {
      desc->group2_1 = shape[rank - 4];
      desc->group2_2 = strides[rank - 4];
      desc->group2_3 = (adjustedBlockSize[rank - 4] << 16);
    }
  }

  // group3 (128 bits / 4 dwords) for 4D-5D tensors:
  // [47:0]:    tensor_dim3_stride (48 bits, we use 32 bits)
  // [79:48]:   tensor_dim4 (5th dimension from end)
  // [95:80]:   tile_dim4
  // [127:96]:  reserved
  if (rank == 5) {
    desc->group3_0 = strides[rank - 5];
    desc->group3_1 = (shape[rank - 5] << 16);
    desc->group3_2 = (shape[rank - 5] >> 16);
    desc->group3_2 |= (adjustedBlockSize[rank - 5] << 16);
  }

  return true;
}

// The list of paths to search for the HIP runtime library. The caller Python
// code should substitute the search path placeholder.
static const char *hipLibSearchPaths[] = {"/*py_libhip_search_path*/"};

// The list of HIP dynamic library symbols and their signature we are interested
// in this file.
// |FOR_EACH_ERR_FN| is a macro to process APIs that return hipError_t;
// |FOR_EACH_STR_FN| is a macro to process APIs that return const char *.
#define HIP_SYMBOL_LIST(FOR_EACH_ERR_FN, FOR_EACH_STR_FN)                      \
  FOR_EACH_STR_FN(hipGetLastError)                                             \
  FOR_EACH_STR_FN(hipGetErrorString, hipError_t hipError)                      \
  FOR_EACH_ERR_FN(hipGetDeviceProperties, hipDeviceProp_t *prop, int deviceId) \
  FOR_EACH_ERR_FN(hipModuleLoadDataEx, hipModule_t *module, const void *image, \
                  unsigned int numOptions, hipJitOption *options,              \
                  void **optionValues)                                         \
  FOR_EACH_ERR_FN(hipModuleGetFunction, hipFunction_t *function,               \
                  hipModule_t module, const char *kname)                       \
  FOR_EACH_ERR_FN(hipFuncGetAttribute, int *, hipFunction_attribute attr,      \
                  hipFunction_t function)                                      \
  FOR_EACH_ERR_FN(hipDrvLaunchKernelEx, const HIP_LAUNCH_CONFIG *config,       \
                  hipFunction_t f, void **kernelParams, void **extra)          \
  FOR_EACH_ERR_FN(hipModuleLaunchKernel, hipFunction_t f,                      \
                  unsigned int gridDimX, unsigned int gridDimY,                \
                  unsigned int gridDimZ, unsigned int blockDimX,               \
                  unsigned int blockDimY, unsigned int blockDimZ,              \
                  unsigned int sharedMemBytes, hipStream_t stream,             \
                  void **kernelParams, void **extra)                           \
  FOR_EACH_ERR_FN(hipModuleLaunchCooperativeKernel, hipFunction_t f,           \
                  unsigned int gridDimX, unsigned int gridDimY,                \
                  unsigned int gridDimZ, unsigned int blockDimX,               \
                  unsigned int blockDimY, unsigned int blockDimZ,              \
                  unsigned int sharedMemBytes, hipStream_t stream,             \
                  void **kernelParams, void **extra)                           \
  FOR_EACH_ERR_FN(hipPointerGetAttribute, void *data,                          \
                  hipPointer_attribute attribute, hipDeviceptr_t ptr)

// HIP driver version format: HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR *
// 100000 + HIP_VERSION_PATCH.
#define TRITON_HIP_DRIVER_EXTRACT_MAJOR_VERSION(version) ((version) / 10000000)
#define TRITON_HIP_DRIVER_EXTRACT_MINOR_VERSION(version)                       \
  (((version) % 10000000) / 100000)
#define TRITON_HIP_DRIVER_EXTRACT_PATCH_VERSION(version) ((version) % 100000)
#define TRITON_HIP_DRIVER_REQ_MAJOR_VERSION (6)

// #define TRITON_HIP_DRIVER_DBG_VERSION
#ifdef TRITON_HIP_DRIVER_DBG_VERSION
#define TRITON_HIP_DRIVER_LOG_VERSION(version, msgBuff)                        \
  do {                                                                         \
    snprintf(msgBuff, sizeof(msgBuff), "libamdhip64 version is: %d.%d.%d",     \
             TRITON_HIP_DRIVER_EXTRACT_MAJOR_VERSION(version),                 \
             TRITON_HIP_DRIVER_EXTRACT_MINOR_VERSION(version),                 \
             TRITON_HIP_DRIVER_EXTRACT_PATCH_VERSION(version));                \
    printf("%s\n", msgBuff);                                                   \
  } while (0);
#else
#define TRITON_HIP_DRIVER_LOG_VERSION(version, msgBuff)                        \
  do {                                                                         \
    (void)msgBuff;                                                             \
    (void)(version);                                                           \
  } while (0);
#endif

#define TRITON_HIP_MSG_BUFF_SIZE (1024U)

// The HIP symbol table for holding resolved dynamic library symbols.
struct HIPSymbolTable {
#define DEFINE_EACH_ERR_FIELD(hipSymbolName, ...)                              \
  hipError_t (*hipSymbolName)(__VA_ARGS__);
#define DEFINE_EACH_STR_FIELD(hipSymbolName, ...)                              \
  const char *(*hipSymbolName)(__VA_ARGS__);

  HIP_SYMBOL_LIST(DEFINE_EACH_ERR_FIELD, DEFINE_EACH_STR_FIELD)
};

static struct HIPSymbolTable hipSymbolTable;

static int checkDriverVersion(void *lib) {
  int hipVersion = -1;
  const char *error = NULL;
  typedef hipError_t (*hipDriverGetVersion_fn)(int *driverVersion);
  hipDriverGetVersion_fn hipDriverGetVersion;
  dlerror(); // Clear existing errors
  hipDriverGetVersion =
      (hipDriverGetVersion_fn)dlsym(lib, "hipDriverGetVersion");
  error = dlerror();
  if (error) {
    PyErr_SetString(PyExc_RuntimeError,
                    "cannot query 'hipDriverGetVersion' from libamdhip64.so");
    dlclose(lib);
    return -1;
  }

  (void)hipDriverGetVersion(&hipVersion);
  char msgBuff[TRITON_HIP_MSG_BUFF_SIZE] = {0};

  const int hipMajVersion = TRITON_HIP_DRIVER_EXTRACT_MAJOR_VERSION(hipVersion);
  if (hipMajVersion < TRITON_HIP_DRIVER_REQ_MAJOR_VERSION) {
    const int hipMinVersion =
        TRITON_HIP_DRIVER_EXTRACT_MINOR_VERSION(hipVersion);
    const int hipPatchVersion =
        TRITON_HIP_DRIVER_EXTRACT_PATCH_VERSION(hipVersion);
    snprintf(msgBuff, sizeof(msgBuff),
             "libamdhip64 version %d.%d.%d is not supported! Required major "
             "version is >=%d.",
             hipMajVersion, hipMinVersion, hipPatchVersion,
             TRITON_HIP_DRIVER_REQ_MAJOR_VERSION);
    PyErr_SetString(PyExc_RuntimeError, msgBuff);
    dlclose(lib);
    return -1;
  }

  TRITON_HIP_DRIVER_LOG_VERSION(hipVersion, msgBuff);

  return hipVersion;
}

bool initSymbolTable() {
  void *lib;

  // Go through the list of search paths to dlopen the first HIP driver library.
  int n = sizeof(hipLibSearchPaths) / sizeof(hipLibSearchPaths[0]);
  for (int i = 0; i < n; ++i) {
    void *handle = dlopen(hipLibSearchPaths[i], RTLD_LAZY | RTLD_LOCAL);
    if (handle) {
      lib = handle;
      // printf("[triton] chosen %s\n", hipLibSearchPaths[i]);
    }
  }

  if (!lib) {
    PyErr_SetString(PyExc_RuntimeError, "cannot open libamdhip64.so");
    return false;
  }

  int hipVersion = checkDriverVersion(lib);
  if (hipVersion == -1)
    return false;

  const char *error = NULL;
  typedef hipError_t (*hipGetProcAddress_fn)(
      const char *symbol, void **pfn, int hipVersion, uint64_t hipFlags,
      hipDriverProcAddressQueryResult *symbolStatus);
  hipGetProcAddress_fn hipGetProcAddress;
  dlerror(); // Clear existing errors

  *(void **)&hipGetProcAddress = dlsym(lib, "hipGetProcAddress");
  error = dlerror();
  if (error) {
    PyErr_SetString(PyExc_RuntimeError,
                    "cannot query 'hipGetProcAddress' from libamdhip64.so");
    dlclose(lib);
    return false;
  }

  // Resolve all symbols we are interested in.
  uint64_t hipFlags = 0;
  hipDriverProcAddressQueryResult symbolStatus;
  hipError_t status = hipSuccess;
#define QUERY_EACH_FN(hipSymbolName, ...)                                      \
  status = hipGetProcAddress(#hipSymbolName,                                   \
                             (void **)&hipSymbolTable.hipSymbolName,           \
                             hipVersion, hipFlags, &symbolStatus);             \
  if (status != hipSuccess) {                                                  \
    PyErr_SetString(PyExc_RuntimeError,                                        \
                    "cannot get address for '" #hipSymbolName                  \
                    "' from libamdhip64.so");                                  \
    dlclose(lib);                                                              \
    return false;                                                              \
  }

  HIP_SYMBOL_LIST(QUERY_EACH_FN, QUERY_EACH_FN)

  return true;
}

static inline void gpuAssert(hipError_t code, const char *file, int line) {
  {
    if (code != HIP_SUCCESS) {
      {
        const char *prefix = "Triton Error [HIP]: ";
        const char *str = hipSymbolTable.hipGetErrorString(code);
        char err[TRITON_HIP_MSG_BUFF_SIZE] = {0};
        snprintf(err, sizeof(err), "%s Code: %d, Messsage: %s", prefix, code,
                 str);
        PyGILState_STATE gil_state;
        gil_state = PyGILState_Ensure();
        PyErr_SetString(PyExc_RuntimeError, err);
        PyGILState_Release(gil_state);
      }
    }
  }
}

#define HIP_CHECK_AND_RETURN_NULL(ans)                                         \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
    if (PyErr_Occurred())                                                      \
      return NULL;                                                             \
  }

#define HIP_CHECK(ans)                                                         \
  {{gpuAssert((ans), __FILE__, __LINE__);                                      \
  }                                                                            \
  }

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;

  hipDeviceProp_t props;
  HIP_CHECK_AND_RETURN_NULL(
      hipSymbolTable.hipGetDeviceProperties(&props, device_id));

  // create a struct to hold device properties
  return Py_BuildValue(
      "{s:i, s:i, s:i, s:i, s:i, s:i, s:s, s:i, s:i, s:i}", "max_shared_mem",
      props.sharedMemPerBlock, "max_num_regs", props.regsPerBlock,
      "multiprocessor_count", props.multiProcessorCount, "sm_clock_rate",
      props.clockRate, "mem_clock_rate", props.memoryClockRate, "mem_bus_width",
      props.memoryBusWidth, "arch", props.gcnArchName, "warpSize",
      props.warpSize, "max_threads_per_sm", props.maxThreadsPerMultiProcessor,
      "cooperativeLaunch", props.cooperativeLaunch);
}

static PyObject *loadBinary(PyObject *self, PyObject *args) {
  const char *name;
  const char *data;
  Py_ssize_t data_size;
  int shared;
  int device;
  if (!PyArg_ParseTuple(args, "ss#ii", &name, &data, &data_size, &shared,
                        &device)) {
    return NULL;
  }

  // set HIP options
  hipJitOption opt[] = {hipJitOptionErrorLogBufferSizeBytes,
                        hipJitOptionErrorLogBuffer,
                        hipJitOptionInfoLogBufferSizeBytes,
                        hipJitOptionInfoLogBuffer, hipJitOptionLogVerbose};
  const unsigned int errbufsize = 8192;
  const unsigned int logbufsize = 8192;
  char _err[errbufsize];
  char _log[logbufsize];
  void *optval[] = {(void *)(uintptr_t)errbufsize, (void *)_err,
                    (void *)(uintptr_t)logbufsize, (void *)_log, (void *)1};

  // launch HIP Binary
  hipModule_t mod;
  hipFunction_t fun;
  HIP_CHECK_AND_RETURN_NULL(
      hipSymbolTable.hipModuleLoadDataEx(&mod, data, 5, opt, optval))
  HIP_CHECK_AND_RETURN_NULL(
      hipSymbolTable.hipModuleGetFunction(&fun, mod, name));

  // get allocated registers and spilled registers from the function
  int n_regs = 0;
  int n_spills = 0;
  int32_t n_max_threads = 0;
  hipSymbolTable.hipFuncGetAttribute(&n_regs, HIP_FUNC_ATTRIBUTE_NUM_REGS, fun);
  hipSymbolTable.hipFuncGetAttribute(&n_spills,
                                     HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun);
  hipSymbolTable.hipFuncGetAttribute(
      &n_max_threads, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fun);
  n_spills /= 4;
  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("(KKiii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                       n_spills, n_max_threads);
}

static PyObject *createTDMDescriptor(PyObject *self, PyObject *args) {
  int elementBitWidth;
  PyObject *blockSize;
  int numWarps;
  int padInterval;
  int padAmount;
  PyObject *shape;
  PyObject *strides;
  unsigned long long globalAddress;

  if (!PyArg_ParseTuple(args, "iOiiiOOK", &elementBitWidth, &blockSize,
                        &numWarps, &padInterval, &padAmount, &shape, &strides,
                        &globalAddress)) {
    return NULL;
  }

  PyTDMDescriptorObject *descObj = (PyTDMDescriptorObject *)PyObject_CallObject(
      (PyObject *)&PyTDMDescriptorType, NULL);
  if (!descObj)
    return NULL;

  PyObject *blockSizeFast = NULL;
  PyObject *shapeFast = NULL;
  PyObject *stridesFast = NULL;

  uint32_t blockSizeInt[5];
  uint32_t shapeInt[5];
  uint32_t stridesInt[5];

  blockSizeFast = PySequence_Fast(blockSize, "blockSize must be a sequence");
  if (!blockSizeFast)
    goto cleanup;
  int rank = PySequence_Fast_GET_SIZE(blockSizeFast);
  if (rank == 0 || rank > 5) {
    PyErr_SetString(PyExc_RuntimeError, "rank must be between 1 and 5");
    goto cleanup;
  }

  for (int i = 0; i < rank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(blockSizeFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "block size must be an int");
      goto cleanup;
    }
    blockSizeInt[i] = PyLong_AsLong(item);
  }

  shapeFast = PySequence_Fast(shape, "shape must be a sequence");
  if (!shapeFast)
    goto cleanup;

  if (rank != PySequence_Fast_GET_SIZE(shapeFast)) {
    PyErr_SetString(PyExc_RuntimeError, "rank mismatch");
    goto cleanup;
  }
  for (int i = 0; i < rank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(shapeFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "shape must be an int");
      goto cleanup;
    }
    shapeInt[i] = PyLong_AsLong(item);
  }

  stridesFast = PySequence_Fast(strides, "strides must be a sequence");
  if (!stridesFast)
    goto cleanup;

  if (rank != PySequence_Fast_GET_SIZE(stridesFast)) {
    PyErr_SetString(PyExc_RuntimeError, "rank mismatch");
    goto cleanup;
  }
  for (int i = 0; i < rank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(stridesFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "shape must be an int");
      goto cleanup;
    }
    stridesInt[i] = PyLong_AsLong(item);
  }

  Py_DECREF(blockSizeFast);
  blockSizeFast = NULL;
  Py_DECREF(shapeFast);
  shapeFast = NULL;
  Py_DECREF(stridesFast);
  stridesFast = NULL;

  bool success = encodeTDMDescriptor(
      &descObj->desc, elementBitWidth, blockSizeInt, numWarps, padInterval,
      padAmount, shapeInt, stridesInt, globalAddress, rank);
  if (!success) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to encode TDM descriptor");
    goto cleanup;
  }

  return (PyObject *)descObj;

cleanup:
  Py_XDECREF(blockSizeFast);
  Py_XDECREF(shapeFast);
  Py_XDECREF(stridesFast);
  Py_XDECREF(descObj);
  return NULL;
}

static void _launch(int gridX, int gridY, int gridZ, int num_warps,
                    int num_ctas, int launch_cooperative_grid,
                    int shared_memory, int warp_size, hipStream_t stream,
                    hipFunction_t function, void **params) {
  if (gridX * gridY * gridZ == 0)
    return;
  if (num_ctas > 1) {
    if (!hipSymbolTable.hipDrvLaunchKernelEx) {
      PyErr_SetString(
          PyExc_RuntimeError,
          "missing hipDrvLaunchKernelEx symbol; please update HIP runtime");
      return;
    }

    hipLaunchAttribute attributes[2];
    // Attribute0: Cluster dimensions
    attributes[0].id = 4;
    int *cluster_dims = (int *)attributes[0].val.pad;
    cluster_dims[0] = num_ctas;
    cluster_dims[1] = 1;
    cluster_dims[2] = 1;
    // Attribute1: Cooperative launch
    attributes[1].id = hipLaunchAttributeCooperative;
    attributes[1].val.cooperative = launch_cooperative_grid;

    HIP_LAUNCH_CONFIG config = {
        gridX * num_ctas,      gridY,  gridZ,        // Grid size
        warp_size * num_warps, 1,      1,            // Block size
        shared_memory,         stream, attributes, 2 // Number of attributes
    };
    HIP_CHECK(
        hipSymbolTable.hipDrvLaunchKernelEx(&config, function, params, 0));
    return;
  } else if (launch_cooperative_grid) {
    HIP_CHECK(hipSymbolTable.hipModuleLaunchCooperativeKernel(
        function, gridX, gridY, gridZ, warp_size * num_warps, 1, 1,
        shared_memory, stream, params, 0));
    return;
  } else {
    HIP_CHECK(hipSymbolTable.hipModuleLaunchKernel(
        function, gridX, gridY, gridZ, warp_size * num_warps, 1, 1,
        shared_memory, stream, params, 0));
  }
}

static PyObject *data_ptr_str = NULL;

bool extractPointer(void *ptr, PyObject *obj) {
  hipDeviceptr_t *dev_ptr = ptr;
  if (obj == Py_None) {
    *dev_ptr = (hipDeviceptr_t)0; // valid nullptr
    return true;
  }
  if (PyLong_Check(obj)) {
    *dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(obj);
    return true;
  }
  PyObject *ret = PyObject_CallMethodNoArgs(obj, data_ptr_str);
  if (!ret) {
    PyErr_SetString(
        PyExc_TypeError,
        "Pointer argument must be either uint64 or have data_ptr method");
    return false;
  }
  if (!PyLong_Check(ret)) {
    PyErr_SetString(PyExc_TypeError,
                    "data_ptr method of Pointer object must return 64-bit int");
    return false;
  }
  *dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(ret);
  Py_DECREF(ret);
  if (*dev_ptr == 0) {
    return true; // valid nullptr
  }
  hipError_t status = hipSymbolTable.hipPointerGetAttribute(
      dev_ptr, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, *dev_ptr);
  if (status == hipErrorInvalidValue) {
    PyErr_Format(PyExc_ValueError, "Pointer argument (at %d) cannot be "
                                   "accessed from Triton (cpu tensor?)");
    // Clear and ignore HIP error
    (void)hipSymbolTable.hipGetLastError();
    return false;
  }
  return true;
}

bool extractI8(void *ptr, PyObject *obj) {
  *((int8_t *)ptr) = PyLong_AsLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractI16(void *ptr, PyObject *obj) {
  *((int16_t *)ptr) = PyLong_AsLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractI32(void *ptr, PyObject *obj) {
  *((int32_t *)ptr) = PyLong_AsLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractI64(void *ptr, PyObject *obj) {
  *((int64_t *)ptr) = PyLong_AsLongLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractU8(void *ptr, PyObject *obj) {
  *((uint8_t *)ptr) = PyLong_AsUnsignedLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractU16(void *ptr, PyObject *obj) {
  *((uint16_t *)ptr) = PyLong_AsUnsignedLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractU32(void *ptr, PyObject *obj) {
  *((uint32_t *)ptr) = PyLong_AsUnsignedLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractU64(void *ptr, PyObject *obj) {
  *((uint64_t *)ptr) = PyLong_AsUnsignedLongLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractFP16(void *ptr, PyObject *obj) {
  double temp_double = PyFloat_AsDouble(obj);
  uint16_t result;
  // from https://github.com/python/pythoncapi-compat
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 &&            \
    !defined(PYPY_VERSION)
  _PyFloat_Pack2(temp_double, (unsigned char *)&result, 1);
#else
  PyFloat_Pack2(temp_double, (char *)&result, 1);
#endif
  *((uint16_t *)ptr) = result;
  return PyErr_Occurred() == NULL;
}

bool extractBF16(void *ptr, PyObject *obj) {
  double temp_double = PyFloat_AsDouble(obj);
  float f32 = (float)temp_double;
  uint32_t u32 = *(uint32_t *)&f32;
  *((uint16_t *)ptr) = (u32 >> 16);
  return PyErr_Occurred() == NULL;
}

bool extractFP32(void *ptr, PyObject *obj) {
  double temp_double = PyFloat_AsDouble(obj);
  float f32 = (float)temp_double;
  *((uint32_t *)ptr) = *(uint32_t *)&f32;
  return PyErr_Occurred() == NULL;
}

bool extractFP64(void *ptr, PyObject *obj) {
  double temp_double = PyFloat_AsDouble(obj);
  *((uint64_t *)ptr) = *(uint64_t *)&temp_double;
  return PyErr_Occurred() == NULL;
}

// Extract a TDM descriptor from a python object, and store it to the
// memory location pointed by ptr.
bool extractTDMDescriptor(void *ptr, PyObject *obj) {
  TDMDescriptor *desc = &((PyTDMDescriptorObject *)obj)->desc;
  if (desc == NULL) {
    PyErr_Format(PyExc_TypeError,
                 "object must be of type PyTDMDescriptor, got %s",
                 Py_TYPE(obj)->tp_name);
    return false;
  }
  *((TDMDescriptor *)ptr) = *desc;
  return true;
}

typedef bool (*ExtractorFunc)(void *ptr, PyObject *obj);

#define MAX_NAMES_PER_EXTRACTOR 2

typedef struct {
  ExtractorFunc extract;
  size_t size;
  const char *name[MAX_NAMES_PER_EXTRACTOR];
} Extractor;

typedef enum {
  EXTRACTOR_UNKOWN_INDEX = 0,
  // pointers
  EXTRACTOR_POINTER_INDEX = 1,
  // ints
  EXTRACTOR_INT8_INDEX = 2,
  EXTRACTOR_INT16_INDEX = 3,
  EXTRACTOR_INT32_INDEX = 4,
  EXTRACTOR_INT64_INDEX = 5,
  // uints
  EXTRACTOR_UINT8_INDEX = 6,
  EXTRACTOR_UINT16_INDEX = 7,
  EXTRACTOR_UINT32_INDEX = 8,
  EXTRACTOR_UINT64_INDEX = 9,
  // floats
  EXTRACTOR_FP16_INDEX = 10,
  EXTRACTOR_BF16_INDEX = 11,
  EXTRACTOR_FP32_INDEX = 12,
  EXTRACTOR_FP64_INDEX = 13,
  // custom
  EXTRACTOR_TDMDESC_INDEX = 14,
  // last entry to have a count
  EXTRACTOR_TYPE_COUNT
} ExtractorTypeIndex;

Extractor extraction_map[EXTRACTOR_TYPE_COUNT] = {
    [EXTRACTOR_UNKOWN_INDEX] =
        (Extractor){.extract = NULL, .size = 0, .name = NULL},
    [EXTRACTOR_POINTER_INDEX] = (Extractor){.extract = extractPointer,
                                            .size = sizeof(hipDeviceptr_t),
                                            .name = NULL},
    [EXTRACTOR_INT8_INDEX] = (Extractor){.extract = extractI8,
                                         .size = sizeof(int8_t),
                                         .name = {"i8"}},
    [EXTRACTOR_INT16_INDEX] = (Extractor){.extract = extractI16,
                                          .size = sizeof(int16_t),
                                          .name = {"i16"}},
    [EXTRACTOR_INT32_INDEX] = (Extractor){.extract = extractI32,
                                          .size = sizeof(int32_t),
                                          .name = {"i1", "i32"}},
    [EXTRACTOR_INT64_INDEX] = (Extractor){.extract = extractI64,
                                          .size = sizeof(int64_t),
                                          .name = {"i64"}},
    [EXTRACTOR_UINT8_INDEX] = (Extractor){.extract = extractU8,
                                          .size = sizeof(uint8_t),
                                          .name = {"u8"}},
    [EXTRACTOR_UINT16_INDEX] = (Extractor){.extract = extractU16,
                                           .size = sizeof(uint16_t),
                                           .name = {"u16"}},
    [EXTRACTOR_UINT32_INDEX] = (Extractor){.extract = extractU32,
                                           .size = sizeof(uint32_t),
                                           .name = {"u1", "u32"}},
    [EXTRACTOR_UINT64_INDEX] = (Extractor){.extract = extractU64,
                                           .size = sizeof(uint64_t),
                                           .name = {"u64"}},
    [EXTRACTOR_FP16_INDEX] = (Extractor){.extract = extractFP16,
                                         .size = sizeof(uint16_t),
                                         .name = {"fp16"}},
    [EXTRACTOR_BF16_INDEX] = (Extractor){.extract = extractBF16,
                                         .size = sizeof(uint16_t),
                                         .name = {"bf16"}},
    [EXTRACTOR_FP32_INDEX] = (Extractor){.extract = extractFP32,
                                         .size = sizeof(uint32_t),
                                         .name = {"fp32", "f32"}},
    [EXTRACTOR_FP64_INDEX] = (Extractor){.extract = extractFP64,
                                         .size = sizeof(uint64_t),
                                         .name = {"fp64"}},
    [EXTRACTOR_TDMDESC_INDEX] = (Extractor){.extract = extractTDMDescriptor,
                                            .size = sizeof(TDMDescriptor),
                                            .name = {"tensordesc"}},
};

Extractor getExtractor(uint8_t index) {
  if (index >= EXTRACTOR_TYPE_COUNT) {
    return extraction_map[EXTRACTOR_UNKOWN_INDEX];
  }
  return extraction_map[index];
}

bool isMatch(const char *type_bytes, ExtractorTypeIndex idx) {
  Extractor extractor = extraction_map[idx];
  for (int j = 0; j < MAX_NAMES_PER_EXTRACTOR; j++) {
    if (extractor.name[j] != NULL &&
        strcmp(type_bytes, extractor.name[j]) == 0) {
      return true;
    }
  }
  return false;
}

ExtractorTypeIndex getExtractorIndex(PyObject *type) {
  Py_ssize_t type_len = 0;
  const char *type_bytes = PyUnicode_AsUTF8AndSize(type, &type_len);
  if (!type_bytes) {
    return EXTRACTOR_UNKOWN_INDEX;
  }
  if (type_len < 2) {
    PyErr_Format(PyExc_RuntimeError, "Unexpected data type: %R", type);
    return EXTRACTOR_UNKOWN_INDEX;
  }
  // Examples: '*fp32', 'fp32', 'i8', etc.
  if (type_bytes[0] == '*') {
    return EXTRACTOR_POINTER_INDEX;
  }
  for (ExtractorTypeIndex i = EXTRACTOR_INT8_INDEX; i < EXTRACTOR_TYPE_COUNT;
       i++) {
    if (isMatch(type_bytes, i)) {
      return i;
    }
  }

  PyErr_Format(PyExc_RuntimeError, "Unknown data type: %R", type);
  return EXTRACTOR_UNKOWN_INDEX;
}

// Takes in a list of types (ex: ['*fp32', 'u8', 'tensordesc']) and returns
// a bytes array that represent extractors for quick argument extraction
// when launching.
static PyObject *buildSignatureMetadata(PyObject *self, PyObject *args) {
  PyObject *signature = NULL;
  if (!PyArg_ParseTuple(args, "O", &signature)) {
    return NULL;
  }
  PyObject *fast_signature = PySequence_Fast(
      signature, "Expected kernel_arg_types to be a sequence or iterable");
  if (!fast_signature) {
    return NULL;
  }
  Py_ssize_t signature_size = PySequence_Fast_GET_SIZE(fast_signature);
  PyObject **signature_items = PySequence_Fast_ITEMS(fast_signature);

  // Create return bytes object.
  PyObject *ret_bytes = PyBytes_FromStringAndSize(NULL, signature_size);
  if (ret_bytes == NULL) {
    Py_XDECREF(fast_signature);
    return NULL;
  }
  char *buffer = PyBytes_AS_STRING(ret_bytes);
  for (Py_ssize_t i = 0; i < signature_size; ++i) {
    ExtractorTypeIndex extractor_idx = getExtractorIndex(signature_items[i]);
    if (extractor_idx == EXTRACTOR_UNKOWN_INDEX) {
      goto cleanup;
    }
    buffer[i] = (uint8_t)extractor_idx;
  }

  Py_XDECREF(fast_signature);
  return ret_bytes;

cleanup:
  Py_XDECREF(fast_signature);
  Py_XDECREF(ret_bytes);
  return NULL;
}

bool extractArgs(PyObject **final_list, int *list_idx, PyObject *kernel_args,
                 PyObject *arg_annotations) {
  // Extract arg annotations
  PyObject *fast_annotations = PySequence_Fast(
      arg_annotations, "Expected arg_annotations to be a sequence or iterable");
  if (!fast_annotations) {
    goto cleanup;
  }
  Py_ssize_t num_annotations = PySequence_Fast_GET_SIZE(fast_annotations);
  PyObject **annotations = PySequence_Fast_ITEMS(fast_annotations);

  PyObject *fast_args = PySequence_Fast(
      kernel_args, "Expected kernel_args to be a sequence or iterable");
  if (!fast_args) {
    goto cleanup;
  }
  PyObject **args = PySequence_Fast_ITEMS(fast_args);

  int arg_idx = 0;
  for (int i = 0; i < num_annotations; ++i) {
    PyKernelArgObject *annotation = (PyKernelArgObject *)annotations[i];
    switch (annotation->type) {
    case ARG_KERNEL:
      final_list[(*list_idx)++] = args[arg_idx++];
      break;
    case ARG_TUPLE:
      if (!extractArgs(final_list, list_idx, args[arg_idx++],
                       annotation->nested_tuple)) {
        goto cleanup;
      }
      break;
    case ARG_CONSTEXPR:
      arg_idx++;
      break;
    }
  }
  Py_DECREF(fast_annotations);
  Py_DECREF(fast_args);
  return true;

cleanup:
  Py_XDECREF(fast_annotations);
  Py_XDECREF(fast_args);
  return false;
}

bool launchHook(PyObject *hook, PyObject *metadata) {
  if (hook != Py_None) {
    PyObject *ret = PyObject_CallOneArg(hook, metadata);
    if (!ret) {
      return false;
    }
    Py_DECREF(ret);
  }
  return true;
}

static PyObject *launchKernel(PyObject *self, PyObject *args) {
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int launch_cooperative_grid;
  PyObject *profile_scratch_obj = NULL;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  int num_warps, num_ctas, shared_memory;
  PyObject *launch_metadata = NULL;
  int warp_size;
  PyObject *arg_annotations = NULL;
  Py_buffer signature;
  PyObject *kernel_args = NULL;
  if (!PyArg_ParseTuple(args, "piiiKKO(iii)OOOiOy*O", &launch_cooperative_grid,
                        &gridX, &gridY, &gridZ, &_stream, &_function,
                        &profile_scratch_obj, &num_warps, &num_ctas,
                        &shared_memory, &launch_metadata, &launch_enter_hook,
                        &launch_exit_hook, &warp_size, &arg_annotations,
                        &signature, &kernel_args)) {
    return NULL;
  }

  // launch entry hook.
  if (!launchHook(launch_enter_hook, launch_metadata)) {
    goto cleanup;
  }

  uint8_t *extractor_data = (uint8_t *)signature.buf;
  Py_ssize_t num_args = signature.len;

  // Extract kernel parameters - flatten tuples & remove constexpr.
  PyObject **args_data = (PyObject **)alloca(num_args * sizeof(PyObject *));
  if (args_data == NULL) {
    goto cleanup;
  }
  int list_idx = 0;
  if (!extractArgs(args_data, &list_idx, kernel_args, arg_annotations)) {
    goto cleanup;
  }

  // Number of parameters passed to kernel. + 2 for global & profile scratch.
  int num_params = num_args + 2;
  void **params = (void **)alloca(num_params * sizeof(void *));
  int params_idx = 0;
  // This loop has to stay in the same function that owns params, since we are
  // using alloca to allocate pointers to it on the stack of the function.
  for (Py_ssize_t i = 0; i < num_args; ++i) {
    // Get extractor that will send back a struct with
    // * size for allocation
    // * function to call to put the parameter in params buffer
    Extractor extractor = getExtractor(extractor_data[i]);
    if (extractor.extract == NULL) {
      goto cleanup;
    }
    PyObject *current_arg = args_data[i];
    params[params_idx] = alloca(extractor.size);
    if (!extractor.extract(params[params_idx++], current_arg)) {
      goto cleanup;
    }
  }
  // Add global scratch object (nullptr).
  params[params_idx] = alloca(sizeof(void *));
  if (!extractPointer(params[params_idx++], Py_None)) {
    goto cleanup;
  }
  // Add profile scratch object.
  params[params_idx] = alloca(sizeof(void *));
  if (!extractPointer(params[params_idx++], profile_scratch_obj)) {
    goto cleanup;
  }

  _launch(gridX, gridY, gridZ, num_warps, num_ctas, launch_cooperative_grid,
          shared_memory, warp_size, (hipStream_t)_stream,
          (hipFunction_t)_function, params);

  if (!launchHook(launch_exit_hook, launch_metadata)) {
    goto cleanup;
  }

  if (PyErr_Occurred()) {
    goto cleanup;
  }
  PyBuffer_Release(&signature);
  Py_RETURN_NONE;

cleanup:
  PyBuffer_Release(&signature);
  return NULL;
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided hsaco into HIP driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {"create_tdm_descriptor", createTDMDescriptor, METH_VARARGS,
     "create a host-side TDM descriptor"},
    {"build_signature_metadata", buildSignatureMetadata, METH_VARARGS,
     "Calling it with a signature list (ex: ['*fp32', 'u8', 'tensordesc']), "
     "will return metadata to be passed into 'launch' for quicker "
     "argument parsing."},
    {"launch", launchKernel, METH_VARARGS,
     "Entry point for all kernels with this signature"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "hip_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_hip_utils(void) {
  if (!initSymbolTable()) {
    return NULL;
  }

  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);

  if (PyType_Ready(&PyTDMDescriptorType) < 0) {
    return NULL;
  }
  if (PyType_Ready(&PyKernelArgType) < 0) {
    return NULL;
  }
  data_ptr_str = PyUnicode_InternFromString("data_ptr");
  if (data_ptr_str == NULL) {
    return NULL;
  }
  Py_INCREF(&PyTDMDescriptorType);
  PyModule_AddObject(m, "PyTDMDescriptor", (PyObject *)&PyTDMDescriptorType);
  Py_INCREF(&PyKernelArgType);
  PyModule_AddObject(m, "PyKernelArg", (PyObject *)&PyKernelArgType);
  PyModule_AddIntConstant(m, "ARG_CONSTEXPR", ARG_CONSTEXPR);
  PyModule_AddIntConstant(m, "ARG_KERNEL", ARG_KERNEL);
  PyModule_AddIntConstant(m, "ARG_TUPLE", ARG_TUPLE);

  return m;
}
