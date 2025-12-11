#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

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

// TODO: Both host-side and device-side TDM descriptor follow the same encoding
// format. Consider to add a common utility to remove duplicate code.
static bool encodeTDMDescriptor(TDMDescriptor *desc, int elementBitWidth,
                                uint32_t *blockSize, int numWarps,
                                int padInterval, int padAmount, uint32_t *shape,
                                uint32_t *strides, uint64_t globalAddress,
                                int rank) {
  // NYI: TDM > 2D cases
  if (rank != 2)
    return false;

  // Get warp distribution
  uint32_t numWarpsDim0 = numWarps;
  for (; numWarpsDim0 > blockSize[0]; numWarpsDim0 /= 2)
    ;
  uint32_t numWarpsDim1 = numWarps / numWarpsDim0;
  if (!(numWarpsDim0 > 0 && blockSize[1] % numWarpsDim1 == 0))
    return false;

  uint32_t blockSize0 = (blockSize[0] + numWarpsDim0 - 1) / numWarpsDim0;
  uint32_t blockSize1 = (blockSize[1] + numWarpsDim1 - 1) / numWarpsDim1;

  // group0 (128 bits / 4 dwords) effective bit encoding:
  // [120:64]:  global address
  // [127:126]: type - currently always set to 0x2
  desc->group0_2 = (uint32_t)(globalAddress & 0xFFFFFFFF);
  desc->group0_3 = (uint32_t)((globalAddress >> 32) & 0x01FFFFFF);
  desc->group0_3 |= (0x1 << 31);

  // group1 (256 bits / 8 dwords) effective bit encoding:
  // [17:16]:   data size - log2(element size in bytes)
  // [20]:      enable padding
  // [24:22]:   pad interval - log2(pad interval in dwords) - 1
  // [31:25]:   pad amount - pad amount in dwords - 1
  // [79:48]:   tensor shape dim inner
  // [111:80]:  tensor shape dim outer
  // [127:112]: block shape dim inner
  // [143:128]: block shape dim outer
  // [207:160]: tensor stride dim outer (we only use 32 bits)
  int elementSizeInBytes = elementBitWidth / 8;
  int dataSize = log2(elementSizeInBytes);
  desc->group1_0 = (dataSize << 16);
  int dwordSize = 32;
  int padIntervalInDwords = padInterval * elementBitWidth / dwordSize;
  int padAmountInDwords = padAmount * elementBitWidth / dwordSize;
  if (padIntervalInDwords > 0 && padAmountInDwords > 0) {
    int log2PadInterval = log2(padIntervalInDwords);
    desc->group1_0 |= (1 << 20);
    desc->group1_0 |= ((log2PadInterval - 1) << 22);
    desc->group1_0 |= ((padAmountInDwords - 1) << 25);
  }
  desc->group1_1 = (shape[1] << 16);
  desc->group1_2 = (shape[1] >> 16);
  desc->group1_2 |= (shape[0] << 16);
  desc->group1_3 = (shape[0] >> 16);
  desc->group1_3 |= (blockSize1 << 16);
  desc->group1_4 = (blockSize0 & 0xFFFF);
  desc->group1_5 = strides[0];

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
  FOR_EACH_STR_FN(hipGetErrorString, hipError_t hipError)                      \
  FOR_EACH_ERR_FN(hipGetDeviceProperties, hipDeviceProp_t *prop, int deviceId) \
  FOR_EACH_ERR_FN(hipModuleLoadDataEx, hipModule_t *module, const void *image, \
                  unsigned int numOptions, hipJitOption *options,              \
                  void **optionValues)                                         \
  FOR_EACH_ERR_FN(hipModuleGetFunction, hipFunction_t *function,               \
                  hipModule_t module, const char *kname)                       \
  FOR_EACH_ERR_FN(hipFuncGetAttribute, int *, hipFunction_attribute attr,      \
                  hipFunction_t function)

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

#define HIP_CHECK(ans)                                                         \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
    if (PyErr_Occurred())                                                      \
      return NULL;                                                             \
  }

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;

  hipDeviceProp_t props;
  HIP_CHECK(hipSymbolTable.hipGetDeviceProperties(&props, device_id));

  // create a struct to hold device properties
  return Py_BuildValue(
      "{s:i, s:i, s:i, s:i, s:i, s:i, s:s, s:i, s:i}", "max_shared_mem",
      props.sharedMemPerBlock, "max_num_regs", props.regsPerBlock,
      "multiprocessor_count", props.multiProcessorCount, "sm_clock_rate",
      props.clockRate, "mem_clock_rate", props.memoryClockRate, "mem_bus_width",
      props.memoryBusWidth, "arch", props.gcnArchName, "warpSize",
      props.warpSize, "max_threads_per_sm", props.maxThreadsPerMultiProcessor);
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
  HIP_CHECK(hipSymbolTable.hipModuleLoadDataEx(&mod, data, 5, opt, optval))
  HIP_CHECK(hipSymbolTable.hipModuleGetFunction(&fun, mod, name));

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

  uint32_t blockSizeInt[2];
  uint32_t shapeInt[2];
  uint32_t stridesInt[2];

  blockSizeFast = PySequence_Fast(blockSize, "blockSize must be a sequence");
  if (!blockSizeFast)
    goto cleanup;
  int rank = PySequence_Fast_GET_SIZE(blockSizeFast);
  if (rank != 2) {
    PyErr_SetString(PyExc_RuntimeError, "rank must be 2");
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

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided hsaco into HIP driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {"create_tdm_descriptor", createTDMDescriptor, METH_VARARGS,
     "create a host-side TDM descriptor"},
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

  if (PyType_Ready(&PyTDMDescriptorType) < 0)
    return NULL;
  Py_INCREF(&PyTDMDescriptorType);
  PyModule_AddObject(m, "PyTDMDescriptor", (PyObject *)&PyTDMDescriptorType);

  return m;
}
