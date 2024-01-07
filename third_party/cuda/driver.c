#include "cuda.h"
#include <dlfcn.h>
#include <stdbool.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Raises a Python exception and returns false if code is not CUDA_SUCCESS.
static bool gpuAssert(CUresult code, const char *file, int line) {
  if (code == CUDA_SUCCESS)
    return true;

  const char *prefix = "Triton Error [CUDA]: ";
  const char *str;
  cuGetErrorString(code, &str);
  char err[1024] = {0};
  strcat(err, prefix);
  strcat(err, str);
  PyGILState_STATE gil_state;
  gil_state = PyGILState_Ensure();
  PyErr_SetString(PyExc_RuntimeError, err);
  PyGILState_Release(gil_state);
  return false;
}

// To be used only *outside* a Py_{BEGIN,END}_ALLOW_THREADS block.
#define CUDA_CHECK_AND_RETURN_NULL(ans)                                        \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__))                                 \
      return NULL;                                                             \
  } while (0)

// To be used inside a Py_{BEGIN,END}_ALLOW_THREADS block.
#define CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(ans)                          \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__)) {                               \
      PyEval_RestoreThread(_save);                                             \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

#define ADD_ENUM_ITEM(value)                                                   \
  do {                                                                         \
    PyObject *py_value = PyLong_FromLong(value);                               \
    PyDict_SetItemString(enum_dict, #value, py_value);                         \
  } while (0)

#define ADD_ENUM_ITEM_0()
#define ADD_ENUM_ITEM_1(v1) ADD_ENUM_ITEM(v1)
#define ADD_ENUM_ITEM_2(v1, v2)                                                \
  ADD_ENUM_ITEM(v1);                                                           \
  ADD_ENUM_ITEM(v2);
#define ADD_ENUM_ITEM_3(v1, v2, v3)                                            \
  ADD_ENUM_ITEM(v1);                                                           \
  ADD_ENUM_ITEM(v2);                                                           \
  ADD_ENUM_ITEM(v3);
#define ADD_ENUM_ITEM_4(v1, v2, v3, v4)                                        \
  ADD_ENUM_ITEM(v1);                                                           \
  ADD_ENUM_ITEM(v2);                                                           \
  ADD_ENUM_ITEM(v3);                                                           \
  ADD_ENUM_ITEM(v4);
#define ADD_ENUM_ITEM_5(v1, v2, v3, v4, v5)                                    \
  ADD_ENUM_ITEM_2(v1, v2);                                                     \
  ADD_ENUM_ITEM_3(v3, v4, v5);
#define ADD_ENUM_ITEM_6(v1, v2, v3, v4, v5, v6)                                \
  ADD_ENUM_ITEM_2(v1, v2);                                                     \
  ADD_ENUM_ITEM_4(v3, v4, v5, v6);
#define ADD_ENUM_ITEM_7(v1, v2, v3, v4, v5, v6, v7)                            \
  ADD_ENUM_ITEM_3(v1, v2, v3);                                                 \
  ADD_ENUM_ITEM_4(v4, v5, v6, v7);
#define ADD_ENUM_ITEM_8(v1, v2, v3, v4, v5, v6, v7, v8)                        \
  ADD_ENUM_ITEM_4(v1, v2, v3, v4);                                             \
  ADD_ENUM_ITEM_4(v5, v6, v7, v8);
#define ADD_ENUM_ITEM_9(v1, v2, v3, v4, v5, v6, v7, v8, v9)                    \
  ADD_ENUM_ITEM_5(v1, v2, v3, v4, v5);                                         \
  ADD_ENUM_ITEM_4(v6, v7, v8, v9);
#define ADD_ENUM_ITEM_10(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)              \
  ADD_ENUM_ITEM_5(v1, v2, v3, v4, v5);                                         \
  ADD_ENUM_ITEM_5(v6, v7, v8, v9, v10);
#define ADD_ENUM_ITEM_11(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11)         \
  ADD_ENUM_ITEM_6(v1, v2, v3, v4, v5, v6);                                     \
  ADD_ENUM_ITEM_5(v7, v8, v9, v10, v11);
#define ADD_ENUM_ITEM_12(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12)    \
  ADD_ENUM_ITEM_6(v1, v2, v3, v4, v5, v6);                                     \
  ADD_ENUM_ITEM_6(v7, v8, v9, v10, v11, v12);
#define ADD_ENUM_ITEM_13(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12,    \
                         v13)                                                  \
  ADD_ENUM_ITEM_7(v1, v2, v3, v4, v5, v6, v7);                                 \
  ADD_ENUM_ITEM_6(v8, v9, v10, v11, v12, v13);
#define ADD_ENUM_ITEM_14(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12,    \
                         v13, v14)                                             \
  ADD_ENUM_ITEM_7(v1, v2, v3, v4, v5, v6, v7);                                 \
  ADD_ENUM_ITEM_7(v8, v9, v10, v11, v12, v13, v14);

#define DISPATCH_ARGS_N(_14, _13, _12, _11, _10, _9, _8, _7, _6, _5, _4, _3,   \
                        _2, _1, N, ...)                                        \
  ADD_ENUM_ITEM_##N
#define DISPATCH_ARGS(...)                                                     \
  DISPATCH_ARGS_N(__VA_ARGS__, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,  \
                  0)                                                           \
  (__VA_ARGS__)

#define ADD_ENUM_TO_MODULE(module, enum_name, ...)                             \
  do {                                                                         \
    PyObject *enum_dict = PyDict_New();                                        \
    DISPATCH_ARGS(__VA_ARGS__)                                                 \
    if (enum_dict != NULL) {                                                   \
      PyObject_SetAttrString(module, #enum_name, enum_dict);                   \
    }                                                                          \
  } while (0)

static void defineEnums(PyObject *self) {
  ADD_ENUM_TO_MODULE(
      self, CUtensorMapDataType, CU_TENSOR_MAP_DATA_TYPE_UINT8,
      CU_TENSOR_MAP_DATA_TYPE_UINT16, CU_TENSOR_MAP_DATA_TYPE_UINT32,
      CU_TENSOR_MAP_DATA_TYPE_INT32, CU_TENSOR_MAP_DATA_TYPE_UINT64,
      CU_TENSOR_MAP_DATA_TYPE_INT64, CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
      CU_TENSOR_MAP_DATA_TYPE_FLOAT32, CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ,
      CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ);

  ADD_ENUM_TO_MODULE(self, CUtensorMapInterleave, CU_TENSOR_MAP_INTERLEAVE_NONE,
                     CU_TENSOR_MAP_INTERLEAVE_16B,
                     CU_TENSOR_MAP_INTERLEAVE_32B);

  ADD_ENUM_TO_MODULE(self, CUtensorMapSwizzle, CU_TENSOR_MAP_SWIZZLE_NONE,
                     CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_SWIZZLE_64B,
                     CU_TENSOR_MAP_SWIZZLE_128B);

  ADD_ENUM_TO_MODULE(
      self, CUtensorMapL2promotion, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);

  ADD_ENUM_TO_MODULE(self, CUtensorMapFloatOOBfill,
                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
}

typedef struct {
  PyObject_HEAD cuuint32_t value;
} PyCUuint32;

typedef struct {
  PyObject_HEAD cuuint64_t value;
} PyCUuint64;

#define DEFINE_CUUINT_CONSTRUCTOR(NAME, TYPE, FORMAT, VALUE_TYPE)              \
  static PyObject *Py##NAME##_New(PyTypeObject *type, PyObject *args,          \
                                  PyObject *kwds) {                            \
    Py##NAME *self;                                                            \
    VALUE_TYPE value;                                                          \
    if (!PyArg_ParseTuple(args, FORMAT, &value))                               \
      return NULL;                                                             \
    self = (Py##NAME *)type->tp_alloc(type, 0);                                \
    if (self != NULL) {                                                        \
      self->value = (TYPE)value;                                               \
    }                                                                          \
    return (PyObject *)self;                                                   \
  }

DEFINE_CUUINT_CONSTRUCTOR(CUuint32, cuuint32_t, "l", long)
DEFINE_CUUINT_CONSTRUCTOR(CUuint64, cuuint64_t, "L", long long)

static PyTypeObject PyCUuint32_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "cuda_utils.cuuint32_t",
    .tp_basicsize = sizeof(PyCUuint32),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyCUuint32_New,
};

static PyTypeObject PyCUuint64_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "cuda_utils.cuuint64_t",
    .tp_basicsize = sizeof(PyCUuint64),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyCUuint64_New,
};

static void defineTypes(PyObject *self) {
  if (PyType_Ready(&PyCUuint32_Type) < 0) {
    PyErr_SetString(PyExc_TypeError, "Failed to ready cuuint32_t type");
    return;
  }
  Py_INCREF(&PyCUuint32_Type);
  if (PyModule_AddObject(self, "cuuint32_t", (PyObject *)&PyCUuint32_Type) <
      0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Failed to add cuuint32_t type to module");
    return;
  }

  if (PyType_Ready(&PyCUuint64_Type) < 0) {
    PyErr_SetString(PyExc_TypeError, "Failed to ready cuuint64_t type");
    return;
  }
  Py_INCREF(&PyCUuint64_Type);
  if (PyModule_AddObject(self, "cuuint64_t", (PyObject *)&PyCUuint64_Type) <
      0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Failed to add cuuint64_t type to module");
    return;
  }
}

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;
  // Get device handle
  CUdevice device;
  cuDeviceGet(&device, device_id);

  // create a struct to hold device properties
  int max_shared_mem;
  int multiprocessor_count;
  int sm_clock_rate;
  int mem_clock_rate;
  int mem_bus_width;
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &max_shared_mem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &multiprocessor_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &sm_clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &mem_clock_rate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &mem_bus_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));

  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i}", "max_shared_mem",
                       max_shared_mem, "multiprocessor_count",
                       multiprocessor_count, "sm_clock_rate", sm_clock_rate,
                       "mem_clock_rate", mem_clock_rate, "mem_bus_width",
                       mem_bus_width);
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
  CUfunction fun;
  CUmodule mod;
  int32_t n_regs = 0;
  int32_t n_spills = 0;
  // create driver handles
  CUcontext pctx = 0;

  Py_BEGIN_ALLOW_THREADS;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuCtxGetCurrent(&pctx));
  if (!pctx) {
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuDevicePrimaryCtxRetain(&pctx, device));
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuCtxSetCurrent(pctx));
  }

  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuModuleLoadData(&mod, data));
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      cuModuleGetFunction(&fun, mod, name));
  // get allocated registers and spilled registers from the function
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      cuFuncGetAttribute(&n_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fun));
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      cuFuncGetAttribute(&n_spills, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
  n_spills /= 4;
  // set dynamic shared memory if necessary
  int shared_optin;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuDeviceGetAttribute(
      &shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  if (shared > 49152 && shared_optin > 49152) {
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuFuncSetCacheConfig(fun, CU_FUNC_CACHE_PREFER_SHARED));
    int shared_total, shared_static;
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuDeviceGetAttribute(
        &shared_total, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        device));
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuFuncGetAttribute(
        &shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fun));
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuFuncSetAttribute(fun, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_optin - shared_static));
  }
  Py_END_ALLOW_THREADS;

  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("(KKii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                       n_spills);
}

static PyObject *memAlloc(PyObject *self, PyObject *args) {
  size_t bytesize;
  CUdeviceptr dptr;
  CUresult result;

  if (!PyArg_ParseTuple(args, "K", &bytesize)) {
    return NULL; // Error parsing arguments
  }

  Py_BEGIN_ALLOW_THREADS;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuMemAlloc(&dptr, bytesize));
  Py_END_ALLOW_THREADS;

  return PyLong_FromUnsignedLongLong((unsigned long long)dptr);
}

static PyObject *memcpyHtoD(PyObject *self, PyObject *args) {
  unsigned long long dstDevicePtr, srcHostPtr;
  size_t byteCount;
  CUdeviceptr dstDevice;
  const void *srcHost;
  CUresult result;

  if (!PyArg_ParseTuple(args, "KKK", &dstDevicePtr, &srcHostPtr, &byteCount)) {
    return NULL; // Error parsing arguments
  }

  dstDevice = (CUdeviceptr)dstDevicePtr;
  srcHost = (const void *)srcHostPtr;

  Py_BEGIN_ALLOW_THREADS;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      cuMemcpyHtoD(dstDevice, srcHost, byteCount));
  Py_END_ALLOW_THREADS;

  Py_RETURN_NONE;
}

static PyObject *memFree(PyObject *self, PyObject *args) {
  CUdeviceptr dptr;

  if (!PyArg_ParseTuple(args, "K", &dptr)) {
    return NULL; // Error parsing arguments
  }

  Py_BEGIN_ALLOW_THREADS;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuMemFree(dptr));
  Py_END_ALLOW_THREADS;

  Py_RETURN_NONE;
}

// Helper function to convert a Python list to a cuuint64_t array
static cuuint64_t *list_to_cuuint64_array(PyObject *listObj) {
  Py_ssize_t len = PyList_Size(listObj);
  cuuint64_t *array = (cuuint64_t *)malloc(len * sizeof(cuuint64_t));
  for (Py_ssize_t i = 0; i < len; i++) {
    PyObject *item = PyList_GetItem(listObj, i);
    array[i] = (cuuint64_t)PyLong_AsUnsignedLongLong(item);
  }
  return array;
}

// Helper function to convert a Python list to a cuuint32_t array
static cuuint32_t *list_to_cuuint32_array(PyObject *listObj) {
  Py_ssize_t len = PyList_Size(listObj);
  cuuint32_t *array = (cuuint32_t *)malloc(len * sizeof(cuuint32_t));
  for (Py_ssize_t i = 0; i < len; i++) {
    PyObject *item = PyList_GetItem(listObj, i);
    array[i] = (cuuint32_t)PyLong_AsUnsignedLong(item);
  }
  return array;
}

typedef CUresult (*cuTensorMapEncodeTiled_t)(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const cuuint32_t *boxDim,
    const cuuint32_t *elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill);

typedef CUresult (*cuOccupancyMaxActiveClusters_t)(
    int *numClusters, CUfunction func, const CUlaunchConfig *config);

#define defineGetFunctionHandle(name, symbolName)                              \
  static symbolName##_t name() {                                               \
    /* Open the shared library */                                              \
    void *libHandle = dlopen("libcuda.so", RTLD_LAZY);                         \
    if (!libHandle) {                                                          \
      PyErr_SetString(PyExc_RuntimeError, "Failed to open libcuda.so");        \
      return NULL;                                                             \
    }                                                                          \
    /* Clear any existing error */                                             \
    dlerror();                                                                 \
    symbolName##_t funcHandle = (symbolName##_t)dlsym(libHandle, #symbolName); \
    /* Check for errors */                                                     \
    const char *err = dlerror();                                               \
    if (err) {                                                                 \
      PyErr_SetString(PyExc_RuntimeError,                                      \
                      "Failed to retrieve " #symbolName " from libcuda.so");   \
      dlclose(libHandle);                                                      \
      return NULL;                                                             \
    }                                                                          \
    return funcHandle;                                                         \
  }

defineGetFunctionHandle(getCuTensorMapEncodeTiledHandle,
                        cuTensorMapEncodeTiled);
defineGetFunctionHandle(getCuOccupancyMaxActiveClustersHandle,
                        cuOccupancyMaxActiveClusters);

static PyObject *tensorMapEncodeTiled(PyObject *self, PyObject *args) {
  CUtensorMap *tensorMap = (CUtensorMap *)malloc(sizeof(CUtensorMap));
  CUtensorMapDataType tensorDataType;
  cuuint32_t tensorRank;
  void *globalAddress;
  PyObject *globalDimObj, *globalStridesObj, *boxDimObj, *elementStridesObj;
  CUtensorMapInterleave interleave;
  CUtensorMapSwizzle swizzle;
  CUtensorMapL2promotion l2Promotion;
  CUtensorMapFloatOOBfill oobFill;

  // Parse arguments
  if (!PyArg_ParseTuple(args, "iiKO!O!O!O!iiii", &tensorDataType, &tensorRank,
                        &globalAddress, &PyList_Type, &globalDimObj,
                        &PyList_Type, &globalStridesObj, &PyList_Type,
                        &boxDimObj, &PyList_Type, &elementStridesObj,
                        &interleave, &swizzle, &l2Promotion, &oobFill)) {
    return NULL; // Error parsing arguments
  }

  // Convert Python lists to C arrays
  cuuint64_t *globalDim = list_to_cuuint64_array(globalDimObj);
  cuuint64_t *globalStrides = list_to_cuuint64_array(globalStridesObj);
  cuuint32_t *boxDim = list_to_cuuint32_array(boxDimObj);
  cuuint32_t *elementStrides = list_to_cuuint32_array(elementStridesObj);

  static cuTensorMapEncodeTiled_t cuTensorMapEncodeTiledHandle = NULL;
  if (cuTensorMapEncodeTiledHandle == NULL) {
    cuTensorMapEncodeTiledHandle = getCuTensorMapEncodeTiledHandle();
  }
  // Call the function
  Py_BEGIN_ALLOW_THREADS;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuTensorMapEncodeTiledHandle(
      tensorMap, tensorDataType, tensorRank, globalAddress, globalDim,
      globalStrides, boxDim, elementStrides, interleave, swizzle, l2Promotion,
      oobFill));
  Py_END_ALLOW_THREADS;

  // Clean up
  free(globalDim);
  free(globalStrides);
  free(boxDim);
  free(elementStrides);
  // Return the tensor map as a normal pointer
  return PyLong_FromUnsignedLongLong((unsigned long long)tensorMap);
}

static PyObject *occupancyMaxActiveClusters(PyObject *self, PyObject *args) {
  int clusterDimX = -1, clusterDimY = -1, clusterDimZ = -1,
      maxActiveClusters = -1;
  int shared = 0;
  CUfunction func;

  if (!PyArg_ParseTuple(args, "Kiiii", &func, &shared, &clusterDimX,
                        &clusterDimY, &clusterDimZ)) {
    return NULL;
  }

  // Let each SM have one block
  int maxActiveBlocks = 1;
  Py_BEGIN_ALLOW_THREADS;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuFuncSetAttribute(
      func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared));
  Py_END_ALLOW_THREADS;

  CUlaunchAttribute launchAttr[1];
  launchAttr[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  launchAttr[0].value.clusterDim.x = clusterDimX;
  launchAttr[0].value.clusterDim.y = clusterDimY;
  launchAttr[0].value.clusterDim.z = clusterDimZ;
  CUlaunchConfig config;
  config.gridDimX = clusterDimX;
  config.gridDimY = maxActiveBlocks * clusterDimY;
  config.gridDimZ = clusterDimZ;
  config.blockDimX = 128;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = shared;
  config.hStream = 0;
  config.numAttrs = 1;
  config.attrs = launchAttr;

  static cuOccupancyMaxActiveClusters_t cuOccupancyMaxActiveClusters = NULL;
  if (cuOccupancyMaxActiveClusters == NULL) {
    cuOccupancyMaxActiveClusters = getCuOccupancyMaxActiveClustersHandle();
  }

  Py_BEGIN_ALLOW_THREADS;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuFuncSetAttribute(
      func, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      cuOccupancyMaxActiveClusters(&maxActiveClusters, func, &config));
  Py_END_ALLOW_THREADS;
  return PyLong_FromLong(maxActiveClusters);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided cubin into CUDA driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {"cuMemAlloc", memAlloc, METH_VARARGS},
    {"cuMemcpyHtoD", memcpyHtoD, METH_VARARGS},
    {"cuMemFree", memFree, METH_VARARGS},
    {"cuTensorMapEncodeTiled", tensorMapEncodeTiled, METH_VARARGS,
     "Python interface for cuTensorMapEncodeTiled function"},
    {"cuOccupancyMaxActiveClusters", occupancyMaxActiveClusters, METH_VARARGS,
     "Python interface for cuOccupancyMaxActiveClusters function"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "cuda_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_cuda_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }

  defineEnums(m);
  defineTypes(m);
  PyModule_AddFunctions(m, ModuleMethods);

  return m;
}
