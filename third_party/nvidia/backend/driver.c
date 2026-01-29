#include "cuda.h"
#include <dlfcn.h>
#include <stdalign.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct {
  PyObject_HEAD;
  _Alignas(128) CUtensorMap tensorMap;
} PyCUtensorMapObject;

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
#define CUDA_CHECK(ans)                                                        \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__))                                 \
      return;                                                                  \
  } while (0)

// To be used only *outside* a Py_{BEGIN,END}_ALLOW_THREADS block.
#define CUDA_CHECK_AND_RETURN_NULL(ans)                                        \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__))                                 \
      goto cleanup;                                                            \
  } while (0)

// To be used inside a Py_{BEGIN,END}_ALLOW_THREADS block.
#define CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(ans)                          \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__)) {                               \
      PyEval_RestoreThread(_save);                                             \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

// Used to check if functions exist in old CUDA driver versions.
#define INITIALIZE_FUNCTION_POINTER_IF_NULL(funcPointer, initializerFunction)  \
  do {                                                                         \
    if ((funcPointer) == NULL) {                                               \
      (funcPointer) = (initializerFunction)();                                 \
      if ((funcPointer) == NULL) {                                             \
        goto cleanup;                                                          \
      }                                                                        \
    }                                                                          \
  } while (0)

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;
  // Get device handle
  CUdevice device;
  cuDeviceGet(&device, device_id);

  // create a struct to hold device properties
  int max_shared_mem;
  int max_num_regs;
  int multiprocessor_count;
  int warp_size;
  int sm_clock_rate;
  int mem_clock_rate;
  int mem_bus_width;
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &max_shared_mem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &max_num_regs, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &multiprocessor_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
  CUDA_CHECK_AND_RETURN_NULL(
      cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &sm_clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &mem_clock_rate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
  CUDA_CHECK_AND_RETURN_NULL(cuDeviceGetAttribute(
      &mem_bus_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));

  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i, s:i, s:i}", "max_shared_mem",
                       max_shared_mem, "max_num_regs", max_num_regs,
                       "multiprocessor_count", multiprocessor_count, "warpSize",
                       warp_size, "sm_clock_rate", sm_clock_rate,
                       "mem_clock_rate", mem_clock_rate, "mem_bus_width",
                       mem_bus_width);

cleanup:
  return NULL;
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
  int32_t n_max_threads = 0;
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
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuFuncGetAttribute(
      &n_max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fun));
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
  return Py_BuildValue("(KKiii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                       n_spills, n_max_threads);
}

typedef CUresult (*cuOccupancyMaxActiveClusters_t)(
    int *numClusters, CUfunction func, const CUlaunchConfig *config);

typedef CUresult (*cuTensorMapEncodeTiled_t)(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const cuuint32_t *boxDim,
    const cuuint32_t *elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill);

typedef CUresult (*cuTensorMapEncodeIm2col_t)(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const int *pixelBoxLowerCorner,
    const int *pixelBoxUpperCorner, cuuint32_t channelsPerPixel,
    cuuint32_t pixelsPerColumn, const cuuint32_t *elementStrides,
    CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill);

typedef CUresult (*cuLaunchKernelEx_t)(const CUlaunchConfig *config,
                                       CUfunction f, void **kernelParams,
                                       void **extra);

#define defineGetFunctionHandle(name, symbolName)                              \
  static symbolName##_t name() {                                               \
    /* Open the shared library */                                              \
    void *libHandle = dlopen("libcuda.so.1", RTLD_LAZY);                       \
    if (!libHandle) {                                                          \
      PyErr_SetString(PyExc_RuntimeError, "Failed to open libcuda.so.1");      \
      return NULL;                                                             \
    }                                                                          \
    /* Clear any existing error */                                             \
    dlerror();                                                                 \
    symbolName##_t funcHandle = (symbolName##_t)dlsym(libHandle, #symbolName); \
    /* Check for errors */                                                     \
    const char *err = dlerror();                                               \
    if (err) {                                                                 \
      PyErr_SetString(PyExc_RuntimeError,                                      \
                      "Failed to retrieve " #symbolName " from libcuda.so.1"); \
      dlclose(libHandle);                                                      \
      return NULL;                                                             \
    }                                                                          \
    return funcHandle;                                                         \
  }

defineGetFunctionHandle(getCuOccupancyMaxActiveClustersHandle,
                        cuOccupancyMaxActiveClusters);

defineGetFunctionHandle(getCuTensorMapEncodeTiledHandle,
                        cuTensorMapEncodeTiled);

defineGetFunctionHandle(getCuTensorMapEncodeIm2colHandle,
                        cuTensorMapEncodeIm2col);

defineGetFunctionHandle(getLaunchKernelExHandle, cuLaunchKernelEx);

static PyObject *occupancyMaxActiveClusters(PyObject *self, PyObject *args) {
  int clusterDim = -1, maxActiveClusters = -1;
  int shared = 0;
  CUfunction func;

  if (!PyArg_ParseTuple(args, "Kii", &func, &shared, &clusterDim)) {
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
  launchAttr[0].value.clusterDim.x = clusterDim;
  launchAttr[0].value.clusterDim.y = 1;
  launchAttr[0].value.clusterDim.z = 1;
  CUlaunchConfig config;
  config.gridDimX = clusterDim * maxActiveBlocks;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 128;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = shared;
  config.hStream = 0;
  config.numAttrs = 1;
  config.attrs = launchAttr;

  static cuOccupancyMaxActiveClusters_t cuOccupancyMaxActiveClusters = NULL;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuOccupancyMaxActiveClusters,
                                      getCuOccupancyMaxActiveClustersHandle);

  Py_BEGIN_ALLOW_THREADS;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuFuncSetAttribute(
      func, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      cuOccupancyMaxActiveClusters(&maxActiveClusters, func, &config));
  Py_END_ALLOW_THREADS;
  return PyLong_FromLong(maxActiveClusters);

cleanup:
  return NULL;
}

static PyObject *setPrintfFifoSize(PyObject *self, PyObject *args) {
  long size;
  if (!PyArg_ParseTuple(args, "l", &size)) {
    return NULL;
  }
  if (size < 0) {
    PyErr_SetString(PyExc_ValueError, "fifo size must be non-negative");
    return NULL;
  }

  Py_BEGIN_ALLOW_THREADS;

  // Ensure we have an active context.
  CUcontext ctx = NULL;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuCtxGetCurrent(&ctx));
  if (!ctx) {
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuDevicePrimaryCtxRetain(&ctx, /*device=*/0));
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuCtxSetCurrent(ctx));
  }

  // We can't set the fifo size after running a kernel that calls printf.  This
  // is true even if the set() call is a nop and the new size is the same as the
  // old size.
  //
  // This is unfriendly, so check if the old size matches the new size, and skip
  // the set() call if so.
  size_t oldSize = 0;
  CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      cuCtxGetLimit(&oldSize, CU_LIMIT_PRINTF_FIFO_SIZE));
  if (oldSize != size) {
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, size));
  }

  Py_END_ALLOW_THREADS;
  Py_RETURN_NONE;
}

static PyObject *PyCUtensorMap_alloc(PyTypeObject *type, Py_ssize_t n_items) {
  PyCUtensorMapObject *self = NULL;
  void *mem = NULL;
  size_t size = type->tp_basicsize;

  if (posix_memalign(&mem, 128, size) != 0) {
    PyErr_NoMemory();
    return NULL;
  }

  self = (PyCUtensorMapObject *)mem;
  PyObject_INIT(self, type);
  return (PyObject *)self;
}

static void PyCUtensorMap_dealloc(PyObject *self) {
  Py_TYPE(self)->tp_free(self);
}

static void PyCUtensorMap_free(void *ptr) { free(ptr); }

// clang-format off
static PyTypeObject PyCUtensorMapType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "triton.backends.nvidia.PyCUtensorMap",
    .tp_basicsize = sizeof(PyCUtensorMapObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "<PyCUtensorMap object>",
    .tp_new = PyType_GenericNew,
    .tp_alloc = PyCUtensorMap_alloc,
    .tp_dealloc = (destructor)PyCUtensorMap_dealloc,
    .tp_free = PyCUtensorMap_free,
};
// clang-format on

static PyObject *fillTMADescriptorTiled(PyObject *self, PyObject *args) {
  unsigned long long global_address;
  int swizzle;
  int elemSize;
  int elemType;
  PyObject *blockSize;
  PyObject *shape;
  PyObject *strides;
  int padding;

  if (!PyArg_ParseTuple(args, "KiiiOOOi", &global_address, &swizzle, &elemSize,
                        &elemType, &blockSize, &shape, &strides, &padding)) {
    return NULL;
  }

  PyCUtensorMapObject *desc = (PyCUtensorMapObject *)PyObject_CallObject(
      (PyObject *)&PyCUtensorMapType, NULL);
  if (!desc) {
    return NULL;
  }

  PyObject *blockSizeFast = NULL;
  PyObject *shapeFast = NULL;
  PyObject *stridesFast = NULL;

  uint32_t blockSizeInt[5];
  uint64_t shapeInt[5];
  uint64_t stridesLL[5];

  blockSizeFast = PySequence_Fast(blockSize, "blockSize must be a sequence");
  if (!blockSizeFast)
    goto cleanup;
  int rank = PySequence_Fast_GET_SIZE(blockSizeFast);

  for (int i = 0; i < rank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(blockSizeFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "block size must be an int");
      goto cleanup;
    }
    blockSizeInt[rank - i - 1] = PyLong_AsLongLong(item);
  }

  shapeFast = PySequence_Fast(shape, "shape must be a sequence");
  if (!shapeFast)
    goto cleanup;

  if (rank != PySequence_Fast_GET_SIZE(shapeFast)) {
    PyErr_SetString(PyExc_RuntimeError, "Rank mismatch");
    goto cleanup;
  }
  for (int i = 0; i < rank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(shapeFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "shape must be an int");
      goto cleanup;
    }
    shapeInt[rank - i - 1] = PyLong_AsLong(item);
  }

  stridesFast = PySequence_Fast(strides, "strides must be a sequence");
  if (!stridesFast)
    goto cleanup;

  if (rank != PySequence_Fast_GET_SIZE(stridesFast)) {
    PyErr_SetString(PyExc_RuntimeError, "Rank mismatch");
    goto cleanup;
  }
  for (int i = 0; i + 1 < rank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(stridesFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "shape must be an int");
      goto cleanup;
    }
    stridesLL[rank - i - 2] = elemSize * PyLong_AsLongLong(item);
  }
  stridesLL[rank - 1] =
      shapeInt[rank - 1] * (rank == 1 ? elemSize : stridesLL[rank - 2]);
  Py_DECREF(blockSizeFast);
  blockSizeFast = NULL;
  Py_DECREF(shapeFast);
  shapeFast = NULL;
  Py_DECREF(stridesFast);
  stridesFast = NULL;

  CUtensorMapFloatOOBfill fill =
      (padding == 1) ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
                     : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  uint32_t elementStrides[5] = {1, 1, 1, 1, 1};
  static cuTensorMapEncodeTiled_t cuTensorMapEncodeTiled = NULL;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuTensorMapEncodeTiled,
                                      getCuTensorMapEncodeTiledHandle);
  CUresult res = cuTensorMapEncodeTiled(
      &desc->tensorMap, elemType, rank, (void *)global_address, shapeInt,
      stridesLL, blockSizeInt, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, fill);
  if (res != CUDA_SUCCESS) {
    const char *str;
    cuGetErrorString(res, &str);
    char err[4096] = {0};
    size_t off = 0;
    off += snprintf(
        err + off, sizeof(err) - off,
        "Triton Error [CUDA]: Failed to create tensor map descriptor: %s\n",
        str ? str : "Unknown error");
    off += snprintf(err + off, sizeof(err) - off,
                    "elemType=%d rank=%d global_address=0x%llx elemSize=%d "
                    "swizzle=%d padding=%d\n",
                    elemType, rank, (unsigned long long)global_address,
                    elemSize, swizzle, padding);
    off += snprintf(err + off, sizeof(err) - off, "shape=[");
    for (int i = 0; i < rank; ++i) {
      off +=
          snprintf(err + off, sizeof(err) - off, "%llu%s",
                   (unsigned long long)shapeInt[i], (i + 1 < rank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "]\n");
    off += snprintf(err + off, sizeof(err) - off, "strides=[");
    for (int i = 0; i + 1 < rank; ++i) {
      off += snprintf(err + off, sizeof(err) - off, "%llu%s",
                      (unsigned long long)stridesLL[i],
                      (i + 2 < rank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "]\n");
    off += snprintf(err + off, sizeof(err) - off, "blockSize=[");
    for (int i = 0; i < rank; ++i) {
      off += snprintf(err + off, sizeof(err) - off, "%u%s",
                      (unsigned)blockSizeInt[i], (i + 1 < rank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "] elementStrides=[");
    for (int i = 0; i < rank; ++i) {
      off += snprintf(err + off, sizeof(err) - off, "%u%s",
                      (unsigned)elementStrides[i], (i + 1 < rank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "] fill=%d\n", (int)fill);
    PyErr_SetString(PyExc_RuntimeError, err);

    goto cleanup;
  }

  // Follow the CUTLASS change for the driver version check
  // https://github.com/NVIDIA/cutlass/commit/b7ecaa605dd70326900433695e11ebfec407edd2#diff-1dfcaf77b33258ff3175540718d9caff1cd471215f741ba42943ef00770e6d04
  int driver_version = 0;
  CUresult driver_version_result = cuDriverGetVersion(&driver_version);
  assert(driver_version_result == CUDA_SUCCESS);

  if (driver_version <= 13010) {
    int max_byte_index = 0;
    for (int i = 0; i < rank; ++i) {
      int bytes_stride = i == 0 ? elemSize : stridesLL[i - 1];
      max_byte_index += (shapeInt[i] - 1) * bytes_stride;
    }
    if (max_byte_index + 1 < 128 * 1024) {
      uint64_t *desc_u64 = (uint64_t *)&desc->tensorMap;
      desc_u64[1] &= ~(1llu << 21);
    }
  }

  return (PyObject *)desc;

cleanup:
  Py_XDECREF(blockSizeFast);
  Py_XDECREF(shapeFast);
  Py_XDECREF(stridesFast);
  Py_XDECREF(desc);
  return NULL;
}

static PyObject *fillTMADescriptorIm2col(PyObject *self, PyObject *args) {
  unsigned long long global_address;
  int swizzle;
  int elemSize;
  int elemType;
  PyObject *blockSize;
  PyObject *shape;
  PyObject *strides;
  int padding;
  PyObject *pixelBoxLower;
  PyObject *pixelBoxUpper;
  int channelsPerPixel;
  int pixelsPerColumn;
  PyObject *elementStrides;

  if (!PyArg_ParseTuple(args, "KiiiOOOiOOiiO", &global_address, &swizzle,
                        &elemSize, &elemType, &blockSize, &shape, &strides,
                        &padding, &pixelBoxLower, &pixelBoxUpper,
                        &channelsPerPixel, &pixelsPerColumn, &elementStrides)) {
    return NULL;
  }

  PyCUtensorMapObject *desc = (PyCUtensorMapObject *)PyObject_CallObject(
      (PyObject *)&PyCUtensorMapType, NULL);
  if (!desc) {
    return NULL;
  }

  PyObject *blockSizeFast = NULL;
  PyObject *shapeFast = NULL;
  PyObject *stridesFast = NULL;
  PyObject *pixelBoxLowerFast = NULL;
  PyObject *pixelBoxUpperFast = NULL;
  PyObject *elementStridesFast = NULL;

  uint32_t blockSizeInt[5];
  uint64_t shapeInt[5];
  uint64_t stridesLL[5];
  int pixelBoxLowerInt[5] = {0};
  int pixelBoxUpperInt[5] = {0};
  uint32_t elementStridesInt[5] = {1, 1, 1, 1, 1}; // Default to all 1s

  // For im2col mode, shape determines the tensor rank, not blockSize
  // blockSize is typically 2D [pixelsPerColumn, channelsPerPixel]
  // while shape can be 4D or 5D (e.g., NHWC or NDHWC)
  shapeFast = PySequence_Fast(shape, "shape must be a sequence");
  if (!shapeFast)
    goto cleanup;
  int rank = PySequence_Fast_GET_SIZE(shapeFast);

  for (int i = 0; i < rank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(shapeFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "shape must be an int");
      goto cleanup;
    }
    shapeInt[rank - i - 1] = PyLong_AsLong(item);
  }

  blockSizeFast = PySequence_Fast(blockSize, "blockSize must be a sequence");
  if (!blockSizeFast)
    goto cleanup;
  int blockRank = PySequence_Fast_GET_SIZE(blockSizeFast);

  for (int i = 0; i < blockRank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(blockSizeFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "block size must be an int");
      goto cleanup;
    }
    blockSizeInt[blockRank - i - 1] = PyLong_AsLongLong(item);
  }

  stridesFast = PySequence_Fast(strides, "strides must be a sequence");
  if (!stridesFast)
    goto cleanup;

  if (rank != PySequence_Fast_GET_SIZE(stridesFast)) {
    PyErr_Format(PyExc_RuntimeError,
                 "Rank mismatch for strides in fillTMADescriptorIm2col: shape "
                 "has rank %d but strides has %zd elements. "
                 "Expected strides to have %d elements.",
                 rank, PySequence_Fast_GET_SIZE(stridesFast), rank);
    goto cleanup;
  }
  for (int i = 0; i + 1 < rank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(stridesFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "strides must be an int");
      goto cleanup;
    }
    stridesLL[rank - i - 2] = elemSize * PyLong_AsLongLong(item);
  }
  stridesLL[rank - 1] =
      shapeInt[rank - 1] * (rank == 1 ? elemSize : stridesLL[rank - 2]);

  // Parse pixel box lower corner
  pixelBoxLowerFast =
      PySequence_Fast(pixelBoxLower, "pixelBoxLower must be a sequence");
  if (!pixelBoxLowerFast)
    goto cleanup;

  int spatialRank = PySequence_Fast_GET_SIZE(pixelBoxLowerFast);
  if (spatialRank > 5) {
    PyErr_SetString(PyExc_RuntimeError, "Pixel box rank too large (max 5)");
    goto cleanup;
  }

  for (int i = 0; i < spatialRank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(pixelBoxLowerFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "pixelBoxLower elements must be int");
      goto cleanup;
    }
    pixelBoxLowerInt[spatialRank - i - 1] = PyLong_AsLong(item);
  }

  // Parse pixel box upper corner
  pixelBoxUpperFast =
      PySequence_Fast(pixelBoxUpper, "pixelBoxUpper must be a sequence");
  if (!pixelBoxUpperFast)
    goto cleanup;

  if (spatialRank != PySequence_Fast_GET_SIZE(pixelBoxUpperFast)) {
    PyErr_SetString(PyExc_RuntimeError, "Pixel box corner rank mismatch");
    goto cleanup;
  }

  for (int i = 0; i < spatialRank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(pixelBoxUpperFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "pixelBoxUpper elements must be int");
      goto cleanup;
    }
    pixelBoxUpperInt[spatialRank - i - 1] = PyLong_AsLong(item);
  }

  // Parse element strides
  elementStridesFast =
      PySequence_Fast(elementStrides, "elementStrides must be a sequence");
  if (!elementStridesFast)
    goto cleanup;

  int elementStridesLen = PySequence_Fast_GET_SIZE(elementStridesFast);
  if (elementStridesLen != rank) {
    PyErr_SetString(PyExc_RuntimeError,
                    "elementStrides length must match tensor rank");
    goto cleanup;
  }

  for (int i = 0; i < rank; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(elementStridesFast, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "elementStrides elements must be int");
      goto cleanup;
    }
    elementStridesInt[rank - i - 1] = PyLong_AsLong(item);
  }

  Py_DECREF(blockSizeFast);
  blockSizeFast = NULL;
  Py_DECREF(shapeFast);
  shapeFast = NULL;
  Py_DECREF(stridesFast);
  stridesFast = NULL;
  Py_DECREF(pixelBoxLowerFast);
  pixelBoxLowerFast = NULL;
  Py_DECREF(pixelBoxUpperFast);
  pixelBoxUpperFast = NULL;
  Py_DECREF(elementStridesFast);
  elementStridesFast = NULL;

  CUtensorMapFloatOOBfill fill =
      (padding == 1) ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
                     : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  static cuTensorMapEncodeIm2col_t cuTensorMapEncodeIm2col = NULL;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(cuTensorMapEncodeIm2col,
                                      getCuTensorMapEncodeIm2colHandle);

  CUresult res = cuTensorMapEncodeIm2col(
      &desc->tensorMap, elemType, rank, (void *)global_address, shapeInt,
      stridesLL, pixelBoxLowerInt, pixelBoxUpperInt, channelsPerPixel,
      pixelsPerColumn, elementStridesInt, CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, fill);

  if (res != CUDA_SUCCESS) {
    const char *str;
    cuGetErrorString(res, &str);
    char err[4096] = {0};
    size_t off = 0;
    off += snprintf(err + off, sizeof(err) - off,
                    "Triton Error [CUDA]: Failed to create im2col tensor map "
                    "descriptor: %s\n",
                    str ? str : "Unknown error");
    off +=
        snprintf(err + off, sizeof(err) - off,
                 "elemType=%d rank=%d global_address=0x%llx elemSize=%d "
                 "swizzle=%d padding=%d channelsPerPixel=%d "
                 "pixelsPerColumn=%d\n",
                 elemType, rank, (unsigned long long)global_address, elemSize,
                 swizzle, padding, channelsPerPixel, pixelsPerColumn);
    off += snprintf(err + off, sizeof(err) - off, "shape=[");
    for (int i = 0; i < rank; ++i) {
      off +=
          snprintf(err + off, sizeof(err) - off, "%llu%s",
                   (unsigned long long)shapeInt[i], (i + 1 < rank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "]\n");
    off += snprintf(err + off, sizeof(err) - off, "strides=[");
    for (int i = 0; i < rank; ++i) {
      off += snprintf(err + off, sizeof(err) - off, "%llu%s",
                      (unsigned long long)stridesLL[i],
                      (i + 1 < rank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "]\n");
    off += snprintf(err + off, sizeof(err) - off, "blockSize=[");
    for (int i = 0; i < blockRank; ++i) {
      off +=
          snprintf(err + off, sizeof(err) - off, "%u%s",
                   (unsigned)blockSizeInt[i], (i + 1 < blockRank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "]\n");
    off += snprintf(err + off, sizeof(err) - off, "pixelBoxLower=[");
    for (int i = 0; i < spatialRank; ++i) {
      off += snprintf(err + off, sizeof(err) - off, "%d%s", pixelBoxLowerInt[i],
                      (i + 1 < spatialRank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "] pixelBoxUpper=[");
    for (int i = 0; i < spatialRank; ++i) {
      off += snprintf(err + off, sizeof(err) - off, "%d%s", pixelBoxUpperInt[i],
                      (i + 1 < spatialRank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "]\n");
    off += snprintf(err + off, sizeof(err) - off, "elementStrides=[");
    for (int i = 0; i < rank; ++i) {
      off +=
          snprintf(err + off, sizeof(err) - off, "%u%s",
                   (unsigned)elementStridesInt[i], (i + 1 < rank) ? ", " : "");
    }
    off += snprintf(err + off, sizeof(err) - off, "]\n");
    PyErr_SetString(PyExc_RuntimeError, err);

    goto cleanup;
  }

  return (PyObject *)desc;

cleanup:
  Py_XDECREF(blockSizeFast);
  Py_XDECREF(shapeFast);
  Py_XDECREF(stridesFast);
  Py_XDECREF(pixelBoxLowerFast);
  Py_XDECREF(pixelBoxUpperFast);
  Py_XDECREF(elementStridesFast);
  Py_XDECREF(desc);
  return NULL;
}

static void ensureCudaContext() {
  CUcontext pctx;
  CUDA_CHECK(cuCtxGetCurrent(&pctx));
  if (!pctx) {
    // Ensure device context.
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&pctx, device));
    CUDA_CHECK(cuCtxSetCurrent(pctx));
  }
}

static void _launch(int gridX, int gridY, int gridZ, int num_warps,
                    int num_ctas, int launch_cooperative_grid, int launch_pdl,
                    int shared_memory, CUstream stream, CUfunction function,
                    void **params) {
  if (gridX * gridY * gridZ > 0) {
    // 4 attributes that we can currently pass maximum
    CUlaunchAttribute launchAttr[4];
    static cuLaunchKernelEx_t cuLaunchKernelExHandle = NULL;
    if (cuLaunchKernelExHandle == NULL) {
      cuLaunchKernelExHandle = getLaunchKernelExHandle();
    }
    CUlaunchConfig config;
    config.gridDimX = gridX * num_ctas;
    config.gridDimY = gridY;
    config.gridDimZ = gridZ;

    config.blockDimX = 32 * num_warps;
    config.blockDimY = 1;
    config.blockDimZ = 1;
    config.sharedMemBytes = shared_memory;
    config.hStream = stream;
    config.attrs = launchAttr;
    int num_attrs = 0;

    if (launch_pdl != 0) {
      CUlaunchAttribute pdlAttr = {
          .id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
          .value = 1};
      launchAttr[num_attrs] = pdlAttr;
      ++num_attrs;
    }

    if (launch_cooperative_grid != 0) {
      CUlaunchAttribute coopAttr = {.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE,
                                    .value = 1};
      launchAttr[num_attrs] = coopAttr;
      ++num_attrs;
    }

    if (num_ctas != 1) {
      CUlaunchAttribute clusterAttr = {};
      clusterAttr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      clusterAttr.value.clusterDim.x = num_ctas;
      clusterAttr.value.clusterDim.y = 1;
      clusterAttr.value.clusterDim.z = 1;
      launchAttr[num_attrs] = clusterAttr;
      ++num_attrs;

      CUlaunchAttribute clusterSchedulingAttr = {};
      clusterSchedulingAttr.id =
          CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      clusterSchedulingAttr.value.clusterSchedulingPolicyPreference =
          CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
      launchAttr[num_attrs] = clusterSchedulingAttr;
      ++num_attrs;
    }

    // num_ctas == 16 is non-portable. Does work for H100 and B200 tho
    config.numAttrs = num_attrs;
    if (num_ctas == 16) {
      CUDA_CHECK(cuFuncSetAttribute(
          function, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
    }

    CUDA_CHECK(cuLaunchKernelExHandle(&config, function, params, 0));
  }
}

static PyObject *data_ptr_str = NULL;

// Extract a CUDA device pointer from a pointer-like PyObject obj, and store
// it to the memory location pointed by ptr.
bool extractPointer(void *ptr, PyObject *obj) {
  CUdeviceptr *dev_ptr = ptr;
  if (obj == Py_None) {
    *dev_ptr = (CUdeviceptr)0; // valid nullptr
    return true;
  }
  if (PyLong_Check(obj)) {
    *dev_ptr = PyLong_AsUnsignedLongLong(obj);
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
  *dev_ptr = PyLong_AsUnsignedLongLong(ret);
  Py_DECREF(ret);
  if (*dev_ptr == 0) {
    return true; // valid nullptr
  }
  CUresult status = cuPointerGetAttribute(
      dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, *dev_ptr);
  if (status == CUDA_ERROR_INVALID_VALUE) {
    PyErr_Format(PyExc_ValueError,
                 "Pointer argument cannot be accessed from Triton "
                 "(cpu tensor?)");
    return false;
  }
  return gpuAssert(status, __FILE__, __LINE__);
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

// Extract a CUtensorMap descriptor from a python object, and store it to the
// memory location pointed by ptr.
bool extractTmaDesc(void *ptr, PyObject *obj) {
  if (sizeof(CUtensorMap *) != 8) {
    PyErr_SetString(PyExc_SystemError,
                    "getTmaDesc() requires 64-bit compilation");
    return false;
  }
  if (Py_TYPE(obj) != &PyCUtensorMapType) {
    PyErr_Format(PyExc_TypeError,
                 "object must be of type PyCUtensorMap, got %s",
                 Py_TYPE(obj)->tp_name);
    return false;
  }
  *((CUtensorMap *)ptr) = ((PyCUtensorMapObject *)obj)->tensorMap;
  uintptr_t align_128 = (uintptr_t)ptr & (128 - 1);
  if (align_128 != 0) {
    PyErr_Format(
        PyExc_ValueError,
        "CUtensorMap must be aligned to 128B, but got (&map) mod 128 = %ld",
        align_128);
    return false;
  }
  return true;
}

typedef bool (*ExtractorFunc)(void *ptr, PyObject *obj);

#define MAX_NAMES_PER_EXTRACTOR 2

typedef struct {
  ExtractorFunc extract;
  size_t size;
  size_t alignment;
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
  EXTRACTOR_NVTMADESC_INDEX = 14,
  // last entry to have a count
  EXTRACTOR_TYPE_COUNT
} ExtractorTypeIndex;

Extractor extraction_map[EXTRACTOR_TYPE_COUNT] = {
    [EXTRACTOR_UNKOWN_INDEX] =
        (Extractor){.extract = NULL, .size = 0, .name = NULL},
    [EXTRACTOR_POINTER_INDEX] = (Extractor){.extract = extractPointer,
                                            .size = sizeof(CUdeviceptr),
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
    [EXTRACTOR_NVTMADESC_INDEX] = (Extractor){.extract = extractTmaDesc,
                                              .size = sizeof(CUtensorMap),
                                              .alignment = alignof(CUtensorMap),
                                              .name = {"nvTmaDesc"}},
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

// Takes in a list of types (ex: ['*fp32', 'u8', 'nvTmaDesc']) and returns
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
  // ensure cuda context is valid before calling any CUDA APIs, e.g. before
  // calls to cuPointerGetAttributes
  ensureCudaContext();

  // Parse the arguments.
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int launch_cooperative_grid;
  int launch_pdl;
  int num_warps, num_ctas, shared_memory;
  PyObject *launch_metadata = NULL;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *global_scratch_obj = NULL;
  PyObject *profile_scratch_obj = NULL;
  PyObject *arg_annotations = NULL;
  Py_buffer signature;
  PyObject *kernel_args = NULL;
  if (!PyArg_ParseTuple(args, "iiiKKpp(iii)OOOOOOy*O", &gridX, &gridY, &gridZ,
                        &_stream, &_function, &launch_cooperative_grid,
                        &launch_pdl, &num_warps, &num_ctas, &shared_memory,
                        &launch_metadata, &launch_enter_hook, &launch_exit_hook,
                        &global_scratch_obj, &profile_scratch_obj,
                        &arg_annotations, &signature, &kernel_args)) {
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

    size_t alignment = extractor.alignment;
    if (alignment != 0) {
      // Allocate enough space on the stack to guarantee an aligned block.
      size_t size_with_alignment = extractor.size + alignment - 1;
      void *storage_ptr = alloca(size_with_alignment);
      void *aligned_ptr = (void *)((((uintptr_t)storage_ptr) + alignment - 1) &
                                   ~(alignment - 1));
      if (aligned_ptr == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to align parameter storage");
        goto cleanup;
      }
      params[params_idx] = aligned_ptr;
    } else {
      params[params_idx] = alloca(extractor.size);
    }

    PyObject *current_arg = args_data[i];
    if (!extractor.extract(params[params_idx++], current_arg)) {
      goto cleanup;
    }
  }
  // Add scratch objects.
  params[params_idx] = alloca(sizeof(void *));
  if (!extractPointer(params[params_idx++], global_scratch_obj)) {
    goto cleanup;
  }
  params[params_idx] = alloca(sizeof(void *));
  if (!extractPointer(params[params_idx++], profile_scratch_obj)) {
    goto cleanup;
  }

  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, launch_cooperative_grid,
          launch_pdl, shared_memory, (CUstream)_stream, (CUfunction)_function,
          params);
  Py_END_ALLOW_THREADS;
  if (PyErr_Occurred()) {
    goto cleanup;
  }

  if (!launchHook(launch_exit_hook, launch_metadata)) {
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
     "Load provided cubin into CUDA driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {"cuOccupancyMaxActiveClusters", occupancyMaxActiveClusters, METH_VARARGS,
     "Python interface for cuOccupancyMaxActiveClusters function"},
    {"set_printf_fifo_size", setPrintfFifoSize, METH_VARARGS,
     "Python interface for cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, x), which "
     "controls how many bytes can be streamed from kernels before data starts "
     "being dropped.  This inherits all the limitations of this call; in "
     "particular it's an error to change this value after launching any kernel "
     "that calls printf()."},
    {"fill_tma_descriptor_tiled", fillTMADescriptorTiled, METH_VARARGS,
     "Create TMA descriptor for tiled mode"},
    {"fill_tma_descriptor_im2col", fillTMADescriptorIm2col, METH_VARARGS,
     "Create TMA descriptor for im2col mode"},
    {"build_signature_metadata", buildSignatureMetadata, METH_VARARGS,
     "Calling it with a signature list (ex: ['*fp32', 'u8', 'nvTmaDesc']), "
     "will return metadata to be passed into 'launch' for quicker "
     "argument parsing."},
    {"launch", launchKernel, METH_VARARGS, "launches cuda kernel"},

    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "cuda_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_cuda_utils(void) {
  if (PyType_Ready(&PyCUtensorMapType) < 0) {
    return NULL;
  }
  if (PyType_Ready(&PyKernelArgType) < 0) {
    return NULL;
  }

  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  data_ptr_str = PyUnicode_InternFromString("data_ptr");
  if (data_ptr_str == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, ModuleMethods);

  Py_INCREF(&PyCUtensorMapType);
  PyModule_AddObject(m, "PyCUtensorMap", (PyObject *)&PyCUtensorMapType);

  Py_INCREF(&PyKernelArgType);
  PyModule_AddObject(m, "PyKernelArg", (PyObject *)&PyKernelArgType);
  PyModule_AddIntConstant(m, "ARG_CONSTEXPR", ARG_CONSTEXPR);
  PyModule_AddIntConstant(m, "ARG_KERNEL", ARG_KERNEL);
  PyModule_AddIntConstant(m, "ARG_TUPLE", ARG_TUPLE);

  return m;
}
