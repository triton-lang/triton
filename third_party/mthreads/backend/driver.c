#include "musa.h"
#include <dlfcn.h>
#include <stdbool.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Raises a Python exception and returns false if code is not MUSA_SUCCESS.
static bool gpuAssert(MUresult code, const char *file, int line) {
  if (code == MUSA_SUCCESS)
    return true;

  const char *prefix = "Triton Error [MUSA]: ";
  const char *str;
  muGetErrorString(code, &str);
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
#define MUSA_CHECK_AND_RETURN_NULL(ans)                                        \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__))                                 \
      return NULL;                                                             \
  } while (0)

// To be used inside a Py_{BEGIN,END}_ALLOW_THREADS block.
#define MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(ans)                          \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__)) {                               \
      PyEval_RestoreThread(_save);                                             \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

// Used to check if functions exist in old MUSA driver versions.
#define INITIALIZE_FUNCTION_POINTER_IF_NULL(funcPointer, initializerFunction)  \
  do {                                                                         \
    if ((funcPointer) == NULL) {                                               \
      (funcPointer) = (initializerFunction)();                                 \
      if ((funcPointer) == NULL) {                                             \
        return NULL;                                                           \
      }                                                                        \
    }                                                                          \
  } while (0)

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;
  // Get device handle
  MUdevice device;
  muDeviceGet(&device, device_id);

  // create a struct to hold device properties
  int max_shared_mem;
  int max_num_regs;
  int multiprocessor_count;
  int warp_size;
  int sm_clock_rate;
  int mem_clock_rate;
  int mem_bus_width;
  MUSA_CHECK_AND_RETURN_NULL(muDeviceGetAttribute(
      &max_shared_mem, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  MUSA_CHECK_AND_RETURN_NULL(muDeviceGetAttribute(
      &max_num_regs, MU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device));
  MUSA_CHECK_AND_RETURN_NULL(muDeviceGetAttribute(
      &multiprocessor_count, MU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
  MUSA_CHECK_AND_RETURN_NULL(
      muDeviceGetAttribute(&warp_size, MU_DEVICE_ATTRIBUTE_WARP_SIZE, device));
  MUSA_CHECK_AND_RETURN_NULL(muDeviceGetAttribute(
      &sm_clock_rate, MU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
  MUSA_CHECK_AND_RETURN_NULL(muDeviceGetAttribute(
      &mem_clock_rate, MU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
  MUSA_CHECK_AND_RETURN_NULL(muDeviceGetAttribute(
      &mem_bus_width, MU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));

  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i, s:i, s:i}", "max_shared_mem",
                       max_shared_mem, "max_num_regs", max_num_regs,
                       "multiprocessor_count", multiprocessor_count, "warpSize",
                       warp_size, "sm_clock_rate", sm_clock_rate,
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
  MUfunction fun;
  MUmodule mod;
  int32_t n_regs = 0;
  int32_t n_spills = 0;
  // create driver handles
  MUcontext pctx = 0;

  Py_BEGIN_ALLOW_THREADS;
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muCtxGetCurrent(&pctx));
  if (!pctx) {
    MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        muDevicePrimaryCtxRetain(&pctx, device));
    MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muCtxSetCurrent(pctx));
  }

  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muModuleLoadData(&mod, data));
  // MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muModuleLoad(&mod, data));
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      muModuleGetFunction(&fun, mod, name));
  // get allocated registers and spilled registers from the function
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      muFuncGetAttribute(&n_regs, MU_FUNC_ATTRIBUTE_NUM_REGS, fun));
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      muFuncGetAttribute(&n_spills, MU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
  n_spills /= 4;
  // set dynamic shared memory if necessary
  int shared_optin;
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muDeviceGetAttribute(
      &shared_optin, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  // supported based on QY2, PH1 is ok here
  if (shared > 73728 && shared_optin > 73728) {
    MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        muFuncSetCacheConfig(fun, MU_FUNC_CACHE_PREFER_SHARED));
    int shared_total, shared_static;
    MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muDeviceGetAttribute(
        &shared_total, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        device));
    MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muFuncGetAttribute(
        &shared_static, MU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fun));
    MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        muFuncSetAttribute(fun, MU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_optin - shared_static));
  }
  Py_END_ALLOW_THREADS;

  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("(KKii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                       n_spills);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided mubin into MUSA driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "musa_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_musa_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, ModuleMethods);

  return m;
}
