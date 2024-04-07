#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

static inline void gpuAssert(hipError_t code, const char *file, int line) {
  {
    if (code != HIP_SUCCESS) {
      {
        const char *prefix = "Triton Error [HIP]: ";
        const char *str = hipGetErrorString(code);
        char err[1024] = {0};
        snprintf(err, 1024, "%s Code: %d, Messsage: %s", prefix, code, str);
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
  HIP_CHECK(hipGetDeviceProperties(&props, device_id));

  // create a struct to hold device properties
  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i, s:s, s:i}", "max_shared_mem",
                       props.sharedMemPerBlock, "multiprocessor_count",
                       props.multiProcessorCount, "sm_clock_rate",
                       props.clockRate, "mem_clock_rate", props.memoryClockRate,
                       "mem_bus_width", props.memoryBusWidth, "arch",
                       props.gcnArchName, "warpSize", props.warpSize);
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
  HIP_CHECK(hipModuleLoadDataEx(&mod, data, 5, opt, optval))
  HIP_CHECK(hipModuleGetFunction(&fun, mod, name));

  // get allocated registers and spilled registers from the function
  int n_regs = 0;
  int n_spills = 0;
  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("(KKii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                       n_spills);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided hsaco into HIP driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "hip_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_hip_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
