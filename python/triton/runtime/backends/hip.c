#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

static bool gpuAssert(hipError_t code, const char *file, int line) {
    if (code == HIP_SUCCESS)
        return true;

    const char *prefix = "Triton Error [HIP]: ";
    const char *str = hipGetErrorString(code);
    char err[1024] = {0};
    snprintf(err, 1024, "%s Code: %d, Messsage: %s", prefix, code, str);
    PyGILState_STATE gil_state;
    gil_state = PyGILState_Ensure();
    PyErr_SetString(PyExc_RuntimeError, err);
    PyGILState_Release(gil_state);
    return false;
}

#define HIP_CHECK(ans)                                                         \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
    if (PyErr_Occurred())                                                      \
      return NULL;                                                             \
  }

#define HIP_CHECK_AND_RETURN_NULL(ans)                                         \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__))                                 \
      return NULL;                                                             \
  } while (0)

#define HIP_CHECK_AND_RETURN_NULL_ALLOW_THREADS(ans)                           \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__)) {                               \
      PyEval_RestoreThread(_save);                                             \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;

  hipDeviceProp_t props;
  HIP_CHECK_AND_RETURN_NULL(hipGetDeviceProperties(&props, device_id));

  // create a struct to hold device properties
  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i}", "max_shared_mem",
                       props.sharedMemPerBlock, "multiprocessor_count",
                       props.multiProcessorCount, "sm_clock_rate",
                       props.clockRate, "mem_clock_rate", props.memoryClockRate,
                       "mem_bus_width", props.memoryBusWidth);
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

  // Open HSACO file
  FILE *hsaco_file;
  if ((hsaco_file = fopen(data, "rb")) == NULL) {
    return NULL;
  }

  // Read HSCAO file into Buffer
  fseek(hsaco_file, 0L, SEEK_END);
  size_t hsaco_file_size = ftell(hsaco_file);
  unsigned char *hsaco =
      (unsigned char *)malloc(hsaco_file_size * sizeof(unsigned char));
  rewind(hsaco_file);
  fread(hsaco, sizeof(unsigned char), hsaco_file_size, hsaco_file);
  fclose(hsaco_file);

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
  int32_t n_regs = 0;
  int32_t n_spills = 0;
  hipModule_t mod;
  hipFunction_t fun;
  Py_BEGIN_ALLOW_THREADS;
  HIP_CHECK_AND_RETURN_NULL_ALLOW_THREADS(hipModuleLoadDataEx(&mod, hsaco, 5, opt, optval));
  HIP_CHECK_AND_RETURN_NULL_ALLOW_THREADS(hipModuleGetFunction(&fun, mod, name));
  HIP_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      hipFuncGetAttribute(&n_regs, HIP_FUNC_ATTRIBUTE_NUM_REGS, fun));
  HIP_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      hipFuncGetAttribute(&n_spills, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
  // somehow ISA dumping seems vgpr_spill_count always need minus 1
  if(n_spills != 0) n_spills = n_spills / 4 - 1;

  Py_END_ALLOW_THREADS;
  free(hsaco);

  // get allocated registers and spilled registers from the function
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
