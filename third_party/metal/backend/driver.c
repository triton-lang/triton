#include <Metal/Metal.h>
#include <dlfcn.h>
#include <stdbool.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Raises a Python exception and returns false.
static bool metalAssert(const char *err, const char *file, int line) {
  PyGILState_STATE gil_state;
  gil_state = PyGILState_Ensure();
  PyErr_SetString(PyExc_RuntimeError, err);
  PyGILState_Release(gil_state);
  return false;
}

#define METAL_CHECK_AND_RETURN_NULL(err) \
  do { \
    if (err != NULL) { \
      metalAssert([err localizedDescription].UTF8String, __FILE__, __LINE__); \
      return NULL; \
    } \
  } while (0)

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;

  NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
  if (device_id >= [devices count]) {
    PyErr_SetString(PyExc_ValueError, "Invalid device ID");
    return NULL;
  }
  id<MTLDevice> device = devices[device_id];

  return Py_BuildValue("{s:s, s:i, s:i, s:i}", "name", [device.name UTF8String],
                       "max_buffer_length", (long)device.maxBufferLength,
                       "max_threads_per_group", (long)device.maxThreadsPerThreadgroup.width * device.maxThreadsPerThreadgroup.height * device.maxThreadsPerThreadgroup.depth,
                       "max_shared_memory_per_group", (long)device.maxThreadgroupMemoryLength);
}

static PyMethodDef ModuleMethods[] = {
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "metal_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_metal_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, ModuleMethods);

  return m;
}
