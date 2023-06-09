#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  // create a struct to hold device properties
  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i}", "max_shared_mem", 1024,
                       "multiprocessor_count", 16, "sm_clock_rate", 2100,
                       "mem_clock_rate", 2300, "mem_bus_width", 2400);
}

static PyObject *loadBinary(PyObject *self, PyObject *args) {
  // get allocated registers and spilled registers from the function
  int n_regs = 0;
  int n_spills = 0;
  int mod = 0;
  int fun = 0;
  return Py_BuildValue("(KKii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                       n_spills);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load dummy binary for the extension device"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for the extension device"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "ext_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_ext_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
