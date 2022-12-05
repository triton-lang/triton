#include "cuda.h"
#include <Python.h>

union ArgUnion {
  PyObject *O;
  float f;
  double d;
  long l;
  uint32_t I;
  int32_t i;
  uint64_t K;
  int64_t L;
  CUdeviceptr CUdptr;
};

static inline void gpuAssert(CUresult code, const char *file, int line) {
  if (code != CUDA_SUCCESS) {
    const char *prefix = "Triton Error [CUDA]: ";
    const char *str;
    cuGetErrorString(code, &str);
    char err[1024] = {0};
    strcat(err, prefix);
    strcat(err, str);
    PyErr_SetString(PyExc_RuntimeError, err);
  }
}
#define CUDA_CHECK(ans)                                                        \
  { gpuAssert((ans), __FILE__, __LINE__); }

void my_launch(int gridX, int gridY, int gridZ, int num_warps,
               int shared_memory, CUstream stream, CUfunction function,
               void **params) {
  if (gridX * gridY * gridZ > 0) {
    CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, 32 * num_warps, 1,
                              1, shared_memory, stream, params, 0));
  }
}

static inline CUdeviceptr getPointer(PyObject *obj, int idx) {
  if (PyLong_Check(obj)) {
    return (CUdeviceptr)PyLong_AsUnsignedLongLong(obj);
  }
  if (obj == Py_None) {
    return (CUdeviceptr)0;
  }
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if (ptr) {
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {
      PyErr_SetString(
          PyExc_TypeError,
          "data_ptr method of Pointer object must return 64-bit int");
    }
    return (CUdeviceptr)PyLong_AsUnsignedLongLong(ret);
  }
  PyErr_SetString(
      PyExc_TypeError,
      "Pointer argument must be either uint64 or have data_ptr method");
  return (CUdeviceptr)0;
}

static PyObject *launch(PyObject *self, PyObject *args) {
  char *format = (char *)PyUnicode_AsUTF8(PyTuple_GetItem(args, 0));
  char *is_const = (char *)PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int num_warps;
  int shared_memory;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *compiled_kernel = NULL;
  PyObject *hook_ret = NULL;

  int ArgC = strlen(format) - strlen("ssiiiiiKKOOO");
  union ArgUnion MyArgs[100];

  if (0)
    printf("shal1t7 ArgC = %d\n", ArgC);

  if (0)
    printf("shal1t7 format = %s\n", format);

  if (0)
    printf("shal1t7  is_const = %s\n", is_const);

  char *arg0;
  char *arg1;

  if (ArgC == 1 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0])) {
    return NULL;
  }

  if (ArgC == 2 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0], &MyArgs[1])) {
    return NULL;
  }

  if (ArgC == 3 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0], &MyArgs[1], &MyArgs[2])) {
    return NULL;
  }

  if (ArgC == 4 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0], &MyArgs[1], &MyArgs[2], &MyArgs[3])) {
    return NULL;
  }

  if (ArgC == 5 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0], &MyArgs[1], &MyArgs[2], &MyArgs[3],
                        &MyArgs[4])) {
    return NULL;
  }

  if (ArgC == 6 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0], &MyArgs[1], &MyArgs[2], &MyArgs[3],
                        &MyArgs[4], &MyArgs[5])) {
    return NULL;
  }

  if (ArgC == 7 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0], &MyArgs[1], &MyArgs[2], &MyArgs[3],
                        &MyArgs[4], &MyArgs[5], &MyArgs[6])) {
    return NULL;
  }

  if (ArgC == 8 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0], &MyArgs[1], &MyArgs[2], &MyArgs[3],
                        &MyArgs[4], &MyArgs[5], &MyArgs[6], &MyArgs[7])) {
    return NULL;
  }

  if (ArgC == 9 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0], &MyArgs[1], &MyArgs[2], &MyArgs[3],
                        &MyArgs[4], &MyArgs[5], &MyArgs[6], &MyArgs[7],
                        &MyArgs[8])) {
    return NULL;
  }

  if (ArgC == 10 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0], &MyArgs[1], &MyArgs[2], &MyArgs[3],
                        &MyArgs[4], &MyArgs[5], &MyArgs[6], &MyArgs[7],
                        &MyArgs[8], &MyArgs[9])) {
    return NULL;
  }

  if (ArgC == 11 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0], &MyArgs[1], &MyArgs[2], &MyArgs[3],
                        &MyArgs[4], &MyArgs[5], &MyArgs[6], &MyArgs[7],
                        &MyArgs[8], &MyArgs[9], &MyArgs[10])) {
    return NULL;
  }

  if (ArgC == 12 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0], &MyArgs[1], &MyArgs[2], &MyArgs[3],
                        &MyArgs[4], &MyArgs[5], &MyArgs[6], &MyArgs[7],
                        &MyArgs[8], &MyArgs[9], &MyArgs[10], &MyArgs[11])) {
    return NULL;
  }

  if (ArgC == 13 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0], &MyArgs[1], &MyArgs[2], &MyArgs[3],
                        &MyArgs[4], &MyArgs[5], &MyArgs[6], &MyArgs[7],
                        &MyArgs[8], &MyArgs[9], &MyArgs[10], &MyArgs[11],
                        &MyArgs[12])) {
    return NULL;
  }

  if (ArgC == 14 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0], &MyArgs[1], &MyArgs[2], &MyArgs[3],
                        &MyArgs[4], &MyArgs[5], &MyArgs[6], &MyArgs[7],
                        &MyArgs[8], &MyArgs[9], &MyArgs[10], &MyArgs[11],
                        &MyArgs[12], &MyArgs[13])) {
    return NULL;
  }

  if (ArgC == 15 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0], &MyArgs[1], &MyArgs[2], &MyArgs[3],
                        &MyArgs[4], &MyArgs[5], &MyArgs[6], &MyArgs[7],
                        &MyArgs[8], &MyArgs[9], &MyArgs[10], &MyArgs[11],
                        &MyArgs[12], &MyArgs[13], &MyArgs[14])) {
    return NULL;
  }

  if (ArgC == 16 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0], &MyArgs[1], &MyArgs[2], &MyArgs[3],
                        &MyArgs[4], &MyArgs[5], &MyArgs[6], &MyArgs[7],
                        &MyArgs[8], &MyArgs[9], &MyArgs[10], &MyArgs[11],
                        &MyArgs[12], &MyArgs[13], &MyArgs[14], &MyArgs[15])) {
    return NULL;
  }

  if (ArgC == 17 &&
      !PyArg_ParseTuple(
          args, format, &arg0, &arg1, &gridX, &gridY, &gridZ, &num_warps,
          &shared_memory, &_stream, &_function, &launch_enter_hook,
          &launch_exit_hook, &compiled_kernel, &MyArgs[0], &MyArgs[1],
          &MyArgs[2], &MyArgs[3], &MyArgs[4], &MyArgs[5], &MyArgs[6],
          &MyArgs[7], &MyArgs[8], &MyArgs[9], &MyArgs[10], &MyArgs[11],
          &MyArgs[12], &MyArgs[13], &MyArgs[14], &MyArgs[15], &MyArgs[16])) {
    return NULL;
  }

  if (ArgC == 18 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0], &MyArgs[1], &MyArgs[2], &MyArgs[3],
                        &MyArgs[4], &MyArgs[5], &MyArgs[6], &MyArgs[7],
                        &MyArgs[8], &MyArgs[9], &MyArgs[10], &MyArgs[11],
                        &MyArgs[12], &MyArgs[13], &MyArgs[14], &MyArgs[15],
                        &MyArgs[16], &MyArgs[17])) {
    return NULL;
  }

  if (ArgC == 19 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0], &MyArgs[1], &MyArgs[2], &MyArgs[3],
                        &MyArgs[4], &MyArgs[5], &MyArgs[6], &MyArgs[7],
                        &MyArgs[8], &MyArgs[9], &MyArgs[10], &MyArgs[11],
                        &MyArgs[12], &MyArgs[13], &MyArgs[14], &MyArgs[15],
                        &MyArgs[16], &MyArgs[17], &MyArgs[18])) {
    return NULL;
  }

  if (ArgC == 20 &&
      !PyArg_ParseTuple(args, format, &arg0, &arg1, &gridX, &gridY, &gridZ,
                        &num_warps, &shared_memory, &_stream, &_function,
                        &launch_enter_hook, &launch_exit_hook, &compiled_kernel,
                        &MyArgs[0], &MyArgs[1], &MyArgs[2], &MyArgs[3],
                        &MyArgs[4], &MyArgs[5], &MyArgs[6], &MyArgs[7],
                        &MyArgs[8], &MyArgs[9], &MyArgs[10], &MyArgs[11],
                        &MyArgs[12], &MyArgs[13], &MyArgs[14], &MyArgs[15],
                        &MyArgs[16], &MyArgs[17], &MyArgs[18], &MyArgs[19])) {
    return NULL;
  }

  if (ArgC >= 21)
    abort();

  if (launch_enter_hook != Py_None) {
    PyObject *new_args = PyTuple_Pack(1, compiled_kernel);
    hook_ret = PyObject_CallObject(launch_enter_hook, new_args);
    Py_DECREF(new_args);
  }

  for (int index = 0; index < ArgC; ++index)
    if (format[strlen("ssiiiiiKKOOO") + index] == 'O')
      MyArgs[index].CUdptr = getPointer(MyArgs[index].O, index);

  void *params[100];
  for (int index = 0, param_index = 0; index < ArgC; ++index)
    if (is_const[strlen("ssiiiiiKKOOO") + index] == 'N')
      params[param_index++] = &MyArgs[index];

  my_launch(gridX, gridY, gridZ, num_warps, shared_memory, (CUstream)_stream,
            (CUfunction)_function, params);

  if (launch_exit_hook != Py_None) {
    PyObject *new_args = NULL;
    if (hook_ret) {
      new_args = PyTuple_Pack(2, compiled_kernel, hook_ret);
    } else {
      new_args = PyTuple_Pack(1, compiled_kernel);
    }
    hook_ret = PyObject_CallObject(launch_exit_hook, new_args);
    Py_DECREF(new_args);
  }

  if (hook_ret) {
    Py_DECREF(hook_ret);
  }
  if (PyErr_Occurred()) {
    return NULL;
  }
  // return None
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef ModuleMethods[] = {
    {"launch", launch, METH_VARARGS,
     "Entry point for all kernels with this signature"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "launcher",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_liblauncher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
