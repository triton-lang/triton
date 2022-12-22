#include "cuda.h"
#include <Python.h>

static inline void gpuAssert(CUresult code, const char *file, int line) {
  {
    if (code != CUDA_SUCCESS) {
      {
        const char *prefix = "Triton Error [CUDA]: ";
        const char *str;
        cuGetErrorString(code, &str);
        char err[1024] = {{0}};
        strcat(err, prefix);
        strcat(err, str);
        PyErr_SetString(PyExc_RuntimeError, err);
      }
    }
  }
}

#define CUDA_CHECK(ans)                                                        \
  {                                                                            \
    { gpuAssert((ans), __FILE__, __LINE__); }                                  \
  }

void _launch(int gridX, int gridY, int gridZ, int num_warps, int shared_memory,
             CUstream stream, CUfunction function, {arg_decls}) {
  {
    void *params[] = {{ {', '.join(f"&arg{i}" for i in signature.keys() if i not in constants)} }};
    if (gridX * gridY * gridZ > 0) {
      {
        CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, 32 * num_warps,
                                  1, 1, shared_memory, stream, params, 0));
      }
    }
  }
}

static inline CUdeviceptr getPointer(PyObject *obj, int idx) {
  {
    if (PyLong_Check(obj)) {
      { return (CUdeviceptr)PyLong_AsUnsignedLongLong(obj); }
    }
    if (obj == Py_None) {
      { return (CUdeviceptr)0; }
    }
    PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
    if (ptr) {
      {
        PyObject *empty_tuple = PyTuple_New(0);
        PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
        Py_DECREF(empty_tuple);
        Py_DECREF(ptr);
        if (!PyLong_Check(ret)) {
          {
            PyErr_SetString(
                PyExc_TypeError,
                "data_ptr method of Pointer object must return 64-bit int");
          }
        }
        return (CUdeviceptr)PyLong_AsUnsignedLongLong(ret);
      }
    }
    PyErr_SetString(
        PyExc_TypeError,
        "Pointer argument must be either uint64 or have data_ptr method");
    return (CUdeviceptr)0;
  }
}

static PyObject *launch(PyObject *self, PyObject *args) {
  {
    int gridX, gridY, gridZ;
    uint64_t _stream;
    uint64_t _function;
    int num_warps;
    int shared_memory;
    PyObject *launch_enter_hook = NULL;
    PyObject *launch_exit_hook = NULL;
    PyObject *compiled_kernel = NULL;
    PyObject *hook_ret = NULL;
    {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])
    }
    if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &shared_memory, &_stream, &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel, {', '.join(f"&_arg{
      i}" for i, ty in signature.items())})) {{
        return NULL;
  }
}

if (launch_enter_hook != Py_None) {
  {
    PyObject *new_args = PyTuple_Pack(1, compiled_kernel);
    hook_ret = PyObject_CallObject(launch_enter_hook, new_args);
    Py_DECREF(new_args);
  }
}

_launch(gridX, gridY, gridZ, num_warps, shared_memory, (CUstream)_stream,
        (CUfunction)_function, {', '.join(f"getPointer(_arg{i},{i})" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())
        });

if (launch_exit_hook != Py_None) {
  {
    PyObject *new_args = NULL;
    if (hook_ret) {
      { new_args = PyTuple_Pack(2, compiled_kernel, hook_ret); }
    } else {
      { new_args = PyTuple_Pack(1, compiled_kernel); }
    }
    hook_ret = PyObject_CallObject(launch_exit_hook, new_args);
    Py_DECREF(new_args);
  }
}

if (hook_ret) {
  { Py_DECREF(hook_ret); }
}
if (PyErr_Occurred()) {
  { return NULL; }
}
// return None
Py_INCREF(Py_None);
return Py_None;
}
}

static PyMethodDef ModuleMethods[] = {{
    {{"launch", launch, METH_VARARGS,
      "Entry point for all kernels with this signature"}},
    {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
    PyModuleDef_HEAD_INIT,
    \"launcher\",
    NULL, //documentation
    -1, //size
    ModuleMethods
    }};

PyMODINIT_FUNC PyInit_launcher(void) {
  {
    PyObject *m = PyModule_Create(&ModuleDef);
    if (m == NULL) {
      { return NULL; }
    }
    PyModule_AddFunctions(m, ModuleMethods);
    return m;
  }
}