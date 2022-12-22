#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <Python.h>
static inline void gpuAssert(hipError_t code, const char *file, int line)
{{
   if (code != HIP_SUCCESS)
   {{
      const char* prefix = "Triton Error [CUDA]: ";
      const char* str = hipGetErrorString(code);
      char err[1024] = {{0}};
      strcat(err, prefix);
      strcat(err, str);
      PyErr_SetString(PyExc_RuntimeError, err);
   }}
}}
#define CUDA_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}
void _launch(int gridX, int gridY, int gridZ, int num_warps, int shared_memory, hipStream_t stream, hipFunction_t function, {arg_decls}) {{
  void *params[] = {{ {', '.join(f"&arg{i}" for i in signature.keys() if i not in constants)} }};
  if(gridX*gridY*gridZ > 0){{
    hipModuleLaunchKernel(function, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, params, 0);
  }}
}}
static inline hipDeviceptr_t getPointer(PyObject *obj, int idx) {{
  if (PyLong_Check(obj)) {{
    return (hipDeviceptr_t)PyLong_AsUnsignedLongLong(obj);
  }}
  if (obj == Py_None) {{
    return (hipDeviceptr_t)0;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
    }}
    return (hipDeviceptr_t)PyLong_AsUnsignedLongLong(ret);
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return (hipDeviceptr_t)0;
}}
static PyObject* launch(PyObject* self, PyObject* args) {{
  // printf("launch(PyObject* self, PyObject* args)");
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int num_warps;
  int shared_memory;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &shared_memory, &_stream, &_function, {', '.join(f"&_arg{i}" for i, ty in signature.items())})) {{
    return NULL;
  }}
  _launch(gridX, gridY, gridZ, num_warps, shared_memory, (hipStream_t)_stream, (hipFunction_t)_function, {', '.join(f"getPointer(_arg{i},{i})" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});
  if(PyErr_Occurred()) {{
    return NULL;
  }}
  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}
static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};
static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};
PyMODINIT_FUNC PyInit_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}