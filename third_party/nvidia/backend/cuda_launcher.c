#include "cuda.h"
#include <stdbool.h>
#include <Python.h>
#include <dlfcn.h>

#define IS_EMPTY_HELPER(x) IS_EMPTY_HELPER_##x
#define IS_EMPTY_HELPER_ 1
#define IS_EMPTY(x) IS_EMPTY_HELPER(x)

// macros that should be filled in by driver.py:
// #define EXTRA_INNER_LAUNCH_PARAM_DECLS
// #define INNER_LAUNCH_CUDA_CHECK_ARGS
// #define LAUNCH_PY_ARGS
// #define PY_ARG_FORMAT_STR
// #define EXTRA_LAUNCH_PARSE_PY_ARGS
// #define DEVICE_PTR_INFO_VARS
// #define TMA_DESC_VARS
// #define EXTRA_INNER_LAUNCH_CALL_ARGS
//
// nomenclature: "EXTRA" means extra args appended to the end of the function call, which
// requires adding a comma to the end of the previous arg in driver.py.
// "INNER" means the inner function call of `_launch()`.
// 

// #ifndef PARAMS
//   #error "PARAMS must be defined"
// #endif

// #ifndef VAR_LIST_IN_LAUNCH
//   #error "VAR_LIST_IN_LAUNCH must be defined"
// #endif

// #ifndef PY_ARG_FORMAT_STR
//   #error "PY_ARG_FORMAT_STR must be defined"
// #endif

// #ifndef LAUNCH_PARSE_PY_ARGS
//   #error "LAUNCH_PARSE_PY_ARGS must be defined"
// #endif

// #ifndef DEVICE_PTR_INFO_VARS
//   #error "DEVICE_PTR_INFO_VARS must be defined"
// #endif

// #ifndef TMA_DESC_VARS
//   #error "TMA_DESC_VARS must be defined"
// #endif

// #ifndef EXTRA_INNER_LAUNCH_CALL_ARGS
//   #error "EXTRA_INNER_LAUNCH_CALL_ARGS must be defined"
// #endif

static inline void gpuAssert(CUresult code, const char *file, int line)
{
   if (code != CUDA_SUCCESS)
   {
      const char* prefix = "Triton Error [CUDA]: ";
      const char* str;
      cuGetErrorString(code, &str);
      char err[1024] = {0};
      strcat(err, prefix);
      strcat(err, str);
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }
}

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

typedef CUresult (*cuLaunchKernelEx_t)(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra);

static cuLaunchKernelEx_t getLaunchKernelExHandle() {
  // Open the shared library
  void* handle = dlopen("libcuda.so.1", RTLD_LAZY);
  if (!handle) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to open libcuda.so.1");
    return NULL;
  }
  // Clear any existing error
  dlerror();
  cuLaunchKernelEx_t cuLaunchKernelExHandle = (cuLaunchKernelEx_t)dlsym(handle, "cuLaunchKernelEx");
  // Check for errors
  const char *dlsym_error = dlerror();
  if (dlsym_error) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to retrieve cuLaunchKernelEx from libcuda.so.1");
    return NULL;
  }
  return cuLaunchKernelExHandle;
}

// define a macro from driver.py to introduce extra args
// e.g.:
//     arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
//     cuda_launcher_src += f"#define ARG_DECLS {arg_decls}".format(arg_decls=arg_decls)
static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, CUstream stream, CUfunction function EXTRA_INNER_LAUNCH_PARAM_DECLS) {
  // define a macro from driver.py to introduce extra params
  // e.g.: cuda_launcher_src += f"#define PARAMS {params}".format(params=', '.join(f"&arg{i}" for i in params))
  void *params[] = { 
    INNER_LAUNCH_CUDA_CHECK_ARGS
   };
  if (gridX*gridY*gridZ > 0) {
    if (num_ctas == 1) {
      CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, params, 0));
    } else {
      CUlaunchAttribute launchAttr[2];
      launchAttr[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      launchAttr[0].value.clusterDim.x = clusterDimX;
      launchAttr[0].value.clusterDim.y = clusterDimY;
      launchAttr[0].value.clusterDim.z = clusterDimZ;
      launchAttr[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      launchAttr[1].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
      CUlaunchConfig config;
      config.gridDimX = gridX * clusterDimX;
      config.gridDimY = gridY * clusterDimY;
      config.gridDimZ = gridZ * clusterDimZ;
      config.blockDimX = 32 * num_warps;
      config.blockDimY = 1;
      config.blockDimZ = 1;
      config.sharedMemBytes = shared_memory;
      config.hStream = stream;
      config.attrs = launchAttr;
      config.numAttrs = 2;
      static cuLaunchKernelEx_t cuLaunchKernelExHandle = NULL;
      if (cuLaunchKernelExHandle == NULL) {
        cuLaunchKernelExHandle = getLaunchKernelExHandle();
      }
      CUDA_CHECK(cuLaunchKernelExHandle(&config, function, params, 0));
    }
  }
}

typedef struct _DevicePtrInfo {
    CUdeviceptr dev_ptr;
    bool valid;
} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }
  if (obj == Py_None) {
    // valid nullptr
    return ptr_info;
  }
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(ret);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    uint64_t dev_ptr;
    int status = cuPointerGetAttribute(&dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
    if (status == CUDA_ERROR_INVALID_VALUE) {
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }
    ptr_info.dev_ptr = dev_ptr;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}

static inline CUtensorMap* getTmaDesc(PyObject *obj) {
  if (sizeof(CUtensorMap*) != 8) {
    PyErr_SetString(PyExc_SystemError, "getTmaDesc() requires 64-bit compilation");
    return NULL;
  }

  PyObject *method_handle = PyObject_GetAttrString(obj, "tma_desc_cpu_ptr");
  if (!method_handle) {
    PyErr_SetString(PyExc_TypeError, "tma_desc_cpu_ptr() method does not exist");
    return NULL;
  }

  PyObject *empty_tuple = PyTuple_New(0);
  if (!empty_tuple) {
    Py_DECREF(method_handle);
    PyErr_SetString(PyExc_SystemError, "Internal Python error!");
    return NULL;
  }
  PyObject *method_ret = PyObject_Call(method_handle, empty_tuple, NULL);
  Py_DECREF(empty_tuple);
  Py_DECREF(method_handle);
  if (!method_ret) {
    PyErr_SetString(PyExc_SystemError, "Internal Python error!");
    return NULL;
  }

  if (!PyLong_Check(method_ret)) {
    PyErr_SetString(PyExc_TypeError, "tma_desc_cpu_ptr() must return 64-bit int");
    Py_DECREF(method_ret);
    return NULL;
  }

  uint64_t ptr_as_uint = PyLong_AsUnsignedLongLong(method_ret);
  Py_DECREF(method_ret);
  if (!ptr_as_uint) {
    PyErr_SetString(PyExc_ValueError, "received NULL ptr from tma_desc_cpu_ptr()");
    return NULL;
  }
  if (ptr_as_uint % 64 != 0) {
    PyErr_SetString(PyExc_ValueError, "tma_desc_cpu_ptr() must be 64-byte aligned");
    return NULL;
  }

  return (CUtensorMap*)(ptr_as_uint);
}

static PyObject* launch(PyObject* self, PyObject* args) {
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  // example python code to generate the arg list:
  // ' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])
  LAUNCH_PY_ARGS;
  if(!PyArg_ParseTuple(args, 
  PY_ARG_FORMAT_STR
  , &gridX, &gridY, &gridZ, &_stream, &_function,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook
                                           EXTRA_LAUNCH_PARSE_PY_ARGS
                                           )) {
    return NULL;
  }

  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  if (!PyArg_ParseTuple(kernel_metadata, "iiiiii", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {
    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
    return NULL;
  }

  // extract launch metadata
  if (launch_enter_hook != Py_None){
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }

  // raise exception asap
  // python string: "".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])
  DEVICE_PTR_INFO_VARS;
  // python string:"".join([f"CUtensorMap* tma_ptr{i} = getTmaDesc(_arg{i}); if (!tma_ptr{i}) return NULL;" if ty == "nvTmaDesc" else "" for i, ty in signature.items()])
  TMA_DESC_VARS;

  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (CUstream)_stream, (CUfunction)_function
    EXTRA_INNER_LAUNCH_CALL_ARGS
  );
  Py_END_ALLOW_THREADS;
  if (PyErr_Occurred()) {
    return NULL;
  }
  if(launch_exit_hook != Py_None){
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;

  }

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef ModuleMethods[] = {
  {"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__triton_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit___triton_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
