import contextlib
import functools
import hashlib
import io
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile

import setuptools

from ..runtime.cache import CacheManager
from ..runtime.jit import version_key


def is_hip():
    import torch
    return torch.version.hip is not None


# ----- stub --------


@functools.lru_cache()
def libcuda_dirs():
    locs = subprocess.check_output(["whereis", "libcuda.so"]).decode().strip().split()[1:]
    return [os.path.dirname(loc) for loc in locs]


@functools.lru_cache()
def rocm_path_dir():
    return os.getenv("ROCM_PATH", default="/opt/rocm")


@contextlib.contextmanager
def quiet():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def _build(name, src, srcdir):
    if is_hip():
        hip_lib_dir = os.path.join(rocm_path_dir(), "lib")
        hip_include_dir = os.path.join(rocm_path_dir(), "include")
    else:
        cuda_lib_dirs = libcuda_dirs()
        base_dir = os.path.join(os.path.dirname(__file__), os.path.pardir)
        cuda_path = os.path.join(base_dir, "third_party", "cuda")

        cu_include_dir = os.path.join(cuda_path, "include")
        triton_include_dir = os.path.join(os.path.dirname(__file__), "include")
        cuda_header = os.path.join(cu_include_dir, "cuda.h")
        triton_cuda_header = os.path.join(triton_include_dir, "cuda.h")
        if not os.path.exists(cuda_header) and os.path.exists(triton_cuda_header):
            cu_include_dir = triton_include_dir
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    # try to avoid setuptools if possible
    cc = os.environ.get("CC")
    if cc is None:
        # TODO: support more things here.
        clang = shutil.which("clang")
        gcc = shutil.which("gcc")
        cc = gcc if gcc is not None else clang
        if cc is None:
            raise RuntimeError("Failed to find C compiler. Please specify via CC environment variable.")
    # This function was renamed and made public in Python 3.10
    if hasattr(sysconfig, 'get_default_scheme'):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]

    if is_hip():
        ret = subprocess.check_call([cc, src, f"-I{hip_include_dir}", f"-I{py_include_dir}", f"-I{srcdir}", "-shared", "-fPIC", f"-L{hip_lib_dir}", "-lamdhip64", "-o", so])
    else:
        cc_cmd = [cc, src, "-O3", f"-I{cu_include_dir}", f"-I{py_include_dir}", f"-I{srcdir}", "-shared", "-fPIC", "-lcuda", "-o", so]
        cc_cmd += [f"-L{dir}" for dir in cuda_lib_dirs]
        ret = subprocess.check_call(cc_cmd)

    if ret == 0:
        return so
    # fallback on setuptools
    extra_compile_args = []
    library_dirs = cuda_lib_dirs
    include_dirs = [srcdir, cu_include_dir]
    libraries = ['cuda']
    # extra arguments
    extra_link_args = []
    # create extension module
    ext = setuptools.Extension(
        name=name,
        language='c',
        sources=[src],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args + ['-O3'],
        extra_link_args=extra_link_args,
        library_dirs=library_dirs,
        libraries=libraries,
    )
    # build extension module
    args = ['build_ext']
    args.append('--build-temp=' + srcdir)
    args.append('--build-lib=' + srcdir)
    args.append('-q')
    args = dict(
        name=name,
        ext_modules=[ext],
        script_args=args,
    )
    with quiet():
        setuptools.setup(**args)
    return so


def make_so_cache_key(version_hash, signature, constants):
    # Get unique key for the compiled code
    signature = {k: 'ptr' if v[0] == '*' else v for k, v in signature.items()}
    key = f"{version_hash}-{''.join(signature.values())}{constants}"
    key = hashlib.md5(key.encode("utf-8")).hexdigest()
    return key


def make_stub(name, signature, constants):
    # name of files that are cached
    so_cache_key = make_so_cache_key(version_key(), signature, constants)
    so_cache_manager = CacheManager(so_cache_key)
    so_name = f"{name}.so"
    # retrieve stub from cache if it exists
    if not so_cache_manager.has_file(so_name):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = generate_launcher(constants, signature)
            src_path = os.path.join(tmpdir, "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir)
            with open(so, "rb") as f:
                so_cache_manager.put(f.read(), so_name, binary=True)
    return so_cache_manager._make_path(so_name)

# ----- source code generation --------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "hipDeviceptr_t" if is_hip() else "CUdeviceptr"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def generate_launcher(constants, signature):
    import torch
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    def _extracted_type(ty):
        if ty[0] == '*':
            return "PyObject*"
        return {
            'i1': 'int32_t',
            'i32': 'int32_t',
            'i64': 'int64_t',
            'u32': 'uint32_t',
            'u64': 'uint64_t',
            'fp16': 'float',
            'bf16': 'float',
            'fp32': 'float',
            'f32': 'float',
            'fp64': 'double',
        }[ty]

    def format_of(ty):
        return {
            "PyObject*": "O",
            "float": "f",
            "double": "d",
            "long": "l",
            "uint32_t": "I",
            "int32_t": "i",
            "uint64_t": "K",
            "int64_t": "L",
        }[ty]

    format = "iiiiiKKOOO" + ''.join([format_of(_extracted_type(ty)) for ty in signature.values()])

    # generate glue code
    if is_hip():
        src = f"""
    #define __HIP_PLATFORM_AMD__
    #include <hip/hip_runtime.h>
    #include <Python.h>
    #include <stdio.h>

    static inline void gpuAssert(hipError_t code, const char *file, int line)
    {{
      if (code != HIP_SUCCESS)
      {{
         const char* prefix = "Triton Error [HIP]: ";
         const char* str = hipGetErrorString(code);
         char err[1024] = {{0}};
         snprintf(err, 1024, "%s Code: %d, Messsage: %s", prefix, code, str );
         PyErr_SetString(PyExc_RuntimeError, err);
      }}
    }}

    #define HIP_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

    static void _launch(int gridX, int gridY, int gridZ, int num_warps, int shared_memory, hipStream_t stream, hipFunction_t function, {arg_decls}) {{
      void *params[] = {{ {', '.join(f"&arg{i}" for i in signature.keys() if i not in constants)} }};
      if (gridX*gridY*gridZ > 0) {{
          HIP_CHECK(hipModuleLaunchKernel(function, gridX, gridY, gridZ, 64*num_warps, 1, 1, shared_memory, stream, params, 0));
      }}
    }}

    typedef struct _DevicePtrInfo {{
      hipDeviceptr_t dev_ptr;
      bool valid;
    }} DevicePtrInfo;

    static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
      DevicePtrInfo ptr_info;
      ptr_info.dev_ptr = 0;
      ptr_info.valid = true;

      if (PyLong_Check(obj)) {{
        ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(obj);
        return ptr_info;
      }}

      if (obj == Py_None) {{
        // valid nullptr
        return ptr_info;
      }}

      PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");

      if (ptr) {{
        PyObject *empty_tuple = PyTuple_New(0);
        PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
        Py_DECREF(empty_tuple);
        Py_DECREF(ptr);

        if (!PyLong_Check(ret)) {{
          PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
          ptr_info.valid = false;
          return ptr_info;
        }}

        ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(ret);

        if (!ptr_info.dev_ptr)
          return ptr_info;

        uint64_t dev_ptr;
        hipError_t status = hipPointerGetAttribute(&dev_ptr, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
        if (status == hipErrorInvalidValue) {{
            PyErr_Format(PyExc_ValueError,
                         "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
            ptr_info.valid = false;
        }}

        ptr_info.dev_ptr = (hipDeviceptr_t)dev_ptr;
        return ptr_info;
      }}

      PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
      return ptr_info;
    }}

    static PyObject* launch(PyObject* self, PyObject* args) {{

      int gridX, gridY, gridZ;
      uint64_t _stream;
      uint64_t _function;
      int num_warps;
      int shared_memory;
      PyObject *launch_enter_hook = NULL;
      PyObject *launch_exit_hook = NULL;
      PyObject *compiled_kernel = NULL;

      {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
      if (!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &shared_memory, &_stream, &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel, {', '.join(f"&_arg{i}" for i, ty in signature.items())})) {{
        return NULL;
      }}

      if (launch_enter_hook != Py_None) {{
        PyObject_CallObject(launch_enter_hook, args);
      }}

      // raise exception asap
      {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
      _launch(gridX, gridY, gridZ, num_warps, shared_memory, (hipStream_t)_stream, (hipFunction_t)_function, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}" for i, ty in signature.items())});
      if (launch_exit_hook != Py_None) {{
        PyObject_CallObject(launch_exit_hook, args);
      }}
      if (PyErr_Occurred()) {{
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
      \"__triton_launcher\",
      NULL, //documentation
      -1, //size
      ModuleMethods
    }};

    PyMODINIT_FUNC PyInit___triton_launcher(void) {{
      PyObject *m = PyModule_Create(&ModuleDef);
      if(m == NULL) {{
        return NULL;
      }}
      PyModule_AddFunctions(m, ModuleMethods);
      return m;
    }}
    """
    else:
        src = f"""
#include \"cuda.h\"
#include <stdbool.h>
#include <Python.h>

static inline void gpuAssert(CUresult code, const char *file, int line)
{{
   if (code != CUDA_SUCCESS)
   {{
      const char* prefix = "Triton Error [CUDA]: ";
      const char* str;
      cuGetErrorString(code, &str);
      char err[1024] = {{0}};
      strcat(err, prefix);
      strcat(err, str);
      PyErr_SetString(PyExc_RuntimeError, err);
   }}
}}

#define CUDA_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int shared_memory, CUstream stream, CUfunction function, {arg_decls}) {{
  void *params[] = {{ {', '.join(f"&arg{i}" for i in signature.keys() if i not in constants)} }};
  if(gridX*gridY*gridZ > 0){{
    CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, params, 0));
  }}
}}

typedef struct _DevicePtrInfo {{
    CUdeviceptr dev_ptr;
    bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(ret);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    uint64_t dev_ptr;
    int status = cuPointerGetAttribute(&dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
    if (status == CUDA_ERROR_INVALID_VALUE) {{
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }}
    ptr_info.dev_ptr = dev_ptr;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int num_warps;
  int shared_memory;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *compiled_kernel = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &shared_memory, &_stream, &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel, {', '.join(f"&_arg{i}" for i, ty in signature.items())})) {{
    return NULL;
  }}

  if (launch_enter_hook != Py_None) {{
    PyObject_CallObject(launch_enter_hook, args);
  }}


  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, num_warps, shared_memory, (CUstream)_stream, (CUfunction)_function, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});

  if (launch_exit_hook != Py_None) {{
    PyObject_CallObject(launch_exit_hook, args);
  }}

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
  \"__triton_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src
