from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections import namedtuple
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import setuptools
import torch

import triton
import triton._C.libtriton.triton as _triton
from ..runtime.cache import CacheManager
from ..runtime.driver import get_cuda_utils, get_hip_utils
from ..tools.disasm import extract
from . import code_generator, errors


def str_to_ty(name):
    if name[0] == "*":
        ty = str_to_ty(name[1:])
        return triton.language.pointer_type(ty)
    tys = {
        "fp8e5": triton.language.float8e5,
        "fp8e4": triton.language.float8e4,
        "fp16": triton.language.float16,
        "bf16": triton.language.bfloat16,
        "fp32": triton.language.float32,
        "fp64": triton.language.float64,
        "i1": triton.language.int1,
        "i8": triton.language.int8,
        "i16": triton.language.int16,
        "i32": triton.language.int32,
        "i64": triton.language.int64,
        "u8": triton.language.uint8,
        "u16": triton.language.uint16,
        "u32": triton.language.uint32,
        "u64": triton.language.uint64,
        "B": triton.language.int1,
    }
    return tys[name]


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def kernel_suffix(signature, specialization):
    # suffix format:
    # <argid><'c' if equal to 1><'d' if divisible by 16>
    suffix = ''
    for i, _ in enumerate(signature):
        suffix += str(i)
        if i in specialization.equal_to_1:
            suffix += 'c'
        if i in specialization.divisible_by_16:
            suffix += 'd'
    return suffix

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def parse_mlir_module(path, context):
    module = _triton.ir.parse_mlir_module(path, context)
    # module takes ownership of the context
    module.context = context
    return module


def build_triton_ir(fn, signature, specialization, constants, debug=False):
    # canonicalize signature
    if isinstance(signature, str):
        signature = {k: v.strip() for k, v in enumerate(signature.split(","))}
    context = _triton.ir.context()
    context.load_triton()
    # create kernel prototype
    cst_key = lambda i: fn.arg_names.index(i) if isinstance(i, str) else i
    constants = {cst_key(key): value for key, value in constants.items()}
    # visit kernel AST
    gscope = fn.__globals__.copy()
    function_name = '_'.join([fn.__name__, kernel_suffix(signature.values(), specialization)])
    tys = list(signature.values())
    new_constants = {k: True if k in tys and tys[k] == "i1" else 1 for k in specialization.equal_to_1}
    new_attrs = {k: ("multiple_of", 16) for k in specialization.divisible_by_16}
    all_constants = constants.copy()
    all_constants.update(new_constants)
    arg_types = [str_to_ty(v) for k, v in signature.items() if k not in constants]

    prototype = triton.language.function_type([], arg_types)
    generator = code_generator.CodeGenerator(context, prototype, gscope=gscope, constants=all_constants,
                                             function_name=function_name, attributes=new_attrs,
                                             is_kernel=True, debug=debug)

    try:
        generator.visit(fn.parse())
    except errors.CompilationError as e:
        if e.src is None:
            e.set_source_code(fn.src)
        raise
    except Exception as e:
        node = generator.last_node
        if node is None:
            raise
        raise errors.CompilationError(fn.src, node, repr(e)) from e
    ret = generator.module
    # module takes ownership of the context
    ret.context = context
    return ret, generator


def optimize_triton_ir(mod):
    pm = _triton.ir.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_inliner_pass()
    pm.add_triton_combine_pass()
    pm.add_canonicalizer_pass()
    pm.add_cse_pass()
    pm.add_licm_pass()
    pm.add_symbol_dce_pass()
    pm.run(mod)
    return mod


def ast_to_ttir(fn, signature, specialization, constants, debug=False):
    mod, _ = build_triton_ir(fn, signature, specialization, constants, debug)
    return optimize_triton_ir(mod)


def ttir_to_ttgir(mod, num_warps):
    pm = _triton.ir.pass_manager(mod.context)
    pm.add_convert_triton_to_tritongpu_pass(num_warps)
    pm.run(mod)
    return mod


def optimize_ttgir(mod, num_stages, compute_capability):
    pm = _triton.ir.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_tritongpu_coalesce_pass()
    pm.add_tritongpu_remove_layout_conversions_pass()
    pm.add_tritongpu_accelerate_matmul_pass(compute_capability)
    pm.add_tritongpu_remove_layout_conversions_pass()
    pm.add_tritongpu_optimize_dot_operands_pass()
    pm.add_tritongpu_pipeline_pass(num_stages)
    pm.add_tritongpu_prefetch_pass()
    pm.add_tritongpu_optimize_dot_operands_pass()
    pm.add_tritongpu_remove_layout_conversions_pass()
    pm.add_tritongpu_decompose_conversions_pass()
    pm.add_tritongpu_reorder_instructions_pass()
    pm.add_cse_pass()
    pm.add_symbol_dce_pass()
    pm.run(mod)
    return mod


def add_external_libs(mod, libs):
    for name, path in libs.items():
        if len(name) == 0 or len(path) == 0:
            return
    _triton.add_external_libs(mod, list(libs.keys()), list(libs.values()))


def ttgir_to_llir(mod, extern_libs, compute_capability):
    if extern_libs:
        add_external_libs(mod, extern_libs)
    return _triton.translate_triton_gpu_to_llvmir(mod, compute_capability)


def llir_to_ptx(mod: Any, compute_capability: int, ptx_version: int = None) -> str:
    '''
    Translate TritonGPU module to PTX code.
    :param mod: a TritonGPU dialect module
    :return: PTX code
    '''
    if ptx_version is None:
        _, cuda_version = path_to_ptxas()
        ptx_version = ptx_get_version(cuda_version)
    return _triton.translate_llvmir_to_ptx(mod, compute_capability, ptx_version)


def ptx_to_cubin(ptx: str, compute_capability: int):
    '''
    Compile TritonGPU module to cubin.
    :param ptx: ptx code
    :param compute_capability: compute capability
    :return: str
    '''
    ptxas, _ = path_to_ptxas()
    return _triton.compile_ptx_to_cubin(ptx, ptxas, compute_capability)


def ptx_get_kernel_name(ptx: str) -> str:
    '''
    Get kernel name from PTX code.
    This Kernel name is required when launching the kernel.
    '''
    # There is a name mangling in PTX codegen, so the original kernel names in Triton IR are not available in PTX/cubin.
    assert ptx
    for line in ptx.split('\n'):
        line = line.strip()
        if line.startswith('// .globl'):
            return line.split()[-1]


def amdgcn_get_kernel_name(amdgcn: str) -> str:
    '''
    Get kernel name from AMDGCN code.
    This Kernel name is required when launching the kernel.
    '''
    assert amdgcn
    for line in amdgcn.split('\n'):
        line = line.strip()
        if line.startswith('.globl'):
            return line.split()[-1].strip()


def llir_to_amdgcn_and_hsaco(mod: Any, gfx_arch: str, gfx_triple: str, gfx_features: str) -> Tuple[str, str]:
    '''
    Translate TritonGPU module to HSACO code based on full details of gpu architecture.
    :param mod: a TritonGPU dialect module
    :return:
        - AMDGCN code
        - Path to HSACO object
    '''
    return _triton.translate_llvmir_to_hsaco(mod, gfx_arch, gfx_triple, gfx_features)


@functools.lru_cache
def ptx_get_version(cuda_version) -> int:
    '''
    Get the highest PTX version supported by the current CUDA driver.
    '''
    assert isinstance(cuda_version, str)
    major, minor = map(int, cuda_version.split('.'))
    if major == 12:
        return 80 + minor
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor
    raise RuntimeError("Triton only support CUDA 10.0 or higher")


def path_to_ptxas():
    base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    paths = [
        os.environ.get("TRITON_PTXAS_PATH", ""),
        os.path.join(base_dir, "third_party", "cuda", "bin", "ptxas")
    ]

    for ptxas in paths:
        if os.path.exists(ptxas) and os.path.isfile(ptxas):
            result = subprocess.check_output([ptxas, "--version"], stderr=subprocess.STDOUT)
            if result is not None:
                version = re.search(r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
                if version is not None:
                    return ptxas, version.group(1)
    raise RuntimeError("Cannot find ptxas")


instance_descriptor = namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1"], defaults=[set(), set()])


# ------------------------------------------------------------------------------
# compiler
# ------------------------------------------------------------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "hipDeviceptr_t" if torch.version.hip is not None else "CUdeviceptr"
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


def generate_name_initializer(signature):
    src = "int i = 0;\n"
    tys = signature.split(',')
    for i, ty in enumerate(tys):
        src


def binary_name_to_header_name(name):
    if len(name) > 128:
        # avoid filename too long errors (filename limit is 255)
        name = "kernel_" + hashlib.sha256(name.encode("utf-8")).hexdigest()
    return f"{name}.h"


def generate_launcher(constants, signature):
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
    if torch.version.hip is not None:
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


# Utilities for generating and compiling C wrappers


@functools.lru_cache()
def libcuda_dirs():
    locs = subprocess.check_output(["whereis", "libcuda.so"]).decode().strip().split()[1:]
    return [os.path.dirname(loc) for loc in locs]


@contextlib.contextmanager
def quiet():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


@functools.lru_cache()
def rocm_path_dir():
    return os.getenv("ROCM_PATH", default="/opt/rocm")


def _build(name, src, srcdir):
    if torch.version.hip is not None:
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

    if torch.version.hip is not None:
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


def make_fn_cache_key(fn_hash, signature, configs, constants, num_warps, num_stages):
    # Get unique key for the compiled code
    get_conf_key = lambda conf: (sorted(conf.divisible_by_16), sorted(conf.equal_to_1))
    configs_key = [get_conf_key(conf) for conf in configs]
    key = f"{fn_hash}-{''.join(signature.values())}-{configs_key}-{constants}-{num_warps}-{num_stages}"
    key = hashlib.md5(key.encode("utf-8")).hexdigest()
    return key


def read_or_execute(cache_manager, force_compile, file_name, metadata,
                    run_if_found: Callable[[str], bytes] = None,
                    run_if_not_found: Callable = None):
    suffix = file_name.split(".")[1]
    if not force_compile and cache_manager.has_file(file_name):
        module = run_if_found(cache_manager._make_path(file_name))
        data = module if isinstance(module, bytes) else str(module).encode("utf-8")
        md5 = hashlib.md5(data).hexdigest()
        has_changed = metadata and md5 != metadata["md5"][suffix]
        return module, md5, has_changed, True
    module = run_if_not_found()
    data = module if isinstance(module, bytes) else str(module).encode("utf-8")
    md5 = hashlib.md5(data).hexdigest()
    cache_manager.put(data, file_name, True if isinstance(data, bytes) else data)
    return module, md5, True, False


def get_amdgpu_arch_fulldetails():
    """
    get the amdgpu fulll ISA details for compiling:
    i.e., arch_triple: amdgcn-amd-amdhsa; arch_name: gfx906; arch_features: sramecc+:xnack-
    """
    try:
        rocminfo = subprocess.check_output(rocm_path_dir() + '/bin/rocminfo').decode()
        gfx_arch_details = re.search('amd.*', rocminfo).group(0).strip().split('--')
        arch_triple = gfx_arch_details[0]
        arch_name_features = gfx_arch_details[1].split(':')
        arch_name = arch_name_features[0]
        arch_features = ""

        if (len(arch_name_features) == 3):
            arch_features = "+" + re.search('\\w+', arch_name_features[1]).group(0) + ","\
                            "-" + re.search('\\w+', arch_name_features[2]).group(0)
        return [arch_triple, arch_name, arch_features]
    except BaseException:
        return None


def make_stub(name, signature, constants):
    # name of files that are cached
    so_cache_key = make_so_cache_key(triton.runtime.jit.version_key(), signature, constants)
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


def convert_type_repr(x):
    match = re.search(r'!tt\.ptr<(.*)>', x)
    if match is not None:
        return '*' + convert_type_repr(match.group(1))
    return x


def make_hash(fn, **kwargs):
    if isinstance(fn, triton.runtime.JITFunction):
        configs = kwargs["configs"]
        signature = kwargs["signature"]
        constants = kwargs.get("constants", dict())
        num_warps = kwargs.get("num_warps", 4)
        num_stages = kwargs.get("num_stages", 3)
        # Get unique key for the compiled code
        get_conf_key = lambda conf: (sorted(conf.divisible_by_16), sorted(conf.equal_to_1))
        configs_key = [get_conf_key(conf) for conf in configs]
        key = f"{fn.cache_key}-{''.join(signature.values())}-{configs_key}-{constants}-{num_warps}-{num_stages}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()
    assert isinstance(fn, str)
    return hashlib.md5((Path(fn).read_text() + triton.runtime.jit.version_key()).encode("utf-8")).hexdigest()


# - ^\s*func\.func\s+ : match the start of the string, any leading whitespace, the keyword func,
#    and any following whitespace
# - (public\s+)? : optionally match the keyword public and any following whitespace
# - (@\w+) : match an @ symbol followed by one or more word characters
#   (letters, digits, or underscores), and capture it as group 1 (the function name)
# - (\((?:%\w+: \S+(?: \{\S+ = \S+ : \S+\})?(?:, )?)*\)) : match a pair of parentheses enclosing
#   zero or more arguments separated by commas, and capture it as group 2 (the argument list)
mlir_prototype_pattern = r'^\s*func\.func\s+(?:public\s+)?(@\w+)(\((?:%\w+: \S+(?: \{\S+ = \S+ : \S+\})?(?:, )?)*\))\s*\{\s*$'
ptx_prototype_pattern = r"\.(?:visible|extern)\s+\.(?:entry|func)\s+(\w+)\s*\(([^)]*)\)"
prototype_pattern = {
    "ttir": mlir_prototype_pattern,
    "ttgir": mlir_prototype_pattern,
    "ptx": ptx_prototype_pattern,
}

mlir_arg_type_pattern = r'%\w+: ([^,^\)\s]+)(?: \{\S+ = \S+ : \S+\})?,?'
ptx_arg_type_pattern = r"\.param\s+\.(\w+)"
arg_type_pattern = {
    "ttir": mlir_arg_type_pattern,
    "ttgir": mlir_arg_type_pattern,
    "ptx": ptx_arg_type_pattern,
}


def _get_jsonable_constants(constants):
    def _is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False
    serialized_constants = {}
    for constant in constants:
        if _is_jsonable(constants[constant]):
            serialized_constants[constant] = constants[constant]
    return serialized_constants

# def compile(fn, signature: str, device: int = -1, constants=dict(), num_warps: int = 4, num_stages: int = 3, extern_libs=None, configs=None):


@static_vars(discovered_gfx_arch_fulldetails=get_amdgpu_arch_fulldetails())
def compile(fn, **kwargs):
    capability = kwargs.get("cc", None)
    if capability is None:
        device = triton.runtime.jit.get_current_device()
        capability = triton.runtime.jit.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
    # we get the kernel, i.e. the first function generated in the module
    # if fn is not a JITFunction, then it
    # has to be a path to a file
    context = _triton.ir.context()
    asm = dict()
    constants = kwargs.get("constants", dict())
    num_warps = kwargs.get("num_warps", 4)
    num_stages = kwargs.get("num_stages", 3 if capability >= 75 else 2)
    extern_libs = kwargs.get("extern_libs", dict())
    debug = kwargs.get("debug", False)
    # build compilation stages
    stages = {
        "ast": (lambda path: fn, None),
        "ttir": (lambda path: parse_mlir_module(path, context),
                 lambda src: ast_to_ttir(src, signature, configs[0], constants)),
        "ttgir": (lambda path: parse_mlir_module(path, context),
                  lambda src: optimize_ttgir(ttir_to_ttgir(src, num_warps), num_stages, capability)),
        "llir": (lambda path: Path(path).read_text(),
                 lambda src: ttgir_to_llir(src, extern_libs, capability)),
    }
    if torch.version.hip is not None:
        _triton.set_rocm()
        if extern_libs is None:
            extern_libs = get_amdgcn_bitcode_paths()
        else:
            extern_libs.update(get_amdgcn_bitcode_paths())

        for key in list(extern_libs):
            if extern_libs[key] == '' or extern_libs[key] is None:
                extern_libs.pop(key)

        gfx_arch_full_details = compile.discovered_gfx_arch_fulldetails
        gfx_arch = os.environ.get('MI_GPU_ARCH', gfx_arch_full_details[1])
        if gfx_arch is None:
            raise RuntimeError('gfx_arch is None (not specified)')
        stages["amdgcn"] = (lambda path: Path(path).read_text(),
                            lambda src: llir_to_amdgcn_and_hsaco(src, gfx_arch,
                                                                 gfx_arch_full_details[0],
                                                                 gfx_arch_full_details[2]))
    else:
        stages["ptx"] = (lambda path: Path(path).read_text(),
                         lambda src: llir_to_ptx(src, capability))
        stages["cubin"] = (lambda path: Path(path).read_bytes(),
                           lambda src: ptx_to_cubin(src, capability))

    # find out the signature of the function
    if isinstance(fn, triton.runtime.JITFunction):
        configs = kwargs.get("configs", None)
        signature = kwargs["signature"]
        if configs is None:
            configs = [instance_descriptor()]
        assert len(configs) == 1
        kwargs["configs"] = configs
        name = fn.__name__
        first_stage = 0
        if isinstance(signature, str):
            signature = {k: v.strip() for k, v in enumerate(signature.split(","))}
        kwargs["signature"] = signature
    else:
        assert isinstance(fn, str)
        _, ir = os.path.basename(fn).split(".")
        src = Path(fn).read_text()
        import re
        match = re.search(prototype_pattern[ir], src, re.MULTILINE)
        name, signature = match.group(1), match.group(2)
        # print(name, signature)
        types = re.findall(arg_type_pattern[ir], signature)
        # print(types)
        param_tys = [convert_type_repr(ty) for ty in types]
        signature = {k: v for k, v in enumerate(param_tys)}
        first_stage = list(stages.keys()).index(ir)

    # cache manager
    so_path = make_stub(name, signature, constants)
    # create cache manager
    fn_cache_manager = CacheManager(make_hash(fn, **kwargs))
    # determine name and extension type of provided function
    if isinstance(fn, triton.runtime.JITFunction):
        name, ext = fn.__name__, "ast"
    else:
        name, ext = os.path.basename(fn).split(".")

    # load metadata if any
    metadata = None
    if fn_cache_manager.has_file(f'{name}.json'):
        with open(fn_cache_manager._make_path(f"{name}.json")) as f:
            metadata = json.load(f)
    else:
        metadata = {"num_warps": num_warps, "num_stages": num_stages,
                    "constants": _get_jsonable_constants(constants), "ctime": dict(), "debug": debug}
        if ext == "ptx":
            assert "shared" in kwargs, "ptx compilation must provide shared memory size"
            metadata["shared"] = kwargs["shared"]

    first_stage = list(stages.keys()).index(ext)
    asm = dict()
    module = fn
    # run compilation pipeline  and populate metadata
    for ir, (parse, compile_kernel) in list(stages.items())[first_stage:]:
        path = fn_cache_manager._make_path(f"{name}.{ir}")
        if ir == ext:
            next_module = parse(fn)
        elif os.path.exists(path) and\
                ir in metadata["ctime"] and\
                os.path.getctime(path) == metadata["ctime"][ir]:
            if ir == "amdgcn":
                next_module = (parse(path), parse(fn_cache_manager._make_path(f"{name}.hsaco_path")))
            else:
                next_module = parse(path)
        else:
            next_module = compile_kernel(module)
            if ir == "amdgcn":
                fn_cache_manager.put(next_module[0], f"{name}.{ir}")
                fn_cache_manager.put(next_module[1], f"{name}.hsaco_path")
            else:
                fn_cache_manager.put(next_module, f"{name}.{ir}")
        if os.path.exists(path):
            metadata["ctime"][ir] = os.path.getctime(path)
        if ir == "cubin":
            asm[ir] = next_module
        elif ir == "amdgcn":
            asm[ir] = str(next_module[0])
        else:
            asm[ir] = str(next_module)
        if ir == "llir" and "shared" not in metadata:
            metadata["shared"] = _triton.get_shared_memory_size(module)
        if ir == "ptx":
            metadata["name"] = ptx_get_kernel_name(next_module)
        if ir == "amdgcn":
            metadata["name"] = amdgcn_get_kernel_name(next_module[0])
            asm["hsaco_path"] = next_module[1]
        module = next_module
    # write-back metadata
    fn_cache_manager.put(json.dumps(metadata), f"{name}.json", binary=False)
    # return handle to compiled kernel
    return CompiledKernel(fn, so_path, metadata, asm)


@static_vars(discovered_gfx_arch_fulldetails=get_amdgpu_arch_fulldetails())
def _get_amdgcn_bitcode_paths():
    if torch.version.hip is not None:
        gpu_arch_agnostic_bitcode_libraries = ["opencl.bc",
                                               "ocml.bc",
                                               "ockl.bc",
                                               "oclc_finite_only_off.bc",
                                               "oclc_daz_opt_off.bc",
                                               "oclc_correctly_rounded_sqrt_on.bc",
                                               "oclc_unsafe_math_off.bc",
                                               "oclc_wavefrontsize64_on.bc"]

        gfx_arch = _get_amdgcn_bitcode_paths.discovered_gfx_arch_fulldetails[1]
        gfx_arch_id = re.search('gfx(\\w+)', gfx_arch).group(1).strip()

        gpu_arch_specific_bitcode_library = 'oclc_isa_version_' + gfx_arch_id + ".bc"
        bitcode_path_dir = os.path.join(Path(__file__).parent.resolve(), "third_party/rocm/lib/bitcode/")

        amdgcn_bitcode_paths = {}
        i = 1
        for bc_lib in gpu_arch_agnostic_bitcode_libraries:
            bc_path = bitcode_path_dir + bc_lib
            if os.path.exists(bc_path):
                amdgcn_bitcode_paths['library_' + str(i)] = bc_path
                i += 1
        bc_gfx_path = bitcode_path_dir + gpu_arch_specific_bitcode_library
        if os.path.exists(bc_gfx_path):
            amdgcn_bitcode_paths['library_' + str(i)] = bc_gfx_path

        return amdgcn_bitcode_paths
    else:
        return {}


@static_vars(amdgcn_bitcode_paths=_get_amdgcn_bitcode_paths())
def get_amdgcn_bitcode_paths():
    return get_amdgcn_bitcode_paths.amdgcn_bitcode_paths


class CompiledKernel:

    # Hooks for external tools to monitor the execution of triton kernels
    launch_enter_hook = None
    launch_exit_hook = None

    def __init__(self, fn, so_path, metadata, asm):
        # initialize launcher
        import importlib.util
        spec = importlib.util.spec_from_file_location("__triton_launcher", so_path)
        mod = importlib.util.module_from_spec(spec)
        self.fn = fn
        spec.loader.exec_module(mod)
        self.c_wrapper = getattr(mod, "launch")
        # initialize metadata
        self.shared = metadata["shared"]
        self.num_warps = metadata["num_warps"]
        self.num_stages = metadata["num_stages"]
        self.constants = metadata["constants"]
        # initialize asm dict
        self.asm = asm
        # binaries are lazily initialized
        # because it involves doing runtime things
        # (e.g., checking amount of shared memory on current device)
        self.metadata = metadata
        self.cu_module = None
        self.cu_function = None

    def _init_handles(self):
        if self.cu_module is not None:
            return
        device = triton.runtime.jit.get_current_device()
        if torch.version.hip is not None:
            hip_utils = get_hip_utils()
            max_shared = hip_utils.get_device_properties(device)["max_shared_mem"]
            if self.shared > max_shared:
                raise OutOfResources(self.shared, max_shared, "shared memory")
            mod, func, n_regs, n_spills = hip_utils.load_binary(self.metadata["name"], self.asm["hsaco_path"], self.shared, device)
        else:
            cuda_utils = get_cuda_utils()
            max_shared = cuda_utils.get_device_properties(device)["max_shared_mem"]
            if self.shared > max_shared:
                raise OutOfResources(self.shared, max_shared, "shared memory")
            mod, func, n_regs, n_spills = cuda_utils.load_binary(self.metadata["name"], self.asm["cubin"], self.shared, device)

        self.n_spills = n_spills
        self.n_regs = n_regs
        self.cu_module = mod
        self.cu_function = func

    def __getattribute__(self, name):
        if name == 'c_wrapper':
            self._init_handles()
        return super().__getattribute__(name)

    def __getitem__(self, grid):
        self._init_handles()

        def runner(*args, stream=None):
            if stream is None:
                stream = triton.runtime.jit.get_cuda_stream()
            self.c_wrapper(grid[0], grid[1], grid[2], self.num_warps, self.shared, stream, self.cu_function,
                           CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook, self, *args)
        return runner

    def get_sass(self, fun=None):
        if 'sass' in self.asm:
            return self.asm['sass']
        fd, path = tempfile.mkstemp()
        try:
            with open(fd, 'wb') as cubin:
                cubin.write(self.asm['cubin'])
            self.sass = extract(path, fun)
        finally:
            os.remove(path)
        self.asm['sass'] = self.sass
        return self.sass
