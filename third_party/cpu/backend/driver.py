import os
import hashlib
import tempfile
from pathlib import Path
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

dirname = os.getenv("TRITON_SYS_PATH", default="/usr/local")
llvm_root = os.getenv("LLVM_PATH", default="~/.triton/llvm")
llvm_root = os.path.expanduser(llvm_root)
llvm_dirs = os.listdir(llvm_root)
if len(llvm_dirs) == 1:
    llvm_root = os.path.join(llvm_root, llvm_dirs[0])
include_dir = [
    os.path.join(dirname, "include"),
    os.path.join(llvm_root, "include"),
]
library_dir = [os.path.join(dirname, "lib"), os.path.join(llvm_root, "lib")]
libraries = [
    "LLVMOrcJIT",
    "LLVMPasses",
    "LLVMX86CodeGen",
    "LLVMX86AsmParser",
    "LLVMX86Desc",
    "LLVMX86Info",
    "LLVMGlobalISel",
    "LLVMSelectionDAG",
    "LLVMHipStdPar",
    "LLVMCoroutines",
    "LLVMipo",
    "LLVMFrontendOpenMP",
    "LLVMInstrumentation",
    "LLVMAsmPrinter",
    "LLVMCodeGen",
    "LLVMObjCARCOpts",
    "LLVMLinker",
    "LLVMVectorize",
    "LLVMScalarOpts",
    "LLVMInstCombine",
    "LLVMFrontendOffloading",
    "LLVMExecutionEngine",
    "LLVMAggressiveInstCombine",
    "LLVMTransformUtils",
    "LLVMTarget",
    "LLVMRuntimeDyld",
    "LLVMJITLink",
    "LLVMIRPrinter",
    "LLVMBitWriter",
    "LLVMAnalysis",
    "LLVMProfileData",
    "LLVMSymbolize",
    "LLVMDebugInfoDWARF",
    "LLVMObject",
    "LLVMTextAPI",
    "LLVMMCParser",
    "LLVMMCDisassembler",
    "LLVMMC",
    "LLVMIRReader",
    "LLVMCFGuard",
    "LLVMBitReader",
    "LLVMAsmParser",
    "LLVMCore",
    "LLVMBinaryFormat",
    "LLVMOrcTargetProcess",
    "LLVMTargetParser",
    "LLVMRemarks",
    "LLVMOrcShared",
    "LLVMOption",
    "LLVMDebugInfoCodeView",
    "LLVMCodeGenTypes",
    "LLVMBitstreamReader",
    "LLVMSupport",
    "LLVMDemangle",
    "stdc++",
]


def compile_module_from_src(src, name):
    key = hashlib.md5(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.cpp")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, library_dir, include_dir, libraries)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------
# Utils
# ------------------------


class CPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass
        dirname = os.path.dirname(os.path.realpath(__file__))
        mod = compile_module_from_src(Path(os.path.join(dirname, "driver.cpp")).read_text(), "cpu_utils")
        self.load_binary = mod.load_binary

    def get_device_properties(self, *args):
        return {"max_shared_mem": 0}


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def make_launcher(constants, signature, ids):
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors.
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    def _extracted_type(ty):
        if ty[0] == '*':
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        return {
            "PyObject*": "O",
            "float": "f",
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "l",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty]

    args_format = ''.join([format_of(_extracted_type(ty)) for ty in signature.values()])
    format = "iiiOKOOOO" + args_format
    arg_ptrs_list = ', '.join(f"&arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''
    kernel_fn_args = [i for i in signature.keys() if i not in constants]
    kernel_fn_args_list = ', '.join(f"arg{i}" for i in kernel_fn_args) if len(kernel_fn_args) > 0 else ''
    kernel_fn_arg_types = (', '.join(f"{ty_to_cpp(signature[i])}" for i in kernel_fn_args) + ", " if len(signature) > 0 else '') + "uint32_t, uint32_t, uint32_t"

    # generate glue code
    src = f"""
#include <cstddef>
#include <string>
#include <iostream>
#include <iomanip>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdio.h>

using kernel_ptr_t = void(*)({kernel_fn_arg_types});

typedef struct _DevicePtrInfo {{
  void* dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(obj);
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
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(ret);
    if(!ptr_info.dev_ptr) {{
      return ptr_info;
    }}
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}}

static void run_omp_kernels(uint32_t gridX, uint32_t gridY, uint32_t gridZ, kernel_ptr_t kernel_ptr {', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  // TODO: add OMP pragmas to run in parallel
  for (uint32_t z = 0; z < gridZ; ++z) {{
    for (uint32_t y = 0; y < gridY; ++y) {{
      for (uint32_t x = 0; x < gridX; ++x) {{
        (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z);
      }}
    }}
  }}
}}

static PyObject* launch(PyObject* self, PyObject* args) {{


  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *py_obj_stream;
  void* pKrnl;

  {' '.join([f"{_extracted_type(ty)} arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &py_obj_stream, &pKrnl,
                                       &kernel_metadata, &launch_metadata,
                                       &launch_enter_hook, &launch_exit_hook {', ' + arg_ptrs_list if len(signature) > 0 else ''})) {{
    return NULL;
  }}

  void *pStream = PyLong_AsVoidPtr(py_obj_stream);
  kernel_ptr_t kernel_ptr = reinterpret_cast<kernel_ptr_t>(pKrnl);

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  run_omp_kernels(gridX, gridY, gridZ, kernel_ptr {',' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''});

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
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
  \"__triton_cpu_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_cpu_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


class CPULauncher(object):

    def __init__(self, src, metadata):
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        src = make_launcher(constants, signature, ids)
        mod = compile_module_from_src(src, "__triton_cpu_launcher")
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class CPUDriver(DriverBase):

    def __init__(self):
        self.utils = CPUUtils()
        self.launcher_cls = CPULauncher
        super().__init__()

    def get_current_device(self):
        return 0

    def get_current_stream(self, device):
        return 0

    def get_current_target(self):
        # Capability and warp size are zeros for CPU.
        # TODO: GPUTarget naming isn't obviously good.
        return GPUTarget("cpu", 0, 0)

    @staticmethod
    def is_active():
        return True
