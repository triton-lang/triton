import functools
import os
import sysconfig
import subprocess
import tempfile
from pathlib import Path

from triton.common.backend import BaseBackend, compute_core_version_key, register_backend
from triton.compiler.make_launcher import make_so_cache_key
from triton.runtime.cache import get_cache_manager


def _get_triton_shared_opt_path() -> str:
    path = os.getenv("TRITON_SHARED_OPT_PATH", "")
    if path == "":
        raise Exception("TRITON_SHARED_OPT_PATH is not set.")
    return path


def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)


def _ttir_to_ttsharedir(mod):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join('/home/nhat/github/triton/third_party/triton_shared/tmp_cpu', "tt.mlir")
        dst_path = os.path.join('/home/nhat/github/triton/third_party/triton_shared/tmp_cpu', "ttshared.mlir")
        Path(src_path).write_text(ttir_code)
        triton_shared_opt_path = _get_triton_shared_opt_path()
        subprocess.check_call([triton_shared_opt_path, src_path, "--triton-to-linalg", "-o", dst_path])
        return Path(dst_path).read_text()


def _optimize_ttsharedir(ttsharedir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return ttsharedir


def _ttsharedir_to_llir(ttsharedir: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttshared_path = os.path.join(tmpdir, "ttshared.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(ttshared_path).write_text(ttsharedir)
        mlir_opt_path = _get_llvm_bin_path("mlir-opt")
        # TritonShared-MLIR to LLVM-MLIR
        subprocess.check_call([mlir_opt_path, ttshared_path,
            "--convert-linalg-to-affine-loops",
            "--eliminate-empty-tensors",
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=allow-return-allocs-from-loops=true",
            "--lower-affine",
            "--convert-linalg-to-loops",
            "--convert-scf-to-cf",
            "--convert-cf-to-llvm",
            "--convert-arith-to-llvm",
            "--convert-math-to-llvm",
            "--convert-complex-to-llvm",
            "--convert-vector-to-llvm",
            "--convert-index-to-llvm",
            "--memref-expand",
            "--expand-strided-metadata",
            "--finalize-memref-to-llvm",
            "--convert-func-to-llvm",
            "--lower-affine",
            "--convert-arith-to-llvm",
            "--reconcile-unrealized-casts",
            "-o",
            llmlir_path])

        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = _get_llvm_bin_path("mlir-translate")
        subprocess.check_call([mlir_translate_path, llmlir_path,
            "--mlir-to-llvmir",
            "-o",
            llir_path])
        return Path(llir_path).read_text()


def _optimize_llir(llir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return llir


def _llir_to_bin(llir: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        dst_path = os.path.join(tmpdir, "kernel.o")
        Path(src_path).write_text(llir)
        llc_path = _get_llvm_bin_path("llc")
        subprocess.check_call([llc_path, src_path, "-o", dst_path])
        # Actually it's text-format assembly.  Use read_text().
        return Path(dst_path).read_text()


def _ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
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

def _extracted_ty(ty):
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

def _format_of(ty):
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

def _generate_launcher(constants, signature, kernel_name):
    arg_decls = ', '.join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    format = "iiiOOO" + ''.join([_format_of(_extracted_ty(ty)) for ty in signature.values()])
    return f"""
#include <assert.h>
#include <stdbool.h>
#include <Python.h>
#include "CRunnerUtils.h"
#include "CRunnerUtils.cpp"

extern "C" {{
  // Pointer type (=Memref) becomes int64_t + MemRef struct
  // FIXME: understand what this int64_t is used for.
  void {kernel_name}({', '.join(_ty_to_cpp(ty) if ty[0] != "*" else f"int64_t, void*" for i, ty in signature.items() if i not in constants)},
                       int, int, int, int, int, int);
}}

static void _launch(int gridX, int gridY, int gridZ, {arg_decls}) {{
  if (gridX*gridY*gridZ > 0) {{
    // Cast "function" to the real function type.
    for(int x = 0; x < gridX; x++) {{
      for(int y = 0; y < gridY; y++) {{
        for(int z = 0; z < gridZ; z++) {{
          // Use some random type "char" here.
          {' '.join(f'StridedMemRefType<char, 0> ptr_arg{i} = {{static_cast<char *>(arg{i}), static_cast<char *>(arg{i}), 0}};' for i, ty in signature.items() if i not in constants and ty[0] == "*")}
          {kernel_name}({', '.join(f"static_cast<{_ty_to_cpp(ty)}>(arg{i})" if ty[0] != "*" else f"0, &ptr_arg{i}" for i, ty in signature.items() if i not in constants)},
                        gridX, gridY, gridZ, x, y, z);
        }}
      }}
    }}
  }}
}}

typedef struct _DevicePtrInfo {{
  void *dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
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
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *compiled_kernel = NULL;
  {' '.join([f"{_extracted_ty(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &launch_enter_hook, &launch_exit_hook, &compiled_kernel
                       {', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''})) {{
    return NULL;
  }}

  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {{
    return NULL;
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});

  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if (launch_exit_hook != Py_None && !PyObject_CallObject(launch_exit_hook, args)) {{
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
  \"__triton_shared_ref_cpu_kernel_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_shared_ref_cpu_kernel_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""


class TritonSharedRefCPUBackend(BaseBackend):
    stub_so_path = ""

    def __init__(self, device_type: str) -> None:
        super(TritonSharedRefCPUBackend, self).__init__(device_type)
        self.version_key = None

    def add_stages(self, arch, extern_libs, stages):
        filter_in_stages = ["ast", "ttir"]
        filter_out_stages = []
        for key, _ in stages.items():
            if key not in filter_in_stages:
                filter_out_stages.append(key)
        for filter_out_key in filter_out_stages:
            stages.pop(filter_out_key)

        stages["ttsharedir"] = (lambda path: Path(path).read_text(),
                               lambda src: _optimize_ttsharedir(_ttir_to_ttsharedir(src)))
        stages["llir"] = (lambda path: Path(path).read_text(),
                               lambda src: _optimize_llir(_ttsharedir_to_llir(src)))
        stages["cpuasm"] = (lambda path: Path(path).read_text(),
                               lambda src: _llir_to_bin(src))

    def add_meta_info(self, ir, module, next_module, metadata, asm):
        metadata["shared"] = 1
        if ir == "llir":
            # We can get a function name (C naming) from
            # LLVM-IR by getting the first "define void @".
            metadata["name"] = asm["llir"].split("define void @")[1].split("(")[0].strip()

    def get_driver(self):
        return None

    def get_version_key(self):
        if self.version_key is None:
            self.version_key = compute_core_version_key()
        return self.version_key

    def get_stream(self, idx=None) -> int:
        # Returns int to make Triton happy.
        return 0

    @functools.lru_cache(None)
    def get_device_properties(self, device):
        # CPU has no property.  Return some values to make the Triton runtime happy.
        return {"max_shared_mem": 2 ** 20}

    def get_current_device(self):
        # CPU doesn't have a device to return.  Return something.
        return "cpu"

    def set_current_device(self, device):
        # CPU doesn't have a device to set
        assert device == "cpu"
        return

    def get_load_binary_fn(self):
        def _load_binary_fn(kernel_name, binary, shared_size, device):
            # Returns mod, func, n_regs, n_spills, but this implementation does not use it.
            # Note: func is a function pointer.
            return None, 0, 0, 0
        return _load_binary_fn

    def get_kernel_bin(self):
        return "cpuasm"

    def get_architecture_descriptor(self, **kwargs):
        # CPU does not have the following parameters, but we need to pass some values to
        # make the Triton runtime happy.
        return {"num_warps": 1, "num_stages": 1}

    def make_launcher_stub(self, name, signature, constants, ids):
        # name of files that are cached
        so_cache_key = make_so_cache_key(self.get_version_key(), signature, constants, ids)
        so_cache_manager = get_cache_manager(so_cache_key)
        so_name = f"{name}.py"
        # retrieve stub from cache if it exists
        cache_path = so_cache_manager.get_file(so_name)
        if cache_path is not None:
            return cache_path

        kernel_placeholder_name = "KERNEL_NAME_PLACEHOLDER"
        with tempfile.TemporaryDirectory() as tmpdir:
            # Later KERNEL_NAME_PLACEHOLDER will be used to assign the kernel name
            # in the following launch function.
            launcher_src = _generate_launcher(constants, signature, kernel_placeholder_name)
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

            py_src = f"""
import os, subprocess, tempfile
import importlib.util
from pathlib import Path

def launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDim0, clusterDim1, clusterDim2,
           shared, stream, cu_function, launch_enter_hook, launch_exit_hook, compiled_kernel,
           *args):
    # Unlike CUDA/HIP, we cannot easily pass function pointer across different pybind libraries.
    # Let's compile a kernel every time.
    asm_src = compiled_kernel.asm["{self.get_kernel_bin()}"]
    launcher_src = '''
{launcher_src}
'''.replace("{kernel_placeholder_name}", compiled_kernel.metadata["name"])
    with tempfile.TemporaryDirectory() as tmpdir:
        asm_src_path = os.path.join(tmpdir, "kernel.s")
        launcher_src_path = os.path.join(tmpdir, "main.cxx")
        so_path = os.path.join(tmpdir, "kernel.so")
        Path(asm_src_path).write_text(asm_src)
        Path(launcher_src_path).write_text(launcher_src)
        # Compile it together.
        subprocess.check_call(["g++", launcher_src_path, asm_src_path, f"-I{py_include_dir}", f"-I{Path(__file__).resolve().parent}", "-shared", "-fPIC", "-o", so_path])

        # Load and launch the compiled kernel.
        spec = importlib.util.spec_from_file_location("__triton_shared_ref_cpu_kernel_launcher", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.launch(gridX, gridY, gridZ, launch_enter_hook, launch_exit_hook, compiled_kernel, *args)
"""
            return so_cache_manager.put(py_src, so_name, binary=False)


register_backend("cpu", TritonSharedRefCPUBackend)
