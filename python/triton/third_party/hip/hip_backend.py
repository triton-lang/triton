import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Tuple


from triton.common import _build
from triton.common.backend import BaseBackend, register_backend, compute_core_version_key
from triton.compiler.make_launcher import get_cache_manager, make_so_cache_key
from triton.compiler.utils import generate_cu_signature
from triton.runtime import jit
from triton.runtime.driver import HIPDriver
from triton.compiler.compiler import optimize_ttgir, parse_mlir_module, ttgir_to_llir, ttir_to_ttgir

HIP_BACKEND_MODE = False

if HIP_BACKEND_MODE:
    from ..._C.librocm_backend_for_triton import triton as _triton
else:
    from ..._C.libtriton import triton as _triton


def make_stub(name, signature, constants, ids, **kwargs):
    # name of files that are cached
    so_cache_key = make_so_cache_key(compute_core_version_key(), signature, constants, ids, **kwargs)
    so_cache_manager = get_cache_manager(so_cache_key)
    so_name = f"{name}.so"
    # retrieve stub from cache if it exists
    cache_path = so_cache_manager.get_file(so_name)
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = generate_launcher_hip(constants, signature, ids)
            src_path = os.path.join(tmpdir, "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir)
            with open(so, "rb") as f:
                return so_cache_manager.put(f.read(), so_name, binary=True)
    else:
        return cache_path


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "hipDeviceptr_t"
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

def generate_launcher_hip(constants, signature, ids):
    start_desc = len(signature)
    warp_size = _triton.get_warp_size()
    #signature = generate_cu_signature(constants, signature, ids)
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

    format = "iiiiiiiiiKKOOO" + ''.join([format_of(_extracted_type(ty)) for ty in signature.values()])

    # generate glue code
    folded_without_constexprs = [c for c in ids['ids_of_folded_args'] if c not in ids['ids_of_const_exprs']]
    params = [i for i in signature.keys() if i >= start_desc or (i not in constants and i not in folded_without_constexprs)]
    src = f"""
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <Python.h>
#include <stdbool.h>
#include <dlfcn.h>

static inline void gpuAssert(hipError_t code, const char *file, int line)
{{
   if (code != HIP_SUCCESS)
   {{
      const char* prefix = "Triton Error [HIP]: ";
      const char* str = hipGetErrorString(code);
      char err[1024] = {{0}};
      snprintf(err, 1024, "%s Code: %d, Messsage: %s", prefix, code, str );
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }}
}}

#define HIP_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, hipStream_t stream, hipFunction_t function{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  // printf("_launch hip kernel\\n");
  void *params[] = {{ {', '.join(f"&arg{i}" for i in params)} }};
  if (gridX*gridY*gridZ > 0) {{
      HIP_CHECK(hipModuleLaunchKernel(function, gridX, gridY, gridZ, {warp_size}*num_warps, 1, 1, shared_memory, stream, params, 0));
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
    ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(ret);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    uint64_t dev_ptr;
    hipError_t status = hipPointerGetAttribute(&dev_ptr, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
    if (status == hipErrorInvalidValue) {{
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }}
    ptr_info.dev_ptr = (hipDeviceptr_t)dev_ptr;
    Py_DECREF(ret);
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
   // printf("launch\\n");
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int num_warps;
  int num_ctas;
  int clusterDimX;
  int clusterDimY;
  int clusterDimZ;
  int shared_memory;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *compiled_kernel = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &num_ctas, &clusterDimX, &clusterDimY, &clusterDimZ, &shared_memory, &_stream, &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel{', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''})) {{
    return NULL;
  }}

  if (launch_enter_hook != Py_None) {{
    PyObject_CallObject(launch_enter_hook, args);
  }}


  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (hipStream_t)_stream, (hipFunction_t)_function{', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items()) if len(signature) > 0 else ''});
  Py_END_ALLOW_THREADS;

  if (launch_exit_hook != Py_None) {{
    PyObject_CallObject(launch_exit_hook, args);
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


def get_amdgcn_bitcode_paths(gfx_arch: str):
    # print("get_amdgcn_bitcode_paths")
    gpu_arch_agnostic_bitcode_libraries = ["opencl.bc",
                                           "ocml.bc",
                                           "ockl.bc",
                                           "oclc_finite_only_off.bc",
                                           "oclc_daz_opt_on.bc",
                                           "oclc_correctly_rounded_sqrt_on.bc",
                                           "oclc_unsafe_math_off.bc",
                                           "oclc_wavefrontsize64_on.bc",
                                           "oclc_abi_version_400.bc", ]

    gfx_arch_id = re.search('gfx(\\w+)', gfx_arch).group(1).strip()

    gpu_arch_specific_bitcode_library = 'oclc_isa_version_' + gfx_arch_id + ".bc"
    current_dir = Path(__file__)
    bitcode_path_dir = os.path.join(current_dir.parent.resolve(), "lib/bitcode/")

    amdgcn_bitcode_paths = {}
    i = 0
    for bc_lib in gpu_arch_agnostic_bitcode_libraries:
        bc_path = bitcode_path_dir + bc_lib
        if os.path.exists(bc_path):
            amdgcn_bitcode_paths['library_' + str(i)] = bc_path
            i += 1
    bc_gfx_path = bitcode_path_dir + gpu_arch_specific_bitcode_library
    if os.path.exists(bc_gfx_path):
        amdgcn_bitcode_paths['library_' + str(i)] = bc_gfx_path

    # print(f"amdgcn_bitcode_paths: {amdgcn_bitcode_paths}")
    return amdgcn_bitcode_paths

def get_arch_name() -> str:
    arch_info = _triton.get_arch_info()
    gfx_arch_details = re.search('amd.*', arch_info).group(0).strip().split('--')
    arch_name_features = gfx_arch_details[1].split(':')
    return arch_name_features[0]

def gpu_matrix_core_version() -> int:
    """ Determine matrix core type available on current GPU.

        0 means no tensor cores are available
        1 corresponds to MFMA in CDNA 1 architecture
        2 corresponds to MFMA in CDNA 2 architecture
        3 corresponds to MFMA in CDNA 3 architecture
    """

    arch_info = _triton.get_arch_info()
    gfx_arch_details = re.search('amd.*', arch_info)
    if gfx_arch_details is None:
        return 0
    gfx_arch_details = gfx_arch_details.group(0).strip().split('--')
    gpu_name = gfx_arch_details[1].split(':')[0]
    if gpu_name in ['gfx908']:
        return 1
    if gpu_name in ['gfx90a']:
        return 2
    if gpu_name in ['gfx940', 'gfx941', 'gfx942']:
        return 3
    return 0

def get_amdgpu_arch_fulldetails():
    """
    get the amdgpu full ISA details for compiling:
    i.e., arch_triple: amdgcn-amd-amdhsa; arch_name: gfx906; arch_features: sramecc+:xnack-
    """
    try:
        # TODO: package rocm.cc with Triton
        arch_info = _triton.get_arch_info()
        gfx_arch_details = re.search('amd.*', arch_info).group(0).strip().split('--')
        arch_triple = gfx_arch_details[0]
        arch_name_features = gfx_arch_details[1].split(':')
        arch_name = arch_name_features[0]
        arch_features = ""

        # overwrite if provided by user
        gfx_arch = os.environ.get('MI_GPU_ARCH', arch_name)
        if gfx_arch is None:
            raise RuntimeError('gfx_arch is None (not specified)')
        mat_core_ver = gpu_matrix_core_version()
        capability = gpu_matrix_core_version() * 100
        warp_size = _triton.get_warp_size()

        return {"gfx_triple": arch_triple, "gfx_arch": gfx_arch, "gfx_features": arch_features,\
                 "capability": capability, "matrix_core_version": mat_core_ver, "warp_size": warp_size}
    except BaseException as e:
        print("Error: Attempting to get amgpu ISA Details {}".format(e))
        return None


def get_kernel_name(src: str, pattern: str) -> str:
    # print("get_kernel_name")
    '''
    Get kernel name from PTX code.
    This Kernel name is required when launching the kernel.
    '''
    # There is a name mangling in PTX codegen, so the original kernel names in Triton IR are not available in PTX/cubin.
    assert src
    for line in src.split('\n'):
        line = line.strip()
        if line.startswith(pattern):
            return line.split()[-1]


def get_arch_details(arch: dict):
    # get arch info
    gfx_arch = os.environ.get('MI_GPU_ARCH', arch["gfx_arch"])
    gfx_triple = arch["gfx_triple"]
    gfx_features = arch["gfx_features"]
    if gfx_arch is None:
        raise RuntimeError('gfx_arch is None (not specified)')

    return gfx_arch, gfx_triple, gfx_features


def update_extern_libs(extern_libs: dict, gfx_arch: str):
    # append extern_libs
    extern_libs.update(get_amdgcn_bitcode_paths(gfx_arch))
    for key in list(extern_libs):
        if extern_libs[key] == '' or extern_libs[key] is None:
            extern_libs.pop(key)

    # check extern libs
    if extern_libs:
        for name, path in extern_libs.items():
            if len(name) == 0 or len(path) == 0:
                raise RuntimeWarning(f"extern_lib has empty value, {name}: {path}")

    names = list(extern_libs.keys())
    paths = list(extern_libs.values())
    return names, paths


# passes
def ttir_to_ttgir_rocm(module: str, compute_capability: int, num_warps: int, num_stages: int):
    return _triton.translate_ttir_to_ttgir_rocm(module, compute_capability, num_warps, num_stages)


def optimize_ttgir_rocm():
    pass


def ttgir_to_llir_rocm(module: str, extern_libs: dict, arch: dict):
    names, paths = update_extern_libs(extern_libs, arch["gfx_arch"])
    llvmIR = _triton.translate_ttgir_to_llvmir(module, names, paths)
    return llvmIR


def llir_to_amdgcn_and_hsaco_rocm(module: str, arch: dict):
    '''
    Translate TritonGPU module to HSACO code based on full details of gpu architecture.
    :param mod: a TritonGPU dialect module
    :return:
        - AMDGCN code
        - Path to HSACO object
    '''
    return _triton.translate_llvmir_to_hsaco(module, arch["gfx_arch"], arch["gfx_triple"], arch["gfx_features"])


def ttir_to_amdgcn_and_hsaco(module, context, arch, num_warps, num_stages, extern_libs) -> Tuple[str, str]:
    gfx_arch, gfx_triple, gfx_features = get_arch_details(arch)
    names, paths = update_extern_libs(extern_libs, gfx_arch)
    return _triton.translate_triton_ir_to_amdgcn_and_hsaco(str(module), gfx_arch, gfx_triple, gfx_features, num_warps, num_stages, names, paths)


def llir_to_amdgcn_and_hsaco(mod: Any, gfx_arch: str, gfx_triple: str, gfx_features: str) -> Tuple[str, str]:
    '''
    Translate TritonGPU module to HSACO code based on full details of gpu architecture.
    :param mod: a TritonGPU dialect module
    :return:
        - AMDGCN code
        - Path to HSACO object
    '''
    return _triton.translate_llvmir_to_hsaco(mod, gfx_arch, gfx_triple, gfx_features)


class HIPBackend(BaseBackend):
    _cached_rocm_version_key = None

    def __init__(self, device_type: str) -> None:
        super(HIPBackend, self).__init__(device_type)
        self.driver = HIPDriver()
        self.stub_so_path = ""

    def get_version_key(self):
        if self._cached_rocm_version_key is None:
            key = compute_core_version_key()
            ### TODO: Append ROCM version here if needed

            self._cached_rocm_version_key = key
        return self._cached_rocm_version_key

    def is_standalone(self):
        return not HIP_BACKEND_MODE

    def add_stages(self, arch: dict, extern_libs: dict, stages: dict, other: dict = {}):
        if self.is_standalone():
            num_warps = arch["num_warps"]
            num_ctas = arch["num_ctas"]
            num_stages = arch["num_stages"]
            gfx_arch = arch["gfx_arch"]
            gfx_triple = arch["gfx_triple"]
            gfx_features = arch["gfx_features"]

            context = other["context"]
            warp_size = other["warp_size"]
            cluster_info = other["cluster_info"]
            enable_warp_specialization = other["enable_warp_specialization"]
            enable_persistent = other["enable_persistent"]
            optimize_epilogue = other["optimize_epilogue"]
            tma_infos = other["tma_infos"]
            waves_per_eu = other["waves_per_eu"]
            slice_k_tile = other["slice_k_tile"]
            matrix_instr_nonkdim = other["matrix_instr_nonkdim"]
            kpack = other["kpack"]

            stages["ttgir"] = (lambda path: parse_mlir_module(path, context),
                               lambda src: optimize_ttgir(ttir_to_ttgir(src, num_warps, warp_size, num_ctas, arch), num_stages, num_warps, num_ctas, arch, cluster_info, enable_warp_specialization, enable_persistent, optimize_epilogue, matrix_instr_nonkdim, slice_k_tile, kpack))
            stages["llir"] = (lambda path: Path(path).read_text(),
                              lambda src: ttgir_to_llir(src, extern_libs, arch, tma_infos, waves_per_eu))

            extern_libs.update(get_amdgcn_bitcode_paths(gfx_arch))
            for key in list(extern_libs):
                if extern_libs[key] == '' or extern_libs[key] is None:
                    extern_libs.pop(key)

            stages["amdgcn"] = (lambda path: Path(path).read_text(),
                                lambda src: llir_to_amdgcn_and_hsaco(src, gfx_arch,
                                                                     gfx_triple,
                                                                     gfx_features))
        else:
            # add stages
            stages["ttgir"] = (lambda path: Path(path).read_text(),
                               lambda src: ttir_to_ttgir_rocm(str(src), 0, arch["num_warps"], arch["num_stages"]))
            stages["llir"] = (lambda path: Path(path).read_text(),
                              lambda src: ttgir_to_llir_rocm(src, extern_libs, arch))
            stages["amdgcn"] = (lambda path: Path(path).read_text(),
                                lambda src: llir_to_amdgcn_and_hsaco_rocm(src, arch))

    def add_meta_info(self, ir, module, next_module, metadata, asm):
        pass

    def get_driver(self):
        return self.driver

    def get_stream(self, idx=None):
        return jit.get_cuda_stream()

    def get_device_properties(self, device):
        return self.driver.utils.get_device_properties(device)

    def get_current_device(self):
        return jit.get_current_device()

    def set_current_device(self, device):
        return jit.set_current_device(device)

    def get_load_binary_fn(self):
        return self.driver.utils.load_binary

    def get_kernel_bin(self):
        return "hsaco_path"

    def get_architecture_descriptor(self, **kwargs):
        # get arch
        arch = get_amdgpu_arch_fulldetails()

        # set default values
        arch["num_warps"] = 4
        arch["num_stages"] = 2
        arch["num_ctas"] = 1
        return arch

    def make_launcher_stub(self, name, signature, constants, ids):
        self.stub_so_path = make_stub(name, signature, constants, ids)
        return self.stub_so_path

    def get_shared_memory_size(self, module):
        if self.is_standalone():
            return _triton.get_shared_memory_size(module)
        else:
            return _triton.get_shared_memory_size(module)

    def get_num_warps(self, module):
        if self.is_standalone():
            return _triton.get_num_warps(module)
        else:
            return _triton.get_num_warps(module)

    def get_matrix_core_version(self):
        return gpu_matrix_core_version()
