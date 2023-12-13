from triton.common.backend import BaseBackend
from ..._C.libtriton.triton import ir, runtime
from ..._C.libtriton.triton import get_num_warps, TMAInfos, translate_triton_gpu_to_llvmir, get_shared_memory_size, add_external_libs, translate_llvmir_to_hsaco
# from ..._C.libtriton.amd_triton import amd_ir
from dataclasses import dataclass
from ...common.backend import get_cuda_version_key
from typing import Any
import hashlib
from triton.compiler.make_launcher import get_cache_manager, make_so_cache_key
import tempfile
from triton.common import _build
import os


def make_stub(name, signature, constants, ids, **kwargs):
    # name of files that are cached
    version_key = 'TODO'
    so_cache_key = make_so_cache_key(version_key, signature, constants, ids, **kwargs)
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
    params = [
        i for i in signature.keys() if i >= start_desc or (i not in constants and i not in folded_without_constexprs)
    ]
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
      PyErr_SetString(PyExc_RuntimeError, err);
   }}
}}

#define HIP_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, hipStream_t stream, hipFunction_t function{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  // printf("_launch hip kernel\\n");
  void *params[] = {{ {', '.join(f"&arg{i}" for i in params)} }};
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
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (hipStream_t)_stream, (hipFunction_t)_function{', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items()) if len(signature) > 0 else ''});

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


@dataclass(frozen=True)
class HIPOptions:
    num_warps: int = 4
    num_stages: int = 3
    num_ctas: int = 1
    extern_libs: dict = None
    cluster_dims: tuple = (1, 1, 1)
    debug: bool = False
    arch: str = None
    # TODO: deprecate when hook interface has changed
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    capability: int = None

    def __post_init__(self):
        # TODO: change API
        if isinstance(self.extern_libs, dict):
            extern_libs = tuple([(k, v) for k, v in self.extern_libs.items() if v])
            object.__setattr__(self, 'extern_libs', extern_libs)
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class HIPBackend(BaseBackend):

    def __init__(self, device_type: tuple) -> None:
        super().__init__(device_type)
        assert isinstance(device_type, tuple) and len(device_type) == 2
        assert device_type[0] == 'hip'
        assert isinstance(device_type[1], str)
        self.device_type = device_type

    def parse_options(self, opts) -> Any:
        device_type = self.device_type
        args = {'arch': device_type[1]}
        args.update({k: opts[k] for k in HIPOptions.__dataclass_fields__.keys() if k in opts})
        return HIPOptions(**args)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        pm.add_inliner_pass()
        pm.add_triton_combine_pass()
        pm.add_canonicalizer_pass()
        pm.add_reorder_broadcast_pass()
        pm.add_cse_pass()
        pm.add_licm_pass()
        pm.add_symbol_dce_pass()
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.add_convert_triton_to_tritongpu_pass(opt.num_warps, 32, 1, 90)
        pm.run(mod)
        return mod

        # pm = amd_ir.pass_manager(mod.context)
        # pm.enable_debug()
        # pm.add_tritonamdgpu_coalesce_pass()
        # pm.add_tritonamdgpu_accelerate_matmul_pass(gpu_matrix_core_version(), opt.matrix_inst_shape)
        # pm.add_tritonamdgpu_remove_layout_conversions_pass()
        # pm.add_tritonamdgpu_optimize_epilogue_pass()
        # pm.add_tritonamdgpu_optimize_dot_operands_pass()
        # if opt.num_stages == 0 and opt.matrix_core_version != 0:
        #     pm.add_tritonamdgpu_stream_pipeline_pass()
        #     pm.add_canonicalizer_pass()
        # pm.add_tritonamdgpu_pipeline_pass(opt.num_stages, opt.num_warps, opt.num_ctas, 0)
        # pm.add_tritonamdgpu_materialize_load_store_pass(opt.num_warps, 0)
        # pm.add_tritonamdgpu_optimize_dot_operands_pass()
        # pm.add_tritonamdgpu_remove_layout_conversions_pass()
        # pm.add_tritonamdgpu_decompose_conversions_pass()
        # # do we even need the old pipeliner anymore?
        # if opt.num_stages != 0:
        #     pm.add_tritonamdgpu_reorder_instructions_pass()
        # pm.add_cse_pass()
        # pm.add_symbol_dce_pass()
        # pm.run(mod)
        # return mod

    @staticmethod
    def make_llir(src, metadata, options, capability):
        metadata["num_warps"] = get_num_warps(src)
        # link libraries
        if options.extern_libs:
            names = [lib[0] for lib in options.extern_libs]
            paths = [lib[1] for lib in options.extern_libs]
            add_external_libs(src, names, paths)
        # TritonGPU -> LLVM-IR
        ret = translate_triton_gpu_to_llvmir(src, capability, TMAInfos(), runtime.TARGET.ROCDL)
        metadata["shared"] = get_shared_memory_size(src)
        return ret

    @staticmethod
    def make_hsaco(src, metadata, options):
        ret, name = translate_llvmir_to_hsaco(src, options.arch, 'amdgcn-amd-amdhsa', '')
        metadata["name"] = name
        return ret

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, 90)
        # TODO: first amdgcn, then hsaco
        # stages["amdgcn"] = lambda src, metadata: self.make_amdgcn(src, metadata, options)
        stages["hsaco"] = lambda src, metadata: self.make_hsaco(src, metadata, options)

    def hash(self):
        return f'{get_cuda_version_key()}-{self.device_type}'

    def make_launcher_stub(self, src, metadata):
        ids = {
            "ids_of_tensormaps": metadata.get("ids_of_tensormaps", tuple()), "ids_of_folded_args":
            metadata.get("ids_of_folded_args",
                         tuple()), "ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()
        }
        constants = src.constants if hasattr(src, "constants") else dict()
        # set constant
        return make_stub(src.name, src.signature, constants, ids)

    @classmethod
    def create_backend(cls, device_type: str):
        return cls(device_type)
