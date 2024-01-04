from triton.third_party.compiler import BaseBackend
from triton.runtime.cache import get_cache_manager, make_so_cache_key
from triton.common import _build
from triton._C.libtriton import ir, passes, nvidia, llvm
from triton.runtime import driver
from dataclasses import dataclass
import functools
from typing import Any
import hashlib
import re
import tempfile
import signal
import os
import subprocess
from pathlib import Path

# ------------- TMA stuff ----------------#
#
# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


def generate_cu_signature(constants, signature, ids):
    # CUtensorMap*s are always the last arguments
    num_regular_signatures = max(signature.keys()) + 1 if len(signature) > 0 else 0
    if ids["ids_of_tensormaps"] is not None:
        for i, _ in enumerate(ids["ids_of_tensormaps"]):
            signature[num_regular_signatures + i] = '*CUtensorMap'
    return signature, num_regular_signatures


def dummy_tensormaps_info(n=2):
    ret = []
    for i in range(n):
        ret.append(InfoFromBackendForTensorMap(dummy=True))
    return ret


def parse_tma_info(infos, ids_of_folded_args):
    ret = []
    for info in infos:
        e = InfoFromBackendForTensorMap(infos=info)
        e.ids_of_folded_args = ids_of_folded_args
        ret.append(e)
    return ret


def get_tma_mapping(tensormaps_info):
    ret = {}
    if tensormaps_info is not None:
        for i, e in enumerate(tensormaps_info):
            ret.update(e.get_address_tma_mapping())
    else:
        ret = None
    return ret


def get_ids_of_tensormaps(tensormaps_info):
    ret = None
    # order is not relevant
    if tensormaps_info is not None:
        ret = [e.get_id_of_tensormap() for e in tensormaps_info]
    return ret


# decouple information for tensormap from backend
# please ignore the naming style, xx_yy is compiler.py style, xxYy is to comply with cuda tensormap style
# mixing style is for readability
class InfoFromBackendForTensorMap:
    N = 2
    n = 0
    ntma = 0

    def __init__(self, infos=None, dummy=False):
        self.dummy = dummy
        self.ids_of_folded_args = ()
        if not dummy and not isinstance(infos, dict):
            self._extract_info_from_backend(infos)
        elif not dummy and isinstance(infos, dict):
            self._extract_info_from_dict(infos)
        elif dummy:
            self._dummy()

    def _dummy(self):
        assert InfoFromBackendForTensorMap.n < InfoFromBackendForTensorMap.N
        if InfoFromBackendForTensorMap.n == 0:
            self.tensorDataType = driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_FLOAT16"]
            self.tensorRank = 4
            self.globalAddressArgIdx = 0
            self.globalStridesArgIdx = [7, 6, -1, -1]
            self.globalDimsArgIdx = [5, 3, -1, -1]
            self.boxDims = [16, 64, 1, 1]
            self.elementStrides = [1, 1, 1, 1]
            self.interleave = driver.utils.CUtensorMapInterleave["CU_TENSOR_MAP_INTERLEAVE_NONE"]
            self.swizzle = driver.utils.CUtensorMapSwizzle["CU_TENSOR_MAP_SWIZZLE_32B"]
            self.l2Promotion = driver.utils.CUtensorMapL2promotion["CU_TENSOR_MAP_L2_PROMOTION_L2_128B"]
            self.TMADescArgIdx = 11
            self.oobFill = driver.utils.CUtensorMapFloatOOBfill["CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE"]
            InfoFromBackendForTensorMap.n += 1
            return
        if InfoFromBackendForTensorMap.n == 1:
            self.tensorDataType = driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_FLOAT16"]
            self.tensorRank = 4
            self.globalAddressArgIdx = 1
            self.globalStridesArgIdx = [7, 6, -1, -1]
            self.globalDimsArgIdx = [5, 3, -1, -1]
            self.boxDims = [16, 64, 1, 1]
            self.elementStrides = [1, 1, 1, 1]
            self.interleave = driver.utils.CUtensorMapInterleave["CU_TENSOR_MAP_INTERLEAVE_NONE"]
            self.swizzle = driver.utils.CUtensorMapSwizzle["CU_TENSOR_MAP_SWIZZLE_32B"]
            self.l2Promotion = driver.utils.CUtensorMapL2promotion["CU_TENSOR_MAP_L2_PROMOTION_L2_128B"]
            self.TMADescArgIdx = 12
            self.oobFill = driver.utils.CUtensorMapFloatOOBfill["CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE"]
            InfoFromBackendForTensorMap.n += 1
            return

    def _extract_info_from_backend(self, infos):
        self.tensorDataType = infos.tensorDataType
        self.tensorRank = infos.tensorRank
        self.globalAddressArgIdx = infos.globalAddressArgIdx
        self.globalStridesArgIdx = infos.globalStridesArgIdx
        self.globalDimsArgIdx = infos.globalDimsArgIdx
        self.boxDims = infos.boxDims
        self.elementStrides = infos.elementStrides
        self.interleave = infos.interleave
        self.swizzle = infos.swizzle
        self.l2Promotion = infos.l2Promotion
        self.oobFill = infos.oobFill
        self.TMADescArgIdx = infos.TMADescArgIdx

    # dict could be from cached metadata json
    def _extract_info_from_dict(self, infos: dict):
        self.tensorDataType = infos['tensorDataType']
        self.tensorRank = infos['tensorRank']
        self.globalAddressArgIdx = infos['globalAddressArgIdx']
        self.globalStridesArgIdx = infos['globalStridesArgIdx']
        self.globalDimsArgIdx = infos['globalDimsArgIdx']
        self.boxDims = infos['boxDims']
        self.elementStrides = infos['elementStrides']
        self.interleave = infos['interleave']
        self.swizzle = infos['swizzle']
        self.l2Promotion = infos['l2Promotion']
        self.oobFill = infos['oobFill']
        self.TMADescArgIdx = infos['TMADescArgIdx']

    def get_address_tma_mapping(self):
        return {self.globalAddressArgIdx: self.TMADescArgIdx + len(self.ids_of_folded_args)}

    def get_id_of_tensormap(self):
        return self.TMADescArgIdx + len(self.ids_of_folded_args)

    def getTMADescArgIdx(self):
        return self.TMADescArgIdx

    # dtype:cuda.CUtensorMapDataType | int
    def bytes_from_type(self, dtype):
        return {
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_UINT8"]: 1,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_UINT16"]: 2,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_UINT32"]: 4,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_INT32"]: 4,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_UINT64"]: 8,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_INT64"]: 8,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_FLOAT16"]: 2,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_FLOAT32"]: 4,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_FLOAT64"]: 8,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_BFLOAT16"]: 2,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ"]: 4,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_TFLOAT32"]: 4,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ"]: 4
        }[dtype]

    def getTensorMapDataType(self):
        return self.tensorDataType

    def getInterleave(self):
        return self.interleave

    def getSwizzle(self):
        return self.swizzle

    def getL2Promotion(self):
        return self.l2Promotion

    def getOobFill(self):
        return self.oobFill

    def getTensorRank(self):
        return self.tensorRank

    def getBoxDims(self):
        return self.boxDims

    def getElementStrides(self):
        return self.elementStrides

    def getGlobalAddress(self, args):
        idx = self.getOriginArgIdx(self.globalAddressArgIdx, args)
        return args[idx]

    # args, captured kernel args in runtime
    def getGlobalDims(self, args):
        shape = []
        for e in self.globalDimsArgIdx:
            t = 1
            # < 0 means folded arg or constant (-1 - value)
            # -1 means extended dim which is 1, -2 means folded arg with constant 1 (-1 - value)
            if e == -1:
                t = 1
            elif e < 0 and e != -1:
                t = -e - 1
            else:
                idx = self.getOriginArgIdx(e, args)
                t = args[idx]
            shape.append(t)
        return shape

    def getGlobalStrides(self, args):
        t_globalDims = [int(e) for e in self.getGlobalDims(args)]
        t_globalStridesArgIdx = self.globalStridesArgIdx.copy()
        strides_in_elements = []
        # todo: get all stride from backend even in extended mode
        for i in range(self.tensorRank):
            t = 1
            if t_globalStridesArgIdx[i] == -1:
                for ii in range(i):
                    t *= t_globalDims[ii]
            # -2 means the sride in arguments is folded constant 1, we don't use 1 because it can not be distinguished from index 1
            elif t_globalStridesArgIdx[i] < 0:
                t = -1 - t_globalStridesArgIdx[i]
            else:
                new_idx = self.getOriginArgIdx(t_globalStridesArgIdx[i], args)
                t = args[new_idx]

            strides_in_elements.append(t)

        strides_in_elements = strides_in_elements[1:]
        strides_in_bytes = [e * self.bytes_from_type(self.tensorDataType) for e in strides_in_elements]
        return strides_in_bytes

    def getOriginArgIdx(self, idx, args):
        if self.ids_of_folded_args:
            ids_before_folding_arg = [i for i in range(len(args)) if i not in self.ids_of_folded_args]
            return ids_before_folding_arg[idx]
        else:
            return idx

    def tensormap(self, args):
        return driver.utils.cuTensorMapEncodeTiled(
            self.getTensorMapDataType(),
            self.getTensorRank(),
            self.getGlobalAddress(args),
            self.getGlobalDims(args),
            self.getGlobalStrides(args),
            self.getBoxDims(),
            self.getElementStrides(),
            self.getInterleave(),
            self.getSwizzle(),
            self.getL2Promotion(),
            self.getOobFill(),
        )

    # make hashable to use as partial key in cache
    def __hash__(self):
        return hash((self.ids_of_folded_args, self.globalAddressArgIdx, tuple(self.globalDimsArgIdx),
                     tuple(self.globalStridesArgIdx), self.tensorDataType, self.tensorRank, tuple(self.boxDims),
                     tuple(self.elementStrides), self.interleave, self.swizzle, self.l2Promotion, self.oobFill))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.ids_of_folded_args, self.globalAddressArgIdx, self.globalDimsArgIdx, self.globalStridesArgIdx,
                self.tensorDataType, self.tensorRank, self.boxDims, self.elementStrides, self.interleave, self.swizzle,
                self.l2Promotion,
                self.oobFill) == (other.ids_of_folded_args, other.globalAddressArgIdx, other.globalDimsArgIdx,
                                  other.globalStridesArgIdx, other.tensorDataType, other.tensorRank, other.boxDims,
                                  other.elementStrides, other.interleave, other.swizzle, other.l2Promotion,
                                  other.oobFill)


# ----------------------------------------------------

# ----- source code generation --------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "CUdeviceptr"
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


def make_launcher(constants, signature, ids):
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors, such as tensor descriptors for CUDA.
    signature, desc_start_idx = generate_cu_signature(constants, signature, ids)
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
        i for i in signature.keys()
        if i >= desc_start_idx or (i not in constants and i not in folded_without_constexprs)
    ]
    src = f"""
#include \"cuda.h\"
#include <stdbool.h>
#include <Python.h>
#include <dlfcn.h>

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
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }}
}}

#define CUDA_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

typedef CUresult (*cuLaunchKernelEx_t)(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra);

static cuLaunchKernelEx_t getLaunchKernelExHandle() {{
  // Open the shared library
  void* handle = dlopen("libcuda.so", RTLD_LAZY);
  if (!handle) {{
    PyErr_SetString(PyExc_RuntimeError, "Failed to open libcuda.so");
    return NULL;
  }}
  // Clear any existing error
  dlerror();
  cuLaunchKernelEx_t cuLaunchKernelExHandle = (cuLaunchKernelEx_t)dlsym(handle, "cuLaunchKernelEx");
  // Check for errors
  const char *dlsym_error = dlerror();
  if (dlsym_error) {{
    PyErr_SetString(PyExc_RuntimeError, "Failed to retrieve cuLaunchKernelEx from libcuda.so");
    return NULL;
  }}
  return cuLaunchKernelExHandle;
}}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, CUstream stream, CUfunction function{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  void *params[] = {{ {', '.join(f"&arg{i}" for i in params)} }};
  if (gridX*gridY*gridZ > 0) {{
    if (num_ctas == 1) {{
      CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, params, 0));
    }} else {{
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
      if (cuLaunchKernelExHandle == NULL) {{
        cuLaunchKernelExHandle = getLaunchKernelExHandle();
      }}
      CUDA_CHECK(cuLaunchKernelExHandle(&config, function, params, 0));
    }}
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
  ptr_info.valid = false;
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
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

  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {{
    return NULL;
  }}


  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (CUstream)_stream, (CUfunction)_function{', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items()) if len(signature) > 0 else ''});
  Py_END_ALLOW_THREADS;
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


# ----------------------------------------------------


@functools.lru_cache()
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


@dataclass(frozen=True)
class CUDAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    cluster_dims: tuple = (1, 1, 1)
    ptx_version: int = None
    enable_warp_specialization: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False

    def __post_init__(self):
        default_libdir = Path(__file__).parent.parent.parent / 'third_party' / 'cuda' / 'lib'
        extern_libs = dict() if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            extern_libs['libdevice'] = str(default_libdir / 'libdevice.10.bc')
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class CUDABackend(BaseBackend):

    @staticmethod
    def supports_target(target: tuple):
        return target[0] == 'cuda'

    def __init__(self, target: tuple) -> None:
        super().__init__(target)
        self.capability = target[1]
        assert isinstance(self.capability, int)

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in CUDAOptions.__dataclass_fields__.keys() if k in opts}
        args["allow_fp8e4nv"] = self.capability >= 89
        args["max_num_imprecise_acc_default"] = 2**30 if self.capability == 90 else 0
        return CUDAOptions(**args)

    def load_dialects(ctx):
        nvidia.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, capability):
        cluster_info = nvidia.ClusterInfo()
        if opt.cluster_dims is not None:
            cluster_info.clusterDimX = opt.cluster_dims[0]
            cluster_info.clusterDimY = opt.cluster_dims[1]
            cluster_info.clusterDimZ = opt.cluster_dims[2]
        # TTIR -> TTGIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, opt.num_warps, 32, opt.num_ctas, capability)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
        nvidia.passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
        nvidia.passes.ttgpuir.add_rewrite_tensor_pointer(pm, capability)
        nvidia.passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_accelerate_matmul(pm, capability)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        if opt.optimize_epilogue:
            passes.ttgpuir.add_optimize_epilogue(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm)
        passes.common.add_cse(pm)
        # `num_warps` does not mean the total number of warps of a CTA when
        # warp specialization is enabled.
        # it's the responsibility of the compiler to figure out the exact
        # `num_warps` to use.
        # TODO: support the case where `num_warps` from user is not 4.
        ws_enabled = False
        if capability // 10 >= 9 and opt.enable_warp_specialization and opt.num_warps == 4:
            nvidia.passes.ttnvgpuir.add_wsfeasibility_checking(pm, capability)
            pm.run(mod)
            ws_enabled = nvidia.passes.ttnvgpuir.is_ws_supported(mod)
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
        metadata["ws_enabled"] = ws_enabled
        if ws_enabled:
            nvidia.passes.ttnvgpuir.add_wsdecomposing(pm, capability)
            nvidia.passes.ttnvgpuir.add_wspipeline(pm, opt.num_stages, opt.num_warps, capability)
            nvidia.passes.ttnvgpuir.add_wsmutex(pm, capability)
            nvidia.passes.ttnvgpuir.add_wsmaterialization(pm, capability)
            passes.common.add_licm(pm)
            passes.common.add_cse(pm)
        else:
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, opt.num_warps, opt.num_ctas, capability)
        nvidia.passes.ttnvgpuir.add_materialize_load_store(pm, opt.num_warps, capability)
        if capability // 10 <= 8:
            passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_decompose_conversions(pm)
        nvidia.passes.ttnvgpuir.add_wsfixup_missing_attrs(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if capability // 10 >= 9:
            nvidia.passes.ttnvgpuir.add_fence_insertion(pm)
        nvidia.passes.ttnvgpuir.add_wsfixup_missing_attrs(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
        return mod

    @staticmethod
    def make_llir(src, metadata, options, capability):
        # warp-specialization mutates num_warps
        num_warp_groups = src.get_int_attr("triton_gpu.num-warp-groups-per-cta")
        if num_warp_groups is not None:
            metadata["num_warps"] *= num_warp_groups
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        tma_infos = nvidia.TMAInfos()
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, tma_infos)
        if metadata["ws_enabled"]:
            passes.common.add_licm(pm)
            passes.common.add_cse(pm)
        nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)
        pm.run(mod)
        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        nvidia.init_llvm()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        nvidia.set_nvvm_reflect_ftz(llvm_mod)
        if options.extern_libs:
            for name, path in options.extern_libs:
                llvm.link_extern_lib(llvm_mod, path)
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
        # Get some metadata
        if len(tma_infos) > 0:
            metadata["tensormaps_info"] = parse_tma_info(tma_infos, metadata["ids_of_folded_args"])
            for i, _ in enumerate(metadata["tensormaps_info"]):
                metadata["tensormaps_info"][i].ids_of_folded_args = metadata["ids_of_folded_args"]
        metadata["ids_of_tensormaps"] = get_ids_of_tensormaps(metadata.get("tensormaps_info", None))
        metadata["shared"] = src.get_int_attr("triton_gpu.shared")
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_ptx(src, metadata, opt, capability):
        proc = 'sm_90a' if capability == 90 else f'sm_{capability}'
        ret = llvm.translate_to_asm(src, 'nvptx64-nvidia-cuda', proc, '', ['nvptx-short-ptr'], opt.enable_fp_fusion,
                                    False)
        # Find kernel names (there should only be one)
        names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
        assert len(names) == 1
        metadata["name"] = names[0]
        # post-process
        ptx_version = opt.ptx_version
        if ptx_version is None:
            _, cuda_version = path_to_ptxas()
            ptx_version = ptx_get_version(cuda_version)
        ptx_version = f'{ptx_version//10}.{ptx_version%10}'
        ret = re.sub(r'\.version \d+\.\d+', f'.version {ptx_version}', ret, flags=re.MULTILINE)
        # Remove the debug flag that prevents ptxas from optimizing the code
        ret = re.sub(r",\s*debug|debug,\s*", "", ret)
        return ret

    @staticmethod
    def make_cubin(src, metadata, opt, capability):
        ptxas, _ = path_to_ptxas()
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc, \
            tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
            fsrc.write(src)
            fsrc.flush()
            fbin = fsrc.name + '.o'

            line_info = '' if os.environ.get('TRITON_DISABLE_LINE_INFO') else ' -lineinfo'
            fmad = '' if opt.enable_fp_fusion else ' --fmad=false'
            suffix = 'a ' if capability == 90 else ' '
            cmd = f'{ptxas}{line_info}{fmad} -v --gpu-name=sm_{capability}{suffix}{fsrc.name} -o {fbin} 2> {flog.name}'

            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                if e.returncode == 255:
                    raise RuntimeError(f'Internal Triton PTX codegen error: \n{log}')
                elif e.returncode == 128 + signal.SIGSEGV:
                    raise RuntimeError(
                        f'Please run `ptxas {fsrc.name}` to confirm that this is a bug in `ptxas`\n{log}')
                else:
                    raise RuntimeError(f'`ptxas` failed with error code {e.returncode}: \n{log}')
            finally:
                if os.path.exists(fsrc.name):
                    os.remove(fsrc.name)
                if os.path.exists(flog.name):
                    os.remove(flog.name)

            with open(fbin, 'rb') as f:
                cubin = f.read()
            if os.path.exists(fbin):
                os.remove(fbin)
        return cubin

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.capability)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, self.capability)
        stages["ptx"] = lambda src, metadata: self.make_ptx(src, metadata, options, self.capability)
        stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.capability)

    @functools.lru_cache()
    def hash(self):
        version = subprocess.check_output([path_to_ptxas()[0], "--version"])
        return f'{version}-{self.capability}'

    def make_launcher(self, src, metadata):
        ids = {
            "ids_of_folded_args": metadata.get("ids_of_folded_args", tuple()), "ids_of_const_exprs":
            src.fn.constexprs if hasattr(src, "fn") else tuple()
        }
        constants = src.constants if hasattr(src, "constants") else dict()
        key = make_so_cache_key('', src.signature, constants, ids)
        src = make_launcher(constants, src.signature, ids)
        return key, src