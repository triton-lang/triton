import functools
import operator
import os
import sysconfig
import hashlib
import subprocess
import tempfile
import triton
from pathlib import Path
from triton import knobs
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.runtime import _allocation
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver, platform_key

from triton.tools.tensor_descriptor import TensorDescriptor

dirname = os.path.dirname(os.path.realpath(__file__))
include_dir = [os.path.join(dirname, "include")]
libdevice_dir = os.path.join(dirname, "lib")
libraries = ['cuda']


@functools.lru_cache()
def libcuda_dirs():
    if env_libcuda_path := knobs.nvidia.libcuda_path:
        return [env_libcuda_path]

    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()
    # each line looks like the following:
    # libcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1
    locs = [line.split()[-1] for line in libs.splitlines() if "libcuda.so.1" in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path and not dirs:
        dirs = [dir for dir in env_ld_library_path.split(":") if os.path.exists(os.path.join(dir, "libcuda.so.1"))]
    msg = 'libcuda.so cannot found!\n'
    if locs:
        msg += 'Possible files are located at %s.' % str(locs)
        msg += 'Please create a symlink of libcuda.so to any of the files.'
    else:
        msg += 'Please make sure GPU is set up and then run "/sbin/ldconfig"'
        msg += ' (requires sudo) to refresh the linker cache.'
    assert any(os.path.exists(os.path.join(path, 'libcuda.so.1')) for path in dirs), msg
    return dirs


@functools.lru_cache()
def library_dirs():
    return [libdevice_dir, *libcuda_dirs()]


def compile_module_from_src(src, name):
    key = hashlib.sha256((src + platform_key()).encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    cache_path = cache.get_file(f"{name}{suffix}")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, library_dirs(), include_dir, libraries)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}{suffix}", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------
# Utils
# ------------------------


class CudaUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CudaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        mod = compile_module_from_src(Path(os.path.join(dirname, "driver.c")).read_text(), "cuda_utils")
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.cuOccupancyMaxActiveClusters = mod.cuOccupancyMaxActiveClusters
        self.set_printf_fifo_size = mod.set_printf_fifo_size
        self.fill_tma_descriptor = mod.fill_tma_descriptor


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "CUdeviceptr"
    if ty.startswith("tensordesc"):
        return "CUtensorMap"
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
        "nvTmaDesc": "CUtensorMap",
    }[ty]


_BASE_ARGS_FORMAT = "iiiKKppOOOOO"


def make_launcher(constants, signature):

    def _expand_signature(sig, output):
        # Expand tensordesc arguments
        if isinstance(sig, str) and sig.startswith("tensordesc"):
            output.append("nvTmaDesc")
            ndim = sig.count(",") + 1
            for _ in range(ndim):
                output.append("i32")
            for _ in range(ndim):
                output.append("i64")
        else:
            output.append(sig)

    def _flatten_signature(sig, output):
        # Flatten tuples
        if isinstance(sig, tuple):
            for x in sig:
                _flatten_signature(x, output)
        else:
            output.append(sig)

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == '*':
            return "PyObject*"
        if ty in ("constexpr", "nvTmaDesc"):
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty in ("constexpr", "nvTmaDesc"):
            return "O"
        if ty.startswith("tensordesc"):
            return "O"
        return {
            "float": "f",
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "L",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty_to_cpp(ty)]

    expand_signature = []
    for sig in signature.values():
        _expand_signature(sig, expand_signature)
    signature = {i: s for i, s in enumerate(expand_signature)}

    args_format = ''.join([format_of(ty) for ty in signature.values()])
    format = _BASE_ARGS_FORMAT + args_format

    flat_signature = []
    for sig in signature.values():
        _flatten_signature(sig, flat_signature)
    signature = {i: s for i, s in enumerate(flat_signature)}
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors, such as tensor descriptors for CUDA.
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items() if ty != "constexpr")
    internal_args_list = []
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"ptr_info{i}.dev_ptr")
        elif ty == "nvTmaDesc":
            # Note: we have to dereference the pointer
            internal_args_list.append(f"*tma_ptr{i}")
        elif ty != "constexpr":
            internal_args_list.append(f"_arg{i}")
    params = range(len(signature))

    # generate glue code
    newline = '\n  '
    ptr_decls = [
        f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;"
        for i, ty in signature.items()
        if ty[0] == "*"
    ]
    tma_decls = [
        f"CUtensorMap* tma_ptr{i} = getTmaDesc(_arg{i}); if (!tma_ptr{i}) return NULL;" for i, ty in signature.items()
        if ty == "nvTmaDesc"
    ]
    params = [f"&arg{i}" for i, ty in signature.items() if ty != "constexpr"]
    params.append("&global_scratch")
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
  void* handle = dlopen("libcuda.so.1", RTLD_LAZY);
  if (!handle) {{
    PyErr_SetString(PyExc_RuntimeError, "Failed to open libcuda.so.1");
    return NULL;
  }}
  // Clear any existing error
  dlerror();
  cuLaunchKernelEx_t cuLaunchKernelExHandle = (cuLaunchKernelEx_t)dlsym(handle, "cuLaunchKernelEx");
  // Check for errors
  const char *dlsym_error = dlerror();
  if (dlsym_error) {{
    PyErr_SetString(PyExc_RuntimeError, "Failed to retrieve cuLaunchKernelEx from libcuda.so.1");
    return NULL;
  }}
  return cuLaunchKernelExHandle;
}}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int launch_cooperative_grid, int launch_pdl, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, CUstream stream, CUfunction function, CUdeviceptr global_scratch{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  void *params[] = {{ {', '.join(params)} }};
  if (gridX*gridY*gridZ > 0) {{
    // 4 attributes that we can currently pass maxmimum
    CUlaunchAttribute launchAttr[4];
    static cuLaunchKernelEx_t cuLaunchKernelExHandle = NULL;
    if (cuLaunchKernelExHandle == NULL) {{
      cuLaunchKernelExHandle = getLaunchKernelExHandle();
    }}
    CUlaunchConfig config;
    config.gridDimX = gridX;
    config.gridDimY = gridY;
    config.gridDimZ = gridZ;

    if (num_ctas != 1) {{
      config.gridDimX *= clusterDimX;
      config.gridDimY *= clusterDimY;
      config.gridDimZ *= clusterDimZ;
    }}

    config.blockDimX = 32 * num_warps;
    config.blockDimY = 1;
    config.blockDimZ = 1;
    config.sharedMemBytes = shared_memory;
    config.hStream = stream;
    config.attrs = launchAttr;
    int num_attrs = 0;

    if (launch_pdl != 0) {{
      CUlaunchAttribute pdlAttr = {{ .id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, .value = 1}};
      launchAttr[num_attrs] = pdlAttr;
      ++num_attrs;
    }}

    if (launch_cooperative_grid != 0) {{
      CUlaunchAttribute coopAttr = {{ .id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE, .value = 1}};
      launchAttr[num_attrs] = coopAttr;
      ++num_attrs;
    }}

    if (num_ctas != 1) {{
      CUlaunchAttribute clusterAttr = {{}};
      clusterAttr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      clusterAttr.value.clusterDim.x = clusterDimX;
      clusterAttr.value.clusterDim.y = clusterDimY;
      clusterAttr.value.clusterDim.z = clusterDimZ;
      launchAttr[num_attrs] = clusterAttr;
      ++num_attrs;

      CUlaunchAttribute clusterSchedulingAttr = {{}};
      clusterSchedulingAttr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      clusterSchedulingAttr.value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
      launchAttr[num_attrs] = clusterSchedulingAttr;
      ++num_attrs;
    }}

    config.numAttrs = num_attrs;

    CUDA_CHECK(cuLaunchKernelExHandle(&config, function, params, 0));
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
    }} else if (status != CUDA_SUCCESS) {{
        CUDA_CHECK(status);  // Catch any other cuda API errors
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

static inline CUtensorMap* getTmaDesc(PyObject *obj) {{
  if (sizeof(CUtensorMap*) != 8) {{
    PyErr_SetString(PyExc_SystemError, "getTmaDesc() requires 64-bit compilation");
    return NULL;
  }}

  PyObject *method_handle = PyObject_GetAttrString(obj, "tma_desc_cpu_ptr");
  if (!method_handle) {{
    PyErr_SetString(PyExc_TypeError, "tma_desc_cpu_ptr() method does not exist");
    return NULL;
  }}

  PyObject *empty_tuple = PyTuple_New(0);
  if (!empty_tuple) {{
    Py_DECREF(method_handle);
    PyErr_SetString(PyExc_SystemError, "Internal Python error!");
    return NULL;
  }}
  PyObject *method_ret = PyObject_Call(method_handle, empty_tuple, NULL);
  Py_DECREF(empty_tuple);
  Py_DECREF(method_handle);
  if (!method_ret) {{
    PyErr_SetString(PyExc_SystemError, "Internal Python error!");
    return NULL;
  }}

  if (!PyLong_Check(method_ret)) {{
    PyErr_SetString(PyExc_TypeError, "tma_desc_cpu_ptr() must return 64-bit int");
    Py_DECREF(method_ret);
    return NULL;
  }}

  uint64_t ptr_as_uint = PyLong_AsUnsignedLongLong(method_ret);
  Py_DECREF(method_ret);
  if (!ptr_as_uint) {{
    PyErr_SetString(PyExc_ValueError, "received NULL ptr from tma_desc_cpu_ptr()");
    return NULL;
  }}
  if (ptr_as_uint % 64 != 0) {{
    PyErr_SetString(PyExc_ValueError, "tma_desc_cpu_ptr() must be 64-byte aligned");
    return NULL;
  }}

  return (CUtensorMap*)(ptr_as_uint);
}}

static void ensureCudaContext() {{
  CUcontext pctx;
  CUDA_CHECK(cuCtxGetCurrent(&pctx));
  if (!pctx) {{
    // Ensure device context.
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&pctx, device));
    CUDA_CHECK(cuCtxSetCurrent(pctx));
  }}
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  // ensure cuda context is valid before calling any CUDA APIs, e.g. before getPointer calls cuPointerGetAttributes
  ensureCudaContext();

  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int launch_cooperative_grid;
  int launch_pdl;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *global_scratch_obj = NULL;
  {newline.join([f"{_extracted_type(ty)} _arg{i};" for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                                           &_stream, &_function, &launch_cooperative_grid, &launch_pdl, &global_scratch_obj,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook{args_list})) {{
    return NULL;
  }}

  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
    return NULL;
  }}

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }}

  CUdeviceptr global_scratch = 0;
  if (global_scratch_obj != Py_None) {{
    DevicePtrInfo global_scratch_info = getPointer(global_scratch_obj, -1);
    if (!global_scratch_info.valid) {{
      return NULL;
    }}
    global_scratch = global_scratch_info.dev_ptr;
  }}

  // raise exception asap
  {newline.join(ptr_decls)}
  {newline.join(tma_decls)}
  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, launch_cooperative_grid, launch_pdl, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (CUstream)_stream, (CUfunction)_function, global_scratch{', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''});
  Py_END_ALLOW_THREADS;
  if (PyErr_Occurred()) {{
    return NULL;
  }}

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }}

  Py_RETURN_NONE;
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


class TmaDescKernelParam:
    TMA_DESC_SIZE = 128

    def __init__(self):
        import torch
        self.desc = torch.empty(self.TMA_DESC_SIZE, dtype=torch.uint8, device="cpu")

    # Return a CUtensorMap* pointer in host memory
    def tma_desc_cpu_ptr(self):
        return self.desc.data_ptr()


# The TMA dtype enum values are slightly different on host vs device...
TMA_DTYPE_DEVICE_TO_HOST = dict((i, i) for i in range(16))
TMA_DTYPE_DEVICE_TO_HOST[8] = 10
TMA_DTYPE_DEVICE_TO_HOST[9] = 8
TMA_DTYPE_DEVICE_TO_HOST[10] = 9


def make_tensordesc_arg(arg, metadata):
    assert isinstance(arg, TensorDescriptor)
    swizzle = metadata["swizzle"]
    elem_size = metadata["elem_size"]
    elem_type = metadata["elem_type"]
    block_size = metadata["block_size"]
    fp4_padded = metadata["fp4_padded"]

    data_ptr = arg.base.data_ptr()
    shape = arg.shape
    strides = arg.strides
    assert strides[-1] == 1

    desc = TmaDescKernelParam()
    result = [desc, *shape, *strides]

    if fp4_padded:
        shape = list(shape)
        shape[-1] *= 2
    triton.runtime.driver.active.utils.fill_tma_descriptor(
        desc.tma_desc_cpu_ptr(),
        data_ptr,
        swizzle,
        elem_size,
        TMA_DTYPE_DEVICE_TO_HOST[elem_type],
        block_size,
        shape,
        strides,
    )
    return result


def wrap_handle_tensordesc(launcher, tensordesc_meta):
    if not tensordesc_meta:
        return launcher

    def inner(*args):
        meta_args = args[:len(_BASE_ARGS_FORMAT)]
        raw_kernel_args = args[len(_BASE_ARGS_FORMAT):]
        tensordesc_idx = 0
        final_args = []
        for i, arg in enumerate(raw_kernel_args):
            if isinstance(arg, TensorDescriptor):
                meta = tensordesc_meta[tensordesc_idx]
                tensordesc_idx += 1
                final_args.extend(make_tensordesc_arg(arg, meta))
            else:
                final_args.append(arg)
        assert tensordesc_idx == len(tensordesc_meta)
        return launcher(*meta_args, *final_args)

    return inner


class CudaLauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        src = make_launcher(constants, signature)
        mod = compile_module_from_src(src, "__triton_launcher")
        self.num_ctas = functools.reduce(operator.mul, metadata.cluster_dims, 1)
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)
        if tensordesc_meta is not None:
            self.launch = wrap_handle_tensordesc(mod.launch, tensordesc_meta)
        else:
            self.launch = mod.launch
        self.global_scratch_size = metadata.global_scratch_size
        self.global_scratch_align = metadata.global_scratch_align
        self.launch_cooperative_grid = metadata.launch_cooperative_grid
        self.launch_pdl = metadata.launch_pdl

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        if self.global_scratch_size > 0:
            grid_size = gridX * gridY * gridZ
            alloc_size = grid_size * self.num_ctas * self.global_scratch_size
            global_scratch = _allocation._allocator(alloc_size, self.global_scratch_align, stream)
        else:
            global_scratch = None
        self.launch(gridX, gridY, gridZ, stream, function, self.launch_cooperative_grid, self.launch_pdl,
                    global_scratch, *args)


class CudaDriver(GPUDriver):

    def __init__(self):
        self.utils = CudaUtils()  # TODO: make static
        self.launcher_cls = CudaLauncher
        super().__init__()

    def get_current_target(self):
        device = self.get_current_device()
        capability = self.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
        warp_size = 32
        return GPUTarget("cuda", capability, warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("cuda", self.get_current_device())

    def get_device_interface(self):
        import torch
        return torch.cuda

    @staticmethod
    def is_active():
        try:
            import torch
            return torch.cuda.is_available() and (torch.version.hip is None)
        except ImportError:
            return False

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')

    def clear_cache(self, cache):
        cache.zero_()
