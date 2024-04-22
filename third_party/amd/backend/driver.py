import functools
import os
import hashlib
import subprocess
import tempfile
from pathlib import Path
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver

dirname = os.path.dirname(os.path.realpath(__file__))
include_dir = [os.path.join(dirname, "include")]


@functools.lru_cache()
def _get_path_to_hip_runtime_dylib():
    lib_name = "libamdhip64.so"

    # If we are told explicitly what HIP runtime dynamic library to use, obey that.
    env_libhip_path = os.getenv("TRITON_LIBHIP_PATH")
    if env_libhip_path:
        if env_libhip_path.endswith(lib_name) and os.path.exists(env_libhip_path):
            return env_libhip_path
        raise RuntimeError(f"TRITON_LIBHIP_PATH '{env_libhip_path}' does not point to a valid {lib_name}")

    paths = []

    import site
    # First search the HIP runtime dynamic library packaged with PyTorch. It's very likely
    # that we run Triton together with PyTorch. This makes sure we use the same dynamic
    # library to avoid version mismatch.
    for path in site.getsitepackages():
        path = os.path.join(path, "torch", "lib", lib_name)
        if os.path.exists(path):
            return path
        paths.append(path)

    # Then try to see if developer provides a HIP runtime dynamic library using LD_LIBARAY_PATH.
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path:
        for d in env_ld_library_path.split(":"):
            f = os.path.join(d, lib_name)
            if os.path.exists(f):
                return f
            paths.append(f)

    # Afterwards try to search the loader dynamic library resolution paths.
    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()
    # each line looks like the following:
    # libamdhip64.so.6 (libc6,x86-64) => /opt/rocm-6.0.2/lib/libamdhip64.so.6
    # libamdhip64.so (libc6,x86-64) => /opt/rocm-6.0.2/lib/libamdhip64.so
    locs = [line.split()[-1] for line in libs.splitlines() if line.strip().endswith(lib_name)]
    for loc in locs:
        if os.path.exists(loc):
            return loc
        paths.append(loc)

    # As a last resort, guess if we have it in some common installation path.
    common_install_path = os.path.join('/opt/rocm/lib/', lib_name)
    if os.path.exists(common_install_path):
        return common_install_path
    paths.append(common_install_path)

    raise RuntimeError(f"cannot locate {lib_name} after attempted paths {paths}")


def compile_module_from_src(src, name):
    key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, [], include_dir, [])
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class HIPUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(HIPUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        libhip_path = _get_path_to_hip_runtime_dylib()
        src = Path(os.path.join(dirname, "driver.c")).read_text()
        # Just do a simple search and replace here instead of templates or format strings.
        # This way we don't need to escape-quote C code curly brackets and we can replace
        # exactly once.
        src = src.replace('/*py_libhip_search_path*/', libhip_path, 1)
        mod = compile_module_from_src(src, "hip_utils")
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties


# -------------------- Launcher ----------------------------
def ty_to_cpp(ty):
    if ty[0] == '*':
        return "hipDeviceptr_t"
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


def make_launcher(constants, signature, ids, warp_size):
    start_desc = len(signature)
    #signature = generate_cu_signature(constants, signature, ids)
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    def _extracted_type(ty):
        if ty[0] == '*':
            return "PyObject*"
        return {
            'i1': 'int32_t',
            'i8': 'int8_t',
            'i16': 'int16_t',
            'i32': 'int32_t',
            'i64': 'int64_t',
            'u1': 'uint32_t',
            'u8': 'uint8_t',
            'u16': 'uint16_t',
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
    format = "iiiKKOOOO" + args_format
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    libhip_path = _get_path_to_hip_runtime_dylib()

    # generate glue code
    params = [i for i in signature.keys() if i not in constants]
    src = f"""
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <Python.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <dlfcn.h>

// The list of paths to search for the HIP runtime library. The caller Python
// code should substitute the search path placeholder.
static const char *hipLibSearchPaths[] = {{"{libhip_path}"}};

// The list of HIP dynamic library symbols and their signature we are interested
// in this file.
#define HIP_SYMBOL_LIST(FOR_EACH_ERR_FN, FOR_EACH_STR_FN)                     \\
  FOR_EACH_STR_FN(hipGetErrorString, hipError_t hipError)                     \\
  FOR_EACH_ERR_FN(hipModuleLaunchKernel, hipFunction_t f,                     \\
                  unsigned int gridDimX, unsigned int gridDimY,               \\
                  unsigned int gridDimZ, unsigned int blockDimX,              \\
                  unsigned int blockDimY, unsigned int blockDimZ,             \\
                  unsigned int sharedMemBytes, hipStream_t stream,            \\
                  void **kernelParams, void **extra)                          \\
  FOR_EACH_ERR_FN(hipPointerGetAttribute, void *data,                         \\
                  hipPointer_attribute attribute, hipDeviceptr_t ptr)

// The HIP symbol table for holding resolved dynamic library symbols.
struct HIPSymbolTable {{
#define DEFINE_EACH_ERR_FIELD(hipSymbolName, ...)                             \\
  hipError_t (*hipSymbolName)(__VA_ARGS__);
#define DEFINE_EACH_STR_FIELD(hipSymbolName, ...)                             \\
  const char *(*hipSymbolName)(__VA_ARGS__);

  HIP_SYMBOL_LIST(DEFINE_EACH_ERR_FIELD, DEFINE_EACH_STR_FIELD)
}};

static struct HIPSymbolTable hipSymbolTable;

bool initSymbolTable() {{
  // Use the HIP runtime library loaded into the existing process if it exits.
  void *lib = dlopen("libamdhip64.so", RTLD_NOLOAD);
  if (lib) {{
    // printf("[triton] chosen loaded libamdhip64.so in the process\\n");
  }}

  // Otherwise, go through the list of search paths to dlopen the first HIP
  // driver library.
  if (!lib) {{
    int n = sizeof(hipLibSearchPaths) / sizeof(hipLibSearchPaths[0]);
    for (int i = 0; i < n; ++i) {{
      void *handle = dlopen(hipLibSearchPaths[i], RTLD_LAZY | RTLD_LOCAL);
      if (handle) {{
        lib = handle;
        // printf("[triton] chosen %s\\n", hipLibSearchPaths[i]);
      }}
    }}
  }}
  if (!lib) {{
    PyErr_SetString(PyExc_RuntimeError, "cannot open libamdhip64.so");
    return false;
  }}

  // Resolve all symbols we are interested in.
  dlerror(); // Clear existing errors
  const char *error = NULL;
#define QUERY_EACH_FN(hipSymbolName, ...)                                     \\
  *(void **)&hipSymbolTable.hipSymbolName = dlsym(lib, #hipSymbolName);       \\
  error = dlerror();                                                          \\
  if (error) {{                                                               \\
    PyErr_SetString(PyExc_RuntimeError,                                       \\
                    "cannot query " #hipSymbolName " from libamdhip64.so");   \\
    dlclose(lib);                                                             \\
    return false;                                                             \\
  }}

  HIP_SYMBOL_LIST(QUERY_EACH_FN, QUERY_EACH_FN)

  return true;
}}

static inline void gpuAssert(hipError_t code, const char *file, int line)
{{
   if (code != HIP_SUCCESS)
   {{
      const char* prefix = "Triton Error [HIP]: ";
       const char* str = hipSymbolTable.hipGetErrorString(code);
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
      HIP_CHECK(hipSymbolTable.hipModuleLaunchKernel(function, gridX, gridY, gridZ, {warp_size}*num_warps, 1, 1, shared_memory, stream, params, 0));
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
    hipError_t status = hipSymbolTable.hipPointerGetAttribute(&dev_ptr, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
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
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &_stream, &_function,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  // extract kernel metadata
  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
    return NULL;
  }}
  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}


  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (hipStream_t)_stream, (hipFunction_t)_function{', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items()) if len(signature) > 0 else ''});

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
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
  if (!initSymbolTable()) {{
    return NULL;
  }}
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


class HIPLauncher(object):

    def __init__(self, src, metadata):
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        src = make_launcher(constants, signature, ids, metadata.warp_size)
        mod = compile_module_from_src(src, "__triton_launcher")
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class HIPDriver(GPUDriver):

    def __init__(self):
        super().__init__()
        self.utils = HIPUtils()
        self.launcher_cls = HIPLauncher

    @staticmethod
    def is_active():
        import torch
        return torch.version.hip is not None

    def get_current_target(self):
        device = self.get_current_device()
        device_properties = self.utils.get_device_properties(device)
        arch = device_properties['arch']
        warp_size = device_properties['warpSize']
        return GPUTarget("hip", arch.split(':')[0], warp_size)
