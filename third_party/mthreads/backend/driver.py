import functools
import os
import hashlib
import subprocess
import tempfile
import shutil
import sysconfig
from pathlib import Path
from triton.runtime.cache import get_cache_manager
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver

dirname = os.path.dirname(os.path.realpath(__file__))


@functools.lru_cache()
def musa_home_dir():
    return os.getenv("MUSA_HOME", default="/usr/local/musa")


@functools.lru_cache()
def musa_include_dir():
    musa_home = musa_home_dir()
    return os.path.join(musa_home, "include")


@functools.lru_cache()
def libmusa_dirs():
    musa_home = musa_home_dir()
    return os.path.join(musa_home, "lib")


def _build(name, src, srcdir):
    musa_lib_dir = libmusa_dirs()
    mu_include_dir = musa_include_dir()
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    # try to avoid setuptools if possible
    cc = os.environ.get("CC")
    if cc is None:
        # TODO: support more things here.
        clang = shutil.which("clang")
        # `musa.h` has some fucking issues recently, which introduce c++ style code into `musa.h`, we have to use `g++` but not `gcc` until these issues fixed by musa team.
        gcc = shutil.which("g++")
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

    cc_cmd = [
        cc, src, "-O3", f"-I{mu_include_dir}", f"-I{py_include_dir}", f"-I{srcdir}", "-shared", "-fPIC",
        f"-L{musa_lib_dir}", "-lmusa", "-o", so
    ]
    # cc_cmd += [f"-L{dir}" for dir in musa_lib_dir]
    ret = subprocess.check_call(cc_cmd)

    if ret == 0:
        return so
    # Backup source file and cmd.
    dst = os.path.join("/tmp", os.path.basename(src))
    with open(dst, 'w') as f:
        f.write("// " + ' '.join(cc_cmd) + "\n" + open(src).read())
    raise RuntimeError(f"Failed to compile stub for {name}. Source file and compile cmd backup to {dst}")
    '''
    # fallback on setuptools
    extra_compile_args = []
    library_dirs = musa_lib_dir
    include_dirs = [srcdir, mu_include_dir]
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
    '''


def compile_module_from_src(src, name):
    key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir)
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


class MusaUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(MusaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        mod = compile_module_from_src(Path(os.path.join(dirname, "driver.c")).read_text(), "musa_utils")
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "MUdeviceptr"
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


def make_launcher(constants, signature, ids, warp_size):
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors, such as tensor descriptors for CUDA.
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
            "uint32_t": "I",
            "int32_t": "i",
            "uint64_t": "K",
            "int64_t": "L",
        }[ty]

    args_format = ''.join([format_of(_extracted_type(ty)) for ty in signature.values()])
    format = "iiiKKOOOO" + args_format
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    # generate glue code
    params = [i for i in signature.keys() if i not in constants]
    src = f"""
#include \"musa.h\"
#include <stdbool.h>
#include <Python.h>
#include <dlfcn.h>

static inline void gpuAssert(MUresult code, const char *file, int line)
{{
   if (code != MUSA_SUCCESS)
   {{
      const char* prefix = "Triton Error [CUDA]: ";
      const char* str;
      muGetErrorString(code, &str);
      char err[1024] = {{0}};
      strcat(err, prefix);
      strcat(err, str);
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }}
}}

#define MUSA_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

typedef MUresult (*muLaunchKernelEx_t)(const MUlaunchConfig* config, MUfunction f, void** kernelParams, void** extra);

static muLaunchKernelEx_t getLaunchKernelExHandle() {{
  // Open the shared library
  void* handle = dlopen("libmusa.so", RTLD_LAZY);
  if (!handle) {{
    PyErr_SetString(PyExc_RuntimeError, "Failed to open libmusa.so");
    return NULL;
  }}
  // Clear any existing error
  dlerror();
  muLaunchKernelEx_t muLaunchKernelExHandle = (muLaunchKernelEx_t)dlsym(handle, "muLaunchKernelEx");
  // Check for errors
  const char *dlsym_error = dlerror();
  if (dlsym_error) {{
    PyErr_SetString(PyExc_RuntimeError, "Failed to retrieve muLaunchKernelEx from libmusa.so");
    return NULL;
  }}
  return muLaunchKernelExHandle;
}}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, MUstream stream, MUfunction function{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  void *params[] = {{ {', '.join(f"&arg{i}" for i in params)} }};
  if (gridX*gridY*gridZ > 0) {{
    if (num_ctas == 1) {{
      MUSA_CHECK(muLaunchKernel(function, gridX, gridY, gridZ, {warp_size}*num_warps, 1, 1, shared_memory, stream, params, 0));
    }} else {{
      MUlaunchAttribute launchAttr[2];
      launchAttr[0].id = MU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      launchAttr[0].value.clusterDim.x = clusterDimX;
      launchAttr[0].value.clusterDim.y = clusterDimY;
      launchAttr[0].value.clusterDim.z = clusterDimZ;
      launchAttr[1].id = MU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      launchAttr[1].value.clusterSchedulingPolicyPreference = MU_CLUSTER_SCHEDULING_POLICY_SPREAD;
      MUlaunchConfig config;
      config.gridDimX = gridX * clusterDimX;
      config.gridDimY = gridY * clusterDimY;
      config.gridDimZ = gridZ * clusterDimZ;
      config.blockDimX = {warp_size} * num_warps;
      config.blockDimY = 1;
      config.blockDimZ = 1;
      config.sharedMemBytes = shared_memory;
      config.hStream = stream;
      config.attrs = launchAttr;
      config.numAttrs = 2;
      static muLaunchKernelEx_t muLaunchKernelExHandle = NULL;
      if (muLaunchKernelExHandle == NULL) {{
        muLaunchKernelExHandle = getLaunchKernelExHandle();
      }}
      MUSA_CHECK(muLaunchKernelExHandle(&config, function, params, 0));
    }}
  }}
}}

typedef struct _DevicePtrInfo {{
    MUdeviceptr dev_ptr;
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
    int status = muPointerGetAttribute(&dev_ptr, MU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
    if (status == MUSA_ERROR_INVALID_VALUE) {{
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
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (MUstream)_stream, (MUfunction)_function{', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items()) if len(signature) > 0 else ''});
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


class MusaLauncher(object):

    def __init__(self, src, metadata):
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        src = make_launcher(constants, signature, ids, metadata.target.warp_size)
        mod = compile_module_from_src(src, "__triton_launcher")
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class MusaDriver(GPUDriver):

    def __init__(self):
        super().__init__()
        self.utils = MusaUtils()  # TODO: make static
        self.launcher_cls = MusaLauncher

        self.get_device_capability = self._get_device_capability
        self.get_current_stream = self._get_current_stream
        self.get_current_device = self._get_current_device
        self.set_current_device = self._set_current_device

    def _get_device_capability(self, device):
        return torch_musa.get_device_capability(device)

    def _get_current_stream(self, idx):
        try:
            # return torch_musa._MUSAC._musa_getCurrentStream(idx)
            return torch_musa._MUSAC._musa_getCurrentRawStream(idx)
        except ImportError:
            return torch_musa.current_stream(idx).musa_stream

    def _get_current_device(self):
        """
        Get current device
        """
        return torch_musa.current_device()

    def _set_current_device(self, device):
        """
        Set current device as the given device
        """
        torch_musa.set_device(device)

    def get_current_target(self):
        device = self.get_current_device()
        warp_size = 128
        capability = self.get_device_capability(device)
        if capability[0] > 2:
            warp_size = 32
        capability = capability[0] * 10 + capability[1]
        return GPUTarget("musa", capability, warp_size)

    @staticmethod
    def is_active():
        try:
            import torch
            import torch_musa
            return torch.musa.is_available()
        except:
            return False


if MusaDriver.is_active():
    import torch
    import torch_musa
