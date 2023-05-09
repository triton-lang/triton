import functools
import gc
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tracemalloc
import setuptools

import torch
import triton
import triton.language as tl
from triton.common.build import quiet
from triton.common.backend import BaseBackend, register_backend
from triton.compiler.make_launcher import make_so_cache_key
from triton.runtime.cache import get_cache_manager
from triton.runtime.driver import DriverBase
from triton.runtime.jit import version_key


def build_for_backend(name, src, srcdir):
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

    ret = subprocess.check_call([cc, src, f"-I{py_include_dir}", f"-I{srcdir}", "-shared", "-fPIC", "-o", so])
    if ret == 0:
        return so
    # fallback on setuptools
    extra_compile_args = []
    library_dirs = []
    include_dirs = [srcdir]
    libraries = []
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


class ExtensionUtils:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ExtensionUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "extension_backend.c")).read_text()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "ext_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "main.c")
                with open(src_path, "w") as f:
                    f.write(src)
                so = build_for_backend("ext_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location("ext_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties


class ExtensionDriver(DriverBase):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ExtensionDriver, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.utils = ExtensionUtils()


class ExtensionBackend(BaseBackend):
    def __init__(self, device_type: str) -> None:
        super(ExtensionBackend, self).__init__(device_type)
        self.driver = ExtensionDriver()

    def add_stages(self, arch, extern_libs, stages):
        filter_in_stages = ["ast", "ttir", "ttgir"]
        filter_out_stages = []
        for key, _ in stages.items():
            if key not in filter_in_stages:
                filter_out_stages.append(key)
        for filter_out_key in filter_out_stages:
            stages.pop(filter_out_key)

    def add_meta_info(self, ir, module, metadata, asm):
        metadata["name"] = "extension_backend_name"

    def device_driver(self):
        return self.driver

    def get_stream(self):
        return ""

    @functools.lru_cache(None)
    def get_device_properties(self):
        return self.driver.utils.get_device_properties()

    def get_current_device(self):
        return torch.device("cpu")

    def set_current_device(self, device):
        pass

    def get_kernel_path(self):
        return "ttgir"

    def get_architecture_descriptor(self, **kwargs):
        return ""

    def make_launcher_stub(self, name, signature, constants):
        # name of files that are cached
        so_cache_key = make_so_cache_key(version_key(), signature, constants)
        so_cache_manager = get_cache_manager(so_cache_key)
        so_name = f"{name}.so"
        # retrieve stub from cache if it exists
        cache_path = so_cache_manager.get_file(so_name)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src = self._generate_launcher(constants, signature)
                src_path = os.path.join(tmpdir, "main.c")
                with open(src_path, "w") as f:
                    f.write(src)
                so = build_for_backend(name, src_path, tmpdir)
                with open(so, "rb") as f:
                    return so_cache_manager.put(f.read(), so_name, binary=True)
        else:
            return cache_path

    def _generate_launcher(self, constants, signature):
        def ty_to_cpp(ty):
            if ty[0] == '*':
                return "void *"

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
        src = f"""
        #define __EXTENSION_BACKEND__
        #include <Python.h>
        #include <stdio.h>

        static PyObject* launch(PyObject* self, PyObject* args) {{
        printf("Launch empty kernel for extension backend\\n");

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

        return src

register_backend("cpu", ExtensionBackend)

@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 10
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)

tracemalloc.start()
try:
    inp = torch.randn(10)
    out = torch.randn(10)
    kernel[(10,)](inp, out, 10, XBLOCK=16)
    gc.collect()
    begin, _ = tracemalloc.get_traced_memory()
    for _ in range(100):
        kernel[(10,)](inp, out, 10, XBLOCK=16)
    gc.collect()
    end, _ = tracemalloc.get_traced_memory()
    assert end - begin < 1000
finally:
    tracemalloc.stop()
