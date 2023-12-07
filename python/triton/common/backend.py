import functools
import hashlib
import importlib
import importlib.util
import os
import re
import subprocess
import traceback
from typing import Dict

from ..runtime.driver import DriverBase

TRITON_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRITON_VERSION = "2.2.0"


class BaseBackend:

    def __init__(self, device_type: str) -> None:
        self.device_type = device_type

    def add_stages(self, arch, extern_libs, stages):
        """
        Custom the arch, extern_libs and stages per backend specific requirement
        """
        raise NotImplementedError

    def add_meta_info(self, ir, cur_module, next_module, metadata, asm):
        """
        Custom the ir, module, metadata and asm per backend specific requirement
        """
        raise NotImplementedError

    def get_load_binary_fn(self):
        """
        Return a callable to load binary
        """
        raise NotImplementedError

    def get_driver(self) -> DriverBase:
        """
        Get the backend driver. Please refer to "DriverBase" for more details
        """
        raise NotImplementedError

    def get_stream(self):
        """
        Get stream for current device
        """
        raise NotImplementedError

    def get_device_properties(self, device):
        raise NotImplementedError

    def get_current_device(self):
        """
        Get current device
        """
        raise NotImplementedError

    def set_current_device(self, device):
        """
        Set current device as the given device
        """
        raise NotImplementedError

    def get_kernel_bin(self):
        raise NotImplementedError

    def make_launcher_stub(self, name, signature, constants):
        """
        Generate the launcher stub to launch the kernel
        """
        raise NotImplementedError

    def get_architecture_descriptor(self, **kwargs):
        """
        Get the architecture descriptor the backend
        """
        raise NotImplementedError

    @classmethod
    def create_backend(cls, device_type: str):
        return cls(device_type)


_backends: Dict[str, BaseBackend] = {}


def register_backend(device_type: str, backend_cls: type):
    if device_type not in _backends:
        _backends[device_type] = backend_cls.create_backend(device_type)


def get_backend(device_type: str):
    if device_type not in _backends:
        device_backend_package_name = f"...third_party.{device_type}"
        if importlib.util.find_spec(device_backend_package_name, package=__spec__.name):
            try:
                importlib.import_module(device_backend_package_name, package=__spec__.name)
            except Exception:
                traceback.print_exc()
        else:
            return None
    return _backends[device_type] if device_type in _backends else None


def _path_to_binary(binary: str):
    base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    paths = [
        os.environ.get(f"TRITON_{binary.upper()}_PATH", ""),
        os.path.join(base_dir, "third_party", "cuda", "bin", binary)
    ]

    for p in paths:
        bin = p.split(" ")[0]
        if os.path.exists(bin) and os.path.isfile(bin):
            result = subprocess.check_output([bin, "--version"], stderr=subprocess.STDOUT)
            if result is not None:
                version = re.search(r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
                if version is not None:
                    return p, version.group(1)
    raise RuntimeError(f"Cannot find {binary}")


@functools.lru_cache()
def path_to_ptxas():
    return _path_to_binary("ptxas")


@functools.lru_cache()
def path_to_cuobjdump():
    return _path_to_binary("cuobjdump")


@functools.lru_cache()
def path_to_nvdisasm():
    return _path_to_binary("nvdisasm")


@functools.lru_cache()
def compute_core_version_key():
    import pkgutil
    contents = []
    # frontend
    with open(__file__, "rb") as f:
        contents += [hashlib.sha1(f.read()).hexdigest()]
    # compiler
    compiler_path = os.path.join(TRITON_PATH, 'compiler')
    for lib in pkgutil.iter_modules([compiler_path]):
        with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
            contents += [hashlib.sha1(f.read()).hexdigest()]
    # backend
    libtriton_hash = hashlib.sha1()
    with open(os.path.join(TRITON_PATH, "_C/libtriton.so"), "rb") as f:
        while True:
            chunk = f.read(1024**2)
            if not chunk:
                break
            libtriton_hash.update(chunk)
    contents.append(libtriton_hash.hexdigest())
    # language
    language_path = os.path.join(TRITON_PATH, 'language')
    for lib in pkgutil.iter_modules([language_path]):
        with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
            contents += [hashlib.sha1(f.read()).hexdigest()]
    return '-'.join(TRITON_VERSION) + '-'.join(contents)


_cached_cuda_version_key = None


def get_cuda_version_key():
    global _cached_cuda_version_key
    if _cached_cuda_version_key is None:
        key = compute_core_version_key()
        try:
            ptxas = path_to_ptxas()[0]
            ptxas_version = subprocess.check_output([ptxas, "--version"])
        except RuntimeError:
            ptxas_version = b"NO_PTXAS"
        _cached_cuda_version_key = key + '-' + hashlib.sha1(ptxas_version).hexdigest()
    return _cached_cuda_version_key
