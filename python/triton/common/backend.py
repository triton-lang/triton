import functools
import os
import re
import subprocess
from ..runtime.driver import DriverBase


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


def _path_to_binary(binary: str):
    base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    paths = [
        os.environ.get(f"TRITON_{binary.upper()}_PATH", ""),
        os.path.join(base_dir, "third_party", "cuda", "bin", binary),
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
def path_to_rocm_lld():
    return "/opt/rocm/llvm/bin/ld.lld"
