
import importlib
import importlib.util
from typing import Dict

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
                return None
        else:
            return None
    return _backends[device_type] if device_type in _backends else None
