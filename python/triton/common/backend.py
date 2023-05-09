
from typing import Dict


class BaseBackend:
    def __init__(self, device_type: str) -> None:
        self.device_type = device_type

    def add_stages(self, arch, extern_libs, stages):
        raise NotImplementedError

    def add_meta_info(self, ir, module, metadata, asm):
        raise NotImplementedError

    def device_driver(self):
        raise NotImplementedError

    def get_stream(self):
        raise NotImplementedError

    def get_device_properties(self):
        raise NotImplementedError

    def get_current_device(self):
        raise NotImplementedError

    def set_current_device(self, device):
        raise NotImplementedError

    def get_kernel_path(self):
        raise NotImplementedError

    def make_launcher_stub(self, name, signature, constants):
        raise NotImplementedError

    def get_architecture_descriptor(self, **kwargs):
        raise NotImplementedError

    @classmethod
    def create_backend(cls, device_type: str):
        return cls(device_type)

_backends: Dict[str, BaseBackend] = {}


def register_backend(device_type: str, backend_cls: type):
    if device_type not in _backends:
        _backends[device_type] = backend_cls.create_backend(device_type)


def get_backend(device_type: str):
    return _backends[device_type] if device_type in _backends else None

