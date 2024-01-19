from abc import ABCMeta, abstractmethod
import os


class DriverBase(metaclass=ABCMeta):

    @classmethod
    def is_active(cls):
        active_driver = os.environ.get("TRITON_ACTIVE_DRIVER")
        if active_driver is not None:
            return active_driver == cls.target_name()
        return cls.should_activate()

    @staticmethod
    @abstractmethod
    def target_name():
        pass

    @abstractmethod
    def get_current_target(self):
        pass

    def __init__(self) -> None:
        pass


class GPUDriver(DriverBase):

    def __init__(self):
        # TODO: support other frameworks than torch
        import torch
        self.get_device_capability = torch.cuda.get_device_capability
        try:
            from torch._C import _cuda_getCurrentRawStream
            self.get_current_stream = _cuda_getCurrentRawStream
        except ImportError:
            self.get_current_stream = lambda idx: torch.cuda.current_stream(idx).cuda_stream
        self.get_current_device = torch.cuda.current_device
        self.set_current_device = torch.cuda.set_device

    # TODO: remove once TMA is cleaned up
    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args
