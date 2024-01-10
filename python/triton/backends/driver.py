from abc import ABCMeta, abstractmethod, abstractclassmethod


class DriverBase(metaclass=ABCMeta):

    @abstractclassmethod
    def is_active(self):
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
