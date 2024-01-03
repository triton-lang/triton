from abc import ABCMeta, abstractmethod, abstractclassmethod
import os
import importlib


class DriverBase(metaclass=ABCMeta):

    @abstractclassmethod
    def is_active(self):
        pass

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


class UnsupportedDriver(DriverBase):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(UnsupportedDriver, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.utils = None
        self.backend = None


class LazyProxy:

    def __init__(self, init_fn):
        self._init_fn = init_fn
        self._obj = None

    def _initialize_obj(self):
        if self._obj is None:
            self._obj = self._init_fn()

    def __getattr__(self, name):
        self._initialize_obj()
        return getattr(self._obj, name)

    def __setattr__(self, name, value):
        if name in ["_init_fn", "_obj"]:
            super().__setattr__(name, value)
        else:
            self._initialize_obj()
            setattr(self._obj, name, value)

    def __delattr__(self, name):
        self._initialize_obj()
        delattr(self._obj, name)

    def __repr__(self):
        if self._obj is None:
            return f"<{self.__class__.__name__} for {self._init_fn} not yet initialized>"
        return repr(self._obj)

    def __str__(self):
        self._initialize_obj()
        return str(self._obj)


def initialize_driver():
    drivers = list()
    backends_dir = os.path.join(os.path.dirname(__file__), 'backends')
    for filename in os.listdir(backends_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            file_path = os.path.join(backends_dir, filename)
            spec = importlib.util.spec_from_file_location(filename[:-3], file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            driver = getattr(module, "driver")
            assert issubclass(driver, DriverBase)
            drivers.append(driver)
    valids = [driver for driver in drivers if driver.is_active()]
    if len(valids) == 0:
        raise RuntimeError("Found 0 active driver")
    if len(valids) > 1:
        raise RuntimeError("Found more than 1 active driver")
    return valids[0]()


driver = LazyProxy(initialize_driver)
