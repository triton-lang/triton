import abc
import hashlib
import os
import tempfile
from pathlib import Path
import functools

from ..common.build import _build
from .cache import get_cache_manager
from ..runtime import driver


class DriverBase(metaclass=abc.ABCMeta):
    CUDA = 0
    HIP = 1

    @staticmethod
    def third_party_dir():
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party")

    def __init__(self) -> None:
        pass


# -----------------------------
# CUDA
# -----------------------------


class CudaUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CudaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "backends", "cuda.c")).read_text()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "cuda_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "main.c")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build("cuda_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util

        spec = importlib.util.spec_from_file_location("cuda_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.CUtensorMapDataType = mod.CUtensorMapDataType
        self.CUtensorMapInterleave = mod.CUtensorMapInterleave
        self.CUtensorMapSwizzle = mod.CUtensorMapSwizzle
        self.CUtensorMapL2promotion = mod.CUtensorMapL2promotion
        self.CUtensorMapFloatOOBfill = mod.CUtensorMapFloatOOBfill
        self.cuTensorMapEncodeTiled = mod.cuTensorMapEncodeTiled
        self.cuMemAlloc = mod.cuMemAlloc
        self.cuMemcpyHtoD = mod.cuMemcpyHtoD
        self.cuMemFree = mod.cuMemFree
        self.cuOccupancyMaxActiveClusters = mod.cuOccupancyMaxActiveClusters


class TensorMapManager:

    def __init__(self):
        self.tensormaps_device = {}

    def __getitem__(self, key: tuple):
        if key in self.tensormaps_device:
            return int(self.tensormaps_device[key])
        else:
            (e, args) = key
            t_tensormap = e.tensormap(args)
            TENSORMAP_SIZE_IN_BYTES = 128
            t_tensormap_device = driver.utils.cuMemAlloc(TENSORMAP_SIZE_IN_BYTES)
            driver.utils.cuMemcpyHtoD(t_tensormap_device, t_tensormap, TENSORMAP_SIZE_IN_BYTES)
            self.tensormaps_device[key] = t_tensormap_device
            return int(self.tensormaps_device[key])

    def __del__(self):
        for _, v in self.tensormaps_device.items():
            driver.utils.cuMemFree(v)


class CudaDriver(DriverBase):
    tensormap_manager = TensorMapManager()

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CudaDriver, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.utils = CudaUtils()
        self.backend = self.CUDA
        self.binary_ext = "cubin"
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

    @functools.lru_cache()
    def get_current_target(self):
        device = self.get_current_device()
        capability = self.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
        return ("cuda", capability)

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        args_with_tma = list(args)
        if tensormaps_info is not None:
            # tuple for hashable
            args_ptr = tuple([arg.data_ptr() if hasattr(arg, 'data_ptr') else arg for arg in args])
            for i, e in enumerate(tensormaps_info):
                args_with_tma.append(CudaDriver.tensormap_manager[(e, args_ptr)])
        return args_with_tma


# -----------------------------
# HIP
# -----------------------------


class HIPUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(HIPUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "backends", "hip.c")).read_text()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "hip_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "main.c")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build("hip_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util

        spec = importlib.util.spec_from_file_location("hip_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties


class HIPDriver(DriverBase):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(HIPDriver, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.utils = HIPUtils()
        self.backend = self.HIP


class UnsupportedDriver(DriverBase):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(UnsupportedDriver, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.utils = None
        self.backend = None


# -----------------------------
# Driver
# -----------------------------


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
    import torch

    if torch.version.hip is not None:
        return HIPDriver()
    elif torch.cuda.is_available():
        return CudaDriver()
    else:
        return UnsupportedDriver()


driver = LazyProxy(initialize_driver)
