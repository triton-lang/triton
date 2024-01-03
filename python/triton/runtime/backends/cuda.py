import os
import hashlib
import tempfile
from pathlib import Path
from triton.common.build import _build
from triton.runtime.cache import get_cache_manager
from triton.runtime.driver import GPUDriver


class CudaUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CudaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "cuda.c")).read_text()
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

    def __init__(self, utils):
        self.tensormaps_device = {}
        self.utils = utils

    def __getitem__(self, key: tuple):
        if key in self.tensormaps_device:
            return int(self.tensormaps_device[key])
        else:
            (e, args) = key
            t_tensormap = e.tensormap(args)
            TENSORMAP_SIZE_IN_BYTES = 128
            t_tensormap_device = self.utils.cuMemAlloc(TENSORMAP_SIZE_IN_BYTES)
            self.utils.cuMemcpyHtoD(t_tensormap_device, t_tensormap, TENSORMAP_SIZE_IN_BYTES)
            self.tensormaps_device[key] = t_tensormap_device
            return int(self.tensormaps_device[key])

    def __del__(self):
        for _, v in self.tensormaps_device.items():
            self.utils.cuMemFree(v)


class CudaDriver(GPUDriver):

    def __init__(self):
        self.utils = CudaUtils()  # TODO: make static
        self.tensormap_manager = TensorMapManager(self.utils)  # TODO: make static
        self.backend = self.CUDA
        self.binary_ext = "cubin"
        super().__init__()

    def get_current_target(self):
        device = self.get_current_device()
        capability = self.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
        return ("cuda", capability)

    @staticmethod
    def is_active():
        import torch
        return torch.version.hip is None

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        args_with_tma = list(args)
        if tensormaps_info is not None:
            # tuple for hashable
            args_ptr = tuple([arg.data_ptr() if hasattr(arg, 'data_ptr') else arg for arg in args])
            for i, e in enumerate(tensormaps_info):
                args_with_tma.append(CudaDriver.tensormap_manager[(e, args_ptr)])
        return args_with_tma


driver = CudaDriver
