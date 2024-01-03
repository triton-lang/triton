import os
import hashlib
import tempfile
from pathlib import Path
from triton.common.build import _build
from triton.runtime.cache import get_cache_manager
from triton.runtime.driver import GPUDriver


class HIPUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(HIPUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "hip.c")).read_text()
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


class HIPDriver(GPUDriver):

    def __init__(self):
        super().__init__()
        self.utils = HIPUtils()
        self.binary_ext = "hsaco"

    @staticmethod
    def is_active():
        import torch
        return torch.version.hip is not None

    def get_current_target(self):
        device = self.get_current_device()
        arch = self.utils.get_device_properties(device)['arch']
        return ("hip", arch.split(':')[0])

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args


driver = HIPDriver
