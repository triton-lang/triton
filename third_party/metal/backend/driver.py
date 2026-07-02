import ctypes
import ctypes.util
import functools
import os
import subprocess
import sys
import tempfile

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase

dirname = os.path.dirname(os.path.realpath(__file__))


def _metal_is_available():
    """Check if Metal is available on this system."""
    if sys.platform != 'darwin':
        return False
    try:
        # Check for Metal framework
        metal_framework = ctypes.util.find_library('Metal')
        if metal_framework is None:
            return False
        # Check for xcrun metal compiler
        result = subprocess.run(
            ["xcrun", "--sdk", "macosx", "metal", "--version"],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


@functools.lru_cache()
def _get_metal_gpu_family():
    """Detect Apple GPU family using system_profiler."""
    try:
        result = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            stderr=subprocess.DEVNULL, timeout=10
        ).decode("utf-8")

        # Detect Apple Silicon chip generation
        if "Apple M4" in result:
            return 10
        elif "Apple M3" in result:
            return 9
        elif "Apple M2" in result:
            return 8
        elif "Apple M1" in result:
            return 7
        # Fallback to apple7 as minimum supported
        if "Apple" in result:
            return 7
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return 7


def _get_metal_device_name():
    """Get the Metal device name."""
    try:
        result = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            stderr=subprocess.DEVNULL, timeout=10
        ).decode("utf-8")
        for line in result.splitlines():
            if "Chipset Model" in line:
                return line.split(":")[-1].strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "Apple GPU"


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "uint64_t"  # Device pointer as uint64
    return {
        "i1": "int8_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint8_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


class MetalLauncher(object):
    """Handles Metal kernel dispatch."""

    def __init__(self, src, metadata):
        self.kernel_name = metadata.get("name", "triton_kernel")
        self.compile_mode = metadata.get("compile_mode", "jit_msl")
        self.shared_memory = metadata.get("shared", 0)
        self.num_warps = metadata.get("num_warps", 4)
        self.global_scratch_size = getattr(metadata, "global_scratch_size", 0)
        self.global_scratch_align = getattr(metadata, "global_scratch_align", 1)
        self.profile_scratch_size = getattr(metadata, "profile_scratch_size", 0)
        self.profile_scratch_align = getattr(metadata, "profile_scratch_align", 1)

    def __call__(self, gridX, gridY, gridZ, stream, function,
                 kernel_metadata, launch_metadata,
                 launch_enter_hook, launch_exit_hook, *args):
        # In a full implementation, this dispatches through the Metal runtime
        # via a compiled native extension. For now, we use the Python-level
        # metal_utils module.
        if launch_enter_hook is not None:
            launch_enter_hook(kernel_metadata, launch_metadata)

        # Dispatch through Metal runtime
        metal_dispatch(
            function, gridX, gridY, gridZ,
            self.num_warps * 32, 1, 1,
            self.shared_memory,
            stream, args
        )

        if launch_exit_hook is not None:
            launch_exit_hook(kernel_metadata, launch_metadata)


def metal_dispatch(function, grid_x, grid_y, grid_z,
                   threads_x, threads_y, threads_z,
                   shared_memory, stream, args):
    """Dispatch a Metal compute kernel.

    In a full implementation, this calls into a compiled ObjC/Swift extension
    that uses the Metal API (MTLComputeCommandEncoder.dispatchThreadgroups).
    """
    # This will be implemented by the native metal_utils extension
    if hasattr(MetalDriver, '_instance') and MetalDriver._instance is not None:
        utils = MetalDriver._instance.utils
        if utils is not None:
            utils.dispatch(function, grid_x, grid_y, grid_z,
                           threads_x, threads_y, threads_z,
                           shared_memory, stream, args)


class MetalUtils(object):
    """Metal runtime utilities - compiled from Objective-C sources."""

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(MetalUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self._device = None
        self._queue = None
        # Load Metal framework
        try:
            self._metal_framework = ctypes.cdll.LoadLibrary(
                '/System/Library/Frameworks/Metal.framework/Metal'
            )
        except OSError:
            self._metal_framework = None

    def get_device_name(self):
        return _get_metal_device_name()

    def get_gpu_family(self):
        return _get_metal_gpu_family()

    def dispatch(self, function, grid_x, grid_y, grid_z,
                 threads_x, threads_y, threads_z,
                 shared_memory, stream, args):
        """Dispatch a compute kernel via Metal API.

        In full implementation, this uses PyObjC or a compiled extension
        to call MTLComputeCommandEncoder.dispatchThreadgroups.
        """
        pass

    def load_binary(self, name, binary, shared_memory, device):
        """Load a metallib binary and return a pipeline state object.

        In full implementation, this creates an MTLLibrary from the binary
        and produces an MTLComputePipelineState.
        """
        return binary


class MetalDriver(DriverBase):
    _instance = None

    def __init__(self):
        MetalDriver._instance = self
        self.utils = MetalUtils()
        self.launcher_cls = MetalLauncher

    @staticmethod
    def is_active():
        return _metal_is_available()

    def get_current_target(self):
        gpu_family = _get_metal_gpu_family()
        warp_size = 32  # Apple GPU SIMD width
        return GPUTarget("metal", gpu_family, warp_size)

    def get_active_torch_device(self):
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def get_device_interface(self):
        import torch
        if hasattr(torch, 'mps'):
            return torch.mps
        return None

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench
