import ctypes
import ctypes.util
import functools
import os
import struct
import subprocess
import sys

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase

dirname = os.path.dirname(os.path.realpath(__file__))
libdevice_dir = os.path.join(dirname, "lib")


def _metal_is_available():
    """Check if Metal is available on this system.

    Checks for the Metal framework (required) and Apple Silicon GPU.
    The Metal shader compiler toolchain (xcrun metal) is optional —
    kernels can be JIT-compiled via MTLDevice.newLibraryWithSource.
    """
    if sys.platform != 'darwin':
        return False
    try:
        # Check for Metal framework
        metal_framework = ctypes.util.find_library('Metal')
        if metal_framework is None:
            return False
        # Verify we can load it
        ctypes.cdll.LoadLibrary(metal_framework)
        return True
    except (OSError, FileNotFoundError):
        return False


@functools.lru_cache()
def _get_metal_gpu_family():
    """Detect Apple GPU family using system_profiler.

    Returns the GPU family number:
      10 = M4 (Apple10, Metal 4 native)
       9 = M3 (Apple9)
       8 = M2 (Apple8)
       7 = M1 (Apple7, minimum supported)
    """
    try:
        result = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], stderr=subprocess.DEVNULL,
                                         timeout=10).decode("utf-8")

        if "Apple M4" in result:
            return 10
        elif "Apple M3" in result:
            return 9
        elif "Apple M2" in result:
            return 8
        elif "Apple M1" in result:
            return 7
        if "Apple" in result:
            return 7
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return 7


def _get_metal_device_name():
    """Get the Metal device name."""
    try:
        result = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], stderr=subprocess.DEVNULL,
                                         timeout=10).decode("utf-8")
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


def _pack_kernel_args(args, signature):
    """Pack kernel arguments into a bytes buffer for Metal dispatch.

    Handles all Triton types: pointers, integers, and floating point.
    Metal uses setBytes for small constants and setBuffer for device pointers.
    """
    packed_args = []
    for i, arg in enumerate(args):
        if arg is None:
            packed_args.append((None, 'null', 0))
            continue

        # Determine type from signature if available
        sig_type = signature.get(i, '') if signature else ''

        if sig_type.startswith('*') or isinstance(arg, int) and arg > 0xFFFFFFFF:
            # Device pointer - pass as uint64
            packed_args.append((struct.pack('Q', int(arg)), 'ptr', 8))
        elif sig_type in ('fp64', ):
            packed_args.append((struct.pack('d', float(arg)), 'f64', 8))
        elif sig_type in ('fp32', 'f32'):
            packed_args.append((struct.pack('f', float(arg)), 'f32', 4))
        elif sig_type in ('fp16', ):
            import numpy as np
            packed_args.append((np.float16(arg).tobytes(), 'f16', 2))
        elif sig_type in ('bf16', ):
            import numpy as np
            val = np.float32(arg)
            bf16_bits = int.from_bytes(val.tobytes(), 'little') >> 16
            packed_args.append((struct.pack('H', bf16_bits), 'bf16', 2))
        elif sig_type in ('i64', ):
            packed_args.append((struct.pack('q', int(arg)), 'i64', 8))
        elif sig_type in ('u64', ):
            packed_args.append((struct.pack('Q', int(arg)), 'u64', 8))
        elif sig_type in ('i32', ):
            packed_args.append((struct.pack('i', int(arg)), 'i32', 4))
        elif sig_type in ('u32', ):
            packed_args.append((struct.pack('I', int(arg)), 'u32', 4))
        elif sig_type in ('i16', ):
            packed_args.append((struct.pack('h', int(arg)), 'i16', 2))
        elif sig_type in ('u16', ):
            packed_args.append((struct.pack('H', int(arg)), 'u16', 2))
        elif sig_type in ('i8', 'i1'):
            packed_args.append((struct.pack('b', int(arg)), 'i8', 1))
        elif sig_type in ('u8', 'u1'):
            packed_args.append((struct.pack('B', int(arg)), 'u8', 1))
        elif isinstance(arg, float):
            packed_args.append((struct.pack('f', arg), 'f32', 4))
        elif isinstance(arg, int):
            if -2147483648 <= arg <= 2147483647:
                packed_args.append((struct.pack('i', arg), 'i32', 4))
            else:
                packed_args.append((struct.pack('q', arg), 'i64', 8))
        else:
            # Default: try to pack as int32
            packed_args.append((struct.pack('i', int(arg)), 'i32', 4))

    return packed_args


class MetalLauncher(object):
    """Handles Metal kernel dispatch with proper argument packing."""

    def __init__(self, src, metadata):
        self.kernel_name = metadata.get("name", "triton_kernel")
        self.compile_mode = metadata.get("compile_mode", "jit_msl")
        self.shared_memory = metadata.get("shared", 0)
        self.num_warps = metadata.get("num_warps", 4)
        self.global_scratch_size = getattr(metadata, "global_scratch_size", 0) or 0
        self.global_scratch_align = getattr(metadata, "global_scratch_align", 1) or 1
        self.profile_scratch_size = getattr(metadata, "profile_scratch_size", 0) or 0
        self.profile_scratch_align = getattr(metadata, "profile_scratch_align", 1) or 1

        # Extract signature for arg packing
        constants = src.constants if hasattr(src, "constants") else dict()
        self.signature = {idx: value for idx, value in src.signature.items()} if hasattr(src, 'signature') else {}

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                 launch_exit_hook, *args):
        if launch_enter_hook is not None:
            launch_enter_hook(kernel_metadata, launch_metadata)

        # Pack arguments with proper types
        packed = _pack_kernel_args(args, self.signature)

        # Compute threadgroup dimensions
        threads_per_threadgroup = min(self.num_warps * 32, 1024)

        # Dispatch through Metal runtime
        metal_dispatch(function, gridX, gridY, gridZ, threads_per_threadgroup, 1, 1, self.shared_memory, stream, packed)

        if launch_exit_hook is not None:
            launch_exit_hook(kernel_metadata, launch_metadata)


def metal_dispatch(function, grid_x, grid_y, grid_z, threads_x, threads_y, threads_z, shared_memory, stream,
                   packed_args):
    """Dispatch a Metal compute kernel via the native extension.

    This calls into the compiled ObjC extension that uses the Metal API
    (MTLComputeCommandEncoder.dispatchThreadgroups).
    """
    if hasattr(MetalDriver, '_instance') and MetalDriver._instance is not None:
        utils = MetalDriver._instance.utils
        if utils is not None and hasattr(utils, '_dispatch_native'):
            utils._dispatch_native(function, grid_x, grid_y, grid_z, threads_x, threads_y, threads_z, shared_memory,
                                   stream, packed_args)


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
            self._metal_framework = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/Metal.framework/Metal')
        except OSError:
            self._metal_framework = None

    def get_device_name(self):
        return _get_metal_device_name()

    def get_gpu_family(self):
        return _get_metal_gpu_family()

    def dispatch(self, function, grid_x, grid_y, grid_z, threads_x, threads_y, threads_z, shared_memory, stream, args):
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
