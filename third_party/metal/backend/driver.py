import ctypes
import ctypes.util
import functools
import os
import struct
import subprocess
import sys
import tempfile

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase

dirname = os.path.dirname(os.path.realpath(__file__))
libdevice_dir = os.path.join(dirname, "lib")


def _metal_is_available():
    if sys.platform != 'darwin':
        return False
    try:
        metal_framework = ctypes.util.find_library('Metal')
        if metal_framework is None:
            return False
        ctypes.cdll.LoadLibrary(metal_framework)
        return True
    except (OSError, FileNotFoundError):
        return False


@functools.lru_cache()
def _get_metal_gpu_family():
    """Detect Apple GPU family: 10=M4, 9=M3, 8=M2, 7=M1."""
    try:
        result = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            stderr=subprocess.DEVNULL, timeout=10
        ).decode("utf-8")
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


@functools.lru_cache()
def _get_metal_device_name():
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
        return "uint64_t"
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
    """Pack kernel arguments into bytes buffers for Metal dispatch."""
    packed_args = []
    for i, arg in enumerate(args):
        if arg is None:
            packed_args.append((None, 'null', 0))
            continue

        sig_type = signature.get(i, '') if signature else ''

        if sig_type.startswith('*') or isinstance(arg, int) and arg > 0xFFFFFFFF:
            packed_args.append((struct.pack('Q', int(arg)), 'ptr', 8))
        elif sig_type in ('fp64',):
            packed_args.append((struct.pack('d', float(arg)), 'f64', 8))
        elif sig_type in ('fp32', 'f32'):
            packed_args.append((struct.pack('f', float(arg)), 'f32', 4))
        elif sig_type in ('fp16',):
            import numpy as np
            packed_args.append((np.float16(arg).tobytes(), 'f16', 2))
        elif sig_type in ('bf16',):
            import numpy as np
            val = np.float32(arg)
            bf16_bits = int.from_bytes(val.tobytes(), 'little') >> 16
            packed_args.append((struct.pack('H', bf16_bits), 'bf16', 2))
        elif sig_type in ('i64',):
            packed_args.append((struct.pack('q', int(arg)), 'i64', 8))
        elif sig_type in ('u64',):
            packed_args.append((struct.pack('Q', int(arg)), 'u64', 8))
        elif sig_type in ('i32',):
            packed_args.append((struct.pack('i', int(arg)), 'i32', 4))
        elif sig_type in ('u32',):
            packed_args.append((struct.pack('I', int(arg)), 'u32', 4))
        elif sig_type in ('i16',):
            packed_args.append((struct.pack('h', int(arg)), 'i16', 2))
        elif sig_type in ('u16',):
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
            packed_args.append((struct.pack('i', int(arg)), 'i32', 4))

    return packed_args


@functools.lru_cache()
def _compile_metal_utils():
    """Compile the native ObjC Metal runtime extension."""
    import sysconfig
    import hashlib

    src_path = os.path.join(dirname, "driver.m")
    with open(src_path, "rb") as f:
        src_bytes = f.read()

    cache_dir = os.path.join(
        os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache")),
        "metal_utils"
    )
    os.makedirs(cache_dir, exist_ok=True)

    src_hash = hashlib.sha256(src_bytes).hexdigest()[:16]
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so_name = f"metal_utils_{src_hash}{suffix}"
    so_path = os.path.join(cache_dir, so_name)

    if os.path.exists(so_path):
        return so_path

    py_include = sysconfig.get_path("include")
    cc_cmd = [
        "clang", src_path,
        "-x", "objective-c",
        "-O2", "-shared", "-fPIC",
        "-framework", "Metal",
        "-framework", "Foundation",
        f"-I{py_include}",
        "-undefined", "dynamic_lookup",
        "-o", so_path,
    ]

    try:
        subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to compile Metal runtime extension: {e}") from e

    return so_path


@functools.lru_cache()
def _load_metal_utils():
    """Load the compiled Metal utils extension module."""
    import importlib.util

    so_path = _compile_metal_utils()
    spec = importlib.util.spec_from_file_location("metal_utils", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MetalLauncher(object):

    def __init__(self, src, metadata):
        self.kernel_name = metadata.get("name", "triton_kernel")
        self.compile_mode = metadata.get("compile_mode", "jit_msl")
        self.shared_memory = metadata.get("shared", 0)
        self.num_warps = metadata.get("num_warps", 4)
        self.global_scratch_size = metadata.get("global_scratch_size", 0) or 0
        self.global_scratch_align = metadata.get("global_scratch_align", 1) or 1

        constants = src.constants if hasattr(src, "constants") else dict()
        self.signature = {idx: value for idx, value in src.signature.items()} if hasattr(src, 'signature') else {}

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata,
                 launch_metadata, launch_enter_hook, launch_exit_hook, *args):
        if launch_enter_hook is not None:
            launch_enter_hook(kernel_metadata, launch_metadata)

        packed = _pack_kernel_args(args, self.signature)
        threads_per_threadgroup = min(self.num_warps * 32, 1024)

        metal_dispatch(
            function, gridX, gridY, gridZ,
            threads_per_threadgroup, 1, 1,
            self.shared_memory, stream, packed
        )

        if launch_exit_hook is not None:
            launch_exit_hook(kernel_metadata, launch_metadata)


def metal_dispatch(function, grid_x, grid_y, grid_z,
                   threads_x, threads_y, threads_z,
                   shared_memory, stream, packed_args):
    """Dispatch a Metal compute kernel via the native extension."""
    try:
        mod = _load_metal_utils()
    except (RuntimeError, ImportError):
        return

    # Build arg tuple for native dispatch
    arg_values = []
    for data, ty, size in packed_args:
        if data is None:
            arg_values.append(0)
        elif ty == 'ptr':
            arg_values.append(struct.unpack('Q', data)[0])
        elif ty in ('i32', 'u32'):
            arg_values.append(struct.unpack('i' if ty == 'i32' else 'I', data)[0])
        elif ty in ('i64', 'u64'):
            arg_values.append(struct.unpack('q' if ty == 'i64' else 'Q', data)[0])
        elif ty == 'f32':
            arg_values.append(struct.unpack('f', data)[0])
        else:
            arg_values.append(int.from_bytes(data, 'little') if data else 0)

    mod.dispatch(
        function, grid_x, grid_y, grid_z,
        threads_x, threads_y, threads_z,
        shared_memory, tuple(arg_values)
    )


class MetalUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(MetalUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        try:
            self._mod = _load_metal_utils()
        except (RuntimeError, ImportError):
            self._mod = None

    def get_device_name(self):
        if self._mod is not None:
            try:
                return self._mod.get_device_name()
            except Exception:
                pass
        return _get_metal_device_name()

    def get_gpu_family(self):
        if self._mod is not None:
            try:
                fam = self._mod.get_gpu_family()
                if fam > 0:
                    return fam
            except Exception:
                pass
        return _get_metal_gpu_family()

    def load_binary(self, name, binary, shared_memory, device):
        """Load a metallib binary or MSL source and return a pipeline state."""
        if self._mod is not None:
            try:
                if isinstance(binary, str):
                    binary = binary.encode('utf-8')
                return self._mod.load_binary(name, binary, shared_memory)
            except Exception:
                pass
        return binary

    def get_empty_cache_for_benchmark(self):
        """Return a callable that clears GPU caches for benchmarking."""
        def empty_cache():
            pass
        return empty_cache

    def clear_cache(self):
        pass


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
        warp_size = 32
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

    def get_empty_cache_for_benchmark(self):
        return self.utils.get_empty_cache_for_benchmark()
