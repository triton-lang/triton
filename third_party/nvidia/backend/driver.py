import array
from collections.abc import Callable, Iterator, MutableSequence, Sequence
import functools
import inspect
import operator
import os
import subprocess
from typing import Any
import triton
from triton import knobs
from triton.runtime import _allocation
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver
from ._C import cuda_utils

from triton.tools.tensor_descriptor import TensorDescriptor

dirname = os.path.dirname(os.path.realpath(__file__))
include_dir = [os.path.join(dirname, "include")]
libdevice_dir = os.path.join(dirname, "lib")


@functools.lru_cache()
def libcuda_dirs():
    if env_libcuda_path := knobs.nvidia.libcuda_path:
        return [env_libcuda_path]

    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()
    # each line looks like the following:
    # libcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1
    locs = [line.split()[-1] for line in libs.splitlines() if "libcuda.so.1" in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path and not dirs:
        dirs = [dir for dir in env_ld_library_path.split(":") if os.path.exists(os.path.join(dir, "libcuda.so.1"))]
    msg = 'libcuda.so cannot found!\n'
    if locs:
        msg += 'Possible files are located at %s.' % str(locs)
        msg += 'Please create a symlink of libcuda.so to any of the files.'
    else:
        msg += 'Please make sure GPU is set up and then run "/sbin/ldconfig"'
        msg += ' (requires sudo) to refresh the linker cache.'
    assert any(os.path.exists(os.path.join(path, 'libcuda.so.1')) for path in dirs), msg
    return dirs


@functools.lru_cache()
def library_dirs():
    return [libdevice_dir, *libcuda_dirs()]


# ------------------------
# Utils
# ------------------------


class CudaUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CudaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.load_binary = cuda_utils.load_binary
        self.get_device_properties = cuda_utils.get_device_properties
        self.cuOccupancyMaxActiveClusters = cuda_utils.cuOccupancyMaxActiveClusters
        self.set_printf_fifo_size = cuda_utils.set_printf_fifo_size
        self.fill_tma_descriptor = cuda_utils.fill_tma_descriptor


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*' or ty == "none":
        return "CUdeviceptr"
    if ty.startswith("tensordesc"):
        return "CUtensorMap"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
        "nvTmaDesc": "CUtensorMap",
    }[ty]


# A nested sequence of arg type strings.
_ArgTypeWithNesting = str | Sequence[Any]

# Nested mask that has True for elements that should be kept and False for
# elements that should be removed. Has the same shape as the signature.
_ArgMask = Sequence[bool | Any]


def _make_nonconst_arg_mask(signature_types: Sequence[_ArgTypeWithNesting]) -> _ArgMask:
    """Makes a mask that keeps non-constexpr args and removes constexpr args."""
    # For example:
    #   Signature: [i32, constexpr, (i32, constexpr)]
    #   Mask:      [True, False, [True, False]]
    return [_make_nonconst_arg_mask(ty) if isinstance(ty, tuple) else ty != "constexpr" for ty in signature_types]


def _flatten_tuples(xs):
    """Recursively flattens tuple elements in xs."""
    for x in xs:
        if isinstance(x, tuple):
            yield from _flatten_tuples(x)
        else:
            yield x


def _flatten_and_apply_arg_mask(args: Sequence[Any], mask: _ArgMask) -> Iterator[Any]:
    """Flattens nested args skipping those filtered out by the mask."""
    if len(mask) != len(args):
        # If the included elements in the mask are the same length as the args,
        # we can assume the caller filtered the args already.
        # Otherwise there is an unexpected length mismatch.
        if mask.count(True) == len(args):
            yield from _flatten_tuples(args)
            return
        else:
            raise ValueError(f"Mask length {len(mask)} does not match arg length {len(args)}")

    for mask_item, arg in zip(mask, args):
        if not mask_item:
            continue
        arg_is_sequence = isinstance(arg, Sequence) and not isinstance(arg, str)
        mask_item_is_sequence = isinstance(mask_item, Sequence)
        if arg_is_sequence and mask_item_is_sequence:
            yield from _flatten_and_apply_arg_mask(arg, mask_item)
        elif arg_is_sequence != mask_item_is_sequence:
            raise ValueError(f"Inconsistent mask {mask_item} and arg {arg}")
        else:
            yield arg


def make_launcher(signature_types: Sequence[_ArgTypeWithNesting]) -> Callable[..., None]:
    # Expands tensordesc with the type and block shapes like <fp16[128, 16]>
    # into an nvTmaDesc, shapes, and strides.
    # This is the signature-handling counterpart to `make_tensordesc_arg`.
    def _expand_signature(sig: _ArgTypeWithNesting, output: MutableSequence[_ArgTypeWithNesting]):
        # Expand tensordesc arguments
        if isinstance(sig, str) and sig.startswith("tensordesc"):
            output.append("nvTmaDesc")
            ndim = sig.count(",") + 1
            for _ in range(ndim):
                output.append("i32")
            for _ in range(ndim):
                output.append("i64")
        else:
            output.append(sig)

    expand_signature = []
    for sig in signature_types:
        _expand_signature(sig, expand_signature)

    signature_types = expand_signature
    non_const_arg_mask = _make_nonconst_arg_mask(signature_types)
    flattened_signature = _flatten_and_apply_arg_mask(signature_types, non_const_arg_mask)

    signature_metadata = cuda_utils.build_signature_metadata(flattened_signature)

    def wrapper(grid_dim_x: int, grid_dim_y: int, grid_dim_z: int, stream: int, kernel: int, global_scratch: any,
                packed_metadata: tuple[int, int, int, int, int, int], hook_args: any,
                launch_enter_hook: Callable[..., None], launch_exit_hook: Callable[..., None], *args: any) -> None:
        non_const_args = _flatten_and_apply_arg_mask(args, non_const_arg_mask)

        cuda_utils.launch(grid_dim_x, grid_dim_y, grid_dim_z, stream, kernel, packed_metadata, hook_args,
                          launch_enter_hook, launch_exit_hook, signature_metadata, global_scratch, non_const_args)

    return wrapper


class TmaDescKernelParam:
    TMA_DESC_SIZE = 128
    _ALIGN = 64

    def __init__(self):
        # Add the alignment to the array size to ensure that the address can be
        # aligned without access going out of bounds.
        self._array = array.array('B', [0] * (self.TMA_DESC_SIZE + self._ALIGN))
        address, num_bytes = self._array.buffer_info()
        # Shift the address to the nearest multiple of the alignment.
        self._aligned_address = address + self._ALIGN - (address % self._ALIGN)
        assert self._aligned_address + self.TMA_DESC_SIZE <= address + num_bytes
        assert self._aligned_address % self._ALIGN == 0

    # Return a CUtensorMap* pointer in host memory
    def tma_desc_cpu_ptr(self):
        return self._aligned_address


# The TMA dtype enum values are slightly different on host vs device...
TMA_DTYPE_DEVICE_TO_HOST = dict((i, i) for i in range(16))
TMA_DTYPE_DEVICE_TO_HOST[8] = 10
TMA_DTYPE_DEVICE_TO_HOST[9] = 8
TMA_DTYPE_DEVICE_TO_HOST[10] = 9


def make_tensordesc_arg(arg, metadata):
    assert isinstance(arg, TensorDescriptor)
    swizzle = metadata["swizzle"]
    elem_size = metadata["elem_size"]
    elem_type = metadata["elem_type"]
    block_size = metadata["block_size"]
    fp4_padded = metadata["fp4_padded"]

    data_ptr = arg.base.data_ptr()
    shape = arg.shape
    strides = arg.strides
    assert strides[-1] == 1

    desc = TmaDescKernelParam()
    result = [desc, *shape, *strides]

    if fp4_padded:
        shape[-1] *= 2
    triton.runtime.driver.active.utils.fill_tma_descriptor(
        desc.tma_desc_cpu_ptr(),
        data_ptr,
        swizzle,
        elem_size,
        TMA_DTYPE_DEVICE_TO_HOST[elem_type],
        block_size,
        shape,
        strides,
    )
    return result


def get_var_positional_arg_index(launcher: Callable[..., None]) -> int | None:
    """Returns the index of the variable positional argument in a callable."""
    launcher_sig = inspect.signature(launcher)
    for i, param in enumerate(launcher_sig.parameters.values()):
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return i
    return None


def wrap_handle_tensordesc(launcher, tensordesc_meta):
    if not tensordesc_meta:
        return launcher

    # Get the index of the `*args` entry in the launcher signature.
    var_positional_arg = get_var_positional_arg_index(launcher)
    assert var_positional_arg is not None

    def inner(*args):
        meta_args = args[:var_positional_arg]
        raw_kernel_args = args[var_positional_arg:]
        tensordesc_idx = 0
        final_args = []
        for i, arg in enumerate(raw_kernel_args):
            if isinstance(arg, TensorDescriptor):
                meta = tensordesc_meta[tensordesc_idx]
                tensordesc_idx += 1
                final_args.extend(make_tensordesc_arg(arg, meta))
            else:
                final_args.append(arg)
        assert tensordesc_idx == len(tensordesc_meta)
        return launcher(*meta_args, *final_args)

    return inner


class CudaLauncher(object):

    def __init__(self, src, metadata):
        signature = {idx: value for idx, value in src.signature.items()}
        self.num_ctas = functools.reduce(operator.mul, metadata.cluster_dims, 1)
        launch = make_launcher(signature.values())
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)
        if tensordesc_meta is not None:
            self.launch = wrap_handle_tensordesc(launch, tensordesc_meta)
        else:
            self.launch = launch
        self.global_scratch_size = metadata.global_scratch_size
        self.global_scratch_align = metadata.global_scratch_align
        self.launch_cooperative_grid = metadata.launch_cooperative_grid

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        if self.global_scratch_size > 0:
            grid_size = gridX * gridY * gridZ
            alloc_size = grid_size * self.num_ctas * self.global_scratch_size
            global_scratch = _allocation._allocator(alloc_size, self.global_scratch_align, stream)
        else:
            global_scratch = None
        self.launch(gridX, gridY, gridZ, stream, function, global_scratch, *args)


class CudaDriver(GPUDriver):

    def __init__(self):
        self.utils = CudaUtils()  # TODO: make static
        self.launcher_cls = CudaLauncher
        super().__init__()

    def get_current_target(self):
        device = self.get_current_device()
        capability = self.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
        warp_size = 32
        return GPUTarget("cuda", capability, warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("cuda", self.get_current_device())

    def get_device_interface(self):
        import torch
        return torch.cuda

    @staticmethod
    def is_active():
        try:
            import torch
            return torch.cuda.is_available() and (torch.version.hip is None)
        except ImportError:
            return False

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')

    def clear_cache(self, cache):
        cache.zero_()
