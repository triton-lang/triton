import functools
import os
import subprocess
import triton
import re
from pathlib import Path
from triton import knobs
from triton.runtime.build import compile_module_from_src
from triton.runtime import _allocation
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver

dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [os.path.join(dirname, "include")]
libdevice_dir = os.path.join(dirname, "lib")
libraries = ['libcuda.so.1']
PyCUtensorMap = None
PyKernelArg = None
ARG_CONSTEXPR = None
ARG_KERNEL = None
ARG_TUPLE = None


@functools.lru_cache()
def libcuda_dirs():
    if env_libcuda_path := knobs.nvidia.libcuda_path:
        return [env_libcuda_path]

    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
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
        mod = compile_module_from_src(
            src=Path(os.path.join(dirname, "driver.c")).read_text(),
            name="cuda_utils",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )
        global PyCUtensorMap
        global PyKernelArg
        global ARG_CONSTEXPR
        global ARG_KERNEL
        global ARG_TUPLE
        PyCUtensorMap = mod.PyCUtensorMap
        PyKernelArg = mod.PyKernelArg
        ARG_CONSTEXPR = mod.ARG_CONSTEXPR
        ARG_KERNEL = mod.ARG_KERNEL
        ARG_TUPLE = mod.ARG_TUPLE
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.cuOccupancyMaxActiveClusters = mod.cuOccupancyMaxActiveClusters
        self.set_printf_fifo_size = mod.set_printf_fifo_size
        self.fill_tma_descriptor_tiled = mod.fill_tma_descriptor_tiled
        self.fill_tma_descriptor_im2col = mod.fill_tma_descriptor_im2col
        self.launch = mod.launch
        self.build_signature_metadata = mod.build_signature_metadata


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "CUdeviceptr"
    if ty.startswith("tensordesc"):
        return "CUtensorMap"
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
        "fp16": "double",
        "bf16": "double",
        "fp32": "double",
        "f32": "double",
        "fp64": "double",
        "nvTmaDesc": "CUtensorMap",
    }[ty]


def expand_signature(signature, tensordesc_meta):
    output = []
    tensordesc_idx = 0
    # Expand tensor descriptor arguments into either nvTmaDesc, shape and
    # strides, or base pointer, shape and strides depending on whether the
    # kernel was lowered to use the nvTmaDesc or not.
    for sig in signature:
        if isinstance(sig, str) and sig.startswith("tensordesc"):
            meta = tensordesc_meta[tensordesc_idx] if tensordesc_meta else None
            tensordesc_idx += 1

            match = re.match("tensordesc<([^[>]*)\\[([^]]*)\\]", sig)
            dtype = match.group(1)
            shape = match.group(2)
            ndim = shape.count(",") + 1

            if meta is None:
                output.append("*" + dtype)
                # Currently the host side tensor descriptors get passed in as a
                # tensor desc, shape, and strides. We have no way to use these
                # shape and strides when processing tensor descriptors which is
                # why we provide our own decomposition above. Sadly this means
                # we have to pass the shape and strides twice.
                for _ in range(2 * ndim):
                    output.append("i64")
                output.append("i1")
                output.append("i1")
            else:
                output.append("nvTmaDesc")

            for _ in range(ndim):
                output.append("i32")
            for _ in range(ndim):
                output.append("i64")
        else:
            output.append(sig)

    assert not tensordesc_meta or tensordesc_idx == len(tensordesc_meta)
    return output


def make_kernel_signature(signature):
    """
    Creates a kernel signature in C to be able to efficiently extract
    arguments in the launcher.
    """

    def _flatten_signature(sig, output):
        # Flatten tuples
        if isinstance(sig, tuple):
            for x in sig:
                _flatten_signature(x, output)
        else:
            output.append(sig)

    flat_signature = []
    for sig in signature:
        _flatten_signature(sig, flat_signature)
    kernel_signature = [x for x in flat_signature if x != "constexpr"]

    return triton.runtime.driver.active.utils.build_signature_metadata(kernel_signature)


def annotate_arguments(signature):
    """
    This recreates the signature with annotations as C objects which can then
    be used to efficiently flatten tuples, and remove constexpr in the launcher.
    """
    annotated_arguments = []
    for sig in signature:
        if isinstance(sig, tuple):
            annotated_arguments.append((PyKernelArg(nested_tuple=annotate_arguments(sig), type=ARG_TUPLE)))
        elif sig != "constexpr":
            annotated_arguments.append(PyKernelArg(nested_tuple=None, type=ARG_KERNEL))
        else:
            annotated_arguments.append(PyKernelArg(nested_tuple=None, type=ARG_CONSTEXPR))
    return annotated_arguments


# The TMA dtype enum values are slightly different on host vs device...
TMA_DTYPE_DEVICE_TO_HOST = dict((i, i) for i in range(16))
TMA_DTYPE_DEVICE_TO_HOST[8] = 10
TMA_DTYPE_DEVICE_TO_HOST[9] = 8
TMA_DTYPE_DEVICE_TO_HOST[10] = 9
TMA_TF32 = 11


def make_tensordesc_arg(arg, metadata):
    if metadata is None:
        # Currently the host side tensor descriptors get decomposed in
        # the frontend to tensor desc, shape, and strides. We have no
        # way to use these shape and strides when processing tensor
        # descriptors which is why we provide our own decomposition
        # above. Sadly this means we have to pass the shape and strides
        # twice.
        return [
            arg.base,
            *arg.shape,
            *arg.strides,
            arg.padding == "nan",
            arg.round_f32_to_tf32,
            *arg.shape,
            *arg.strides,
        ]

    swizzle = metadata["swizzle"]
    elem_size = metadata["elem_size"]
    elem_type = metadata["elem_type"]
    block_size = metadata["block_size"]
    fp4_padded = metadata["fp4_padded"]

    shape = arg.shape
    strides = arg.strides
    assert strides[-1] == 1
    padding = 1 if arg.padding == "nan" else 0

    if fp4_padded:
        expanded_shape = list(shape)
        expanded_shape[-1] *= 2
    else:
        expanded_shape = shape

    if arg.round_f32_to_tf32:
        elem_type = TMA_TF32

    cu_tensor_map = triton.runtime.driver.active.utils.fill_tma_descriptor_tiled(
        arg.base.data_ptr(),
        swizzle,
        elem_size,
        TMA_DTYPE_DEVICE_TO_HOST[elem_type],
        block_size,
        expanded_shape,
        strides,
        padding,
    )

    return [cu_tensor_map, *shape, *strides]


def wrap_handle_tensordesc(launcher, signature, tensordesc_meta):
    has_tensor_desc_arg = any(isinstance(sig, str) and sig.startswith("tensordesc") for sig in signature.values())
    if not has_tensor_desc_arg:
        return launcher

    tensordesc_indices = set(
        [i for i, sig in enumerate(signature.values()) if isinstance(sig, str) and sig.startswith("tensordesc")])
    assert not tensordesc_meta or len(tensordesc_meta) == len(tensordesc_indices)
    if not tensordesc_meta:
        tensordesc_meta = [None] * len(tensordesc_indices)

    def inner(*args):
        base_args = args[:-1]
        kernel_args = args[-1]

        final_kernel_args = []
        tensordesc_idx = 0
        for i, arg in enumerate(kernel_args):
            if i in tensordesc_indices:
                final_kernel_args.extend(make_tensordesc_arg(arg, tensordesc_meta[tensordesc_idx]))
                tensordesc_idx += 1
            else:
                final_kernel_args.append(arg)

        return launcher(*base_args, final_kernel_args)

    return inner


class CudaLauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)

        launcher = triton.runtime.driver.active.utils.launch
        expanded_signature = expand_signature(signature.values(), tensordesc_meta)
        self.arg_annotations = annotate_arguments(expanded_signature)
        self.kernel_signature = make_kernel_signature(expanded_signature)
        self.num_ctas = getattr(metadata, "num_ctas", 1)
        self.launch = wrap_handle_tensordesc(launcher, signature, tensordesc_meta)
        self.global_scratch_size = metadata.global_scratch_size
        self.global_scratch_align = metadata.global_scratch_align
        self.profile_scratch_size = metadata.profile_scratch_size
        self.profile_scratch_align = metadata.profile_scratch_align
        self.launch_cooperative_grid = metadata.launch_cooperative_grid
        self.launch_pdl = metadata.launch_pdl

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                 launch_exit_hook, *args):

        def allocate_scratch(size, align, allocator):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * self.num_ctas * size
                alloc_fn = allocator.get()
                return alloc_fn(alloc_size, align, stream)
            return None

        global_scratch = allocate_scratch(self.global_scratch_size, self.global_scratch_align, _allocation._allocator)
        profile_scratch = allocate_scratch(self.profile_scratch_size, self.profile_scratch_align,
                                           _allocation._profile_allocator)

        self.launch(gridX, gridY, gridZ, stream, function, self.launch_cooperative_grid, self.launch_pdl,
                    kernel_metadata, launch_metadata, launch_enter_hook, launch_exit_hook, global_scratch,
                    profile_scratch, self.arg_annotations, self.kernel_signature, args)


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

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

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
