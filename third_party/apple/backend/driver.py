"""
Apple MPS Triton backend driver.

Dispatch pipeline:
  metallib bytes (from compiler.py make_metallib stage)
    → torch._C._mps_loadMetalllib(bytes)   [patched PyTorch]
    → _mps_PrecompiledShaderLibrary
    → lib.kernel_name(*args, threads=grid, group_size=block, arg_casts=...)
"""

import struct as _struct
import torch
from triton.backends.driver import DriverBase, decompose_descriptor, expand_signature
from triton.tools.tensor_descriptor import TensorDescriptor


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int32_t", "i8": "int8_t", "i16": "int16_t",
        "i32": "int32_t", "i64": "int64_t",
        "u32": "uint32_t", "u64": "uint64_t",
        "fp16": "float", "bf16": "float", "fp32": "float", "fp64": "double",
    }[ty]


# Scalar type → (struct.pack format char, byte size, alignment)
_SCALAR_PACK_INFO = {
    "i1":  ("b", 1, 1),   # i1 stored as 1 byte
    "i8":  ("b", 1, 1),
    "i16": ("h", 2, 2),
    "i32": ("i", 4, 4),
    "i64": ("q", 8, 8),
    "u32": ("I", 4, 4),
    "u64": ("Q", 8, 8),
    "fp16": ("e", 2, 2),
    "bf16": ("e", 2, 2),   # bf16 → pack as fp16 (Metal treats both as 2-byte)
    "fp32": ("f", 4, 4),
    "fp64": ("d", 8, 8),
}


def _is_pointer_type(ty):
    """Check if a type string represents a pointer (tensor) argument."""
    return isinstance(ty, str) and ty.startswith('*')


def _compute_scalar_layout(scalar_types):
    """Compute packed buffer layout matching MetalASM Pass 5b.

    Returns (total_size, field_offsets) where field_offsets[i] is the byte
    offset for scalar i in the packed buffer.
    """
    offsets = []
    current = 0
    for ty in scalar_types:
        info = _SCALAR_PACK_INFO.get(ty)
        if info is None:
            raise ValueError(f"Unknown scalar type for packing: {ty}")
        _, size, align = info
        padding = (align - (current % align)) % align
        current += padding
        offsets.append(current)
        current += size
    return current, offsets


def _pack_scalars(scalar_types, scalar_values, total_size, offsets):
    """Pack scalar values into a bytes buffer matching MetalASM struct layout."""
    buf = bytearray(total_size)
    for ty, val, offset in zip(scalar_types, scalar_values, offsets):
        fmt, size, _ = _SCALAR_PACK_INFO[ty]
        if ty == "i1":
            val = 1 if val else 0
        elif ty == "bf16":
            # bf16: convert float → bf16 bits, pack as uint16
            import numpy as np
            bf16_bits = int.from_bytes(
                np.array([val], dtype=np.float32).view(np.uint32).item().to_bytes(4, 'little')[2:4],
                'little'
            )
            _struct.pack_into("H", buf, offset, bf16_bits)
            continue
        _struct.pack_into(fmt, buf, offset, val)
    return bytes(buf)


class MPSUtils:
    """
    Drop-in for CudaUtils — pure Python, no C extension needed.
    Triton calls driver.active.utils.load_binary(...).
    """

    def load_binary(self, name, metallib_bytes, shared_mem, device):
        """
        Returns (module, function, n_regs, n_spills, n_max_threads).
        - module   = the _mps_PrecompiledShaderLibrary (held alive to prevent GC)
        - function = the _mps_MetalKernel for `name`
        - n_regs, n_spills = 0 (not tracked on Metal)
        - n_max_threads    = 1024 (hardware max threadgroup size on Apple Silicon)
        """
        if not hasattr(torch._C, '_mps_loadMetalllib'):
            raise RuntimeError(
                "torch._C._mps_loadMetalllib not found — "
                "rebuild PyTorch with the MPS patch (see mps-flash-attention repo)"
            )
        module = torch._C._mps_loadMetalllib(bytes(metallib_bytes))
        function = getattr(module, name)
        return module, function, 0, 0, 1024

    def unload_module(self, module):
        del module

    def get_device_properties(self, device):
        return {
            "warpSize":            32,
            "max_shared_mem":      32768,
            "multiprocessorCount": torch._C._mps_get_core_count(),
        }

    def get_current_device(self):
        return 0

    def set_current_device(self, device):
        pass  # single MPS device

    def get_current_stream(self, device):
        return 0  # MPS manages its own stream internally


class MPSLauncher:
    """
    Called by Triton's JIT runtime to dispatch a compiled kernel.
    `function` = _mps_MetalKernel returned by load_binary.
    """

    def __init__(self, src, metadata):
        self.signature  = dict(src.signature)
        self.constants  = getattr(src, "constants", {})

        # Constexpr args appear in Python *args but NOT in the compiled IR.
        # We strip them before passing to _mps_MetalKernel so Metal buffer slots
        # match IR arg positions exactly. self.constexpr_py_slots is the set of
        # Python *args indices that are constexpr (to be stripped at launch).
        self.constexpr_py_slots = frozenset(
            i for i, (k, ty) in enumerate(self.signature.items())
            if ty == 'constexpr'
        )

        # Expand tensor descriptor types into flat scalar types.
        # MPS has no hardware TMA, so tensordesc_meta is always None —
        # descriptors are decomposed to (ptr, *shape, *strides, padding, tf32, *shape, *strides).
        non_constexpr_sig = [ty for ty in self.signature.values() if ty != 'constexpr']
        expanded = expand_signature(non_constexpr_sig, None, None)

        # Classify each expanded arg as pointer or scalar.
        # MetalASM Pass 5b packs all scalars into ONE device buffer,
        # so the IR param order is: [pointers..., packed_scalar_buf, system_values].
        # We need to separate pointers from scalars at launch time.
        self.ptr_indices = []    # indices into expanded that are pointers
        self.scalar_indices = [] # indices into expanded that are scalars
        self.scalar_types = []   # type strings for scalars (for packing)
        for slot, ty in enumerate(expanded):
            if _is_pointer_type(ty):
                self.ptr_indices.append(slot)
            else:
                self.scalar_indices.append(slot)
                self.scalar_types.append(ty)

        # Pre-compute packed buffer layout
        if self.scalar_types:
            self.total_size, self.field_offsets = _compute_scalar_layout(self.scalar_types)
        else:
            self.total_size = 0
            self.field_offsets = []

        # Threadgroup size: num_warps * 32 threads, capped at 1024
        warp_size = 32
        self.lx = min(getattr(metadata, "num_warps", 4) * warp_size, 1024)
        self.ly = 1
        self.lz = 1

    def __call__(self, gridX, gridY, gridZ, stream, function,
                 kernel_metadata, launch_metadata,
                 launch_enter_hook, launch_exit_hook, *args):

        if launch_enter_hook:
            launch_enter_hook(launch_metadata)

        # Strip constexpr args and decompose TensorDescriptors.
        from triton.runtime.jit import TensorWrapper
        flat_args = []
        for i, a in enumerate(args):
            if i in self.constexpr_py_slots:
                continue
            if isinstance(a, TensorWrapper):
                flat_args.append(a.base)
            elif isinstance(a, TensorDescriptor):
                flat_args.extend(decompose_descriptor(a))
            else:
                flat_args.append(a)

        # Separate pointer args from scalar args.
        # IR param order after Pass 5b: [ptr0, ptr1, ..., packed_scalar_buf]
        ptr_args = [flat_args[i] for i in self.ptr_indices]
        scalar_values = [flat_args[i] for i in self.scalar_indices]

        if scalar_values:
            # Pack all scalars into a small MPS tensor (acts as device buffer)
            packed_bytes = _pack_scalars(
                self.scalar_types, scalar_values,
                self.total_size, self.field_offsets
            )
            scalar_buf = torch.frombuffer(bytearray(packed_bytes), dtype=torch.uint8).to('mps')
            reordered_args = tuple(ptr_args) + (scalar_buf,)
        else:
            reordered_args = tuple(ptr_args)

        import os as _os
        if _os.environ.get('TRITON_MPS_DEBUG'):
            _threads = [gridX * self.lx, gridY * self.ly, gridZ * self.lz]
            _gs = [self.lx, self.ly, self.lz]
            print(f'[MPS] threads={_threads} group_size={_gs} grid=({gridX},{gridY},{gridZ})')
            print(f'[MPS] reordered_args={reordered_args}')
            if scalar_values:
                print(f'[MPS] scalar_types={self.scalar_types} scalar_values={scalar_values}')
                print(f'[MPS] packed_bytes={packed_bytes.hex()} total_size={self.total_size}')
        function(
            *reordered_args,
            threads    =[gridX * self.lx, gridY * self.ly, gridZ * self.lz],
            group_size =[self.lx, self.ly, self.lz],
        )

        if launch_exit_hook:
            launch_exit_hook(launch_metadata)


class MPSDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils        = MPSUtils()
        self.launcher_cls = MPSLauncher

    @staticmethod
    def is_active():
        try:
            return torch.backends.mps.is_available()
        except Exception:
            return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_current_target(self):
        from triton.backends.compiler import GPUTarget
        return GPUTarget("mps", "apple_m", 32)

    def get_active_torch_device(self):
        return torch.device("mps", 0)

    def get_current_device(self):
        return 0

    def get_current_stream(self, device):
        return 0  # MPS manages its own stream internally

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        return torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device='mps')

    def clear_cache(self, cache):
        cache.zero_()
