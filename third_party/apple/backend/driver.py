"""
Apple MPS Triton backend driver.

Dispatch pipeline:
  metallib bytes (from compiler.py make_metallib stage)
    → torch._C._mps_loadMetalllib(bytes)   [patched PyTorch]
    → _mps_PrecompiledShaderLibrary
    → lib.kernel_name(*args, threads=grid, group_size=block, arg_casts=...)
"""

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


# Metal setBytes only handles these casts; fp32 + tensors pass through naturally.
# i64/u64: no cast — PyTorch MPS setArg handles int64_t natively (8 bytes via setBytes).
_SCALAR_CAST = {
    "i1": "int32", "i8": "int8", "i16": "int16",
    "i32": "int32", "u32": "uint32",
    "fp16": "fp16", "bf16": "bf16",
    # i64, u64, fp32, fp64: pass as Python int/float — no cast needed
}


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

        # Build arg_casts from the expanded (flat) signature.
        slot = 0
        self.arg_casts = {}
        for ty in expanded:
            if isinstance(ty, str) and ty[0] != '*' and ty in _SCALAR_CAST:
                self.arg_casts[slot] = _SCALAR_CAST[ty]
            slot += 1

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

        # _mps_MetalKernel.__call__ signature:
        #   (*args, threads=[gx,gy,gz], group_size=[lx,ly,lz], arg_casts=dict|None)
        # Strip constexpr args — they're not in the compiled IR, so Metal slots
        # must be contiguous over non-constexpr args only.
        # Then decompose TensorDescriptor args into flat (ptr, shape, strides, ...).
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
        filtered_args = tuple(flat_args)
        import os as _os
        if _os.environ.get('TRITON_MPS_DEBUG'):
            _threads = [gridX * self.lx, gridY * self.ly, gridZ * self.lz]
            _gs = [self.lx, self.ly, self.lz]
            print(f'[MPS] threads={_threads} group_size={_gs} grid=({gridX},{gridY},{gridZ})')
            print(f'[MPS] filtered_args={filtered_args} arg_casts={self.arg_casts}')
        function(
            *filtered_args,
            threads    =[gridX * self.lx, gridY * self.ly, gridZ * self.lz],
            group_size =[self.lx, self.ly, self.lz],
            arg_casts  = self.arg_casts if self.arg_casts else None,
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
