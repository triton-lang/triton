"""Metal-specific language extensions for Triton.

Provides Metal-specific intrinsics and helper functions for Apple Silicon GPUs.
"""

import triton
from triton import language as tl


@triton.jit
def simdgroup_barrier():
    """Issue a SIMD group barrier (equivalent to __syncwarp in CUDA)."""
    return tl.inline_asm_elementwise(
        "",  # MSL: simdgroup_barrier handled at IR level
        "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def threadgroup_barrier():
    """Issue a threadgroup barrier (equivalent to __syncthreads in CUDA)."""
    return tl.inline_asm_elementwise(
        "",  # MSL: threadgroup_barrier handled at IR level
        "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


def get_simd_width():
    """Return the SIMD group width for Apple GPUs (always 32)."""
    return 32


def get_max_threadgroup_size():
    """Return the maximum threads per threadgroup on Apple Silicon."""
    return 1024


def get_max_threadgroup_memory():
    """Return the maximum threadgroup memory in bytes (32 KB on Apple Silicon)."""
    return 32768
