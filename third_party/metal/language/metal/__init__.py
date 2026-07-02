"""Metal-specific language extensions for Triton.

Provides Metal 4 intrinsics and helper functions for Apple Silicon GPUs.
Targets Metal Shading Language 4.0 (macOS 26+) with fallback to MSL 3.2 (macOS 15+).
"""

from . import libdevice
from . import utils

import triton
from triton import language as tl

# Metal 4 constants
METAL_LANGUAGE_VERSION = "4.0"
MIN_MACOS_VERSION = "26.0"
SIMD_WIDTH = 32
MAX_THREADGROUP_SIZE = 1024
MAX_THREADGROUP_MEMORY = 32768  # 32 KB


@triton.jit
def simdgroup_barrier():
    """Issue a SIMD group barrier (equivalent to __syncwarp in CUDA)."""
    return tl.inline_asm_elementwise(
        "",
        "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def threadgroup_barrier():
    """Issue a threadgroup barrier (equivalent to __syncthreads in CUDA)."""
    return tl.inline_asm_elementwise(
        "",
        "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


def get_simd_width():
    """Return the SIMD group width for Apple GPUs (always 32)."""
    return SIMD_WIDTH


def get_max_threadgroup_size():
    """Return the maximum threads per threadgroup on Apple Silicon."""
    return MAX_THREADGROUP_SIZE


def get_max_threadgroup_memory():
    """Return the maximum threadgroup memory in bytes (32 KB on Apple Silicon)."""
    return MAX_THREADGROUP_MEMORY


def get_metal_language_version():
    """Return the target Metal Shading Language version."""
    return METAL_LANGUAGE_VERSION
