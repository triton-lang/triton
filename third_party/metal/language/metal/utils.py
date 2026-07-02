"""Metal-specific utility intrinsics for Triton.

Provides hardware query functions and compile-time builtins for Apple Silicon GPUs.
This is the Metal equivalent of CUDA's utils.py, adapted for the Metal execution model.

Metal terminology mapping:
  - SIMD group     = warp (32 threads on Apple Silicon)
  - Threadgroup    = CTA / block
  - Thread         = individual lane
  - Grid           = NDRange

Apple GPUs always have a SIMD width of 32 (simdgroup size).
"""

from triton.language import core

# ---------------------------------------------------------------------------
# Hardware timer
# ---------------------------------------------------------------------------


@core.extern
def gpu_timestamp(_semantic=None):
    """Read the Metal GPU timestamp counter.

    Returns a 64-bit timestamp value from the GPU's hardware clock.
    Useful for fine-grained performance measurement within a kernel.

    Note: On Apple Silicon, the timestamp counter is accessible via
    the Metal GPU timestamp intrinsic. The resolution and behavior
    may vary across GPU families (M1/M2/M3/M4).
    """
    return core.inline_asm_elementwise("mov.u64 $0, %clock64;",  # Placeholder: lowered by Metal backend
                                       "=l", [], dtype=core.int64, is_pure=False, pack=1, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Thread / SIMD group / Threadgroup position queries
# ---------------------------------------------------------------------------


@core.extern
def simdgroup_id(_semantic=None):
    """Return the SIMD group index within the threadgroup.

    Equivalent to CUDA's smid() but at threadgroup scope.
    In MSL: [[simdgroup_index_in_threadgroup]]

    Returns:
        int32: The index of this thread's SIMD group within the threadgroup.
    """
    return core.inline_asm_elementwise("mov.u32 $0, %simdgroup_index_in_threadgroup;", "=r", [], dtype=core.int32,
                                       is_pure=True, pack=1, _semantic=_semantic)


@core.extern
def threadgroup_id(_semantic=None):
    """Return the threadgroup position in the compute grid (1D).

    In MSL: [[threadgroup_position_in_grid]].x

    Returns:
        int32: The linear threadgroup index within the dispatch grid.
    """
    return core.inline_asm_elementwise("mov.u32 $0, %threadgroup_position_in_grid;", "=r", [], dtype=core.int32,
                                       is_pure=True, pack=1, _semantic=_semantic)


@core.extern
def thread_index_in_simdgroup(_semantic=None):
    """Return the thread's lane index within its SIMD group.

    In MSL: [[thread_index_in_simdgroup]]
    Equivalent to CUDA's lane_id (value in [0, 31]).

    Returns:
        int32: Lane index within the SIMD group, range [0, 31].
    """
    return core.inline_asm_elementwise("mov.u32 $0, %thread_index_in_simdgroup;", "=r", [], dtype=core.int32,
                                       is_pure=True, pack=1, _semantic=_semantic)


@core.extern
def thread_index_in_threadgroup(_semantic=None):
    """Return the thread's linear index within the threadgroup.

    In MSL: [[thread_index_in_threadgroup]]
    Equivalent to CUDA's threadIdx.x (for 1D threadgroups).

    Returns:
        int32: Thread position within the threadgroup.
    """
    return core.inline_asm_elementwise("mov.u32 $0, %thread_index_in_threadgroup;", "=r", [], dtype=core.int32,
                                       is_pure=True, pack=1, _semantic=_semantic)


@core.extern
def thread_position_in_grid(_semantic=None):
    """Return the thread's global position in the compute grid (1D).

    In MSL: [[thread_position_in_grid]].x

    Returns:
        int32: Global thread index across the entire dispatch.
    """
    return core.inline_asm_elementwise("mov.u32 $0, %thread_position_in_grid;", "=r", [], dtype=core.int32,
                                       is_pure=True, pack=1, _semantic=_semantic)


@core.extern
def threads_per_threadgroup(_semantic=None):
    """Return the number of threads per threadgroup (runtime value).

    In MSL: [[threads_per_threadgroup]].x

    Returns:
        int32: Total threads in this threadgroup.
    """
    return core.inline_asm_elementwise("mov.u32 $0, %threads_per_threadgroup;", "=r", [], dtype=core.int32,
                                       is_pure=True, pack=1, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Compile-time constants
# ---------------------------------------------------------------------------


@core.builtin
def num_threads_per_simdgroup(_semantic=None):
    """Return the number of threads per SIMD group (compile-time constant).

    On Apple Silicon GPUs, the SIMD group width is always 32.

    Returns:
        constexpr(32): The SIMD group width.
    """
    return core.constexpr(32)


@core.builtin
def num_simdgroups_per_threadgroup(_semantic=None):
    """Return the number of SIMD groups per threadgroup (compile-time).

    This is determined by the num_warps compilation option. Each "warp"
    in Triton terminology maps to one SIMD group on Metal.

    Returns:
        constexpr: Number of SIMD groups (== num_warps).
    """
    return core.constexpr(_semantic.builder.options.num_warps)


@core.builtin
def num_threads(_semantic=None):
    """Return the total number of threads per threadgroup (compile-time).

    Computed as num_warps * 32 (SIMD group width).
    This is the Metal equivalent of CUDA's num_threads().

    Returns:
        constexpr: Total threads per threadgroup.
    """
    return core.constexpr(_semantic.builder.options.num_warps * 32)


@core.builtin
def num_warps(_semantic=None):
    """Return the number of warps (SIMD groups) per threadgroup (compile-time).

    On Metal, each warp corresponds to one SIMD group of 32 threads.
    This is an alias for num_simdgroups_per_threadgroup for compatibility
    with code that uses CUDA-style terminology.

    Returns:
        constexpr: Number of SIMD groups per threadgroup.
    """
    return core.constexpr(_semantic.builder.options.num_warps)


@core.builtin
def warp_size(_semantic=None):
    """Return the warp (SIMD group) size (compile-time constant).

    Always 32 on Apple Silicon.

    Returns:
        constexpr(32): The SIMD group width.
    """
    return core.constexpr(32)
