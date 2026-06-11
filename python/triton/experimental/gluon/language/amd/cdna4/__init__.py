from triton.runtime.jit import constexpr_function
from triton._C.libtriton.gluon_ir import (
    get_amd_mfma_scale_layout as _get_mfma_scale_layout,
    compute_amd_efficient_padded_shared_layout as _compute_efficient_padded_shared_layout,
)

from ..._core import builtin, int8, uint8, _unwrap_if_constexpr
from ..._layouts import DotOperandLayout
from .._layouts import AMDMFMALayout
from .._ops import _mma_scaled, _scaled_upcast
from ..cdna3 import _buffer_atomic_rmw_impl, _convert_e8m0_scale_to_bf16
from ..cdna3 import *  # NOQA: F403
from ..cdna3 import __all__ as __cdna3_all
from . import async_copy

__all__ = [
    *__cdna3_all,
    "async_copy",
    "mfma_scaled",
    "scaled_upcast",
    "get_mfma_scale_layout",
    "compute_efficient_padded_shared_layout",
]


@builtin
def mfma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, _semantic=None):
    """
    AMD Scaled MFMA operation.

    ```
    c = a * a_scale @ b * b_scale + acc
    ```

    `a` and `b` use microscaling formats described in
    "OCP Microscaling Formats (MX) Specification":
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf.
    Currently supported only on CDNA4 hardware.

    Args:
        a (tensor): The operand A to be multiplied.
        a_scale (Optional[tensor]): Scale factor for operand A.
        a_format (str): Format of the operand A. Available formats: `e2m1`, `e4m3`, `e5m2`.
        b (tensor): The operand B to be multiplied.
        b_scale (Optional[tensor]): Scale factor for operand B.
        b_format (str): Format of the operand B. Available formats: `e2m1`, `e4m3`, `e5m2`.
        acc (tensor): Accumulator tensor.
    """
    layout = acc.type.layout
    assert isinstance(layout, AMDMFMALayout), "Expected layout to be an instance of AMDMFMALayout"
    assert (isinstance(a.type.layout, DotOperandLayout) and a.type.layout.parent== layout), \
            "Expected lhs layout to be a DotOperandLayout with parent matching MFMA layout"
    assert (isinstance(b.type.layout, DotOperandLayout) and b.type.layout.parent == layout), \
            "Expected rhs layout to be a DotOperandLayout with parent matching MFMA layout"

    a_format = _unwrap_if_constexpr(a_format)
    b_format = _unwrap_if_constexpr(b_format)
    a_scale = _unwrap_if_constexpr(a_scale)
    b_scale = _unwrap_if_constexpr(b_scale)
    assert a_format in {"e2m1", "e4m3", "e5m2"}, f"Unsupported lhs_format: {a_format}"
    assert b_format in {"e2m1", "e4m3", "e5m2"}, f"Unsupported rhs_format: {b_format}"

    return _mma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, get_mfma_scale_layout, _semantic)


@builtin
def scaled_upcast(src, scale, elem_type, axis=None, _semantic=None):
    """
    Upcast an fp4 or fp8 tensor and fold raw E8M0 scale payload into the
    CDNA4 scaled-upcast op.

    The scale tensor must use raw E8M0 payload in `int8` or `uint8`, and must
    already have the expanded output shape and scaled-upcast result layout.
    For fp4 inputs, that is the canonical unpacked layout implied by `src`
    and `axis`. `elem_type` must be `fp16` or `bf16`. CDNA4 converts those
    bytes to the internal `bf16` scale form expected by the AMD op.
    """
    axis = _unwrap_if_constexpr(axis)
    elem_type = _unwrap_if_constexpr(elem_type)
    assert scale.dtype in (int8, uint8), \
        f"Expected scale to use raw E8M0 payload in int8/uint8 but got {scale.dtype}"
    scale = _convert_e8m0_scale_to_bf16(scale, _semantic=_semantic)
    return _scaled_upcast(src, scale, elem_type, axis, _semantic)


def _get_mfma_scale_layout_impl(*args, **kwargs):
    return _get_mfma_scale_layout(*args, **kwargs)


_get_mfma_scale_layout_impl.__triton_builtin__ = True


@constexpr_function
def get_mfma_scale_layout(dot_operand_layout, shape, scale_factor=32):
    """ Get the scale layout for MFMA scaled operands.

    Args:
        dot_operand_layout (DotOperandLayout): The dot operand layout.
        shape (List[int]): The shape of the scale tensor.
        scale_factor (int): The scale factor.
    Return:
        layout (DistributedLinearLayout): The scale layout.
    """
    assert scale_factor == 32, "Only scale factor 32 is supported for CDNA4 Scaled MFMA"
    op_idx = dot_operand_layout.operand_index
    parent = dot_operand_layout.parent
    assert isinstance(parent, AMDMFMALayout), "Expected parent to be an instance of AMDMFMALayout"
    mdim = parent.instr_shape[0]
    tiles_per_warp = parent.tiles_per_warp
    warps_per_cta = parent.warps_per_cta
    return _get_mfma_scale_layout_impl(op_idx, shape, mdim, tiles_per_warp, warps_per_cta)


def _compute_efficient_padded_shared_layout_impl(*args, **kwargs):
    return _compute_efficient_padded_shared_layout(*args, **kwargs)


_compute_efficient_padded_shared_layout_impl.__triton_builtin__ = True


@constexpr_function
def compute_efficient_padded_shared_layout(dot_operand_layout, shape, dtype, is_k_contig=True):
    """Compute an efficient padded shared layout for the given parameters
    that avoids bank conflicts as much as possible.

    Args:
        dot_operand_layout (DotOperandLayout): The layout for the dot operand
            that will be copied to shared memory with padding. Must have an
            AMDMFMALayout v4 (CDNA4) parent.
        shape (List[int]): Shared memory tile shape for the dot operand —
            ``[BM, BK]`` for operand A or ``[BK, BN]`` for operand B.
        dtype (dtype): Element type of the tensor that will live in this
            shared memory allocation (e.g. ``ttgl.float16``, ``ttgl.float8e4nv``).
            Only types with bitwidth in {4, 8, 16} are supported. For packed
            fp4 (two values per byte), pass ``ttgl.uint8`` — at the LDS level
            4-bit shares the 8-bit padding pattern.
        is_k_contig (bool): K is the contiguous dim in shared memory.
    Return:
        layout (PaddedSharedLayout): or None if the input falls outside the
            supported set. Common reasons for None: ``k_width`` not in
            {4, 8, 16}; element bitwidth not in {4, 8, 16}; MFMA instruction
            shape or kWidth combination not handled by the underlying
            algorithm.
    """
    parent = dot_operand_layout.parent
    assert isinstance(parent, AMDMFMALayout), \
        "Expected dot operand's parent to be an AMDMFMALayout"
    assert parent.version == 4, \
        "compute_efficient_padded_shared_layout only supports MFMA v4 (CDNA4)"
    return _compute_efficient_padded_shared_layout_impl(
        dot_operand_layout.operand_index,
        dot_operand_layout.k_width,
        parent.version,
        list(parent.warps_per_cta),
        list(parent.instr_shape),
        parent.transposed,
        list(parent.tiles_per_warp),
        parent.element_bitwidth,
        list(parent.cga_layout),
        list(shape),
        dtype.primitive_bitwidth,
        is_k_contig,
    )


"""
buffer_atomic_rmw of cnda4 shares the same signature and functionalities as cdna3.buffer_atomic_rmw.
The cdna4 version additionally supports `fadd` with `bf16`.
"""


@builtin
def buffer_atomic_max(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):
    return _buffer_atomic_rmw_impl('max', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_min(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('min', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_add(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('add', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_and(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('and', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_or(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('or', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_xor(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('xor', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_xchg(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('xchg', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)
