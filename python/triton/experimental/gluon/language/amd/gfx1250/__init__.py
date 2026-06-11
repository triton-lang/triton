from triton.runtime.jit import constexpr_function
from triton._C.libtriton.gluon_ir import get_amd_wmma_scale_layout as _get_wmma_scale_layout

from ..._core import builtin, int8, uint8, int32, float8e4nv, tensor, _unwrap_if_constexpr
from .._ops import _wmma, _verify_wmma, _mma_scaled, _scaled_upcast
from .._layouts import AMDWMMALayout
from ..cdna3 import buffer_load, buffer_store
from ._layouts import PartitionedSharedLayout, make_partitioned_dot_layouts
from . import tdm
from . import async_copy
from . import mbarrier
from . import cluster

__all__ = [
    "async_copy", "tdm", "mbarrier", "cluster", "wmma", "wmma_scaled", "scaled_upcast", "buffer_load", "buffer_store",
    "get_wmma_scale_layout", "PartitionedSharedLayout", "make_partitioned_dot_layouts"
]


@builtin
def wmma(a, b, acc, _semantic=None):
    """
    Computes matrix-multiplication of a * b + acc using AMD WMMA instruction.

    Args:
        a (tensor): The operand a to be multiplied.
        b (tensor): The operand b to be multiplied.
        acc (tensor): The accumulator tensor.
    """
    return _wmma(3, a, b, acc, _semantic)


# List all valid combinations directly for readability
_valid_dtype_combinations = set([(dtype_a, dtype_b, "e8m0", "e8m0")
                                 for dtype_a in ("e4m3", "e5m2", "e2m1")
                                 for dtype_b in ("e4m3", "e5m2", "e2m1")] + [(dtype_a, "e2m1", "e8m0", dtype_b_scale)
                                                                             for dtype_a in ("e4m3", "e5m2")
                                                                             for dtype_b_scale in ("e4m3", )] +
                                [("e2m1", dtype_b, dtype_a_scale, "e8m0")
                                 for dtype_b in ("e4m3", "e5m2")
                                 for dtype_a_scale in ("e4m3", )] + [("e2m1", "e2m1", dtype_scale, dtype_scale)
                                                                     for dtype_scale in ("e4m3", )])


@builtin
def wmma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, _semantic=None):
    """
    AMD Scaled WMMA operation.

    ```
    c = a * a_scale @ b * b_scale + acc
    ```

    `a` and `b` use microscaling formats described in
    "OCP Microscaling Formats (MX) Specification":
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf.

    Args:
        a (tensor): The operand A to be multiplied.
        a_scale (Optional[tensor]): Scale factor for operand A.
        a_format (str): Format of the operand A. Available formats: `e2m1`, `e4m3`, `e5m2`.
        b (tensor): The operand B to be multiplied.
        b_scale (Optional[tensor]): Scale factor for operand B.
        b_format (str): Format of the operand B. Available formats: `e2m1`, `e4m3`, `e5m2`.
        acc (tensor): Accumulator tensor.
    """
    _verify_wmma(3, a, b, acc)
    a_format = _unwrap_if_constexpr(a_format)
    b_format = _unwrap_if_constexpr(b_format)
    a_scale = _unwrap_if_constexpr(a_scale)
    b_scale = _unwrap_if_constexpr(b_scale)

    if a_format == "e2m1":
        wmma_layout = a.type.layout.parent
        assert isinstance(wmma_layout, AMDWMMALayout) and wmma_layout.instr_shape in [[16, 16, 64], [32, 16, 64]], \
            "e2m1 format expects instr_shape to be [16, 16, 64] or [32, 16, 64]"
    if b_format == "e2m1":
        wmma_layout = b.type.layout.parent
        assert isinstance(wmma_layout, AMDWMMALayout) and wmma_layout.instr_shape in [[16, 16, 64], [32, 16, 64]], \
            "e2m1 format expects instr_shape to be [16, 16, 64] or [32, 16, 64]"
    acc_shapes = [[16, 16, 128]]
    if a_format == "e2m1" and b_format == "e2m1":
        acc_shapes.append([32, 16, 128])

    acc_layout = acc.type.layout
    assert isinstance(acc_layout, AMDWMMALayout) and acc_layout.instr_shape in acc_shapes, \
        f"accumulator tensor's layout must be one of {acc_shapes}"

    assert a_format in {"e2m1", "e4m3", "e5m2"}, f"Unsupported lhs_format: {a_format}"
    assert b_format in {"e2m1", "e4m3", "e5m2"}, f"Unsupported rhs_format: {b_format}"

    scale_dtype_to_format = {float8e4nv: "e4m3"}

    # E8M0 scale has various representation in frontend.
    scale_dtype_to_format.update({x: "e8m0" for x in (int8, uint8)})

    if isinstance(a_scale, tensor) and isinstance(b_scale, tensor):
        assert a_scale.dtype in scale_dtype_to_format, f"Unsupported a_scale dtype: {a_scale.dtype}"
        assert b_scale.dtype in scale_dtype_to_format, f"Unsupported b_scale dtype: {b_scale.dtype}"

        a_scale_format = scale_dtype_to_format[a_scale.dtype]
        b_scale_format = scale_dtype_to_format[b_scale.dtype]

        assert (a_format, b_format, a_scale_format, b_scale_format) in _valid_dtype_combinations, \
            f"Unsupported dtype combination: {a_format}, {b_format}, {a_scale_format}, {b_scale_format}."

    return _mma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, get_wmma_scale_layout, _semantic)


@builtin
def scaled_upcast(src, scale, elem_type, axis=None, _semantic=None):
    """
    Upcast an fp4 or fp8 tensor and fold raw E8M0 scale payload into the
    GFX1250 scaled-upcast op.

    The scale tensor must use raw E8M0 payload in `int8` or `uint8`, and must
    already have the expanded output shape and scaled-upcast result layout.
    For fp4 inputs, that is the canonical unpacked layout implied by `src`
    and `axis`. `elem_type` must be `fp16` or `bf16`. GFX1250 keeps those
    bytes in the native `cvt.scale.pk8` payload form.
    """
    axis = _unwrap_if_constexpr(axis)
    elem_type = _unwrap_if_constexpr(elem_type)
    assert scale.dtype in (int8, uint8), \
        f"Expected scale to use raw E8M0 payload in int8/uint8 but got {scale.dtype}"
    return _scaled_upcast(src, scale, elem_type, axis, _semantic)


def _get_wmma_scale_layout_impl(*args, **kwargs):
    return _get_wmma_scale_layout(*args, **kwargs)


_get_wmma_scale_layout_impl.__triton_builtin__ = True


@constexpr_function
def get_wmma_scale_layout(dot_operand_layout, shape, scale_factor=32):
    """ Get the scale layout for WMMA scaled operands.

    Args:
        dot_operand_layout (DotOperandLayout): The dot operand layout.
        shape (List[int]): The shape of the scale tensor.
        scale_factor (int): The scale factor, i.e. the number of elements of operand sharing a single scale.

    Return:
        layout (DistributedLinearLayout): The scale layout.
    """
    assert scale_factor in (16, 32), "Only support 16 or 32 scale factor"
    op_idx = dot_operand_layout.operand_index
    parent = dot_operand_layout.parent
    assert isinstance(parent, AMDWMMALayout), "Expected parent to be an instance of AMDWMMALayout"
    mdim = parent.instr_shape[0]
    ndim = parent.instr_shape[1]
    transposed = parent.transposed
    reg_bases = parent.reg_bases
    warp_bases = parent.warp_bases
    cga_bases = parent.cga_layout
    return _get_wmma_scale_layout_impl(op_idx, shape, mdim, ndim, transposed, scale_factor, reg_bases, warp_bases,
                                       cga_bases)
