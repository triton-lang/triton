from triton.experimental.gluon.language import _core as ttgl
from ..._core import builtin, float32, _unwrap_if_constexpr
from ..cdna3 import buffer_load_to_shared, buffer_load, buffer_store, mfma

__all__ = ["buffer_load_to_shared", "buffer_load", "buffer_store", "mfma", "mfma_scaled"]


@builtin
def mfma_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, fast_math=False, lhs_k_pack=True,
                rhs_k_pack=True, out_dtype=float32, layout=None, _semantic=None):
    out_dtype = _unwrap_if_constexpr(out_dtype)
    assert out_dtype == float32, "Only float32 is supported for out_dtype at the moment"
    tensor = _semantic.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, fast_math, lhs_k_pack,
                                  rhs_k_pack, out_dtype)

    assert layout is not None, "Layout must be specified for mfma_scaled"
    layout = _unwrap_if_constexpr(layout)
    ret_ty = ttgl.distributed_type(tensor.dtype, tensor.shape, layout)
    return ttgl.tensor(tensor.handle, ret_ty)
