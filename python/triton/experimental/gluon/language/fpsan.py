from triton._instrumentation import is_enabled
from triton.language import fpsan as tl_fpsan

from ._core import _unwrap_if_constexpr, builtin, dtype, float8e4nv, float8e5, float16, float32, tensor
from ._core import expect_zero as _expect_zero

barrier = builtin(tl_fpsan.barrier)
compact_exp2 = builtin(tl_fpsan.compact_exp2)
homomorphic_cast = builtin(tl_fpsan.homomorphic_cast)


@builtin
def probability_cast(value, dtype, expect_zero=None, _semantic=None):
    """Convert an exp2 probability tensor between FP32 and E4M3 or E5M2.

    With FPSan enabled, narrowing packs a compact_exp2 result into one byte.
    Widening expands it with an FP8 format marker.
    Pass a_type="fpsan_probability" or b_type="fpsan_probability" to
    tcgen05_mma for the same expansion at the operand read.

    Otherwise this is a regular cast. expect_zero marks masked entries after
    rounding, with the semantics of gl.expect_zero.
    """
    dtype = _unwrap_if_constexpr(dtype)
    fp8 = (float8e4nv, float8e5)
    if not isinstance(value, tensor) or not value.type.is_block():
        raise ValueError("probability_cast requires a tensor")
    if not ((value.dtype == float32 and dtype in fp8) or (value.dtype in fp8 and dtype == float32)):
        raise ValueError("probability_cast requires a conversion between FP32 and E4M3 or E5M2")
    builder = _semantic.builder
    fpsan = is_enabled(builder.options, "fpsan")
    if fpsan:
        ty = value.type.with_element_ty(dtype)
        handle = builder.create_fpsan_probability_cast(value.handle, ty.to_ir(builder))
        result = tensor(handle, ty)
    else:
        result = _semantic.cast(value, dtype)
    mask = _unwrap_if_constexpr(expect_zero)
    if mask is not None:
        if fpsan:
            return _expect_zero(result, mask, _semantic=_semantic)
        if builder.options.debug:
            check = _semantic.cast(result, float16) if dtype in fp8 else result
            _expect_zero(check, mask, _semantic=_semantic)
    return result


@builtin
def embed(input, _semantic=None):
    """Embed a floating-point tensor in the same-width FPSan integer ring."""
    input = _semantic.to_tensor(input)
    result_ty = input.type.with_element_ty(dtype(f"int{input.dtype.primitive_bitwidth}"))
    handle = _semantic.builder.create_experimental_fpsan_embed(input.handle, result_ty.to_ir(_semantic.builder))
    return tensor(handle, result_ty)


@builtin
def unembed(input, dtype: dtype, _semantic=None):
    """Unembed an FPSan integer payload into the given floating-point dtype."""
    input = _semantic.to_tensor(input)
    result_ty = input.type.with_element_ty(_unwrap_if_constexpr(dtype))
    handle = _semantic.builder.create_experimental_fpsan_unembed(input.handle, result_ty.to_ir(_semantic.builder))
    return tensor(handle, result_ty)
