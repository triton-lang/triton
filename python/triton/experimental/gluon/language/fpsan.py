from triton.language import fpsan as tl_fpsan

from ._core import _unwrap_if_constexpr, builtin, dtype, tensor

barrier = builtin(tl_fpsan.barrier)
homomorphic_cast = builtin(tl_fpsan.homomorphic_cast)


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


@builtin
def store_probability(dst, value, expected_zero=None, _semantic=None):
    """Round an FP32 probability tile to the destination FP8 format, store it in TMEM and return it.

    expected_zero has the semantics of expect_zero on the rounded value.
    In FPSan, an unscaled FP32-accumulating tcgen05_mma reads the tagged
    full-width payload from the stored bytes. Register uses see the narrow result.
    """
    from triton._instrumentation import is_enabled
    from ._core import float8e4nv, float8e5, float16, float32, expect_zero
    from .nvidia.blackwell import tensor_memory_descriptor

    if not isinstance(dst, tensor_memory_descriptor) or dst.dtype not in (float8e4nv, float8e5):
        raise ValueError("store_probability requires an E4M3 or E5M2 TMEM destination")
    if value.dtype != float32 or value.shape != dst.shape or len(value.shape) != 2:
        raise ValueError("store_probability requires a matching 2D FP32 tile")
    expected_zero = _unwrap_if_constexpr(expected_zero)
    options = _semantic.builder.options
    if is_enabled(options, "fpsan"):
        if expected_zero is not None:
            value = expect_zero(value, expected_zero, _semantic=_semantic)
        ty = value.type.with_element_ty(dst.dtype)
        handle = _semantic.builder.create_fpsan_store_probability(ty.to_ir(_semantic.builder), dst.handle, value.handle)
        return tensor(handle, ty)

    rounded = _semantic.cast(value, dst.dtype)
    if options.debug and expected_zero is not None:
        expect_zero(_semantic.cast(rounded, float16), expected_zero, _semantic=_semantic)
    dst.store(rounded, _semantic=_semantic)
    return rounded
