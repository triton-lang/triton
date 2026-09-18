from ._core import _unwrap_if_constexpr, builtin, dtype, tensor


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
