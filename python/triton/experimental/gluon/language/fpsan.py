from triton._instrumentation import is_enabled
from triton.language.core import _aggregate
from triton.language import fpsan as tl_fpsan

from ._core import _unwrap_if_constexpr, builtin, constexpr, dtype, expect_zero, float8e4nv, float8e5, float16, float32, tensor

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


@_aggregate
class _ProbabilityStore:
    value: tensor
    mask: tensor | constexpr

    def _store(self, dst, semantic, pred=None):
        value = self.value
        mask = _unwrap_if_constexpr(self.mask)
        if dst.dtype not in (float8e4nv, float8e5):
            raise ValueError("probability_store requires an E4M3 or E5M2 destination")
        if value.dtype != float32 or len(value.shape) != 2 or value.shape != dst.shape:
            raise ValueError("probability_store requires a matching 2D FP32 tile")
        options = semantic.builder.options
        if is_enabled(options, "fpsan"):
            if mask is not None:
                value = expect_zero(value, mask, _semantic=semantic)
            ty = value.type.with_element_ty(dst.dtype)
            predicate = pred.handle if pred is not None else None
            handle = semantic.builder.create_fpsan_store_probability(ty.to_ir(semantic.builder), dst.handle,
                                                                     value.handle, predicate)
            return tensor(handle, ty)

        rounded = semantic.cast(value, dst.dtype)
        if options.debug and mask is not None:
            expect_zero(semantic.cast(rounded, float16), mask, _semantic=semantic)
        if pred is None:
            dst.store(rounded, _semantic=semantic)
        else:
            dst.store(rounded, pred=pred, _semantic=semantic)
        return rounded


@builtin
def probability_store(value, expect_zero=None, _semantic=None):
    """Prepare a 2D FP32 probability tile for an FP8 SMEM or TMEM descriptor store.

    Pass the result to `dst.store(...)`, which rounds to the destination format
    (E4M3 or E5M2), writes the tile and returns the rounded tensor. The optional
    mask has the semantics of `gl.expect_zero` applied to the rounded tensor.
    In FPSan, an unscaled FP32-accumulating tcgen05_mma reads the tagged
    full-width payload from the stored bytes and their residuals.
    """
    return _ProbabilityStore(value, expect_zero)
