from dataclasses import dataclass

from triton._C.libtriton import ir
from triton._instrumentation import is_enabled
from triton.language import fpsan as tl_fpsan

from ._core import (_unwrap_if_constexpr, base_type, base_value, builtin, distributed_type, dtype, float8e4nv, float8e5,
                    float16, float32, int32, tensor)
from ._core import expect_zero as _expect_zero

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


@dataclass(frozen=True)
class _ProbabilityType(base_type):
    rounded_type: distributed_type
    _has_residual: bool

    def mangle(self):
        return f"Probability{int(self._has_residual)}_{self.rounded_type.mangle()}"

    def _flatten_ir_types(self, builder, out):
        out.append(self.rounded_type.to_ir(builder))
        if self._has_residual:
            out.append(self.rounded_type.with_element_ty(int32).to_ir(builder))

    def _unflatten_ir(self, handles, cursor):
        end = cursor + (2 if self._has_residual else 1)
        return _Probability(self, *handles[cursor:end]), end


@dataclass(frozen=True, eq=False)
class _Probability(base_value):
    # Expose the cast as one Gluon value across calls and control flow.
    # Only descriptor stores unpack its IR handles.
    type: _ProbabilityType
    _rounded: ir.value
    _residual: ir.value | None = None

    def _flatten_ir(self, handles):
        handles.append(self._rounded)
        if self._residual is not None:
            handles.append(self._residual)

    def _set_name(self, builder, name):
        self._rounded.set_loc(builder.create_name_loc(name, self._rounded.get_loc()))
        if self._residual is not None:
            self._residual.set_loc(builder.create_name_loc(name + ".residual", self._residual.get_loc()))

    def _store(self, dst, semantic, pred=None):
        rounded = tensor(self._rounded, self.type.rounded_type)
        if rounded.dtype != dst.dtype or rounded.shape != dst.shape:
            raise ValueError("probability_cast shape and dtype must match the destination")
        if self._residual is not None:
            predicate = pred.handle if pred is not None else None
            semantic.builder.create_fpsan_store_probability(dst.handle, self._rounded, self._residual, predicate)
        elif pred is None:
            dst.store(rounded, _semantic=semantic)
        else:
            dst.store(rounded, pred=pred, _semantic=semantic)
        return rounded


@builtin
def probability_cast(value, dtype, expect_zero=None, _semantic=None):
    """Cast a 2D FP32 probability tile to E4M3 or E5M2 for a subsequent descriptor store.

    Pass the opaque result to a matching SMEM or TMEM descriptor's `store`
    method, which writes the tile and returns the rounded FP8 tensor. The
    optional mask has the semantics of `gl.expect_zero` on the rounded value.
    In FPSan, the cast also produces a residual that travels with the FP8 tile.
    An unscaled FP32-accumulating tcgen05_mma reconstructs the tagged full-width
    payload from the stored bytes and residuals.
    """
    dtype = _unwrap_if_constexpr(dtype)
    if dtype not in (float8e4nv, float8e5):
        raise ValueError("probability_cast requires an E4M3 or E5M2 dtype")
    if not isinstance(value, tensor) or value.dtype != float32 or len(value.shape) != 2:
        raise ValueError("probability_cast requires a 2D FP32 tile")
    mask = _unwrap_if_constexpr(expect_zero)
    builder = _semantic.builder
    if is_enabled(builder.options, "fpsan"):
        if mask is not None:
            value = _expect_zero(value, mask, _semantic=_semantic)
        ty = value.type.with_element_ty(dtype)
        residual_ty = value.type.with_element_ty(int32)
        rounded, residual = builder.create_fpsan_probability_cast(ty.to_ir(builder), residual_ty.to_ir(builder),
                                                                  value.handle)
        return _Probability(_ProbabilityType(ty, True), rounded, residual)

    rounded = _semantic.cast(value, dtype)
    if builder.options.debug and mask is not None:
        _expect_zero(_semantic.cast(rounded, float16), mask, _semantic=_semantic)
    return _Probability(_ProbabilityType(rounded.type, False), rounded.handle)
