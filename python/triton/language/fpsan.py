"""Explicit controls for floating-point sanitization."""

from typing import Optional

from . import core


@core.builtin
def barrier(input, _semantic=None):
    """
    Mark an intermediate expression as significant to floating-point sanitization.

    With FPSan enabled, apply a deterministic, non-homomorphic transformation to
    the input's payload. For example, ``barrier(a * b) * c`` can differ
    from ``a * barrier(b * c)``, and ``barrier(a * b) + c`` can
    differ from ``fma(a, b, c)``. Without FPSan, return the input unchanged.

    High payload bits are mixed into low bits, so moving an
    :func:`homomorphic_cast` across a barrier can change the signature.

    This is not a synchronization or optimization barrier and does not enforce
    floating-point rounding or prevent FMA contraction in normal execution.

    :param input: A floating-point scalar or tensor.
    :return: A value with the same type, shape, and layout as ``input``.
    """
    input = _semantic.to_tensor(input)
    return _semantic.fpsan_barrier(input)


@core.builtin
def compact_exp2(input, _semantic=None):
    """
    Compute exp2 on an FP32 value with a compact FPSan signature.

    Under FPSan the result has payload ``(1 + 2**25)**x mod 2**32`` for
    input payload ``x``. Its 128 powers fit in an FP8 probability code.
    Use this operation for both the centered weights and their rescaling
    factors to preserve ``E(x + y) = E(x) * E(y)`` across an FP8 round trip.
    Native execution computes the ordinary base-2 exponential.
    """
    return _semantic.fpsan_compact_exp2(_semantic.to_tensor(input))


@core.builtin
def homomorphic_cast(input, dtype: core.dtype, fp_downcast_rounding: Optional[str] = None, _semantic=None):
    """
    Cast to a same-width or narrower floating-point type, preserving FPSan algebra.

    With FPSan enabled, retain the low bits of the input's payload. This preserves
    addition and multiplication, allowing a downcast to move through arithmetic
    without changing the signature. Use :func:`barrier` to mark a boundary
    across which moving the cast should be observable.

    Without FPSan, perform an ordinary floating-point cast. Widening and integer
    conversions are not supported. The input's shape and layout are preserved.

    :param input: A floating-point scalar or tensor.
    :param dtype: A floating-point destination type no wider than the input.
    :param fp_downcast_rounding: Rounding mode for normal execution: ``"rtne"``
        (default) or ``"rtz"``. Only valid for a narrower destination type;
        FPSan payload truncation is independent of this choice.
    """
    return _semantic.fpsan_homomorphic_cast(_semantic.to_tensor(input), core._unwrap_if_constexpr(dtype),
                                            core._unwrap_if_constexpr(fp_downcast_rounding))
