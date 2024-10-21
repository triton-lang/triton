import triton.language as tl
from triton.language import core
from triton.language.core import builtin
from triton import jit


@core.extern
def acos(arg0, _builder=None):
    return core.tensor(_builder.create_acos(arg0.handle), arg0.type)


@core.extern
def acosh(arg0, _builder=None):
    return core.tensor(_builder.create_acosh(arg0.handle), arg0.type)


@core.extern
def asin(arg0, _builder=None):
    return core.tensor(_builder.create_asin(arg0.handle), arg0.type)


@core.extern
def asinh(arg0, _builder=None):
    return core.tensor(_builder.create_asinh(arg0.handle), arg0.type)


@core.extern
def atan(arg0, _builder=None):
    return core.tensor(_builder.create_atan(arg0.handle), arg0.type)


@core.extern
def atanh(arg0, _builder=None):
    return core.tensor(_builder.create_atanh(arg0.handle), arg0.type)


@core.extern
def cbrt(arg0, _builder=None):
    return core.tensor(_builder.create_cbrt(arg0.handle), arg0.type)


@core.extern
def cos(arg0, _builder=None):
    return core.tensor(_builder.create_cos(arg0.handle), arg0.type)


@core.extern
def cosh(arg0, _builder=None):
    return core.tensor(_builder.create_cosh(arg0.handle), arg0.type)


@core.extern
def erf(arg0, _builder=None):
    return core.tensor(_builder.create_erf(arg0.handle), arg0.type)


@core.extern
def exp(arg0, _builder=None):
    return core.tensor(_builder.create_exp(arg0.handle), arg0.type)


@core.extern
def exp2(arg0, _builder=None):
    return core.tensor(_builder.create_exp2(arg0.handle), arg0.type)


@core.extern
def expm1(arg0, _builder=None):
    return core.tensor(_builder.create_expm1(arg0.handle), arg0.type)


@core.extern
def floor(arg0, _builder=None):
    return core.tensor(_builder.create_floor(arg0.handle), arg0.type)


@core.extern
def log(arg0, _builder=None):
    return core.tensor(_builder.create_log(arg0.handle), arg0.type)


@core.extern
def log2(arg0, _builder=None):
    return core.tensor(_builder.create_log2(arg0.handle), arg0.type)


@core.extern
def log10(arg0, _builder=None):
    return core.tensor(_builder.create_log10(arg0.handle), arg0.type)


@core.extern
def log1p(arg0, _builder=None):
    return core.tensor(_builder.create_log1p(arg0.handle), arg0.type)


@core.extern
def sin(arg0, _builder=None):
    return core.tensor(_builder.create_sin(arg0.handle), arg0.type)


@core.extern
def rsqrt(arg0, _builder=None):
    return core.tensor(_builder.create_rsqrt(arg0.handle), arg0.type)


@core.extern
def sqrt(arg0, _builder=None):
    return core.tensor(_builder.create_sqrt(arg0.handle), arg0.type)


@core.extern
def sinh(arg0, _builder=None):
    return core.tensor(_builder.create_sinh(arg0.handle), arg0.type)


@core.extern
def tan(arg0, _builder=None):
    return core.tensor(_builder.create_tan(arg0.handle), arg0.type)


@core.extern
def tanh(arg0, _builder=None):
    return core.tensor(_builder.create_tanh(arg0.handle), arg0.type)


@core.extern
def trunc(arg0, _builder=None):
    return core.tensor(_builder.create_trunc(arg0.handle), arg0.type)


@core.extern
def ceil(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Sleef_ceilf%(numel)", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Sleef_ceild%(numel)", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def pow(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Sleef_powf%(numel)_u10", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Sleef_powd%(numel)_u10", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fmod(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Sleef_fmodf%(numel)", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Sleef_fmodd%(numel)", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@jit
def _const(v, dtype):
    """
    Create a tensor with a single value of type :dtype.
    """
    return tl.full((1, ), v, dtype)


@jit
def _is_special_float(arg0, uint_dtype, kind: tl.constexpr):
    # By default, Triton assumes constexprs are int32. Thus, when we do operations with constants,
    # we end up auto-promoting smaller integer types to int32, which is undesirable. Thus we
    # explicitly cast them to our desired type here.
    one = _const(1, uint_dtype)
    zero = _const(0, uint_dtype)

    bitwidth: tl.constexpr = arg0.dtype.primitive_bitwidth
    exponent_width: tl.constexpr = bitwidth - 1 - arg0.dtype.fp_mantissa_width
    mantissa_width: tl.constexpr = arg0.dtype.fp_mantissa_width

    uintval = arg0.to(uint_dtype, bitcast=True)
    exponent = uintval << one >> _const(mantissa_width, uint_dtype) + one
    exp_is_all_ones = exponent == (one << _const(exponent_width, uint_dtype)) - one
    shifted_mantissa = uintval << _const(exponent_width, uint_dtype) + one

    if kind == "nan":
        return exp_is_all_ones & (shifted_mantissa != zero)
    elif kind == "inf":
        return exp_is_all_ones & (shifted_mantissa == zero)
    else:
        raise ValueError(f"Unexpected kind {kind}")


@builtin
def isnan(arg0, _builder=None, _generator=None):
    if not arg0.dtype.is_floating():
        raise ValueError("isnan expects a floating point type")
    bitwidth = arg0.dtype.primitive_bitwidth
    uint_dtype = tl.core.get_int_dtype(bitwidth, signed=False)
    return _generator.call_JitFunction(_is_special_float, (arg0, uint_dtype, "nan"), kwargs={})


@builtin
def isinf(arg0, _builder=None, _generator=None):
    if not arg0.dtype.is_floating():
        raise ValueError("isinf expects a floating point type")
    bitwidth = arg0.dtype.primitive_bitwidth
    uint_dtype = tl.core.get_int_dtype(bitwidth, signed=False)
    return _generator.call_JitFunction(_is_special_float, (arg0, uint_dtype, "inf"), kwargs={})


@jit
def _signbit(arg0, uint_dtype: tl.constexpr):
    bitwidth: tl.constexpr = arg0.dtype.primitive_bitwidth
    return arg0.to(uint_dtype, bitcast=True) >> (bitwidth - 1)


@builtin
def signbit(arg0, _builder=None, _generator=None):
    if not arg0.dtype.is_floating():
        raise ValueError("signbit expects a floating point type")
    bitwidth = arg0.dtype.primitive_bitwidth
    uint_dtype = tl.core.get_int_dtype(bitwidth, signed=False)
    return _generator.call_JitFunction(_signbit, (arg0, uint_dtype), kwargs={})
