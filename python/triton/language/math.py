from . import core
from functools import wraps
from typing import List

T = core.TypeVar('T')


def _check_dtype(dtypes: List[str]) -> T:
    """
    We're following libdevice's convention to check accepted data types for math functions.
    It is not a good practice to support all data types as accelerators/GPUs don't support
    many float16 and bfloat16 math operations.
    We should let the users know that they are using and invoke explicit cast to convert
    the data type to the supported one.
    """

    def wrapper(fn):

        @wraps(fn)
        def check(*args, **kwargs):
            # concatenate args and kwargs
            all_args = list(args) + list(kwargs.values())
            for arg in [a for a in all_args if isinstance(a, core.tensor)]:
                if arg.type.scalar.name not in dtypes:
                    raise ValueError(f"Expected dtype {dtypes} but got {arg.type.scalar.name}")
            return fn(*args, **kwargs)

        return check

    return wrapper


def _add_math_1arg_docstr(name: str) -> core.Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Computes the element-wise {name} of :code:`x`.

    :param x: the input values
    :type x: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


def _add_math_2arg_docstr(name: str) -> core.Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Computes the element-wise {name} of :code:`x` and :code:`y`.

    :param x: the input values
    :type x: Block
    :param y: the input values
    :type y: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


def _add_math_3arg_docstr(name: str) -> core.Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Computes the element-wise {name} of :code:`x`, :code:`y`, and :code:`z`.

    :param x: the input values
    :type x: Block
    :param y: the input values
    :type y: Block
    :param z: the input values
    :type z: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@core.builtin
@_check_dtype(dtypes=["int32", "int64", "uint32", "uint64"])
@_add_math_2arg_docstr("most significant N bits of the 2N-bit product")
def umulhi(x, y, _semantic=None):
    x = core._unwrap_if_constexpr(x)
    y = core._unwrap_if_constexpr(y)
    x, y = core.binary_op_type_legalization(x, y, _semantic)
    return core.tensor(_semantic.builder.create_umulhi(x.handle, y.handle), x.type)


@core._tensor_member_fn
@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("exponential")
def exp(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_exp(x.handle), x.type)


@core._tensor_member_fn
@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("exponential (base 2)")
def exp2(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_exp2(x.handle), x.type)


@core._tensor_member_fn
@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("natural logarithm")
def log(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_log(x.handle), x.type)


@core._tensor_member_fn
@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("logarithm (base 2)")
def log2(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_log2(x.handle), x.type)


@core._tensor_member_fn
@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("cosine")
def cos(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_cos(x.handle), x.type)


@core._tensor_member_fn
@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("sine")
def sin(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_sin(x.handle), x.type)


@core._tensor_member_fn
@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("fast square root")
def sqrt(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_sqrt(x.handle), x.type)


@core._tensor_member_fn
@core.builtin
@_check_dtype(dtypes=["fp32"])
@_add_math_1arg_docstr("precise square root (rounding to nearest wrt the IEEE standard)")
def sqrt_rn(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_precise_sqrt(x.handle), x.type)


@core._tensor_member_fn
@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("inverse square root")
def rsqrt(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_rsqrt(x.handle), x.type)


@core._tensor_member_fn
@core.builtin
@_add_math_1arg_docstr("absolute value")
def abs(x, _semantic=None):
    x = _semantic.to_tensor(x)
    dtype = x.dtype
    if dtype.is_fp8e4b15():
        mask = core.full(x.shape, 0x7F, core.int8, _semantic=_semantic)
        return core.tensor(_semantic.builder.create_and(x.handle, mask.handle), x.type)
    elif dtype.is_floating():
        return core.tensor(_semantic.builder.create_fabs(x.handle), x.type)
    elif dtype.is_int_signed():
        return core.tensor(_semantic.builder.create_iabs(x.handle), x.type)
    elif dtype.is_int_unsigned():
        return x  # no-op
    else:
        assert False, f"Unexpected dtype {dtype}"


@core.builtin
def fdiv(x, y, ieee_rounding=False, approx=False, _semantic=None):
    """
    Computes the element-wise division of :code:`x` by :code:`y`.

    :param x: the numerator
    :param y: the denominator
    :param ieee_rounding: if True, use IEEE round-to-nearest division.
    :param approx: if True, allow approximate division for float32 operands.
        On NVIDIA, this uses ``div.approx.f32``, with a maximum error of 2 ULP
        for denominator magnitudes in [2**-126, 2**126]. Results outside this
        range may differ from full-range division. On AMD, this uses
        ``llvm.amdgcn.fdiv.fast``, with a maximum error of 2.5 ULP and no support
        for denormal inputs or results. Cannot be combined with
        ``ieee_rounding=True``.
    """
    ieee_rounding = core._unwrap_if_constexpr(ieee_rounding)
    approx = core._unwrap_if_constexpr(approx)
    x = core._unwrap_if_constexpr(x)
    y = core._unwrap_if_constexpr(y)
    return _semantic.fdiv(x, y, ieee_rounding, approx)


@core.builtin
@_check_dtype(dtypes=["fp32"])
@_add_math_2arg_docstr("precise division (rounding to nearest wrt the IEEE standard)")
def div_rn(x, y, _semantic=None):
    x = core._unwrap_if_constexpr(x)
    y = core._unwrap_if_constexpr(y)
    x, y = core.binary_op_type_legalization(x, y, _semantic)
    return core.tensor(_semantic.builder.create_precise_divf(x.handle, y.handle), x.type)


@core._tensor_member_fn
@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("error function")
def erf(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_erf(x.handle), x.type)


@core._tensor_member_fn
@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("floor")
def floor(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_floor(x.handle), x.type)


@core._tensor_member_fn
@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("ceil")
def ceil(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_ceil(x.handle), x.type)


@core.builtin
def fma(x, y, z, rounding_mode="rn", _semantic=None):
    """
    Computes the element-wise fused multiply-add of :code:`x`, :code:`y`, and
    :code:`z`: ``x * y + z`` with a single rounding.

    :param x: the first multiplicand
    :param y: the second multiplicand
    :param z: the addend
    :param rounding_mode: the rounding mode for the final result. Must be a
        compile-time constant: ``"rn"`` (round to nearest, ties to even; the
        default), ``"rz"`` (round toward zero), ``"rd"`` (round
        toward negative infinity), or ``"ru"`` (round toward positive
        infinity). The directed modes (``"rz"``, ``"rd"``, ``"ru"``) are only
        supported for ``fp32`` and ``fp64`` operands: NVIDIA PTX provides
        ``fma.rn``/``fma.rz``/``fma.rm``/``fma.rp`` for ``f32``/``f64`` on all
        targets since ``sm_20``/``sm_13`` (including Hopper ``sm_90``,
        Blackwell ``sm_100`` and later), while the half-precision
        ``fma.{f16,bf16}`` forms are ``.rn``-only; AMD CDNA3 (``gfx942``),
        CDNA4 (``gfx950``) and CDNA5 (``gfx1250``) expose all four modes for
        single and double precision through the ``MODE.FP_ROUND`` fields
        (0=nearest-even, 1=+inf, 2=-inf, 3=toward-zero), lowered here via the
        LLVM constrained-FMA intrinsic, as in the OCML
        ``__ocml_fma_rt*`` device functions.
    """
    rounding_mode = core._unwrap_if_constexpr(rounding_mode)
    if rounding_mode not in ("rn", "rz", "rd", "ru"):
        raise ValueError(f"Invalid rounding mode: {rounding_mode}. "
                         "Supported rounding modes are 'rn', 'rz', 'rd' and 'ru'.")
    x = core._unwrap_if_constexpr(x)
    y = core._unwrap_if_constexpr(y)
    z = core._unwrap_if_constexpr(z)
    x, y, z = _semantic.ternary_op_type_checking_impl(x, y, z)
    if rounding_mode != "rn" and x.type.scalar.name not in ("fp32", "fp64"):
        raise ValueError(f"rounding_mode='{rounding_mode}' is only supported for fp32 and fp64 operands, "
                         f"but got {x.type.scalar.name}. Half-precision FMA is round-to-nearest-even only "
                         "on NVIDIA and has no directed libdevice equivalent to preserve.")
    return core.tensor(_semantic.builder.create_fma(x.handle, y.handle, z.handle, rounding_mode), x.type)
