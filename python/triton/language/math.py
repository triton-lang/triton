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


@core.builtin
@_check_dtype(dtypes=["int32", "int64", "uint32", "uint64"])
@_add_math_2arg_docstr("most significant N bits of the 2N-bit product")
def umulhi(x, y, _builder=None):
    x = core._to_tensor(x, _builder)
    y = core._to_tensor(y, _builder)
    x, y = core.binary_op_type_legalization(x, y, _builder)
    return core.tensor(_builder.create_umulhi(x.handle, y.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("exponential")
def exp(x, _builder=None):
    x = core._to_tensor(x, _builder)
    return core.tensor(_builder.create_exp(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("exponential (base 2)")
def exp2(x, _builder=None):
    x = core._to_tensor(x, _builder)
    return core.tensor(_builder.create_exp2(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("natural logarithm")
def log(x, _builder=None):
    x = core._to_tensor(x, _builder)
    return core.tensor(_builder.create_log(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("logarithm (base 2)")
def log2(x, _builder=None):
    x = core._to_tensor(x, _builder)
    return core.tensor(_builder.create_log2(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("cosine")
def cos(x, _builder=None):
    x = core._to_tensor(x, _builder)
    return core.tensor(_builder.create_cos(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("sine")
def sin(x, _builder=None):
    x = core._to_tensor(x, _builder)
    return core.tensor(_builder.create_sin(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("square root")
def sqrt(x, _builder=None):
    x = core._to_tensor(x, _builder)
    return core.tensor(_builder.create_sqrt(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32"])
@_add_math_1arg_docstr("square root (rounding to nearest)")
def sqrt_rn(x, _builder=None):
    x = core._to_tensor(x, _builder)
    return core.tensor(_builder.create_precise_sqrt(x.handle), x.type)


@core.builtin
@_add_math_1arg_docstr("absolute value")
def abs(x, _builder=None):
    x = core._to_tensor(x, _builder)
    dtype = x.dtype
    if dtype.is_floating():
        return core.tensor(_builder.create_fabs(x.handle), x.type)
    elif dtype.is_int_signed():
        return core.tensor(_builder.create_iabs(x.handle), x.type)
    elif dtype.is_int_unsigned():
        return x  # no-op
    else:
        assert False, f"Unexpected dtype {dtype}"


@core.builtin
@_check_dtype(dtypes=["fp32"])
@_add_math_1arg_docstr("division (rounding to nearest)")
def div_rn(x, y, _builder=None):
    x = core._to_tensor(x, _builder)
    y = core._to_tensor(y, _builder)
    core.binary_op_type_legalization(x, y, _builder)
    return core.tensor(_builder.create_precise_divf(x.handle, y.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("error function")
def erf(x, _builder=None):
    x = core._to_tensor(x, _builder)
    return core.tensor(_builder.create_erf(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("floor")
def floor(x, _builder=None):
    x = core._to_tensor(x, _builder)
    return core.tensor(_builder.create_floor(x.handle), x.type)
