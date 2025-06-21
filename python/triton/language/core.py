from __future__ import annotations

from warnings import warn
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
import typing
from typing import Union, Callable, List, Sequence, TypeVar, Optional, Tuple
from dataclasses import dataclass
import builtins
from .. import knobs
from ..runtime.jit import jit, JITFunction
import inspect

from .._C.libtriton import ir
from .._utils import TRITON_MAX_TENSOR_NUMEL, validate_block_shape, get_primitive_bitwidth

T = TypeVar('T')

TRITON_BUILTIN = "__triton_builtin__"

PropagateNan = ir.PROPAGATE_NAN


def must_use_result(x, s=True):
    """If the result of this function is unused, throw an error."""
    if isinstance(x, str):
        return (lambda fn: must_use_result(fn, x))
    x._must_use_result = s
    return x


def builtin(fn: T) -> T:
    """Mark a function as a builtin."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "_semantic" not in kwargs or kwargs["_semantic"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_semantic` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    setattr(wrapper, TRITON_BUILTIN, True)

    return wrapper


def _tensor_member_fn(fn: T) -> T:
    """Decorator that adds this free function as a member fn on class tensor.

    When called as a member function on class tensor, the first argument to `fn`
    is `self`, i.e. the tensor object.

    If there are multiple decorators on a function, you probably want this one
    to be the highest one (i.e. furthest from the function's `def`), so it's
    applied last.

    Unfortunately you still need to add a type stub to the body of class tensor
    in order for pytype to know about it.
    """
    assert callable(fn)
    orig_sig = inspect.signature(fn)
    # Does fn take args other than _semantic, _generator, and the tensor itself?
    has_args = len(orig_sig.parameters.keys() - {"_semantic", "_generator"}) > 1

    if not fn.__doc__:
        fn.__doc__ = ""
    fn.__doc__ += f"""
    This function can also be called as a member function on :py:class:`tensor`,
    as :code:`x.{fn.__name__}({"..." if has_args else ""})` instead of
    :code:`{fn.__name__}(x{", ..." if has_args else ""})`.
    """

    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    # Match the signature of `fn`, but change the first arg to `self` so the
    # docs are a little less weird.
    new_params = list(orig_sig.parameters.values())
    new_params[0] = new_params[0].replace(name='self')
    new_sig = orig_sig.replace(parameters=new_params)
    wrapper.__signature__ = new_sig
    wrapper.__doc__ = f"Forwards to :py:func:`{fn.__name__}` free function"
    # If fn is a builtin, mark the wrapper as a builtin too.
    if is_builtin(fn):
        setattr(wrapper, TRITON_BUILTIN, True)

    setattr(tensor, fn.__name__, fn if isinstance(fn, JITFunction) else wrapper)
    return fn


def _unwrap_iterable(x):
    """Returns x[0] if x has one element and x[0] is iterable."""
    if len(x) == 1:
        # Determine whether x[0] is iterable.
        #
        # You might want to use collections.abc.Iterable instead of this
        # try/except block.  Unfortunately, this doesn't work with constexpr.
        #
        # The problem is that abc.Iterable checks for __iter__ on the *class*.
        # But we want constexpr to expose an __iter__ method if and only if the
        # wrapped *object* (i.e. self.value) is iterable.  Therefore there's no
        # right answer for whether the class constexpr defines __iter__, and
        # abc.Iterable doesn't work (at least not without some metaclass magic).
        try:
            iter(x[0])
            return x[0]
        except TypeError:
            pass

    return x


def is_builtin(fn) -> bool:
    """Is this a registered triton builtin function?"""
    return getattr(fn, TRITON_BUILTIN, False)


@builtin
def to_tensor(x, _semantic=None):
    return _semantic.to_tensor(x)


# -----------------------
# constexpr
# -----------------------


class const:
    """
    This class is used as a type annotation to mark pointers to constant data.
    The `store` function cannot be called with a pointer to const. Constness
    is part of the pointer type and the usual Triton type consistency rules
    apply. For example you cannot have a function that returns constant pointer
    in one return statement and non-constant pointer in another.
    """
    pass


class base_value:
    """Base class of values that exist in the triton IR (i.e. not constexprs).
    """
    type: base_type

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        """Flatten frontend value into a sequence of mlir handles, which are appended
        to the output list
        """
        raise NotImplementedError


class base_type:

    def __eq__(self, other):
        raise NotImplementedError("Types must implement __eq__")

    def __ne__(self, other):
        return not (self == other)

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        """Build a frontend value with the current dtype, wrapping a list of existing handles.
        cursor is the index of the first handle relevant to this value, and the function
        should return the updated cursor position after any handles consumed by the created value.
        """
        raise NotImplementedError

    def mangle(self) -> str:
        raise NotImplementedError(f"NYI: Type mangling for type {self.__class__}")

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        raise NotImplementedError


class constexpr_type(base_type):

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self) -> str:
        return f"constexpr[{self.value}]"

    def mangle(self) -> str:
        return repr(self)

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        return

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        return constexpr(self.value), cursor


class constexpr(base_value):
    """
    This class is used to store a value that is known at compile-time.
    """

    def __init__(self, value):
        if isinstance(value, constexpr):
            self.value = value.value
        else:
            self.value = value
        self.type = constexpr_type(value)

    def __repr__(self) -> str:
        return f"constexpr[{self.value}]"

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        return

    def __index__(self):
        return self.value

    # In interpreter mode, constant values are not wrapped in constexpr,
    # and therefore do not have a .value attribute.
    # As a result, from here and below, we need to call the _unwrap_if_constexpr
    # function to obtain either constexpr.value or the value itself.
    def __add__(self, other):
        return constexpr(self.value + _unwrap_if_constexpr(other))

    def __radd__(self, other):
        return constexpr(_unwrap_if_constexpr(other) + self.value)

    def __sub__(self, other):
        return constexpr(self.value - _unwrap_if_constexpr(other))

    def __rsub__(self, other):
        return constexpr(_unwrap_if_constexpr(other) - self.value)

    def __mul__(self, other):
        return constexpr(self.value * _unwrap_if_constexpr(other))

    def __mod__(self, other):
        return constexpr(self.value % _unwrap_if_constexpr(other))

    def __rmul__(self, other):
        return constexpr(_unwrap_if_constexpr(other) * self.value)

    def __truediv__(self, other):
        return constexpr(self.value / _unwrap_if_constexpr(other))

    def __rtruediv__(self, other):
        return constexpr(_unwrap_if_constexpr(other) / self.value)

    def __floordiv__(self, other):
        return constexpr(self.value // _unwrap_if_constexpr(other))

    def __rfloordiv__(self, other):
        return constexpr(_unwrap_if_constexpr(other) // self.value)

    def __gt__(self, other):
        return constexpr(self.value > _unwrap_if_constexpr(other))

    def __rgt__(self, other):
        return constexpr(_unwrap_if_constexpr(other) > self.value)

    def __ge__(self, other):
        return constexpr(self.value >= _unwrap_if_constexpr(other))

    def __rge__(self, other):
        return constexpr(_unwrap_if_constexpr(other) >= self.value)

    def __lt__(self, other):
        return constexpr(self.value < _unwrap_if_constexpr(other))

    def __rlt__(self, other):
        return constexpr(_unwrap_if_constexpr(other) < self.value)

    def __le__(self, other):
        return constexpr(self.value <= _unwrap_if_constexpr(other))

    def __rle__(self, other):
        return constexpr(_unwrap_if_constexpr(other) <= self.value)

    def __eq__(self, other):
        return constexpr(self.value == _unwrap_if_constexpr(other))

    def __ne__(self, other):
        return constexpr(self.value != _unwrap_if_constexpr(other))

    def __bool__(self):
        return bool(self.value)

    def __neg__(self):
        return constexpr(-self.value)

    def __and__(self, other):
        return constexpr(self.value & _unwrap_if_constexpr(other))

    def logical_and(self, other):
        return constexpr(self.value and _unwrap_if_constexpr(other))

    def __or__(self, other):
        return constexpr(self.value | _unwrap_if_constexpr(other))

    def __xor__(self, other):
        return constexpr(self.value ^ _unwrap_if_constexpr(other))

    def logical_or(self, other):
        return constexpr(self.value or _unwrap_if_constexpr(other))

    def __pos__(self):
        return constexpr(+self.value)

    def __invert__(self):
        return constexpr(~self.value)

    def __pow__(self, other):
        return constexpr(self.value**_unwrap_if_constexpr(other))

    def __rpow__(self, other):
        return constexpr(_unwrap_if_constexpr(other)**self.value)

    def __rshift__(self, other):
        return constexpr(self.value >> _unwrap_if_constexpr(other))

    def __lshift__(self, other):
        return constexpr(self.value << _unwrap_if_constexpr(other))

    def __not__(self):
        return constexpr(not self.value)

    def __iter__(self):
        return iter(self.value)

    def __call__(self, *args, **kwds):
        return self.value(*args, **kwds)

    def __getitem__(self, *args):
        args = (_unwrap_if_constexpr(x) for x in _normalize_tuple(args))
        return self.value.__getitem__(*args)


def constexpr_function(f):
    """
    Wraps an arbitrary Python function so that it can be called at
    compile-time on constexpr arguments in a Triton function and
    returns a constexpr result.
    """

    @wraps(f)
    def wrapper(*args, _semantic=None, **kwargs):
        # de-constexpr arguments and discard the _semantic keyword argument:
        args = [_unwrap_if_constexpr(x) for x in args]
        kwargs = {k: _unwrap_if_constexpr(v) for (k, v) in kwargs.items()}

        # call the raw Python function f:
        res = f(*args, **kwargs)

        # convert result back to a Triton constexpr:
        return constexpr(res)

    # disguise the function as a Triton builtin to avoid raising an error
    # that we're calling a non-JIT function from within a Triton kernel:
    wrapper.__triton_builtin__ = True
    wrapper.__module__ = constexpr_function.__module__
    return wrapper


CONSTEXPR_0 = constexpr(0)


def _unwrap_if_constexpr(o):
    if isinstance(o, list):
        return [_unwrap_if_constexpr(x) for x in o]
    if isinstance(o, builtins.tuple):
        return builtins.tuple(_unwrap_if_constexpr(x) for x in o)
    if isinstance(o, tuple):
        return tuple(_unwrap_if_constexpr(x) for x in o)
    return o.value if isinstance(o, constexpr) else o


def _normalize_tuple(t):
    normalized_tuple = _unwrap_if_constexpr(t)
    if isinstance(normalized_tuple, (list, builtins.tuple)):
        normalized_tuple = tuple(normalized_tuple)
    return normalized_tuple


def check_bit_width(value, shift_value):
    if isinstance(value, tensor) and isinstance(shift_value, constexpr):
        bitwidth = value.type.scalar.primitive_bitwidth
        if shift_value.value >= bitwidth:
            warn(
                f"Value {shift_value.value} exceeds the maximum bitwidth ({bitwidth}) for type '{value.dtype}'. This may result in undefined behavior."
            )


# -----------------------
# dtype
# -----------------------


class dtype(base_type):
    SINT_TYPES = ['int8', 'int16', 'int32', 'int64']
    UINT_TYPES = ['int1', 'uint8', 'uint16', 'uint32', 'uint64']
    FP_TYPES = ['fp8e4b15', 'fp8e4nv', 'fp8e4b8', 'fp8e5', 'fp8e5b16', 'fp16', 'bf16', 'fp32', 'fp64']
    STANDARD_FP_TYPES = ['fp16', 'bf16', 'fp32', 'fp64']
    OTHER_TYPES = ['void']

    class SIGNEDNESS(Enum):
        SIGNED = 0
        UNSIGNED = 1

    class KIND(Enum):
        BOOLEAN = 0
        INTEGRAL = 1
        FLOATING = 2

    def __init__(self, name):
        name = _unwrap_if_constexpr(name)
        self.name = name
        assert name in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES, name
        self.primitive_bitwidth = get_primitive_bitwidth(name)
        self.itemsize = self.primitive_bitwidth // 8
        if name in dtype.SINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.SIGNED
            self.int_bitwidth = self.primitive_bitwidth
        elif name in dtype.UINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.UNSIGNED
            self.int_bitwidth = self.primitive_bitwidth
        elif name in dtype.FP_TYPES:
            if name == 'fp8e4b15':
                self.fp_mantissa_width = 3
                self.exponent_bias = 15
            elif name == 'fp8e4nv':
                self.fp_mantissa_width = 3
                self.exponent_bias = 7
            elif name == 'fp8e4b8':
                self.fp_mantissa_width = 3
                self.exponent_bias = 8
            elif name == 'fp8e5':
                self.fp_mantissa_width = 2
                self.exponent_bias = 15
            elif name == 'fp8e5b16':
                self.fp_mantissa_width = 2
                self.exponent_bias = 16
            elif name == 'fp16':
                self.fp_mantissa_width = 10
                self.exponent_bias = 15
            elif name == 'bf16':
                self.fp_mantissa_width = 7
                self.exponent_bias = 127
            elif name == 'fp32':
                self.fp_mantissa_width = 23
                self.exponent_bias = 127
            elif name == 'fp64':
                self.fp_mantissa_width = 52
                self.exponent_bias = 1023
            else:
                raise RuntimeError(f'Unsupported floating-point type {name}')

    def is_fp8(self):
        return 'fp8' in self.name

    def is_fp8e4nv(self):
        return self.name == 'fp8e4nv'

    def is_fp8e4b8(self):
        return self.name == 'fp8e4b8'

    def is_fp8e4b15(self):
        return self.name == 'fp8e4b15'

    def is_fp8e5(self):
        return self.name == 'fp8e5'

    def is_fp8e5b16(self):
        return self.name == 'fp8e5b16'

    def is_fp16(self):
        return self.name == 'fp16'

    def is_bf16(self):
        return self.name == 'bf16'

    def is_fp32(self):
        return self.name == 'fp32'

    def is_fp64(self):
        return self.name == 'fp64'

    def is_int1(self):
        return self.name == 'int1'

    def is_int8(self):
        return self.name == 'int8'

    def is_int16(self):
        return self.name == 'int16'

    def is_int32(self):
        return self.name == 'int32'

    def is_int64(self):
        return self.name == 'int64'

    def is_uint8(self):
        return self.name == 'uint8'

    def is_uint16(self):
        return self.name == 'uint16'

    def is_uint32(self):
        return self.name == 'uint32'

    def is_uint64(self):
        return self.name == 'uint64'

    def is_floating(self):
        return self.name in dtype.FP_TYPES

    def is_standard_floating(self):
        return self.name in dtype.STANDARD_FP_TYPES

    def is_int_signed(self):
        return self.name in dtype.SINT_TYPES

    def is_int_unsigned(self):
        return self.name in dtype.UINT_TYPES

    def is_int(self):
        return self.name in dtype.SINT_TYPES + dtype.UINT_TYPES

    def is_bool(self):
        return self.is_int1()

    def kind(self):
        # Return int value following the type ordering bool < integer < fp
        if self.is_bool():
            return dtype.KIND.BOOLEAN
        elif self.is_int():
            return dtype.KIND.INTEGRAL
        else:
            assert self.is_floating()
            return dtype.KIND.FLOATING

    def get_int_max_value(self):
        if self.is_int_signed():
            return 2**(self.int_bitwidth - 1) - 1
        if self.is_int_unsigned():
            return 2**self.int_bitwidth - 1
        assert False

    def get_int_min_value(self):
        if self.is_int_signed():
            return -2**(self.int_bitwidth - 1)
        if self.is_int_unsigned():
            return 0
        assert False

    @staticmethod
    def is_dtype(type_str):
        return type_str in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES

    @staticmethod
    def is_void():
        raise RuntimeError("Not implemented")

    @staticmethod
    def is_block():
        return False

    @staticmethod
    def is_ptr():
        return False

    @staticmethod
    def is_const():
        return False

    def __eq__(self, other: dtype):
        if not isinstance(other, dtype):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash((self.name, ))

    @property
    def scalar(self):
        return self

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def to_ir(self, builder: ir.builder) -> ir.type:
        if self.name.startswith("fp8"):
            if self.name not in builder.options.supported_fp8_dtypes:
                raise ValueError(f'type {self} not supported in this architecture. '
                                 f'The supported fp8 dtypes are {builder.options.supported_fp8_dtypes}')

        if self.name == 'void':
            return builder.get_void_ty()
        elif self.name == 'int1':
            return builder.get_int1_ty()
        elif self.name in ('int8', 'uint8'):
            return builder.get_int8_ty()
        elif self.name in ('int16', 'uint16'):
            return builder.get_int16_ty()
        elif self.name in ('int32', 'uint32'):
            return builder.get_int32_ty()
        elif self.name in ('int64', 'uint64'):
            return builder.get_int64_ty()
        elif self.name == 'fp8e5':
            return builder.get_fp8e5_ty()
        elif self.name == 'fp8e5b16':
            return builder.get_fp8e5b16_ty()
        elif self.name == 'fp8e4nv':
            return builder.get_fp8e4nv_ty()
        elif self.name == 'fp8e4b8':
            return builder.get_fp8e4b8_ty()
        elif self.name == 'fp8e4b15':
            return builder.get_fp8e4b15_ty()
        elif self.name == 'fp16':
            return builder.get_half_ty()
        elif self.name == 'bf16':
            return builder.get_bf16_ty()
        elif self.name == 'fp32':
            return builder.get_float_ty()
        elif self.name == 'fp64':
            return builder.get_double_ty()
        raise ValueError(f'fail to convert {self} to ir type')

    def __str__(self):
        return self.name

    def codegen_name(self):
        if self.name.startswith("fp"):
            return "float" + self.name[2:]
        elif self.name.startswith("bf"):
            return "bfloat" + self.name[2:]
        else:
            return self.name

    @property
    def cache_key_part(self) -> str:
        """See cache_key_part() in triton.cc."""
        return self.name

    def __repr__(self):
        """Output of repr needs to be an evaluatable expression"""
        return f'triton.language.{self.codegen_name()}'

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        return tensor(handles[cursor], self), cursor + 1

    def mangle(self) -> str:
        if self.is_int():
            SIGNED = dtype.SIGNEDNESS.SIGNED
            prefix = 'i' if self.int_signedness == SIGNED else 'u'
            return prefix + str(self.int_bitwidth)
        if self.is_floating():
            return str(self)
        if self.is_void():
            return 'V'
        return super().mangle()

    def with_element_ty(self, element_ty: dtype):
        assert not self.is_block()
        return element_ty


# Some functions have a param named `dtype`, which shadows the `dtype` class.
# We can't change the param name because it is part of function's public API.
# Declare an alias so those functions can still reference the dtype class.
_DtypeClass = dtype


class pointer_type(dtype):

    def __init__(self, element_ty: dtype, address_space: int = 1, const: bool = False):
        element_ty = _unwrap_if_constexpr(element_ty)
        if not isinstance(element_ty, dtype):
            raise TypeError(f'element_ty has type `{type(element_ty).__name__}`; expected `dtype`.')
        self.element_ty = element_ty
        self.address_space = address_space
        self.const = const
        self.name = f'pointer<{element_ty}>' if not const else f'const_pointer<{element_ty}>'

    def to_ir(self, builder: ir.builder) -> ir.pointer_type:
        return builder.get_ptr_ty(self.element_ty.to_ir(builder), self.address_space)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def is_ptr(self):
        return True

    def is_const(self):
        return self.const

    def __eq__(self, other: pointer_type) -> bool:
        if not isinstance(other, pointer_type):
            return False
        return self.element_ty == other.element_ty and self.address_space == other.address_space and self.const == other.const

    @property
    def scalar(self):
        return self

    def mangle(self) -> str:
        return f"P{self.element_ty.mangle()}"


class block_type(dtype):

    def __init__(self, element_ty: dtype, shape: List):
        self.element_ty = element_ty

        # Note that block_type's shape is a list of int
        # while tensor's shape is a list of constexpr.
        assert (isinstance(shape, (list, tuple)))

        # shape can be empty ([]) when an input is a 0D tensor.
        self.shape = tuple(_unwrap_shape(shape))
        if not self.shape:
            raise TypeError('0d block_type is forbidden')

        self.numel = validate_block_shape(self.shape)
        self.name = f'<{self.shape}, {self.element_ty}>'

    def to_ir(self, builder: ir.builder) -> ir.block_type:
        return builder.get_block_ty(self.element_ty.to_ir(builder), self.shape)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def is_block(self):
        return True

    def get_block_shapes(self) -> Tuple[int]:
        return self.shape

    def with_element_ty(self, scalar_ty: dtype) -> block_type:
        return block_type(scalar_ty, self.shape)

    def __eq__(self, other) -> bool:
        if not isinstance(other, block_type):
            return False
        return self.element_ty == other.element_ty and self.shape == other.shape

    @property
    def scalar(self):
        return self.element_ty

    def mangle(self) -> str:
        elt = self.scalar.mangle()
        shape = '_'.join(map(str, self.shape))
        return f'{elt}S{shape}S'


class tuple_type(base_type):

    def __init__(self, types, fields=None):
        self.types = types
        self.fields = fields or [''] * len(types)
        self.name = '[' + ','.join([f"{k}:{v}" for k, v in zip(self.fields, self.types)]) + ']'

    def __str__(self):
        return self.name

    def __iter__(self):
        return iter(self.types)

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]):
        for ty in self.types:
            if not isinstance(ty, constexpr):
                ty._flatten_ir_types(builder, out)

    def __getitem__(self, index: int) -> dtype:
        return self.types[index]

    def __eq__(self, other):
        return type(self) is type(other) and self.types == other.types and self.fields == other.fields

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[tuple, int]:
        values = []
        for ty in self.types:
            value, cursor = ty._unflatten_ir(handles, cursor)
            values.append(value)
        return tuple(values, self), cursor

    def mangle(self):
        return 'T' + '_'.join(ty.mangle for ty in self.types) + 'T'


class slice_type(dtype):

    def __init__(self):
        self.name = 'slice_type'


# scalar types
void = dtype('void')
int1 = dtype('int1')
int8 = dtype('int8')
int16 = dtype('int16')
int32 = dtype('int32')
int64 = dtype('int64')
uint8 = dtype('uint8')
uint16 = dtype('uint16')
uint32 = dtype('uint32')
uint64 = dtype('uint64')
float8e5 = dtype('fp8e5')
float8e5b16 = dtype('fp8e5b16')
float8e4nv = dtype('fp8e4nv')
float8e4b8 = dtype('fp8e4b8')
float8e4b15 = dtype('fp8e4b15')
float16 = dtype('fp16')
bfloat16 = dtype('bf16')
float32 = dtype('fp32')
float64 = dtype('fp64')
# pointer types
pi32_t = pointer_type(int32)


def get_int_dtype(bitwidth: int, signed: bool) -> dtype:
    if bitwidth == 1:
        return int1
    elif bitwidth == 8 and signed:
        return int8
    elif bitwidth == 8 and not signed:
        return uint8
    elif bitwidth == 16 and signed:
        return int16
    elif bitwidth == 16 and not signed:
        return uint16
    elif bitwidth == 32 and signed:
        return int32
    elif bitwidth == 32 and not signed:
        return uint32
    elif bitwidth == 64 and signed:
        return int64
    elif bitwidth == 64 and not signed:
        return uint64
    else:
        raise ValueError(f'Unsupported bitwidth {bitwidth} and signedness {signed}')


# -----------------------
# tensor
# -----------------------


class tensor(base_value):
    """Represents an N-dimensional array of values or pointers.

    :code:`tensor` is the fundamental data structure in Triton programs.  Most
    functions in :py:mod:`triton.language` operate on and return tensors.

    Most of the named member functions here are duplicates of the free functions
    in :code:`triton.language`.  For example, :code:`triton.language.sqrt(x)` is
    equivalent to :code:`x.sqrt()`.

    :code:`tensor` also defines most of the magic/dunder methods, so you can
    write :code:`x+y`, :code:`x << 2`, etc.

    .. rubric:: Constructors
    ..
       For some reason Sphinx includes __init__ before printing the full table
       of methods.  Not what I want, but I can't figure out how to fix it.  Give
       it its own section so it looks intentional. :)
    """

    def __init__(self, handle, type: dtype):
        """Not called by user code."""
        super().__init__()
        # IR handle
        self.handle = handle
        # Block shape
        self.shape = type.shape if type.is_block() else ()
        self.numel = 1
        for s in self.shape:
            self.numel *= s
        self.numel = constexpr(self.numel)
        self.type = type  # Tensor type (can be block_type)
        # Following the practice in pytorch, dtype is scalar type
        self.dtype = type.scalar
        self.shape = tuple([constexpr(s) for s in self.shape])

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    def __str__(self) -> str:
        # ex. "float32[16, 32]"
        return str(self.dtype) + '[' + ', '.join(str(s) for s in self.shape) + ']'

    @builtin
    def __add__(self, other, _semantic=None):
        return add(self, other, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __radd__(self, other, _semantic=None):
        return add(other, self, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __sub__(self, other, _semantic=None):
        return sub(self, other, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __rsub__(self, other, _semantic=None):
        return sub(other, self, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __mul__(self, other, _semantic=None):
        return mul(self, other, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __rmul__(self, other, _semantic=None):
        return mul(other, self, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __truediv__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.truediv(self, other)

    @builtin
    def __rtruediv__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.truediv(other, self)

    @builtin
    def __floordiv__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.floordiv(self, other)

    @builtin
    def __rfloordiv__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.floordiv(other, self)

    @builtin
    def __mod__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.mod(self, other)

    @builtin
    def __rmod__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.mod(other, self)

    # unary operators
    @builtin
    def __neg__(self, _semantic=None):
        return _semantic.minus(self)

    @builtin
    def __invert__(self, _semantic=None):
        return _semantic.invert(self)

    # bitwise operators

    @builtin
    def __and__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.and_(self, other)

    @builtin
    def __rand__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.and_(other, self)

    @builtin
    def __or__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.or_(self, other)

    @builtin
    def __ror__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.or_(other, self)

    @builtin
    def __xor__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.xor_(self, other)

    @builtin
    def __rxor__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.xor_(other, self)

    @builtin
    def __lshift__(self, other, _semantic=None):
        check_bit_width(self, other)
        other = _unwrap_if_constexpr(other)
        return _semantic.shl(self, other)

    @builtin
    def __rlshift__(self, other, _semantic=None):
        check_bit_width(other, self)
        other = _unwrap_if_constexpr(other)
        return _semantic.shl(other, self)

    @builtin
    def __rshift__(self, other, _semantic=None):
        check_bit_width(self, other)
        other = _unwrap_if_constexpr(other)
        if self.dtype.is_int_signed():
            return _semantic.ashr(self, other)
        else:
            return _semantic.lshr(self, other)

    @builtin
    def __rrshift__(self, other, _semantic=None):
        check_bit_width(other, self)
        other = _unwrap_if_constexpr(other)
        if self.dtype.is_int_signed():
            return _semantic.ashr(other, self)
        else:
            return _semantic.lshr(other, self)

    # >
    @builtin
    def __gt__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.greater_than(self, other)

    @builtin
    def __rgt__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.greater_than(other, self)

    # >=
    @builtin
    def __ge__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.greater_equal(self, other)

    @builtin
    def __rge__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.greater_equal(other, self)

    # <
    @builtin
    def __lt__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.less_than(self, other)

    @builtin
    def __rlt__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.less_than(other, self)

    # <=
    @builtin
    def __le__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.less_equal(self, other)

    @builtin
    def __rle__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.less_equal(other, self)

    # ==
    @builtin
    def __eq__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.equal(self, other)

    @builtin
    def __req__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.equal(other, self)

    @builtin
    def __ne__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.not_equal(self, other)

    @builtin
    def __rne__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.not_equal(other, self)

    @builtin
    def logical_and(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.logical_and(self, other)

    @builtin
    def logical_or(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.logical_or(self, other)

    # note: __not__ isn't actually a magic method in python
    # but it's ok because our ASTVisitor handles it
    @builtin
    def __not__(self, _semantic=None):
        return _semantic.not_(self)

    @builtin
    def __getitem__(self, slices, _semantic=None):
        if isinstance(slices, (builtins.slice, slice, constexpr)) or slices is None:
            slices = [slices]
        if isinstance(slices, tuple):
            slices = slices.values
        ret = self
        for dim, sl in enumerate(slices):
            if _unwrap_if_constexpr(sl) is None:
                ret = _semantic.expand_dims(ret, dim)
            elif isinstance(sl, (builtins.slice, slice)) and all(
                    _unwrap_if_constexpr(arg) is None for arg in (sl.start, sl.stop, sl.step)):
                pass  # an unsqueeze
            else:
                raise ValueError(f"unsupported tensor index: {sl}")
        return ret

    @property
    def T(self):
        """Transposes a 2D tensor."""
        assert False, "Transposition must be created by the AST Visitor"

    @builtin
    def to(self, dtype: dtype, fp_downcast_rounding: Optional[str] = None, bitcast: bool = False, _semantic=None):
        """
        Alias for :py:func:`tensor.cast`.
        """
        return cast(self, dtype, fp_downcast_rounding, bitcast, _semantic=_semantic)

    # Type stubs for functions added by the _tensor_member_fn decorator.
    # (Unfortunately these can't be created automatically.)
    #
    # We couldn't write these definitions out even if we wanted to, because some
    # of these functions are defined in standard.py.
    def broadcast_to(self, *shape) -> tensor:
        ...

    def trans(self, *dims) -> tensor:
        ...

    def permute(self, *dims) -> tensor:
        ...

    def split(self) -> tuple[tensor, tensor]:
        ...

    def view(self, *shape) -> tensor:
        ...

    def reshape(self, *shape) -> tensor:
        ...

    def expand_dims(self, axis) -> tensor:
        ...

    def cast(self, dtype, fp_downcast_rounding=None, bitcast=False) -> tensor:
        ...

    def store(self, value, mask=None, boundary_check=(), cache_modifier="", eviction_policy="") -> tensor:
        ...

    def advance(self, offsets) -> tensor:
        ...

    def atomic_cas(self, cmp, val, sem=None, scope=None) -> tensor:
        ...

    def atomic_xchg(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_add(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_max(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_min(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_and(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_or(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_xor(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def exp(self) -> tensor:
        ...

    def log(self) -> tensor:
        ...

    def cos(self) -> tensor:
        ...

    def sin(self) -> tensor:
        ...

    def sqrt(self) -> tensor:
        ...

    def rsqrt(self) -> tensor:
        ...

    def abs(self) -> tensor:
        ...

    def reduce(self, axis, combine_fn, keep_dims=False) -> tensor:
        ...

    def associative_scan(self, axis, combine_fn, reverse=False) -> tensor:
        ...

    def gather(self, indices, axis) -> tensor:
        ...

    def histogram(self, num_bins) -> tensor:
        ...

    def cdiv(self, div) -> tensor:
        ...

    def sigmoid(self) -> tensor:
        ...

    def softmax(self, dim=None, keep_dims=False, ieee_rounding=False) -> tensor:
        ...

    def ravel(self) -> tensor:
        ...

    def max(self, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def argmax(self, axis, tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def min(self, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def argmin(self, axis, tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def sum(self, axis=None, keep_dims=False, dtype=None) -> tensor:
        ...

    def xor_sum(self, axis=None, keep_dims=False) -> tensor:
        ...

    def reduce_or(self, axis=None, keep_dims=False) -> tensor:
        ...

    def cumsum(self, axis=0, reverse=False) -> tensor:
        ...

    def cumprod(self, axis=0, reverse=False) -> tensor:
        ...

    def sort(self, dim: constexpr = None, descending: constexpr = CONSTEXPR_0) -> tensor:
        ...

    def flip(self, dim=None) -> tensor:
        ...


class tuple(base_value):

    def __init__(self, args: Sequence, type: tuple_type = None):
        self.values = [i for i in args]

        def get_type(x):
            if isinstance(x, dtype):
                return dtype
            if isinstance(x, (int, float)):
                return constexpr
            return x.type

        self.type = type or tuple_type([get_type(x) for x in self.values])

    def __getitem__(self, idx: constexpr):
        if isinstance(idx, int):
            idx = constexpr(idx)
        if isinstance(idx, constexpr):
            return self.values[idx]
        else:
            assert isinstance(idx, (slice, builtins.slice))
            return tuple(self.values[idx.start:idx.stop:idx.step])

    def __getattr__(self, name):
        return self.values[self.type.fields.index(name)]

    # TODO: remove
    def __setitem__(self, idx: constexpr, value):
        if isinstance(idx, int):
            idx = constexpr(idx)
        assert isinstance(idx, constexpr)
        self.values[idx] = value

    def __add__(self, other):
        other = _normalize_tuple(other)
        return tuple(self.values + other.values)
        # return tuple(a + b for a, b in zip(self.values, other.values))

    def __mul__(self, other):
        assert isinstance(other, constexpr)
        return tuple(self.values * other.value)

    def __eq__(self, other):
        other = _normalize_tuple(other)
        return constexpr(self.values == other.values)

    def __hash__(self):
        return hash(builtins.tuple(self.values))

    def __str__(self):
        return str([str(x) for x in self.values])

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def _flatten_ir(self, handles: List[ir.value]):
        for v in self.values:
            v._flatten_ir(handles)

    def __repr__(self):
        return f"({' ,'.join(repr(x) for x in self.values)})"


class slice:

    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step
        self.type = slice_type()


class tensor_descriptor_base_type(base_type):

    def __init__(self, block_type: block_type):
        self.block_type = block_type

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[tensor_descriptor_base, int]:
        value = tensor_descriptor_base(handles[cursor], self.block_type)
        return value, cursor + 1

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        is_signed = self.block_type.element_ty.is_int_signed()
        out.append(builder.create_tensor_descriptor_type(self.block_type.to_ir(builder), is_signed))

    def __str__(self) -> str:
        # ex. "tensor_descriptor<float32[16, 32]>"
        return f"tensor_descriptor<{self.block_type}>"

    def __eq__(self, other) -> bool:
        if type(other) is not type(self):
            return False
        return self.block_type == other.block_type

    def __neq__(self, other) -> bool:
        return not (self == other)

    def mangle(self) -> str:
        return f"TD{self.block_type.mangle()}"


class tensor_descriptor_base(base_value):
    """"
    A tensor descriptor with unknown shape and strides
    """

    def __init__(self, handle, block_type: block_type):
        """Not called by user code."""
        super().__init__()

        self.handle = handle  # IR handle
        self.type = tensor_descriptor_base_type(block_type)  # Tensor type (block_type)

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    @property
    def block_type(self):
        return self.type.block_type

    @property
    def block_shape(self):
        return self.type.block_type.shape

    @property
    def dtype(self):
        return self.type.block_type.element_ty

    def __str__(self) -> str:
        return str(self.type)

    @builtin
    def load(self, offsets: Sequence[constexpr | tensor], _semantic=None) -> tensor:
        """Load a block from the descriptor starting at the given element offsets.

        Values outside of the tensor bounds will be filled with zeros.

        :note: Offset must be a multiple of 16-bytes
        """
        return _semantic.descriptor_load(self, offsets, "", "")

    @builtin
    def store(self, offsets: Sequence[constexpr | tensor], value: tensor, _semantic=None) -> tensor:
        """Store a block from the descriptor starting at the given element offsets.

        Values outside of the tensor bounds will be ignored.

        :note: Offset must be a multiple of 16-bytes
        """
        return _semantic.descriptor_store(self, value, offsets)

    @builtin
    def atomic_add(self, offsets: Sequence[constexpr | tensor], value: tensor, _semantic=None) -> tensor:
        return _semantic.descriptor_atomic_add(self, value, offsets)

    @builtin
    def atomic_min(self, offsets: Sequence[constexpr | tensor], value: tensor, _semantic=None) -> tensor:
        return _semantic.descriptor_atomic_min(self, value, offsets)

    @builtin
    def atomic_max(self, offsets: Sequence[constexpr | tensor], value: tensor, _semantic=None) -> tensor:
        return _semantic.descriptor_atomic_max(self, value, offsets)

    @builtin
    def atomic_and(self, offsets: Sequence[constexpr | tensor], value: tensor, _semantic=None) -> tensor:
        return _semantic.descriptor_atomic_and(self, value, offsets)

    @builtin
    def atomic_or(self, offsets: Sequence[constexpr | tensor], value: tensor, _semantic=None) -> tensor:
        return _semantic.descriptor_atomic_or(self, value, offsets)

    @builtin
    def atomic_xor(self, offsets: Sequence[constexpr | tensor], value: tensor, _semantic=None) -> tensor:
        return _semantic.descriptor_atomic_xor(self, value, offsets)

    @builtin
    def gather(self, *args, _semantic=None) -> tensor:
        """Gather multiple descriptors worth of data"""
        assert len(args) == 2, f"descriptor gather only supports 2D indexing, but got {len(args)}"
        x_offsets = args[0]
        y_offset = args[1]
        return _semantic.descriptor_gather(self, x_offsets, y_offset, "", "")

    @builtin
    def scatter(self, value, *args, _semantic=None) -> tensor:
        """Scatter multiple descriptors worth of data"""
        assert len(args) == 2, f"descriptor scatter only supports 2D indexing, but got {len(args)}"
        x_offsets = args[0]
        y_offset = args[1]
        return _semantic.descriptor_scatter(self, value, x_offsets, y_offset)


class tensor_descriptor_type(tensor_descriptor_base_type):

    def __init__(self, block_type: block_type, shape_type: tuple_type, strides_type: tuple_type):
        self.block_type = block_type
        self.shape_type = shape_type
        self.strides_type = strides_type

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[tensor_descriptor_base, int]:
        handle = handles[cursor]
        cursor += 1
        shape, cursor = self.shape_type._unflatten_ir(handles, cursor)
        strides, cursor = self.strides_type._unflatten_ir(handles, cursor)
        shape = shape.values
        strides = strides.values
        value = tensor_descriptor(handle, shape, strides, self.block_type)
        return value, cursor

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        super()._flatten_ir_types(builder, out)
        self.shape_type._flatten_ir_types(builder, out)
        self.strides_type._flatten_ir_types(builder, out)

    def __eq__(self, other):
        return super().__eq__(other) and (self.shape_type == other.shape_type) and (self.strides_type
                                                                                    == other.strides_type)


class tensor_descriptor(tensor_descriptor_base):
    """A descriptor representing a tensor in global memory.
    """

    def __init__(self, handle, shape: List[tensor], strides: List[tensor], block_type: block_type):
        """Not called by user code."""
        # IR handle
        super().__init__(handle, block_type)
        # Global shape
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        self.type = tensor_descriptor_type(
            block_type,
            shape_type=self.shape.type,
            strides_type=self.strides.type,
        )

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)
        self.shape._flatten_ir(handles)
        self.strides._flatten_ir(handles)


# -----------------------
# aggregate
# -----------------------


@dataclass(frozen=True)
class _aggregate_type(base_type):
    """A generic base type for all Triton aggregate types.

    This class contains a reference to the original user-defined Python class
    and a list of class fields with their Triton types.
    """

    base_cls: type
    fields: List[Tuple[str, base_type]]

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[ir.value, int]:
        instance = self.base_cls._get_instance()
        for name, ty in self.fields:
            value, cursor = ty._unflatten_ir(handles, cursor)
            setattr(instance, name, value)
        return instance, cursor

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        for name, ty in self.fields:
            ty._flatten_ir_types(builder, out)

    def mangle(self) -> str:
        name = f"{self.base_cls.__module__}.{self.base_cls.__qualname__}"
        fields = [ty.mangle() for (name, ty) in self.fields]
        return f"{name}<{', '.join(fields)}>"


def _aggregate(cls):

    # Define the wrapped Triton value type.
    class aggregate_value(base_value):
        __triton_builtin__ = True
        __triton_aggregate__ = True

        @classmethod
        def _get_instance(this_cls):
            return super().__new__(this_cls)

        def __new__(this_cls, *args, _semantic=None, _generator=None, **kwargs):
            # Call into the user-defined constructor.
            instance = this_cls._get_instance()
            if isinstance(cls.__init__, JITFunction):
                raise ValueError(f"{cls.__name__}.__init__ cannot be a @triton.jit function")
            extra_kwargs = {}
            if "_semantic" in inspect.signature(cls.__init__).parameters:
                extra_kwargs["_semantic"] = _semantic
            if "_generator" in inspect.signature(cls.__init__).parameters:
                extra_kwargs["_generator"] = _generator
            cls.__init__(instance, *args, **extra_kwargs, **kwargs)

            # Require that the user-defined constructor initialized all fields.
            for name in cls.__annotations__.keys():
                if not hasattr(instance, name):
                    raise AttributeError(f"constructor for {cls.__name__} did not initialize attribute '{name}'")

            return instance

        # Only allow setting attributes defined in the class annotations.
        def __setattr__(self, name, value):
            if name not in cls.__annotations__:
                raise AttributeError(f"{cls.__name__} has no attribute '{name}'")
            if not isinstance(value, cls.__annotations__[name]):
                raise TypeError(f"Expected {cls.__annotations__[name]} for attribute '{name}', got {type(value)}")
            super().__setattr__(name, value)

        def _flatten_ir(self, handles: List[ir.value]) -> None:
            for name in cls.__annotations__.keys():
                getattr(self, name)._flatten_ir(handles)

        @property
        def type(self):
            return _aggregate_type(aggregate_value,
                                   [(name, getattr(self, name).type) for name in cls.__annotations__.keys()])

    for (name, member) in inspect.getmembers(cls):
        if inspect.isfunction(member) or inspect.ismethod(member) or isinstance(member, JITFunction):
            if name != "__init__":
                setattr(aggregate_value, name, member)

    aggregate_value.__name__ = cls.__name__
    aggregate_value.__module__ = cls.__module__
    aggregate_value.__qualname__ = cls.__qualname__
    aggregate_value.__doc__ = cls.__doc__

    return aggregate_value


# -----------------------
# SPMD Programming Model
# -----------------------


@builtin
def program_id(axis, _semantic=None):
    """
    Returns the id of the current program instance along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Must be 0, 1 or 2.
    :type axis: int
    """
    # if axis == -1:
    #     pid0 = _semantic.program_id(0)
    #     pid1 = _semantic.program_id(1)
    #     pid2 = _semantic.program_id(2)
    #     npg0 = _semantic.num_programs(0)
    #     npg1 = _semantic.num_programs(1)
    #     return pid0 + pid1*npg0 + pid2*npg0*npg1
    axis = _unwrap_if_constexpr(axis)
    return _semantic.program_id(axis)


@builtin
def num_programs(axis, _semantic=None):
    """
    Returns the number of program instances launched along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Must be 0, 1 or 2.
    :type axis: int
    """
    axis = _unwrap_if_constexpr(axis)
    return _semantic.num_programs(axis)


# -----------------------
# Block Initialization
# -----------------------


@builtin
def arange(start, end, _semantic=None):
    start = _unwrap_if_constexpr(start)
    end = _unwrap_if_constexpr(end)
    return _semantic.arange(start, end)


arange.__doc__ = f"""
    Returns contiguous values within the half-open interval :code:`[start,
    end)`.  :code:`end - start` must be less than or equal to
    :code:`TRITON_MAX_TENSOR_NUMEL = {TRITON_MAX_TENSOR_NUMEL}`

    :param start: Start of the interval. Must be a power of two.
    :type start: int32
    :param end: End of the interval. Must be a power of two greater than
        :code:`start`.
    :type end: int32
"""


def _unwrap_shape(shape):
    shape = _unwrap_if_constexpr(shape)
    return [_unwrap_if_constexpr(s) for s in shape]


def _shape_check_impl(shape):
    shape = _unwrap_shape(shape)
    validate_block_shape(shape)
    return shape


@builtin
def full(shape, value, dtype, _semantic=None):
    """
    Returns a tensor filled with the scalar value for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :type shape: tuple of ints
    :param value: A scalar value to fill the array with
    :type value: scalar
    :param dtype: Data type of the new array, e.g., :code:`tl.float16`
    :type dtype: tl.dtype
    """
    shape = _shape_check_impl(shape)
    value = _unwrap_if_constexpr(value)
    dtype = _unwrap_if_constexpr(dtype)
    return _semantic.full(shape, value, dtype)


# -----------------------
# Shape Manipulation
# -----------------------


@builtin
def broadcast(input, other, _semantic=None):
    """
    Tries to broadcast the two given blocks to a common compatible shape.

    :param input: The first input tensor.
    :type input: Block
    :param other: The second input tensor.
    :type other: Block
    """
    return _semantic.broadcast_impl_value(input, other)


@_tensor_member_fn
@builtin
def broadcast_to(input, *shape, _semantic=None):
    """
    Tries to broadcast the given tensor to a new :code:`shape`.

    :param input: The input tensor.
    :type input: Block
    :param shape: The desired shape.
    :type shape:

    :code:`shape` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        broadcast_to(x, (32, 32))
        broadcast_to(x, 32, 32)
    """
    shape = _shape_check_impl(_unwrap_iterable(shape))
    return _semantic.broadcast_impl_shape(input, shape)


@_tensor_member_fn
@builtin
def trans(input: tensor, *dims, _semantic=None):
    """
    Permutes the dimensions of a tensor.

    If the parameter :code:`dims` is not specified, the function defaults to a (1,0) permutation,
    effectively transposing a 2D tensor.

    :param input: The input tensor.
    :param dims: The desired ordering of dimensions.  For example,
        :code:`(2, 1, 0)` reverses the order dims in a 3D tensor.

    :code:`dims` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        trans(x, (2, 1, 0))
        trans(x, 2, 1, 0)

    :py:func:`permute` is equivalent to this function, except it doesn't
    have the special case when no permutation is specified.
    """
    dims = _unwrap_iterable(dims)
    if not dims:
        dims = (1, 0)
    return _semantic.permute(input, dims)


@_tensor_member_fn
@builtin
def permute(input, *dims, _semantic=None):
    """
    Permutes the dimensions of a tensor.

    :param input: The input tensor.
    :type input: Block
    :param dims: The desired ordering of dimensions.  For example,
        :code:`(2, 1, 0)` reverses the order dims in a 3D tensor.

    :code:`dims` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        permute(x, (2, 1, 0))
        permute(x, 2, 1, 0)

    :py:func:`trans` is equivalent to this function, except when
    :code:`dims` is empty, it tries to do a (1,0) permutation.
    """
    dims = _unwrap_iterable(dims)
    return _semantic.permute(input, dims)


@builtin
def cat(input, other, can_reorder=False, _semantic=None):
    """
    Concatenate the given blocks

    :param input: The first input tensor.
    :type input: Tensor
    :param other: The second input tensor.
    :type other: Tensor
    :param reorder: Compiler hint. If true, the compiler is
        allowed to reorder elements while concatenating inputs.  Only use if the
        order does not matter (e.g., result is only used in reduction ops).
        Current implementation of `cat` supports only can_reorder=True.
    """
    return _semantic.cat(input, other, can_reorder)


@builtin
def join(a, b, _semantic=None):
    """
    Join the given tensors in a new, minor dimension.

    For example, given two tensors of shape (4,8), produces a new tensor of
    shape (4,8,2).  Given two scalars, returns a tensor of shape (2).

    The two inputs are broadcasted to be the same shape.

    If you want to join more than two elements, you can use multiple calls to
    this function.  This reflects the constraint in Triton that tensors must
    have power-of-two sizes.

    join is the inverse of split.

    :param a: The first input tensor.
    :type a: Tensor
    :param b: The second input tensor.
    :type b: Tensor
    """
    return _semantic.join(a, b)


@jit
def _take_first(a, b):
    return a


def _unsplat(x, _semantic=None, _generator=None):
    """
    Convert a single-element tensor to a scalar.
    """
    if len(x.shape) == 0:
        return x
    numel = 1
    for d in x.shape:
        numel *= d
    assert numel == 1, "can only unsplat single-element tensors"
    if len(x.shape) >= 2:
        x = _semantic.reshape(x, [1])
    x = typing.cast(tensor, reduce(x, 0, _take_first, _semantic=_semantic, _generator=_generator))
    return x


@_tensor_member_fn
@builtin
def split(a, _semantic=None, _generator=None) -> tuple[tensor, tensor]:
    """
    Split a tensor in two along its last dim, which must have size 2.

    For example, given a tensor of shape (4,8,2), produces two tensors of shape
    (4,8).  Given a tensor of shape (2), returns two scalars.

    If you want to split into more than two pieces, you can use multiple calls
    to this function (probably plus calling reshape).  This reflects the
    constraint in Triton that tensors must have power-of-two sizes.

    split is the inverse of join.

    :param a: The tensor to split.
    :type a: Tensor
    """
    # If len(a.shape) == 1, i.e. a.shape == [2], we should return two scalars.
    # But _semantic.split can only handle returning tensors.  Work around this by
    # expanding the input to shape [1,2] and then reducing the result.
    was_rank_1 = len(a.shape) == 1
    if was_rank_1:
        a = _semantic.expand_dims(a, 0)

    out_lhs, out_rhs = _semantic.split(a)

    if was_rank_1:
        # Currently `reduce` is the best way to convert a tensor of shape [1] to a scalar.
        out_lhs = _unsplat(out_lhs, _semantic=_semantic, _generator=_generator)
        out_rhs = _unsplat(out_rhs, _semantic=_semantic, _generator=_generator)

    return out_lhs, out_rhs


@_tensor_member_fn
@builtin
def view(input, *shape, _semantic=None):
    """
    Returns a tensor with the same elements as `input` but a different shape.
    The order of the elements may not be preserved.

    :param input: The input tensor.
    :type input: Block
    :param shape: The desired shape.

    :code:`shape` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        view(x, (32, 32))
        view(x, 32, 32)
    """
    warn("view is deprecated, please use reshape with can_reorder being true.")
    shape = _shape_check_impl(_unwrap_iterable(shape))
    return _semantic.reshape(input, shape, can_reorder=True)


@_tensor_member_fn
@builtin
def item(input, _semantic=None, _generator=None):
    """
    Converts a single-element tensor into a scalar.
    """
    return _unsplat(input, _semantic=_semantic, _generator=_generator)


@_tensor_member_fn
@builtin
def reshape(input, *shape, can_reorder=False, _semantic=None, _generator=None):
    """
    Returns a tensor with the same number of elements as input but with the
    provided shape.

    :param input: The input tensor.
    :type input: Block
    :param shape: The new shape.

    :code:`shape` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        reshape(x, (32, 32))
        reshape(x, 32, 32)
    """
    shape = _shape_check_impl(_unwrap_iterable(shape))
    if len(shape) == 0:
        return _unsplat(input, _semantic=_semantic, _generator=_generator)
    return _semantic.reshape(input, shape, can_reorder)


def _wrap_axis(axis, ndim):
    if not (-ndim <= axis < ndim):
        raise ValueError(f"invalid axis {axis}. Expected {-ndim} <= axis < {ndim}")

    return axis if axis >= 0 else axis + ndim


@_tensor_member_fn
@builtin
def expand_dims(input, axis, _semantic=None):
    """
    Expand the shape of a tensor, by inserting new length-1 dimensions.

    Axis indices are with respect to the resulting tensor, so
    ``result.shape[axis]`` will be 1 for each axis.

    :param input: The input tensor.
    :type input: tl.tensor
    :param axis: The indices to add new axes
    :type axis: int | Sequence[int]

    """
    input = _semantic.to_tensor(input)
    axis = _unwrap_if_constexpr(axis)
    axes = list(axis) if isinstance(axis, (Sequence, tuple)) else [axis]
    new_ndim = len(input.shape) + len(axes)
    axes = [_wrap_axis(_unwrap_if_constexpr(d), new_ndim) for d in axes]

    if len(set(axes)) != len(axes):
        raise ValueError(f"expand_dims received duplicate axes, normalized axes = {axes}")

    ret = input
    for a in sorted(axes):
        ret = _semantic.expand_dims(ret, a)
    return ret


@_tensor_member_fn
@builtin
def cast(input, dtype: dtype, fp_downcast_rounding: Optional[str] = None, bitcast: bool = False, _semantic=None):
    """
    Casts a tensor to the given :code:`dtype`.

    :param dtype: The target data type.
    :type dtype: tl.dtype
    :param fp_downcast_rounding: The rounding mode for downcasting
        floating-point values. This parameter is only used when self is a
        floating-point tensor and dtype is a floating-point type with a
        smaller bitwidth. Supported values are :code:`"rtne"` (round to
        nearest, ties to even) and :code:`"rtz"` (round towards zero).
    :type fp_downcast_rounding: str, optional
    :param bitcast: If true, the tensor is bitcasted to the given
        :code:`dtype`, instead of being numerically casted.
    :type bitcast: bool, optional
    """
    input = _semantic.to_tensor(input)
    dtype = _unwrap_if_constexpr(dtype)
    fp_downcast_rounding = _unwrap_if_constexpr(fp_downcast_rounding)
    bitcast = _unwrap_if_constexpr(bitcast)
    if bitcast:
        return _semantic.bitcast(input, dtype)
    return _semantic.cast(input, dtype, fp_downcast_rounding)


# -----------------------
# Linear Algebra
# -----------------------


@builtin
def dot(input, other, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None, out_dtype=float32,
        _semantic=None):
    """
    Returns the matrix product of two blocks.

    The two blocks must both be two-dimensional or three-dimensional and have compatible inner dimensions.
    For three-dimensional blocks, `tl.dot` performs the batched matrix product,
    where the first dimension of each block represents the batch dimension.

    :param input: The first tensor to be multiplied.
    :type input: 2D or 3D tensor of scalar-type in {:code:`int8`, :code:`float8_e5m2`, :code:`float16`, :code:`bfloat16`, :code:`float32`}
    :param other: The second tensor to be multiplied.
    :type other: 2D or 3D tensor of scalar-type in {:code:`int8`, :code:`float8_e5m2`, :code:`float16`, :code:`bfloat16`, :code:`float32`}
    :param acc: The accumulator tensor. If not None, the result is added to this tensor.
    :type acc: 2D or 3D tensor of scalar-type in {:code:`float16`, :code:`float32`, :code:`int32`}
    :param input_precision: How to exercise the Tensor Cores for f32 x f32. If
      the device does not have Tensor Cores or the inputs are not of dtype f32,
      this option is ignored. For devices that do have tensor cores, the
      default precision is tf32.
    :type input_precision: string. Available options for nvidia: :code:`"tf32"`, :code:`"tf32x3"`, :code:`"ieee"`. Default: :code:`"tf32"`. Available options for amd: :code:`"ieee"`, (CDNA3 only) :code:`"tf32"`.
    :param allow_tf32: *Deprecated.* If true, input_precision is set to "tf32".
      Only one of :code:`input_precision` and :code:`allow_tf32` can be
      specified (i.e. at least one must be :code:`None`).
    """
    assert input_precision is None or allow_tf32 is None, "Only one of input_precision and allow_tf32 can be specified"
    if input_precision is None:
        supports_tf32 = "tf32" in _semantic.builder.options.allowed_dot_input_precisions
        input_precision = knobs.language.fp32_default or ("tf32" if (supports_tf32 and
                                                                     (allow_tf32 or allow_tf32 is None)) else "ieee")

    input_precision = _unwrap_if_constexpr(input_precision)
    out_dtype = _unwrap_if_constexpr(out_dtype)
    max_num_imprecise_acc = _unwrap_if_constexpr(max_num_imprecise_acc)
    acc = _unwrap_if_constexpr(acc)
    return _semantic.dot(input, other, acc, input_precision, max_num_imprecise_acc, out_dtype)


@builtin
def dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, fast_math=False, lhs_k_pack=True,
               rhs_k_pack=True, out_dtype=float32, _semantic=None):
    """
    Returns the matrix product of two blocks in microscaling format.

    lhs and rhs use microscaling formats described here:
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

    Software emulation enables targeting hardware architectures without native microscaling
    operation support. Right now for such case, microscaled lhs/rhs are upcasted to
    :code:`bf16` element type beforehand for dot computation, with one exception:
    for AMD CDNA3 specifically, if one of the inputs is of :code:`fp16` element type,
    the other input is also upcasted to :code:`fp16` element type instead.
    This behavior is experimental and may be subject to change in the future.

    :param lhs: The first tensor to be multiplied.
    :type lhs: 2D tensor representing fp4, fp8 or bf16 elements. Fp4 elements are packed into uint8 inputs with the first element in lower bits. Fp8 are stored as uint8 or the corresponding fp8 type.
    :param lhs_scale: Scale factor for lhs tensor.
    :type lhs_scale: e8m0 type represented as an uint8 tensor.
    :param lhs_format: format of the lhs tensor. Available formats: {:code:`e2m1`, :code:`e4m3`, :code:`e5m2`, :code:`bf16`, :code:`fp16`}.
    :type lhs_format: str
    :param rhs: The second tensor to be multiplied.
    :type rhs: 2D tensor representing fp4, fp8 or bf16 elements. Fp4 elements are packed into uint8 inputs with the first element in lower bits. Fp8 are stored as uint8 or the corresponding fp8 type.
    :param rhs_scale: Scale factor for rhs tensor.
    :type rhs_scale: e8m0 type represented as an uint8 tensor.
    :param rhs_format: format of the rhs tensor. Available formats: {:code:`e2m1`, :code:`e4m3`, :code:`e5m2`, :code:`bf16`, :code:`fp16`}.
    :type rhs_format: str
    :param acc: The accumulator tensor. If not None, the result is added to this tensor.
    :param lhs_k_pack: If false, the lhs tensor is packed into uint8 along M dimension.
    :type lhs_k_pack: bool, optional
    :param rhs_k_pack: If false, the rhs tensor is packed into uint8 along N dimension.
    :type rhs_k_pack: bool, optional
    """
    out_dtype = _unwrap_if_constexpr(out_dtype)
    assert out_dtype == float32, "Only float32 is supported for out_dtype at the moment"
    return _semantic.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, fast_math, lhs_k_pack,
                                rhs_k_pack, out_dtype)


# -----------------------
# Non-Atomic Memory Operations
# -----------------------


@builtin
def load(pointer, mask=None, other=None, boundary_check=(), padding_option="", cache_modifier="", eviction_policy="",
         volatile=False, _semantic=None):
    """
    Return a tensor of data whose values are loaded from memory at location defined by `pointer`:

        (1) If `pointer` is a single element pointer, a scalar is be loaded.  In
            this case:

            - `mask` and `other` must also be scalars,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (2) If `pointer` is an N-dimensional tensor of pointers, an
            N-dimensional tensor is loaded.  In this case:

            - `mask` and `other` are implicitly broadcast to `pointer.shape`,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (3) If `pointer` is a block pointer defined by `make_block_ptr`, a
            tensor is loaded.  In this case:

            - `mask` and `other` must be `None`, and
            - `boundary_check` and `padding_option` can be specified to control the behavior of out-of-bound access.

    :param pointer: Pointer to the data to be loaded
    :type pointer: `triton.PointerType`, or block of `dtype=triton.PointerType`
    :param mask: if `mask[idx]` is false, do not load the data at address `pointer[idx]`
        (must be `None` with block pointers)
    :type mask: Block of `triton.int1`, optional
    :param other: if `mask[idx]` is false, return `other[idx]`
    :type other: Block, optional
    :param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check
    :type boundary_check: tuple of ints, optional
    :param padding_option: should be one of {"", "zero", "nan"}, the padding value to use while out of bounds. "" means an undefined value.
    :param cache_modifier: changes cache option in NVIDIA PTX
    :type cache_modifier: str, optional, should be one of {"", ".ca", ".cg", ".cv"}, where ".ca" stands for
        cache at all levels, ".cg" stands for cache at global level (cache in L2 and below, not L1),
        and ".cv" means don’t cache and fetch again. see
        `cache operator <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_ for more details.
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional
    :param volatile: changes volatile option in NVIDIA PTX
    :type volatile: bool, optional
    """
    # `mask` and `other` can be constexpr
    mask = _unwrap_if_constexpr(mask)
    other = _unwrap_if_constexpr(other)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    if other is not None:
        other = _semantic.to_tensor(other)
    padding_option = _unwrap_if_constexpr(padding_option)
    cache_modifier = _unwrap_if_constexpr(cache_modifier)
    eviction_policy = _unwrap_if_constexpr(eviction_policy)
    volatile = _unwrap_if_constexpr(volatile)
    return _semantic.load(pointer, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy,
                          volatile)


@builtin
def load_tensor_descriptor(desc: tensor_descriptor_base, offsets: Sequence[constexpr | tensor],
                           _semantic=None) -> tensor:
    """Load a block of data from a tensor descriptor."""
    return desc.load(offsets, _semantic=_semantic)


@builtin
def store_tensor_descriptor(desc: tensor_descriptor_base, offsets: Sequence[constexpr | tensor], value: tensor,
                            _semantic=None) -> tensor:
    """Store a block of data to a tensor descriptor."""
    return desc.store(offsets, value, _semantic=_semantic)


@_tensor_member_fn
@builtin
def store(pointer, value, mask=None, boundary_check=(), cache_modifier="", eviction_policy="", _semantic=None):
    """
    Store a tensor of data into memory locations defined by `pointer`.

        (1) If `pointer` is a single element pointer, a scalar is stored.  In
            this case:

            - `mask` must also be scalar, and
            - `boundary_check` and `padding_option` must be empty.

        (2) If `pointer` is an N-dimensional tensor of pointers, an
            N-dimensional block is stored.  In this case:

            - `mask` is implicitly broadcast to `pointer.shape`, and
            - `boundary_check` must be empty.

        (3) If `pointer` is a block pointer defined by `make_block_ptr`, a block
            of data is stored.  In this case:

            - `mask` must be None, and
            - `boundary_check` can be specified to control the behavior of out-of-bound access.

    `value` is implicitly broadcast to `pointer.shape` and typecast to `pointer.dtype.element_ty`.

    :param pointer: The memory location where the elements of `value` are stored
    :type pointer: `triton.PointerType`, or block of `dtype=triton.PointerType`
    :param value: The tensor of elements to be stored
    :type value: Block
    :param mask: If `mask[idx]` is false, do not store `value[idx]` at `pointer[idx]`
    :type mask: Block of triton.int1, optional
    :param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check
    :type boundary_check: tuple of ints, optional
    :param cache_modifier: changes cache option in NVIDIA PTX
    :type cache_modifier: str, optional, should be one of {"", ".wb", ".cg", ".cs", ".wt"}, where ".wb" stands for
        cache write-back all coherent levels, ".cg" stands for cache global, ".cs" stands for cache streaming, ".wt"
        stands for cache write-through, see `cache operator <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_ for more details.
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional, should be one of {"", "evict_first", "evict_last"}
    """
    # `value` can be constexpr
    value = _semantic.to_tensor(value)
    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    cache_modifier = _unwrap_if_constexpr(cache_modifier)
    eviction_policy = _unwrap_if_constexpr(eviction_policy)
    return _semantic.store(pointer, value, mask, boundary_check, cache_modifier, eviction_policy)


@builtin
def make_block_ptr(base: tensor, shape, strides, offsets, block_shape, order, _semantic=None):
    """
    Returns a pointer to a block in a parent tensor

    :param base: The base pointer to the parent tensor
    :param shape: The shape of the parent tensor
    :param strides: The strides of the parent tensor
    :param offsets: The offsets to the block
    :param block_shape: The shape of the block
    :param order: The order of the original data format
    """
    return _semantic.make_block_ptr(base, shape, strides, offsets, block_shape, order)


@must_use_result(
    "Note that tl.advance does not have any side effects. To move the block pointer, you need to assign the result of tl.advance to a variable."
)
@_tensor_member_fn
@builtin
def advance(base, offsets, _semantic=None):
    """
    Advance a block pointer

    :param base: the block pointer to advance
    :param offsets: the offsets to advance, a tuple by dimension
    """
    return _semantic.advance(base, offsets)


@builtin
def make_tensor_descriptor(
    base: tensor,
    shape: List[tensor],
    strides: List[tensor],
    block_shape: List[constexpr],
    _semantic=None,
) -> tensor_descriptor:
    """Make a tensor descriptor object

    :param base: the base pointer of the tensor, must be 16-byte aligned
    :param shape: A list of non-negative integers representing the tensor shape
    :param strides: A list of tensor strides. Leading dimensions must be multiples
        of 16-byte strides and the last dimension must be contiguous.
    :param block_shape: The shape of block to be loaded/stored from global memory

    Notes
    *****
    On NVIDIA GPUs with TMA support, this will result in a TMA descriptor object
    and loads and stores from the descriptor will be backed by the TMA hardware.

    Currently only 2-5 dimensional tensors are supported.

    Example
    *******
    .. code-block:: python

        @triton.jit
        def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
            desc = tl.make_tensor_descriptor(
                in_out_ptr,
                shape=[M, N],
                strides=[N, 1],
                block_shape=[M_BLOCK, N_BLOCK],
            )

            moffset = tl.program_id(0) * M_BLOCK
            noffset = tl.program_id(1) * N_BLOCK

            value = desc.load([moffset, noffset])
            desc.store([moffset, noffset], tl.abs(value))

        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

        M, N = 256, 256
        x = torch.randn(M, N, device="cuda")
        M_BLOCK, N_BLOCK = 32, 32
        grid = (M / M_BLOCK, N / N_BLOCK)
        inplace_abs[grid](x, M, N, M_BLOCK, N_BLOCK)

    """
    return _semantic.make_tensor_descriptor(base, shape, strides, block_shape)


# -----------------------
# Atomic Memory Operations
# -----------------------


def _add_atomic_docstr(name: str, has_cmp: bool = False) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = f"""
    Performs an atomic {name} at the memory location specified by :code:`pointer`.

    Return the data stored at :code:`pointer` before the atomic operation.

    :param pointer: The memory locations to operate on
    :type pointer: Block of dtype=triton.PointerDType"""
        if has_cmp:
            docstr += """
    :param cmp: The values expected to be found in the atomic object
    :type cmp: Block of dtype=pointer.dtype.element_ty"""
        docstr += """
    :param val: The values with which to perform the atomic operation
    :type val: Block of dtype=pointer.dtype.element_ty
    :param sem: Specifies the memory semantics for the operation. Acceptable values are "acquire",
        "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided,
        the function defaults to using "acq_rel" semantics.
    :type sem: str, optional
    :param scope: Defines the scope of threads that observe the synchronizing effect of the atomic operation.
        Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".
    :type scope: str, optional
    """
        func.__doc__ = docstr
        return func

    return _decorator


@_tensor_member_fn
@builtin
@_add_atomic_docstr("compare-and-swap", has_cmp=True)
def atomic_cas(pointer, cmp, val, sem=None, scope=None, _semantic=None):
    cmp = _semantic.to_tensor(cmp)
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    return _semantic.atomic_cas(pointer, cmp, val, sem, scope)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("exchange")
def atomic_xchg(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_xchg(pointer, val, mask, sem, scope)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("add")
def atomic_add(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_add(pointer, val, mask, sem, scope)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("max")
def atomic_max(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_max(pointer, val, mask, sem, scope)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("min")
def atomic_min(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_min(pointer, val, mask, sem, scope)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("logical and")
def atomic_and(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_and(pointer, val, mask, sem, scope)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("logical or")
def atomic_or(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_or(pointer, val, mask, sem, scope)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("logical xor")
def atomic_xor(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_xor(pointer, val, mask, sem, scope)


# -----------------------
# Conditioning
# -----------------------


@builtin
def where(condition, x, y, _semantic=None):
    """
    Returns a tensor of elements from either :code:`x` or :code:`y`, depending on :code:`condition`.

    Note that :code:`x` and :code:`y` are always evaluated regardless of the value of :code:`condition`.

    If you want to avoid unintended memory operations, use the :code:`mask` arguments in `triton.load` and `triton.store` instead.

    The shape of :code:`x` and :code:`y` are both broadcast to the shape of :code:`condition`.
    :code:`x` and :code:`y` must have the same data type.

    :param condition: When True (nonzero), yield x, otherwise yield y.
    :type condition: Block of triton.bool
    :param x: values selected at indices where condition is True.
    :param y: values selected at indices where condition is False.
    """
    condition = _semantic.to_tensor(condition)
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return _semantic.where(condition, x, y)


# -----------------------
# Math
# -----------------------


@builtin
def add(x, y, sanitize_overflow: constexpr = True, _semantic=None):
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return _semantic.add(x, y, sanitize_overflow)


@builtin
def sub(x, y, sanitize_overflow: constexpr = True, _semantic=None):
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return _semantic.sub(x, y, sanitize_overflow)


@builtin
def mul(x, y, sanitize_overflow: constexpr = True, _semantic=None):
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return _semantic.mul(x, y, sanitize_overflow)


@builtin
def minimum(x, y, propagate_nan: constexpr = PropagateNan.NONE, _semantic=None):
    """
    Computes the element-wise minimum of :code:`x` and :code:`y`.

    :param x: the first input tensor
    :type x: Block
    :param y: the second input tensor
    :type y: Block
    :param propagate_nan: whether to propagate NaN values.
    :type propagate_nan: tl.PropagateNan

    .. seealso:: :class:`tl.PropagateNan`
    """
    x = _semantic.to_tensor(x)
    y = _semantic.to_tensor(y)
    x = _promote_bfloat16_to_float32(x, _semantic=_semantic)
    y = _promote_bfloat16_to_float32(y, _semantic=_semantic)
    propagate_nan = _unwrap_if_constexpr(propagate_nan)
    return _semantic.minimum(x, y, propagate_nan)


@builtin
def maximum(x, y, propagate_nan: constexpr = PropagateNan.NONE, _semantic=None):
    """
    Computes the element-wise maximum of :code:`x` and :code:`y`.

    :param x: the first input tensor
    :type x: Block
    :param y: the second input tensor
    :type y: Block
    :param propagate_nan: whether to propagate NaN values.
    :type propagate_nan: tl.PropagateNan

    .. seealso:: :class:`tl.PropagateNan`
    """
    x = _semantic.to_tensor(x)
    y = _semantic.to_tensor(y)
    x = _promote_bfloat16_to_float32(x, _semantic=_semantic)
    y = _promote_bfloat16_to_float32(y, _semantic=_semantic)
    propagate_nan = _unwrap_if_constexpr(propagate_nan)
    return _semantic.maximum(x, y, propagate_nan)


@builtin
def clamp(x, min, max, propagate_nan: constexpr = PropagateNan.NONE, _semantic=None):
    """
    Clamps the input tensor :code:`x` within the range [min, max].
    Behavior when :code:`min` > :code:`max` is undefined.

    :param x: the input tensor
    :type x: Block
    :param min: the lower bound for clamping
    :type min: Block
    :param max: the upper bound for clamping
    :type max: Block
    :param propagate_nan: whether to propagate NaN values. Applies only to the :code:`x` tensor.
        If either :code:`min` or :code:`max` is NaN, the result is undefined.
    :type propagate_nan: tl.PropagateNan

    .. seealso:: :class:`tl.PropagateNan`
    """
    x = _semantic.to_tensor(x)
    min = _semantic.to_tensor(min)
    max = _semantic.to_tensor(max)
    x = _promote_bfloat16_to_float32(x, _semantic=_semantic)
    min = _promote_bfloat16_to_float32(min, _semantic=_semantic)
    max = _promote_bfloat16_to_float32(max, _semantic=_semantic)

    propagate_nan = _unwrap_if_constexpr(propagate_nan)

    return _semantic.clamp(x, min, max, propagate_nan)


# -----------------------
# Reductions
# -----------------------


def _add_reduction_docstr(name: str, return_indices_arg: str = None, tie_break_arg: str = None,
                          dtype_arg: str = None) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Returns the {name} of all elements in the :code:`input` tensor along the provided :code:`axis`

    :param input: the input values
    :type input: Tensor
    :param axis: the dimension along which the reduction should be done. If None, reduce all dimensions
    :type axis: int
    :param keep_dims: if true, keep the reduced dimensions with length 1
    :type keep_dims: bool"""
        if return_indices_arg is not None:
            docstr += f"""
    :param {return_indices_arg}: if true, return index corresponding to the {name} value
    :type {return_indices_arg}: bool"""
        if tie_break_arg is not None:
            docstr += f"""
    :param {tie_break_arg}: if true, in case of a tie (i.e., multiple elements have the same {name} value), return the left-most index for values that aren't NaN
    :type {tie_break_arg}: bool"""
        if dtype_arg is not None:
            docstr += f"""
    :param {dtype_arg}: the desired data type of the returned tensor. If specified, the input tensor is casted to :code:`{dtype_arg}` before the operation is performed. This is useful for preventing data overflows. If not specified, integer and bool dtypes are upcasted to :code:`tl.int32` and float dtypes are upcasted to at least :code:`tl.float32`.
    :type {dtype_arg}: tl.dtype"""

        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@contextmanager
def _insertion_guard(builder):
    ip = builder.get_insertion_point()
    yield
    builder.restore_insertion_point(ip)


@_tensor_member_fn
@builtin
def reduce(input, axis, combine_fn, keep_dims=False, _semantic=None, _generator=None):
    """Applies the combine_fn to all elements in :code:`input` tensors along the provided :code:`axis`

    :param input: the input tensor, or tuple of tensors
    :type input: Tensor
    :param axis: the dimension along which the reduction should be done. If None, reduce all dimensions
    :type axis: int | None
    :param combine_fn: a function to combine two groups of scalar tensors (must be marked with @triton.jit)
    :type combine_fn: Callable
    :param keep_dims: if true, keep the reduced dimensions with length 1
    :type keep_dims: bool

    """
    if isinstance(input, tensor):
        return reduce((input, ), axis, combine_fn, keep_dims=keep_dims, _semantic=_semantic, _generator=_generator)[0]

    def make_combine_region(reduce_op):
        param_types = [t.type.scalar for t in input] * 2
        region = reduce_op.get_region(0)
        builder = _semantic.builder
        with _insertion_guard(builder):
            to_ir = lambda T: T.to_ir(builder)
            block = builder.create_block_with_parent(region, list(map(to_ir, param_types)))
            args = [tensor(block.arg(i), ty) for i, ty in enumerate(param_types)]
            results = _generator.call_JitFunction(combine_fn, args, kwargs={})
            if isinstance(results, tensor):
                handles = [results.handle]
            else:
                handles = [r.handle for r in results]
            builder.create_reduce_ret(*handles)

    def expand_ndims(t, ndims):
        for _ in builtins.range(ndims):
            t = expand_dims(t, 0, _semantic=_semantic)
        return t

    axis = _unwrap_if_constexpr(axis)
    keep_dims = _unwrap_if_constexpr(keep_dims)
    if axis is not None:
        axis = _wrap_axis(axis, len(input[0].shape))
    ret = _semantic.reduction(input, axis, make_combine_region)
    if keep_dims:
        if axis is not None:
            ret = tuple(expand_dims(t, axis, _semantic=_semantic) for t in ret)
        else:
            ret = tuple(expand_ndims(t, len(input[0].shape)) for t in ret)
    return ret


@builtin
def _promote_bfloat16_to_float32(t, _semantic=None):
    scalar_ty = t.type.scalar

    # hardware doesn't support FMAX, FMIN, CMP for bfloat16
    if scalar_ty is bfloat16:
        return t.to(float32, _semantic=_semantic)
    return t


@builtin
def _reduce_with_indices(input, axis, combine_fn, keep_dims=False, _semantic=None, _generator=None):
    axis = _unwrap_if_constexpr(axis)
    n = input.shape[axis]
    index = arange(0, n, _semantic=_semantic)

    if len(input.shape) > 1:
        # Broadcast index across the non-reduced axes
        axes_to_expand = [constexpr(d) for d in builtins.range(len(input.shape))]
        del axes_to_expand[axis]
        index = expand_dims(index, axes_to_expand, _semantic=_semantic)
        index = broadcast_to(index, input.shape, _semantic=_semantic)

    rvalue, rindices = reduce((input, index), axis, combine_fn, keep_dims=keep_dims, _semantic=_semantic,
                              _generator=_generator)
    return rvalue, rindices


# -----------------------
# Scans
# -----------------------


def _add_scan_docstr(name: str, dtype_arg: str = None) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Returns the {name} of all elements in the :code:`input` tensor along the provided :code:`axis`

    :param input: the input values
    :type input: Tensor
    :param axis: the dimension along which the scan should be done
    :type axis: int
    :param reverse: if true, the scan is performed in the reverse direction
    :type reverse: bool"""

        if dtype_arg is not None:
            docstr += f"""
    :param {dtype_arg}: the desired data type of the returned tensor. If specified, the input tensor is casted to :code:`{dtype_arg}` before the operation is performed. If not specified, small integer types (< 32 bits) are upcasted to prevent overflow. Note that :code:`tl.bfloat16` inputs are automatically promoted to :code:`tl.float32`.
    :type {dtype_arg}: tl.dtype"""

        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@_tensor_member_fn
@builtin
def associative_scan(input, axis, combine_fn, reverse=False, _semantic=None, _generator=None):
    """Applies the combine_fn to each elements with a carry in :code:`input` tensors along the provided :code:`axis` and update the carry

    :param input: the input tensor, or tuple of tensors
    :type input: Tensor
    :param axis: the dimension along which the reduction should be done
    :type axis: int
    :param combine_fn: a function to combine two groups of scalar tensors (must be marked with @triton.jit)
    :type combine_fn: Callable
    :param reverse: whether to apply the associative scan in the reverse direction along axis
    :type reverse: bool

    """
    if isinstance(input, tensor):
        return associative_scan((input, ), axis, combine_fn, reverse, _semantic=_semantic, _generator=_generator)[0]

    def make_combine_region(scan_op):
        param_types = [t.type.scalar for t in input] * 2
        region = scan_op.get_region(0)
        builder = _semantic.builder
        with _insertion_guard(builder):
            to_ir = lambda T: T.to_ir(builder)
            block = builder.create_block_with_parent(region, list(map(to_ir, param_types)))
            args = [tensor(block.arg(i), ty) for i, ty in enumerate(param_types)]
            results = _generator.call_JitFunction(combine_fn, args, kwargs={})
            if isinstance(results, tensor):
                handles = [results.handle]
            else:
                handles = [r.handle for r in results]
            builder.create_scan_ret(*handles)

    axis = _unwrap_if_constexpr(axis)
    if axis is not None:
        axis = _wrap_axis(axis, len(input[0].shape))
    return _semantic.associative_scan(input, axis, make_combine_region, reverse)


@_tensor_member_fn
@builtin
def histogram(input, num_bins, mask=None, _semantic=None, _generator=None):
    """computes an histogram based on input tensor with num_bins bins, the bins have a width of 1 and start at 0.

    :param input: the input tensor
    :type input: Tensor
    :param num_bins: number of histogram bins
    :type num_bins: int
    :param mask: if `mask[idx]` is false, exclude `input[idx]` from histogram
    :type mask: Block of `triton.int1`, optional

    """
    num_bins = _unwrap_if_constexpr(num_bins)
    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    return _semantic.histogram(input, num_bins, mask)


@_tensor_member_fn
@builtin
def gather(src, index, axis, _semantic=None):
    """Gather from a tensor along a given dimension.

    :param src: the source tensor
    :type src: Tensor
    :param index: the index tensor
    :type index: Tensor
    :param axis: the dimension to gather along
    :type axis: int

    """
    axis = _unwrap_if_constexpr(axis)
    return _semantic.gather(src, index, axis)


# -----------------------
# Compiler Hint Ops
# -----------------------


@builtin
def debug_barrier(_semantic=None):
    '''
    Insert a barrier to synchronize all threads in a block.
    '''
    return _semantic.debug_barrier()


@builtin
def multiple_of(input, values, _semantic=None):
    """
    Let the compiler know that the values in :code:`input` are all multiples of :code:`value`.
    """
    if isinstance(values, constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]")
    values = [x.value for x in values]
    return _semantic.multiple_of(input, values)


@builtin
def max_contiguous(input, values, _semantic=None):
    """
    Let the compiler know that the `value` first values in :code:`input` are contiguous.
    """
    if isinstance(values, constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]")
    values = [x.value for x in values]
    return _semantic.max_contiguous(input, values)


@builtin
def max_constancy(input, values, _semantic=None):
    """
    Let the compiler know that the `value` first values in :code:`input` are constant.

    e.g. if :code:`values` is [4], then each group of 4 values in :code:`input` should all be equal,
    for example [0, 0, 0, 0, 1, 1, 1, 1].
    """
    if isinstance(values, constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]")
    values = [x.value for x in values]
    return _semantic.max_constancy(input, values)


@builtin
def assume(cond, _semantic=None):
    '''
    Allow compiler to assume the :code:`cond` is True.
    '''
    return _semantic.assume(_semantic.to_tensor(cond))


# -----------------------
# Debugging functions
# -----------------------


@builtin
def static_print(*values, sep: str = " ", end: str = "\n", file=None, flush=False, _semantic=None):
    '''
    Print the values at compile time.  The parameters are the same as the builtin :code:`print`.

    NOTE: Calling the Python builtin :code:`print` is not the same as calling this, it instead maps to :code:`device_print`,
    which has special requirements for the arguments.

    .. highlight:: python
    .. code-block:: python

        tl.static_print(f"BLOCK_SIZE={BLOCK_SIZE}")
    '''
    pass


@builtin
def static_assert(cond, msg="", _semantic=None):
    '''
    Assert the condition at compile time.  Does not require that the :code:`TRITON_DEBUG` environment variable
    is set.

    .. highlight:: python
    .. code-block:: python

        tl.static_assert(BLOCK_SIZE == 1024)
    '''
    pass


@builtin
def device_print(prefix, *args, hex=False, _semantic=None):
    '''
    Print the values at runtime from the device.  String formatting does not work for runtime values, so you should
    provide the values you want to print as arguments.  The first value must be a string, all following values must
    be scalars or tensors.

    Calling the Python builtin :code:`print` is the same as calling this function, and the requirements for the arguments will match
    this function (not the normal requirements for :code:`print`).

    .. highlight:: python
    .. code-block:: python

        tl.device_print("pid", pid)
        print("pid", pid)

    On CUDA, printfs are streamed through a buffer of limited size (on one host,
    we measured the default as 6912 KiB, but this may not be consistent across
    GPUs and CUDA versions).  If you notice some printfs are being dropped, you
    can increase the buffer size by calling

    .. highlight:: python
    .. code-block:: python

        triton.runtime.driver.active.utils.set_printf_fifo_size(size_bytes)

    CUDA may raise an error if you try to change this value after running a
    kernel that uses printfs.  The value set here may only affect the current
    device (so if you have multiple GPUs, you'd need to call it multiple times).

    :param prefix: a prefix to print before the values. This is required to be a string literal.
    :param args: the values to print. They can be any tensor or scalar.
    :param hex: print all values as hex instead of decimal
    '''
    import string
    prefix = _unwrap_if_constexpr(prefix)
    assert isinstance(prefix, str), f"{prefix} is not string"
    b_ascii = True
    for ch in prefix:
        if ch not in string.printable:
            b_ascii = False
            break
    assert b_ascii, f"{prefix} is not an ascii string"
    new_args = []
    for arg in args:
        new_args.append(_semantic.to_tensor(arg))
    return _semantic.device_print(prefix, new_args, hex)


@builtin
def device_assert(cond, msg="", _semantic=None):
    '''
    Assert the condition at runtime from the device.  Requires that the environment variable :code:`TRITON_DEBUG`
    is set to a value besides :code:`0` in order for this to have any effect.

    Using the Python :code:`assert` statement is the same as calling this function, except that the second argument
    must be provided and must be a string, e.g. :code:`assert pid == 0, "pid != 0"`.  The environment variable must
    be set for this :code:`assert` statement to have any effect.

    .. highlight:: python
    .. code-block:: python

        tl.device_assert(pid == 0)
        assert pid == 0, f"pid != 0"

    :param cond: the condition to assert. This is required to be a boolean tensor.
    :param msg: the message to print if the assertion fails. This is required to be a string literal.
    '''
    msg = _unwrap_if_constexpr(msg)
    return _semantic.device_assert(_semantic.to_tensor(cond), msg)


@builtin
def inline_asm_elementwise(asm: str, constraints: str, args: Sequence, dtype: Union[dtype, Sequence[dtype]],
                           is_pure: bool, pack: int, _semantic=None):
    '''
        Execute inline assembly over a tensor.  Essentially, this is :code:`map`
        where the function is inline assembly.

        The input tensors :code:`args` are implicitly broadcasted to the same shape.

        :code:`dtype` can be a tuple of types, in which case the output is a
        tuple of tensors.

        Each invocation of the inline asm processes :code:`pack` elements at a
        time.  Exactly which set of inputs a block receives is unspecified.
        Input elements of size less than 4 bytes are packed into 4-byte
        registers.

        This op does not support empty :code:`dtype` -- the inline asm must
        return at least one tensor, even if you don't need it.  You can work
        around this by returning a dummy tensor of arbitrary type; it shouldn't
        cost you anything if you don't use it.

        Example using
        `PTX <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>`_
        assembly:

        .. highlight:: python
        .. code-block:: python

            @triton.jit
            def kernel(A, B, C, D, BLOCK: tl.constexpr):
                a = tl.load(A + tl.arange(0, BLOCK)) # uint8 tensor
                b = tl.load(B + tl.arange(0, BLOCK)) # float32 tensor

                # For each (a,b) in zip(a,b), perform the following:
                # - Let ai be `a` converted to int32.
                # - Let af be `a` converted to float.
                # - Let m be the max of ai and b.
                # - Return ai and mi.
                # Do the above 4 elements at a time.
                (c, d) = tl.inline_asm_elementwise(
                    asm="""
                    {
                        // Unpack `a` into `ai`.
                        .reg .b8 tmp<4>;
                        mov.b32 {tmp0, tmp1, tmp2, tmp3}, $8;
                        cvt.u32.u8 $0, tmp0;
                        cvt.u32.u8 $1, tmp1;
                        cvt.u32.u8 $2, tmp2;
                        cvt.u32.u8 $3, tmp3;
                    }
                    // Convert `ai` to float.
                    cvt.rn.f32.s32 $4, $0;
                    cvt.rn.f32.s32 $5, $1;
                    cvt.rn.f32.s32 $6, $2;
                    cvt.rn.f32.s32 $7, $3;
                    // Take max of `ai` and `b`.
                    max.f32 $4, $4, $9;
                    max.f32 $5, $5, $10;
                    max.f32 $6, $6, $11;
                    max.f32 $7, $7, $12;
                    """,
                    constraints=(
                        # 8 output registers, namely
                        #   $0=ai0, $1=ai1, $2=ai2, $3=ai3,
                        #   $4=m0,  $5=m1,  $6=m2,  $7=m3.
                        "=r,=r,=r,=r,=r,=r,=r,=r,"
                        # 5 input registers, namely
                        #   $8=ai,
                        #   $9=b0, $10=b1, $11=b2, $12=b3.
                        # The four elements from `a` are all packed into one register.
                        "r,r,r,r,r"),
                    args=[a, b],
                    dtype=(tl.int32, tl.float32),
                    is_pure=True,
                    pack=4,
                )
                tl.store(C + tl.arange(0, BLOCK), c)
                tl.store(D + tl.arange(0, BLOCK), d)

        :param asm: assembly to run.  Must match target's assembly format.
        :param constraints: asm constraints in
            `LLVM format <https://llvm.org/docs/LangRef.html#inline-asm-constraint-string>`_
        :param args: the input tensors, whose values are passed to the asm block
        :param dtype: the element type(s) of the returned tensor(s)
        :param is_pure: if true, the compiler assumes the asm block has no side-effects
        :param pack: the number of elements to be processed by one instance of inline assembly
        :return: one tensor or a tuple of tensors of the given dtypes
    '''
    asm = _unwrap_if_constexpr(asm)
    constraints = _unwrap_if_constexpr(constraints)
    pack = _unwrap_if_constexpr(pack)
    is_pure = _unwrap_if_constexpr(is_pure)

    # Wrap `dtype` in a tuple if it's not already.
    try:
        iter(dtype)  # type: ignore
        has_multiple_outputs = True
    except TypeError:
        has_multiple_outputs = False
        dtype = (dtype, )  # type: ignore

    dtype = typing.cast(Sequence[_DtypeClass], dtype)

    res_tys = dtype
    if dispatch_args := [_semantic.to_tensor(arg) for arg in args]:
        bin_op_type_checking = partial(
            _semantic.binary_op_type_checking_impl,
            arithmetic_check=False,
            allow_lhs_ptr=True,
            allow_rhs_ptr=True,
        )
        broadcast_arg = dispatch_args[0]
        # Get the broadcast shape over all the arguments
        for item in dispatch_args:
            _, broadcast_arg = bin_op_type_checking(item, broadcast_arg)
        if broadcast_arg.shape:
            # Change the shape of each argument based on the broadcast shape
            for i, item in enumerate(dispatch_args):
                dispatch_args[i], _ = bin_op_type_checking(item, broadcast_arg)
            res_tys = [broadcast_arg.type.with_element_ty(dt) for dt in dtype]
    handles = [t.handle for t in dispatch_args]
    builder = _semantic.builder
    call = builder.create_inline_asm(asm, constraints, handles, [ty.to_ir(builder) for ty in res_tys], is_pure, pack)

    if not has_multiple_outputs:
        return tensor(call.get_result(0), res_tys[0])
    return tuple(tensor(call.get_result(i), ty) for i, ty in enumerate(res_tys))


# -----------------------
# Iterators
# -----------------------


class static_range:
    """
    Iterator that counts upward forever.

    .. highlight:: python
    .. code-block:: python

        @triton.jit
        def kernel(...):
            for i in tl.static_range(10):
                ...
    :note: This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
        :code:`triton.jit` functions. In addition, it also guides the compiler to unroll the loop aggressively.
    :param arg1: the start value.
    :param arg2: the end value.
    :param step: the step value.
    """

    def __init__(self, arg1, arg2=None, step=None):
        assert isinstance(arg1, constexpr), f"{arg1} used as tl.static_range start value is not a constexpr"
        if step is None:
            self.step = constexpr(1)
        else:
            assert isinstance(step, constexpr), f"{step} used as tl.static_range step value is not a constexpr"
            self.step = step
        if arg2 is None:
            self.start = constexpr(0)
            self.end = arg1
        else:
            assert isinstance(arg2, constexpr), f"{arg2} used as tl.static_range end value is not a constexpr"
            self.start = arg1
            self.end = arg2

    def __iter__(self):
        raise RuntimeError("static_range can only be used in @triton.jit'd functions")

    def __next__(self):
        raise RuntimeError("static_range can only be used in @triton.jit'd functions")


class range:
    """
    Iterator that counts upward forever.

    .. highlight:: python
    .. code-block:: python

        @triton.jit
        def kernel(...):
            for i in tl.range(10, num_stages=3):
                ...
    :note: This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
        :code:`triton.jit` functions. In addition, it allows user to pass extra attributes to the compiler.
    :param arg1: the start value.
    :param arg2: the end value.
    :param step: the step value.
    :param num_stages: pipeline the loop into this many stages (so there are
        :code:`num_stages` iterations of the loop in flight at once).

        Note this is subtly different than passing :code:`num_stages` as a
        kernel argument.  The kernel argument only pipelines loads that feed
        into :code:`dot` operations, while this attribute tries to pipeline most
        (though not all) loads in this loop.
    :param loop_unroll_factor: Tells the Triton IR level loop unroller how many
        times to unroll a for loop that this range is used with. Less than 2 for
        this value implies no unrolling.
    :param disallow_acc_multi_buffer: If true, prevent the accumulator of the dot
        operation in the loop to be multi-buffered, if applicable.
    :param flatten: automatically flatten the loop nest starting at this loop to
        create a single flattened loop. The compiler will try to pipeline the
        flattened loop which can avoid stage stalling.
    :param warp_specialize: Enable automatic warp specialization on the loop.
        The compiler will attempt to partition memory, MMA, and vector
        operations in the loop into separate async partitions. This will
        increase the total number of warps required by the kernel.

        Note that warp specialization is only supported on Blackwell GPUs and
        only works on simple matmul loops. Support for arbitrary loops will be
        expanded over time.
    """

    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None,
                 disallow_acc_multi_buffer=False, flatten=False, warp_specialize=False):
        if step is None:
            self.step = constexpr(1)
        else:
            self.step = step
        if arg2 is None:
            self.start = constexpr(0)
            self.end = arg1
        else:
            self.start = arg1
            self.end = arg2
        self.num_stages = num_stages
        self.loop_unroll_factor = loop_unroll_factor
        self.disallow_acc_multi_buffer = disallow_acc_multi_buffer
        self.flatten = flatten
        self.warp_specialize = warp_specialize

    def __iter__(self):
        raise RuntimeError("tl.range can only be used in @triton.jit'd functions")

    def __next__(self):
        raise RuntimeError("tl.range can only be used in @triton.jit'd functions")


# -----------------------
# Extern functions
# -----------------------


def dispatch(func, lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, ret_shape: tuple,
             is_pure: bool, _semantic):
    '''
        Dispatch a function to a library
        :param func: the function to dispatch
        :param lib_name: the name of the library
        :param lib_path: the path of the library
        :param args: the arguments of the function
        :param arg_type_symbol_dict: the type of the arguments
        :param ret_shape: the shape of the return value
        :return: the return value of the function
    '''
    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(f"length of input args does not match."
                         f"Expect {len(args)}, got {num_args}")

    arg_types = []
    arg_list = []
    for arg in args:
        if isinstance(arg, tensor):
            arg_types.append(arg.dtype)
            arg_list.append(arg.handle)
        else:
            arg_types.append(type(arg))
            arg_list.append(arg)
    arg_types = tuple(arg_types)

    if arg_types not in arg_type_symbol_dict:
        raise ValueError(f"input arg type does not match."
                         f"Expect one of {arg_type_symbol_dict.keys()}, got {arg_types}")
    else:
        symbol = arg_type_symbol_dict[arg_types][0]
        ret_type = arg_type_symbol_dict[arg_types][1]
        if ret_shape:
            ret_type = block_type(ret_type, ret_shape)
        builder = _semantic.builder
        return tensor(func(lib_name, lib_path, symbol, arg_list, ret_type.to_ir(builder), is_pure), ret_type)


@builtin
def extern_elementwise(lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, is_pure: bool,
                       _semantic=None):
    '''
        Dispatch an elementwise function to a library
        :param lib_name: the name of the library
        :param lib_path: the path of the library
        :param args: the arguments of the function
        :param arg_type_symbol_dict: the type of the arguments
        :param is_pure: whether the function is pure
        :return: the return value of the function
    '''
    dispatch_args = args.copy()
    all_scalar = True
    ret_shape = None
    arg_types = []
    for i in builtins.range(len(dispatch_args)):
        dispatch_args[i] = _semantic.to_tensor(dispatch_args[i])
        arg_types.append(dispatch_args[i].dtype)
        if dispatch_args[i].type.is_block():
            all_scalar = False
    if len(arg_types) > 0:
        arg_types = tuple(arg_types)
        arithmetic_check = True
        # If there's a type tuple that is not supported by the library, we will do arithmetic check
        if arg_types in arg_type_symbol_dict:
            arithmetic_check = False
        broadcast_arg = dispatch_args[0]
        # Get the broadcast shape over all the arguments
        for item in dispatch_args:
            _, broadcast_arg = _semantic.binary_op_type_checking_impl(item, broadcast_arg,
                                                                      arithmetic_check=arithmetic_check)
        # Change the shape of each argument based on the broadcast shape
        for i in builtins.range(len(dispatch_args)):
            dispatch_args[i], _ = _semantic.binary_op_type_checking_impl(dispatch_args[i], broadcast_arg,
                                                                         arithmetic_check=arithmetic_check)
        if not all_scalar:
            ret_shape = broadcast_arg.shape
    func = _semantic.builder.create_extern_elementwise
    return dispatch(func, lib_name, lib_path, dispatch_args, arg_type_symbol_dict, ret_shape, is_pure, _semantic)


def binary_op_type_legalization(lhs, rhs, semantic):
    '''
        Convert both operands to a single common type
        :param lhs: the left operand
        :param rhs: the right operand
        :param builder: the builder
    '''
    return semantic.binary_op_type_checking_impl(lhs, rhs)


def extern(fn):
    """A decorator for external functions."""
    return builtin(fn)
