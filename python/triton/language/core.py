from enum import Enum
from functools import wraps
from typing import List

import triton
from triton._C.libtriton.triton import ir
# from .semantic import *


# convert tensor/dtype to ir values
def _to_ir(x, builder):
    if isinstance(x, bool):
        return builder.get_int1(x)
    elif isinstance(x, int):
        if -2**31 <= x < 2**31:
            return builder.get_int32(x)
        elif 2**31 <= x < 2**32:
            return builder.get_uint32(x)
        elif -2**63 <= x < 2**63:
            return builder.get_int64(x)
        elif 2**63 <= x < 2**64:
            return builder.get_uint64(x)
        else:
            raise RuntimeError(f'Nonrepresentable integer {x}.')
    elif isinstance(x, float):
        return builder.get_float32(x)
    elif isinstance(x, constexpr):
        return _to_ir(x.value, builder)
    elif isinstance(x, tensor):
        return x.handle
    elif isinstance(x, dtype):
        return x.handle(builder)
    return x


# def _patch(fn):
#     def _from_ir(x):
#         if isinstance(x, ir.value):
#             if x.type.is_void():
#                 return None
#             return tensor(x)
#         return x

#     def wrapper(*args, **kwargs):
#         builder = args[-1]
#         assert isinstance(builder, ir.builder)
#         args = [_to_ir(x, builder) for x in args]
#         # for i, arg in enumerate(args):
#         #     if arg is None:
#         #         raise ValueError(f"Unexpected `None` at position {i} for function {fn.__name__}")
#         kwargs = {k: _to_ir(v, builder) for k, v in kwargs.items()}
#         ret = fn(*args, **kwargs)
#         if isinstance(ret, tuple):
#             return map(_from_ir, ret)
#         return _from_ir(ret)

#     return wrapper


# for name in dir(frontend):
#     fn = getattr(frontend, name)
#     if callable(fn) and "impl" not in name:
#         setattr(frontend, name, _patch(fn))


def builtin(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if '_builder' not in kwargs or \
           kwargs['_builder'] is None:
            raise ValueError("Did you forget to add @triton.jit ? (`_builder` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    return wrapper


class dtype:
    SINT_TYPES = ['int1', 'int8', 'int16', 'int32', 'int64']
    UINT_TYPES = ['uint8', 'uint16', 'uint32', 'uint64']
    FP_TYPES = ['fp8', 'fp16', 'bf16', 'fp32', 'fp64']

    class SIGNEDNESS(Enum):
        SIGNED = 0
        UNSIGNED = 1

    def __init__(self, name):
        self.name = name
        assert name in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES, name
        if name in dtype.SINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.SIGNED
            self.int_bitwidth = int(name.split('int')[-1])
        elif name in dtype.UINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.UNSIGNED
            self.int_bitwidth = int(name.split('int')[-1])
        elif name in dtype.FP_TYPES:
            if name == 'fp8':
                self.fp_mantissa_width = 3
            elif name == 'fp16':
                self.fp_mantissa_width = 10
            elif name == 'bf16':
                self.fp_mantissa_width = 7
            elif name == 'fp32':
                self.fp_mantissa_width = 23
            elif name == 'fp64':
                self.fp_mantissa_width = 53
        
    def is_fp8(self):
        return self.name == 'fp8'

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

    def is_int_signed(self):
        return self.name in dtype.SINT_TYPES

    def is_int(self):
        return self.name in dtype.SINT_TYPES + dtype.UINT_TYPES

    def is_bool(self):
        return self.is_int1()

    def is_void(self):
        raise RuntimeError("Not implemented")

    def is_block(self):
        return False

    def is_ptr(self):
        return False

    # @property
    # def name(self) -> str:
    #     # The init functions are named something like 'get_int8'. Strip the prefix.
    #     nom = self.init.__name__
    #     prefix = 'get_'
    #     assert nom.startswith(prefix)
    #     return nom[len(prefix):]

    def to_ir(self, builder: ir.builder) -> ir.type:
        if self.name == 'int1':
            return builder.get_int1_ty()
        elif self.name == 'int8' or self.name == 'uint8':
            return builder.get_int8_ty()
        elif self.name == 'int16' or self.name == 'uint16':
            return builder.get_int16_ty()
        elif self.name == 'int32' or self.name == 'uint32':
            return builder.get_int32_ty()
        elif self.name == 'int64' or self.name == 'uint64':
            return builder.get_int64_ty()
        elif self.name == 'fp16':
            return builder.get_half_ty()
        elif self.name == 'fp32':
            return builder.get_float_ty()
        elif self.name == 'fp64':
            return builder.get_double_ty()

    def __str__(self):
        return self.name

    @property
    def cache_key_part(self) -> str:
        """See cache_key_part() in triton.cc."""
        return self.name

    def __repr__(self):
        return f'triton.language.{self.name}'


class pointer_type(dtype):
    def __init__(self, element_ty: dtype, address_space: int = 1):
        if not isinstance(element_ty, dtype):
            raise TypeError('element_ty is a {type(element_ty).__name__}.')
        self.element_ty = element_ty
        self.address_space = address_space

    def to_ir(self, builder: ir.builder) -> ir.pointer_type:
        return ir.type.make_ptr(self.element_ty.to_ir(builder), 1)

    def __str__(self):
        return f'pointer<{self.element_ty}>'

    def is_ptr(self):
        return True


class block_type(dtype):
    def __init__(self, element_ty: dtype, shape: List[int]):
        self.element_ty = element_ty
        self.shape = shape

    def to_ir(self, builder: ir.builder) -> ir.block_type:
        return ir.type.make_block(self.element_ty.to_ir(builder), self.shape)

    def __str__(self):
        return f'<{self.shape}, {self.element_ty}>'

    def is_block(self):
        return True


# scalar types
int1 = dtype('int1')
int8 = dtype('int8')
int16 = dtype('int16')
int32 = dtype('int32')
int64 = dtype('int64')
uint8 = dtype('uint8')
uint16 = dtype('uint16')
uint32 = dtype('uint32')
uint64 = dtype('uint64')
float8 = dtype('fp8')
float16 = dtype('fp16')
bfloat16 = dtype('bf16')
float32 = dtype('fp32')
float64 = dtype('fp64')
# pointer types
pi32_t = pointer_type(int32)

# -----------------------
# constexpr
# -----------------------


class constexpr:
    """
    This class is used to store a value that is known at compile-time.
    """

    def __init__(self, value):
        if isinstance(value, constexpr):
            self.value = value.value
        else:
            self.value = value

    def __repr__(self) -> str:
        return f"constexpr[{self.value}]"

    def __add__(self, other):
        return self.value + other.value

    def __radd__(self, other):
        return other.value + self.value

    def __sub__(self, other):
        return self.value - other.value

    def __rsub__(self, other):
        return other.value - self.value

    def __mul__(self, other):
        return self.value * other.value

    def __rmul__(self, other):
        return other.value * self.value

    def __truediv__(self, other):
        return self.value / other.value

    def __rtruediv__(self, other):
        return other.value / self.value

    def __floordiv__(self, other):
        return self.value // other.value

    def __rfloordiv__(self, other):
        return other.value // self.value

    #

    def __gt__(self, other):
        return self.value > other.value

    def __rgt__(self, other):
        return other.value > self.value

    def __ge__(self, other):
        return self.value >= other.value

    def __rge__(self, other):
        return other.value >= self.value

    def __lt__(self, other):
        return self.value < other.value

    def __rlt__(self, other):
        return other.value < self.value

    def __le__(self, other):
        return self.value <= other.value

    def __rle__(self, other):
        return other.value <= self.value

    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return self.value != other.value

    def __bool__(self):
        return bool(self.value)

    def __call__(self, *args, **kwds):
        return self.value(*args, **kwds)


class tensor:
    @staticmethod
    def _init_dtype(ir_type):
        # primitive type
        if ir_type.is_int1(): return int1
        if ir_type.is_int8(): return int8
        if ir_type.is_int16(): return int16
        if ir_type.is_int32(): return int32
        if ir_type.is_int64(): return int64
        if ir_type.is_fp8(): return float8
        if ir_type.is_fp16(): return float16
        if ir_type.is_bf16(): return bfloat16
        if ir_type.is_fp32(): return float32
        if ir_type.is_fp64(): return float64
        # pointer type
        if ir_type.is_ptr():
            element_ty = tensor._init_dtype(ir_type.element)
            return pointer_type(element_ty)
        raise ValueError(f"Unsupported type {ir_type}")

    def __init__(self, handle, type: dtype = None):
        # IR handle
        self.handle = handle
        # Block shape
        self.shape = (1, )
        if self.handle.type.is_block():
            self.shape = self.handle.type.shape
        self.numel = 1
        for s in self.shape:
            self.numel *= s
        self.numel = constexpr(self.numel)
        # Data-type wrapper
        self.type = type
        # if type is not provided, infer from ir type
        if not self.type:
            self.type = tensor._init_dtype(self.handle.type.scalar)
        # Shape is a constexpr
        self.shape = [constexpr(s) for s in self.shape]

    def __str__(self) -> str:
        # ex. "float32[3,4]"
        return str(self.dtype) + '[' + ','.join(str(s) for s in self.shape) + ']'

    @builtin
    def __add__(self, other, _builder=None):
        return add(self, other, _builder)

    def __radd__(self, other, _builder=None):
        return self.__add__(other, _builder=_builder)

    @builtin
    def __sub__(self, other, _builder=None):
        return sub(self, other, _builder)

    def __rsub__(self, other, _builder=None):
        return sub(other, self, _builder)

    @builtin
    def __mul__(self, other, _builder=None):
        return mul(self, other, _builder)

    def __rmul__(self, other, _builder=None):
        return self.__mul__(other, _builder=_builder)

    @builtin
    def __truediv__(self, other, _builder=None):
        return truediv(self, other, _builder)

    def __rtruediv__(self, other, _builder=None):
        return truediv(other, self, _builder)

    @builtin
    def __floordiv__(self, other, _builder=None):
        return floordiv(self, other, _builder)

    @builtin
    def __mod__(self, other, _builder=None):
        return mod(self, other, _builder)

    # unary operators
    @builtin
    def __neg__(self, _builder=None):
        return minus(self, _builder)

    @builtin
    def __invert__(self, _builder=None):
        return invert(self, _builder)

    # bitwise operators

    @builtin
    def __and__(self, other, _builder=None):
        return and_(self, other, _builder)

    @builtin
    def __or__(self, other, _builder=None):
        return or_(self, other, _builder)

    @builtin
    def __xor__(self, other, _builder=None):
        return xor_(self, other, _builder)

    @builtin
    def __lshift__(self, other, _builder=None):
        return shl(self, other, _builder)

    @builtin
    def __rshift__(self, other, _builder=None):
        return lshr(self, other, _builder)

    # comparison operators

    # >
    @builtin
    def __gt__(self, other, _builder=None):
        return greater_than(self, other, _builder)

    @builtin
    def __rgt__(self, other, _builder=None):
        return greater_than(other, self, _builder)

    # >=
    @builtin
    def __ge__(self, other, _builder=None):
        return greater_equal(self, other, _builder)

    def __rge__(self, other, _builder=None):
        return greater_equal(other, self, _builder)

    # <
    @builtin
    def __lt__(self, other, _builder=None):
        return less_than(self, other, _builder)

    @builtin
    def __rlt__(self, other, _builder=None):
        return less_than(other, self, _builder)

    # <=
    @builtin
    def __le__(self, other, _builder=None):
        return less_equal(self, other, _builder)

    @builtin
    def __rle__(self, other, _builder=None):
        return less_equal(other, self, _builder)

    # ==
    @builtin
    def __eq__(self, other, _builder=None):
        return equal(self, other, _builder)

    @builtin
    def __ne__(self, other, _builder=None):
        return not_equal(self, other, _builder)

    @builtin
    def __getitem__(self, slices, _builder=None):
        if isinstance(slices, slice):
            slices = [slices]
        src_shape = self.shape
        dst_shape = []
        curr = 0
        for sl in slices:
            if isinstance(sl, constexpr) and sl.value is None:
                dst_shape.append(1)
            elif sl == slice(None, None, None):
                dst_shape.append(src_shape[curr].value)
                curr += 1
        ret = reshape(self, dst_shape, _builder)
        return ret

    @builtin
    def to(self, dtype, bitcast=False, _builder=None):
        dtype = dtype.handle(_builder)
        if bitcast:
            return bitcast(self, dtype, _builder)
        return cast_impl(self, dtype, _builder)


# -----------------------
# SPMD Programming Model
# -----------------------


@builtin
def program_id(axis, _builder=None):
    """
    Returns the id of the current program instance along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Has to be either 0, 1 or 2.
    :type axis: int
    """
    # if axis == -1:
    #     pid0 = program_id(0, _builder)
    #     pid1 = program_id(1, _builder)
    #     pid2 = program_id(2, _builder)
    #     npg0 = num_programs(0, _builder)
    #     npg1 = num_programs(0, _builder)
    #     return pid0 + pid1*npg0 + pid2*npg0*npg1
    return program_id(axis, _builder)


@builtin
def num_programs(axis, _builder=None):
    """
    Returns the number of program instances launched along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Has to be either 0, 1 or 2.
    :type axis: int
    """
    return num_programs(axis, _builder)


# -----------------------
# Block Initialization
# -----------------------


@builtin
def arange(start, end, _builder=None):
    """
    Returns contiguous values within the open interval [:code:`start`, :code:`end`).

    :param start: Start of the interval. Must be a power of two.
    :type start: int
    :param stop: End of the interval. Must be a power of two >= start.
    :type stop: int
    """
    return arange(start, end, _builder)


@builtin
def zeros(shape, dtype, _builder=None):
    """
    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :type shape: tuple of ints
    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`
    :type dtype: DType
    """
    for i, d in enumerate(shape):
        if not isinstance(d, constexpr):
            raise TypeError(f"Shape element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]")
    shape = [x.value for x in shape]
    return zeros(shape, dtype, _builder)


# -----------------------
# Shape Manipulation
# -----------------------


@builtin
def broadcast(input, other, _builder=None):
    """
    Tries to broadcast the two given blocks to a common compatible shape.

    :param input: The first input tensor.
    :type input: Block
    :param other: The second input tensor.
    :type other: Block
    """
    return broadcast_impl_value(input, other, _builder)


@builtin
def broadcast_to(input, shape, _builder=None):
    """
    Tries to broadcast the given tensor to a new :code:`shape`.

    :param input: The input tensor.
    :type input: Block
    :param shape: The desired shape.
    :type shape: Tuple[int]
    """
    return broadcast_to(input, shape, _builder)


@builtin
def cat(input, other, _builder=None):
    """
    Concatenate the given blocks

    :param input: The first input tensor.
    :type input:
    :param other: The second input tensor.
    :type other:
    """
    return cat(input, other, _builder)


@builtin
def reshape(input, shape, _builder=None):
    """
    Tries to reshape the given tensor to a new shape.

    :param input: The input tensor.
    :type input:
    :param shape: The desired shape.
    :type shape: Tuple[int]

    """
    shape = [x.value for x in shape]
    return reshape(input, shape, _builder)


# -----------------------
# Linear Algebra
# -----------------------


@builtin
def dot(input, other, allow_tf32=True, _builder=None):
    """
    Returns the matrix product of two blocks.

    The two blocks must be two dimensionals and have compatible inner dimensions.

    :param input: The first tensor to be multiplied.
    :type input: 2D tensor of scalar-type in {:code:`float16`, :code:`bfloat16`, :code:`float32`}
    :param other: The second tensor to be multiplied.
    :type other: 2D tensor of scalar-type in {:code:`float16`, :code:`bfloat16`, :code:`float32`}
    """
    return dot(input, other, allow_tf32, _builder)


# -----------------------
# Non-Atomic Memory Operations
# -----------------------


@builtin
def load(pointer, mask=None, other=None, cache_modifier="", eviction_policy="", volatile=False, _builder=None):
    """
    Return a tensor of data whose values are, elementwise, loaded from memory at location defined by :code:`pointer`.

    :code:`mask` and :code:`other` are implicitly broadcast to :code:`pointer.shape`.

    :code:`other` is implicitly typecast to :code:`pointer.dtype.element_ty`.

    :param pointer: Pointers to the data to be loaded.
    :type pointer: Block of dtype=triton.PointerDType
    :param mask: if mask[idx] is false, do not load the data at address :code:`pointer[idx]`.
    :type mask: Block of triton.int1, optional
    :param other: if mask[idx] is false, return other[idx]
    :type other: Block, optional
    :param cache_modifier: changes cache option in nvidia ptx
    'type cache_modifier: str, optional
    """
    return load(pointer, mask, other, cache_modifier, eviction_policy, volatile, _builder)


@builtin
def store(pointer, value, mask=None, _builder=None):
    """
    Stores :code:`value` tensor of elements in memory, element-wise, at the memory locations specified by :code:`pointer`.

    :code:`value` is implicitly broadcast to :code:`pointer.shape` and typecast to :code:`pointer.dtype.element_ty`.

    :param pointer: The memory locations where the elements of :code:`value` are stored.
    :type pointer: Block of dtype=triton.PointerDType
    :param value: The tensor of elements to be stored.
    :type value: Block
    :param mask: If mask[idx] is false, do not store :code:`value[idx]` at :code:`pointer[idx]`.
    :type mask: Block of triton.int1, optional
    """
    return store(pointer, value, mask, _builder)


# -----------------------
# Atomic Memory Operations
# -----------------------

def _add_atomic_docstr(name):

    def _decorator(func):
        docstr = """
    Performs an atomic {name} at the memory location specified by :code:`pointer`.

    Return the data stored at :code:`pointer` before the atomic operation.

    :param pointer: The memory locations to compare-and-swap.
    :type pointer: Block of dtype=triton.PointerDType
    :param cmp: The values expected to be found in the atomic object
    :type cmp: Block of dtype=`pointer.dtype.element_ty`
    :param val: The values to copy in case the expected value matches the contained value.
    :type val: Block of dtype=`pointer.dtype.element_ty`
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@builtin
@_add_atomic_docstr("compare-and-swap")
def atomic_cas(pointer, cmp, val, _builder=None):
    return atomic_cas(pointer, cmp, val, _builder)


@builtin
@_add_atomic_docstr("exchange")
def atomic_xchg(pointer, val, mask=None, _builder=None):
    return atomic_xchg(pointer, val, mask, _builder)


@builtin
@_add_atomic_docstr("add")
def atomic_add(pointer, val, mask=None, _builder=None):
    return atomic_add(pointer, val, mask, _builder)


@builtin
@_add_atomic_docstr("max")
def atomic_max(pointer, val, mask=None, _builder=None):
    return atomic_max(pointer, val, mask, _builder)


@builtin
@_add_atomic_docstr("min")
def atomic_min(pointer, val, mask=None, _builder=None):
    return atomic_min(pointer, val, mask, _builder)


@builtin
@_add_atomic_docstr("logical and")
def atomic_and(pointer, val, mask=None, _builder=None):
    return atomic_and(pointer, val, mask, _builder)


@builtin
@_add_atomic_docstr("logical or")
def atomic_or(pointer, val, mask=None, _builder=None):
    return atomic_or(pointer, val, mask, _builder)


@builtin
@_add_atomic_docstr("logical xor")
def atomic_xor(pointer, val, mask=None, _builder=None):
    return atomic_xor(pointer, val, mask, _builder)


# -----------------------
# Conditioning
# -----------------------


@builtin
def where(condition, x, y, _builder=None):
    """
    Returns a tensor of elements from either :code:`x` or :code:`y`, depending on :code:`condition`.

    Note that :code:`x` and :code:`y` are always evaluated regardless of the value of :code:`condition`.

    If you want to avoid unintented memory operations, use the :code:`mask` arguments in `triton.load` and `triton.store` instead.

    The shape of :code:`x` and :code:`y` are both broadcast to the shape of :code:`condition`.
    :code:`x` and :code:`y` must have the data type.

    :param condition: When True (nonzero), yield x, otherwise yield y.
    :type condition: Block of triton.bool
    :param x: values selected at indices where condition is True.
    :param y: values selected at indices where condition is False.
    """
    return where(condition, x, y, _builder)


# -----------------------
# Math
# -----------------------

@builtin
def umulhi(x, y, _builder=None):
    return umulhi(x, y, _builder)


@builtin
def fdiv(x, y, ieee_rounding=False, _builder=None):
    return fdiv(x, y, ieee_rounding, _builder)


def _add_math_1arg_docstr(name):

    def _decorator(func):
        docstr = """
    Computes the element-wise {name} of :code:`x`

    :param x: the input values
    :type x: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@builtin
@_add_math_1arg_docstr("exponential")
def exp(x, _builder=None):
    return exp(x, _builder)


@builtin
@_add_math_1arg_docstr("natural logarithm")
def log(x, _builder=None):
    return log(x, _builder)


@builtin
@_add_math_1arg_docstr("cosine")
def cos(x, _builder=None):
    return cos(x, _builder)


@builtin
@_add_math_1arg_docstr("sine")
def sin(x, _builder=None):
    return sin(x, _builder)


@builtin
@_add_math_1arg_docstr("square root")
def sqrt(x, _builder=None):
    return sqrt(x, _builder)


# -----------------------
# Reductions
# -----------------------

def _add_reduction_docstr(name):

    def _decorator(func):
        docstr = """
    Returns the {name} of all elements in the :code:`input` tensor along the provided :code:`axis`

    :param input: the input values
    :param axis: the dimension along which the reduction should be done
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@builtin
@_add_reduction_docstr("maximum")
def max(input, axis, _builder=None):
    return max(input, axis, _builder)


@builtin
@_add_reduction_docstr("minimum")
def min(input, axis, _builder=None):
    return min(input, axis, _builder)


@builtin
@_add_reduction_docstr("sum")
def sum(input, axis, _builder=None):
    return sum(input, axis, _builder)


@builtin
@_add_reduction_docstr("xor sum")
def xor_sum(input, axis, _builder=None):
    return xor_sum(input, axis, _builder)


# -----------------------
# Internal for debugging
# -----------------------


@builtin
def debug_barrier(_builder=None):
    return debug_barrier(_builder)


@builtin
def multiple_of(input, value, _builder=None):
    """
    Let the compiler knows that the values in :code:`input` are all multiples of :code:`value`.
    """
    return multiple_of(input, value, _builder)


@builtin
def max_contiguous(input, value, _builder=None):
    """
    Let the compiler knows that the `value` first values in :code:`input` are contiguous.
    """
    return max_contiguous(input, value, _builder)


# -----------------------
# Standard library
# -----------------------

@triton.jit
def abs(x):
    return where(x >= 0, x, -x)


@triton.jit
def cdiv(x, div):
    """
    Computes the ceiling division of :code:`x` by :code:`div`

    :param x: the input number
    :type input: Block
    :param div: the divisor
    :param div: Block
    """
    return (x + div - 1) // div


@triton.jit
def minimum(x, y):
    """
    Computes the element-wise minimum of :code:`x` and :code:`y`.

    :param input: the first input tensor
    :type input: Block
    :param other: the second input tensor
    :type other: Block
    """
    return triton.language.where(x < y, x, y)


@triton.jit
def maximum(x, y):
    """
    Computes the element-wise maximum of :code:`x` and :code:`y`.

    :param input: the first input tensor
    :type input: Block
    :param other: the second input tensor
    :type other: Block
    """
    return triton.language.where(x > y, x, y)


@triton.jit
@_add_math_1arg_docstr("sigmoid")
def sigmoid(x):
    return 1 / (1 + triton.language.exp(-x))


@triton.jit
@_add_math_1arg_docstr("softmax")
def softmax(x, ieee_rounding=False):
    z = x - triton.language.max(x, 0)
    num = triton.language.exp(z)
    den = triton.language.sum(num, 0)
    return fdiv(num, den, ieee_rounding)


@triton.jit
def ravel(x):
    """
    Returns a contiguous flattened view of :code:`x`

    :param x: the input tensor
    :type x: Block
    """
    return triton.language.reshape(x, [x.numel])


@triton.jit
def swizzle2d(i, j, size_i, size_j, size_g):
    """
    transformes indices of a row-major size_i*size_j matrix into those
    of one where indices are row major for each group of size_j rows.
    For example, for size_i = size_j = 4 and size_g = 2, it will transform
    [[0 , 1 , 2 , 3 ],
     [4 , 5 , 6 , 7 ],
     [8 , 9 , 10, 11],
     [12, 13, 14, 15]]
    into
    [[0, 2,  4 , 6 ],
     [1, 3,  5 , 7 ],
     [8, 10, 12, 14],
     [9, 11, 13, 15]]
    """
    # "unrolled index in array"
    ij = i * size_j + j
    # number of elements in `size_g` groups
    # of `size_j` columns
    size_gj = size_g * size_j
    # index of the group in which (i,j) is
    group_id = ij // size_gj
    # row-index of the first element of this group
    off_i = group_id * size_g
    # last group may have fewer rows
    size_g = minimum(size_i - off_i, size_g)
    # new row and column indices
    new_i = off_i + (ij % size_g)
    new_j = (ij % size_gj) // size_g
    return new_i, new_j


@triton.jit
def zeros_like(input):
    return zeros(input.shape, input.dtype)


## Create custom exception that prints message "hello"
class IncompatibleTypeErrorimpl(Exception):
  def __init__(self, type_a, type_b):
    self.type_a = type_a
    self.type_b = type_b
    self.message = "invalid operands of type " + self.type_a.repr() + " and " + self.type_b.repr()
    super(IncompatibleTypeErrorimpl, self).__init__(self.message)


##===----------------------------------------------------------------------===##
##                              Programming Model
##===----------------------------------------------------------------------===##

def program_id(axis, builder):
  return tensor(builder.create_get_program_id(axis), int32)

def num_programs(axis, builder):
  return tensor(builder.create_get_num_programs(axis), int32)

#===----------------------------------------------------------------------===//
#                               Implicit Casting Utilities
#===----------------------------------------------------------------------===//

def integer_promote_impl(a_ty: dtype, b_ty: dtype) -> dtype:
  a_rank = a_ty.int_bitwidth
  b_rank = b_ty.int_bitwidth
  a_sn = a_ty.int_signedness
  b_sn = b_ty.int_signedness
  # Rules for signedness taken from "Usual arithmetic conversions" on
  # https://en.cppreference.com/w/c/language/conversion.
  if a_sn == b_sn:
    return a_ty if a_rank > b_rank else b_ty
  elif a_sn == dtype.SIGNEDNESS.UNSIGNED:
    return a_ty if a_rank >= b_rank else b_ty
  elif b_sn == dtype.SIGNEDNESS.UNSIGNED:
    return b_ty if b_rank >= a_rank else a_ty
  else:
    assert False
  

def computation_type_impl(a_ty: dtype, b_ty: dtype, div_or_mod: bool) -> dtype:
  # 1) if one operand is double, the other is implicitly
  #    converted to double
  if a_ty.is_fp64() or b_ty.is_fp64():
    return float64
  # 2) if one operand is float, the other is implicitly
  #    converted to float
  if a_ty.is_fp32() or b_ty.is_fp32():
    return float32
  # 3 ) if one operand is half, the other is implicitly converted to half
  #     unless we're doing / or %, which do not exist natively in PTX for fp16.
  if a_ty.is_fp16() or b_ty.is_fp16():
    if div_or_mod:
      return float32
    else:
      return float16
  if not a_ty.is_int() or not b_ty.is_int():
    assert False
  # 4 ) both operands are integer and undergo
  #    integer promotion
  if div_or_mod and a_ty.int_signedness != b_ty.int_signedness:
    raise ValueError("Cannot use /, #, or % with " + a_ty.repr() + " and " + b_ty.repr() + " because they have different signedness;" 
                        "this is unlikely to result in a useful answer. Cast them to the same signedness.")
  return integer_promote_impl(a_ty, b_ty)

#===----------------------------------------------------------------------===//
#                               Binary Operators
#===----------------------------------------------------------------------===//

def check_ptr_type_impl(type_a: dtype, type_b: dtype, allow_ptr_a: bool) -> None:
  if type_a.is_ptr():
    if not allow_ptr_a:
      raise IncompatibleTypeErrorimpl(type_a, type_b)
    # T* + U* with T != U
    if type_b.is_ptr() and (type_a != type_b):
      raise IncompatibleTypeErrorimpl(type_a, type_b)
    # T* + float
    if type_b.is_floating():
      raise IncompatibleTypeErrorimpl(type_a, type_b)

def binary_op_type_checking_impl(lhs: tensor,
                                 rhs: tensor,
                                 builder: ir.builder,
                                 allow_lhs_ptr = False, allow_rhs_ptr = False,
                                 arithmetic_check = True, div_or_mod = False
                                ) -> Tuple[tensor, tensor]:
  # implicit broadcasting
  lhs, rhs = broadcast_impl_value(lhs, rhs, builder)
  # implicit typecasting
  lhs_sca_ty = lhs.type.scalar
  rhs_sca_ty = rhs.type.scalar
  check_ptr_type_impl(lhs_sca_ty, rhs_sca_ty, allow_lhs_ptr)
  check_ptr_type_impl(rhs_sca_ty, lhs_sca_ty, allow_rhs_ptr)
  if arithmetic_check and not lhs_sca_ty.is_ptr() and not rhs_sca_ty.is_ptr():
    ret_sca_ty = computation_type_impl(lhs_sca_ty, rhs_sca_ty, div_or_mod)
    lhs = cast(lhs, ret_sca_ty, builder)
    rhs = cast(rhs, ret_sca_ty, builder)
  return lhs, rhs
  

def add(input: tensor, 
        other: tensor, 
        builder: ir.builder) -> tensor:
  input, other = binary_op_type_checking_impl(input, other, builder, True, True)
  input_scalar_ty = input.type.scalar
  other_scalar_ty = other.type.scalar
  # offset + ptr
  # ptr + offset
  if other_scalar_ty.is_ptr() and not input_scalar_ty.is_ptr():
    input, other = other, input
  if input_scalar_ty.is_ptr():
    return tensor(builder.create_gep(input, [other]), input.type)
  # float + float
  elif input_scalar_ty.is_floating():
    return tensor(builder.create_fadd(input, other), input.type)
  # int + int
  elif input_scalar_ty.is_int():
    return tensor(builder.create_add(input, other), input.type)
  assert False

def sub(input: tensor,
        other: tensor,
        builder: ir.builder) -> tensor:
  input, other = binary_op_type_checking_impl(input, other, builder, True, False)
  scalar_ty = input.type.scalar
  # ptr - offset
  if scalar_ty.is_ptr():
    return tensor(builder.create_gep(input.handle, [minus(other, builder).handle]),
                      input.type)
  # float - float
  if scalar_ty.is_floating():
    return tensor(builder.create_fsub(input.handle, other.handle), input.type)
  # int - int
  elif scalar_ty.is_int():
    return tensor(builder.create_sub(input.handle, other.handle), input.type)
  assert False

def mul(input: tensor,
        other: tensor,
        builder: ir.builder) -> tensor:
  input, other = binary_op_type_checking_impl(input, other, builder)
  scalar_ty = input.type.scalar
  # float * float
  if scalar_ty.is_floating():
    return tensor(builder.create_fmul(input.handle, other.handle), input.type)
  # * int
  elif scalar_ty.is_int():
    return tensor(builder.create_mul(input.handle, other.handle), input.type)
  assert False

def truediv(input: tensor,
            other: tensor,
            builder: ir.builder) -> tensor:
  input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
  input_scalar_ty = input.type.scalar
  other_scalar_ty = other.type.scalar
  # float / int
  if input_scalar_ty.is_floating() and other_scalar_ty.is_int():
    other = cast(other, input_scalar_ty, builder)
  # int / float
  elif input_scalar_ty.is_int() and other_scalar_ty.is_floating():
    input = cast(input, other_scalar_ty, builder)
  # int / int (cast to float32)
  elif input_scalar_ty.is_int() and other_scalar_ty.is_int():
    input = cast(input, float32, builder)
    other = cast(other, float32, builder)
  # float / float (cast to highest exponent type)
  elif input_scalar_ty.is_floating() and other_scalar_ty.is_floating():
    if input_scalar_ty.get_fp_mantissa_width() > other_scalar_ty.get_fp_mantissa_width():
      other = cast(other, input_scalar_ty, builder)
    else:
      input = cast(input, other_scalar_ty, builder)
  # unreachable
  else:
    assert False
  return tensor(builder.create_fdiv(input.handle, other.handle), input.type)

def floordiv(input: tensor,
            other: tensor,
            builder: ir.builder) -> tensor:
  input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
  input_scalar_ty = input.type.scalar
  other_scalar_ty = other.type.scalar
  if input_scalar_ty.is_int() and other_scalar_ty.is_int():
    ret_ty = integer_promote_impl(input_scalar_ty, other_scalar_ty)
    input = cast(input, ret_ty, builder)
    other = cast(other, ret_ty, builder)
    if ret_ty.is_int_signed():
      return tensor(builder.create_sdiv(input.handle, other.handle), input.type)
    else:
      return tensor(builder.create_udiv(input.handle, other.handle), input.type)
  assert False

def fdiv(input: tensor,
         other: tensor,
         ieee_rounding: bool,
         builder: ir.builder) -> tensor:
  input_scalar_ty = input.type.scalar
  other_scalar_ty = other.type.scalar
  if not input_scalar_ty.is_floating() or not other_scalar_ty.is_floating():
    raise ValueError("both operands of fdiv must have floating poscalar type")
  input, other = binary_op_type_checking_impl(input, other, builder, False, False, False, True)
  ret = builder.create_fdiv(input, other)
  ret.set_fdiv_ieee_rounding(ieee_rounding.value)
  return tensor(ret, input.type)

def mod(input: tensor,
        other: tensor,
        builder: ir.builder) -> tensor:
  input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
  scalar_ty = input.type.scalar
  other_scalar_ty = other.type.scalar
  # float % float
  if scalar_ty.is_floating():
    return tensor(builder.create_frem(input.handle, other.handle), input.type)
  # % int
  elif scalar_ty.is_int():
    if scalar_ty.int_signedness != other_scalar_ty.int_signedness:
      raise ValueError("Cannot mod " + scalar_ty.repr() + " by " + other_scalar_ty.repr() + \
                       " because they have different signedness;"
                       "this is unlikely to result in a useful answer. Cast them to the same signedness.")
    if scalar_ty.is_int_signed():
      return tensor(builder.create_srem(input.handle, other.handle), input.type)
    else:
      return tensor(builder.create_urem(input.handle, other.handle), input.type)
  assert False

##############
# bitwise ops
##############
def bitwise_op_type_checking_impl(input: tensor,
                                  other: tensor,
                                  builder: ir.builder) -> Tuple[tensor, tensor]:
  input, other = binary_op_type_checking_impl(input, other, builder, False, False, False)
  input_sca_ty = input.type.scalar
  other_sca_ty = other.type.scalar
  if not input_sca_ty.is_int() or not other_sca_ty.is_int():
    raise IncompatibleTypeErrorimpl(input_sca_ty, other_sca_ty)
  ret_sca_ty = integer_promote_impl(input_sca_ty, other_sca_ty)
  if ret_sca_ty != input_sca_ty:
    input = cast(input, ret_sca_ty, builder)
  if ret_sca_ty != other_sca_ty:
    other = cast(other, ret_sca_ty, builder)
  return input, other

def and_(input: tensor,
         other: tensor,
         builder: ir.builder) -> tensor:
  input, other = bitwise_op_type_checking_impl(input, other, builder)
  return tensor(builder.create_and(input.handle, other.handle), input.type)

def or_(input: tensor,
         other: tensor,
         builder: ir.builder) -> tensor:
  input, other = bitwise_op_type_checking_impl(input, other, builder)
  return tensor(builder.create_or(input.handle, other.handle), input.type)


def xor_(input: tensor,
         other: tensor,
         builder: ir.builder) -> tensor:
  input, other = bitwise_op_type_checking_impl(input, other, builder)
  return tensor(builder.create_xor(input.handle, other.handle), input.type)


def lshr(input: tensor,
         other: tensor,
         builder: ir.builder) -> tensor:
  input, other = bitwise_op_type_checking_impl(input, other, builder)
  return tensor(builder.create_lshr(input.handle, other.handle), input.type)


def shl(input: tensor,
         other: tensor,
         builder: ir.builder) -> tensor:
  input, other = bitwise_op_type_checking_impl(input, other, builder)
  return tensor(builder.create_shl(input.handle, other.handle), input.type)

#===----------------------------------------------------------------------===//
#                               Unary Operators
#===----------------------------------------------------------------------===//

def plus(input: tensor) -> tensor:
  return input

def minus(input: tensor,
          builder: ir.builder) -> tensor:
  input_sca_ty = input.type.scalar
  if input_sca_ty.is_ptr():
    raise ValueError("wrong type argument to unary minus (" + input_sca_ty.repr() + ")")
  _0 = tensor(ir.constant.get_null_value(input_sca_ty.to_ir(builder)), input_sca_ty)
  return sub(_0, input, builder)

def invert(input: tensor,
           builder: tensor) -> tensor:
  input_sca_ty = input.type.scalar
  if input_sca_ty.is_ptr() or input_sca_ty.is_floating():
    raise ValueError("wrong type argument to unary invert (" + input_sca_ty.repr() + ")")
  _1 = tensor(ir.constant.get_all_ones_value(input_sca_ty.to_ir(builder)), input_sca_ty)
  return xor_(input, _1, builder)


#===----------------------------------------------------------------------===//
#                               Comparison Operators
#===----------------------------------------------------------------------===//

def greater_than(input: tensor,
               other: tensor,
               builder: ir.builder) -> tensor:
  input, other = binary_op_type_checking_impl(input, other, builder)
  scalar_ty = input.type.scalar
  # float > float
  if scalar_ty.is_floating():
    return builder.create_fcmpOGT(input, other)
  # > int
  elif scalar_ty.is_int():
    if scalar_ty.is_int_signed():
      return builder.create_icmpSGT(input, other)
    else:
      return builder.create_icmpUGT(input, other)
  assert False

def greater_equal(input: tensor,
               other: tensor,
               builder: ir.builder) -> tensor:
  input, other = binary_op_type_checking_impl(input, other, builder)
  scalar_ty = input.type.scalar
  # float >= float
  if scalar_ty.is_floating():
    return tensor(builder.create_fcmpOGE(input, other))
  # >= int
  elif scalar_ty.is_int():
    if scalar_ty.is_int_signed():
      return tensor(builder.create_icmpSGE(input, other))
    else:
      return tensor(builder.create_icmpUGE(input, other))
  assert False

def less_than(input: tensor,
               other: tensor,
               builder: ir.builder) -> tensor:
  input, other = binary_op_type_checking_impl(input, other, builder)
  scalar_ty = input.type.scalar
  # float < float
  if scalar_ty.is_floating():
    return tensor(builder.create_fcmpOLT(input.handle, other.handle))
  # < int
  elif scalar_ty.is_int():
    if scalar_ty.is_int_signed():
      return tensor(builder.create_icmpSLT(input.handle, other.handle))
    else:
      return tensor(builder.create_icmpULT(input.handle, other.handle))
  assert False

def less_equal(input: tensor,
               other: tensor,
               builder: ir.builder) -> tensor:
  input, other = binary_op_type_checking_impl(input, other, builder)
  scalar_ty = input.type.scalar
  # float < float
  if scalar_ty.is_floating():
    return tensor(builder.create_fcmpOLE(input.handle, other.handle))
  # < int
  elif scalar_ty.is_int():
    if scalar_ty.is_int_signed():
      return tensor(builder.create_icmpSLE(input.handle, other.handle))
    else:
      return tensor(builder.create_icmpULE(input.handle, other.handle))
  assert False

def equal(input: tensor,
          other: tensor,
          builder: ir.builder) -> tensor:
  input, other = binary_op_type_checking_impl(input, other, builder)
  scalar_ty = input.type.scalar
  # float == float
  if scalar_ty.is_floating():
    return tensor(builder.create_fcmpOEQ(input.handle, other.handle))
  # == int
  elif scalar_ty.is_int():
    return tensor(builder.create_icmpEQ(input.handle, other.handle))
  assert False

def not_equal(input: tensor,
              other: tensor,
              builder: ir.builder) -> tensor:
  input, other = binary_op_type_checking_impl(input, other, builder)
  scalar_ty = input.type.scalar
  # float == float
  if scalar_ty.is_floating():
    return tensor(builder.create_fcmpUNE(input.handle, other.handle))
  # == int
  elif scalar_ty.is_int():
    return tensor(builder.create_icmpNE(input.handle, other.handle))
  assert False

#===----------------------------------------------------------------------===//
#                               Block Creation
#===----------------------------------------------------------------------===//

def arange(start: int, end: int, builder: ir.builder) -> tensor:
  return tensor(builder.get_range(start, end), int32)

def zeros(shape: List[int], dtype: dtype, builder: ir.builder) -> tensor:
  _0 = ir.constant.get_null_value(dtype.to_ir(builder))
  ret_ty = block_type(dtype, shape)
  return tensor(builder.create_splat(_0, shape), ret_ty)

#===----------------------------------------------------------------------===//
#                               Shape Manipulation
#===----------------------------------------------------------------------===//

def reshape(input: tensor,
            dst_shape: List[int],
            builder: ir.builder) -> tensor:
  numel = 1
  for s in dst_shape: 
    numel *= s
  if input.type.numel != numel:
    raise ValueError("cannot reshape block of different shape")
  ret_ty = block_type(input.type.scalar, dst_shape)
  return tensor(builder.create_reshape(input.handle, dst_shape), ret_ty)

def cat(lhs: tensor, rhs: tensor, builder: ir.builder) -> tensor:
  # TODO: check types
  return tensor(builder.create_cat(lhs.handle, rhs.handle), lhs.type)

def broadcast_impl_shape(input: tensor,
                         shape: List[int],
                         builder: ir.builder) -> tensor:
  if not input.type.is_block():
    return builder.create_splat(input, shape)
  src_shape = input.type.get_block_shapes()
  if len(src_shape) != len(shape):
    raise ValueError("Cannot broadcast")
  if shape == src_shape:
    return input
  return builder.create_broadcast(input, shape)

def broadcast_impl_value(lhs: tensor,
                         rhs: tensor,
                         builder: ir.builder) -> tensor:
  lhs_ty = lhs.type
  rhs_ty = rhs.type

  # make_shape_compatible(block, scalar)
  if lhs_ty.is_block() and not rhs_ty.is_block():
    ret_ty = lhs_ty
    rhs = tensor(builder.create_splat(rhs.handle, lhs_ty.get_block_shapes()), ret_ty)
  # make_shape_compatible(scalar, block)
  elif not lhs_ty.is_block() and rhs_ty.is_block():
    ret_ty = rhs_ty
    lhs = tensor(builder.create_splat(lhs.handle, rhs_ty.get_block_shapes()), ret_ty)
  # make_shape_compatible(block, block)
  elif lhs_ty.is_block() and rhs_ty.is_block():
    lhs_shape = lhs_ty.get_block_shapes()
    rhs_shape = rhs_ty.get_block_shapes()
    if len(lhs_shape) != len(rhs_shape):
      raise ValueError("Cannot make_shape_compatible: blocks must have the same rank")
    ret_shape = []
    for i in range(len(lhs_shape)):
      left = lhs_shape[i]
      right = rhs_shape[i]
      if left == 1:
        ret_shape.append(right)
      elif right == 1:
        ret_shape.append(left)
      elif left == right:
        ret_shape.append(left)
      else:
        raise ValueError("Cannot make_shape_compatible: incompatible dimensions at index " + str(i) +
                                 ": " + str(left) + " and " + str(right))
    if lhs_shape != ret_shape:
      ret_ty = block_type(lhs_ty.scalar, ret_shape)
      lhs = tensor(builder.create_broadcast(lhs.handle, ret_shape), ret_ty)
    if rhs_shape != ret_shape:
      ret_ty = block_type(rhs_ty.scalar, ret_shape)
      rhs = tensor(builder.create_broadcast(rhs.handle, ret_shape), ret_ty)
  # (scalar, scalar) => returns original blocks
  return lhs, rhs

#######
# cast
#######
def bitcast(input: tensor,
            dst_ty: dtype,
            builder: ir.builder) -> tensor:
  src_ty = input.type
  if src_ty.is_block():
    dst_ty = block_type(dst_ty, input.type.get_block_shapes())
  if src_ty == dst_ty:
    return input
  src_sca_ty = src_ty.scalar
  dst_sca_ty = dst_ty.scalar
  if src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
    return cast(input, dst_ty, builder)
  # Bitcast
  src_bits = src_sca_ty.primitive_bitwidth
  dst_bits = dst_sca_ty.primitive_bitwidth
  if src_bits != dst_bits:
    raise ValueError("Cannot bitcast data-type of size " + str(src_bits) +
                             "to data-type of size " + str(dst_bits))
  return tensor(builder.create_bitcast(input.handle, dst_ty.to_ir(builder)),
                    dst_ty)

def cast(input: tensor,
         dst_ty: dtype,
         builder: ir.builder) -> tensor:
  src_ty = input.type
  if src_ty.is_block():
    dst_ty = block_type(dst_ty, input.type.get_block_shapes())
  if src_ty == dst_ty:
    return input
  src_sca_ty = src_ty.scalar
  dst_sca_ty = dst_ty.scalar

  # bf16 <=> (not fp32)
  if (src_sca_ty.is_bf16() and not dst_sca_ty.is_fp32()) or \
     (dst_sca_ty.is_bf16() and not src_sca_ty.is_fp32()):
    return cast(cast(input, float32, builder), dst_sca_ty, builder)

  # FP Truncation
  truncate_fp = src_sca_ty.is_floating() and \
                dst_sca_ty.is_floating() and \
                src_sca_ty.get_fp_mantissa_width() > dst_sca_ty.get_fp_mantissa_width()
  if truncate_fp:
    return tensor(builder.create_fp_trunc(input.handle, 
                                              dst_ty.to_ir(builder)),
                      dst_ty)

  # FP Extension
  ext_fp = src_sca_ty.is_floating() and \
                dst_sca_ty.is_floating() and \
                src_sca_ty.get_fp_mantissa_width() < dst_sca_ty.get_fp_mantissa_width()
  if ext_fp:
    return tensor(builder.create_fp_ext(input.handle,
                                            dst_ty.to_ir(builder)),
                      dst_ty)

  # Int cast
  if src_sca_ty.is_int() and dst_sca_ty.is_int() and \
    (src_sca_ty.int_bitwidth != dst_sca_ty.int_bitwidth or
     src_sca_ty.int_signedness != dst_sca_ty.int_signedness):
    sign_extend = src_sca_ty.is_int_signed() and src_sca_ty != builder.get_int1_ty()
    return tensor(builder.create_int_cast(input.handle,
                                              dst_ty.to_ir(builder), sign_extend),
                      dst_ty)

  # Float to Int
  if src_sca_ty.is_floating() and dst_sca_ty.is_int():
    # TODO: is this correct?
    if dst_sca_ty.is_bool():
      return tensor(builder.create_fp_to_ui(input.handle,
                                                dst_ty.to_ir(builder)),
                        dst_ty)
    else:
      return tensor(builder.create_fp_to_si(input.handle,
                                                dst_ty.to_ir(builder)),
                        dst_ty)

  # int => float
  if src_sca_ty.is_int() and dst_sca_ty.is_floating():
    if src_sca_ty.is_bool() or not src_sca_ty.is_int_signed():
      return tensor(builder.create_ui_to_fp(input.handle,
                                                dst_ty.to_ir(builder)),
                        dst_ty)
    else:
      return tensor(builder.create_si_to_fp(input.handle,
                                                dst_ty.to_ir(builder)),
                        dst_ty)

  # ptr => int
  if src_sca_ty.is_ptr() and dst_sca_ty.is_int():
    bitwidth = dst_sca_ty.int_bitwidth
    if bitwidth == 64:
      return tensor(builder.create_cast(ir.PtrToInt, input.handle, dst_ty.to_ir(builder)),
                        dst_ty)
    if bitwidth == 1:
      return not_equal(cast(input, int64, builder),
                       tensor(builder.get_int64(0), int64),
                       builder)

  if not src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
    return tensor(builder.create_int_to_ptr(input.handle, dst_ty.to_ir(builder)), dst_ty)
  # Ptr . Ptr
  if src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
    return tensor(builder.create_bitcast(input.handle, dst_ty.to_ir(builder)), dst_ty)
  # * . Bool
  if dst_sca_ty.is_bool():
    if src_sca_ty.is_ptr():
      input = cast(input, int64, builder)
    other = builder.get_int64(0)
    if src_ty.is_bool():
      other = builder.create_splat(other, src_ty.get_block_shapes())
    return tensor(builder.create_icmpNE(input.handle, other), dst_ty)
  assert False

#===----------------------------------------------------------------------===//
#                               Memory Operators
#===----------------------------------------------------------------------===//

def load(ptr: tensor,
         mask: Optional[tensor],
         other: Optional[tensor],
         cache_modifier: str,
         eviction_policy: str,
         is_volatile: bool,
         builder: ir.builder) -> tensor:
  if not ptr.type.scalar.is_ptr():
    raise ValueError("Pointer argument of load instruction is " + ptr.type.repr())
  if ptr.type.is_block():
    if mask:
      mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
    if other:
      other = broadcast_impl_shape(other, ptr.type.get_block_shapes(), builder)
  
  if other:
    other = cast(other, ptr.type.scalar.element, builder)
  ptr_ty = ptr.type.scalar
  elt_ty = ptr_ty.element
  # treat bool* as int8*
  if elt_ty == int1:
    elt_ty = int8
    ptr_ty = pointer_type(elt_ty, ptr_ty.address_space)
    ptr = cast(ptr, ptr_ty, builder)
  
  # cache modifier
  cache = ir.CACHE_MODIFIER.NONE; # default
  if cache_modifier:
    if cache_modifier == ".ca":
      cache = ir.CACHE_MODIFIER.CA
    elif cache_modifier == ".cg":
      cache = ir.CACHE_MODIFIER.CG
    else:
      raise ValueError(f"Cache modifier {cache_modifier} not supported")
  
  # eviction policy
  eviction = ir.EVICTION_POLICY.NORMAL; #default
  if eviction_policy:
    if eviction_policy == "evict_last":
        eviction = ir.EVICTION_POLICY.EVICT_LAST
    elif eviction_policy == "evict_first":
        eviction = ir.EVICTION_POLICY.EVICT_FIRST
    else:
        raise ValueError(f"Eviction policy {eviction_policy} not supported")

  assert ptr.type.is_block()
  shape = ptr.type.get_block_shapes()
  dst_ty = block_type(elt_ty, shape)
  if not mask and not other:
    return tensor(builder.create_load(ptr.handle, cache, eviction, is_volatile),
                      dst_ty)
  if not mask:
    raise ValueError("`other` cannot be provided without `mask`")
  
  if not other:
    other_ir = ir.undef.get(elt_ty.to_ir(builder))
    if ptr.type.is_block():
      other_ir = builder.create_splat(other_ir, ptr.type.get_block_shapes())
    other = tensor(other_ir, dst_ty)
  
  return tensor(builder.create_masked_load(ptr.handle,
                                               mask.handle,
                                               other.handle,
                                               cache, eviction, is_volatile),
                    dst_ty)

def store(ptr: tensor,
          val: tensor,
          mask: Optional[tensor],
          builder: ir.builder) -> tensor:
  if not ptr.type.scalar.is_ptr():
    raise ValueError("Pointer argument of store instruction is " + ptr.type.repr())
  if ptr.type.is_block():
    val = broadcast_impl_shape(val, ptr.type.get_block_shapes(), builder)
  if mask:
    mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
  ptr_ty = ptr.type.scalar
  elt_ty = ptr_ty.element
  # treat bool* as int8*
  if elt_ty == int1:
    elt_ty = int8
    ptr_ty = pointer_type(elt_ty, ptr_ty.address_space)
    ptr = cast(ptr, ptr_ty, builder)
  
  # cast to target data-type
  val = cast(val, elt_ty, builder)
  if not mask:
    return tensor(builder.create_store(ptr.handle, val.handle))
  if not mask.type.scalar.is_bool():
    raise ValueError("Mask must have boolean scalar type")
  return tensor(builder.create_masked_store(ptr.handle, val.handle, mask.handle))

#########
# atomic
#########
def atomic_cas(ptr: tensor,
               cmp: tensor,
               val: tensor,
               builder: ir.builder) -> tensor:
  # TODO: type checking
  return tensor(builder.create_atomic_cas(ptr.handle, cmp.handle, val.handle), val.type)

def atom_red_typechecking_impl(ptr: tensor,
                               val: tensor,
                               mask: tensor,
                               builder: ir.builder) -> Tuple[tensor, tensor, tensor]:
  if not ptr.type.scalar.is_ptr():
    raise ValueError("Pointer argument of store instruction is " + ptr.type.repr())
  if ptr.type.is_block():
    if mask:
      mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
    if val:
      val = broadcast_impl_shape(val, ptr.type.get_block_shapes(), builder)
  val = cast(val, ptr.type.scalar.element, builder)
  if not mask:
    mask_ir = builder.get_int1(True)
    if ptr.type.is_block():
      mask_ir = builder.create_splat(mask_ir, ptr.type.get_block_shapes())
    mask = tensor(mask_ir)
  return ptr, val, mask
  

def atomic_max(ptr: tensor,
               val: tensor,
               mask: tensor,
               builder: ir.builder) -> tensor:
  ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
  sca_ty = val.type.scalar
  # direct call to atomic_max for integers
  if sca_ty.is_int():
    if sca_ty.is_int_signed():
      return tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, 
                                                 ptr.handle,
                                                 val.handle,
                                                 mask.handle),
                        val.type)
    else:
      return tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.UMAX,
                                                 ptr.handle,
                                                 val.handle,
                                                 mask.handle),
                        val.type)
  # for float
  # return atomic_smax(i_ptr, i_val) if val >= 0
  # return atomic_umin(i_ptr, i_val) if val < 0
  i_val = bitcast(val, int32, builder)
  i_ptr = bitcast(ptr, pointer_type(int32, 1), builder)
  pos = greater_equal(val, tensor(ir.constant_float.get(sca_ty.to_ir(builder), 0), sca_ty), builder)
  neg = less_than(val, tensor(ir.constant_float.get(sca_ty.to_ir(builder), 0), sca_ty), builder)
  pos_ret = builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, i_ptr.handle, i_val.handle, and_(mask, pos, builder).handle)
  neg_ret = builder.create_atomic_rmw(ir.ATOMIC_OP.UMIN, i_ptr.handle, i_val.handle, and_(mask, neg, builder).handle)
  return where(pos, pos_ret, neg_ret, builder)

def atomic_min(ptr: tensor,
               val: tensor,
               mask: tensor,
               builder: ir.builder) -> tensor:
  ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
  sca_ty = val.type.scalar
  # direct call to atomic_min for integers
  if sca_ty.is_int():
    if sca_ty.is_int_signed():
      return tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.MIN,
                                                  ptr.handle,
                                                  val.handle,
                                                  mask.handle),
                        val.type)
    else:
      return tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.UMIN,
                                                  ptr.handle,
                                                  val.handle,
                                                  mask.handle),
                        val.type)
  # for float
  # return atomic_smin(i_ptr, i_val) if val >= 0
  # return atomic_umax(i_ptr, i_val) if val < 0
  i_val = bitcast(val, builder.get_int32_ty(), builder)
  i_ptr = bitcast(ptr, ir.type.make_ptr(builder.get_int32_ty(), 1), builder)
  pos = greater_equal(val, ir.constant_float.get(sca_ty, 0), builder)
  neg = less_than(val, ir.constant_float.get(sca_ty, 0), builder)
  pos_ret = tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.MIN, 
                                                 i_ptr.handle,
                                                 i_val.handle,
                                                 and_(mask, pos, builder).handle),
                       i_val.type)
  neg_ret = tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.UMAX,
                                                 i_ptr.handle,
                                                 i_val.handle,
                                                 and_(mask, neg, builder).handle),
                       i_val.type)
  return where(pos, pos_ret, neg_ret, builder)

def atomic_add(ptr: tensor,
               val: tensor,
               mask: tensor,
               builder: ir.builder) -> tensor:
  ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
  sca_ty = val.type.scalar
  op = ir.ATOMIC_OP.FADD if sca_ty.is_floating() else ir.ATOMIC_OP.ADD
  return tensor(builder.create_atomic_rmw(op, ptr.handle, val.handle, mask.handle), val.type)

def atomic_and(ptr: tensor,
               val: tensor,
               mask: tensor, 
               builder: ir.builder) -> tensor:
  ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
  return tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.AND, ptr.handle, val.handle, mask.handle), val.type)

def atomic_or(ptr: tensor,
              val: tensor,
              mask: tensor, 
              builder: ir.builder) -> tensor:
  ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
  return tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.OR, ptr.handle, val.handle, mask.handle), val.type)

def atomic_xor(ptr: tensor,
               val: tensor,
               mask: tensor, 
               builder: ir.builder) -> tensor:
  ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
  return tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.XOR, ptr.handle, val.handle, mask.handle), val.type)

def atomic_xchg(ptr: tensor,
                val: tensor,
                mask: tensor, 
                builder: ir.builder) -> tensor:
  ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, builder)
  return tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.XCHG, ptr.handle, val.handle, mask.handle), val.type)

#===----------------------------------------------------------------------===//
#                               Linear Algebra
#===----------------------------------------------------------------------===//

def dot(lhs: tensor,
        rhs: tensor,
        allow_tf32: bool,
        builder: ir.builder) -> tensor:
  if lhs.type.is_int_or_tileint():
    _0 = builder.get_int32(0)
  else:
    _0 = builder.get_float32(0)
  M = lhs.type.shape[0]
  N = rhs.type.shape[1]
  _0 = builder.create_splat(_0, [M, N])
  ret_ty = block_type(lhs.type.scalar, [M, N])
  return tensor(builder.create_dot(lhs.handle, rhs.handle, _0, allow_tf32),
                   ret_ty)


#===----------------------------------------------------------------------===//
#                               Indexing
#===----------------------------------------------------------------------===//

def where(condition: tensor,
          x: tensor,
          y: tensor,
          builder: ir.builder) -> tensor:
  condition = cast(condition, int1, builder)
  if condition.type.is_block():
    x = broadcast_impl_shape(x, condition.type.get_block_shapes(), builder)
    y = broadcast_impl_shape(y, condition.type.get_block_shapes(), builder)
  
  x_ty = x.type.scalar
  y_ty = y.type.scalar
  ty = computation_type_impl(x_ty, y_ty, div_or_mod=False)
  x = cast(x, ty, builder)
  y = cast(y, ty, builder)
  return tensor(builder.create_select(condition.handle, x.handle, y.handle), ty)


#===----------------------------------------------------------------------===//
#                               Reductions
#===----------------------------------------------------------------------===//

def reduce_impl(input: tensor, axis: int, builder: ir.builder, name: str, 
                FLOAT_OP: ir.REDUCE_OP, INT_OP: ir.REDUCE_OP) -> tensor:
  scalar_ty = input.type.scalar
  # input is extended to 32-bits if necessary
  # this increases numerical accuracy and can be done pretty much for free
  # on GPUs
  if scalar_ty.is_int() and scalar_ty.int_bitwidth <= 32:
    input = cast(input, int32, builder)
  if scalar_ty.is_floating():
    return builder.create_reduce(input, FLOAT_OP, axis)
  elif scalar_ty.is_int():
    return builder.create_reduce(input, INT_OP, axis)
  assert False

def min(input: tensor, axis: int, builder: ir.builder) -> tensor:
  return reduce_impl(input, axis, builder, "min", ir.REDUCE_OP.FMIN, ir.REDUCE_OP.MIN)

def max(input: tensor, axis: int, builder: ir.builder) -> tensor:
  return reduce_impl(input, axis, builder, "max", ir.REDUCE_OP.FMAX, ir.REDUCE_OP.MAX)

def sum(input: tensor, axis: int, builder: ir.builder) -> tensor:
  return reduce_impl(input, axis, builder, "sum", ir.REDUCE_OP.FADD, ir.REDUCE_OP.ADD)

def xor_sum(input: tensor, axis: int, builder: ir.builder) -> tensor:
  scalar_ty = input.type.scalar
  if not scalar_ty.is_int():
    raise ValueError("xor_sum only supported for integers")
  return reduce_impl(input, axis, builder, "sum", ir.REDUCE_OP.XOR, ir.REDUCE_OP.XOR)


#===----------------------------------------------------------------------===//
#                               Math
#===----------------------------------------------------------------------===//

def umulhi(x: tensor,  y: tensor, builder: ir.builder) -> tensor:
  binary_op_type_checking_impl(x, y, builder)
  return tensor(builder.insert(ir.umulhi_inst.create(x.handle, y.handle)), x.type)

def exp(x: tensor, builder: ir.builder) -> tensor:
  return tensor(builder.create_exp(x.handle), x.type)

def log(x: tensor, builder: ir.builder) -> tensor:
  return tensor(builder.create_log(x.handle), x.type)

def cos(x: tensor, builder: ir.builder) -> tensor:
  return tensor(builder.create_cos(x.handle), x.type)

def sin(x: tensor, builder: ir.builder) -> tensor:
  return tensor(builder.create_sin(x.handle), x.type)

def sqrt(x: tensor, builder: ir.builder) -> tensor:
  return tensor(builder.create_sqrt(x.handle), x.type)


##

def multiple_of(x: tensor, value: int) -> tensor:
  i = x.handle
  if not i:
    assert False
  i.set_metadata(ir.metadata.multiple_of, value)
  return x

def max_contiguous(x: tensor, value: int) -> tensor:
  i = x.handle
  if not i:
    assert False
  i.set_metadata(ir.metadata.max_contiguous, value)
  return x

def debug_barrier(builder: ir.builder) -> tensor:
  return tensor(builder.create_barrier())


