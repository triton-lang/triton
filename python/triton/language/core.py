from __future__ import annotations

from enum import Enum
from functools import wraps
from typing import List

import triton
from . import semantic
from triton._C.libtriton.triton import ir


def _to_tensor(x, builder):
    if isinstance(x, bool):
        return tensor(builder.get_int1(x), int1)
    # Note: compile-time const integers are represented by unsigned values
    elif isinstance(x, int):
        if -2**31 <= x < 2**31:
            return tensor(builder.get_int32(x), int32)
        elif 2**31 <= x < 2**32:
            return tensor(builder.get_uint32(x), uint32)
        elif -2**63 <= x < 2**63:
            return tensor(builder.get_int64(x), int64)
        elif 2**63 <= x < 2**64:
            return tensor(builder.get_uint64(x), uint64)
        else:
            raise RuntimeError(f'Nonrepresentable integer {x}.')
    elif isinstance(x, float):
        return tensor(builder.get_float32(x), float32)
    elif isinstance(x, constexpr):
        if x.value is None:
            return None
        return _to_tensor(x.value, builder)
    elif isinstance(x, tensor):
        return x
    elif x is None:
        return None
    assert False, f'cannot convert {x} to tensor'


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
    OTHER_TYPES = ['void']

    class SIGNEDNESS(Enum):
        SIGNED = 0
        UNSIGNED = 1

    def __init__(self, name):
        self.name = name
        assert name in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES, name
        if name in dtype.SINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.SIGNED
            self.int_bitwidth = int(name.split('int')[-1])
            self.primitive_bitwidth = self.int_bitwidth
        elif name in dtype.UINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.UNSIGNED
            self.int_bitwidth = int(name.split('int')[-1])
            self.primitive_bitwidth = self.int_bitwidth
        elif name in dtype.FP_TYPES:
            if name == 'fp8':
                self.fp_mantissa_width = 3
                self.primitive_bitwidth = 8
            elif name == 'fp16':
                self.fp_mantissa_width = 10
                self.primitive_bitwidth = 16
            elif name == 'bf16':
                self.fp_mantissa_width = 7
                self.primitive_bitwidth = 16
            elif name == 'fp32':
                self.fp_mantissa_width = 23
                self.primitive_bitwidth = 32
            elif name == 'fp64':
                self.fp_mantissa_width = 53
                self.primitive_bitwidth = 64
        elif name == 'void':
            self.primitive_bitwidth = 0

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

    def is_int_unsigned(self):
        return self.name in dtype.UINT_TYPES

    def is_int(self):
        return self.name in dtype.SINT_TYPES + dtype.UINT_TYPES

    def is_bool(self):
        return self.is_int1()

    def is_void(self):
        return self.name == 'void'

    def is_block(self):
        return False

    def is_ptr(self):
        return False

    def __eq__(self, other: dtype):
        if not isinstance(other, dtype):
            return False
        return self.name == other.name

    def __ne__(self, other: dtype):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name,))

    @property
    def scalar(self):
        return self

    def to_ir(self, builder: ir.builder) -> ir.type:
        if self.name == 'void':
            return builder.get_void_ty()
        elif self.name == 'int1':
            return builder.get_int1_ty()
        elif self.name == 'int8' or self.name == 'uint8':
            return builder.get_int8_ty()
        elif self.name == 'int16' or self.name == 'uint16':
            return builder.get_int16_ty()
        elif self.name == 'int32' or self.name == 'uint32':
            return builder.get_int32_ty()
        elif self.name == 'int64' or self.name == 'uint64':
            return builder.get_int64_ty()
        elif self.name == 'fp8':
            return builder.get_fp8_ty()
        elif self.name == 'fp16':
            return builder.get_half_ty()
        elif self.name == 'bf16':
            return builder.get_bf16_ty()
        elif self.name == 'fp32':
            return builder.get_float_ty()
        elif self.name == 'fp64':
            return builder.get_double_ty()
        raise ValueError(f'fail to covert {self} to ir type')

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

        self.name = self.__str__()

    def to_ir(self, builder: ir.builder) -> ir.pointer_type:
        return ir.type.make_ptr(self.element_ty.to_ir(builder), 1)

    def __str__(self):
        return f'pointer<{self.element_ty}>'

    def __repr__(self):
        return self.__str__()

    def is_ptr(self):
        return True

    def __eq__(self, other: pointer_type) -> bool:
        if not isinstance(other, pointer_type):
            return False
        return self.element_ty == other.element_ty and self.address_space == other.address_space

    def __ne__(self, other: pointer_type) -> bool:
        return not self.__eq__(other)

    @property
    def scalar(self):
        return self


class block_type(dtype):
    def __init__(self, element_ty: dtype, shape: List[int]):
        self.element_ty = element_ty
        # FIXME:
        # block_type's shape is a list of int
        # while tensor's shape is a list of constexpr
        self.shape = shape
        self.numel = 1
        for i, s in enumerate(self.shape):
            if isinstance(s, constexpr):
                self.shape[i] = s.value
            self.numel *= self.shape[i]

        self.name = self.__str__()

    def to_ir(self, builder: ir.builder) -> ir.block_type:
        return ir.type.make_block(self.element_ty.to_ir(builder), self.shape)

    def __str__(self):
        return f'<{self.shape}, {self.element_ty}>'

    def __repr__(self):
        return self.__str__()

    def is_block(self):
        return True

    def get_block_shapes(self) -> List[int]:
        return self.shape

    def __eq__(self, other: block_type) -> bool:
        if not isinstance(other, block_type):
            return False
        return self.element_ty == other.element_ty and self.shape == other.shape

    def __ne__(self, other: block_type) -> bool:
        return not self.__eq__(other)

    @property
    def scalar(self):
        return self.element_ty


class function_type(dtype):
    def __init__(self, ret_type: dtype, param_types: List[dtype]) -> None:
        self.ret_type = ret_type
        self.param_types = param_types

    def __str__(self):
        return f'fn ({self.param_types}) -> {self.ret_type}'

    def to_ir(self, builder: ir.builder):
        ir_param_types = [ty.to_ir(builder) for ty in self.param_types]
        return ir.type.make_function(self.ret_type.to_ir(builder), ir_param_types)


class tuple_type(dtype):
    def __init__(self, element_types: List[dtype]) -> None:
        self.element_types = element_types

    def __str__(self):
        return f'<{self.element_types}>'

    def to_ir(self, builder: ir.builder):
        ir_element_types = [ty.to_ir(builder) for ty in self.element_types]
        return ir.struct_type.get(ir_element_types, True)


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

    def __bool__(self):
        return bool(self.value)

    def __ge__(self, other):
        other = other.value if isinstance(other, constexpr) else other
        return self.value >= other

    def __gt__(self, other):
        other = other.value if isinstance(other, constexpr) else other
        return self.value > other

    def __le__(self, other):
        other = other.value if isinstance(other, constexpr) else other
        return self.value <= other

    def __lt__(self, other):
        other = other.value if isinstance(other, constexpr) else other
        return self.value < other

    def __eq__(self, other):
        other = other.value if isinstance(other, constexpr) else other
        return self.value == other

    def __call__(self, *args, **kwds):
        return self.value(*args, **kwds)

    def to(self, dtype, bitcast=False, _builder=None):
        if dtype in [float8, float16, bfloat16]:
            raise ValueError("floating point constexpr must be float64")
        if dtype.is_int():
            ret_ty = int
        elif dtype.is_bool():
            ret_ty = bool
        elif dtype.is_floating():
            ret_ty = float
        return constexpr(ret_ty(self.value))


class tensor:
    # infer dtype from ir type
    @staticmethod
    def _to_dtype(ir_type):
        # block type
        if ir_type.is_block():
            scalar_ty = tensor._to_dtype(ir_type.scalar)
            return block_type(scalar_ty, ir_type.get_block_shapes())
        # pointer type
        if ir_type.is_ptr():
            element_ty = tensor._to_dtype(ir_type.element)
            return pointer_type(element_ty)
        # primitive type
        if ir_type.is_void(): return void
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
        raise ValueError(f"Unsupported type {ir_type.repr()}")

    def __init__(self, handle, type: dtype):
        # IR handle
        self.handle = handle
        # Block shape
        self.shape = (1, )
        if self.handle.type.is_block():
            self.shape = self.handle.type.shape
        self.numel = 1
        for s in self.shape:
            self.numel *= s
        is_pow2 = (self.numel and (not(self.numel & (self.numel - 1))))
        if not is_pow2:
            raise ValueError("Triton tensors must have a power-of-two number of elements")
        self.numel = constexpr(self.numel)
        self.type = type  # Tensor type (can be block_type)
        # Following the practice in pytorch, dtype is scalar type
        self.dtype = type.scalar
        self.shape = [constexpr(s) for s in self.shape]

    def __str__(self) -> str:
        # ex. "float32[3,4]"
        return str(self.dtype) + '[' + ','.join(str(s) for s in self.shape) + ']'

    @builtin
    def __add__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.add(self, other, _builder)

    def __radd__(self, other, _builder=None):
        return self.__add__(other, _builder=_builder)

    @builtin
    def __sub__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.sub(self, other, _builder)

    def __rsub__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.sub(other, self, _builder)

    @builtin
    def __mul__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.mul(self, other, _builder)

    def __rmul__(self, other, _builder=None):
        return self.__mul__(other, _builder=_builder)

    @builtin
    def __truediv__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.truediv(self, other, _builder)

    def __rtruediv__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.truediv(other, self, _builder)

    @builtin
    def __floordiv__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.floordiv(self, other, _builder)

    @builtin
    def __mod__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.mod(self, other, _builder)

    @builtin
    def __rmod__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.mod(other, self, _builder)

    # unary operators
    @builtin
    def __neg__(self, _builder=None):
        return semantic.minus(self, _builder)

    @builtin
    def __invert__(self, _builder=None):
        return semantic.invert(self, _builder)

    # bitwise operators

    @builtin
    def __and__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.and_(self, other, _builder)

    @builtin
    def __or__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.or_(self, other, _builder)

    @builtin
    def __xor__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.xor_(self, other, _builder)

    @builtin
    def __lshift__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.shl(self, other, _builder)

    @builtin
    def __rshift__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.lshr(self, other, _builder)

    # comparison operators

    # >
    @builtin
    def __gt__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.greater_than(self, other, _builder)

    @builtin
    def __rgt__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.greater_than(other, self, _builder)

    # >=
    @builtin
    def __ge__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.greater_equal(self, other, _builder)

    @builtin
    def __rge__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.greater_equal(other, self, _builder)

    # <
    @builtin
    def __lt__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.less_than(self, other, _builder)

    @builtin
    def __rlt__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.less_than(other, self, _builder)

    # <=
    @builtin
    def __le__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.less_equal(self, other, _builder)

    @builtin
    def __rle__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.less_equal(other, self, _builder)

    # ==
    @builtin
    def __eq__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.equal(self, other, _builder)

    @builtin
    def __ne__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.not_equal(self, other, _builder)

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
        ret = semantic.reshape(self, dst_shape, _builder)
        return ret

    @builtin
    def to(self, dtype, bitcast=False, _builder=None):
        if isinstance(bitcast, constexpr):
            bitcast = bitcast.value
        if bitcast:
            return semantic.bitcast(self, dtype, _builder)
        return semantic.cast(self, dtype, _builder)


# -----------------------
# SPMD Programming Model
# -----------------------
def _constexpr_to_value(v):
    if isinstance(v, constexpr):
        return v.value
    return v


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
    axis = _constexpr_to_value(axis)
    return semantic.program_id(axis, _builder)


@builtin
def num_programs(axis, _builder=None):
    """
    Returns the number of program instances launched along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Has to be either 0, 1 or 2.
    :type axis: int
    """
    axis = _constexpr_to_value(axis)
    return semantic.num_programs(axis, _builder)


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
    start = _constexpr_to_value(start)
    end = _constexpr_to_value(end)
    return semantic.arange(start, end, _builder)


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
    dtype = _constexpr_to_value(dtype)
    return semantic.zeros(shape, dtype, _builder)


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
    return semantic.broadcast_impl_value(input, other, _builder)


@builtin
def broadcast_to(input, shape, _builder=None):
    """
    Tries to broadcast the given tensor to a new :code:`shape`.

    :param input: The input tensor.
    :type input: Block
    :param shape: The desired shape.
    :type shape: Tuple[int]
    """
    return semantic.broadcast_impl_shape(input, shape, _builder)


@builtin
def cat(input, other, _builder=None):
    """
    Concatenate the given blocks

    :param input: The first input tensor.
    :type input:
    :param other: The second input tensor.
    :type other:
    """
    return semantic.cat(input, other, _builder)


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
    return semantic.reshape(input, shape, _builder)


# -----------------------
# Linear Algebra
# -----------------------


@builtin
def dot(input, other, trans_a=False, trans_b=False, allow_tf32=True, _builder=None):
    """
    Returns the matrix product of two blocks.

    The two blocks must be two dimensionals and have compatible inner dimensions.

    :param input: The first tensor to be multiplied.
    :type input: 2D tensor of scalar-type in {:code:`float16`, :code:`bfloat16`, :code:`float32`}
    :param other: The second tensor to be multiplied.
    :type other: 2D tensor of scalar-type in {:code:`float16`, :code:`bfloat16`, :code:`float32`}
    """
    allow_tf32 = _constexpr_to_value(allow_tf32)
    return semantic.dot(input, other, trans_a, trans_b, allow_tf32, _builder)


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
    # mask, other can be constexpr
    if mask is not None:
        mask = _to_tensor(mask, _builder)
    if other is not None:
        other = _to_tensor(other, _builder)
    cache_modifier = _constexpr_to_value(cache_modifier)
    eviction_policy = _constexpr_to_value(eviction_policy)
    volatile = _constexpr_to_value(volatile)
    return semantic.load(pointer, mask, other, cache_modifier, eviction_policy, volatile, _builder)


@builtin
def store(pointer, value, mask=None, eviction_policy="", _builder=None):
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
    # value can be constexpr
    value = _to_tensor(value, _builder)
    if mask is not None:
        mask = _to_tensor(mask, _builder)
    return semantic.store(pointer, value, mask, eviction_policy, _builder)


# -----------------------
# Atomic Memory Operations
# -----------------------

@builtin
def atomic_cas(pointer, cmp, val, _builder=None):
    """
        Performs an atomic compare-and-swap at the memory location specified by :code:`pointer`.

        Return the data stored at :code:`pointer` before the atomic operation.

        :param pointer: The memory locations to compare-and-swap.
        :type pointer: Block of dtype=triton.PointerDType
        :param cmp: The values expected to be found in the atomic object
        :type cmp: Block of dtype=`pointer.dtype.element_ty`
        :param val: The values to copy in case the expected value matches the contained value.
        :type val: Block of dtype=`pointer.dtype.element_ty`
    """
    cmp = _to_tensor(cmp, _builder)
    val = _to_tensor(val, _builder)
    return semantic.atomic_cas(pointer, cmp, val, _builder)


def _add_atomic_docstr(name):

    def _decorator(func):
        docstr = """
    Performs an atomic {name} at the memory location specified by :code:`pointer`.

    Return the data stored at :code:`pointer` before the atomic operation.

    :param pointer: The memory locations to apply {name}.
    :type pointer: Block of dtype=triton.PointerDType
    :param val: The values to {name} in the atomic object.
    :type val: Block of dtype=`pointer.dtype.element_ty`
    :param mask: If mask[idx] is false, do not apply {name}.
    :type mask: Block of triton.int1, optional
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@builtin
@_add_atomic_docstr("exchange")
def atomic_xchg(pointer, val, mask=None, _builder=None):
    val = _to_tensor(val, _builder)
    return semantic.atomic_xchg(pointer, val, mask, _builder)


@builtin
@_add_atomic_docstr("add")
def atomic_add(pointer, val, mask=None, _builder=None):
    val = _to_tensor(val, _builder)
    return semantic.atomic_add(pointer, val, mask, _builder)


@builtin
@_add_atomic_docstr("max")
def atomic_max(pointer, val, mask=None, _builder=None):
    val = _to_tensor(val, _builder)
    return semantic.atomic_max(pointer, val, mask, _builder)


@builtin
@_add_atomic_docstr("min")
def atomic_min(pointer, val, mask=None, _builder=None):
    val = _to_tensor(val, _builder)
    return semantic.atomic_min(pointer, val, mask, _builder)


@builtin
@_add_atomic_docstr("logical and")
def atomic_and(pointer, val, mask=None, _builder=None):
    val = _to_tensor(val, _builder)
    return semantic.atomic_and(pointer, val, mask, _builder)


@builtin
@_add_atomic_docstr("logical or")
def atomic_or(pointer, val, mask=None, _builder=None):
    val = _to_tensor(val, _builder)
    return semantic.atomic_or(pointer, val, mask, _builder)


@builtin
@_add_atomic_docstr("logical xor")
def atomic_xor(pointer, val, mask=None, _builder=None):
    val = _to_tensor(val, _builder)
    return semantic.atomic_xor(pointer, val, mask, _builder)


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
    condition = _to_tensor(condition, _builder)
    x = _to_tensor(x, _builder)
    y = _to_tensor(y, _builder)
    return semantic.where(condition, x, y, _builder)


# -----------------------
# Math
# -----------------------

@builtin
def umulhi(x, y, _builder=None):
    x = _to_tensor(x, _builder)
    y = _to_tensor(y, _builder)
    return semantic.umulhi(x, y, _builder)


@builtin
def fdiv(x, y, ieee_rounding=False, _builder=None):
    ieee_rounding = _constexpr_to_value(ieee_rounding)
    return semantic.fdiv(x, y, ieee_rounding, _builder)


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
    return semantic.exp(x, _builder)


@builtin
@_add_math_1arg_docstr("natural logarithm")
def log(x, _builder=None):
    return semantic.log(x, _builder)


@builtin
@_add_math_1arg_docstr("cosine")
def cos(x, _builder=None):
    return semantic.cos(x, _builder)


@builtin
@_add_math_1arg_docstr("sine")
def sin(x, _builder=None):
    return semantic.sin(x, _builder)


@builtin
@_add_math_1arg_docstr("square root")
def sqrt(x, _builder=None):
    return semantic.sqrt(x, _builder)


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
    axis = _constexpr_to_value(axis)
    return semantic.max(input, axis, _builder)


@builtin
@_add_reduction_docstr("maximum index")
def argmax(input, axis, _builder=None):
    axis = _constexpr_to_value(axis)
    return semantic.argmax(input, axis, _builder)


@builtin
@_add_reduction_docstr("minimum")
def min(input, axis, _builder=None):
    axis = _constexpr_to_value(axis)
    return semantic.min(input, axis, _builder)


@builtin
@_add_reduction_docstr("minimum index")
def argmin(input, axis, _builder=None):
    axis = _constexpr_to_value(axis)
    return semantic.argmin(input, axis, _builder)


@builtin
@_add_reduction_docstr("sum")
def sum(input, axis, _builder=None):
    axis = _constexpr_to_value(axis)
    return semantic.sum(input, axis, _builder)


@builtin
@_add_reduction_docstr("xor sum")
def xor_sum(input, axis, _builder=None):
    axis = _constexpr_to_value(axis)
    return semantic.xor_sum(input, axis, _builder)

# -----------------------
# Utilities
# -----------------------


@builtin
def globaltimer(_builder=None):
    return semantic.globaltimer(_builder)


@builtin
def clock(_builder=None):
    return semantic.clock(_builder)

# -----------------------
# Internal for debugging
# -----------------------


@builtin
def debug_barrier(_builder=None):
    return semantic.debug_barrier(_builder)


@builtin
def multiple_of(input, value, _builder=None):
    """
    Let the compiler knows that the values in :code:`input` are all multiples of :code:`value`.
    """
    value = _constexpr_to_value(value)
    return semantic.multiple_of(input, value)


@builtin
def max_contiguous(input, value, _builder=None):
    """
    Let the compiler knows that the `value` first values in :code:`input` are contiguous.
    """
    value = _constexpr_to_value(value)
    return semantic.max_contiguous(input, value)


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
def softmax(x, ieee_rounding: constexpr = False):
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
# -----------------------
# Dynamic Parallelism
# -----------------------


# class LaunchProxy:

#     def __init__(self, fn, args, constants, grid, num_warps) -> None:
#         self.args = args
#         self.grid = grid
#         self.constants = constants
#         self.num_warps = num_warps
#         self.fn = fn


# @builtin
# def launch(fn, args, grid, num_warps=None, _builder=None):
#     constants = {i: x for i, x in enumerate(args) if isinstance(x, constexpr)}
#     args = [_to_ir(x, builder=_builder) for x in args if not isinstance(x, constexpr)]
#     grid = [_to_ir(x, builder=_builder) for x in grid]
#     if num_warps is None:
#         num_warps = _to_ir(4, builder=_builder)
#     return LaunchProxy(fn, args, constants, grid, num_warps)
