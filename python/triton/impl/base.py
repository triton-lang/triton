from __future__ import annotations, division

import enum
import functools
from enum import Enum
from typing import (
    Any,
    cast,
    Tuple,
    List,
    Sequence,
    TypeVar,
    Optional,
    Union,
    Generic,
    overload,
)

import torch

from . import ir


T = TypeVar("T")

TRITON_BUILTIN = "__triton_builtin__"


def builtin(fn: T) -> T:
    """Mark a function as builtin.

    decorates with `_builder` guard.
    """
    assert callable(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if kwargs.get("_builder") is None:
            raise ValueError(
                "Did you forget to add @triton.jit ? "
                "(`_builder` argument must be provided outside of JIT functions.)"
            )
        return fn(*args, **kwargs)

    setattr(wrapper, TRITON_BUILTIN, True)

    return cast(T, wrapper)


def is_builtin(fn) -> bool:
    """Is this a registered triton builtin function?"""
    return getattr(fn, TRITON_BUILTIN, False)


class dtype:
    SINT_TYPES = ["int1", "int8", "int16", "int32", "int64"]
    UINT_TYPES = ["uint8", "uint16", "uint32", "uint64"]
    FP_TYPES = ["fp8", "fp16", "bf16", "fp32", "fp64"]
    OTHER_TYPES = ["void"]

    class SIGNEDNESS(Enum):
        SIGNED = 0
        UNSIGNED = 1

    name: str
    numel: int = 1
    shape: Tuple[int, ...] = tuple()
    int_signedness: SIGNEDNESS
    int_bitwidth: int
    primitive_bitwidth: int
    fp_mantissa_width: int = 0

    def __init__(self, name):
        self.name = name
        assert (
            name
            in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES
        ), name
        if name in dtype.SINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.SIGNED
            self.int_bitwidth = int(name.split("int")[-1])
            self.primitive_bitwidth = self.int_bitwidth
        elif name in dtype.UINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.UNSIGNED
            self.int_bitwidth = int(name.split("int")[-1])
            self.primitive_bitwidth = self.int_bitwidth
        elif name in dtype.FP_TYPES:
            if name == "fp8":
                self.fp_mantissa_width = 3
                self.primitive_bitwidth = 8
            elif name == "fp16":
                self.fp_mantissa_width = 10
                self.primitive_bitwidth = 16
            elif name == "bf16":
                self.fp_mantissa_width = 7
                self.primitive_bitwidth = 16
            elif name == "fp32":
                self.fp_mantissa_width = 23
                self.primitive_bitwidth = 32
            elif name == "fp64":
                self.fp_mantissa_width = 53
                self.primitive_bitwidth = 64
        elif name == "void":
            self.primitive_bitwidth = 0

    def is_fp8(self) -> bool:
        return self.name == "fp8"

    def is_fp16(self) -> bool:
        return self.name == "fp16"

    def is_bf16(self) -> bool:
        return self.name == "bf16"

    def is_fp32(self) -> bool:
        return self.name == "fp32"

    def is_fp64(self) -> bool:
        return self.name == "fp64"

    def is_int1(self) -> bool:
        return self.name == "int1"

    def is_int8(self) -> bool:
        return self.name == "int8"

    def is_int16(self) -> bool:
        return self.name == "int16"

    def is_int32(self) -> bool:
        return self.name == "int32"

    def is_int64(self) -> bool:
        return self.name == "int64"

    def is_uint8(self) -> bool:
        return self.name == "uint8"

    def is_uint16(self) -> bool:
        return self.name == "uint16"

    def is_uint32(self) -> bool:
        return self.name == "uint32"

    def is_uint64(self) -> bool:
        return self.name == "uint64"

    def is_floating(self) -> bool:
        return self.name in dtype.FP_TYPES

    def is_int_signed(self) -> bool:
        return self.name in dtype.SINT_TYPES

    def is_int_unsigned(self) -> bool:
        return self.name in dtype.UINT_TYPES

    def is_int(self) -> bool:
        return self.name in dtype.SINT_TYPES + dtype.UINT_TYPES

    def is_bool(self) -> bool:
        return self.is_int1()

    def is_void(self) -> bool:
        return self.name == "void"

    def is_block(self) -> bool:
        return False

    def is_ptr(self) -> bool:
        return False

    def __eq__(self, other: Any):
        if not isinstance(other, dtype):
            return False
        return self.name == other.name

    def __ne__(self, other: Any):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name,))

    @property
    def scalar(self) -> dtype:
        return self

    def to_ir(self, builder: ir.builder) -> ir.type:
        if self.name == "void":
            return builder.get_void_ty()
        elif self.name == "int1":
            return builder.get_int1_ty()
        elif self.name == "int8" or self.name == "uint8":
            return builder.get_int8_ty()
        elif self.name == "int16" or self.name == "uint16":
            return builder.get_int16_ty()
        elif self.name == "int32" or self.name == "uint32":
            return builder.get_int32_ty()
        elif self.name == "int64" or self.name == "uint64":
            return builder.get_int64_ty()
        elif self.name == "fp8":
            return builder.get_fp8_ty()
        elif self.name == "fp16":
            return builder.get_half_ty()
        elif self.name == "bf16":
            return builder.get_bf16_ty()
        elif self.name == "fp32":
            return builder.get_float_ty()
        elif self.name == "fp64":
            return builder.get_double_ty()
        raise ValueError(f"fail to convert {self} to ir type")

    def __str__(self):
        return self.name

    @property
    def cache_key_part(self) -> str:
        """See cache_key_part() in triton.cc."""
        return self.name

    def __repr__(self):
        return f"{self.name}"


class pointer_type(dtype):
    element_ty: dtype
    address_space: int

    def __init__(self, element_ty: dtype, address_space: int = 1):
        if not isinstance(element_ty, dtype):
            raise TypeError("element_ty is a {type(element_ty).__name__}.")
        self.element_ty = element_ty
        self.address_space = address_space

        self.name = self.__str__()

    def to_ir(self, builder: ir.builder) -> ir.pointer_type:
        return ir.type.make_ptr(self.element_ty.to_ir(builder), 1)

    def __str__(self):
        return f"pointer<{self.element_ty}>"

    def __repr__(self):
        return self.__str__()

    def is_ptr(self):
        return True

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, pointer_type):
            return False
        return (
            self.element_ty == other.element_ty
            and self.address_space == other.address_space
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @property
    def scalar(self) -> pointer_type:
        return self


class block_type(dtype):
    element_ty: dtype
    shape: Tuple[int, ...]
    numel: int

    def __init__(self, element_ty: dtype, shape: Sequence[int]):
        self.element_ty = element_ty
        # FIXME:
        # block_type's shape is a tuple of int
        # while tensor's shape is a list of constexpr
        self.numel = 1

        shape = list(shape)
        for i, s in enumerate(shape):
            if isinstance(s, constexpr):
                shape[i] = s.value
            self.numel *= shape[i]

        self.shape = tuple(shape)

        self.name = self.__str__()

    def to_ir(self, builder: ir.builder) -> ir.block_type:
        return ir.type.make_block(self.element_ty.to_ir(builder), self.shape)

    def __str__(self) -> str:
        return f"<{self.shape}, {self.element_ty}>"

    def __repr__(self) -> str:
        return self.__str__()

    def is_block(self) -> bool:
        return True

    def get_block_shapes(self) -> Tuple[int, ...]:
        return self.shape

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, block_type):
            return False
        return self.element_ty == other.element_ty and self.shape == other.shape

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @property
    def scalar(self) -> dtype:
        return self.element_ty


class function_type(dtype):
    def __init__(self, ret_type: dtype, param_types: List[dtype]) -> None:
        self.ret_type = ret_type
        self.param_types = param_types

    def __str__(self):
        return f"fn ({self.param_types}) -> {self.ret_type}"

    def to_ir(self, builder: ir.builder):
        ir_param_types = [ty.to_ir(builder) for ty in self.param_types]
        return ir.type.make_function(
            self.ret_type.to_ir(builder),
            ir_param_types,
        )


class tuple_type(dtype):
    def __init__(self, element_types: List[dtype]) -> None:
        self.element_types = element_types

    def __str__(self):
        return f"<{self.element_types}>"

    def to_ir(self, builder: ir.builder):
        ir_element_types = [ty.to_ir(builder) for ty in self.element_types]
        return ir.struct_type.get(
            ir_element_types,
            True,
        )


# scalar types
void = dtype("void")
int1 = dtype("int1")
int8 = dtype("int8")
int16 = dtype("int16")
int32 = dtype("int32")
int64 = dtype("int64")
uint8 = dtype("uint8")
uint16 = dtype("uint16")
uint32 = dtype("uint32")
uint64 = dtype("uint64")
float8 = dtype("fp8")
float16 = dtype("fp16")
bfloat16 = dtype("bf16")
float32 = dtype("fp32")
float64 = dtype("fp64")
# pointer types
pi32_t = pointer_type(int32)


# -----------------------
# constexpr
# -----------------------


class constexpr(Generic[T]):
    """
    This class is used to store a value that is known at compile-time.
    """

    value: T

    def __init__(self, value: T):
        self.value = cvalue(value)

    def __repr__(self) -> str:
        return f"constexpr[{self.value}]"

    def __bool__(self) -> bool:
        return bool(self.value)

    def __ge__(self, other) -> bool:
        other = other.value if isinstance(other, constexpr) else other
        return self.value >= other

    def __gt__(self, other) -> bool:
        other = other.value if isinstance(other, constexpr) else other
        return self.value > other

    def __le__(self, other) -> bool:
        other = other.value if isinstance(other, constexpr) else other
        return self.value <= other

    def __lt__(self, other) -> bool:
        other = other.value if isinstance(other, constexpr) else other
        return self.value < other

    def __eq__(self, other) -> bool:
        other = other.value if isinstance(other, constexpr) else other
        return self.value == other

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __call__(self, *args, **kwds):
        return self.value(*args, **kwds)

    def to(self, dtype, bitcast=False, _builder: ir.builder = None) -> constexpr:
        if dtype in [float8, float16, bfloat16]:
            raise ValueError("floating point constexpr must be float64")
        ret_ty: type
        if dtype.is_int():
            ret_ty = int
        elif dtype.is_bool():
            ret_ty = bool
        elif dtype.is_floating():
            ret_ty = float
        return constexpr(ret_ty(self.value))


@overload
def cvalue(v: constexpr[T]) -> T:
    ...


@overload
def cvalue(v: T) -> T:
    ...


def cvalue(v: Union[constexpr[T], T]) -> T:
    """Unpack a constexpr.

    This unrolls to a no-op in JITed code.
    """
    if isinstance(v, constexpr):
        return v.value
    return v


@overload
def _constexpr_to_value(v: constexpr[T]) -> T:
    ...


@overload
def _constexpr_to_value(v: T) -> T:
    ...


def _constexpr_to_value(v: Union[constexpr[T], T]) -> T:
    """Legacy wrapper for cvalue()."""
    return cvalue(v)


dtype_t = dtype
"""
dtype type alias.

This type alias exists because some types (such as tensor)
both have a class member annotation named `dtype`, and declare
method parameters with the `dtype` type; so we need to prevent
the class annotation from shadowing the parameter type.
"""


class tensor:
    handle: ir.pointer_type
    type: dtype_t
    dtype: dtype_t
    shape: List[constexpr[int]]
    numel: constexpr[int]

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
        if ir_type.is_void():
            return void
        if ir_type.is_int1():
            return int1
        if ir_type.is_int8():
            return int8
        if ir_type.is_int16():
            return int16
        if ir_type.is_int32():
            return int32
        if ir_type.is_int64():
            return int64
        if ir_type.is_fp8():
            return float8
        if ir_type.is_fp16():
            return float16
        if ir_type.is_bf16():
            return bfloat16
        if ir_type.is_fp32():
            return float32
        if ir_type.is_fp64():
            return float64
        raise ValueError(f"Unsupported type {ir_type.repr()}")

    def __init__(self, handle, type: dtype_t):
        # IR handle
        self.handle = handle
        # Block shape
        shape = (1,)
        if self.handle.type.is_block():
            shape = self.handle.type.shape
        numel = 1
        for s in shape:
            numel *= s
        is_pow2 = numel and (not (numel & (numel - 1)))
        if not is_pow2:
            raise ValueError(
                "Triton tensors must have a power-of-two number of elements"
            )
        self.numel = constexpr(numel)
        self.shape = [constexpr(s) for s in shape]

        self.type = type  # Tensor type (can be block_type)
        # Following the practice in pytorch, dtype is scalar type
        self.dtype = type.scalar

    def __str__(self) -> str:
        # ex. "float32[3,4]"
        return str(self.dtype) + "[" + ",".join(str(s) for s in self.shape) + "]"

    @builtin
    def __add__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _add(self, other, _builder)

    def __radd__(self, other, _builder: ir.builder = None):
        return self.__add__(other, _builder=_builder)

    @builtin
    def __sub__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _sub(self, other, _builder)

    def __rsub__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _sub(other, self, _builder)

    @builtin
    def __mul__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _mul(self, other, _builder)

    def __rmul__(self, other, _builder: ir.builder = None) -> tensor:
        return self.__mul__(other, _builder=_builder)

    @builtin
    def __truediv__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _truediv(self, other, _builder)

    def __rtruediv__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _truediv(other, self, _builder)

    @builtin
    def __floordiv__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _floordiv(self, other, _builder)

    @builtin
    def __rfloordiv__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _floordiv(other, self, _builder)

    @builtin
    def __mod__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _mod(self, other, _builder)

    @builtin
    def __rmod__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _mod(other, self, _builder)

    # unary operators
    @builtin
    def __neg__(self, _builder: ir.builder = None) -> tensor:
        return _minus(self, _builder)

    @builtin
    def __invert__(self, _builder: ir.builder = None) -> tensor:
        return _invert(self, _builder)

    # bitwise operators

    @builtin
    def __and__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _and_(self, other, _builder)

    @builtin
    def __or__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _or_(self, other, _builder)

    @builtin
    def __xor__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _xor_(self, other, _builder)

    @builtin
    def __lshift__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _shl(self, other, _builder)

    @builtin
    def __rshift__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _lshr(self, other, _builder)

    # comparison operators

    # >
    @builtin
    def __gt__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _greater_than(self, other, _builder)

    @builtin
    def __rgt__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _greater_than(other, self, _builder)

    # >=
    @builtin
    def __ge__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _greater_equal(self, other, _builder)

    @builtin
    def __rge__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _greater_equal(other, self, _builder)

    # <
    @builtin
    def __lt__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _less_than(self, other, _builder)

    @builtin
    def __rlt__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _less_than(other, self, _builder)

    # <=
    @builtin
    def __le__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _less_equal(self, other, _builder)

    @builtin
    def __rle__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _less_equal(other, self, _builder)

    # ==
    @builtin
    def __eq__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _equal(self, other, _builder)

    @builtin
    def __ne__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _not_equal(self, other, _builder)

    @builtin
    def __getitem__(self, slices, _builder: ir.builder = None) -> tensor:
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
        ret = _reshape(self, dst_shape, _builder)
        return ret

    @builtin
    def to(self, dtype, bitcast=False, _builder: ir.builder = None) -> tensor:
        if isinstance(bitcast, constexpr):
            bitcast = bitcast.value
        if bitcast:
            return _bitcast(self, dtype, _builder)
        return _cast(self, dtype, _builder)


def is_triton_tensor(value: object) -> bool:
    return isinstance(value, tensor)


class TensorWrapper:
    def __init__(self, base, dtype):
        self.dtype = dtype
        self.base = base
        self.is_cuda = base.is_cuda
        self.device = base.device

    def data_ptr(self):
        return self.base.data_ptr()

    def __str__(self) -> str:
        return f"TensorWrapper[{self.dtype}]({self.base})"


def reinterpret(
    t: Union[tensor, TensorWrapper],
    dtype: dtype,
) -> Union[tensor, TensorWrapper]:
    if isinstance(t, TensorWrapper):
        if dtype == t.base.dtype:
            # Reinterpreting to the original interpretation; return the base.
            return t.base
        else:
            # Reinterpreting a wrapped tensor to a different type.
            return TensorWrapper(t.base, dtype)
    elif isinstance(t, torch.Tensor):
        # A new wrapper is needed around an unwrapped tensor.
        return TensorWrapper(t, dtype)
    else:
        raise TypeError(f"Cannot reinterpret a {type(t)}.")


def _to_tensor(x, builder) -> Optional[tensor]:
    if isinstance(x, bool):
        return tensor(builder.get_int1(x), int1)
    # Note: compile-time const integers are represented by unsigned values
    elif isinstance(x, int):
        if -(2**31) <= x < 2**31:
            return tensor(builder.get_int32(x), int32)
        elif 2**31 <= x < 2**32:
            return tensor(builder.get_uint32(x), uint32)
        elif -(2**63) <= x < 2**63:
            return tensor(builder.get_int64(x), int64)
        elif 2**63 <= x < 2**64:
            return tensor(builder.get_uint64(x), uint64)
        else:
            raise RuntimeError(f"Nonrepresentable integer {x}.")
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
    assert False, f"cannot convert {x} to tensor"


# Create custom exception that prints message "hello"
class IncompatibleTypeErrorimpl(Exception):
    def __init__(self, type_a, type_b):
        self.type_a = type_a
        self.type_b = type_b
        self.message = (
            "invalid operands of type "
            + self.type_a.__repr__()
            + " and "
            + self.type_b.__repr__()
        )
        super(IncompatibleTypeErrorimpl, self).__init__(self.message)


def _integer_promote_impl(
    a_ty: dtype,
    b_ty: dtype,
) -> dtype:
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
    assert False


def _computation_type_impl(
    a_ty: dtype,
    b_ty: dtype,
    *,
    div_or_mod: bool,
) -> dtype:
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
    #     Supported PTX op: add, sub, mul, fma, neg, abs, min, max, tanh, ex2, setp
    if a_ty.is_fp16() or b_ty.is_fp16():
        if div_or_mod:
            return float32
        else:
            return float16
    # 4) return bf16 only if both operands are of bf16
    if a_ty.is_bf16() or b_ty.is_bf16():
        if div_or_mod:
            return float32
        if a_ty.is_bf16() and b_ty.is_bf16():
            return bfloat16
        return float32
    if not a_ty.is_int() or not b_ty.is_int():
        assert False
    # 5 ) both operands are integer and undergo
    #    integer promotion
    if div_or_mod and a_ty.int_signedness != b_ty.int_signedness:
        raise ValueError(
            "Cannot use /, #, or % with "
            + a_ty.__repr__()
            + " and "
            + b_ty.__repr__()
            + " because they have different signedness;"
            "this is unlikely to result in a useful answer. Cast them to the same signedness."
        )
    return _integer_promote_impl(a_ty, b_ty)


# ===----------------------------------------------------------------------===//
#                               Binary Operators
# ===----------------------------------------------------------------------===//


def _check_ptr_type_impl(
    type_a: dtype,
    type_b: dtype,
    *,
    allow_ptr_a: bool,
) -> None:
    if type_a.is_ptr():
        if not allow_ptr_a:
            raise IncompatibleTypeErrorimpl(type_a, type_b)
        # T* + U* with T != U
        if type_b.is_ptr() and (type_a != type_b):
            raise IncompatibleTypeErrorimpl(type_a, type_b)
        # T* + float
        if type_b.is_floating():
            raise IncompatibleTypeErrorimpl(type_a, type_b)


def _binary_op_type_checking_impl(
    lhs: tensor,
    rhs: tensor,
    *,
    builder: ir.builder,
    allow_lhs_ptr: bool = False,
    allow_rhs_ptr: bool = False,
    arithmetic_check: bool = True,
    div_or_mod: bool = False,
) -> Tuple[tensor, tensor]:
    # implicit broadcasting
    lhs, rhs = _broadcast_impl_value(lhs, rhs, builder)
    # implicit typecasting
    lhs_sca_ty = lhs.type.scalar
    rhs_sca_ty = rhs.type.scalar
    _check_ptr_type_impl(lhs_sca_ty, rhs_sca_ty, allow_ptr_a=allow_lhs_ptr)
    _check_ptr_type_impl(rhs_sca_ty, lhs_sca_ty, allow_ptr_a=allow_rhs_ptr)
    if arithmetic_check and not lhs_sca_ty.is_ptr() and not rhs_sca_ty.is_ptr():
        ret_sca_ty = _computation_type_impl(
            lhs_sca_ty, rhs_sca_ty, div_or_mod=div_or_mod
        )
        lhs = _cast(lhs, ret_sca_ty, builder)
        rhs = _cast(rhs, ret_sca_ty, builder)
    return lhs, rhs


def _add(
    input: tensor,
    other: tensor,
    builder: ir.builder,
) -> tensor:
    input, other = _binary_op_type_checking_impl(
        input,
        other,
        builder=builder,
        allow_lhs_ptr=True,
        allow_rhs_ptr=True,
    )
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar

    # offset + ptr
    # ptr + offset
    if other_scalar_ty.is_ptr() and not input_scalar_ty.is_ptr():
        input, other = other, input
    if input_scalar_ty.is_ptr():
        return tensor(
            builder.create_gep(input.handle, [other.handle]),
            input.type,
        )
    # float + float
    elif input_scalar_ty.is_floating():
        return tensor(
            builder.create_fadd(input.handle, other.handle),
            input.type,
        )
    # int + int
    elif input_scalar_ty.is_int():
        return tensor(
            builder.create_add(input.handle, other.handle),
            input.type,
        )
    assert False


def _sub(
    input: tensor,
    other: tensor,
    builder: ir.builder,
) -> tensor:
    input, other = _binary_op_type_checking_impl(
        input,
        other,
        builder=builder,
        allow_lhs_ptr=True,
        allow_rhs_ptr=False,
    )
    scalar_ty = input.type.scalar
    # ptr - offset
    if scalar_ty.is_ptr():
        return tensor(
            builder.create_gep(input.handle, [_minus(other, builder).handle]),
            input.type,
        )
    # float - float
    if scalar_ty.is_floating():
        return tensor(
            builder.create_fsub(input.handle, other.handle),
            input.type,
        )
    # int - int
    elif scalar_ty.is_int():
        return tensor(
            builder.create_sub(input.handle, other.handle),
            input.type,
        )
    assert False


def _mul(
    input: tensor,
    other: tensor,
    builder: ir.builder,
) -> tensor:
    input, other = _binary_op_type_checking_impl(input, other, builder=builder)
    scalar_ty = input.type.scalar
    # float * float
    if scalar_ty.is_floating():
        return tensor(
            builder.create_fmul(input.handle, other.handle),
            input.type,
        )
    # * int
    elif scalar_ty.is_int():
        return tensor(
            builder.create_mul(input.handle, other.handle),
            input.type,
        )
    assert False


def _truediv(
    input: tensor,
    other: tensor,
    builder: ir.builder,
) -> tensor:
    input, other = _binary_op_type_checking_impl(
        input,
        other,
        builder=builder,
        allow_lhs_ptr=False,
        allow_rhs_ptr=False,
        arithmetic_check=True,
        div_or_mod=True,
    )
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    # float / int
    if input_scalar_ty.is_floating() and other_scalar_ty.is_int():
        other = _cast(other, input_scalar_ty, builder)
    # int / float
    elif input_scalar_ty.is_int() and other_scalar_ty.is_floating():
        input = _cast(input, other_scalar_ty, builder)
    # int / int (cast to float32)
    elif input_scalar_ty.is_int() and other_scalar_ty.is_int():
        input = _cast(input, float32, builder)
        other = _cast(other, float32, builder)
    # float / float (cast to highest exponent type)
    elif input_scalar_ty.is_floating() and other_scalar_ty.is_floating():
        if input_scalar_ty.fp_mantissa_width > other_scalar_ty.fp_mantissa_width:
            other = _cast(other, input_scalar_ty, builder)
        else:
            input = _cast(input, other_scalar_ty, builder)
    # unreachable
    else:
        assert False
    return tensor(
        builder.create_fdiv(input.handle, other.handle),
        input.type,
    )


def _floordiv(
    input: tensor,
    other: tensor,
    builder: ir.builder,
) -> tensor:
    input, other = _binary_op_type_checking_impl(
        input,
        other,
        builder=builder,
        allow_lhs_ptr=False,
        allow_rhs_ptr=False,
        arithmetic_check=True,
        div_or_mod=True,
    )
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    if input_scalar_ty.is_int() and other_scalar_ty.is_int():
        ret_ty = _integer_promote_impl(input_scalar_ty, other_scalar_ty)
        input = _cast(input, ret_ty, builder)
        other = _cast(other, ret_ty, builder)
        if ret_ty.is_int_signed():
            return tensor(
                builder.create_sdiv(input.handle, other.handle),
                input.type,
            )
        else:
            return tensor(
                builder.create_udiv(input.handle, other.handle),
                input.type,
            )
    assert False


def _fdiv(
    input: tensor,
    other: tensor,
    ieee_rounding: bool,
    builder: ir.builder,
) -> tensor:
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    if not input_scalar_ty.is_floating() or not other_scalar_ty.is_floating():
        raise ValueError("both operands of fdiv must have floating poscalar type")
    input, other = _binary_op_type_checking_impl(
        input,
        other,
        builder=builder,
        allow_lhs_ptr=False,
        allow_rhs_ptr=False,
        arithmetic_check=False,
        div_or_mod=True,
    )
    ret = builder.create_fdiv(input.handle, other.handle)
    ret.set_fdiv_ieee_rounding(ieee_rounding)
    return tensor(ret, input.type)


def _mod(
    input: tensor,
    other: tensor,
    builder: ir.builder,
) -> tensor:
    input, other = _binary_op_type_checking_impl(
        input,
        other,
        builder=builder,
        allow_lhs_ptr=False,
        allow_rhs_ptr=False,
        arithmetic_check=True,
        div_or_mod=True,
    )
    scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    # float % float
    if scalar_ty.is_floating():
        return tensor(
            builder.create_frem(input.handle, other.handle),
            input.type,
        )
    # % int
    elif scalar_ty.is_int():
        if scalar_ty.int_signedness != other_scalar_ty.int_signedness:
            raise ValueError(
                "Cannot mod "
                + scalar_ty.__repr__()
                + " by "
                + other_scalar_ty.__repr__()
                + " "
                "because they have different signedness;"
                "this is unlikely to result in a useful answer. Cast them to the same signedness."
            )
        if scalar_ty.is_int_signed():
            return tensor(
                builder.create_srem(input.handle, other.handle),
                input.type,
            )
        else:
            return tensor(
                builder.create_urem(input.handle, other.handle),
                input.type,
            )
    assert False


def _bitwise_op_type_checking_impl(
    input: tensor,
    other: tensor,
    builder: ir.builder,
) -> Tuple[tensor, tensor]:
    input, other = _binary_op_type_checking_impl(
        input,
        other,
        builder=builder,
        allow_lhs_ptr=False,
        allow_rhs_ptr=False,
        arithmetic_check=False,
    )
    input_sca_ty = input.type.scalar
    other_sca_ty = other.type.scalar
    if not input_sca_ty.is_int() or not other_sca_ty.is_int():
        raise IncompatibleTypeErrorimpl(input_sca_ty, other_sca_ty)
    ret_sca_ty = _integer_promote_impl(input_sca_ty, other_sca_ty)
    if ret_sca_ty != input_sca_ty:
        input = _cast(input, ret_sca_ty, builder)
    if ret_sca_ty != other_sca_ty:
        other = _cast(other, ret_sca_ty, builder)
    return input, other


def _and_(
    input: tensor,
    other: tensor,
    builder: ir.builder,
) -> tensor:
    input, other = _bitwise_op_type_checking_impl(input, other, builder)
    return tensor(
        builder.create_and(input.handle, other.handle),
        input.type,
    )


def _or_(
    input: tensor,
    other: tensor,
    builder: ir.builder,
) -> tensor:
    input, other = _bitwise_op_type_checking_impl(input, other, builder)
    return tensor(
        builder.create_or(input.handle, other.handle),
        input.type,
    )


def _xor_(
    input: tensor,
    other: tensor,
    builder: ir.builder,
) -> tensor:
    input, other = _bitwise_op_type_checking_impl(input, other, builder)
    return tensor(
        builder.create_xor(input.handle, other.handle),
        input.type,
    )


def _lshr(
    input: tensor,
    other: tensor,
    builder: ir.builder,
) -> tensor:
    input, other = _bitwise_op_type_checking_impl(input, other, builder)
    return tensor(
        builder.create_lshr(input.handle, other.handle),
        input.type,
    )


def _shl(
    input: tensor,
    other: tensor,
    builder: ir.builder,
) -> tensor:
    input, other = _bitwise_op_type_checking_impl(input, other, builder)
    return tensor(
        builder.create_shl(input.handle, other.handle),
        input.type,
    )


def _plus(input: tensor) -> tensor:
    return input


def _minus(input: tensor, builder: ir.builder) -> tensor:
    input_sca_ty = input.type.scalar
    if input_sca_ty.is_ptr():
        raise ValueError(
            "wrong type argument to unary minus (" + input_sca_ty.__repr__() + ")"
        )
    _0 = tensor(
        ir.constant.get_null_value(input_sca_ty.to_ir(builder)),
        input_sca_ty,
    )
    return _sub(_0, input, builder)


def _invert(input: tensor, builder: tensor) -> tensor:
    input_sca_ty = input.type.scalar
    if input_sca_ty.is_ptr() or input_sca_ty.is_floating():
        raise ValueError(
            "wrong type argument to unary invert (" + input_sca_ty.__repr__() + ")"
        )
    _1 = tensor(
        ir.constant.get_all_ones_value(input_sca_ty.to_ir(builder)),
        input_sca_ty,
    )
    return _xor_(input, _1, builder)


def _bool_like(v: tensor) -> dtype:
    type = v.type
    if isinstance(type, block_type):
        shape = type.shape
        return block_type(int1, shape)

    return int1


def _greater_than(
    input: tensor,
    other: tensor,
    builder: ir.builder,
) -> tensor:
    input, other = _binary_op_type_checking_impl(input, other, builder=builder)
    scalar_ty = input.type.scalar
    # float > float
    if scalar_ty.is_floating():
        return tensor(
            builder.create_fcmpOGT(input.handle, other.handle),
            _bool_like(input),
        )
    # > int
    elif scalar_ty.is_int():
        if scalar_ty.is_int_signed():
            return tensor(
                builder.create_icmpSGT(input.handle, other.handle),
                _bool_like(input),
            )
        else:
            return tensor(
                builder.create_icmpUGT(input.handle, other.handle),
                _bool_like(input),
            )
    assert False


def _greater_equal(
    input: tensor,
    other: tensor,
    builder: ir.builder,
) -> tensor:
    input, other = _binary_op_type_checking_impl(input, other, builder=builder)
    scalar_ty = input.type.scalar
    # float >= float
    if scalar_ty.is_floating():
        return tensor(
            builder.create_fcmpOGE(input.handle, other.handle),
            _bool_like(input),
        )
    # >= int
    elif scalar_ty.is_int():
        if scalar_ty.is_int_signed():
            return tensor(
                builder.create_icmpSGE(input.handle, other.handle),
                _bool_like(input),
            )
        else:
            return tensor(
                builder.create_icmpUGE(input.handle, other.handle),
                _bool_like(input),
            )
    assert False


def _less_than(input: tensor, other: tensor, builder: ir.builder) -> tensor:
    input, other = _binary_op_type_checking_impl(input, other, builder=builder)
    scalar_ty = input.type.scalar
    # float < float
    if scalar_ty.is_floating():
        return tensor(
            builder.create_fcmpOLT(input.handle, other.handle),
            _bool_like(input),
        )
    # < int
    elif scalar_ty.is_int():
        if scalar_ty.is_int_signed():
            return tensor(
                builder.create_icmpSLT(input.handle, other.handle),
                _bool_like(input),
            )
        else:
            return tensor(
                builder.create_icmpULT(input.handle, other.handle),
                _bool_like(input),
            )
    assert False


def _less_equal(
    input: tensor,
    other: tensor,
    builder: ir.builder,
) -> tensor:
    input, other = _binary_op_type_checking_impl(input, other, builder=builder)
    scalar_ty = input.type.scalar
    # float < float
    if scalar_ty.is_floating():
        return tensor(
            builder.create_fcmpOLE(input.handle, other.handle),
            _bool_like(input),
        )
    # < int
    elif scalar_ty.is_int():
        if scalar_ty.is_int_signed():
            return tensor(
                builder.create_icmpSLE(input.handle, other.handle),
                _bool_like(input),
            )
        else:
            return tensor(
                builder.create_icmpULE(input.handle, other.handle),
                _bool_like(input),
            )
    assert False


def _equal(
    input: tensor,
    other: tensor,
    builder: ir.builder,
) -> tensor:
    input, other = _binary_op_type_checking_impl(input, other, builder=builder)
    scalar_ty = input.type.scalar
    # float == float
    if scalar_ty.is_floating():
        return tensor(
            builder.create_fcmpOEQ(input.handle, other.handle),
            _bool_like(input),
        )
    # == int
    elif scalar_ty.is_int():
        return tensor(
            builder.create_icmpEQ(input.handle, other.handle),
            _bool_like(input),
        )
    assert False


def _not_equal(
    input: tensor,
    other: Optional[tensor],
    builder: ir.builder,
) -> tensor:
    if other is None:
        return True

    input, other = _binary_op_type_checking_impl(input, other, builder=builder)
    scalar_ty = input.type.scalar
    # float == float
    if scalar_ty.is_floating():
        return tensor(
            builder.create_fcmpUNE(input.handle, other.handle),
            _bool_like(input),
        )
    # == int
    elif scalar_ty.is_int():
        return tensor(
            builder.create_icmpNE(input.handle, other.handle),
            _bool_like(input),
        )
    assert False


def _reshape(
    input: tensor,
    dst_shape: List[int],
    builder: ir.builder,
) -> tensor:
    numel = 1
    for s in dst_shape:
        numel *= s
    if input.type.numel != numel:
        raise ValueError("cannot reshape block of different shape")
    ret_ty = block_type(input.type.scalar, dst_shape)
    return tensor(builder.create_reshape(input.handle, dst_shape), ret_ty)


def _broadcast_impl_shape(
    input: tensor,
    shape: List[int],
    builder: ir.builder,
) -> tensor:
    type = input.type
    if not isinstance(type, block_type):
        ret_ty = block_type(input.type, shape)
        return tensor(builder.create_splat(input.handle, shape), ret_ty)

    src_shape = type.get_block_shapes()
    if len(src_shape) != len(shape):
        raise ValueError(f"Cannot broadcast, rank mismatch: {src_shape}, {shape}")
    if shape == src_shape:
        return input
    for i in range(len(src_shape)):
        if shape[i] != src_shape[i] and src_shape[i] != 1:
            raise ValueError(
                f"Cannot broadcast, the expanded size of the tensor ({shape[i]})"
                f" must match the existing size ({src_shape[1]}) at non-singleton dimension"
                f" {i}: {src_shape}, {shape}"
            )
    ret_ty = block_type(input.type.scalar, shape)
    return tensor(builder.create_broadcast(input.handle, shape), ret_ty)


def _broadcast_impl_value(
    lhs: tensor,
    rhs: tensor,
    builder: ir.builder,
) -> Tuple[tensor, tensor]:
    lhs_ty = lhs.type
    rhs_ty = rhs.type

    # scalar, scalar
    if not isinstance(lhs_ty, block_type) and not isinstance(rhs_ty, block_type):
        return lhs, rhs

    # make_shape_compatible(block, block)
    if isinstance(lhs_ty, block_type) and isinstance(rhs_ty, block_type):
        lhs_shape = lhs_ty.get_block_shapes()
        rhs_shape = rhs_ty.get_block_shapes()
        if len(lhs_shape) != len(rhs_shape):
            raise ValueError(
                "Cannot make_shape_compatible: blocks must have the same rank"
            )
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
                raise ValueError(
                    "Cannot make_shape_compatible: incompatible dimensions "
                    "at index " + str(i) + ": " + str(left) + " and " + str(right)
                )
        if lhs_shape != ret_shape:
            ret_ty = block_type(lhs_ty.scalar, ret_shape)
            lhs = tensor(
                builder.create_broadcast(lhs.handle, ret_shape),
                ret_ty,
            )
        if rhs_shape != ret_shape:
            ret_ty = block_type(rhs_ty.scalar, ret_shape)
            rhs = tensor(builder.create_broadcast(rhs.handle, ret_shape), ret_ty)

        return lhs, rhs

    # make_shape_compatible(block, scalar)
    if isinstance(lhs_ty, block_type) and not isinstance(rhs_ty, block_type):
        rhs_ty = block_type(rhs_ty.scalar, lhs_ty.shape)
        rhs = tensor(
            builder.create_splat(rhs.handle, lhs_ty.get_block_shapes()),
            rhs_ty,
        )

        return lhs, rhs

    # make_shape_compatible(scalar, block)
    if not isinstance(lhs_ty, block_type) and isinstance(rhs_ty, block_type):
        lhs_ty = block_type(lhs_ty.scalar, rhs_ty.shape)
        lhs = tensor(
            builder.create_splat(lhs.handle, rhs_ty.get_block_shapes()),
            lhs_ty,
        )

    return lhs, rhs


def _bitcast(input: tensor, dst_ty: dtype, builder: ir.builder) -> tensor:
    src_ty = input.type
    if isinstance(src_ty, block_type):
        dst_ty = block_type(dst_ty, src_ty.get_block_shapes())
    if src_ty == dst_ty:
        return input
    src_sca_ty = src_ty.scalar
    dst_sca_ty = dst_ty.scalar
    if src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
        return _cast(input, dst_ty, builder)
    # Bitcast
    src_bits = src_sca_ty.primitive_bitwidth
    dst_bits = dst_sca_ty.primitive_bitwidth
    if src_bits != dst_bits:
        raise ValueError(
            "Cannot bitcast data-type of size " + str(src_bits) + "to "
            "data-type of size " + str(dst_bits)
        )
    return tensor(
        builder.create_bitcast(input.handle, dst_ty.to_ir(builder)),
        dst_ty,
    )


def _cast(input: tensor, dst_ty: dtype, builder: ir.builder) -> tensor:
    src_ty = input.type
    if isinstance(src_ty, block_type) and not isinstance(dst_ty, block_type):
        dst_ty = block_type(dst_ty, src_ty.get_block_shapes())
    if src_ty == dst_ty:
        return input
    src_sca_ty = src_ty.scalar
    dst_sca_ty = dst_ty.scalar
    # fp8 <=> bf16/fp16
    if (src_sca_ty.is_bf16() or src_sca_ty.is_fp16()) and dst_sca_ty.is_fp8():
        return tensor(
            builder.create_fp_trunc(input.handle, dst_ty.to_ir(builder)),
            dst_ty,
        )
    if src_sca_ty.is_fp8() and (dst_sca_ty.is_bf16() or dst_sca_ty.is_fp16()):
        return tensor(
            builder.create_fp_ext(input.handle, dst_ty.to_ir(builder)),
            dst_ty,
        )
    # bf16 <=> (not fp32)
    if (src_sca_ty.is_bf16() and not dst_sca_ty.is_fp32()) or (
        dst_sca_ty.is_bf16() and not src_sca_ty.is_fp32()
    ):
        return _cast(_cast(input, float32, builder), dst_sca_ty, builder)

    # FP Truncation
    truncate_fp = (
        src_sca_ty.is_floating()
        and dst_sca_ty.is_floating()
        and src_sca_ty.fp_mantissa_width > dst_sca_ty.fp_mantissa_width
    )
    if truncate_fp:
        return tensor(
            builder.create_fp_trunc(input.handle, dst_ty.to_ir(builder)),
            dst_ty,
        )

    # FP Extension
    ext_fp = (
        src_sca_ty.is_floating()
        and dst_sca_ty.is_floating()
        and src_sca_ty.fp_mantissa_width < dst_sca_ty.fp_mantissa_width
    )
    if ext_fp:
        return tensor(
            builder.create_fp_ext(input.handle, dst_ty.to_ir(builder)),
            dst_ty,
        )

    # Int cast
    if (
        src_sca_ty.is_int()
        and dst_sca_ty.is_int()
        and (
            src_sca_ty.int_bitwidth != dst_sca_ty.int_bitwidth
            or src_sca_ty.int_signedness != dst_sca_ty.int_signedness
        )
    ):
        sign_extend = src_sca_ty.is_int_signed() and not src_sca_ty.is_bool()
        return tensor(
            builder.create_int_cast(input.handle, dst_ty.to_ir(builder), sign_extend),
            dst_ty,
        )

    # Float to Int
    if src_sca_ty.is_floating() and dst_sca_ty.is_int():
        # TODO: is this correct?
        if dst_sca_ty.is_bool():
            return _not_equal(input, _to_tensor(0, builder), builder)
        else:
            return tensor(
                builder.create_fp_to_si(input.handle, dst_ty.to_ir(builder)),
                dst_ty,
            )

    # int => float
    if src_sca_ty.is_int() and dst_sca_ty.is_floating():
        if src_sca_ty.is_bool() or not src_sca_ty.is_int_signed():
            return tensor(
                builder.create_ui_to_fp(input.handle, dst_ty.to_ir(builder)),
                dst_ty,
            )
        else:
            return tensor(
                builder.create_si_to_fp(input.handle, dst_ty.to_ir(builder)),
                dst_ty,
            )

    # ptr => int
    if src_sca_ty.is_ptr() and dst_sca_ty.is_int():
        bitwidth = dst_sca_ty.int_bitwidth
        if bitwidth == 64:
            return tensor(
                builder.create_ptr_to_int(input.handle, dst_ty.to_ir(builder)),
                dst_ty,
            )
        if bitwidth == 1:
            return _not_equal(
                _cast(input, int64, builder),
                tensor(builder.get_int64(0), int64),
                builder,
            )

    if not src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
        return tensor(
            builder.create_int_to_ptr(input.handle, dst_ty.to_ir(builder)),
            dst_ty,
        )
    # Ptr . Ptr
    if src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
        return tensor(
            builder.create_bitcast(input.handle, dst_ty.to_ir(builder)),
            dst_ty,
        )
    # * . Bool
    if dst_sca_ty.is_bool():
        if src_sca_ty.is_ptr():
            input = _cast(input, int64, builder)
        other = builder.get_int64(0)
        if src_ty.is_bool():
            assert isinstance(src_ty, block_type)
            other = builder.create_splat(other, src_ty.get_block_shapes())
        return tensor(builder.create_icmpNE(input.handle, other), dst_ty)
    assert False, f"cannot cast {input} to {dst_ty}"


def _where(condition: tensor, x: tensor, y: tensor, builder: ir.builder) -> tensor:
    condition = _cast(condition, int1, builder)
    if condition.type.is_block():
        condition, x = _broadcast_impl_value(condition, x, builder)
        x, y = _broadcast_impl_value(x, y, builder)
        condition, x = _broadcast_impl_value(condition, x, builder)

    x, y = _binary_op_type_checking_impl(
        x,
        y,
        builder=builder,
        allow_lhs_ptr=True,
        allow_rhs_ptr=True,
    )
    if not condition.type.is_block():
        condition, _ = _broadcast_impl_value(condition, x, builder)
    ret_ty = x.type
    return tensor(
        builder.create_select(condition.handle, x.handle, y.handle),
        ret_ty,
    )
