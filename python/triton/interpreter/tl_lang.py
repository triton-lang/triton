from __future__ import annotations

from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap

torch = torch_wrapper.torch


def _primitive_to_tensor(x):
    """
    Converts various Python primitive data types to PyTorch tensor.
    """
    tensor_args = {"device": "cuda"}
    if isinstance(x, bool):
        return torch.tensor([x], dtype=torch.bool, **tensor_args)
    elif isinstance(x, int):
        if -(2**31) <= x < 2**31:
            return torch.tensor([x], dtype=torch.int32, **tensor_args)
        elif -(2**63) <= x < 2**63:
            return torch.tensor([x], dtype=torch.int64, **tensor_args)
        else:
            raise RuntimeError(f"Nonrepresentable integer {x}.")
    elif isinstance(x, float):
        return torch.tensor([x], dtype=torch.float32, **tensor_args)
    elif torch.is_tensor(x):
        return x
    elif isinstance(x, WrappedTensor):
        return x
    elif isinstance(x, debugger_constexpr):
        if x.value is None:
            return None
        return _primitive_to_tensor(x.value)
    elif x is None:
        return None
    assert False, f"cannot convert {x} of type {type(x)} to tensor"


def _infer_tensor(func):
    """
    A decorator function to harmonize function args:
        - converts primitives to PyTorch tensors
        - wraps PyTorch tensors with WrappedTensors
    """
    def wrapper(*args):
        new_args = tuple(map(lambda v: _primitive_to_tensor(v), args))
        new_args = tuple(map(lambda v: WrappedTensor(v) if torch.is_tensor(v) else v, new_args))

        return func(*new_args)

    return wrapper


def _tensor_operation(func):
    """
    A decorator function to unwrap WrappedTensors and debugger_constexpr before calling the function.
    Can be combined with _infer_tensor decorator to harmonize args (everything to torch tensor).
    """
    def wrapper(*args, **kwargs):
        for arg in args:
            assert not torch.is_tensor(arg), "unexpected tensor argument"

        def unwrap_tensor(v):
            if isinstance(v, WrappedTensor):
                return v.tensor
            if isinstance(v, debugger_constexpr):
                return v.value
            return v

        new_args = tuple(map(unwrap_tensor, args))
        new_kwargs = {k: unwrap_tensor(v) for k, v in kwargs.items()}

        result = func(args[0], *new_args[1:], **new_kwargs)
        return WrappedTensor(result) if torch.is_tensor(result) else result

    return wrapper


class debugger_constexpr:
    def __init__(self, value):
        if isinstance(value, debugger_constexpr):
            self.value = value.value
        else:
            self.value = value

    def __str__(self) -> str:
        return "debugger_constexpr(" + str(self.value) + ")"

    def __index__(self) -> int:
        return self.value

    def __bool__(self):
        return bool(self.value)

    def __ge__(self, other):
        other = other.value if isinstance(other, debugger_constexpr) else other
        return self.value >= other

    def __gt__(self, other):
        other = other.value if isinstance(other, debugger_constexpr) else other
        return self.value > other

    def __le__(self, other):
        other = other.value if isinstance(other, debugger_constexpr) else other
        return self.value <= other

    def __lt__(self, other):
        other = other.value if isinstance(other, debugger_constexpr) else other
        return self.value < other

    def __eq__(self, other):
        other = other.value if isinstance(other, debugger_constexpr) else other
        return self.value == other

    def __or__(self, other):
        other = other.value if isinstance(other, debugger_constexpr) else other
        return self.value | other

    def __ror__(self, other):
        other = other.value if isinstance(other, debugger_constexpr) else other
        return self.value | other

    def __and__(self, other):
        other = other.value if isinstance(other, debugger_constexpr) else other
        return self.value & other

    def __rand__(self, other):
        other = other.value if isinstance(other, debugger_constexpr) else other
        return self.value & other

    def to(self, dtype, bitcast=False, _builder=None):
        if dtype in [torch.int64]:
            ret_ty = int
        elif dtype == torch.bool:
            ret_ty = bool
        elif dtype in [torch.float64]:
            ret_ty = float
        else:
            raise ValueError("dtype not supported in debugger")
        return debugger_constexpr(ret_ty(self.value))


class WrappedTensor:
    def __init__(self, tensor):
        self.tensor = tensor

    def __index__(self) -> int:
        return self.tensor.item()

    def __str__(self) -> str:
        return "wrapped_" + str(self.tensor)

    def __bool__(self) -> bool:
        return torch.all(self.tensor == True).item()  # noqa: E712

    @property
    def dtype(self):
        return self.tensor.dtype

    @_infer_tensor
    @_tensor_operation
    def __add__(self, other):
        return torch.add(self.tensor, other)

    @_infer_tensor
    @_tensor_operation
    def __radd__(self, other):
        return self.__add__(other)

    @_infer_tensor
    @_tensor_operation
    def __sub__(self, other):
        return torch.sub(self.tensor, other)

    @_infer_tensor
    @_tensor_operation
    def __rsub__(self, other):
        return torch.sub(other, self.tensor)

    @_infer_tensor
    @_tensor_operation
    def __mul__(self, other):
        return torch.mul(self.tensor, other)

    @_infer_tensor
    @_tensor_operation
    def __rmul__(self, other):
        return self.__mul__(other)

    @_infer_tensor
    @_tensor_operation
    def __truediv__(self, other):
        return torch.div(self.tensor, other)

    @_infer_tensor
    @_tensor_operation
    def __rtruediv__(self, other):
        return torch.div(other, self.tensor)

    @_infer_tensor
    @_tensor_operation
    def __floordiv__(self, other):
        return torch.floor_divide(self.tensor, other)

    @_infer_tensor
    @_tensor_operation
    def __rfloordiv__(self, other):
        return torch.floor_divide(other, self.tensor)

    @_infer_tensor
    @_tensor_operation
    def __mod__(self, other):
        return torch.remainder(self.tensor, other)

    @_infer_tensor
    @_tensor_operation
    def __rmod__(self, other):
        return torch.remainder(other, self.tensor)

    @_infer_tensor
    @_tensor_operation
    def __neg__(self):
        return -self.tensor

    @_infer_tensor
    @_tensor_operation
    def __invert__(self):
        return ~self.tensor

    @_infer_tensor
    @_tensor_operation
    def __and__(self, other):
        return torch.bitwise_and(self.tensor, other)

    @_infer_tensor
    @_tensor_operation
    def __or__(self, other):
        return torch.bitwise_or(self.tensor, other)

    @_infer_tensor
    @_tensor_operation
    def __xor__(self, other):
        return torch.bitwise_xor(self.tensor, other)

    @_infer_tensor
    @_tensor_operation
    def __lshift__(self, other):
        return torch.bitwise_left_shift(self.tensor, other)

    @_infer_tensor
    @_tensor_operation
    def __rshift__(self, other):
        return torch.bitwise_right_shift(self.tensor, other)

    @_infer_tensor
    @_tensor_operation
    def __gt__(self, other):
        return self.tensor > other

    @_infer_tensor
    @_tensor_operation
    def __rgt__(self, other):
        return other > self.tensor

    @_infer_tensor
    @_tensor_operation
    def __ge__(self, other):
        return self.tensor >= other

    @_infer_tensor
    @_tensor_operation
    def __rge__(self, other):
        return other >= self.tensor

    @_infer_tensor
    @_tensor_operation
    def __lt__(self, other):
        return self.tensor < other

    @_infer_tensor
    @_tensor_operation
    def __rlt__(self, other):
        return other < self.tensor

    @_infer_tensor
    @_tensor_operation
    def __le__(self, other):
        return self.tensor <= other

    @_infer_tensor
    @_tensor_operation
    def __rle__(self, other):
        return other <= self.tensor

    @_infer_tensor
    @_tensor_operation
    def __eq__(self, other):
        return torch.equal(self.tensor, other)

    @_infer_tensor
    @_tensor_operation
    def __ne__(self, other):
        return not torch.equal(self.tensor, other)

    @_tensor_operation
    def __getitem__(self, slices):
        return self.tensor.__getitem__(slices)
        # if isinstance(slices, slice):
        #     slices = [slices]
        # src_shape = self.shape
        # dst_shape = []
        # curr = 0
        # for sl in slices:
        #     if isinstance(sl, constexpr) and sl.value is None:
        #         dst_shape.append(1)
        #     elif sl == slice(None, None, None):
        #         dst_shape.append(src_shape[curr].value)
        #         curr += 1
        # ret = torch.reshape(self.tensor, dst_shape, )
        # return ret

    @_tensor_operation
    def to(self, dtype, bitcast=False):
        return self.tensor.to(dtype)
        # if isinstance(bitcast, constexpr):
        #     bitcast = bitcast.value
        # if bitcast:
        #     return semantic.bitcast(self, dtype, )
        # return semantic.cast(self, dtype, )


def _constexpr_to_value(v):
    if isinstance(v, debugger_constexpr):
        return v.value
    return v


class TritonLangProxy:
    _memory_map: MemoryMap
    _context: ExecutionContext

    def __init__(self, memory_map: MemoryMap, context: ExecutionContext):
        self._memory_map = memory_map
        self._context = context

    # Types
    # Removed void, int1, float8, uint16, uint32, uint64, pi32_t

    # constexpr = debugger_constexpr

    # Program functions

    @_tensor_operation
    def load(
        self,
        pointer: torch.Tensor,
        mask: torch.Tensor = None,
        other=0.0,
        cache_modifier="",
        eviction_policy="",
        volatile=False,
    ):
        return self._memory_map.load(pointer, mask, other)

    @_tensor_operation
    def store(self, pointer: torch.Tensor, value: torch.Tensor, mask=None):
        return self._memory_map.store(pointer, value, mask)

    @_tensor_operation
    def program_id(self, axis):
        assert axis < len(self._context.program_id)
        return torch.tensor([self._context.program_id[axis]], dtype=torch.int32, device="cuda")

    @_tensor_operation
    def num_programs(self, axis):
        assert axis < len(self._context.program_size)
        return torch.tensor([self._context.program_size[axis]], dtype=torch.int32, device="cuda")

    @_tensor_operation
    def arange(self, start, end):
        return torch.arange(start=start, end=end, dtype=torch.int32, device="cuda")

    @_tensor_operation
    def zeros(self, shape, dtype):
        for i, d in enumerate(shape):
            if not isinstance(d, debugger_constexpr):
                raise TypeError(f"Shape element {i} must have type `constexpr`")
            if not isinstance(d.value, int):
                raise TypeError(f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]")
        shape = [x.value for x in shape]
        if isinstance(dtype, lcore.dtype):
            if dtype.is_fp32():
                dtype = torch.float32
            elif dtype.is_fp16():
                dtype = torch.float16
            elif dtype.is_bf16():
                dtype = torch.bfloat16
            elif dtype.is_int32():
                dtype = torch.int32
            elif dtype.is_int16():
                dtype = torch.int16
            elif dtype.is_int8():
                dtype = torch.int8
            else:
                raise TypeError(f"Unsupported dtype {dtype}")
        return torch.zeros(size=shape, dtype=dtype, device="cuda")

    @_tensor_operation
    def dequantize(self, input, scale, shift, nbit, dst_ty=None):
        if dst_ty is None:
            dst_ty = torch.float16
        raise NotImplementedError()

    @_tensor_operation
    def broadcast(self, input, other):
        raise NotImplementedError()

    @_tensor_operation
    def broadcast_to(self, input, shape):
        raise NotImplementedError()

    @_tensor_operation
    def cat(self, input, shape):
        raise NotImplementedError()

    @_tensor_operation
    def reshape(self, input, shape):
        raise NotImplementedError()

    @_tensor_operation
    def dot(self, input, other, trans_a=False, trans_b=False, allow_tf32=True):
        assert input.dtype == other.dtype
        if trans_a:
            input = input.T
        if trans_b:
            other = other.T
        return torch.matmul(input=input, other=other)

    @_tensor_operation
    def atomic_cas(self, pointer, cmp, val):
        stored = self._memory_map.load(pointer, None, 0.0)
        if not isinstance(cmp, torch.Tensor):
            cmp = torch.tensor([cmp], dtype=stored.dtype, device="cuda")
        if not isinstance(val, torch.Tensor):
            val = torch.tensor([val], dtype=stored.dtype, device="cuda")
        if stored == cmp:
            self._memory_map.store(pointer, val, None)
        return stored

    @_tensor_operation
    def atomic_xchg(self, pointer, val, mask=None):
        if isinstance(val, int):
            val = torch.tensor([val], dtype=torch.int32, device="cuda")
        stored = self._memory_map.load(pointer, mask, 0.0)
        self._memory_map.store(pointer, val, mask)
        return stored

    @_tensor_operation
    def atomic_add(self, pointer, val, mask=None):
        # arbitrary other value as it will masked during storing
        stored = self._memory_map.load(pointer, mask, 0.0)
        result = stored + val
        self._memory_map.store(pointer, result, mask)
        return stored

    @_tensor_operation
    def atomic_max(self, pointer, val, mask=None):
        stored = self._memory_map.load(pointer, mask, 0.0)
        result = torch.maximum(stored, val)
        self._memory_map.store(pointer, result, mask)
        return stored

    @_tensor_operation
    def atomic_min(self, pointer, val, mask=None):
        stored = self._memory_map.load(pointer, mask, 0.0)
        result = torch.minimum(stored, val)
        self._memory_map.store(pointer, result, mask)
        return stored

    @_tensor_operation
    def atomic_and(self, pointer, val, mask=None):
        stored = self._memory_map.load(pointer, mask, 0)
        result = torch.bitwise_and(stored, val)
        self._memory_map.store(pointer, result, mask)
        return stored

    @_tensor_operation
    def atomic_or(self, pointer, val, mask=None):
        stored = self._memory_map.load(pointer, mask, 0)
        result = torch.bitwise_or(stored, val)
        self._memory_map.store(pointer, result, mask)
        return stored

    @_tensor_operation
    def atomic_xor(self, pointer, val, mask=None):
        stored = self._memory_map.load(pointer, mask, 0)
        result = torch.bitwise_xor(stored, val)
        self._memory_map.store(pointer, result, mask)
        return stored

    @_tensor_operation
    def where(self, condition, x, y):
        condition = _primitive_to_tensor(condition)
        x = _primitive_to_tensor(x)
        y = _primitive_to_tensor(y)
        return torch.where(condition, x, y)

    @_tensor_operation
    def umulhi(self, x, y):
        raise NotImplementedError()

    @_tensor_operation
    def fdiv(self, x, y, ieee_rounding=False):
        raise NotImplementedError()

    @_tensor_operation
    def exp(self, x):
        return torch.exp(x)

    @_tensor_operation
    def log(self, x):
        return torch.log(x)

    @_tensor_operation
    def cos(self, x):
        return torch.cos(x)

    @_tensor_operation
    def sin(self, x):
        return torch.sin(x)

    @_tensor_operation
    def sqrt(self, x):
        return torch.sqrt(x)

    @_tensor_operation
    def globaltimer(self):
        raise NotImplementedError()

    @_tensor_operation
    def clock(self):
        raise NotImplementedError()

    @_tensor_operation
    def debug_barrier(self):
        raise NotImplementedError()

    @_tensor_operation
    def multiple_of(self, input, values):
        return input

    @_tensor_operation
    def max_contiguous(self, input, values):
        return input

    @_tensor_operation
    def max_constancy(self, input, values):
        return input

    @_tensor_operation
    def abs(self, x):
        return torch.abs(x)

    @_tensor_operation
    def cdiv(self, x, div):
        return (x + div - 1) // div

    @_tensor_operation
    def minimum(self, x, y):
        if isinstance(x, int):
            x = torch.tensor(x, device="cuda")
        if isinstance(y, int):
            y = torch.tensor(y, device="cuda")
        return torch.minimum(x, y)

    @_tensor_operation
    def maximum(self, x, y):
        return torch.maximum(x, y)

    @_tensor_operation
    def sigmoid(self, x):
        raise NotImplementedError()

    @_tensor_operation
    def softmax(self, x, ieee_rounding=False):
        raise NotImplementedError()

    @_tensor_operation
    def ravel(self, x):
        raise NotImplementedError()

    @_tensor_operation
    def swizzle2d(self, i, j, size_i, size_j, size_g):
        raise NotImplementedError()

    @_tensor_operation
    def zeros_like(self, input):
        raise NotImplementedError()

    @_tensor_operation
    def max(self, input, axis=None):
        if axis is None:
            return torch.max(input)
        return torch.max(input, dim=axis).values

    @_tensor_operation
    def argmax(self, input, axis):
        raise NotImplementedError()

    @_tensor_operation
    def min(self, input, axis=None):
        if axis is None:
            return torch.min(input)
        return torch.min(input, dim=axis).values

    @_tensor_operation
    def argmin(self, input, axis):
        raise NotImplementedError()

    @_tensor_operation
    def sum(self, input, axis=None):
        if axis is None:
            return torch.sum(input)
        return torch.sum(input, dim=axis)

    @_tensor_operation
    def xor_sum(self, input, axis):
        raise NotImplementedError()

    @_tensor_operation
    def cumsum(self, input, axis=None):
        if axis is None:
            return torch.cumsum(input)
        return torch.cumsum(input, dim=axis)

    @_tensor_operation
    def cumprod(self, input, axis=None):
        if axis is None:
            return torch.cumprod(input)
        return torch.cumprod(input, dim=axis)
