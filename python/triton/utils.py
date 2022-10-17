from __future__ import annotations

import torch


def cdiv(x, y):
    return (x + y - 1) // y


def next_power_of_2(n):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n


class MockTensor:
    """
    Can be used in place of real tensors when calling:
        kernel.warmup(MockTensor(torch.float32), ...)
    """
    @staticmethod
    def wrap_dtype(arg):
        if isinstance(arg, torch.dtype):
            return MockTensor(arg)
        return arg

    def __init__(self, dtype):
        self.dtype = dtype

    def data_ptr(self):
        return 0  # optimistically assumes multiple of 16


class TensorWrapper:
    def __init__(self, base, dtype):
        self.dtype = dtype
        self.base = base
        self.is_cuda = base.is_cuda
        self.device = base.device

    def data_ptr(self):
        return self.base.data_ptr()

    def __str__(self) -> str:
        return f'TensorWrapper[{self.dtype}]({self.base})'


def reinterpret(tensor, dtype):
    if isinstance(tensor, TensorWrapper):
        if dtype == tensor.base.dtype:
            # Reinterpreting to the original interpretation; return the base.
            return tensor.base
        else:
            # Reinterpreting a wrapped tensor to a different type.
            return TensorWrapper(tensor.base, dtype)
    elif isinstance(tensor, torch.Tensor):
        # A new wrapper is needed around an unwrapped tensor.
        return TensorWrapper(tensor, dtype)
    else:
        raise TypeError(f'Cannot reinterpret a {type(tensor)}.')
