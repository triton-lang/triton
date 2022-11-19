from __future__ import annotations

import torch


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def next_power_of_2(n: int) -> int:
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
