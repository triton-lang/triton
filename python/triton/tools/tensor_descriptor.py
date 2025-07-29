from dataclasses import dataclass
from typing import List, Any
from triton._utils import validate_block_shape


@dataclass
class TensorDescriptor:
    base_ptr: Any
    dtype: Any
    shape: List[int]
    strides: List[int]
    block_shape: List[int]

    def __post_init__(self):
        rank = len(self.shape)
        assert len(self.strides) == rank, f"rank mismatch: {self}"
        assert len(self.block_shape) == rank, f"rank mismatch: {self}"
        assert rank > 0, "rank must not be zero"
        assert rank <= 5, "rank cannot be more than 5"
        if self.base_ptr is not None:
            assert self.base_ptr % 16 == 0, "base must be 16-byte aligned"
        validate_block_shape(self.block_shape)
        elem_bytes = self.dtype.itemsize
        for stride in self.strides[:-1]:
            assert (stride * elem_bytes) % 16 == 0, "strides must be 16-byte aligned"
        assert self.strides[-1] == 1, "Last dimension must be contiguous"

    @staticmethod
    def from_tensor(tensor: Any, block_shape: List[int]):
        data_ptr = None
        if type(tensor).__name__ not in ("FakeTensor", "FunctionalTensor"):
            data_ptr = tensor.data_ptr()
        return TensorDescriptor(
            data_ptr,
            tensor.dtype,
            tensor.shape,
            tensor.stride(),
            block_shape,
        )
