from dataclasses import dataclass
from typing import List, Any


@dataclass
class TensorDescriptor:
    base: Any
    shape: List[int]
    strides: List[int]
    block_shape: List[int]

    def __post_init__(self):
        rank = len(self.shape)
        assert len(self.strides) == rank, f"rank mismatch: {self}"
        assert len(self.block_shape) == rank, f"rank mismatch: {self}"

    @staticmethod
    def from_tensor(tensor: Any, block_shape: List[int]):
        return TensorDescriptor(
            tensor,
            tensor.shape,
            tensor.stride(),
            block_shape,
        )
