from dataclasses import dataclass
from typing import List, Any


@dataclass
class TensorDescriptor:
    base: Any
    shape: List[int]
    strides: List[int]
    block_shape: List[int]

    @staticmethod
    def from_tensor(tensor: Any, block_shape: List[int]):
        return TensorDescriptor(
            tensor,
            tensor.shape,
            tensor.stride(),
            block_shape,
        )
