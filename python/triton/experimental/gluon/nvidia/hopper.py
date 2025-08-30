from dataclasses import dataclass
from typing import List, Any
from triton._utils import validate_block_shape, canonicalize_dtype, get_primitive_bitwidth
from triton.experimental.gluon.language._layouts import NVMMASharedLayout

__all__ = ["TensorDescriptor"]


@dataclass
class TensorDescriptor:
    base: Any
    shape: List[int]
    strides: List[int]
    block_shape: List[int]
    layout: NVMMASharedLayout
    padding: str = "zero"

    def __post_init__(self):
        rank = len(self.shape)
        assert len(self.strides) == rank, f"rank mismatch: {self}"
        assert len(self.block_shape) == rank, f"rank mismatch: {self}"
        assert rank > 0, "rank must not be zero"
        assert rank <= 5, "rank cannot be more than 5"
        assert self.base.data_ptr() % 16 == 0, "base must be 16-byte aligned"
        validate_block_shape(self.block_shape)
        dtype_str = canonicalize_dtype(self.base.dtype)
        elem_bytes = get_primitive_bitwidth(dtype_str) // 8
        for stride in self.strides[:-1]:
            assert (stride * elem_bytes) % 16 == 0, "strides must be 16-byte aligned"
        assert self.strides[-1] == 1, "Last dimension must be contiguous"
        assert isinstance(self.layout, NVMMASharedLayout), "Layout must be NVMMASharedLayout"
        assert self.padding == "zero" or self.padding == "nan", "Illegal value for padding"
        if self.padding == "nan":
            assert self.base.dtype.is_floating_point, "Padding option `nan` is only supported for floating point tensors"

    @staticmethod
    def from_tensor(tensor: Any, block_shape: List[int], layout: NVMMASharedLayout, padding="zero"):
        return TensorDescriptor(
            tensor,
            tensor.shape,
            tensor.stride(),
            block_shape,
            layout,
            padding,
        )
