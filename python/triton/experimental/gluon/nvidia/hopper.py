from dataclasses import dataclass
from typing import List, Any
from triton._utils import validate_block_shape, canonicalize_dtype, get_primitive_bitwidth
from triton.experimental.gluon.language._layouts import NVMMASharedLayout
import triton.language as tl

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
        for shape_dim in self.shape:
            assert shape_dim > 0, "shape must be positive"
        assert self.strides[-1] == 1, "Last dimension must be contiguous"
        assert isinstance(self.layout, NVMMASharedLayout), "Layout must be NVMMASharedLayout"
        assert self.padding == "zero" or self.padding == "nan", "Illegal value for padding"
        if self.padding == "nan":
            assert self.base.dtype.is_floating_point, "Padding option `nan` is only supported for floating point tensors"
        assert elem_bytes * 8 == self.layout.element_bitwidth
        padding_factor = 2 if self.layout.fp4_padded else 1
        min_block = self.layout.swizzle_byte_width // (elem_bytes * padding_factor)
        assert self.block_shape[-1] >= min_block, \
            f"Expected block_shape[-1] to be at least {min_block} but got {self.block_shape[-1]}"
        if self.layout.fp4_padded:
            for stride in self.strides[:-1]:
                assert (stride * elem_bytes) % 32 == 0, "For fp4_padded, tensor strides must be 32-byte aligned"
            assert tl.target_info.cuda_capability_geq(10, 0), "fp4_padded requires blackwell or newer"
        assert not self.layout.fp4_padded or self.layout.swizzle_byte_width == 128, f"FP4 padded operands must be swizzled with 128-byte width, but got {self.layout.swizzle_byte_width}"
        assert self.layout.element_bitwidth in [
            8, 16, 32
        ], f"tensor descriptor dtype must be 8, 16, or 32 bits, but got {self.layout.element_bitwidth}"

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
