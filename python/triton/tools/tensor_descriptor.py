from dataclasses import dataclass
from typing import List, Any
from triton._utils import validate_block_shape, canonicalize_dtype, fp6_logical_shape, validate_fp6_box


@dataclass
class TensorDescriptor:
    base: Any
    shape: List[int]
    strides: List[int]
    block_shape: List[int]
    padding: str = "zero"
    round_f32_to_tf32: bool = False
    fp6_format: str | None = None

    def __post_init__(self):
        rank = len(self.shape)
        assert len(self.strides) == rank, f"rank mismatch: {self}"
        assert len(self.block_shape) == rank, f"rank mismatch: {self}"
        assert rank > 0, "rank must not be zero"
        assert rank <= 5, "rank cannot be more than 5"
        ty = type(self.base)
        if ty.__name__ not in ("FakeTensor", "FunctionalTensor"):
            assert self.base.data_ptr() % 16 == 0, "base must be 16-byte aligned"
        validate_block_shape(self.block_shape)
        if self.fp6_format is not None:
            if self.fp6_format not in ("e2m3", "e3m2") or canonicalize_dtype(self.base.dtype) != "u8":
                raise ValueError("FP6 descriptors require uint8 storage and format 'e2m3' or 'e3m2'")
            validate_fp6_box(self.block_shape[-1], self.shape[-1], self.strides[:-1], self.padding, host=True)
        elem_bytes = self.base.dtype.itemsize
        for stride in self.strides[:-1]:
            assert (stride * elem_bytes) % 16 == 0, "strides must be 16-byte aligned"
        for shape_dim in self.shape:
            assert shape_dim > 0, "shape must be positive"
        assert self.strides[-1] == 1, "Last dimension must be contiguous"
        assert self.padding == "zero" or self.padding == "nan", "Illegal value for padding"
        if self.padding == "nan":
            assert self.base.dtype.is_floating_point, "Padding option `nan` is only supported for floating point tensors"
        if self.round_f32_to_tf32:
            dtype_name = canonicalize_dtype(self.base.dtype)
            assert dtype_name == "fp32", "round_f32_to_tf32 is only supported for float32 tensors"

    @staticmethod
    def from_tensor(tensor: Any, block_shape: List[int], padding="zero", round_f32_to_tf32=False, fp6_format=None):
        """For packed FP6 storage, block_shape and descriptor offsets count logical FP6 values."""
        shape = fp6_logical_shape(tensor.shape) if fp6_format is not None else tensor.shape
        return TensorDescriptor(tensor, shape, tensor.stride(), block_shape, padding, round_f32_to_tf32, fp6_format)
