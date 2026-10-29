from dataclasses import dataclass
from typing import List, Any
from triton._utils import validate_block_shape, canonicalize_dtype


@dataclass
class TensorDescriptor:
    base: Any
    shape: List[int]
    strides: List[int]
    block_shape: List[int]
    padding: str = "zero"
    round_f32_to_tf32: bool = False
    dtype: Any = None

    def __post_init__(self):
        rank = len(self.shape)
        assert len(self.strides) == rank, f"rank mismatch: {self}"
        assert len(self.block_shape) == rank, f"rank mismatch: {self}"
        assert rank > 0, "rank must not be zero"
        assert rank <= 5, "rank cannot be more than 5"
        check_address = type(self.base).__name__ not in ("FakeTensor", "FunctionalTensor")
        if self.dtype is not None:
            from triton.runtime.jit import reinterpret
            if not self.dtype.is_fp6() or canonicalize_dtype(self.base.dtype) != "u8":
                raise ValueError("a descriptor dtype override requires FP6 on a uint8 base")
            self.base = reinterpret(self.base, self.dtype)
        if getattr(self.base.dtype, "name", None) in ("fp6e2m3", "fp6e3m2"):
            if self.block_shape[-1] != 128 or self.shape[-1] % 128:
                raise ValueError("host FP6 descriptors require a 128-value inner box")
            if (check_address and self.base.data_ptr() % 32) or any(s % 32 for s in self.strides[:-1]):
                raise ValueError("FP6 tensor map base and outer strides must be 32-byte aligned")
            if self.padding != "zero" or self.round_f32_to_tf32:
                raise ValueError("FP6 tensor maps require zero padding")
        if check_address:
            assert self.base.data_ptr() % 16 == 0, "base must be 16-byte aligned"
        validate_block_shape(self.block_shape)
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
    def from_tensor(tensor: Any, block_shape: List[int], padding="zero", round_f32_to_tf32=False, dtype=None):
        """Infer logical extents from storage. An FP6 dtype interprets uint8 rows as packed codes."""
        shape = tensor.shape
        if (dtype is not None and dtype.is_fp6()) or getattr(tensor.dtype, "name", None) in ("fp6e2m3", "fp6e3m2"):
            if shape[-1] % 96:
                raise ValueError("packed FP6 rows must contain a multiple of 96 bytes")
            shape = [*shape[:-1], shape[-1] // 96 * 128]
        return TensorDescriptor(tensor, shape, tensor.stride(), block_shape, padding, round_f32_to_tf32, dtype)
