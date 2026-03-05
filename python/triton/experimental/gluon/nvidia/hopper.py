from dataclasses import dataclass
from typing import List, Any, Optional
from triton._utils import validate_block_shape, canonicalize_dtype, get_primitive_bitwidth
from triton.experimental.gluon.language._layouts import NVMMASharedLayout
import triton.language as tl

__all__ = ["TensorDescriptor", "TensorDescriptorIm2Col"]


def _validate_common_descriptor(tensor, shape, strides, layout, padding, round_f32_to_tf32, block_shape):
    rank = len(shape)
    assert len(strides) == rank, "strides rank mismatch"
    assert 0 < rank <= 5, "rank must be 1-5"
    assert tensor.data_ptr() % 16 == 0, "base must be 16-byte aligned"

    dtype_str = canonicalize_dtype(tensor.dtype)
    elem_bytes = get_primitive_bitwidth(dtype_str) // 8
    for stride in strides[:-1]:
        assert (stride * elem_bytes) % 16 == 0, "strides must be 16-byte aligned"
    for shape_dim in shape:
        assert shape_dim > 0, "shape must be positive"
    assert strides[-1] == 1, "Last dimension must be contiguous"
    assert isinstance(layout, NVMMASharedLayout), "Layout must be NVMMASharedLayout"
    assert padding == "zero" or padding == "nan", "Illegal value for padding"
    if padding == "nan":
        assert tensor.dtype.is_floating_point, "Padding option `nan` is only supported for floating point tensors"
    if round_f32_to_tf32:
        assert dtype_str == "fp32", "round_f32_to_tf32 is only supported for float32 tensors"
    assert elem_bytes * 8 == layout.element_bitwidth
    padding_factor = 2 if layout.fp4_padded else 1
    min_block = layout.swizzle_byte_width // (elem_bytes * padding_factor)
    assert block_shape[-1] >= min_block, \
        f"Expected block_shape[-1] to be at least {min_block} but got {block_shape[-1]}"
    if layout.fp4_padded:
        assert tensor.data_ptr() % 32 == 0, "For fp4_padded, base must 32-byte aligned"
        for stride in strides[:-1]:
            assert (stride * elem_bytes) % 32 == 0, "For fp4_padded, tensor strides must be 32-byte aligned"
        assert tl.target_info.cuda_capability_geq(10, 0), "fp4_padded requires blackwell or newer"
    assert not layout.fp4_padded or layout.swizzle_byte_width == 128, (
        f"FP4 padded operands must be swizzled with 128-byte width, but got {layout.swizzle_byte_width}")
    assert layout.element_bitwidth in [
        8, 16, 32
    ], (f"tensor descriptor dtype must be 8, 16, or 32 bits, but got {layout.element_bitwidth}")
    return rank


@dataclass
class TensorDescriptor:
    base: Any
    shape: List[int]
    strides: List[int]
    block_shape: List[int]
    layout: NVMMASharedLayout
    padding: str = "zero"
    round_f32_to_tf32: bool = False

    def __post_init__(self):
        rank = len(self.shape)
        assert len(self.block_shape) == rank, f"tiled: block_shape must match rank {rank}"
        rank = _validate_common_descriptor(
            self.base,
            self.shape,
            self.strides,
            self.layout,
            self.padding,
            self.round_f32_to_tf32,
            self.block_shape,
        )
        validate_block_shape(self.block_shape)

    @property
    def mode(self) -> str:
        return "tiled"

    def __mangle__(self):
        """Generate a type string matching MLIR types (!ttng.tensordesc or !ttng.tensordesc_im2col)."""
        dtype_str = canonicalize_dtype(self.base.dtype)
        block_shape_str = ','.join(map(str, self.block_shape))
        return f"tensordesc<{dtype_str}[{block_shape_str}],{repr(self.layout)}>"

    @staticmethod
    def from_tensor(tensor: Any, block_shape: List[int], layout: NVMMASharedLayout, padding="zero",
                    round_f32_to_tf32=False):
        """
        Create a TensorDescriptor from a tensor.

        Args:
            tensor: Input tensor
            block_shape: Block dimensions for TMA copy.
                Tiled mode: must match tensor rank.
            layout: NVMMASharedLayout for shared memory
            padding: "zero" (default) or "nan" for out-of-bounds padding
            round_f32_to_tf32: Round float32 to TF32 precision (default False)
        """
        return TensorDescriptor(
            tensor,
            tensor.shape,
            tensor.stride(),
            block_shape,
            layout,
            padding,
            round_f32_to_tf32,
        )


@dataclass
class TensorDescriptorIm2Col:
    base: Any
    shape: List[int]
    strides: List[int]
    block_shape: List[int]
    layout: NVMMASharedLayout
    padding: str = "zero"
    round_f32_to_tf32: bool = False
    element_strides: Optional[List[int]] = None  # Element strides per dimension (optional)
    pixel_box_lower_corner: Optional[List[int]] = None  # Im2col: box start offsets (DHW)
    pixel_box_upper_corner: Optional[List[int]] = None  # Im2col: box end offsets (DHW)

    def __post_init__(self):
        assert len(self.block_shape) == 2, "im2col: block_shape must be 2D"
        rank = _validate_common_descriptor(
            self.base,
            self.shape,
            self.strides,
            self.layout,
            self.padding,
            self.round_f32_to_tf32,
            self.block_shape,
        )
        # Validate element_strides if provided
        if self.element_strides is not None:
            assert len(self.element_strides
                       ) == rank, f"element_strides length mismatch: expected {rank}, got {len(self.element_strides)}"
            for i, s in enumerate(self.element_strides):
                assert 0 < s <= 8, f"element_strides[{i}] must be in (0, 8], got {s}"
            assert self.element_strides[-1] == 1, f"element_strides[-1] must be 1, got {self.element_strides[-1]}"

        assert rank in [3, 4, 5], f"im2col mode requires rank 3, 4, or 5, got {rank}"
        spatial_rank = rank - 2

        assert self.pixel_box_lower_corner is not None, "pixel_box_lower_corner required for im2col"
        assert self.pixel_box_upper_corner is not None, "pixel_box_upper_corner required for im2col"
        assert len(self.pixel_box_lower_corner) == spatial_rank, "pixel_box_lower_corner length mismatch"
        assert len(self.pixel_box_upper_corner) == spatial_rank, "pixel_box_upper_corner length mismatch"

        # Validate box corner ranges based on rank
        offset_ranges = {3: (-32768, 32767), 4: (-128, 127), 5: (-16, 15)}
        lo, hi = offset_ranges[rank]
        for corner, name in [(self.pixel_box_lower_corner, "Lower"), (self.pixel_box_upper_corner, "Upper")]:
            for i, val in enumerate(corner):
                assert lo <= val <= hi, f"pixel_box_{name.lower()}_corner[{i}] must be in [{lo}, {hi}], got {val}"

        # block_shape is [pixelsPerColumn, channelsPerPixel], both must be powers of 2
        def is_power_of_2(n):
            return n > 0 and (n & (n - 1)) == 0

        assert is_power_of_2(self.block_shape[0]), f"block_shape[0] must be power of 2, got {self.block_shape[0]}"
        assert is_power_of_2(self.block_shape[1]), f"block_shape[1] must be power of 2, got {self.block_shape[1]}"

    @property
    def mode(self) -> str:
        return "im2col"

    def __mangle__(self):
        """Generate a type string matching MLIR types (!ttng.tensordesc or !ttng.tensordesc_im2col)."""
        dtype_str = canonicalize_dtype(self.base.dtype)
        block_shape_str = ','.join(map(str, self.block_shape))
        return f"tensordesc_im2col<{dtype_str}[{block_shape_str}],{repr(self.layout)}>"

    @staticmethod
    def from_tensor(tensor: Any, block_shape: List[int], layout: NVMMASharedLayout, padding="zero",
                    round_f32_to_tf32=False, element_strides=None, pixel_box_lower_corner=None,
                    pixel_box_upper_corner=None):
        """
        Create a TensorDescriptorIm2Col from a tensor.

        Args:
            tensor: Input tensor
            block_shape: Block dimensions for TMA copy (2D [pixelsPerColumn, channelsPerPixel])
            layout: NVMMASharedLayout for shared memory
            padding: "zero" (default) or "nan" for out-of-bounds padding
            round_f32_to_tf32: Round float32 to TF32 precision (default False)
            element_strides: Element strides per dimension (optional, each in range (0, 8])
            pixel_box_lower_corner: Im2col mode - box start offsets (DHW dimensions)
            pixel_box_upper_corner: Im2col mode - box end offsets (DHW dimensions)
        """
        return TensorDescriptorIm2Col(
            tensor,
            tensor.shape,
            tensor.stride(),
            block_shape,
            layout,
            padding,
            round_f32_to_tf32,
            element_strides,
            pixel_box_lower_corner,
            pixel_box_upper_corner,
        )
