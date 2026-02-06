from dataclasses import dataclass
from typing import List, Any, Optional
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
    mode: str = "tiled"  # "tiled" or "im2col"
    round_f32_to_tf32: bool = False

    elementStrides: Optional[List[int]] = None  # Element strides per dimension (optional)
    pixelBoxLowerCorner: Optional[List[int]] = None  # Im2col: box start offsets (DHW)
    pixelBoxUpperCorner: Optional[List[int]] = None  # Im2col: box end offsets (DHW)

    def __post_init__(self):
        rank = len(self.shape)
        assert len(self.strides) == rank, "strides rank mismatch"
        assert self.mode in ["tiled", "im2col"], f"invalid mode: {self.mode}"
        assert 0 < rank <= 5, "rank must be 1-5"
        assert self.base.data_ptr() % 16 == 0, "base must be 16-byte aligned"

        if self.mode == "tiled":
            assert len(self.block_shape) == rank, f"tiled: block_shape must match rank {rank}"
            validate_block_shape(self.block_shape)
        else:
            assert len(self.block_shape) == 2, "im2col: block_shape must be 2D"

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
        if self.round_f32_to_tf32:
            assert dtype_str == "fp32", "round_f32_to_tf32 is only supported for float32 tensors"
        assert elem_bytes * 8 == self.layout.element_bitwidth
        padding_factor = 2 if self.layout.fp4_padded else 1
        min_block = self.layout.swizzle_byte_width // (elem_bytes * padding_factor)
        assert self.block_shape[-1] >= min_block, \
            f"Expected block_shape[-1] to be at least {min_block} but got {self.block_shape[-1]}"
        if self.layout.fp4_padded:
            assert self.base.data_ptr() % 32 == 0, "For fp4_padded, base must 32-byte aligned"
            for stride in self.strides[:-1]:
                assert (stride * elem_bytes) % 32 == 0, "For fp4_padded, tensor strides must be 32-byte aligned"
            assert tl.target_info.cuda_capability_geq(10, 0), "fp4_padded requires blackwell or newer"
        assert not self.layout.fp4_padded or self.layout.swizzle_byte_width == 128, f"FP4 padded operands must be swizzled with 128-byte width, but got {self.layout.swizzle_byte_width}"
        assert self.layout.element_bitwidth in [
            8, 16, 32
        ], f"tensor descriptor dtype must be 8, 16, or 32 bits, but got {self.layout.element_bitwidth}"

        # Validate elementStrides if provided
        if self.elementStrides is not None:
            assert len(self.elementStrides
                       ) == rank, f"elementStrides length mismatch: expected {rank}, got {len(self.elementStrides)}"
            for i, s in enumerate(self.elementStrides):
                assert 0 < s <= 8, f"elementStrides[{i}] must be in (0, 8], got {s}"
            assert self.elementStrides[-1] == 1, f"elementStrides[-1] must be 1, got {self.elementStrides[-1]}"

        # Validate IM2COL mode parameters
        if self.mode == "im2col":
            assert rank in [3, 4, 5], f"im2col mode requires rank 3, 4, or 5, got {rank}"
            spatial_rank = rank - 2

            assert self.pixelBoxLowerCorner is not None, "pixelBoxLowerCorner required for im2col"
            assert self.pixelBoxUpperCorner is not None, "pixelBoxUpperCorner required for im2col"
            assert len(self.pixelBoxLowerCorner) == spatial_rank, "pixelBoxLowerCorner length mismatch"
            assert len(self.pixelBoxUpperCorner) == spatial_rank, "pixelBoxUpperCorner length mismatch"

            # Validate box corner ranges based on rank
            offset_ranges = {3: (-32768, 32767), 4: (-128, 127), 5: (-16, 15)}
            lo, hi = offset_ranges[rank]
            for corner, name in [(self.pixelBoxLowerCorner, "Lower"), (self.pixelBoxUpperCorner, "Upper")]:
                for i, val in enumerate(corner):
                    assert lo <= val <= hi, f"pixelBox{name}Corner[{i}] must be in [{lo}, {hi}], got {val}"

            # block_shape is [pixelsPerColumn, channelsPerPixel], both must be powers of 2
            def is_power_of_2(n):
                return n > 0 and (n & (n - 1)) == 0

            assert is_power_of_2(self.block_shape[0]), f"block_shape[0] must be power of 2, got {self.block_shape[0]}"
            assert is_power_of_2(self.block_shape[1]), f"block_shape[1] must be power of 2, got {self.block_shape[1]}"

    def __mangle__(self):
        """Generate a type string matching MLIR types (!ttng.tensordesc or !ttng.tensordesc_im2col)."""
        dtype_str = canonicalize_dtype(self.base.dtype)
        block_shape_str = ','.join(map(str, self.block_shape))
        type_name = "tensordesc_im2col" if self.mode == "im2col" else "tensordesc"
        return f"{type_name}<{dtype_str}[{block_shape_str}],{repr(self.layout)}>"

    @staticmethod
    def from_tensor(tensor: Any, block_shape: List[int], layout: NVMMASharedLayout, padding="zero", mode="tiled",
                    round_f32_to_tf32=False, elementStrides=None, pixelBoxLowerCorner=None, pixelBoxUpperCorner=None):
        """
        Create a TensorDescriptor from a tensor.

        Args:
            tensor: Input tensor
            block_shape: Block dimensions for TMA copy.
                Tiled mode: must match tensor rank.
                Im2col mode: must be 2D [pixelsPerColumn, channelsPerPixel], both powers of 2.
            layout: NVMMASharedLayout for shared memory
            padding: "zero" (default) or "nan" for out-of-bounds padding
            mode: "tiled" (default) or "im2col"
            round_f32_to_tf32: Round float32 to TF32 precision (default False)
            elementStrides: Element strides per dimension (optional, each in range (0, 8])
            pixelBoxLowerCorner: Im2col mode - box start offsets (DHW dimensions)
            pixelBoxUpperCorner: Im2col mode - box end offsets (DHW dimensions)
        """
        return TensorDescriptor(
            tensor,
            tensor.shape,
            tensor.stride(),
            block_shape,
            layout,
            padding,
            mode,
            round_f32_to_tf32,
            elementStrides,
            pixelBoxLowerCorner,
            pixelBoxUpperCorner,
        )
