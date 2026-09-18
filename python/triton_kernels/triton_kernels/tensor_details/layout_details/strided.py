from dataclasses import dataclass
from .base import Layout, LayoutTransformation
from .torch_utils import repack
import torch


# ------------------- Layout Definition -------------------
@dataclass(frozen=True)
class StridedLayout(Layout):

    # NOTE: We only encode the (logical) major dimension; the full dimension order is
    # derived from the tensor rank. This keeps the API minimal while still allowing
    # "which dim is contiguous/packed" to be expressed.
    #
    # For a tensor of rank `R`, the derived order is:
    #   base = list(reversed(range(R)))
    #   swap base[0] with base[index(major_dim)]
    #   order = base
    #
    # This matches the previous default `order=list(reversed(range(R)))` when
    # `major_dim == R - 1`.
    major_dim: int = -1

    def __post_init__(self):
        if not isinstance(self.major_dim, int):
            raise TypeError(f"StridedLayout(major_dim=...) must be an int, got {type(self.major_dim)}")

    def make_transformation(self, shape: list[int], is_fp4: bool) -> LayoutTransformation:
        return StridedLayoutTransformation(shape, is_fp4, self.order(len(shape)))

    @property
    def name(self):
        return "STRIDED"

    def can_preserve_storage_as(self, other: Layout, rank: int) -> bool:
        return isinstance(other, StridedLayout) and self.order(rank) == other.order(rank)

    def swizzle_block_shape(self, block_shape):
        return block_shape

    def order(self, rank: int) -> list[int]:
        """
        Returns the minor->major dimension order for a given tensor rank.

        `self.major_dim` supports negative indexing (like Python).
        """
        if rank <= 0:
            return []
        if not (-rank <= self.major_dim < rank):
            raise ValueError(f"Invalid StridedLayout.major_dim={self.major_dim} for rank={rank}")
        major_dim = self.major_dim if self.major_dim >= 0 else self.major_dim + rank
        base = list(reversed(range(rank)))
        # Preserve the previous behavior: derive from canonical reversed order, then
        # swap the requested major dimension into position 0.
        idx = base.index(major_dim)
        base[0], base[idx] = base[idx], base[0]
        return base


@dataclass(frozen=True)
class StridedLayoutTransformation(LayoutTransformation):

    order: list[int]

    def _convert_data_from(self, data, source: LayoutTransformation, *, out):
        if (not isinstance(source, StridedLayoutTransformation) or (self.is_fp4 and data.dtype != torch.uint8)):
            return super()._convert_data_from(data, source, out=out)
        if out is None:
            out = torch.empty_strided(self.storage_shape, self.storage_strides, dtype=data.dtype, device=data.device)
        return repack(data, source.order[0], self.order[0], self.is_fp4, out=out)

    def _can_convert_fp4(self, data):
        """Packed shape identifies nibble pairs; readers use actual strides."""
        return (self.is_fp4 and len(self.shape) >= 2 and self.order[0] >= len(self.shape) - 2
                and data.device.type == "cuda" and data.dtype == torch.uint8 and list(data.shape) == self.storage_shape)

    @property
    def storage_shape(self) -> list[int]:
        shape = list(self.shape)
        if self.is_fp4:
            packing_dim = self.order[0]
            if shape[packing_dim] % 2:
                raise ValueError(
                    f"FP4 packing dimension {packing_dim} must have an even size, got {shape[packing_dim]}")
            shape[packing_dim] //= 2
        return shape

    @property
    def storage_strides(self) -> list[int]:
        shape = self.storage_shape
        strides, size = [0] * len(shape), 1
        for dim in self.order:
            strides[dim], size = size, size * shape[dim]
        return strides

    def swizzle_data(self, data):
        r = len(self.shape)
        if r == 0:
            return self._validate_storage_shape(data)
        pd = self.order[0]  # packed/contiguous dim in output
        out = torch.empty_strided(self.storage_shape, self.storage_strides, dtype=data.dtype, device=data.device)
        repack(data, -1, pd, self.is_fp4, out=out)
        return self._validate_storage_shape(out)

    def unswizzle_data(self, data):
        out_shape = list(self.shape)
        if self.is_fp4:
            out_shape[-1] //= 2
        ret = torch.empty(out_shape, dtype=data.dtype, device=data.device)
        repack(data, self.order[0], -1, self.is_fp4, out=ret)
        return ret
