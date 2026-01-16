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

    def swizzle_data(self, data):
        assert data.stride(-1) == 1
        r = len(self.shape)
        if r == 0:
            return data
        pd = self.order[0]  # packed/contiguous dim in output
        out_shape = list(self.shape)
        if self.is_fp4:
            out_shape[pd] //= 2
        # dense strides in minor->major `self.order`
        stride, s = [0] * r, 1
        for d in self.order:
            stride[d], s = s, s * out_shape[d]
        out = torch.empty_strided(out_shape, stride, dtype=data.dtype, device=data.device)
        repack(data, -1, pd, self.is_fp4, out=out)
        return out

    def unswizzle_data(self, data):
        assert data.stride(self.order[0]) == 1
        out_shape = list(self.shape)
        if self.is_fp4:
            out_shape[-1] //= 2
        ret = torch.empty(out_shape, dtype=data.dtype, device=data.device)
        repack(data, self.order[0], -1, self.is_fp4, out=ret)
        assert ret.stride(-1) == 1
        return ret
