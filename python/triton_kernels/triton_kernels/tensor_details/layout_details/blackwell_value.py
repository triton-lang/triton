from dataclasses import dataclass
import torch
from .base import Layout, LayoutTransformation
from .torch_utils import repack


# ------------------- Blackwell MX Value Layout -------------------
@dataclass(frozen=True)
class BlackwellMXValueLayout(Layout):

    @property
    def name(self):
        return "BLACKWELL_MX_VALUE"

    def make_transformation(self, shape: list[int], is_fp4: bool) -> LayoutTransformation:
        return BlackwellMXValueLayoutTransformation(shape, is_fp4)

    def swizzle_block_shape(self, block_shape):
        return block_shape


def strides_major_dim_m2(shape):
    n = len(shape)
    if n <= 1:
        return [1] * n
    order = [n - 2, n - 1] + list(range(n - 3, -1, -1))  # fastest -> slowest
    st = [0] * n
    st[order[0]] = 1
    for prev, d in zip(order, order[1:]):
        st[d] = st[prev] * shape[prev]
    return st


# ------------------- Blackwell MX Value Layout Transformation -------------------
@dataclass(frozen=True)
class BlackwellMXValueLayoutTransformation(LayoutTransformation):

    def swizzle_data(self, data):
        assert data.stride(-1) == 1
        # re-pack as column-major
        data = repack(data, -1, -2, self.is_fp4)
        out_shape = list(data.shape)
        out_shape[-2] += (-out_shape[-2]) % 128
        # gc.collect()
        # torch.cuda.empty_cache()
        ret = torch.empty_strided(out_shape, strides_major_dim_m2(out_shape), device=data.device, dtype=data.dtype)
        ret[..., :data.shape[-2], :] = data
        return ret

    def unswizzle_data(self, data: torch.Tensor):
        assert data.stride(-2) == 1
        sizes = [self.shape[i] for i in range(data.ndim)]
        sizes[-2] //= 2
        data = data[tuple(slice(0, s) for s in sizes)]
        data = repack(data, -2, -1, self.is_fp4)
        data = data.contiguous()
        return data
