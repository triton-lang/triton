import torch

from ... import jit
from ... import language as tl
from ... import next_power_of_2


def num_warps(n):
    if n <= 128:
        return 1
    if n <= 256:
        return 2
    if n <= 512:
        return 4
    if n <= 4096:
        return 8
    return 16


@jit
def _blocksparse_softmax_fwd(Out, A, stride_xz, LUT,  #
                             R, extent, stride_zr, stride_hr,  # relative attention
                             scale, is_causal,  #
                             ROW_SIZE: tl.constexpr,  #
                             BLOCK_SIZE: tl.constexpr,  #
                             IS_DENSE: tl.constexpr  #
                             ):
    h = tl.program_id(0)
    m = tl.program_id(1)
    z = tl.program_id(2)
    # create index ranges
    hm = h * tl.num_programs(1) + m
    lane_n = tl.arange(0, ROW_SIZE) % BLOCK_SIZE
    block_n = tl.arange(0, ROW_SIZE) // BLOCK_SIZE
    # extract information from LUT
    header = LUT + (hm // BLOCK_SIZE) * 2
    size = tl.load(header + 0)
    offset = tl.load(header + 1)
    # pointer offset
    off_a = z * stride_xz
    off_a += (offset + block_n) * BLOCK_SIZE * BLOCK_SIZE  # block indx
    off_a += (m % BLOCK_SIZE) * BLOCK_SIZE  # row indx
    # do not need to read column indices in the dense case
    if IS_DENSE:
        ns = tl.arange(0, ROW_SIZE)
    else:
        off_lut = offset + 2 * tl.num_programs(0) * tl.num_programs(1) // BLOCK_SIZE
        start_n = tl.load(LUT + off_lut + block_n, mask=block_n < size, other=0)
        ns = start_n * BLOCK_SIZE + lane_n
    # load X
    mask = block_n < size
    a = tl.load(A + off_a + lane_n, mask=mask, other=-float("inf"))
    a = a.to(tl.float32)
    # compute
    out = a
    out *= scale
    # apply relative attention
    if R is not None:
        R += z * stride_zr
        R += h * stride_hr
        off_lo = (extent - m - 1) + ns
        mask_lo = (off_lo >= 0) & (off_lo < extent)
        rel_logits = tl.load(R + m * extent + off_lo, mask=mask_lo, other=0.0)
        out += rel_logits
    out = out.to(tl.float32)
    # apply causal mask
    out = tl.where((ns > m) & is_causal, -float("inf"), out)
    # computation
    out = tl.softmax(out)
    # write-back
    tl.store(Out + off_a + lane_n, out, mask=mask)


@jit
def _blocksparse_softmax_bwd(DA, stride_zdx,  #
                             DOut, stride_zdout,  #
                             Out, stride_zout,  #
                             scale,  #
                             LUT,  #
                             DR, extent, stride_zr, stride_hr, stride_er,  #
                             is_causal,  #
                             ROW_SIZE: tl.constexpr,  #
                             BLOCK_SIZE: tl.constexpr,  #
                             IS_DENSE: tl.constexpr):
    h = tl.program_id(0)
    m = tl.program_id(1)
    z = tl.program_id(2)
    # create index ranges
    hm = h * tl.num_programs(1) + m
    lane_n = tl.arange(0, ROW_SIZE) % BLOCK_SIZE
    block_n = tl.arange(0, ROW_SIZE) // BLOCK_SIZE
    # extract information from LUT
    header = LUT + (hm // BLOCK_SIZE) * 2
    size = tl.load(header + 0)
    offset = tl.load(header + 1)
    # row-col offset
    off_mn = (offset + block_n) * BLOCK_SIZE * BLOCK_SIZE
    off_mn += (m % BLOCK_SIZE) * BLOCK_SIZE
    mask = block_n < size
    # pointers
    As = Out + z * stride_zout + off_mn
    DOuts = DOut + z * stride_zdout + off_mn
    # do not need to read column indices in the dense case
    if IS_DENSE:
        ns = tl.arange(0, ROW_SIZE)
    else:
        off_lut = offset + 2 * tl.num_programs(0) * tl.num_programs(1) // BLOCK_SIZE
        start_n = tl.load(LUT + off_lut + block_n, mask=mask, other=0)
        ns = start_n * BLOCK_SIZE + lane_n
    # load data
    a = tl.load(As + lane_n, mask=mask, other=0.0)
    a = a.to(tl.float32)
    dout = tl.load(DOuts + lane_n, mask=mask, other=0.0)
    dout = dout.to(tl.float32)
    # compute
    a = tl.where((ns > m) & is_causal & (a == a), 0., a)
    da = a * (dout - tl.sum(a * dout, 0))
    # apply relative attention
    if DR is not None:
        DR += z * stride_zr
        DR += h * stride_hr
        off_lo = (extent - m - 1) + ns
        mask_lo = (off_lo >= 0) & (off_lo < extent) & mask
        tl.store(DR + m * extent + off_lo, da, mask=mask_lo)
    da = da * scale
    # convert da
    # write-back
    DAs = DA + z * stride_zdx + off_mn
    tl.store(DAs + lane_n, da, mask=mask)


class _softmax(torch.autograd.Function):

    @staticmethod
    def make_lut(layout, block, device):
        _empty = torch.tensor([], dtype=torch.int64, device=layout.device)
        sizes = _empty.clone()
        # sizes along rows
        for h in range(layout.shape[0]):
            sizes = torch.cat((sizes, layout[h, :, :].sum(-1)))
        total_sizes = sizes * block
        # offsets in block format
        offsets = torch.zeros_like(sizes)
        offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
        # block indices
        columns = layout.nonzero(as_tuple=False)[:, 2]
        header = torch.stack((sizes, offsets), dim=1).view(-1)
        lut = torch.cat((header, columns)).type(torch.int32).to(device)
        return lut, int(total_sizes.max())

    @staticmethod
    def forward(ctx, a, scale, rel_logits, is_causal, spdims, block, lut, maxlut, is_dense):
        if scale is not None and isinstance(scale, torch.Tensor):
            assert scale.device.type == "cpu"
            scale = scale.item()
        M = a.shape[0]
        grid = [spdims[0], spdims[1] * block, M]
        rel_shape = (1, 1, 1, 1) if rel_logits is None else rel_logits.shape
        rel_strides = (1, 1, 1, 1) if rel_logits is None else rel_logits.stride()
        # enqueue kernel
        out = torch.empty_like(a)
        _blocksparse_softmax_fwd[grid](
            out, a, a.stride(0), lut,  #
            rel_logits, rel_shape[-1], rel_strides[0], rel_strides[1],  # relative attn#
            scale,  #
            is_causal,  #
            BLOCK_SIZE=block,  #
            ROW_SIZE=next_power_of_2(maxlut),  #
            IS_DENSE=is_dense,  #
            num_warps=num_warps(maxlut)  #
        )
        # save to context
        # ctx.mark_dirty(x)
        ctx.save_for_backward(out, lut)
        ctx.spdims = spdims
        ctx.block = block
        ctx.maxlut = maxlut
        ctx.scale = scale
        ctx.rel_shape = rel_shape
        ctx.rel_strides = rel_strides
        ctx.rel_dtype = a.dtype
        ctx.is_dense = is_dense
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, dout):
        # retrieve from context
        out, lut = ctx.saved_tensors
        # relative logits gradients
        dr = None
        if ctx.needs_input_grad[3]:
            dr = torch.zeros(ctx.rel_shape, dtype=ctx.rel_dtype, device=out.device)
        # run kernel
        M = out.shape[0]
        grid = (ctx.spdims[0], ctx.spdims[1] * ctx.block, M)
        da = torch.empty_like(dout)
        _blocksparse_softmax_bwd[grid](
            da, da.stride(0),  #
            dout, dout.stride(0),  #
            out, out.stride(0),  #
            ctx.scale,  #
            lut,  #
            dr, ctx.rel_shape[-1], ctx.rel_strides[0], ctx.rel_strides[1], ctx.rel_strides[2],  #
            ctx.is_causal,  #
            BLOCK_SIZE=ctx.block,  #
            ROW_SIZE=next_power_of_2(ctx.maxlut),  #
            IS_DENSE=ctx.is_dense,  #
            num_warps=num_warps(ctx.maxlut)  #
        )
        return (da, None, None, dr, None, None, None, None, None, None, None, None, None, None, None, None, None, None)


class softmax:

    def __init__(self, layout, block, device, is_dense=False):
        self.spdims = layout.shape
        self.layout = layout
        self.block = block
        self.lut, self.maxlut = _softmax.make_lut(self.layout, self.block, device)
        self.is_dense = is_dense

    def __call__(self, a, *, scale=1.0, rel_logits=None, is_causal=False):
        if rel_logits is not None and rel_logits.dtype != a.dtype:
            raise ValueError(f"relative position embedding must be {a.dtype}")
        a = _softmax.apply(a, scale, rel_logits, is_causal, self.spdims, self.block, self.lut, self.maxlut,
                           self.is_dense)
        return a
