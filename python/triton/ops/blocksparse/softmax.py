import triton
import torch
import os

fwd_src = triton.read(os.path.join(os.path.dirname(__file__), 'softmax.c'), kernel_names=['forward'])
fwd_kernels = dict()

bwd_src = triton.read(os.path.join(os.path.dirname(__file__), 'softmax.c'), kernel_names=['backward'])
bwd_kernels = dict()

class _softmax(torch.autograd.Function):
    @staticmethod
    def next_power_of_2(n):
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n += 1
        return n

    @staticmethod
    def make_lut(layout, block, device):
        _empty = torch.tensor([], dtype=torch.int64, device=layout.device)
        sizes = _empty.clone()
        # sizes along rows
        for h in range(layout.shape[0]):
            sizes = torch.cat((sizes, layout[h, :, :].sum(-1)))
        # offsets in block format
        offsets = torch.zeros_like(sizes)
        offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
        # block indices
        idx = torch.arange(layout.sum())
        head = layout.nonzero(as_tuple=False)[:, 0]
        rows = layout.nonzero(as_tuple=False)[:, 1]
        columns = layout.nonzero(as_tuple=False)[:, 2]
        core = torch.stack((idx, columns, rows, head), dim=1).view(-1)
        # construct look-up table
        offsets = offsets * 4 + 2 * sizes.numel()
        header = torch.stack((sizes, offsets), dim=1).view(-1)
        lut = torch.cat((header, core)).type(torch.int32).to(device)
        return lut, int(sizes.max())

    @staticmethod
    def make_kernel(cache, src, max_k, device, dtype, block, apply_scale, apply_rpe, apply_kp_mask, apply_attn_mask,
                    kp_mask_mode, attn_mask_mode):
        if max_k >= 32768:
            raise NotImplementedError('Reductions larger than 32768 elements '\
                                      'are not yet implemented')
        num_warps = 4 if max_k < 512 else (8 if max_k < 2048 else 16)
        TN = _softmax.next_power_of_2(max_k)
        # just-in-time compile kernel
        key = (block, device, dtype, num_warps, TN, apply_scale, apply_rpe, apply_kp_mask, apply_attn_mask,
               kp_mask_mode, attn_mask_mode)
        if key not in cache:
            defines = {
                'TM': 1, 'TN': TN, 'TYPE': dtype, 'BLOCK': block, 'INFINITY':
                {torch.float32: 'F32_INFINITY', torch.float16: 'F16_INFINITY'}[dtype]
            }
            if apply_scale:
                defines['APPLY_SCALE'] = True
            if apply_rpe:
                defines['APPLY_RPE'] = True
            if apply_kp_mask:
                defines['APPLY_KP_MASK'] = True
                if kp_mask_mode == 'mul':
                    defines['KP_MASK_MUL'] = True
            if apply_attn_mask:
                defines['APPLY_ATTN_MASK'] = True
                if attn_mask_mode == 'mul':
                    defines['ATTN_MASK_MUL'] = True
            kernel = triton.kernel(src, device=device, defines=defines, num_warps=num_warps)
            cache[key] = kernel
        return cache[key]

    @staticmethod
    def forward(ctx, x, scale, rpe, key_padding_mask, attn_mask, kp_mask_mode, attn_mask_mode, spdims, block, lut,
                maxlut, bench, time):
        apply_scale = False if scale == 1.0 else True

        # handle None rpe
        if rpe is None:
            apply_rpe = False
            stride_zrpe, stride_hrpe, stride_srpe = 0, 0, 0
            rpe = torch.empty(0, dtype=x.dtype, device=x.device)
        else:
            apply_rpe = True
            stride_zrpe, stride_hrpe, stride_srpe = rpe.stride(0), rpe.stride(1), rpe.stride(2)

        # handle None key_padding_mask
        if key_padding_mask is None:
            apply_kp_mask = False
            stride_zkpm = 0
            key_padding_mask = torch.empty(0, dtype=x.dtype, device=x.device)
        else:
            apply_kp_mask = True
            stride_zkpm = key_padding_mask.stride(0)

        # handle None attention_mask
        if attn_mask is None:
            apply_attn_mask = False
            stride_zattnm = 0
            attn_mask = torch.empty(0, dtype=x.dtype, device=x.device)
        else:
            apply_attn_mask = True
            stride_zattnm = attn_mask.stride(0)

        # run kernel
        kernel = _softmax.make_kernel(fwd_kernels, fwd_src, maxlut * block, x.device, x.dtype, block, apply_scale,
                                      apply_rpe, apply_kp_mask, apply_attn_mask, kp_mask_mode, attn_mask_mode)
        M = x.shape[0]
        grid = lambda opt: [spdims[0] * spdims[1] * block, M]

        # run kernel
        kernel(x.data_ptr(),
               scale,
               lut.data_ptr(),
               rpe.data_ptr(),
               key_padding_mask.data_ptr(),
               attn_mask.data_ptr(),
               maxlut,
               x.stride(0),
               stride_zrpe,
               stride_hrpe,
               stride_srpe,
               stride_zkpm,
               stride_zattnm,
               grid=grid)
        # save to context
        ctx.mark_dirty(x)
        ctx.save_for_backward(x, lut)
        ctx.spdims = spdims
        ctx.block = block
        ctx.maxlut = maxlut
        ctx.scale = scale
        ctx.apply_scale = apply_scale
        ctx.apply_rpe = apply_rpe
        ctx.apply_kp_mask = apply_kp_mask
        ctx.apply_attn_mask = apply_attn_mask
        ctx.kp_mask_mode = kp_mask_mode
        ctx.attn_mask_mode = attn_mask_mode
        return x

    @staticmethod
    def backward(ctx, dx):
        # retrieve from context
        x, lut = ctx.saved_tensors
        # run kernel
        kernel = _softmax.make_kernel(bwd_kernels, bwd_src, ctx.maxlut * ctx.block, x.device, x.dtype, ctx.block,
                                      ctx.apply_scale, ctx.apply_rpe, ctx.apply_kp_mask, ctx.apply_attn_mask,
                                      ctx.kp_mask_mode, ctx.attn_mask_mode)
        M = x.shape[0]
        grid = lambda opt: [ctx.spdims[0] * ctx.spdims[1] * ctx.block, M]
        kernel(x.data_ptr(), ctx.scale, dx.data_ptr(), lut.data_ptr(), ctx.maxlut, x.stride(0), dx.stride(0), grid=grid)
        return dx, None, None, None, None, None, None, None, None, None, None, None, None, None, None

class softmax:

    apply_softmax = _softmax.apply

    def make_lut(self, device):
        key = (device, )
        if key not in self.lut_cache:
            self.lut_cache[key] = _softmax.make_lut(self.layout, self.block, device)
        return self.lut_cache[key]

    def __init__(self, layout, block, bench=False):
        self.spdims = layout.shape
        self.layout = layout
        self.block = block
        self.bench = bench
        self.lut_cache = dict()

    def __call__(self,
                 x,
                 scale=1.,
                 rpe=None,
                 key_padding_mask=None,
                 attn_mask=None,
                 key_padding_mask_mode='add',
                 attn_mask_mode='add'):
        time_y = [None]
        if rpe is not None and rpe.dtype != x.dtype:
            raise ValueError('relative position embedding must be %s' % x.dtype)
        if attn_mask is not None and attn_mask.dtype != x.dtype:
            raise ValueError('Attention mask must be %s' % x.dtype)
        if key_padding_mask is not None and key_padding_mask.dtype != x.dtype:
            raise ValueError('Key padding mask must be %s' % x.dtype)
        lut, maxlut = self.make_lut(x.device)
        x = softmax.apply_softmax(x, scale, rpe, key_padding_mask, attn_mask, key_padding_mask_mode, attn_mask_mode,
                                  self.spdims, self.block, lut, maxlut, self.bench, time_y)
        return x