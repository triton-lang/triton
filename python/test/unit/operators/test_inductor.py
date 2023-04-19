import torch

import triton
import triton.language as tl


def test_normalization_with_remat():

    @triton.jit
    def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
        xnumel = 512
        rnumel = 4096
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        rbase = tl.arange(0, RBLOCK)[None, :]
        x3 = xindex
        x0 = xindex % 64
        tmp1 = tl.load(in_ptr0 + (x0), xmask)
        tmp3 = tl.load(in_ptr1 + (x0), xmask)
        tmp11 = tl.load(in_ptr2 + (x0), xmask)
        tmp13 = tl.load(in_ptr3 + (x0), xmask)
        _tmp17 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
        for roffset in range(0, rnumel, RBLOCK):
            rindex = roffset + rbase
            rmask = rindex < rnumel
            r2 = rindex
            tmp0 = tl.load(in_out_ptr0 + (r2 + (4096 * x3)), rmask & xmask, eviction_policy='evict_last', other=0)
            tmp2 = tmp0 - tmp1
            tmp4 = 1e-05
            tmp5 = tmp3 + tmp4
            tmp6 = tl.sqrt(tmp5)
            tmp7 = 1 / tmp6
            tmp8 = 1.0
            tmp9 = tmp7 * tmp8
            tmp10 = tmp2 * tmp9
            tmp12 = tmp10 * tmp11
            tmp14 = tmp12 + tmp13
            _tmp17 = tl.where(rmask & xmask, _tmp17 + tmp14, _tmp17)
            tl.store(in_out_ptr0 + (r2 + (4096 * x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp14, rmask & xmask)
        tmp17 = tl.sum(_tmp17, 1)[:, None]
        tmp18 = 4096.0
        tmp19 = tmp17 / tmp18
        tl.store(in_out_ptr1 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp19, xmask)

    torch.manual_seed(123)

    buf14 = torch.rand(8, 64, 64, 64, device="cuda")
    buf16 = torch.rand(8, 1, 64, device="cuda")
    arg114_1 = torch.rand(64, device="cuda")
    arg115_1 = torch.rand(64, device="cuda")
    arg8_1 = torch.rand(64, device="cuda")
    arg9_1 = torch.rand(64, device="cuda")
    triton_[(512,)](buf14, buf16, arg114_1, arg115_1, arg8_1, arg9_1, 512, 4096, 1, 2048)
    torch.testing.assert_allclose(buf16.mean().item(), buf14.mean().item(), atol=1e-7, rtol=0)
