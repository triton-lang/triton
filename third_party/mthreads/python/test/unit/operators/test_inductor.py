import pytest
import torch

import triton
import triton.language as tl


def test_normalization_with_remat(device):

    @triton.jit
    def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK: tl.constexpr,
                RBLOCK: tl.constexpr):
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

    buf14 = torch.rand(8, 64, 64, 64, device=device)
    buf16 = torch.rand(8, 1, 64, device=device)
    arg114_1 = torch.rand(64, device=device)
    arg115_1 = torch.rand(64, device=device)
    arg8_1 = torch.rand(64, device=device)
    arg9_1 = torch.rand(64, device=device)
    triton_[(512, )](buf14, buf16, arg114_1, arg115_1, arg8_1, arg9_1, 512, 4096, 1, 2048)
    torch.testing.assert_close(buf16.mean().item(), buf14.mean().item(), atol=1e-7, rtol=0)


def test_avg_pool_bw(device):

    @triton.jit
    def triton_(in_ptr0, out_ptr0, XBLOCK: tl.constexpr):
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        x1 = (xindex // 8) % 8
        x0 = xindex % 8
        x2 = (xindex // 64)
        x5 = xindex
        tmp0 = (-1) + x1
        tmp1 = (-1) + x0
        tmp2 = 2 + x1
        tmp3 = 2 + x0
        tmp4 = 0
        tmp5 = tl.where(tmp0 != tmp0, tmp0, tl.where(tmp0 > tmp4, tmp0, tmp4))
        tmp6 = tl.where(tmp1 != tmp1, tmp1, tl.where(tmp1 > tmp4, tmp1, tmp4))
        tmp7 = 8
        tmp8 = tl.where(tmp2 != tmp2, tmp2, tl.where(tmp2 < tmp7, tmp2, tmp7))
        tmp9 = tl.where(tmp3 != tmp3, tmp3, tl.where(tmp3 < tmp7, tmp3, tmp7))
        tmp10 = tmp5 + tmp4
        tmp11 = tmp6 + tmp4
        tmp12 = 1
        tmp13 = tmp8 - tmp12
        tmp14 = tl.where(tmp10 != tmp10, tmp10, tl.where(tmp10 < tmp13, tmp10, tmp13))
        tmp15 = tmp9 - tmp12
        tmp16 = tl.where(tmp11 != tmp11, tmp11, tl.where(tmp11 < tmp15, tmp11, tmp15))
        tmp17 = tl.load(in_ptr0 + (tmp16 + (8 * tmp14) + (64 * x2)), None).to(tl.float32)
        tmp18 = tmp17 / 9
        tmp19 = tmp10 < tmp8
        tmp20 = tmp11 < tmp9
        tmp21 = tmp19 & tmp20
        tmp22 = 0.0
        tmp23 = tl.where(tmp21, tmp18, tmp22)
        tmp24 = tmp6 + tmp12
        tmp25 = tl.where(tmp24 != tmp24, tmp24, tl.where(tmp24 < tmp15, tmp24, tmp15))
        tmp26 = tl.load(in_ptr0 + (tmp25 + (8 * tmp14) + (64 * x2)), None).to(tl.float32)
        tmp27 = tmp26 / 9
        tmp28 = tmp24 < tmp9
        tmp29 = tmp19 & tmp28
        tmp30 = tmp23 + tmp27
        tmp31 = tl.where(tmp29, tmp30, tmp23)
        tmp32 = 2
        tmp33 = tmp6 + tmp32
        tmp34 = tl.where(tmp33 != tmp33, tmp33, tl.where(tmp33 < tmp15, tmp33, tmp15))
        tmp35 = tl.load(in_ptr0 + (tmp34 + (8 * tmp14) + (64 * x2)), None).to(tl.float32)
        tmp36 = tmp35 / 9
        tmp37 = tmp33 < tmp9
        tmp38 = tmp19 & tmp37
        tmp39 = tmp31 + tmp36
        tmp40 = tl.where(tmp38, tmp39, tmp31)
        tmp41 = tmp5 + tmp12
        tmp42 = tl.where(tmp41 != tmp41, tmp41, tl.where(tmp41 < tmp13, tmp41, tmp13))
        tmp43 = tl.load(in_ptr0 + (tmp16 + (8 * tmp42) + (64 * x2)), None).to(tl.float32)
        tmp44 = tmp43 / 9
        tmp45 = tmp41 < tmp8
        tmp46 = tmp45 & tmp20
        tmp47 = tmp40 + tmp44
        tmp48 = tl.where(tmp46, tmp47, tmp40)
        tmp49 = tl.load(in_ptr0 + (tmp25 + (8 * tmp42) + (64 * x2)), None).to(tl.float32)
        tmp50 = tmp49 / 9
        tmp51 = tmp45 & tmp28
        tmp52 = tmp48 + tmp50
        tmp53 = tl.where(tmp51, tmp52, tmp48)
        tmp54 = tl.load(in_ptr0 + (tmp34 + (8 * tmp42) + (64 * x2)), None).to(tl.float32)
        tmp55 = tmp54 / 9
        tmp56 = tmp45 & tmp37
        tmp57 = tmp53 + tmp55
        tmp58 = tl.where(tmp56, tmp57, tmp53)
        tmp59 = tmp5 + tmp32
        tmp60 = tl.where(tmp59 != tmp59, tmp59, tl.where(tmp59 < tmp13, tmp59, tmp13))
        tmp61 = tl.load(in_ptr0 + (tmp16 + (8 * tmp60) + (64 * x2)), None).to(tl.float32)
        tmp62 = tmp61 / 9
        tmp63 = tmp59 < tmp8
        tmp64 = tmp63 & tmp20
        tmp65 = tmp58 + tmp62
        tmp66 = tl.where(tmp64, tmp65, tmp58)
        tmp67 = tl.load(in_ptr0 + (tmp25 + (8 * tmp60) + (64 * x2)), None).to(tl.float32)
        tmp68 = tmp67 / 9
        tmp69 = tmp63 & tmp28
        tmp70 = tmp66 + tmp68
        tmp71 = tl.where(tmp69, tmp70, tmp66)
        tmp72 = tl.load(in_ptr0 + (tmp34 + (8 * tmp60) + (64 * x2)), None).to(tl.float32)
        tmp73 = tmp72 / 9
        tmp74 = tmp63 & tmp37
        tmp75 = tmp71 + tmp73
        tmp76 = tl.where(tmp74, tmp75, tmp71)
        tl.store(out_ptr0 + (x5 + tl.zeros([XBLOCK], tl.int32)), tmp76, None)

    inp = torch.ones(8, 2048, 8, 8, device=device, dtype=torch.half)
    out = torch.ones_like(inp) * 3
    numel = inp.numel()
    triton_[(numel // 1024, )](inp, out, 1024)
    out_ref = torch.ones_like(inp)
    out_ref[:, :, 1:7, 0::7] = 2 / 3
    out_ref[:, :, 0::7, 1:7] = 2 / 3
    out_ref[:, :, 0::7, 0::7] = 4 / 9
    torch.testing.assert_close(out, out_ref)


@pytest.mark.parametrize("RBLOCK", [1, 16, 32, 64, 128])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_scan2d_broadcast(RBLOCK, num_warps, device):

    @triton.jit(debug=True)
    def fn(in_ptr, out_ptr, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
        rindex = tl.arange(0, RBLOCK)[None, :]
        xindex = tl.arange(0, XBLOCK)[:, None]
        data = tl.load(in_ptr + rindex)
        scan = tl.cumsum(data, 1)
        expected_max = tl.sum(data, 1)
        tl.device_assert(scan <= expected_max)
        tl.store(out_ptr + xindex * RBLOCK + rindex, scan)

    XBLOCK = 4
    input = torch.randint(0, 10, (1, RBLOCK), dtype=torch.int64, device=device)
    output = torch.empty((XBLOCK, RBLOCK), dtype=torch.int64, device=device)
    fn[(1, )](input, output, XBLOCK, RBLOCK, num_warps=num_warps)
    ref = input.cumsum(1).broadcast_to((XBLOCK, RBLOCK))
    torch.testing.assert_close(output, ref)


def test_scan2d_for(device):

    @triton.jit
    def fn(out_ptr0, rnumel, RBLOCK: tl.constexpr):
        rbase = tl.arange(0, RBLOCK)[None, :]
        for roffset in range(0, rnumel, RBLOCK):
            rindex = roffset + rbase
            rmask = rindex < rnumel
            tmp3 = tl.where(rmask, 1, 0)
            tmp6 = tl.cumsum(tmp3, 1)
            tl.store(out_ptr0 + rindex, tmp6, rmask)

    RBLOCK = 8
    out0 = torch.empty(RBLOCK, device=device, dtype=torch.int64)
    fn[(1, )](out0, RBLOCK, RBLOCK)
    ref = torch.arange(RBLOCK, device=device, dtype=torch.int64) + 1
    torch.testing.assert_close(out0, ref)
