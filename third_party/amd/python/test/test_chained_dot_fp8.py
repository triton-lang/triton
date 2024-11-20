"""
Testing the (FP8) case of a dot op that consumes the output (MFMA) of
another dot op as an input.

"""

import math
import pytest
import sys
import torch

import triton
import triton.language as tl

TORCH_HAS_FP8E4 = hasattr(torch, 'float8_e4m3fnuz')
float8: tl.constexpr = None if not TORCH_HAS_FP8E4 else torch.float8_e4m3fnuz
torch.manual_seed(42)


@triton.jit
def _chained_dot(
    Q,
    K,
    V,
    Out,
    q_desc,
    k_desc,
    v_desc,
    s_sc,
    s_desc,
    o_sc,
    stride_qz,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_kn,
    stride_kd,
    stride_vz,
    stride_vd,
    stride_vn,
    stride_oz,
    stride_om,
    stride_od,
    Z,
    M,
    N,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_FP8: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)
    qkv_offset = off_z * stride_qz
    Q_block_ptr = tl.make_block_ptr(base=Q + qkv_offset, shape=(N, BLOCK_D), strides=(stride_qm, stride_qd),
                                    offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_D), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + qkv_offset, shape=(BLOCK_D, N), strides=(stride_kd, stride_kn),
                                    offsets=(0, 0), block_shape=(BLOCK_D, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + qkv_offset, shape=(N, BLOCK_D), strides=(stride_vn, stride_vd),
                                    offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_D), order=(0, 1))

    s_scale = q_desc * k_desc * s_sc
    acc_scale = s_desc * v_desc * o_sc

    q = tl.load(Q_block_ptr)

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    lo, hi = 0, N
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = tl.load(K_block_ptr)
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k)

        if USE_FP8:
            s *= s_scale

        v = tl.load(V_block_ptr)
        acc += tl.dot(s.to(v.dtype), v)

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    if USE_FP8:
        acc *= acc_scale

    O_block_ptr = tl.make_block_ptr(base=Out + qkv_offset, shape=(N, BLOCK_D), strides=(stride_om, stride_od),
                                    offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_D), order=(1, 0))
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


class chained_dot_fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, msize=32, q_desc=1.0, k_desc=1.0, v_desc=1.0, s_sc=1.0, s_desc=1.0, o_sc=1.0):
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-2]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        assert msize in {16, 32}
        o = torch.empty_like(q, dtype=v.dtype)

        BLOCK_M = 128 if q.dtype == float8 else 256
        if BLOCK_M > q.shape[1]:
            BLOCK_M = int(math.pow(2, math.floor(math.log2(q.shape[1]))))
        BLOCK_N = 32
        if BLOCK_N > k.shape[1]:
            BLOCK_N = int(math.pow(2, math.floor(math.log2(k.shape[1]))))
        waves_per_eu = 2
        num_warps = 4 if q.dtype == float8 else 8
        num_stages = 1

        grid = (triton.cdiv(q.shape[1], BLOCK_M), q.shape[0], 1)

        _chained_dot[grid](q, k, v, o, q_desc,
                           k_desc, v_desc, s_sc, s_desc, o_sc, q.stride(0), q.stride(1), q.stride(2), k.stride(0),
                           k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1),
                           o.stride(2), Z=q.shape[0], M=q.shape[1], N=k.shape[1], BLOCK_D=Lk, BLOCK_M=BLOCK_M,
                           BLOCK_N=BLOCK_N, USE_FP8=(q.dtype == float8), waves_per_eu=waves_per_eu, num_warps=num_warps,
                           num_stages=num_stages, matrix_instr_nonkdim=msize)

        return o


chained_dot = chained_dot_fn.apply


def to_float8(x, dtype=float8, margin: float = 1.0):
    finfo = torch.finfo(dtype)
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    scale = math.pow(2, math.floor(math.log2(scale.float().item())) - margin)
    x_scaled = (x.float() * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scaled.to(dtype), scale, 1.0 / scale


@pytest.mark.parametrize('M, N, D, dtype, msize', [(*shape, dtype, msize)
                                                   for shape in [(128, 64, 32), (256, 128, 128)]
                                                   for dtype in ['fp8']
                                                   for msize in [16, 32]])
def test_chained_dot(M, N, D, dtype, msize):
    if dtype == 'fp8':
        assert float8 is not None

    BATCH = 1
    q = torch.empty((BATCH, M, D), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    k = torch.empty((BATCH, N, D), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    v = torch.empty((BATCH, D, N), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)

    if dtype == 'fp8':
        q_f8, _, q_desc = to_float8(q)
        k_f8, _, k_desc = to_float8(k)
        v_f8, _, v_desc = to_float8(v)

        s = torch._scaled_mm(q_f8[0], k_f8[0].transpose(0, 1), out_dtype=torch.float32,
                             scale_a=torch.tensor(q_desc, dtype=torch.float32, device="cuda"),
                             scale_b=torch.tensor(k_desc, dtype=torch.float32, device="cuda"))
        s_f8, s_sc, s_desc = to_float8(s)
        ref = torch._scaled_mm(s_f8, v_f8[0].transpose(0, 1), out_dtype=torch.float32,
                               scale_a=torch.tensor(s_desc, dtype=torch.float32, device="cuda"),
                               scale_b=torch.tensor(v_desc, dtype=torch.float32, device="cuda"))
        ref_f8, ref_sc, _ = to_float8(ref)

        tri_out = chained_dot(q_f8, k_f8, v_f8, msize, q_desc, k_desc, v_desc, s_sc, s_desc, ref_sc)

        assert tri_out.isnan().sum() == 0
        torch.testing.assert_close(tri_out[0].float(), ref_f8.float(), atol=1e-2, rtol=0)

    else:
        s = torch.matmul(q, k.transpose(1, 2))
        ref = torch.matmul(s, v.transpose(1, 2))

        tri_out = chained_dot(q, k, v, msize)
        torch.testing.assert_close(tri_out, ref, atol=1e-2, rtol=0)
