import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q, K, V,
    TMP, L, M,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kk, stride_kn,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_qm = tl.program_id(0)
    off_hz = tl.program_id(1)
    # initialize offsets
    offs_m = start_qm * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = off_hz * stride_qh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    off_v = off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    t_ptrs = TMP + off_hz * N_CTX + offs_m

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    q = tl.load(q_ptrs)
    for start_n in range(0, start_qm + 1):
        # -- compute qk ----
        k = tl.load(k_ptrs)
        qk = tl.dot(q, k)
        qk += tl.where(offs_m[:, None] >= (start_n * BLOCK_N + offs_n[None, :]), 0, float("-inf"))
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        p = p.to(tl.float16)
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        tl.store(t_ptrs, acc_scale)
        acc_scale = tl.load(t_ptrs)  # BUG: have to store and immediately load
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(v_ptrs)
        acc += tl.dot(p, v)
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        # r_ptrs += BLOCK_N
        l_i = l_i_new
        m_i = m_i_new

    start_qm = tl.program_id(0)
    offs_m = start_qm * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)
    # initialize pointers to output
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_out = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_ptrs = Out + off_out
    tl.store(out_ptrs, acc)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v):
        BLOCK = 128
        # shape constraints
        Lq, Lk = q.shape[-1], k.shape[-2]
        assert Lq == Lk
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1])
        tmp = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _fwd_kernel[grid](
            q, k, v,
            tmp, L, m,
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=64, num_warps=4,
            num_stages=1,
        )
        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.BLOCK = BLOCK
        ctx.grid = grid
        return o


attention = _attention.apply


@pytest.mark.parametrize('Z, H, N_CTX, D_MODEL', [(2, 3, 1024, 64)])
def test_op(Z, H, N_CTX, D_MODEL, dtype=torch.float16):
    torch.manual_seed(20)
    q = .5 * torch.randn((Z, H, N_CTX, D_MODEL), dtype=dtype, device="cuda", requires_grad=True)
    k = .5 * torch.randn((Z, H, D_MODEL, N_CTX), dtype=dtype, device="cuda", requires_grad=True)
    v = .5 * torch.randn((Z, H, N_CTX, D_MODEL), dtype=dtype, device="cuda", requires_grad=True)
    # triton implementation
    tri_out = attention(q, k, v)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    ref_qk = torch.matmul(q, k)
    for z in range(Z):
        for h in range(H):
            ref_qk[:, :, M == 0] = float("-inf")
    ref_qk = torch.softmax(ref_qk, dim=-1)
    ref_out = torch.matmul(ref_qk, v)
    # compare
    triton.testing.assert_almost_equal(ref_out, tri_out)


try:
    from flash_attn.flash_attn_interface import flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
# vary batch size for fixed heads / seq
batch_bench = triton.testing.Benchmark(
    x_names=['BATCH'],
    x_vals=[2**i for i in range(0, 8)],
    line_arg='provider',
    line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
    line_names=['Triton'] + (['Flash'] if HAS_FLASH else []),
    styles=[('red', '-'), ('blue', '-')],
    ylabel='ms',
    plot_name=f'fused-attention-seq{N_CTX}-head{N_HEADS}-d{D_HEAD}',
    args={'H': N_HEADS, 'N_CTX': N_CTX, 'D_MODEL': D_HEAD, 'dtype': torch.float16}
)
# vary seq length for fixed head and batch=4
seq_bench = triton.testing.Benchmark(
    x_names=['N_CTX'],
    x_vals=[2**i for i in range(10, 16)],
    line_arg='provider',
    line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
    line_names=['Triton'] + (['Flash'] if HAS_FLASH else []),
    styles=[('red', '-'), ('blue', '-')],
    ylabel='ms',
    plot_name=f'fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}',
    args={'H': D_HEAD, 'BATCH': BATCH, 'D_MODEL': D_HEAD, 'dtype': torch.float16}
)


@triton.testing.perf_report([batch_bench, seq_bench])
def bench_flash_attention(BATCH, H, N_CTX, D_MODEL, provider, dtype=torch.float16, device="cuda"):
    warmup = 25
    rep = 500
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_MODEL), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, D_MODEL, N_CTX), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_MODEL), dtype=dtype, device="cuda", requires_grad=True)
        fn = lambda: attention(q, k, v)
        ms = triton.testing.do_bench(fn, percentiles=None, warmup=warmup, rep=rep)
        return ms
    if provider == "flash":
        lengths = torch.full((BATCH,), fill_value=N_CTX, device=device)
        cu_seqlens = torch.zeros((BATCH + 1,), device=device, dtype=torch.int32)
        cu_seqlens[1:] = lengths.cumsum(0)
        qkv = torch.randn((BATCH * N_CTX, 3, H, D_MODEL), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, cu_seqlens, 0., N_CTX, causal=True)
        ms = triton.testing.do_bench(fn, percentiles=None, warmup=warmup, rep=rep)
        return ms


bench_flash_attention.run(save_path='.', print_data=True)
