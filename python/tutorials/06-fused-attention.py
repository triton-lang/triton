import torch
import triton
import triton.language as tl
import pytest

# fmt: off

# ********************************************************

@triton.jit
def _kernel(
    Q, K, V,
    TMP, stride_hzt, #NOTE: scratchpad buffer to workaround a compiler bug
    Out, 
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kk, stride_kn,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, 
    BLOCK_M: tl.constexpr, BLOCK_DMODEL : tl.constexpr, 
    BLOCK_N: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    start_am = tl.program_id(1)
    start_bn = 0
    # initialize pointers to Q
    offs_qm = start_am * BLOCK_M + tl.arange(0, BLOCK_M) 
    offs_qk = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (off_z * stride_qz \
                + off_h * stride_qh \
                + offs_qm[:, None] * stride_qm \
                + offs_qk[None, :] * stride_qk)
    # initialize pointers to K
    offs_kk = tl.arange(0, BLOCK_DMODEL)
    offs_kn = start_bn * BLOCK_N + tl.arange(0, BLOCK_N)
    k_ptrs = K + (off_z * stride_kz \
                + off_h * stride_kh \
                + offs_kn[None, :] * stride_kn \
                + offs_kk[:, None] * stride_kk)
    # initialize pointers to V
    off_vk = tl.arange(0, BLOCK_N)
    off_vn = tl.arange(0, BLOCK_DMODEL)
    v_ptrs = V +  off_z * stride_vz \
                + off_h * stride_vh \
                + off_vk[:, None] * stride_vk \
                + off_vn[None, :] * stride_vn
    # initialize pointer to workaround scratchpad
    t_ptrs = TMP + off_hz*stride_hzt + offs_qm
    # compute acc, m_i, l_i
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    num_blocks_for_row = start_am + 1
    for start_n in range(0, num_blocks_for_row):
        q = tl.load(q_ptrs) # BUG: fails when moved out of the loop
        v = tl.load(v_ptrs)
        # -- compute qk ----
        k = tl.load(k_ptrs)
        qk = tl.dot(q, k)
        qk = tl.where(offs_qm[:, None] >= (start_n*BLOCK_N + offs_kn[None, :]), qk, float("-inf"))
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        l_i_new = tl.exp(m_i - m_i_new) * l_i + tl.exp(m_ij - m_i_new) * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = tl.exp(m_ij - m_i_new) / l_i_new
        p = p * p_scale[:, None]
        p = p.to(tl.float16)
        # scale acc
        acc_scale = l_i / l_i_new * tl.exp(m_i - m_i_new)
        tl.store(t_ptrs, acc_scale)
        acc_scale = tl.load(t_ptrs) # BUG: have to store and immediately load
        acc = acc * acc_scale[:, None]
        # update acc
        acc += tl.dot(p, v)
        k_ptrs += BLOCK_N*stride_kn
        v_ptrs += BLOCK_N*stride_vk
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    offs_om = offs_qm
    offs_on = tl.arange(0, BLOCK_DMODEL)
    out_ptrs = Out + off_z * stride_oz \
                 + off_h * stride_oh \
                 + offs_om[:, None] * stride_om \
                 + offs_on[None, :] * stride_on
    # write-back
    tl.store(out_ptrs, acc)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v):
        BLOCK = 128
        D_MODEL = q.shape[-1]
        assert D_MODEL == 64
        # shape constraints
        Lq, Lk = q.shape[-1], k.shape[-2]
        assert Lq == Lk
        o = torch.empty_like(q)
        grid = (q.shape[0]*q.shape[1], triton.cdiv(q.shape[2], BLOCK))
        tmp = torch.empty((q.shape[0]*q.shape[1], q.shape[2]), device=q.device, dtype=torch.float16)
        _kernel[grid](
            q, k, v,
            tmp, tmp.stride(0), o, 
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1],
            BLOCK_M=BLOCK, BLOCK_N=BLOCK, 
            BLOCK_DMODEL = D_MODEL, num_warps=4,
        )
        return o
    
attention = _attention.apply

@pytest.mark.parametrize('Z, H, N_CTX, D_MODEL', [(2, 3, 1024, 64)])
def test_op(Z, H, N_CTX, D_MODEL, dtype=torch.float16):
    torch.manual_seed(20)
    q = .5*torch.randn((Z, H, N_CTX, D_MODEL), dtype=dtype, device="cuda", requires_grad=True)
    k = .5*torch.randn((Z, H, D_MODEL, N_CTX), dtype=dtype, device="cuda", requires_grad=True)
    v = .5*torch.randn((Z, H, N_CTX, D_MODEL), dtype=dtype, device="cuda", requires_grad=True)
    # triton implementation
    tri_out = attention(q, k, v)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    ref_qk = torch.matmul(q, k)
    for z in range(Z):
        for h in range(H):
            ref_qk[:,:,M==0] = float("-inf")
    ref_qk = torch.softmax(ref_qk, dim=-1)
    ref_out = torch.matmul(ref_qk, v)
    # compare
    triton.testing.assert_almost_equal(ref_out, tri_out)


benchmarks = [
    (64, 16, 1024, 64)
]

@pytest.mark.parametrize('Z, H, N_CTX, D_MODEL', benchmarks)
def test_perf(Z, H, N_CTX, D_MODEL, dtype=torch.float16):
    q = torch.randn((Z, H, N_CTX, D_MODEL), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((Z, H, D_MODEL, N_CTX), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn((Z, H, N_CTX, D_MODEL), dtype=dtype, device="cuda", requires_grad=True)
    fn = lambda: attention(q, k, v)
    ms = triton.testing.do_bench(fn, percentiles=None, warmup=25, rep=500)
    print(ms)