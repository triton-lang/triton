"""
Fused Attention with TMA
========================


"""

try:  # for pytest
    from .fmha_common import *
except:  # for benchmark
    from fmha_common import *

import pytest
import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attn_fwd_inner(
    acc0,
    acc1,
    l_i_0,
    l_i_1,
    m_i_0,
    m_i_1,
    q0,  #
    q1,  #
    k_desc_ptr,
    v_desc_ptr,  #
    start_m,
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    offs_hz: tl.constexpr,
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
    ld1_group: tl.constexpr,
    ld2_group: tl.constexpr,
    mma1_group: tl.constexpr,
    mma2_group: tl.constexpr,
    sm1_group: tl.constexpr,
    sm2_group: tl.constexpr,
    corr1_group: tl.constexpr,
    corr2_group: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX

    offs_kv0 = (offs_hz * 2) * N_CTX + lo
    offs_kv1 = (offs_hz * 2 + 1) * N_CTX + lo

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        with ld1_group:
            k0 = k_desc_ptr.load([offs_kv0, 0],)
            if fp8_v:
                v0 = v_desc_ptr.load([(offs_hz * 2) * HEAD_DIM, start_n],)
            else:
                v0 = v_desc_ptr.load([offs_kv0, 0],)
        with ld2_group:
            k1 = k_desc_ptr.load([offs_kv1, 0],)
            if fp8_v:
                v1 = v_desc_ptr.load([(offs_hz * 2 + 1) * HEAD_DIM, start_n],)
            else:
                v1 = v_desc_ptr.load([offs_kv1, 0],)
            
        with mma1_group:
            qk0 = tl.dot(q0, k0.T)
            
        with sm1_group:
            if STAGE == 2:
                # tl.device_print("start_n:", start_n)
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk0 = qk0 * qk_scale + tl.where(mask, 0, -1.0e6)
                m_ij_0 = tl.maximum(m_i_0, tl.max(qk0, 1))
                qk0 -= m_ij_0[:, None]
            else:
                m_ij_0 = tl.maximum(m_i_0, tl.max(qk0, 1) * qk_scale)
                qk0 = qk0 * qk_scale - m_ij_0[:, None]
            alpha0 = tl.math.exp2(m_i_0 - m_ij_0)
            p0 = tl.math.exp2(qk0)
            l_ij_0 = tl.sum(p0, 1)
            p0 = p0.to(tl.float8e5) if fp8_v else p0.to(tl.float16)
            # -- update m_i and l_i
            m_i_0 = m_ij_0
            l_i_0 = l_i_0 * alpha0 + l_ij_0
            
        with corr1_group:
            # -- update output accumulator --
            acc0 = acc0 * alpha0[:, None]
            
        with mma2_group:
            if fp8_v:
                acc0 = tl.dot(p0, v0.T, acc0)
            else:
                acc0 = tl.dot(p0, v0, acc0)
            
        with mma1_group:
            qk1 = tl.dot(q1, k1.T)

        with sm2_group:
            if STAGE == 2:
                # tl.device_print("start_n:", start_n)
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk1 = qk1 * qk_scale + tl.where(mask, 0, -1.0e6)
                m_ij_1 = tl.maximum(m_i_1, tl.max(qk1, 1))
                qk1 -= m_ij_1[:, None]
            else:
                m_ij_1 = tl.maximum(m_i_1, tl.max(qk1, 1) * qk_scale)
                qk1 = qk1 * qk_scale - m_ij_1[:, None]
            alpha1 = tl.math.exp2(m_i_1 - m_ij_1)
            p1 = tl.math.exp2(qk1)
            l_ij_1 = tl.sum(p1, 1)
            p1 = p1.to(tl.float8e5) if fp8_v else p1.to(tl.float16)
            # -- update m_i and l_i
            m_i_1 = m_ij_1
            l_i_1 = l_i_1 * alpha1 + l_ij_1

        with corr1_group:
            # -- update output accumulator --
            acc1 = acc1 * alpha1[:, None]
            
        with mma2_group:
            if fp8_v:
                acc1 = tl.dot(p1, v1.T, acc1)
            else:
                acc1 = tl.dot(p1, v1, acc1)

        # update m_i and l_i
        offs_kv0 += BLOCK_N
        offs_kv1 += BLOCK_N

    return acc0, acc1, l_i_0, l_i_1, m_i_0, m_i_1


# @triton.autotune(configs_tma, key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    q_desc_ptr,
    k_desc_ptr,
    v_desc_ptr,
    o_desc_ptr,
    m_desc_ptr,
    sm_scale,
    M,
    Out,  #
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    Z,
    H,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = (off_hz * 2) // H
    off_h_0 = (off_hz * 2) % H
    off_h_1 = (off_hz * 2 + 1) % H
    qvk_offset_0 = off_z.to(tl.int64) * stride_qz + off_h_0.to(tl.int64) * stride_qh
    qvk_offset_1 = off_z.to(tl.int64) * stride_qz + off_h_1.to(tl.int64) * stride_qh

    O_block_ptr_0 = tl.make_block_ptr(
        base=Out + qvk_offset_0,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    O_block_ptr_1 = tl.make_block_ptr(
        base=Out + qvk_offset_1,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i_0 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i_1 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i_0 = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    l_i_1 = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc_0 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    mma1_group = tl.group(name="mma1", start=12, size=1, reg_count=40)
    ld1_group = tl.group(name="ld1_group", start=13, size=1, reg_count=40)
    ld2_group = tl.group(name="ld2_group", start=14, size=1, reg_count=40)
    mma2_group = tl.group(name="mma2", start=15, size=1, reg_count=40)
    sm1_group = tl.group(name="sm1", start=0, size=4, reg_count=192)
    sm2_group = tl.group(name="sm2", start=4, size=4, reg_count=192)
    corr1_group = tl.group(name="correction", start=8, size=4, reg_count=88)
    #corr2_group = tl.group(name="corr2", start=12, size=4)

    # load q: it will stay in SRAM throughout
    with ld1_group:
        q_0 = q_desc_ptr.load(
            [off_hz * 2 * N_CTX + start_m * BLOCK_M, 0],
        )
    with ld2_group:
        q_1 = q_desc_ptr.load(
            [(off_hz * 2 + 1) * N_CTX + start_m * BLOCK_M, 0],
        )

    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc_0, acc_1, l_i_0, l_i_1, m_i_0, m_i_1 = _attn_fwd_inner(
            acc_0,
            acc_1,
            l_i_0,
            l_i_1,
            m_i_0,
            m_i_1,
            q_0,
            q_1,
            k_desc_ptr,
            v_desc_ptr,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            off_hz,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,  #
            ld1_group,
            ld2_group,
            mma1_group,
            mma2_group,
            sm1_group,
            sm2_group,
            corr1_group,
            None, #corr2_group,
        )
    # stage 2: on-band
    if STAGE & 2:
        acc_0, acc_1, l_i_0, l_i_1, m_i_0, m_i_1 = _attn_fwd_inner(
            acc_0,
            acc_1,
            l_i_0,
            l_i_1,
            m_i_0,
            m_i_1,
            q_0,
            q_1,
            k_desc_ptr,
            v_desc_ptr,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            off_hz,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,  #
            ld1_group,
            ld2_group,
            mma1_group,
            mma2_group,
            sm1_group,
            sm2_group,
            corr1_group,
            None, #corr2_group,
        )

    # epilogue
    with sm1_group:
        m_ptrs_0 = M + (off_hz * 2) * N_CTX + offs_m
        m_i_0 += tl.math.log2(l_i_0)
        acc_0 = acc_0 / l_i_0[:, None]
        tl.store(m_ptrs_0, m_i_0)
        
        acc0= tl.reshape(acc_0, (BLOCK_M, 2, HEAD_DIM // 2))
        acc0= tl.permute(acc0, (0, 2, 1))
        acc0A, acc0B = tl.split(acc0)
        
        o_desc_ptr.store(
            [off_hz * 2 * N_CTX + start_m * BLOCK_M, 0],
            acc0A.to(Out.type.element_ty),
        )
        o_desc_ptr.store(
           [off_hz * 2 * N_CTX + start_m * BLOCK_M, HEAD_DIM//2],
           acc0B.to(Out.type.element_ty),
        )


    with sm2_group:
        m_ptrs_1 = M + (off_hz * 2 + 1) * N_CTX + offs_m
        m_i_1 += tl.math.log2(l_i_1)
        acc_1 = acc_1 / l_i_1[:, None]
        tl.store(m_ptrs_1, m_i_1)
        
        acc1= tl.reshape(acc_1, (BLOCK_M, 2, HEAD_DIM // 2))
        acc1= tl.permute(acc1, (0, 2, 1))
        acc1A, acc1B = tl.split(acc1)
        o_desc_ptr.store(
            [(off_hz * 2 + 1) * N_CTX + start_m * BLOCK_M, 0],
            acc1A.to(Out.type.element_ty),
        )
        o_desc_ptr.store(
           [(off_hz * 2 + 1) * N_CTX + start_m * BLOCK_M, HEAD_DIM//2],
           acc1B.to(Out.type.element_ty),
        )


def attention(q, k, v, causal, sm_scale, NUM_WARPS, USE_TTG_WS):
    if v.dtype == torch.float8_e5m2:
        Z, H, HEAD_DIM, _ = v.shape
    else:
        Z, H, _, HEAD_DIM = v.shape
    assert Z*H%2 == 0, "Z*H must be even"
    BLOCK_N = min(128, HEAD_DIM)
    return run_attention(
        _attn_fwd,
        q,
        k,
        v,
        causal,
        sm_scale,
        BLOCK_M=128,
        BLOCK_N=BLOCK_N,
        NUM_STAGES=1,
        NUM_WARPS=NUM_WARPS,
        USE_TTG_WS=USE_TTG_WS,
        WG_SPEC=False,
        MATH_WG_PIPE=False,
        FORCE_MEMBAR=False,
        NUM_BLOCK_M=2,
        Osplit=2,
    )


@pytest.mark.parametrize(
    "Z, H, N_CTX, HEAD_DIM, NUM_WARPS, causal, USE_TTG_WS",
    [
        (2, 2, 256, 128, 4, False, True),
        (2, 2, 512, 128, 4, False, False),
        (2, 2, 1024, 128, 4, False, True),
        (2, 2, 2048, 128, 4, False, False),
        (2, 2, 4096, 128, 4, False, True),
        (2, 2, 256, 32, 4, True, True),
        (2, 2, 512, 32, 4, True, False),
        (2, 2, 1024, 32, 4, True, True),
        (2, 2, 2048, 32, 4, True, False),
        (2, 2, 4096, 32, 4, True, True),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float8_e5m2])
def test_op(Z, H, N_CTX, HEAD_DIM, NUM_WARPS, causal, dtype, USE_TTG_WS):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Blackwell attention isn't supported on sm != 10.x")

    torch.manual_seed(20)
    sm_scale = 0.5
    q, k, v = init_tensors(Z, H, N_CTX, HEAD_DIM, dtype)
    # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale, NUM_WARPS, USE_TTG_WS)
    # official FA implementation (broken for fp8)
    if supports_hopper() and dtype == torch.float16:
        fa2_out = triton_reference(q, k, v, causal, sm_scale)
        assert torch.allclose(tri_out, fa2_out, atol=1e-2, rtol=1.0 - 3)

    ref_out = torch_reference(q, k, v, causal, sm_scale, N_CTX, dtype)
    # compare
    # assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    # print("ref_out :", ref_out)
    # print("tri_out :", tri_out)
    RTOL = 2e-3 if dtype == torch.float16 else 3e-2
    ATOL = 2.0e-3 if dtype == torch.float16 else 1.0/8
    assert_close_verbose(
        tri_out.to(torch.float16),
        ref_out.to(torch.float16),
        rtol=RTOL,
        atol=ATOL,
    )
    # print("PASS")


BATCH, N_HEADS = 4, 32
HAS_FLASH_BENCH = HAS_FLASH
HAS_FLASH_BENCH = False
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd"]:
    for causal, HEAD_DIM in [(False, 128)]:
        for provider in ["triton-fp16", "triton-fp8"]:
            if mode == "bwd" and not causal:
                continue
            configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    x_vals=[2**i for i in range(10, 15)],
                    line_arg="provider",
                    line_vals=[f"{provider}"] + (["flash"] if HAS_FLASH_BENCH else []),
                    line_names=[f"{provider}"]
                    + (
                        ["Flash-3"]
                        if HAS_FLASH_BENCH and supports_hopper()
                        else ["Flash-2"] if HAS_FLASH_BENCH else []
                    ),
                    styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                    ylabel="ms",
                    plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{provider}-{mode}-causal={causal}",
                    args={
                        "H": N_HEADS,
                        "BATCH": BATCH,
                        "HEAD_DIM": HEAD_DIM,
                        "mode": mode,
                        "causal": causal,
                        "USE_TTG_WS": False,
                    },
                )
            )


@triton.testing.perf_report(configs)
def bench_flash_attention(
    BATCH,
    H,
    N_CTX,
    HEAD_DIM,
    causal,
    mode,
    provider,
    USE_TTG_WS,
    NUM_WARPS=4,
    device="cuda",
):
    def bench_fn(q, k, v, causal, sm_scale):
        return attention(
            q,
            k,
            v,
            causal,
            sm_scale,
            NUM_WARPS=NUM_WARPS,
            USE_TTG_WS=USE_TTG_WS,
        )

    return bench_flash_attention_with_configs(
        bench_fn, BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device
    )


if __name__ == "__main__":
    Z, H, N_CTX, HEAD_DIM, NUM_WARPS = (8, 2, 4096, 128, 4)

    # causal = True
    dtype = torch.float16
    dtype = torch.float8_e5m2
    use_ttg_ws = True
    use_ttg_ws = False

    causal = False
    test_op(Z, H, N_CTX, HEAD_DIM, NUM_WARPS, causal=causal, dtype=dtype, USE_TTG_WS=use_ttg_ws)

    causal = True
    #test_op(Z, H, N_CTX, HEAD_DIM, NUM_WARPS, causal=causal, dtype=dtype, USE_TTG_WS=use_ttg_ws)

    bench_flash_attention.run(save_path=".", print_data=True)
