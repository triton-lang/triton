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
    acc,
    l_i,
    m_i,
    q,  #
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
    softmax_group: tl.constexpr,
    correction_group: tl.constexpr,
):

    # to support correction, group we need to use warpSize= 1 in load/groupA/groupB
    # that would contain only mma's
    # correction_group = tl.group(name="correction", start=16, size=4)
    # range of values handled by this stage

    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offs_kv = offs_hz * N_CTX + lo

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        with ld1_group:
            k = k_desc_ptr.load(
                [offs_kv, 0],
            )

        with mma1_group:
            qk = tl.dot(q, k.T)

        with softmax_group:
            if STAGE == 2:
                # tl.device_print("start_n:", start_n)
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                qk -= m_ij[:, None]
            else:
                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                qk = qk * qk_scale - m_ij[:, None]
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            p = p.to(tl.float8e5) if fp8_v else p.to(tl.float16)
            # -- update m_i and l_i
            m_i = m_ij
            l_i = l_i * alpha + l_ij

        with correction_group:
            # -- update output accumulator --
            acc = acc * alpha[:, None]

        # update acc
        # with softmax_group:
        with ld2_group:
            if fp8_v:
                v = v_desc_ptr.load(
                    [offs_hz * HEAD_DIM, start_n],
                )
            else:
                v = v_desc_ptr.load(
                    [offs_kv, 0],
                )

        # with softmax_group:
        with mma2_group:
            if fp8_v:
                acc = tl.dot(p, v.T, acc)
            else:
                acc = tl.dot(p, v, acc)

        # update m_i and l_i
        offs_kv += BLOCK_N

    return acc, l_i, m_i


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
    num_warps: tl.constexpr,
):
    # tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
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
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    if num_warps == 8:
        mma1_group = tl.group(name="mma1", start=19, size=1, reg_count=24)
        ld1_group = tl.group(name="ld1_group", start=18, size=1, reg_count=24)
        ld2_group = tl.group(name="ld2_group", start=17, size=1, reg_count=24)
        mma2_group = tl.group(name="mma2", start=16, size=1, reg_count=24)
        softmax_group = tl.group(name="softmax", start=0, size=8, reg_count=176)
        correction_group = tl.group(name="correction", start=8, size=8, reg_count=48)
    else:
        mma1_group = tl.group(name="mma1", start=9, size=1)
        ld1_group = tl.group(name="ld1_group", start=10, size=1)
        ld2_group = tl.group(name="ld2_group", start=11, size=1)
        mma2_group = tl.group(name="mma2", start=8, size=1)
        softmax_group = tl.group(name="softmax", start=0, size=4)
        correction_group = tl.group(name="correction", start=4, size=4)

    # load q: it will stay in SRAM throughout
    with ld1_group:
        q = q_desc_ptr.load(
            [off_hz * N_CTX + start_m * BLOCK_M, 0],
        )

    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
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
            softmax_group,
            correction_group,
        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
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
            softmax_group,
            correction_group,
        )

    with softmax_group:
        # epilogue
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        #tl.store(O_block_ptr, acc.to(Out.type.element_ty))
        o_desc_ptr.store(
            [off_hz * N_CTX + start_m * BLOCK_M, 0], acc.to(Out.type.element_ty)
        )


def attention(q, k, v, causal, sm_scale, NUM_WARPS, BLOCK_N):
    if v.dtype == torch.float8_e5m2:
        _, _, HEAD_DIM, _ = v.shape
    else:
        _, _, _, HEAD_DIM = v.shape
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
        USE_TTG_WS=True,
        WG_SPEC=False,
        MATH_WG_PIPE=False,
        FORCE_MEMBAR=False,
    )


f8 = torch.float8_e5m2
f16 = torch.float16
@pytest.mark.parametrize(
    "Z, H, N_CTX, HEAD_DIM, NUM_WARPS, causal, BLOCK_N, dtype",
    [
       #f8 non-causal
       (2, 2, 256, 128, 8, False, 256, f8),
       (2, 2, 512, 128, 8, False, 256, f8),
       (2, 2, 1024, 128, 8, False, 256, f8),
       (2, 2, 2048, 128, 8, False, 256, f8),
       (2, 2, 4096, 128, 8, False, 256, f8),
       #f8 causal, runs out of tmem, can't use BLOCK_N=256
       (2, 2, 256, 128, 8, True, 128, f8),
       (2, 2, 512, 128, 8, True, 128, f8),
       (2, 2, 1024, 128, 8, True, 128, f8),
       (2, 2, 2048, 128, 8, True, 128, f8),
       (2, 2, 4096, 128, 8, True, 128, f8),
      #f16 non-causal, codegen bug, can't use BLOCK_N=256
       (2, 2, 256, 128, 8, False, 128, f16),
       (2, 2, 512, 128, 8, False, 128, f16),
       (2, 2, 1024, 128, 8, False, 128, f16),
       (2, 2, 2048, 128, 8, False, 128, f16),
       (2, 2, 4096, 128, 8, False, 128, f16),
       #f16 causal
       (2, 2, 256, 128, 8, True, 128, f16),
       (2, 2, 512, 128, 8, True, 128, f16),
       (2, 2, 1024, 128, 8, True, 128, f16),
       (2, 2, 2048, 128, 8, True, 128, f16),
       (2, 2, 4096, 128, 8, True, 128, f8),
    ],
)
def test_op(Z, H, N_CTX, HEAD_DIM, NUM_WARPS, causal, dtype, BLOCK_N):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Blackwell attention isn't supported on sm != 10.x")

    torch.manual_seed(20)
    sm_scale = 0.5
    q, k, v = init_tensors(Z, H, N_CTX, HEAD_DIM, dtype)
    # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale, NUM_WARPS, BLOCK_N)
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
    ATOL = 2.0e-3 if dtype == torch.float16 else 2.0 / 8
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
    for causal, HEAD_DIM, BLOCK_N, NUM_WARPS in [(False, 128, 256, 8), (True, 128, 128, 8)]:
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
                        "BLOCK_N": BLOCK_N if provider == "triton-fp8" else 128,
                        "NUM_WARPS": NUM_WARPS if provider == "triton-fp8" else 4,
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
    BLOCK_N,
    NUM_WARPS,
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
            BLOCK_N=BLOCK_N,
        )

    return bench_flash_attention_with_configs(
        bench_fn, BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device
    )


if __name__ == "__main__":
    Z, H, N_CTX, HEAD_DIM, NUM_WARPS = (2, 2, 4096, 128, 8)

    # causal = True
    dtype = torch.float16
    dtype = torch.float8_e5m2

    causal = False
    test_op(Z, H, N_CTX, HEAD_DIM, NUM_WARPS, causal=causal, dtype=dtype, BLOCK_N=256)

    causal = True
    test_op(Z, H, N_CTX, HEAD_DIM, NUM_WARPS, causal=causal, dtype=dtype, BLOCK_N=128)

    bench_flash_attention.run(save_path=".", print_data=True)
