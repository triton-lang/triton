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
):

    softmax_group = tl.group(name="fa_bw_softmax", start=0, size=4)
    load_group = tl.group(name="tma_load", start=4, size=4)
    groupA = tl.group(name="groupA", start=8, size=4)
    groupB = tl.group(name="groupB", start=12, size=4)

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
        with load_group:
            k = k_desc_ptr.load(
                [offs_kv, 0],
            )

        with groupA:
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
            p = tl.math.exp2(qk)
            p = p.to(tl.float8e5) if fp8_v else p.to(tl.float16)
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            alpha = tl.math.exp2(m_i - m_ij)
            m_i = m_ij
            l_i = l_i * alpha + l_ij

        with groupA:
            # -- update output accumulator --
            acc = acc * alpha[:, None]

        # update acc
        # with softmax_group:
        with load_group:
            if fp8_v:
                v = v_desc_ptr.load(
                    [offs_hz * HEAD_DIM, start_n],
                )
            else:
                v = v_desc_ptr.load(
                    [offs_kv, 0],
                )

        # with softmax_group:
        with groupB:
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
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
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

    # load q: it will stay in SRAM throughout
    q = q_desc_ptr.load(
        [off_hz * N_CTX + start_m * BLOCK_M, 0],
    )

    # XXX: we can't pass groups yet to other triton function
    # so unfortunatelly we need to redefine duplicate groupB here and in _attn_fwd_inner
    groupB = tl.group(name="groupB", start=12, size=4)

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
        )

    with groupB:
        # epilogue
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def attention(q, k, v, causal, sm_scale, NUM_WARPS):
    return run_attention(
        _attn_fwd,
        q,
        k,
        v,
        causal,
        sm_scale,
        BLOCK_M=128,
        BLOCK_N=64,
        NUM_STAGES=1,
        NUM_WARPS=NUM_WARPS,
        USE_TTG_WS=False,
        WG_SPEC=False,
        MATH_WG_PIPE=False,
        FORCE_MEMBAR=False,
    )


@pytest.mark.parametrize(
    "Z, H, N_CTX, HEAD_DIM, NUM_WARPS",
    [
        (2, 2, 256, 64, 4),
        (2, 2, 512, 64, 4),
        (2, 2, 1024, 64, 4),
        (2, 2, 2048, 64, 4),
        (2, 2, 4096, 64, 4),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_op(Z, H, N_CTX, HEAD_DIM, NUM_WARPS, causal, dtype):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Blackwell attention isn't supported on sm != 10.x")

    torch.manual_seed(20)
    sm_scale = 0.5
    q, k, v = init_tensors(Z, H, N_CTX, HEAD_DIM, dtype)
    # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale, NUM_WARPS)
    # official FA implementation (broken for fp8)
    if supports_hopper() and dtype == torch.float16:
        fa2_out = triton_reference(q, k, v, causal, sm_scale)
        assert torch.allclose(tri_out, fa2_out, atol=1e-2, rtol=1.0 - 3)

    ref_out = torch_reference(q, k, v, causal, sm_scale, N_CTX, dtype)
    # compare
    # assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    # print("ref_out :", ref_out)
    # print("tri_out :", tri_out)
    ERROR_TOLERANCE = 2e-3 if dtype == torch.float16 else 3e-2
    assert_close_verbose(
        tri_out.to(torch.float16),
        ref_out.to(torch.float16),
        rtol=ERROR_TOLERANCE,
        atol=ERROR_TOLERANCE,
    )
    # print("PASS")


BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
HAS_FLASH_BENCH = HAS_FLASH
HAS_FLASH_BENCH = False
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd"]:
    for causal in [False, True]:
        # for provider in ["triton-fp16", "triton-fp8"]:
        for provider in ["triton-fp16"]:
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
        )

    return bench_flash_attention_with_configs(
        bench_fn, BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device
    )


if __name__ == "__main__":
    Z, H, N_CTX, HEAD_DIM, NUM_WARPS = (2, 2, 4096, 64, 4)

    causal = True
    # causal = False

    dtype = torch.float16
    # dtype = torch.float8_e5m2

    test_op(Z, H, N_CTX, HEAD_DIM, NUM_WARPS, causal=causal, dtype=dtype)

    bench_flash_attention.run(save_path=".", print_data=True)
