"""
Fused Attention with TMA
========================


"""

import pytest
import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


configs_tma = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=s, num_warps=w)
    for s in [2, 3]
    for w in [4, 8]
]


def supports_hopper():
    return False
    return torch.cuda.get_device_capability()[0] >= 9


try:
    if supports_hopper():
        print("Hopper kernel")
        from flash_attn_interface import (
            _flash_attn_forward as flash_attn_func,
        )
    else:
        print("Ampere kernel")
        from flash_attn.flash_attn_interface import (
            flash_attn_qkvpacked_func as flash_attn_func,
        )
    HAS_FLASH = True
except BaseException:
    print("no flash attn")
    HAS_FLASH = False


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
        k = k_desc_ptr.load(
            [offs_kv, 0],
        )
        qk = tl.dot(q, k.T)
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
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        if fp8_v:
            v = v_desc_ptr.load(
                [offs_hz * HEAD_DIM, start_n],
            )
            p = p.to(tl.float8e5)
            acc = tl.dot(p, v.T, acc)
        else:
            v = v_desc_ptr.load(
                [offs_kv, 0],
            )
            p = p.to(tl.float16)
            acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
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
    # tl.inline_asm_elementwise(
    #     "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
    #     "=r, l",
    #     [q_desc_ptr],
    #     dtype=tl.int32,
    #     is_pure=False,
    #     pack=1,
    # )
    # tl.inline_asm_elementwise(
    #     "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
    #     "=r, l",
    #     [k_desc_ptr],
    #     dtype=tl.int32,
    #     is_pure=False,
    #     pack=1,
    # )
    # tl.inline_asm_elementwise(
    #     "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
    #     "=r, l",
    #     [v_desc_ptr],
    #     dtype=tl.int32,
    #     is_pure=False,
    #     pack=1,
    # )
    # """
    # tl.inline_asm_elementwise(
    #     "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
    #     "=r, l",
    #     [o_desc_ptr],
    #     dtype=tl.int32,
    #     is_pure=False,
    #     pack=1,
    # )
    # tl.inline_asm_elementwise(
    #     "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
    #     "=r, l",
    #     [m_desc_ptr],
    #     dtype=tl.int32,
    #     is_pure=False,
    #     pack=1,
    # )
    # """
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
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    """
    tl._experimental_descriptor_store(
        m_desc_ptr,
        m_i,
        [off_hz * N_CTX + start_m * BLOCK_M],
    )
    tl._experimental_descriptor_store(
        o_desc_ptr,
        acc.to(Out.type.element_ty),
        [off_hz * N_CTX + start_m * BLOCK_M, 0],
    )
    """
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


# Cached fwd TMA descriptors
cached_desc_q = None
cached_desc_k = None
cached_desc_v = None
cached_desc_o = None
cached_desc_m = None
cached_args = None


def get_fwd_tma_descriptors(*args):
    global cached_desc_q, cached_desc_k, cached_desc_v, cached_desc_o, cached_desc_m, cached_args
    # FIXME: args != cached_args fails
    if True: # or args != cached_args:
        cached_args = args
        # reuse cached tma descriptors if input matches
        (
            q,
            k,
            v,
            o,
            m,
            Z,
            H,
            N_CTX,
            HEAD_DIM,
            BLOCK_M,
            BLOCK_N,
            qkvo_element_size,
            m_element_size,
            fp8_v,
        ) = args
        cached_desc_q = TensorDescriptor(
            q,
            [Z * H * N_CTX, HEAD_DIM],
            [HEAD_DIM, 1],
            [BLOCK_M, HEAD_DIM],
        )
        cached_desc_k = TensorDescriptor(
            k,
            [Z * H * N_CTX, HEAD_DIM],
            [HEAD_DIM, 1],
            [BLOCK_N, HEAD_DIM],
        )
        if fp8_v:
            cached_desc_v = TensorDescriptor(
                    v,
                    [Z * H * HEAD_DIM, N_CTX],
                    [N_CTX, 1],
                    [HEAD_DIM, BLOCK_N]
                )
        else:
            cached_desc_v = TensorDescriptor(
                    v,
                    [Z * H * N_CTX, HEAD_DIM],
                    [HEAD_DIM, 1],
                    [BLOCK_N, HEAD_DIM]
                )
        cached_desc_o = TensorDescriptor(
            o,
            [Z * H * N_CTX, HEAD_DIM],
            [HEAD_DIM, 1],
            [BLOCK_M, HEAD_DIM]
        )
        cached_desc_m = TensorDescriptor(
            m,
            [Z * H * N_CTX,],
            [1,],
            [BLOCK_M,]
        )
    return cached_desc_q, cached_desc_k, cached_desc_v, cached_desc_o, cached_desc_m


def attention(q, k, v, causal, sm_scale, USE_TTG_WS, WG_SPEC, MATH_WG_PIPE, NUM_WARPS, FORCE_MEMBAR):
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    if v.dtype == torch.float8_e5m2:
        HEAD_DIM_V = v.shape[-2]
    else:
        HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    o = torch.empty_like(q)
    stage = 3 if causal else 1
    dtype = k.dtype

    grid = lambda args: (
        triton.cdiv(q.shape[2], args["BLOCK_M"]),
        q.shape[0] * q.shape[1],
        1,
    )
    M = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )
    # K: (Z, H, N_CTX, HEAD_DIM)
    Z, H, N_CTX = k.shape[:3]

    BLOCK_M, BLOCK_N = 128, 128
    # FIXME: should test num_stages=1
    NUM_STAGES = 2

    # Note: When using ttg.warp_specialized, more shared memory (smem) is allocated.
    # To avoid running out of smem, we reduce NUM_STAGES by 1.
    if USE_TTG_WS:
        NUM_STAGES = max(1, NUM_STAGES - 1)

    Z, H, N_CTX = k.shape[:3]
    desc_q, desc_k, desc_v, desc_o, desc_m = get_fwd_tma_descriptors(
        q,
        k,
        v,
        o,
        M,
        Z,
        H,
        N_CTX,
        HEAD_DIM_Q,
        BLOCK_M,
        BLOCK_N,
        q.element_size(),
        M.element_size(),
        v.dtype == torch.float8_e5m2,
    )
    if WG_SPEC == "mma_first":
        WG_SPEC = (("tma_load", NUM_WARPS,4), ("mma", 0,NUM_WARPS))
    elif WG_SPEC == "tma_load_first":
        WG_SPEC = (("tma_load", 0,4), ("mma", 4,NUM_WARPS))
    else:
        WG_SPEC = ()
    pgm = _attn_fwd[grid](
        q,
        k,
        v,  #
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        desc_m,  #
        sm_scale,
        M,
        o,  #
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),  #
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),  #
        q.shape[0],
        q.shape[1],  #
        N_CTX=q.shape[2],  #
        HEAD_DIM=HEAD_DIM_K,  #
        STAGE=stage,  #
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=NUM_STAGES,
        num_warps=NUM_WARPS,
        mma_depth=1,
        enable_warp_specialization=True,
        math_wg_pipe=MATH_WG_PIPE,
        use_ttg_ws=USE_TTG_WS,
        wg_spec_override=WG_SPEC,
        force_membar=FORCE_MEMBAR,
    )
    #print(pgm.asm['ptx'])

    return o

def assert_close_verbose(actual, expected, rtol=2e-3, atol=2e-3, max_mismatches=5):
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        print("PASS")
    except AssertionError as e:
        print("Tensors are not close.")
        diff = torch.abs(actual - expected)
        mismatch_mask = diff > (atol + rtol * torch.abs(expected))

        mismatches = torch.nonzero(mismatch_mask, as_tuple=False)
        num_mismatches = mismatches.shape[0]

        if num_mismatches == 0:
            print("No mismatches above tolerance, but torch.testing.assert_close still failed.")
            return

        print(f"Total mismatches: {num_mismatches}")
        top_diff = diff[mismatch_mask].flatten()
        top_k = min(max_mismatches, top_diff.numel())
        top_k_vals, top_k_indices = torch.topk(top_diff, top_k)

        for i in range(top_k):
            flat_index = top_k_indices[i]
            mismatch_idx = mismatches[flat_index]
            act_val = actual[tuple(mismatch_idx.tolist())]
            exp_val = expected[tuple(mismatch_idx.tolist())]
            print(f"Mismatch at {tuple(mismatch_idx.tolist())}: actual={act_val}, expected={exp_val}, diff={top_k_vals[i]}")


@pytest.mark.parametrize(
    "Z, H, N_CTX, HEAD_DIM, WG_SPEC",
    [
        (2, 2, 1024, 128, "tma_load_first"),
        (2, 2, 2048, 128, "mma_first"),
        (2, 2, 4096, 128, None),
        (2, 2, 8192, 128, "mma_first"),
        (2, 2, 16384, 128, None),
    ],
)
@pytest.mark.parametrize("math_wg_pipe", [False, True])
@pytest.mark.parametrize("causal", [False, True])
# Disable USE_TTG_WS until proper integration is done
# @pytest.mark.parametrize("USE_TTG_WS", [False, True])
@pytest.mark.parametrize("USE_TTG_WS", [False])
def test_op(Z, H, N_CTX, HEAD_DIM, WG_SPEC, math_wg_pipe, causal, USE_TTG_WS, dtype=torch.float16):
    if torch.cuda.get_device_capability()[0] >= 10:
        if causal == True:
            pytest.skip("causal attention isn't supported on Blackwell yet")
        # math_wg_pipe=True isn't supported on Blackwell yet
        if math_wg_pipe == True:
            pytest.skip("math wg pipelining isn't supported on Blackwell")
        # on blackwell there is a race in 8 warps mode
        # we will first make 8 warps to work for matmul then look at FMHA
        NUM_WARPS = 4
        MATH_WG_PIPE = False
    elif torch.cuda.get_device_capability()[0] >= 9:
        # FIXME: math_wg_pipe=False hangs on Hopper
        if math_wg_pipe == False:
            pytest.skip("turning off math wg pipelining hangs on Hopper")
        MATH_WG_PIPE = math_wg_pipe
        NUM_WARPS = 8
    else:
        pytest.skip("causal attention isn't supported on sm <= 9.0")

    torch.manual_seed(20)
    sm_scale = 0.5
    if dtype == torch.float8_e5m2:
        q = (
            torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=torch.float16, device="cuda")
            .normal_(mean=0.0, std=0.5)
            .requires_grad_()
            .to(dtype)
        )
        k = (
            torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=torch.float16, device="cuda")
            .normal_(mean=0.0, std=0.5)
            .requires_grad_()
            .to(dtype)
        )
        v = (
            torch.empty((Z, H, HEAD_DIM, N_CTX), dtype=torch.float16, device="cuda")
            .normal_(mean=0.0, std=0.5)
            .requires_grad_()
            .to(dtype)
        )
    else:
        q = (
            torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
            .normal_(mean=0.0, std=0.5)
            .requires_grad_()
        )
        k = (
            torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
            .normal_(mean=0.0, std=0.5)
            .requires_grad_()
        )
        v = (
            torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
            .normal_(mean=0.0, std=0.5)
            .requires_grad_()
        )
    # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale, USE_TTG_WS, WG_SPEC, MATH_WG_PIPE, NUM_WARPS, FORCE_MEMBAR=False)
    # official FA implementation (broken for fp8)
    if supports_hopper() and dtype == torch.float16:
        q_nhd = q.transpose(1, 2).contiguous()
        k_nhd = k.transpose(1, 2).contiguous()
        v_nhd = v.transpose(1, 2).contiguous()
        fa2_out = (
            flash_attn_func(q_nhd, k_nhd, v_nhd, sm_scale, causal)[0]
            .transpose(1, 2)
            .contiguous()
        )
        assert torch.allclose(tri_out, fa2_out, atol=1e-2, rtol=1.0 - 3)
    # reference implementation
    if dtype == torch.float8_e5m2:
        v = v.transpose(2, 3).contiguous()
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    # compare
    # assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    # print("ref_out :", ref_out)
    # print("tri_out :", tri_out)
    ERROR_TOLERANCE = 2e-3 if dtype == torch.float16 else 3e-2
    if 0:
        torch.testing.assert_close(
            tri_out.to(torch.float16),
            ref_out.to(torch.float16),
            rtol=ERROR_TOLERANCE,
            atol=ERROR_TOLERANCE,
    )
    else:
        assert_close_verbose(
            tri_out.to(torch.float16),
            ref_out.to(torch.float16),
            rtol=ERROR_TOLERANCE,
            atol=ERROR_TOLERANCE,
        )
    #print("PASS")


BATCH, N_HEADS, HEAD_DIM = 4, 32, 128
HAS_FLASH_BENCH = HAS_FLASH
HAS_FLASH_BENCH = False
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd"]:
    for causal in [False, True]:
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
    USE_TTG_WS=False,
    MATH_WG_PIPE=True,
    NUM_WARPS=8,
    WG_SPEC="mma_first",
    FORCE_MEMBAR=False,
    device="cuda",
):
    print(f"BENCH: {BATCH=}, {H=}, {N_CTX=}, {HEAD_DIM=}, {causal=}, {mode=}")
    global desc_k, desc_v
    desc_k = None
    desc_v = None
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 1000
    dtype = torch.float16
    if "triton" in provider:
        if mode == "fwd" and "fp8" in provider:
            v_shape = (BATCH, H, HEAD_DIM, N_CTX)
        else:
            v_shape = (BATCH, H, N_CTX, HEAD_DIM)
        q = torch.randn(
            (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
        )
        k = torch.randn(
            (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
        )
        v = torch.randn(v_shape, dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale, USE_TTG_WS, WG_SPEC, MATH_WG_PIPE, NUM_WARPS, FORCE_MEMBAR)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        if supports_hopper():
            q = torch.randn(
                (BATCH, N_CTX, H, HEAD_DIM),
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
            k = torch.randn(
                (BATCH, N_CTX, H, HEAD_DIM),
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
            v = torch.randn(
                (BATCH, N_CTX, H, HEAD_DIM),
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
            sm_scale = 1.3
            fn = lambda: flash_attn_func(q, k, v, sm_scale, causal)
        else:
            qkv = torch.randn(
                (BATCH, N_CTX, 3, H, HEAD_DIM),
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
            fn = lambda: flash_attn_func(qkv, causal=causal)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


if __name__ == "__main__":
    Z, H, N_CTX, HEAD_DIM = (2, 2, 16384, 128)
    wg_spec = ()
    #wg_spec = "mma_first"
    #wg_spec = "tma_load_first"
    use_ttg_ws = False
    force_membar = False
    math_wg_pipe = True
    #use_ttg_ws = True
    #force_membar = True  # when using with ttng.wg perf is closer to ttg.ws=true
    if 1:
        test_op(
            Z, H, N_CTX, HEAD_DIM, USE_TTG_WS=use_ttg_ws, WG_SPEC=wg_spec, math_wg_pipe=math_wg_pipe, causal=False, dtype=torch.float16
        )
    else:
        test_op(
            Z, H, N_CTX, HEAD_DIM, USE_TTG_WS=use_ttg_ws, WG_SPEC=wg_spec, math_wg_pipe=math_wg_pipe, causal=False, dtype=torch.float8_e5m2
        )

    bench_flash_attention.run(USE_TTG_WS=use_ttg_ws, WG_SPEC=wg_spec, FORCE_MEMBAR=force_membar, save_path=".", print_data=True)
