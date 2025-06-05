import triton
from triton.tools.tensor_descriptor import TensorDescriptor
import torch


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
    if True:  # or args != cached_args:
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
                v, [Z * H * HEAD_DIM, N_CTX], [N_CTX, 1], [HEAD_DIM, BLOCK_N]
            )
        else:
            cached_desc_v = TensorDescriptor(
                v, [Z * H * N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_N, HEAD_DIM]
            )
        cached_desc_o = TensorDescriptor(
            o, [Z * H * N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_M, HEAD_DIM]
        )
        cached_desc_m = TensorDescriptor(
            m,
            [
                Z * H * N_CTX,
            ],
            [
                1,
            ],
            [
                BLOCK_M,
            ],
        )
    return cached_desc_q, cached_desc_k, cached_desc_v, cached_desc_o, cached_desc_m


def myprint(what, x):
    print(what)
    if 0:
        print(x)
        return
    m, n = x.shape
    for i in range(0, m):
        print(f"{i:3}: {x[i,:]}")


def assert_close_verbose(actual, expected, rtol=2e-3, atol=2e-3, max_mismatches=5):
    if 0:
        print(f"actual:")
        myprint("actual00", actual[0, 0, :, :])
        print("expected")
        myprint("expected00", expected[0, 0, :, :])

        print(f"actual:")
        myprint("actual10", actual[1, 0, :, :])
        print("expected")
        myprint("expected10", expected[1, 0, :, :])

        print(f"actual:")
        myprint("actual11", actual[1, 1, :, :])
        print("expected")
        myprint("expected11", expected[1, 1, :, :])

        print(f"actual:")
        myprint("actual01", actual[0, 1, :, :])
        print("expected")
        myprint("expected01", expected[0, 1, :, :])
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
            print(
                "No mismatches above tolerance, but torch.testing.assert_close still failed."
            )
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
            print(
                f"Mismatch at {tuple(mismatch_idx.tolist())}: actual={act_val}, expected={exp_val}, diff={top_k_vals[i]}"
            )
        print("FAIL")
        assert False


def init_tensors(Z, H, N_CTX, HEAD_DIM, dtype, sm_scale):
    torch.manual_seed(20)
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
    return (
        q,
        k,
        v,
    )


def run_attention(
    attn_kernel,
    q,
    k,
    v,
    causal,
    sm_scale,
    BLOCK_M,
    BLOCK_N,
    NUM_STAGES,
    NUM_WARPS,
    USE_TTG_WS,
    WG_SPEC,
    MATH_WG_PIPE,
    FORCE_MEMBAR,
):
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
        WG_SPEC = (("tma_load", NUM_WARPS, 4), ("mma", 0, NUM_WARPS))
    elif WG_SPEC == "tma_load_first":
        WG_SPEC = (("tma_load", 0, 4), ("mma", 4, NUM_WARPS))
    else:
        WG_SPEC = ()
    pgm = attn_kernel[grid](
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
    # print(pgm.asm['ptx'])

    return o


def init_tensors(Z, H, N_CTX, HEAD_DIM, dtype):
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
    return q, k, v


def bench_flash_attention_with_configs(
    bench_fn,
    BATCH,
    H,
    N_CTX,
    HEAD_DIM,
    causal,
    mode,
    provider,
    device="cuda",
):

    def run():
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
                (BATCH, H, N_CTX, HEAD_DIM),
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
            k = torch.randn(
                (BATCH, H, N_CTX, HEAD_DIM),
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
            v = torch.randn(v_shape, dtype=dtype, device=device, requires_grad=True)
            if mode == "fwd" and "fp8" in provider:
                q = q.to(torch.float8_e5m2)
                k = k.to(torch.float8_e5m2)
                v = v.to(torch.float8_e5m2)
            sm_scale = 1.3
            fn = lambda: bench_fn(q, k, v, causal, sm_scale)
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

    return run()


def triton_reference(q, k, v, causal, sm_scale):
    # official FA implementation (broken for fp8)
    q_nhd = q.transpose(1, 2).contiguous()
    k_nhd = k.transpose(1, 2).contiguous()
    v_nhd = v.transpose(1, 2).contiguous()
    fa2_out = (
        flash_attn_func(q_nhd, k_nhd, v_nhd, sm_scale, causal)[0]
        .transpose(1, 2)
        .contiguous()
    )
    return fa2_out


def torch_reference(q, k, v, causal, sm_scale, N_CTX, dtype):
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

    return ref_out
