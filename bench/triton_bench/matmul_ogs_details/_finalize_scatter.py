import triton
import triton.language as tl
from triton_bench.numerics_details.flexpoint import float_to_flex, load_scale, update_scale


@tl.constexpr_function
def is_hip():
    from triton_bench import target_info
    return target_info.is_hip()


def _finalize_scatter_launch_metadata(grid, kernel, args):
    ret = dict()
    Out, A, ScatterSrcIndx, K, N, EXPT_PER_TOK = [
        args[name] for name in ["Out", "A", "ScatterSrcIndx", "K", "N", "EXPT_PER_TOK"]
    ]
    ret["name"] = f"{kernel.name} [{N=} {K=} {EXPT_PER_TOK=}]"

    is_active = (ScatterSrcIndx != -1).view((-1, EXPT_PER_TOK))
    need_accum = is_active.sum(dim=1) >= (1 if K > 1 else 2)
    is_active &= need_accum[:, None]
    active_input_rows = is_active.sum()
    active_output_rows = need_accum.sum()

    ret["bytes"] = (active_input_rows * K * A.shape[-1] * A.element_size() +
                    active_output_rows * Out.shape[-1] * Out.element_size() +
                    ScatterSrcIndx.numel() * ScatterSrcIndx.element_size())
    nbits = Out.dtype.itemsize * 8
    ret[f"flops{nbits}"] = active_input_rows * K * A.shape[-1]
    return ret


@triton.jit
def _thread_local_absmax(x, BLOCK_SIZE: tl.constexpr, NUM_THREADS: tl.constexpr):
    return tl.max(tl.reshape(tl.abs(x), [NUM_THREADS, BLOCK_SIZE // NUM_THREADS], can_reorder=True), axis=1)


# Copied from _gather_expert_fwd with some modification.
@triton.jit(launch_metadata=_finalize_scatter_launch_metadata)
def _finalize_scatter(
    Out,
    OutExpectedScale,
    OutActualScale,
    OutChecksumScale,
    A,
    stride_a_k,
    stride_a_m,
    AScale,
    ScatterSrcIndx,
    K: tl.constexpr,
    M,
    N,
    NTokens,
    EXPT_PER_TOK: tl.constexpr,
    flexpoint_saturate_inf: tl.constexpr,
    BLOCK_N: tl.constexpr,
    M_BLOCKS,
    HAS_FUSED_SCRATCHPAD: tl.constexpr,
):
    # TODO: M_BLOCKS, N_BLOCKS, could be tl.num_programs(axis=0).
    # But, that builtin is very slow on AMD. Remove when the compiler is fixed.
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)

    if HAS_FUSED_SCRATCHPAD:
        M = M  # remove constexpr
        A += M.to(tl.int64) * stride_a_m  # need to read from the scratchpad region.

    THREADS_PER_BLOCK: tl.constexpr = tl.extra.cuda.num_threads()
    local_max = tl.full([THREADS_PER_BLOCK], 0.0, tl.float32)
    scale_a = load_scale(AScale)

    # Offset pointers to the starting row
    ScatterSrcIndxPtr = ScatterSrcIndx + pid_m * EXPT_PER_TOK
    OutPtr = Out + pid_m * N + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    for row in range(pid_m, M, M_BLOCKS):
        out = tl.zeros([BLOCK_N], dtype=tl.float32)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        tl.static_assert(EXPT_PER_TOK > 1)
        scatter_src_idxs = tl.load(ScatterSrcIndxPtr + tl.arange(0, EXPT_PER_TOK))

        is_src_active = (scatter_src_idxs != -1)
        # number of actually active experts
        n_active_experts = tl.sum(is_src_active.to(tl.int32))

        # Set inactive rows to 0
        if n_active_experts == 0:
            tl.store(OutPtr, tl.zeros([BLOCK_N], dtype=Out.dtype.element_ty), mask=n_mask)

        if HAS_FUSED_SCRATCHPAD:
            # Rows with single active expert are already written to the correct
            # location, so there's nothing to do.
            need_accum = (n_active_experts > 1)
        else:
            need_accum = (n_active_experts >= 1)

        if K > 1 and NTokens is not None:
            need_accum = need_accum and row < tl.load(NTokens)

        if need_accum:
            # Load from active rows of A and accumulate into out.
            # NOTE: important to zero-out masked values here as all values
            # (including masked) are incorporated into the flexpoint max.
            # When conversion to the out type is relatively cheap, we load blocks of A,
            # convert, and do a sum reduction. This also requires K to be 1 to preserve
            # the order in which the accumulation is done.
            # Otherwise, we'll load and accumulate active rows of A individually.
            do_sum_reduction = (
                K == 1 and
                (tl.constexpr(A.dtype.element_ty == tl.float32) or
                 (tl.constexpr(A.dtype.element_ty == tl.float16) or tl.constexpr(A.dtype.element_ty == tl.bfloat16)))
            ) and not is_hip()  # This path performs slower on AMD
            if do_sum_reduction:
                if n_active_experts == EXPT_PER_TOK:  # no mask on is_src_active needed
                    APtrs = A + scatter_src_idxs[:, None].to(tl.int64) * stride_a_m + offs_n[None, :]
                    for _ in range(K):
                        out += tl.sum(tl.load(APtrs, mask=n_mask[None, :], other=0.0).to(tl.float32), axis=0)
                        APtrs += stride_a_k
                else:
                    APtrs = A + scatter_src_idxs[:, None].to(tl.int64) * stride_a_m + offs_n[None, :]
                    mask = is_src_active[:, None] & n_mask[None, :]
                    for _ in range(K):
                        out += tl.sum(tl.load(APtrs, mask=mask, other=0.0).to(tl.float32), axis=0)
                        APtrs += stride_a_k
            else:
                for i in tl.static_range(0, EXPT_PER_TOK, 1):
                    src_idx = tl.load(ScatterSrcIndxPtr + i)
                    if src_idx != -1:
                        APtrs = A + src_idx.to(tl.int64) * stride_a_m + offs_n
                        for _ in range(K):
                            out += tl.load(APtrs, mask=n_mask, other=0.0)
                            APtrs += stride_a_k

            out = out * scale_a

            if OutActualScale is not None:
                absmax = _thread_local_absmax(out, BLOCK_N, THREADS_PER_BLOCK)
                local_max = tl.maximum(local_max, absmax)
            out = float_to_flex(out, OutExpectedScale, None, OutChecksumScale, n_mask, Out, flexpoint_saturate_inf)
            tl.store(OutPtr, out, mask=n_mask)
            out = out.to(tl.float32)

        ScatterSrcIndxPtr += M_BLOCKS * EXPT_PER_TOK
        OutPtr += M_BLOCKS * N

    update_scale(local_max, OutActualScale, Out)
