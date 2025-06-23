import triton
import triton.language as tl
from triton_kernels.numerics_details.flexpoint import float_to_flex, load_scale, update_scale
from triton_kernels.target_info import cuda_capability_geq as _cuda_capability_geq
from triton_kernels.target_info import is_hip as _is_hip


@tl.constexpr_function
def is_hip():
    return _is_hip()


@tl.constexpr_function
def cuda_capability_geq(x, y):
    return _cuda_capability_geq(x, y)


@tl.constexpr_function
def log2(n):
    return len(bin(n)) - 3


@tl.constexpr_function
def _permute_to_end_order(n: int, axis: int):
    """
    Returns the order of the axes of a tensor to permute `axis` to the end.
    """
    order = tuple(range(n))
    return order[:axis] + order[(axis + 1):] + (axis, )


@triton.jit
def permute_to_end(x, axis: tl.constexpr):
    """
    Permutes `x` so that `axis` is the last axis.
    """
    N: tl.constexpr = len(x.shape)
    return tl.permute(x, _permute_to_end_order(N, axis).value)


@triton.jit
def split_n(x, N: tl.constexpr):
    """
    Given `x`, a tensor of shape AxB...x2x2...x2, split it N times.
    Return a tuple of the results.
    """
    xs = (x, )
    for i in tl.static_range(N):
        next = tl.split(xs[0])
        for j in tl.static_range(2**i - 1):
            next = next + tl.split(xs[j + 1])
        xs = next
    return xs


@triton.jit
def thread_local_absmax(x, BLOCK_SIZE: tl.constexpr = None, NUM_THREADS: tl.constexpr = None):
    N: tl.constexpr = tl.extra.cuda.num_threads() if NUM_THREADS is None else NUM_THREADS
    BS: tl.constexpr = x.numel if BLOCK_SIZE is None else BLOCK_SIZE
    tl.static_assert(BS % N == 0, "BLOCK_SIZE must be divisible by NUM_THREADS")
    return tl.max(tl.reshape(tl.abs(x), [N, BS // N], can_reorder=True), axis=1)


def _finalize_matmul_launch_metadata(grid, kernel, args):
    ret = dict()
    Out, A, ScatterSrcIndx, FinalizeScatterIdxs, K, M, N, EXPT_PER_TOK, NumRows = [
        args[name]
        for name in ["Out", "A", "ScatterSrcIndx", "FinalizeScatterIdxs", "K", "M", "N", "EXPT_PER_TOK", "NumRows"]
    ]
    ret["name"] = f"{kernel.name} [M={M}x{EXPT_PER_TOK} {N=} {K=}]"

    if FinalizeScatterIdxs is not None:
        M = FinalizeScatterIdxs[-1].item()

    if ScatterSrcIndx is not None:
        is_active = (ScatterSrcIndx != -1).view((-1, EXPT_PER_TOK))
        n_active = is_active.sum(dim=1)
        need_accum = n_active >= (1 if K > 1 else 2)
        is_active &= need_accum[:, None]
        active_input_rows = is_active.sum()
        active_output_rows = need_accum.sum()
        if EXPT_PER_TOK > 1:
            # Masked rows are set to zero.
            active_output_rows += (n_active == 0).sum()
    else:
        if NumRows is not None:
            if isinstance(NumRows, int):
                active_input_rows = NumRows
            else:
                active_input_rows = NumRows.item()
        else:
            active_input_rows = M
        active_output_rows = M

    ret["bytes"] = (active_input_rows * K * A.shape[-1] * A.element_size() +
                    active_output_rows * Out.shape[-1] * Out.element_size())
    if FinalizeScatterIdxs is not None:
        ret["bytes"] += FinalizeScatterIdxs.numel() * FinalizeScatterIdxs.element_size()
    elif ScatterSrcIndx is not None and EXPT_PER_TOK > 1:
        ret["bytes"] += ScatterSrcIndx.numel() * ScatterSrcIndx.element_size()
    nbits = Out.dtype.itemsize * 8
    ret[f"flops{nbits}"] = active_input_rows * K * A.shape[-1]
    return ret


@tl.constexpr_function
def _accumulate_f16_into_f32_and_track_absmax_ptx(n_inputs: int, src_type: str, absmax_reg_name: str | None):
    """
    Generate PTX code to take fp16 inputs and sum them into an f32 accumulator using mixed-precision
    adds. If `absmax_reg_name` is provided, the absolute maximum value seen so far is tracked inside
    that register.

    Generates code something like:

    add.f32.f16 $0, $2, $1;
    add.f32.f16 $0, $3, $0;
    add.f32.f16 $0, $4, $0;
    add.f32.f16 $0, $5, $0;

    .reg .f32 b;
    abs.f32 b, $0;
    max.f32 my_abs_max, my_abs_max, b;
    """
    # Add the first f16 value to the input $1, store into the output $0.
    ptx = f"\nadd.f32.{src_type} $0, $2, $1;"
    # Accumulate the rest of the inputs into the output $0.
    for i in range(1, n_inputs):
        ptx += f"\nadd.f32.{src_type} $0, ${2 + i}, $0;"
    if absmax_reg_name is not None:
        # Update `absmax_reg_name` with the absolute maximum value seen so far.
        ptx += f"""
        .reg .f32 b;
        abs.f32 b, $0;
        max.f32 {absmax_reg_name}, {absmax_reg_name}, b;
        """
    # Return the PTX snippet, brace-enclosed so we don't pollute the global namespace.
    return f"{{{ptx}}}"


@triton.jit
def _mixed_precision_accumulate_and_track_absmax(acc, x, axis: tl.constexpr, absmax_reg_name: tl.constexpr = None):
    """Given an fp8/bf16/fp16 tensor, accumulate into `acc` along `axis`.
    Values are first converted to bf16/fp16, packed into 32-bit registers, and then accumulated using
    mixed-precision adds.

    If `absmax_reg_name` is provided, the absolute maximum value seen so far is tracked inside that
    register.
    """
    REDUCTION_SIZE: tl.constexpr = x.shape[axis]
    tl.static_assert(2**log2(REDUCTION_SIZE) == REDUCTION_SIZE,
                     f"Reduction size must be a power of 2, was {REDUCTION_SIZE}")
    # move `axis` to the last axis and reshape for iterative splitting.
    x = permute_to_end(x, axis)
    x = tl.reshape(x, x.shape[:-1] + (2, ) * log2(REDUCTION_SIZE))
    # Split into a tuple of AxB tensors.
    xs = split_n(x, log2(REDUCTION_SIZE))
    if (tl.constexpr(x.dtype == tl.float8e4nv) or tl.constexpr(x.dtype == tl.float8e5)):
        # Convert fp8 to fp16.
        fp16_xs = ()
        for i in tl.static_range(len(xs)):
            fp16_xs += (xs[i].to(tl.float16), )
        xs = fp16_xs
        src_type: tl.constexpr = "f16"
    elif x.dtype == tl.float16:
        src_type: tl.constexpr = "f16"
    elif x.dtype == tl.bfloat16:
        src_type: tl.constexpr = "bf16"
    else:
        tl.static_assert(False, f"Unsupported dtype: {x.dtype}")
    return tl.inline_asm_elementwise(
        _accumulate_f16_into_f32_and_track_absmax_ptx(REDUCTION_SIZE, src_type, absmax_reg_name),
        "=r,r" + (",h" * len(xs)),
        (acc, ) + xs,
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


def _finalize_matmul_repr(specialization):
    signature = specialization.signature
    suffix = "" if "ScatterSrcIndx" in specialization.constants else "_scatter"
    return f"_finalize_matmul{suffix}_{signature['A'][1:]}"


@triton.jit(repr=_finalize_matmul_repr, launch_metadata=_finalize_matmul_launch_metadata)
def _finalize_matmul(
    Out,
    OutExpectedScale,
    OutActualScale,
    OutChecksumScale,
    A,
    stride_a_k,
    stride_a_m,
    AScale,
    ScatterSrcIndx,
    FinalizeScatterIdxs,
    K: tl.constexpr,
    M,
    N,
    NumRows,
    # fused activation function
    ACTIVATION_FN: tl.constexpr,
    activation_fn_args,
    ACTIVATION_REDUCTION_N: tl.constexpr,
    # epilogue transform
    EPILOGUE_FN: tl.constexpr,
    epilogue_fn_args,
    EXPT_PER_TOK: tl.constexpr,
    flexpoint_saturate_inf: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGES: tl.constexpr,
    HAS_FUSED_SCRATCHPAD: tl.constexpr,
):
    if HAS_FUSED_SCRATCHPAD:
        # Bump A to the scratchpad region.
        A += tl.cast(M, tl.int64) * stride_a_m

    USE_FUSED_MIXED_PREC_ACC: tl.constexpr = (cuda_capability_geq(10, 0)
                                              and tl.constexpr(A.dtype.element_ty != tl.float32))
    USE_FUSED_ABSMAX: tl.constexpr = (USE_FUSED_MIXED_PREC_ACC and OutActualScale is not None) and ACTIVATION_FN is None

    THREADS_PER_BLOCK: tl.constexpr = tl.extra.cuda.num_threads()
    local_max = tl.full([THREADS_PER_BLOCK], 0.0, tl.float32)
    if USE_FUSED_ABSMAX:
        local_max = tl.inline_asm_elementwise(
            """
            .reg .f32 my_abs_max;
            mov.b32 my_abs_max, 0;
            mov.b32 $0, 0;
            """, "=r,r", [local_max], dtype=tl.float32, is_pure=False, pack=1)

    out_scale = load_scale(OutExpectedScale)
    a_scale = load_scale(AScale)

    if FinalizeScatterIdxs is not None:
        MBound = tl.load(FinalizeScatterIdxs + M + M * EXPT_PER_TOK)
        if tl.program_id(0) >= MBound:
            return
    else:
        MBound = M

    if NumRows is not None:
        NumRows = NumRows  # remove constexpr
        if NumRows.dtype.is_ptr():
            NumRows = tl.load(NumRows)

    if FinalizeScatterIdxs is not None or (ScatterSrcIndx is not None and EXPT_PER_TOK > 1):
        n_active_experts = 0
    else:
        n_active_experts: tl.constexpr = EXPT_PER_TOK

    OUT_BLOCK_N: tl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    outN = N // ACTIVATION_REDUCTION_N

    for pid_m in tl.range(tl.program_id(0), MBound, tl.num_programs(0)):
        src_offs = pid_m * EXPT_PER_TOK + tl.arange(0, EXPT_PER_TOK)
        if FinalizeScatterIdxs is not None:
            row = tl.load(FinalizeScatterIdxs + pid_m)
            src_idxs = tl.load(FinalizeScatterIdxs + M + src_offs)
            n_active_experts = tl.sum((src_idxs != -1).to(tl.int32))
        elif ScatterSrcIndx is not None and EXPT_PER_TOK > 1:
            row = pid_m
            src_idxs = tl.load(ScatterSrcIndx + src_offs)
            n_active_experts = tl.sum((src_idxs != -1).to(tl.int32))
        else:
            row = pid_m
            src_idxs = src_offs
            if NumRows is not None:
                src_idxs = tl.where(src_idxs < NumRows, src_idxs, -1)

        if n_active_experts == 0:
            for off_n in tl.range(tl.program_id(1) * OUT_BLOCK_N, outN, tl.num_programs(1) * OUT_BLOCK_N):
                offs_n = off_n + tl.arange(0, OUT_BLOCK_N)
                n_mask = offs_n < outN
                tl.store(Out + row * outN + offs_n, tl.zeros([OUT_BLOCK_N], dtype=Out.dtype.element_ty), mask=n_mask)
        else:
            for off_n in tl.range(tl.program_id(1) * BLOCK_N, N, tl.num_programs(1) * BLOCK_N, num_stages=STAGES):
                offs_n = off_n + tl.arange(0, BLOCK_N)
                n_mask = offs_n < N

                acc = tl.zeros([BLOCK_N], dtype=tl.float32)
                if is_hip():
                    if EXPT_PER_TOK > 1:
                        src_idxs_tup = split_n(tl.reshape(src_idxs, (2, ) * log2(EXPT_PER_TOK)), log2(EXPT_PER_TOK))
                    else:
                        # Convert 1D tensor to 1D tuple.
                        src_idxs_tup = tl.split(tl.reshape(tl.join(src_idxs, src_idxs), (2, )))[:1]
                    for i in tl.static_range(0, EXPT_PER_TOK, 1):
                        src_idx = src_idxs_tup[i]
                        if src_idx != -1:
                            As = A + src_idx.to(tl.int64) * stride_a_m + offs_n
                            for ki in tl.static_range(K):
                                acc += tl.load(As, mask=n_mask, other=0.0)
                                As += stride_a_k
                else:
                    As = A + src_idxs.to(tl.int64)[:, None] * stride_a_m + offs_n[None, :]
                    for ki in tl.static_range(K):
                        a = tl.load(As, mask=(src_idxs != -1)[:, None] & n_mask[None, :], other=0.0)
                        As += stride_a_k
                        if USE_FUSED_MIXED_PREC_ACC:
                            acc = _mixed_precision_accumulate_and_track_absmax(
                                acc, a, axis=0,
                                absmax_reg_name="my_abs_max" if USE_FUSED_ABSMAX and ki == K - 1 else None)
                        else:
                            acc += tl.sum(a, dtype=tl.float32, axis=0)
                acc = acc * a_scale
                if ACTIVATION_FN is not None:
                    out = ACTIVATION_FN(tl.reshape(acc, (1, BLOCK_N)), *activation_fn_args)
                    out = tl.reshape(out, (OUT_BLOCK_N, ))
                else:
                    tl.static_assert(ACTIVATION_REDUCTION_N == 1,
                                     "Activation reduction must be 1 if no activation fn is provided")
                    out = acc
                if not USE_FUSED_ABSMAX and OutActualScale is not None:
                    local_max = tl.maximum(local_max, thread_local_absmax(out))
                out = float_to_flex(out, out_scale if OutExpectedScale is not None else None, None, OutChecksumScale,
                                    None, Out, flexpoint_saturate_inf)
                if EPILOGUE_FN is not None:
                    out = EPILOGUE_FN(out, *epilogue_fn_args, target_dtype=Out.dtype.element_ty,
                                      pid=row * tl.num_programs(1) + tl.program_id(1))
                offs_n = off_n // ACTIVATION_REDUCTION_N + tl.arange(0, OUT_BLOCK_N)
                n_mask = offs_n < outN
                tl.store(Out + row * outN + offs_n, out, mask=n_mask)

    persisent_m = tl.num_programs(0) < MBound
    if not persisent_m and n_active_experts == 0:
        # Skip updating the scale if there were no active experts and this is a non-persistent launch.
        # The loop ran only once, and inside it we only stored zeros.
        return

    if USE_FUSED_ABSMAX:
        local_max = tl.inline_asm_elementwise(
            "mov.b32 $0, my_abs_max;",
            "=r,r",
            [local_max],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        local_max *= a_scale
    update_scale(local_max, OutActualScale, Out)
