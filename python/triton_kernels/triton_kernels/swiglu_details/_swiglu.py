import triton
import triton.language as tl


@triton.jit
def clip(x, limit, clip_lower: tl.constexpr):
    if clip_lower:
        res = tl.clamp(x, -limit, limit)
    else:
        res = tl.minimum(x, limit)
    return res


@triton.jit
def load_scale(scale_ptr):
    return 1.0 if scale_ptr is None else tl.load(scale_ptr)


def swiglu_repr(specialization):
    signature = specialization.signature
    constants = specialization.constants
    convert_dtype = lambda dtype: "mxfp4" if "u8" in dtype else dtype
    dtypes = "x".join([convert_dtype(f"{signature[i][1:]}") for i in ["Out", "A"]])
    blocks = "x".join([f"{constants[i]}" for i in ["BLOCK_M", "BLOCK_N"]])
    return f"_swiglu_{dtypes}_{blocks}"


def swiglu_launch_metadata(grid, kernel, args):
    M, N = args["M"], args["N"]
    ret = dict()
    ret["name"] = f"{kernel.name} [M = {M}, N = {N}]"
    A, Out = args["A"], args["Out"]
    ret["bytes"] = Out.numel() * Out.element_size() + A.numel() * A.element_size()
    return ret


@triton.jit
def exp_ftz(x):
    if tl.target_info.is_cuda():
        log2_e: tl.constexpr = 1.4426950408889634
        x *= log2_e
        return tl.inline_asm_elementwise(
            "ex2.approx.ftz.f32 $0, $1;",
            "=r, r",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        return tl.exp(x)


@triton.jit
def compute_swiglu(gelu, linear, scale, alpha, limit):
    gelu = gelu.to(tl.float32) * scale
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(tl.float32) * scale
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)
    s = gelu / (1 + exp_ftz(-alpha * gelu))
    return tl.fma(s, linear, s)  # (s * (linear + 1))


@triton.jit(repr=lambda _: "_swiglu")
def _swiglu_fn(input, alpha, limit):
    gelu, linear = tl.split(tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    return compute_swiglu(gelu, linear, 1.0, alpha, limit)


@triton.jit(repr=swiglu_repr, launch_metadata=swiglu_launch_metadata)
def _swiglu(Out, OutExpectedScale, OutActualScale, OutChecksumScale, A, AScale, alpha, M, N, stride_am, stride_an,
            stride_outm, stride_outn, limit: tl.constexpr, NTokens, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
            EVEN_N: tl.constexpr, M_BLOCKS, N_BLOCKS, flexpoint_saturate_inf: tl.constexpr):
    if NTokens is not None:
        M = tl.load(NTokens)
        M_BLOCKS = (M + BLOCK_M - 1) // BLOCK_M

    a_scale = load_scale(AScale)
    out_expected_scale = load_scale(OutExpectedScale)

    for pid in tl.range(tl.program_id(0), M_BLOCKS * N_BLOCKS, tl.num_programs(0), num_stages=2):
        pid_m = (pid // N_BLOCKS)
        pid_n = (pid % N_BLOCKS)
        off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = off_m < M
        mask_n = off_n < N
        packed_off_n = pid_n * BLOCK_N + tl.arange(0, 2 * BLOCK_N) // 2
        packed_mask_n = packed_off_n < N
        packed_mask_n = tl.max_constancy(packed_mask_n, [16])
        # load a
        packed_off_n = pid_n * 2 * BLOCK_N + tl.arange(0, 2 * BLOCK_N)
        packed_offs = off_m[:, None] * stride_am + packed_off_n[None, :] * stride_an
        if EVEN_N:
            a_packed = tl.load(A + packed_offs, mask=mask_m[:, None], other=0.)
        else:
            if pid_n * BLOCK_N + BLOCK_N <= N:
                a_packed = tl.load(A + packed_offs, mask=mask_m[:, None], other=0.)
            else:
                packed_mask = mask_m[:, None] & packed_mask_n[None, :]
                a_packed = tl.load(A + packed_offs, mask=packed_mask, other=0.)
        a_gelu, a_linear = tl.split(tl.reshape(a_packed, (BLOCK_M, BLOCK_N, 2)))
        out = compute_swiglu(a_gelu, a_linear, a_scale, alpha, limit)
        out = out / out_expected_scale
        mask = mask_m[:, None] if EVEN_N else mask_m[:, None] & mask_n[None, :]
        tl.store(Out + off_m[:, None] * stride_outm + off_n[None, :] * stride_outn, out, mask)
