from dataclasses import dataclass
from triton_bench.numerics import flex_to_float, float_to_flex, update_scale, InFlexData, OutFlexData
from triton_bench.meta import num_sms, is_hip, threads_per_warp
import torch
import triton
import triton.language as tl


@triton.jit
def clip(x, limit, clip_lower: tl.constexpr):
    res = tl.minimum(x, limit)
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


@triton.jit
def thread_local_absmax(x, BLOCK_SIZE: tl.constexpr, NUM_THREADS: tl.constexpr):
    return tl.max(tl.reshape(tl.abs(x), [NUM_THREADS, BLOCK_SIZE // NUM_THREADS], can_reorder=True), axis=1)


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


@triton.jit(repr=swiglu_repr, launch_metadata=swiglu_launch_metadata)
def _swiglu(Out, OutExpectedScale, OutActualScale, A, AScale, alpha, M, N, stride_am, stride_an, stride_outm,
            stride_outn,
            # optional PID-indexed arrays for tracking RMS of linear and nonlinear parts
            limit: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, M_BLOCKS, NUM_THREADS: tl.constexpr,
            flexpoint_saturate_inf: tl.constexpr):
    pid_m = tl.program_id(axis=0).to(tl.int64)
    pid_n = tl.program_id(axis=1).to(tl.int64)
    # iterate over M blocks until M is reached
    base_m = pid_m * BLOCK_M
    local_max = tl.full([NUM_THREADS], 0.0, tl.float32)
    while base_m < M:
        off_m = base_m + tl.arange(0, BLOCK_M)
        off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = off_m[:, None] < M and off_n[None, :] < N
        # column offsets
        off_a_gelu = off_m[:, None] * stride_am + 2 * off_n[None, :] * stride_an
        off_a_linear = off_m[:, None] * stride_am + 2 * off_n[None, :] * stride_an + 1
        # a gelu
        a_gelu = tl.load(A + off_a_gelu, mask=mask, other=0.)
        a_gelu = flex_to_float(a_gelu, scale_ptr=AScale)
        if limit is not None:
            a_gelu = clip(a_gelu, limit, clip_lower=False)
        # a linear
        a_linear = tl.load(A + off_a_linear, mask=mask, other=0.)
        a_linear = flex_to_float(a_linear, scale_ptr=AScale)
        if limit is not None:
            a_linear = clip(a_linear, limit, clip_lower=True)
        # compute output
        out_gelu = a_gelu / (1 + tl.exp(-alpha * a_gelu))
        out = out_gelu * (a_linear + 1)
        # update flexpoint stats and divide by scale
        # we don't need masking because of the `other` when loading `A`
        if OutActualScale is not None:
            absmax = thread_local_absmax(out, BLOCK_M * BLOCK_N, NUM_THREADS)
            local_max = tl.maximum(local_max, absmax)
        out = float_to_flex(out, OutExpectedScale, None, None, Out, None, flexpoint_saturate_inf)
        # write-back
        tl.store(Out + off_m[:, None] * stride_outm + off_n[None, :] * stride_outn, out, mask=mask)
        # increment block base
        base_m += (M_BLOCKS * BLOCK_M)
    # reduce flexpoint scale
    update_scale(local_max, OutActualScale, Out)


@dataclass(frozen=True)
class FlexCtx:
    out_data: OutFlexData = OutFlexData()
    inp_data: InFlexData = InFlexData()
    saturate_inf: bool = False


@dataclass(frozen=True)
class PrecisionConfig:
    limit: float
    flex_ctx: FlexCtx = FlexCtx()


class SwiGLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, alpha, precision_config):
        N = a.shape[-1]
        M = a.numel() // N
        assert a.stride()[-1] == 1
        assert a.shape[-1] % 2 == 0
        out = torch.empty(size=(M, N // 2), dtype=a.dtype, device=a.device)
        flex_ctx = precision_config.flex_ctx
        # optimization hyperparameters
        BLOCK_M, BLOCK_N = 8, 128
        num_warps = 2
        waves_per_sm = 32 if is_hip() else 128
        # launch semi-persistent kernel
        num_pid = num_sms() * (waves_per_sm // num_warps)
        N_BLOCKS = triton.cdiv(N // 2, BLOCK_N)
        M_BLOCKS = max(1, triton.cdiv(num_pid, N_BLOCKS))
        grid = (M_BLOCKS, N_BLOCKS)
        _swiglu[grid](
            flex_ctx.out_data.reinterpret(out),
            flex_ctx.out_data.expected_scale,
            flex_ctx.out_data.actual_scale,
            flex_ctx.inp_data.reinterpret(a),
            flex_ctx.inp_data.scale,
            alpha,
            M,
            N // 2,
            a.shape[-1],
            1,
            out.shape[-1],
            1,
            precision_config.limit,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            M_BLOCKS=M_BLOCKS,
            NUM_THREADS=num_warps * threads_per_warp(),
            flexpoint_saturate_inf=flex_ctx.saturate_inf,
            num_warps=num_warps,
        )
        out = out.view(a.shape[:-1] + out.shape[-1:])
        return out


def swiglu(a, alpha, precision_config):
    return SwiGLU.apply(a, alpha, precision_config)


def swiglu_torch(a, alpha, precision_config):
    limit = precision_config.limit
    a_gelu = a[..., ::2]
    if limit is not None:
        a_gelu = a_gelu.clamp(max=limit)
    a_linear = a[..., 1::2]
    if limit is not None:
        a_linear = a_linear.clamp(min=-limit, max=limit)

    out_gelu = a_gelu * torch.sigmoid(alpha * a_gelu)
    out = out_gelu * (a_linear + 1)
    return out
