from dataclasses import dataclass
from triton_bench.numerics import flex_to_float, float_to_flex, load_scale, update_scale, InFlexData, OutFlexData
from triton_bench.meta import num_sms, is_hip, threads_per_warp, cuda_capability_geq
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


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
def _swiglu(out_desc, Out, OutExpectedScale, OutActualScale, OutChecksumScale, a_desc, A, AScale, alpha, M, N,
            stride_am, stride_an, stride_outm, stride_outn, limit: tl.constexpr, ExptData, NUM_EXPERTS: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, EVEN_N: tl.constexpr, M_BLOCKS, N_BLOCKS,
            flexpoint_saturate_inf: tl.constexpr):
    if ExptData is not None:
        M = tl.load(ExptData + 2 * NUM_EXPERTS)
        M_BLOCKS = (M + BLOCK_M - 1) // BLOCK_M

    local_max = tl.full([tl.extra.cuda.num_threads()], 0.0, tl.float32)

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
        if a_desc is not None:
            a_packed = a_desc.load([pid_m * BLOCK_M, pid_n * 2 * BLOCK_N])
        if EVEN_N:
            a_packed = tl.load(A + packed_offs, mask=mask_m[:, None], other=0.)
        else:
            if pid_n * BLOCK_N + BLOCK_N <= N:
                a_packed = tl.load(A + packed_offs, mask=mask_m[:, None], other=0.)
            else:
                packed_mask = mask_m[:, None] and packed_mask_n[None, :]
                a_packed = tl.load(A + packed_offs, mask=packed_mask, other=0.)
        a_gelu, a_linear = tl.split(tl.reshape(a_packed, (BLOCK_M, BLOCK_N, 2)))
        # a gelu
        a_gelu = a_gelu.to(tl.float32) * a_scale
        if limit is not None:
            a_gelu = clip(a_gelu, limit, clip_lower=False)
        # a linear
        a_linear = a_linear.to(tl.float32) * a_scale
        if limit is not None:
            a_linear = clip(a_linear, limit, clip_lower=True)
        # compute output
        s = a_gelu / (1 + tl.exp(-alpha * a_gelu))
        out = tl.fma(s, a_linear, s)  # (s * (a_linear + 1))
        # update flexpoint stats and divide by scale
        # we don't need masking because of the `other` when loading `A`
        if OutActualScale is not None:
            absmax = thread_local_absmax(out, out.numel, tl.extra.cuda.num_threads())
            local_max = tl.maximum(local_max, absmax)
        out = float_to_flex(out, out_expected_scale,
                            None,  # ActualScale: local absmax is tracked and updated after the loop
                            OutChecksumScale, None, Out, flexpoint_saturate_inf)

        if out_desc is not None:
            out_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], out.to(Out.dtype.element_ty))
        else:
            mask = mask_m[:, None] if EVEN_N else mask_m[:, None] and mask_n[None, :]
            tl.store(Out + off_m[:, None] * stride_outm + off_n[None, :] * stride_outn, out, mask)

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
    def forward(ctx, a, alpha, precision_config, expt_data, num_experts):
        N = a.shape[-1]
        M = a.numel() // N
        assert a.stride()[-1] == 1
        assert a.shape[-1] % 2 == 0
        out = torch.empty(size=(M, N // 2), dtype=a.dtype, device=a.device)
        flex_ctx = precision_config.flex_ctx
        # optimization hyperparameters
        BLOCK_M, BLOCK_N = 32 // a.itemsize, 128
        num_warps = 4
        kwargs = {'maxnreg': 64} if not is_hip() else {}
        # TMA descriptors
        out_desc = None
        a_desc = None
        if cuda_capability_geq(9, 0) and flex_ctx.out_data.actual_scale is not None:
            # We need TMA to store the outputs otherwise Triton will aggressively removing layout conversions at
            # the cost of duplicating too much compute. With TMA, the layout conversion gets folded into the TMA store,
            # and the duplication doesn't occur.
            assert out.shape[-1] * out.element_size() % 16 == 0
            out_desc = TensorDescriptor.from_tensor(out, (BLOCK_M, BLOCK_N))
            assert a.shape[-1] * a.element_size() % 16 == 0
            a_desc = TensorDescriptor.from_tensor(a, (BLOCK_M, 2 * BLOCK_N))
        # launch semi-persistent kernel
        N_BLOCKS = triton.cdiv(N // 2, BLOCK_N)
        if expt_data is not None:
            waves_per_sm = 32 if is_hip() else 128
            num_pid = num_sms() * (waves_per_sm // num_warps)
            M_BLOCKS = max(1, triton.cdiv(num_pid, N_BLOCKS))
            grid = (min(M_BLOCKS * N_BLOCKS, 4 * num_sms()), )
        else:
            M_BLOCKS = triton.cdiv(M, BLOCK_M)
            if M_BLOCKS * N_BLOCKS >= 8 * num_sms():
                grid = (8 * num_sms(), )
            else:
                grid = (min(M_BLOCKS * N_BLOCKS, 4 * num_sms()), )
        _swiglu[grid](
            out_desc,
            flex_ctx.out_data.reinterpret(out),
            flex_ctx.out_data.expected_scale,
            flex_ctx.out_data.actual_scale,
            flex_ctx.out_data.checksum_scale,
            a_desc,
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
            expt_data,
            num_experts,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            EVEN_N=(N // 2) % 2 == 0,
            M_BLOCKS=M_BLOCKS,
            N_BLOCKS=N_BLOCKS,
            flexpoint_saturate_inf=flex_ctx.saturate_inf,
            num_warps=num_warps,
            **kwargs,
        )
        out = out.view(a.shape[:-1] + out.shape[-1:])
        return out


def swiglu(a, alpha, precision_config, expt_data=None, num_experts=0):
    return SwiGLU.apply(a, alpha, precision_config, expt_data, num_experts)


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
