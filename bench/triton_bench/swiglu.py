from dataclasses import dataclass
from triton_bench.numerics import InFlexData, OutFlexData
from triton_bench.meta import num_sms, is_hip, threads_per_warp
import torch
from .swiglu_details._swiglu import _swiglu


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
        cdiv = lambda a, b: (a + b - 1) // b
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
        N_BLOCKS = cdiv(N // 2, BLOCK_N)
        M_BLOCKS = max(1, cdiv(num_pid, N_BLOCKS))
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
