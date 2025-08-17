import torch
import triton

from triton_kernels.numerics import InFlexData, OutFlexData
from ..matmul_ogs import get_kernels
from triton_kernels.numerics_details.flexpoint import float_to_flex, load_scale
from triton_kernels.numerics_details.mxfp import dequantize_mxfp8_fn
import triton.language as tl


@triton.jit
def _reduce_grouped(X, stride_xb, stride_xm, stride_xn,  #
                    XScale,  # input scalar flex scale
                    Out, stride_om, stride_on,  # output tensor
                    OutExpectedScale, OutActualScale, OutChecksumScale,  # output scalar flex scales
                    InIndx, B, N,  #
                    XMxScale, stride_mxb, stride_mxs,  # optional per-32-col output MXFP scales (uint8)
                    OutMxScale, stride_omxs,  # optional per-32-col output MXFP scales (uint8)
                    # fused activation function
                    ACTIVATION_FN: tl.constexpr, activation_fn_args, ACTIVATION_REDUCTION_N: tl.constexpr,
                    # epilogue transform
                    EPILOGUE_FN: tl.constexpr, epilogue_fn_args,
                    #
                    HAS_IN_MX_SCALE: tl.constexpr, HAS_OUT_MX_SCALE: tl.constexpr, FLEXPOINT_SATURATE_INF: tl.constexpr,
                    K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_t = tl.program_id(0)
    BLOCK_N_OUT: tl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    # persistent along N: single program on N, iterate tiles of size BLOCK_N
    start = pid_t * K
    # load indices into a tuple
    if InIndx is None:
        indxs = (pid_t, )
    else:
        indxs = ()
        for i in tl.static_range(0, K):
            indxs = indxs + (tl.load(InIndx + start + i), )
    # determine first valid topk row
    fi = indxs[(K - 1)]
    for i in tl.static_range(K - 2, -1, -1):
        fi = tl.where(indxs[i] != -1, indxs[i], fi)
    # record overwritten row index (may be -1 if none)
    XPtrs = X + tl.arange(0, BLOCK_N) * stride_xn
    OutPtrs = Out + tl.arange(0, BLOCK_N_OUT) * stride_on
    if HAS_IN_MX_SCALE:
        XScalePtrs = XMxScale + tl.arange(0, BLOCK_N // 32) * stride_xn
    if HAS_OUT_MX_SCALE:
        OutScalePtrs = OutMxScale + tl.arange(0, BLOCK_N_OUT // 32) * stride_on
    x_scale = load_scale(XScale)
    for n_curr in tl.range(0, N, BLOCK_N, num_stages=4):
        acc = tl.zeros([BLOCK_N_OUT], dtype=tl.float32)
        x_n_mask = tl.arange(0, BLOCK_N) < N - n_curr
        x_n_mask_scale = tl.arange(0, BLOCK_N // 32) < tl.cdiv(N - n_curr, 32)
        # accumulate contributions for this tile
        for i in tl.static_range(0, K):
            curr = tl.zeros([BLOCK_N], dtype=tl.float32)
            # iterate over split_k partial values
            for b in tl.range(0, B):
                is_valid = indxs[i] != -1
                x_row_ptr = XPtrs + indxs[i] * stride_xm + b * stride_xb
                vals = tl.load(x_row_ptr, mask=x_n_mask & is_valid, other=0.0)
                vals = vals.to(tl.float32)
                if HAS_IN_MX_SCALE:
                    scale_row_ptr = XScalePtrs + indxs[i] * stride_mxs + b * stride_mxb
                    scale = tl.load(scale_row_ptr, mask=x_n_mask_scale & is_valid, other=0.)
                    scale = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
                    vals = vals.reshape([BLOCK_N // 32, 32])
                    vals = (scale[:, None] * vals).reshape([BLOCK_N])
                curr += vals
            # apply nonlinearity to split-k output
            if ACTIVATION_FN is not None:
                curr = ACTIVATION_FN(curr[None, :], *activation_fn_args)
            curr = tl.reshape(curr, [curr.shape[-1]])
            # update final accumulator
            acc += curr
        acc *= x_scale
        # Compute per-32-col MXFP scales for this tile if requested
        Nrem = (N - n_curr) // ACTIVATION_REDUCTION_N
        out_n_mask = tl.arange(0, BLOCK_N_OUT) < Nrem
        out_n_mask_scale = tl.arange(0, BLOCK_N_OUT // 32) < tl.cdiv(Nrem, 32)
        if HAS_OUT_MX_SCALE:
            acc, acc_scale = dequantize_mxfp8_fn(acc[None, :], out_n_mask[None, :])
            acc = tl.reshape(acc, [acc.shape[-1]])
            acc_scale = tl.reshape(acc_scale, [acc_scale.shape[-1]])
        # Convert to flexpoint output if configured (scalar scales)
        acc = float_to_flex(acc, OutExpectedScale, OutActualScale, OutChecksumScale, None, Out, FLEXPOINT_SATURATE_INF)
        # write-back for this tile
        out_ptr = OutPtrs + pid_t * stride_om
        tl.store(out_ptr, acc, mask=out_n_mask)
        if HAS_OUT_MX_SCALE:
            out_scale_ptr = OutScalePtrs + pid_t * stride_omxs
            tl.store(out_scale_ptr, acc_scale, mask=out_n_mask_scale)
        XPtrs += BLOCK_N * stride_xn
        OutPtrs += BLOCK_N_OUT * stride_on
        if HAS_IN_MX_SCALE:
            XScalePtrs += BLOCK_N // 32 * stride_xn
        if HAS_OUT_MX_SCALE:
            OutScalePtrs += BLOCK_N_OUT // 32 * stride_xn


__all__ = [
    "reduce_grouped",
    "_reduce_grouped",
]


def reduce_grouped(
    x: torch.Tensor,
    indx: torch.Tensor,
    fused_activation,
    epilogue,
    x_flex: InFlexData | None = None,
    out_flex: OutFlexData | None = None,
    x_mx_scale: torch.Tensor | None = None,
    has_out_mx_scale: bool = False,
    out_dtype: torch.dtype | None = None,
    flexpoint_saturate_inf: bool = False,
):
    """
    In-place grouped row reduction.

    Arguments
    - x: Tensor[AnyFloat] expected shape [B, 1, (num_groups * K), N]
    - indx: Tensor[Int] of shape [num_groups, K] (global row indices; -1 for invalid)

    Returns
    - (out, out_mx_scale): out has shape [1, num_groups, N // fused_activation.reduction_n]
    """
    if indx is None and x.shape[0] == 1:
        return x.squeeze(0), None
    if indx is not None:
        assert x.shape[-2] == indx.numel()
    K = 1 if indx is None else indx.shape[1]
    num_groups = x.shape[-2] // K
    out_dtype = x.dtype if out_dtype is None else out_dtype
    assert x.shape[-1] % fused_activation.reduction_n == 0
    out_shape_n = x.shape[-1] // fused_activation.reduction_n
    out = torch.empty((x.shape[1], num_groups, out_shape_n), dtype=out_dtype, device=x.device)
    if has_out_mx_scale:
        out_mx_scale = torch.empty((num_groups, triton.cdiv(out_shape_n, 32)), dtype=torch.uint8, device=x.device)
    else:
        out_mx_scale = None
    BLOCK_N = 512
    # Resolve scalar flex scales (may be None)
    x_expected_scale = None if x_flex is None else x_flex.scale
    out_expected_scale = None if out_flex is None else out_flex.expected_scale
    out_actual_scale = None if out_flex is None else out_flex.actual_scale
    out_checksum_scale = None if out_flex is None else out_flex.checksum_scale
    # Resolve MXFP output scale row stride
    stride_mxb = 0 if x_mx_scale is None else x_mx_scale.stride(0)
    stride_mxs = 0 if x_mx_scale is None else x_mx_scale.stride(1)
    stride_omxs = 0 if out_mx_scale is None else out_mx_scale.stride(0)
    kernels = get_kernels(epilogue.specs, fused_activation.specs)
    kernels._reduce_grouped[(num_groups, )](
        x, x.stride(0), x.stride(2), x.stride(3),  #
        x_expected_scale,  # scalar input scale
        out, out.stride(1), out.stride(2),  #
        out_expected_scale, out_actual_scale, out_checksum_scale, indx,  #
        x.shape[0], x.shape[-1],  #
        x_mx_scale, stride_mxb, stride_mxs,  #
        out_mx_scale, stride_omxs,  #
        *fused_activation.fn_args, fused_activation.reduction_n, *epilogue.fn_arg_values_finalize,
        HAS_IN_MX_SCALE=x_mx_scale is not None, HAS_OUT_MX_SCALE=out_mx_scale is not None,
        FLEXPOINT_SATURATE_INF=flexpoint_saturate_inf,  #
        BLOCK_N=BLOCK_N, K=K,  #
        num_warps=1,  #
    )
    return out, out_mx_scale
