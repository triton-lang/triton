import torch
import triton
import triton.language as tl
from triton_kernels.numerics import InFlexData, OutFlexData
from triton_kernels.numerics_details.flexpoint import float_to_flex, load_scale
from triton_kernels.numerics_details.mxfp import dequantize_mxfp8_fn


@triton.jit
def _reduce_grouped(X, stride_xb, stride_xm, stride_xn,  #
                    XScale,  # input scalar flex scale
                    Out, stride_om, stride_on,  # output tensor
                    OutExpectedScale, OutActualScale, OutChecksumScale,  # output scalar flex scales
                    InIndx, B, N,  #
                    XMxScale, stride_mxb, stride_mxs,  # optional per-32-col output MXFP scales (uint8)
                    HAS_MX_SCALE: tl.constexpr, FLEXPOINT_SATURATE_INF: tl.constexpr, K: tl.constexpr,
                    BLOCK_N: tl.constexpr):
    pid_t = tl.program_id(0)
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
    ColPtrs = X + tl.arange(0, BLOCK_N) * stride_xn
    OColPtrs = Out + tl.arange(0, BLOCK_N) * stride_on
    if HAS_MX_SCALE:
        ColScalePtrs = XMxScale + tl.arange(0, BLOCK_N // 32) * stride_xn
    x_scale = load_scale(XScale)
    for n_curr in tl.range(0, N, BLOCK_N, num_stages=4):
        n_mask = tl.arange(0, BLOCK_N) < N - n_curr
        n_mask_scale = tl.arange(0, BLOCK_N // 32) < tl.cdiv(N - n_curr, 32)
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        # accumulate contributions for this tile
        for b in tl.range(0, B):
            for i in tl.static_range(0, K):
                is_valid = indxs[i] != -1
                x_row_ptr = ColPtrs + indxs[i] * stride_xm + b * stride_xb
                vals = tl.load(x_row_ptr, mask=n_mask & is_valid, other=0.0)
                vals = vals.to(tl.float32)
                if HAS_MX_SCALE:
                    scale_row_ptr = ColScalePtrs + indxs[i] * stride_mxs + b * stride_mxb
                    scale = tl.load(scale_row_ptr, mask=n_mask_scale & is_valid, other=0.)
                    scale = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
                    vals = vals.reshape([BLOCK_N // 32, 32])
                    vals = (scale[:, None] * vals).reshape([BLOCK_N])
                acc += vals
        acc *= x_scale
        # Compute per-32-col MXFP scales for this tile if requested
        if HAS_MX_SCALE:
            acc, acc_scale = dequantize_mxfp8_fn(acc[None, :], n_mask[None, :])
            acc = tl.reshape(acc, [acc.shape[-1]])
            acc_scale = tl.reshape(acc_scale, [acc_scale.shape[-1]])
            # Flatten to 1D vector for this tile and store directly
            scale_row_ptr = ColScalePtrs + fi * stride_mxs
            tl.store(scale_row_ptr, acc_scale, mask=n_mask_scale)
        # Convert to flexpoint output if configured (scalar scales)
        acc = float_to_flex(acc, OutExpectedScale, OutActualScale, OutChecksumScale, None, X, FLEXPOINT_SATURATE_INF)
        # write-back for this tile
        out_ptr = OColPtrs + pid_t * stride_om
        tl.store(out_ptr, acc, mask=n_mask)
        ColPtrs += BLOCK_N * stride_xn
        OColPtrs += BLOCK_N * stride_on
        if HAS_MX_SCALE:
            ColScalePtrs += BLOCK_N // 32 * stride_xn


def reduce_grouped(x: torch.Tensor, indx: torch.Tensor, x_flex: InFlexData | None = None,
                   out_flex: OutFlexData | None = None, x_mx_scale: torch.Tensor | None = None,
                   flexpoint_saturate_inf: bool = False):
    """
    In-place grouped row reduction.

    Arguments
    - x: Tensor[AnyFloat] of shape [(num_groups * K), N]
    - indx: Tensor[Int] of shape [num_groups, K]

    Description
    For each group g in [0, num_groups), this routine sums the K rows of `x`
    specified by `indx[g, :]` and overwrites the row corresponding to the first
    valid (non-negative) index with the per-group sum. Accumulation is performed
    in float32 for numerical stability, and the result is written back in the
    dtype of `x`.

    Behavior and edge cases
    - Invalid (-1) entries are skipped during accumulation and do not generate
      memory traffic. If a group has no valid entries, nothing is written for
      that group.
    - Reduction is performed tile-by-tile along the N dimension within a single
      kernel launch (persistent along N) to minimize launch overhead.

    Performance notes
    - Memory traffic per group is approximately (valid_rows_read + 1) * N * sizeof(x),
      plus index reads. With no invalid entries, this becomes (K + 1) reads/writes
      of length N per group.

    Returns
    - The input tensor `x` (modified in place).
    """
    if indx is not None:
        assert x.shape[-2] == indx.numel()
    K = 1 if indx is None else indx.shape[1]
    num_groups = x.shape[-2] // K
    out = torch.empty((num_groups, x.shape[-1]), dtype=x.dtype, device=x.device)
    BLOCK_N = 512
    # Resolve scalar flex scales (may be None)
    x_expected_scale = None if x_flex is None else x_flex.scale
    out_expected_scale = None if out_flex is None else out_flex.expected_scale
    out_actual_scale = None if out_flex is None else out_flex.actual_scale
    out_checksum_scale = None if out_flex is None else out_flex.checksum_scale
    # Resolve MXFP output scale row stride
    has_mx_scale = x_mx_scale is not None
    stride_mxb = 0 if not has_mx_scale else x_mx_scale.stride(0)
    stride_mxs = 0 if not has_mx_scale else x_mx_scale.stride(1)
    _reduce_grouped[(num_groups, )](
        x, x.stride(0), x.stride(1), x.stride(2),  #
        x_expected_scale,  # scalar input scale
        out, out.stride(0), out.stride(1),  #
        out_expected_scale, out_actual_scale, out_checksum_scale, indx,  #
        x.shape[0], x.shape[-1],  #
        x_mx_scale, stride_mxb, stride_mxs,  #
        HAS_MX_SCALE=has_mx_scale, FLEXPOINT_SATURATE_INF=flexpoint_saturate_inf,  #
        BLOCK_N=BLOCK_N, K=K,  #
        num_warps=1,  #
    )
    return out


def reduce_grouped_torch(x: torch.Tensor, indx: torch.Tensor, inplace: bool = True):
    """
    Torch reference for grouped reduction; same semantics as reduce_grouped.
    - x: [(num_groups*K), N]
    - indx: [num_groups, K] with -1 marking invalid entries
    Returns (x_out, overwritten_idx)

    Note: Because this reference is sequential and writes after summation,
    it will deterministically resolve overlapping indices across groups.
    The Triton in-place kernel does not protect against cross-group write
    conflicts; provide disjoint per-group indices to match behavior.
    """
    assert indx.ndim == 2
    num_groups, k = indx.shape
    src = indx.to(torch.long).reshape(-1)
    positions = torch.arange(src.numel(), device=src.device, dtype=torch.long)
    valid = src != -1
    # Sum contributions per group across valid rows (fp32)
    token_id = positions // k
    valid_pos = torch.nonzero(valid, as_tuple=False).squeeze(1)
    contrib = x.index_select(-2, src.index_select(0, valid_pos)).to(torch.float32)
    sums = torch.zeros((x.shape[0], num_groups, x.shape[-1]), dtype=torch.float32, device=x.device)
    sums.index_add_(-2, token_id.index_select(0, valid_pos), contrib)
    # First valid per group (left-to-right)
    big = torch.full_like(src, src.numel(), dtype=torch.long)
    masked_pos = torch.where(valid, positions, big)
    first_pos_flat = masked_pos.view(num_groups, k).min(dim=1).values
    valid_group_mask = first_pos_flat != big[0]
    overwritten = -torch.ones((num_groups, ), dtype=torch.int32, device=x.device)
    vals = torch.zeros((num_groups, x.shape[-1]), dtype=x.dtype, device=x.device)
    if valid_group_mask.any():
        idx_keep = torch.nonzero(valid_group_mask, as_tuple=False).squeeze(1)
        first_rows = src.index_select(0, first_pos_flat[valid_group_mask])
        vals2 = sums.index_select(-2, idx_keep).to(x.dtype).sum(0)
        vals.index_copy_(-2, first_rows, vals2)
        overwritten.index_copy_(0, idx_keep, first_rows.to(torch.int32))
    return vals, overwritten
