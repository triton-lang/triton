import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _reduce_broadcast(
    X,
    Y,
    stride_xr,
    stride_x0,
    stride_x1,
    Mask,
    stride_mr,
    stride_m0,
    stride_m1,
    K,
    S0,
    S1,
    IS_MASK_NONE: tl.constexpr,
    BROADCAST_R: tl.constexpr,
    BROADCAST_S0: tl.constexpr,
    BROADCAST_S1: tl.constexpr,
    BLOCK_S0: tl.constexpr,
    BLOCK_S1: tl.constexpr,
):
    pid_s0 = tl.program_id(0)
    pid_s1 = tl.program_id(1)
    offs_s0 = pid_s0 * BLOCK_S0 + tl.arange(0, BLOCK_S0)
    offs_s1 = pid_s1 * BLOCK_S1 + tl.arange(0, BLOCK_S1)
    valid_s0 = offs_s0 < S0
    valid_s1 = offs_s1 < S1
    acc = tl.zeros((BLOCK_S0, BLOCK_S1), dtype=tl.float32)
    for k in tl.range(0, K, num_stages=2):
        x_ptrs = X + offs_s0[:, None] * stride_x0 + offs_s1[None, :] * stride_x1 + k * stride_xr
        x = tl.load(x_ptrs, mask=valid_s0[:, None] & valid_s1[None, :], other=0.0)
        if not IS_MASK_NONE:
            k_term = 0 if BROADCAST_R else (k * stride_mr)
            s0_term = 0 if BROADCAST_S0 else (offs_s0[:, None] * stride_m0)
            s1_term = 0 if BROADCAST_S1 else (offs_s1[None, :] * stride_m1)
            m_ptrs = Mask + k_term + s0_term + s1_term
            m = tl.load(m_ptrs, mask=valid_s0[:, None] & valid_s1[None, :], other=1)
            x = tl.where(m != 0, x, 0.0)
        acc += x.to(tl.float32)
    y_ptrs = Y + offs_s0[:, None] * S1 + offs_s1[None, :]
    tl.store(y_ptrs, acc.to(Y.dtype.element_ty), mask=valid_s0[:, None] & valid_s1[None, :])


def reduce(x: torch.Tensor, dim: int, mask: Optional[torch.Tensor] = None):
    """
    Performs a reduction over the specified dimension of the input tensor,
    optionally ignoring masked elements.

    Arguments:
        - x: Tensor
          input tensor to reduce.
        - dim: int
          dimension along which `x` should be reduce.
        - mask: Optional[torch.Tensor]
          integer mask of the same shape as `x` (or broadcastable to it).
          entries that are `0` are ignored in the reduction.
          if `mask is None`, all elements are included.

    Returns:
        - output: torch.Tensor
          The reduced tensor with `dim` removed.
    """
    if x.ndim != 3:
        raise NotImplementedError("reduce only supports 3D inputs in this implementation")
    if dim < 0:
        dim += x.ndim
    if dim not in (0, 1, 2):
        raise ValueError("dim must be in {0,1,2}")
    # input shapes
    B, M, N = x.shape
    dims = (0, 1, 2)
    nonred = tuple(d for d in dims if d != dim)
    S0, S1 = x.shape[nonred[0]], x.shape[nonred[1]]
    y = torch.empty((S0, S1), device=x.device, dtype=x.dtype)
    # Strides for X along reduced and non-reduced dims
    stride_xr = x.stride(dim)
    stride_x0 = x.stride(nonred[0])
    stride_x1 = x.stride(nonred[1])
    # Mask strides (broadcast allowed via stride 0)
    if mask is not None:
        mstr0, mstr1, mstr2 = mask.stride()
        stride_mr = (mstr0 if dim == 0 else (mstr1 if dim == 1 else mstr2))
        stride_m0 = (mstr0 if nonred[0] == 0 else (mstr1 if nonred[0] == 1 else mstr2))
        stride_m1 = (mstr0 if nonred[1] == 0 else (mstr1 if nonred[1] == 1 else mstr2))
    else:
        stride_mr = stride_m0 = stride_m1 = 0
    K = x.shape[dim]
    # Always use the 2D tiled kernel with constexpr metaprogramming for mask broadcasting
    BLOCK_S0 = 64
    BLOCK_S1 = 128
    grid = (triton.cdiv(S0, BLOCK_S0), triton.cdiv(S1, BLOCK_S1))
    _reduce_broadcast[grid](
        x,
        y,
        stride_xr,
        stride_x0,
        stride_x1,
        mask,
        stride_mr,
        stride_m0,
        stride_m1,
        K,
        S0,
        S1,
        IS_MASK_NONE=(mask is None),
        BROADCAST_R=(stride_mr == 0),
        BROADCAST_S0=(stride_m0 == 0),
        BROADCAST_S1=(stride_m1 == 0),
        BLOCK_S0=BLOCK_S0,
        BLOCK_S1=BLOCK_S1,
        num_warps=4,
    )
    return y


def reduce_torch(x: torch.Tensor, dim: int, mask: Optional[torch.Tensor] = None):
    if mask is None:
        return x.sum(dim=dim)
    m = mask.to(dtype=torch.bool, device=x.device)
    zeros = torch.zeros(1, dtype=x.dtype, device=x.device)
    return torch.where(m, x, zeros).sum(dim=dim)
