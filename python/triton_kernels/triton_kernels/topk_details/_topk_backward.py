import triton
import triton.language as tl


@triton.jit
def _topk_backward(
    Yi,
    stride_ym,  # topk indices
    DY,
    stride_dym,  # output gradient values
    X,
    stride_xm,  # input values
    DX,
    stride_dxm,  # input gradient values
    n_rows,
    NRows,
    n_expts_tot,
    APPLY_SOFTMAX: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    N_EXPTS_PAD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if NRows is not None:
        n_rows = tl.load(NRows)
    if pid_m >= n_rows:
        return
    Yi += pid_m * stride_ym
    DY += pid_m * stride_dym
    X += pid_m * stride_xm
    DX += pid_m * stride_dxm
    # --
    offs_xn = tl.arange(0, N_EXPTS_PAD)
    offs_yn = tl.arange(0, N_EXPTS_ACT)
    mask_xn = offs_xn < n_expts_tot
    y_indx = tl.load(Yi + offs_yn)
    # recompute softmax
    x = tl.load(X + y_indx)
    x = x.to(tl.float32)
    y = tl.softmax(x)
    # compute input-gradient
    dy = tl.load(DY + offs_yn)
    dy = dy.to(tl.float32)
    s = tl.sum(y * dy, 0)
    if APPLY_SOFTMAX:
        dx_topk = y * (dy - s)
    else:
        dx_topk = dy
    # full gradient using vectorized operations
    dx_full = tl.zeros([N_EXPTS_PAD], dtype=tl.float32)
    offs_xn_expanded = offs_xn[:, None]
    y_indx_expanded = y_indx[None, :]
    match_mask = (offs_xn_expanded == y_indx_expanded)
    dx_topk_expanded = dx_topk[None, :]
    dx_full = tl.sum(tl.where(match_mask, dx_topk_expanded, 0.0), axis=1)
    # write back
    tl.store(DX + offs_xn, dx_full, mask=mask_xn)
