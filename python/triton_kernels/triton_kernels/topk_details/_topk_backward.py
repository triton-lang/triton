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
    N_EXPTS_ACT_PAD: tl.constexpr = triton.next_power_of_2(N_EXPTS_ACT)
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
    offs_yn = tl.arange(0, N_EXPTS_ACT_PAD)
    mask_yn = offs_yn < N_EXPTS_ACT
    mask_xn = offs_xn < n_expts_tot
    # recompute softmax
    y_indx = tl.load(Yi + offs_yn, mask=mask_yn, other=0)
    x = tl.load(X + y_indx, mask=mask_yn, other=float("-inf"))
    x = x.to(tl.float32)
    y = tl.softmax(x)
    # compute input-gradient
    dy = tl.load(DY + offs_yn, mask=mask_yn, other=0.0)
    dy = dy.to(tl.float32)
    s = tl.sum(y * dy, 0)
    # write-back input gradient
    tl.store(DX + offs_xn, 0, mask=mask_xn)
    tl.fence()
    if APPLY_SOFTMAX:
        dx = y * (dy - s)
    else:
        dx = dy
    tl.store(DX + y_indx, dx, mask=mask_yn)
