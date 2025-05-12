import triton
import triton.language as tl


@triton.jit
def _weight_transpose(
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    X,
    stride_xe: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    Y,
    stride_ye: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    pid_e = tl.program_id(2).to(tl.int64)

    X += stride_xe * pid_e
    Y += stride_ye * pid_e

    m_exact: tl.constexpr = (M % BLOCK_M) == 0
    n_exact: tl.constexpr = (N % BLOCK_N) == 0

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = off_m < M
    mask_n = off_n < N

    if m_exact:
        if n_exact:
            mask = None
            other = None
        else:
            mask = mask_n[None, :]
            other = 0
    else:
        if n_exact:
            mask = mask_m[:, None]
            other = 0
        else:
            mask = mask_m[:, None] & mask_n[None, :]
            other = 0

    X_ptrs = X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn
    Y_ptrs = Y + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn

    tile = tl.load(X_ptrs, mask=mask, other=other)
    tl.store(Y_ptrs, tile, mask=mask)
