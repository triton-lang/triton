import triton
import triton.language as tl


@triton.jit
def streaming_topk(X, stride_xm, n_expts_tot, offs_m, mask_m, N_EXPTS_PAD: tl.constexpr, N_EXPTS_ACT: tl.constexpr,
                   BLOCK_N: tl.constexpr):

    # subtract 1 from loop iterations because we peel the first (masked) iteration:
    loop_iterations: tl.constexpr = N_EXPTS_PAD // BLOCK_N - 1

    offs_x_n = loop_iterations * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_x_n[None, :] < n_expts_tot

    # first iteration:
    X_ptrs = X + offs_m[:, None] * stride_xm + offs_x_n[None, :]
    x = tl.load(X_ptrs, mask=(mask_m & mask_n), other=float("-inf"))
    x = (x.to(tl.uint16, bitcast=True).to(tl.int32) << 16) | offs_x_n[None, :]
    x = x.to(tl.float32, bitcast=True)

    acc = tl.topk(x, N_EXPTS_ACT, dim=1)

    # subsequent iterations:
    for _i in range(loop_iterations):
        acc = tl.bitonic_merge(acc)  # ensure sorted ascending for the merge
        X_ptrs -= BLOCK_N
        offs_x_n -= BLOCK_N
        x = tl.load(X_ptrs, mask=mask_m, other=float("-inf"))
        x = (x.to(tl.uint16, bitcast=True).to(tl.int32) << 16) | offs_x_n[None, :]
        x = x.to(tl.float32, bitcast=True)
        acc = tl.maximum(acc, tl.topk(x, N_EXPTS_ACT, dim=1))

    return acc


@triton.jit
def _topk(X, stride_xm,  # inputs
          Yv, Yi, stride_ym,  # topk values/indices
          Bits, stride_rm, n_rows,  # bitmatrix
          n_expts_tot, BLOCK_M: tl.constexpr, N_EXPTS_PAD: tl.constexpr, N_EXPTS_ACT: tl.constexpr,
          BLOCK_N: tl.constexpr):

    tl.static_assert(BLOCK_N % 32 == 0)
    tl.static_assert(N_EXPTS_PAD % BLOCK_N == 0)
    x_dtype: tl.constexpr = X.dtype.element_ty

    # load logits
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m[:, None] < n_rows
    y = streaming_topk(X, stride_xm, n_expts_tot, offs_m, mask_m, N_EXPTS_PAD, N_EXPTS_ACT, BLOCK_N)
    y = y.to(tl.uint32, bitcast=True)

    # sort result in direction of ascending expert index
    y = (y << 16) | (y >> 16)
    y = tl.sort(y, dim=1)
    y_indices = y >> 16
    y_values = (y & 0x0000FFFF).to(tl.uint16).to(x_dtype, bitcast=True)
    y_values = tl.softmax(y_values.to(tl.float32), dim=1, keep_dims=True).to(x_dtype)

    # write back
    offs_y_n = tl.arange(0, N_EXPTS_ACT)
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    Yi_ptrs = Yi + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    tl.store(Yv_ptrs, y_values, mask=mask_m)
    tl.store(Yi_ptrs, y_indices, mask=mask_m)

    # pack into bitmatrix
    y_div = y_indices // 32
    y_rem = y_indices % 32
    loop_iterations = N_EXPTS_PAD // BLOCK_N
    for i in range(loop_iterations):
        offs_r_n = tl.arange(0, BLOCK_N // 32) + i * (BLOCK_N // 32)
        y2 = tl.where(y_div[:, :, None] == offs_r_n[None, None, :], (1 << y_rem)[:, :, None], 0)
        r = tl.reduce_or(y2, axis=1)
        BitsPtrs = Bits + offs_m[:, None] * stride_rm + offs_r_n[None, :]
        tl.store(BitsPtrs, r, mask=mask_m)
