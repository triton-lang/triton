import triton
import triton.language as tl


@triton.jit
def or_combine(x, y):
    return x | y


@triton.jit
def softmax(x):
    x = x.to(tl.float32)
    z = x - tl.max(x, 1)[:, None]
    num = tl.math.exp(z)
    den = tl.sum(num, 1)[:, None]
    x = tl.math.fdiv(num, den, False)
    return x


@triton.jit
def _compute_bitmatrix(X, stride_xm,  # logits
                       Yv, Yi, stride_ym,  # topk values/indices
                       Routing, stride_rm, n_rows,  # routing bitmatrix
                       n_expts_tot, BLOCK_M: tl.constexpr, N_EXPTS_PAD: tl.constexpr, N_EXPTS_ACT: tl.constexpr):
    tl.static_assert(N_EXPTS_PAD % 32 == 0)
    x_dtype: tl.constexpr = X.dtype.element_ty
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_x_n = tl.arange(0, N_EXPTS_PAD)
    mask_m = offs_m[:, None] < n_rows
    mask_n = offs_x_n[None, :] < n_expts_tot
    mask = mask_m & mask_n
    # load
    X_ptrs = X + offs_m[:, None] * stride_xm + offs_x_n[None, :]
    x = tl.load(X_ptrs, mask=mask, other=float("-inf"))
    x = softmax(x).to(x_dtype)
    x = (x.to(tl.uint16, bitcast=True).to(tl.int32) << 16) | offs_x_n[None, :]
    # top-k experts
    x = x.to(tl.float32, bitcast=True)
    y = tl.topk(x, N_EXPTS_ACT, dim=1)
    y = y.to(tl.uint32, bitcast=True)
    # sort result in direction of ascending expert index
    y = (y << 16) | (y >> 16)
    y = tl.sort(y, dim=1)
    y_indices = y >> 16
    y_values = (y & 0x0000FFFF).to(tl.uint16).to(x_dtype, bitcast=True)
    # write back
    offs_y_n = tl.arange(0, N_EXPTS_ACT)
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    Yi_ptrs = Yi + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    tl.store(Yv_ptrs, y_values, mask=mask_m)
    tl.store(Yi_ptrs, y_indices, mask=mask_m)
    # pack into bitmatrix
    y_div = y_indices // 32
    y_rem = y_indices % 32
    y = tl.where(y_div[:, :, None] == tl.arange(0, N_EXPTS_PAD // 32)[None, :, :], (1 << y_rem)[:, :, None], 0)
    r = tl.reduce(y, combine_fn=or_combine, axis=1)
    offs_r_n = tl.arange(0, N_EXPTS_PAD // 32)
    RoutingPtrs = Routing + offs_m[:, None] * stride_rm + offs_r_n[None, :]
    tl.store(RoutingPtrs, r, mask=mask_m)
