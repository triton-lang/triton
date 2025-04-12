import torch
import triton
import triton.language as tl


@triton.jit
def vertical_popcount(x):
    """
    Input  x : uint32[..., N]
    Output y : uint32[..., 32]
    semantics : y[..., i] = sum_j((x[..., j] >> i) & 1)
    credits: @apgoucher
    """

    tl.static_assert(x.dtype == tl.uint32, "x should consist of 32-bit unsigned integers")

    BLOCK_N: tl.constexpr = x.shape[-1]  # summation axis
    BATCHES: tl.constexpr = x.numel // BLOCK_N  # number of batches
    if BLOCK_N >= 8:
        sa1: tl.constexpr = 8
    else:
        sa1: tl.constexpr = BLOCK_N
    # create 8-way sums in 4-bit fields:
    y = tl.reshape(x, [BATCHES, BLOCK_N // sa1, sa1, 1])
    y = (y >> tl.arange(0, 4)[None, None, None, :]) & 0x11111111
    y = tl.sum(y, 2)  # [BATCHES, BLOCK_N // sa1, 4]
    if BLOCK_N >= 128:
        sa2: tl.constexpr = 16
    else:
        sa2: tl.constexpr = BLOCK_N // sa1
    # create 128-way sums in 8-bit fields:
    y = tl.reshape(y, [BATCHES, BLOCK_N // (sa1 * sa2), sa2, 1, 4])
    y = (y >> (4 * tl.arange(0, 2))[None, None, None, :, None]) & 0x0f0f0f0f
    y = tl.sum(y, 2)  # [BATCHES, BLOCK_N // (sa1 * sa2), 2, 4]
    sa3: tl.constexpr = BLOCK_N // (sa1 * sa2)
    # create N-way sums in 32-bit fields:
    y = tl.reshape(y, [BATCHES, 1, sa3, 8])
    y = (y >> (8 * tl.arange(0, 4))[None, :, None, None]) & 0x000000ff
    y = tl.sum(y, 2)  # [BATCHES, 4, 8]
    y = tl.reshape(y, x.shape[:-1] + [32])
    return y


@triton.jit
def or_combine(x, y):
    return x | y


@triton.jit
def routing(X, stride_xm, stride_xn, Yv, Yi, stride_ym, stride_yn, R, stride_rm, stride_rn, BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    # load
    offs_x_n = tl.arange(0, BLOCK_N)
    X_ptrs = X + offs_m[:, None] * stride_xm + offs_x_n[None, :]
    x = tl.load(X_ptrs)
    x = (x.to(tl.int16, bitcast=True).to(tl.int32) << 16) | offs_x_n[None, :]
    # sort
    y = tl.topk(x, K, dim=1)
    y_values = ((y >> 16) & 0xFFFF).to(tl.bfloat16)
    y_indices = y & 0xFFFF
    # write back
    offs_y_n = tl.arange(0, K)
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    Yi_ptrs = Yi + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    tl.store(Yv_ptrs, y_values)
    tl.store(Yi_ptrs, y_indices)
    # pack into bitmatrix
    y_div = y_indices // 32
    y_rem = y_indices % 32
    y0 = tl.reduce(tl.where(y_div == 0, 1 << y_rem, 0), combine_fn=or_combine, axis=1)  # 0th packed int32
    y1 = tl.reduce(tl.where(y_div == 1, 1 << y_rem, 0), combine_fn=or_combine, axis=1)  # 1st packed int32
    y2 = tl.reduce(tl.where(y_div == 2, 1 << y_rem, 0), combine_fn=or_combine, axis=1)  # 2nd packed int32
    y3 = tl.reduce(tl.where(y_div == 3, 1 << y_rem, 0), combine_fn=or_combine, axis=1)  # 4th packed int32
    r = tl.join(tl.join(y0, y1), tl.join(y2, y3))
    r = tl.reshape(tl.permute(r, [0, 2, 1]), [BLOCK_M, 4])
    offs_r_n = tl.arange(0, BLOCK_N // 32)
    R_ptrs = R + offs_m[:, None] * stride_rm + offs_r_n[None, :]
    tl.store(R_ptrs, r)


@triton.jit
def hist(RoutingBits, stride_rm, stride_rn, Hist, stride_hm, BLOCK_M: tl.constexpr):
    BLOCK_N: tl.constexpr = 128
    BLOCK_B: tl.constexpr = 128 // 32
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_b = tl.arange(0, BLOCK_B)
    r = tl.load(RoutingBits + offs_m[:, None] * stride_rm + offs_b[None, :])
    r = tl.permute(r, [1, 0])
    hist = vertical_popcount(r)
    tl.store(Hist + offs_n, tl.reshape(hist, [BLOCK_N]))


M = 4
BLOCK_M, BLOCK_N = 4, 128
K = 8
torch.manual_seed(0)
dev = "cuda"
x = torch.randn((M, BLOCK_N), dtype=torch.bfloat16, device=dev)
yv = torch.empty((M, K), dtype=torch.bfloat16, device=dev)
yi = torch.empty((M, K), dtype=torch.int16, device=dev)
r = torch.empty((M, BLOCK_N // 32), dtype=torch.uint32, device=dev)
h = torch.empty((BLOCK_N, ), dtype=torch.uint32, device=dev)

routing[(M // BLOCK_M, 1)](x, x.stride(0), x.stride(1), yv, yi, yv.stride(0), yv.stride(1), r, r.stride(0), r.stride(1),
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, K=K, num_warps=4)
hist[(1, 1)](r, r.stride(0), r.stride(1), h, h.stride(0), BLOCK_M)

# print(yi)
# print(yv)
# print(yi)
# print(r)
# print(h.reshape(4, 32))
# print(torch.topk(x))

gate_vals, expt_indx = torch.topk(x, k=K, dim=1)
expt_indx, sort_indices = torch.sort(expt_indx, dim=1)
expt_indx = expt_indx.reshape(-1).to(torch.int32)
#     # expt_scal = torch.gather(expt_scal, 1, sort_indices)
#     # # flatten topk data
#     # expt_scal = expt_scal.reshape(-1)
#     # # sort by expert_id so experts are contiguous for the matmul
#     # topk_indx = torch.argsort(expt_indx, stable=True)
#     # gate_indx = torch.argsort(topk_indx)
#     # gate_scal = expt_scal[topk_indx]
h_ref = torch.histc(expt_indx, bins=BLOCK_N, max=BLOCK_N - 1)  # histogram of tokens over experts
# print(h_ref.reshape(4, 32))
assert torch.all(h.int() == h_ref.int())
# # print(h)
# # print(torch.sort(expt_indx).values)
# # for x in h.reshape(4, 32).tolist():
# #     print(x)

# # # print(expt_indx)
# # print(torch.histc(expt_idxs, bins=BLOCK_N))
