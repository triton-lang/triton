import triton
import triton.language as tl


@triton.jit
def unswizzle_mxfp4_scale_hopper(x, num_warps: tl.constexpr):
    """
    Triton inverse of swizzle_mxfp4_scale_hopper
    """
    tl.static_assert(len(x.shape) == 2, "NYI")
    M: tl.constexpr = x.shape[0]
    K: tl.constexpr = x.shape[1]
    tl.static_assert(M % num_warps == 0, f"M must be divisible by {num_warps}. Got {M}")
    tl.static_assert(K % 64 == 0, f"K must be divisible by 64. Got {K}")
    x = x.reshape(M // num_warps, num_warps, K // 64, 2, 8, 2, 2)
    x = x.trans(0, 3, 1, 6, 4, 2, 5)
    x = x.reshape(M * 32, K // 32)
    return x
