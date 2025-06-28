import torch
import triton
import triton.language as tl


def swizzle_mxfp4_scale_hopper(x: torch.Tensor, num_warps: int, allow_pad: bool = True):
    """
    Make the 64x2 tile of scales of a 64x64 tile of mxfp4 values contiguous.
    """
    *batch, M, K = x.shape
    SWIZZLE_ALIGN_M = 2 * num_warps * 2 * 8
    SWIZZLE_ALIGN_K = 2
    pad_m = (SWIZZLE_ALIGN_M - (M % SWIZZLE_ALIGN_M)) % SWIZZLE_ALIGN_M
    pad_k = (SWIZZLE_ALIGN_K - (K % SWIZZLE_ALIGN_K)) % SWIZZLE_ALIGN_K
    if pad_m or pad_k > 0:
        assert allow_pad, "Padding is required for swizzling, but it was explicitly disabled."
    x = torch.nn.functional.pad(x, (0, pad_k, 0, pad_m))
    *batch, M, K = x.shape
    assert x.is_contiguous()
    assert num_warps & (num_warps - 1) == 0, "warps_n must be a power of 2"
    assert M % (2 * num_warps * 2 *
                8) == 0 and K % 2 == 0, f"Input tensor must have a subtile of shape (..., {2 * num_warps * 2 * 8}, 2)"
    b = len(batch)
    x = x.reshape(*batch, M // (2 * num_warps * 2 * 8), 2, num_warps, 2, 8, K // 2, 2)
    perm = [0, 2, 5, 1, 4, 6, 3]
    perm = list(range(b)) + [b + p for p in perm]
    x = x.permute(*perm)
    x = x.flatten(-5, -1)
    x = x.flatten(-3, -2)
    assert x.shape[-2] == M // 32
    assert x.shape[-1] == K * 32
    return x


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


def unswizzle_mxfp4_scale_hopper_torch(x: torch.Tensor, num_warps: int):
    """
    PyTorch inverse of unswizzle_mxfp4_scale_hopper
    """
    assert num_warps & (num_warps - 1) == 0, "num_warps must be a power of 2"
    *batch, M, K = x.shape
    b = len(batch)
    x = x.reshape(*batch, M // num_warps, num_warps, K // 64, 2, 8, 2, 2)
    perm = [0, 3, 1, 6, 4, 2, 5]
    perm = list(range(b)) + [b + p for p in perm]
    x = x.permute(*perm)
    x = x.reshape(*batch, M * 32, K // 32)
    return x
