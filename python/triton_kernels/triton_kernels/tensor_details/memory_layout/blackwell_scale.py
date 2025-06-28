import torch
import triton
import triton.language as tl

SWIZZLE_ALIGN_INNER = 8
SWIZZLE_SIZE_INNER = 4
SWIZZLE_SIZE_OUTER = 128


@triton.jit
def unswizzle_mx_scale_bw(x, SIZE_OUTER: tl.constexpr = SWIZZLE_SIZE_OUTER,
                          SIZE_INNER: tl.constexpr = SWIZZLE_SIZE_INNER,
                          ALIGN_INNER: tl.constexpr = SWIZZLE_ALIGN_INNER):
    shape_0: tl.constexpr = x.shape[0]
    shape_1: tl.constexpr = x.shape[1]
    tl.static_assert(shape_1 % SIZE_OUTER == 0)
    tl.static_assert(shape_1 // SIZE_OUTER <= ALIGN_INNER)
    x = x.reshape(shape_0, (shape_1 // SIZE_OUTER) // SIZE_INNER, 32, SIZE_OUTER // 32, SIZE_INNER)
    x = x.trans(0, 3, 2, 1, 4).reshape(shape_0 * SIZE_OUTER, shape_1 // SIZE_OUTER)
    return x


def swizzle_mx_scale_bw(tensor: torch.Tensor, allow_pad=True):
    """
    Swizzle the input tensor of shape (A, B, ... N, K) to (A, B, ... N // 128, K // 4, 32, 4, 4).
    Padding is applied if N and K are not multiples of 128 and 4 respectively.
    Returns the swizzled tensor repacked as (A, B, ... N, K), with padding.
    """
    *leading_shape, N, K, = tensor.shape
    pad_k = (SWIZZLE_ALIGN_INNER - (K % SWIZZLE_ALIGN_INNER)) % SWIZZLE_ALIGN_INNER
    pad_n = (SWIZZLE_SIZE_OUTER - (N % SWIZZLE_SIZE_OUTER)) % SWIZZLE_SIZE_OUTER
    if pad_k or pad_n > 0:
        assert allow_pad, "Padding is required for swizzling, but it was explicitly disabled."
        tensor = torch.nn.functional.pad(tensor, (0, pad_k, 0, pad_n))
    padded_shape = tensor.shape
    tensor = tensor.reshape(*leading_shape, padded_shape[-2] // SWIZZLE_SIZE_OUTER, SWIZZLE_SIZE_OUTER // 32, 32,
                            padded_shape[-1] // SWIZZLE_SIZE_INNER, SWIZZLE_SIZE_INNER)
    permute_order = list(range(len(tensor.shape)))
    permute_order[-2], permute_order[-4] = permute_order[-4], permute_order[-2]
    return tensor.permute(permute_order).reshape(*padded_shape)


def unswizzle_mx_scale_bw_torch(tensor: torch.Tensor):
    """
    Unswizzle the input tensor of shape (A, B, ... N // 128, K // 4, 32, 4, 4) packed as (A, B, ... N, K). (Testing only)
    """
    assert tensor.shape[-1] % SWIZZLE_SIZE_INNER == 0, f"{tensor.shape[-1]=} must be a multiple of {SWIZZLE_SIZE_INNER}"
    assert tensor.shape[-2] % SWIZZLE_SIZE_OUTER == 0, f"{tensor.shape[-2]=} must be a multiple of {SWIZZLE_SIZE_OUTER}"
    *leading_shape, N, K, = tensor.shape
    tensor = tensor.reshape(*leading_shape, N // SWIZZLE_SIZE_OUTER, K // SWIZZLE_SIZE_INNER, 32,
                            SWIZZLE_SIZE_OUTER // 32, SWIZZLE_SIZE_INNER)
    permute_order = list(range(len(tensor.shape)))
    permute_order[-2], permute_order[-4] = permute_order[-4], permute_order[-2]
    return tensor.permute(permute_order).reshape(*leading_shape, N, K)
