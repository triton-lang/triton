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
