import triton
import triton.language as tl


@triton.jit
def unswizzle_mxfp4_value_hopper(x, op_idx: tl.constexpr, mma_version: tl.constexpr):
    """
    Triton inverse of swizzle_mxfp4_value_hopper
    """
    tl.static_assert(op_idx == 0 or op_idx == 1, "op_idx must be 0 or 1")
    tl.static_assert(len(x.shape) == 2, "NYI")
    tl.static_assert(mma_version == 2 or mma_version == 3, "mma_version must be 2 or 3")
    if op_idx == 1:
        x = x.trans()

    # We have two times the elements if we already upcasted to bfloat16
    mult: tl.constexpr = 2 if x.dtype == tl.bfloat16 else 1
    M: tl.constexpr = x.shape[0]
    K: tl.constexpr = x.shape[1]
    tl.static_assert(M % 4 == 0, "M must be divisible by 4")
    tl.static_assert(K % (4 * 8 * 2 * 2 * mult) == 0, f"K must be divisible by {4 * 8 * 2 * 2 * mult}")

    # We are loading 8 bf16 elements per thread to use ld.global.v4
    # Every u8 represents 2 mxfp4 elements
    u8_kwidth: tl.constexpr = 8 // 2 if mma_version == 2 else 1
    x = x.reshape(M // 4, 4, K // (4 * 8 * 2 * 2 * mult), 2, 4, 8 // u8_kwidth, 2, u8_kwidth * mult)
    x = x.trans(0, 6, 1, 3, 2, 5, 4, 7)
    x = x.reshape(M * 4, K // 4)
    if op_idx == 1:
        x = x.trans()
    return x
