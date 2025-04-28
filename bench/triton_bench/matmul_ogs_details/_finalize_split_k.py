import triton
import triton.language as tl
from triton_bench.numerics_details.flexpoint import float_to_flex


def _finalize_split_k_launch_metadata(grid, kernel, args):
    ret = dict()
    ret["name"] = f"{kernel.name}"
    P, Y = args["P"], args["Y"]
    ret["bytes"] = P.numel() * P.element_size() + Y.numel() * Y.element_size()
    nbits = Y.dtype.itemsize * 8
    ret[f"flops{nbits}"] = P.numel()
    return ret


@triton.jit(launch_metadata=_finalize_split_k_launch_metadata)
def _finalize_split_k(P, stride_p_z, stride_p_m, Y, stride_y_m, YExpectedScale, YActualScale, YChecksumScale, M, N,
                      K: tl.constexpr, NTokens, num_rows, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      FLEXPOINT_SATURATE_INF: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    PPtrs = P + off_m[:, None] * stride_p_m + off_n[None, :]
    mask = off_m[:, None] < M and off_n[None, :] < N

    row_count = num_rows if num_rows is not None else tl.load(NTokens) if NTokens is not None else None
    if row_count is not None:
        load_mask = mask and off_m[:, None] < row_count
    else:
        load_mask = mask
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for _k in range(K):
        acc += tl.load(PPtrs, mask=load_mask)
        PPtrs += stride_p_z
    acc = float_to_flex(acc, YExpectedScale, YActualScale, YChecksumScale, mask, Y, FLEXPOINT_SATURATE_INF)
    YPtrs = Y + off_m[:, None] * stride_y_m + off_n[None, :]
    tl.store(YPtrs, acc, mask=mask)
