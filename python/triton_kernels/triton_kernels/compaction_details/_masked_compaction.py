import triton
import triton.language as tl


@triton.jit
def _masked_compaction(Yv, Yi, BitMask, stride_bm, RetYv, RetYi, sentinel, K: tl.constexpr):
    pid_m = tl.program_id(0)
    yv = tl.load(Yv + pid_m * K + tl.arange(0, K))
    yi = tl.load(Yi + pid_m * K + tl.arange(0, K))
    div = yi // 32
    rem = yi % 32
    active_bits = (tl.load(BitMask + pid_m * stride_bm + div) >> rem) & 1
    exc_cumsum = tl.cumsum(active_bits, 0) - active_bits
    rev_arange = tl.where(active_bits, 0, K - 1 - tl.arange(0, K))
    write_indx = exc_cumsum + rev_arange
    yv = tl.where(active_bits, yv, sentinel)
    yi = tl.where(active_bits, yi, sentinel)
    tl.store(RetYv + pid_m * K + write_indx, yv)
    tl.store(RetYi + pid_m * K + write_indx, yi)
