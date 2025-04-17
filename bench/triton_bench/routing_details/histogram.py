import triton
import triton.language as tl


@triton.jit
def _vertical_popcount(x):
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
def _memset_hist(Hist, hist_size, TokStarts, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(TokStarts, 0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(Hist + offs, 0, mask=offs < hist_size)


@triton.jit
def _compute_hist(R, shape_rm, stride_rm,  # routing bitmatrix
                  Hist, TokensStart, PartialHist, stride_pm, shape_pn,  # histogram
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_N2: tl.constexpr):
    tl.static_assert(BLOCK_N % 32 == 0)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    BLOCK_B: tl.constexpr = BLOCK_N // 32
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = pid_n * BLOCK_B + tl.arange(0, BLOCK_B)
    r = tl.load(R + offs_m[None, :] * stride_rm + offs_b[:, None], mask=offs_m[None, :] < shape_rm)
    hist = tl.reshape(_vertical_popcount(r), [BLOCK_N])
    mask = offs_n < shape_pn
    tl.atomic_add(Hist + offs_n, hist, mask=mask)
    tl.store(PartialHist + pid_m * stride_pm + offs_n, hist, mask=mask)
    # update atomic block counter (reuse tokens offset memory)
    tl.debug_barrier()
    if tl.atomic_add(TokensStart, 1) != tl.num_programs(0) * tl.num_programs(1) - 1:
        return
    tl.debug_barrier()
    # we are the only block left and all atomics are visible; compute cumsum

    loop_iterations = (shape_pn + BLOCK_N2 - 1) // BLOCK_N2

    x = tl.zeros([BLOCK_N2], Hist.dtype.element_ty)
    offs_n = tl.arange(0, BLOCK_N2)

    for i in range(loop_iterations):
        hist2 = tl.load(Hist + offs_n)
        tok_starts = tl.cumsum(hist2, 0) + x
        x += tl.sum(hist2, 0)
        tl.store(TokensStart, 0)
        tl.store(TokensStart + 1 + offs_n, tok_starts, mask=offs_n < shape_pn)
        offs_n += BLOCK_N2


@triton.jit
def _finalize_hist(TokensStart, PartialHist, PartialOffs, shape_pm, stride_pm, BLOCK_M: tl.constexpr):
    expt_id = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    # initialize first row of the output
    tokens_off = tl.load(TokensStart + expt_id)
    tl.store(PartialOffs + expt_id, tokens_off)
    # iterate over input data
    curr_sum = tokens_off
    for _ in range(0, shape_pm, BLOCK_M):
        offs = offs_m * stride_pm + expt_id
        curr = tl.load(PartialHist + offs, mask=offs_m < shape_pm)
        out = tl.cumsum(curr, 0) + curr_sum
        curr_sum += tl.sum(curr, 0)
        offs = (1 + offs_m) * stride_pm + expt_id
        tl.store(PartialOffs + offs, out, mask=offs_m < shape_pm - 1)
        offs_m += BLOCK_M
