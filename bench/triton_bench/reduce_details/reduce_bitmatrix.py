import triton
import triton.language as tl


@triton.jit
def vpopc(x):
    """
    Vertical popcount
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
def _sum_bitmatrix_memset(Ret, ret_size, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(Ret + offs, 0, mask=offs < ret_size)


@triton.jit
def _sum_bitmatrix_rows(B, shape_bm, stride_bm,  # input bitmatrix
                        Ret, Partials, stride_pm, shape_pn,  # outputs
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    tl.static_assert(BLOCK_N % 32 == 0)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    BLOCK_B: tl.constexpr = BLOCK_N // 32
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = pid_n * BLOCK_B + tl.arange(0, BLOCK_B)
    bits = tl.load(B + offs_m[None, :] * stride_bm + offs_b[:, None], mask=offs_m[None, :] < shape_bm)
    ret = tl.reshape(vpopc(bits), [BLOCK_N])
    mask = offs_n < shape_pn
    tl.atomic_add(Ret + offs_n, ret, mask=mask)
    tl.store(Partials + pid_m * stride_pm + offs_n, ret, mask=mask)


def sum_bitmatrix_rows(x, out_ret, out_partials, partials_block_size=None):
    assert partials_block_size is not None
    cdiv = triton.cdiv
    PARTIALS_BLOCK_M = partials_block_size
    BLOCK_N = 32
    MEMSET_BLOCK = 512
    n_rows, n_cols = x.shape
    assert out_ret.shape == (n_cols, )
    assert out_partials.shape == (cdiv(n_rows, PARTIALS_BLOCK_M), n_cols)
    # output tensors
    _sum_bitmatrix_memset[(cdiv(out_ret.shape[0], MEMSET_BLOCK), )](
        out_ret, out_ret.shape[0],  # outputs
        BLOCK=512  # tunable parameter
    )
    _sum_bitmatrix_rows[(cdiv(n_rows, PARTIALS_BLOCK_M), cdiv(n_cols, BLOCK_N))](
        x.data, x.data.shape[0], x.data.stride(0),  # input
        out_ret,  # output [final reduction]
        out_partials, out_partials.stride(0), out_partials.shape[1],  # output [partial reductions]
        BLOCK_N=BLOCK_N,  # tunable parameters
        BLOCK_M=PARTIALS_BLOCK_M,  # constants
    )
    return out_ret, out_partials
