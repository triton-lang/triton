import torch
import triton


@triton.heuristics({
    'EVEN_K': lambda *args, **meta: args[5] % (meta['BLOCK_K'] * meta['SPLIT_K']) == 0,
})
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1, 'GROUP_M': 8}, num_warps=4),
        # triton.Config({'BLOCK_M': 64 , 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1, 'GROUP_M': 8}, num_warps=4),\
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64 , 'BLOCK_K': 32, 'SPLIT_K': 1, 'GROUP_M': 8}, num_warps=4),\
        # triton.Config({'BLOCK_M': 64 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 1, 'GROUP_M': 8}, num_warps=4),\
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1, 'GROUP_M': 8}, num_warps=4),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 1, 'GROUP_M': 8}, num_warps=4),\
        # triton.Config({'BLOCK_M': 64 , 'BLOCK_N': 32 , 'BLOCK_K': 64, 'SPLIT_K': 1, 'GROUP_M': 8}, num_warps=2),\
        # triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64 , 'BLOCK_K': 64, 'SPLIT_K': 1, 'GROUP_M': 8}, num_warps=2),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def _kernel(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, LOCKS, **META):
    # extract meta-parameters
    BLOCK_M = META['BLOCK_M']
    BLOCK_N = META['BLOCK_N']
    BLOCK_K = META['BLOCK_K']
    GROUP_M = META['GROUP_M']
    SPLIT_K = META['SPLIT_K']
    # matrix multiplication
    pid = triton.program_id(0)
    pid_z = triton.program_id(1)
    grid_m = (M + BLOCK_M - 1) / BLOCK_M
    grid_n = (N + BLOCK_N - 1) / BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid / width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) / (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + triton.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + triton.arange(0, BLOCK_N)
    rk = triton.arange(0, BLOCK_K)
    # pointers
    K = K / SPLIT_K
    A = A + (pid_z * K * stride_ak + rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (pid_z * K * stride_bk + rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    acc = triton.zeros((BLOCK_M, BLOCK_N), dtype=triton.float32)
    for k in range(K, 0, -BLOCK_K):
        if META['EVEN_K']:
            a = triton.load(A)
            b = triton.load(B)
        else:
            a = triton.load(A, mask=rk[None, :] < k, other=0.)
            b = triton.load(B, mask=rk[:, None] < k, other=0.)
        acc += triton.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    acc = acc.to(triton.float16)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + triton.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + triton.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        triton.store(C, acc, mask=mask)
    else:
        LOCKS = LOCKS + triton.program_id(0)
        COUNT = LOCKS + triton.num_programs(0)
        while triton.atomic_cas(LOCKS, 0, 1) == 1:
            pass
        count = triton.load(COUNT)
        if count == 0:
            triton.store(C, acc, mask=mask)
        else:
            curr = triton.load(C, mask=mask, other=0.)
            triton.store(C, acc + curr, mask=mask)
        triton.atomic_xchg(COUNT, (count + 1) % SPLIT_K)
        triton.atomic_xchg(LOCKS, 0)


class _matmul(torch.autograd.Function):
    kernel = _kernel

    _locks = dict()

    @staticmethod
    def _call(a, b):
        device = a.device
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape
        # allocates output
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        # allocate locks for split-k
        if a.device not in _matmul._locks:
            _matmul._locks[device] = torch.zeros(1024 * 1024, dtype=torch.int32, device=device)
        locks = _matmul._locks[device]
        # launch kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
        _kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), locks)
        # done
        return c

    @staticmethod
    def forward(ctx, a, b):
        return _matmul._call(a, b)


matmul = _matmul.apply
