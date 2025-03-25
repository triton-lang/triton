# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
# Licensed under the MIT License

import torch

import triton
import triton.language as tl
from .matmul_perf_model import early_config_prune, estimate_matmul_time


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound():
    configs = []
    for num_stages in [1]:
        # TODO support block size 16 for MFMA dot op
        for block_m in [16, 32] if torch.version.hip is None and not hasattr(torch, "corex") else [32, 64]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 4 if block_n <= 64 else 8
                    configs.append(
                        triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1},
                                      num_stages=num_stages, num_warps=num_warps))
                    # split_k
                    #for split_k in [2, 4, 8, 16]:
                    #    configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k},
                    #                                 num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
    return configs


def get_configs_compute_bound():
    configs = []
    for block_m in [64, 128, 256]:
        for block_n in [64, 128, 256]:
            for block_k in [32, 64, 128]:
                num_warps = 8 if block_n <= 64 else 16
                configs.append(
                    triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1},
                                  num_stages=1, num_warps=num_warps))
    return configs


@triton.autotune(
    configs=[] + get_configs_compute_bound() + get_configs_io_bound(),
    key=['M', 'N', 'K'],
    prune_configs_by={'early_config_prune': early_config_prune, 'perf_model': estimate_matmul_time, 'top_k': 10},
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['BLOCK_K'] == 0,
})
@triton.jit
def _bmm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_aq,
    stride_am,
    stride_ak,
    stride_bq,
    stride_bk,
    stride_bn,
    stride_cq,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q * stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q * stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn + idx_q * stride_cq)
    mask = (idx_m < M) & (idx_n < N)
    # handles write-back with reduction-splitting
    tl.store(C, acc, mask=mask)


class _bmm(torch.autograd.Function):
    kernel = _bmm_kernel

    _locks = {}

    @staticmethod
    def _call(a, b, dot_out_dtype):
        device = a.device
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()

        #only MR support Trans layout
        if hasattr(torch, "corex"):
            capability = torch.cuda.get_device_capability(device)
            capability = capability[0] * 10 + capability[1]
            if (capability < 71):
                if a.stride(0) >= 1 and a.stride(1) > 1:
                    a = a.contiguous()
                if b.stride(0) >= 1 and b.stride(1) > 1:
                    b = b.contiguous()
        # checks constraints
        assert a.shape[0] == b.shape[0], "incompatible dimensions"
        assert a.shape[2] == b.shape[1], "incompatible dimensions"
        B, M, K = a.shape
        _, _, N = b.shape
        # allocates output
        c = torch.empty((B, M, N), device=device, dtype=a.dtype)
        if dot_out_dtype is None:
            if a.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                dot_out_dtype = tl.float32
            else:
                dot_out_dtype = tl.int32
        else:
            assert isinstance(dot_out_dtype, torch.dtype), "dot_out_dtype must be a torch.dtype"
            if dot_out_dtype == torch.float16:
                dot_out_dtype = tl.float16
            elif dot_out_dtype in [torch.float32, torch.bfloat16]:
                dot_out_dtype = tl.float32
            else:
                dot_out_dtype = tl.int32
        # launch kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), B, 1)
        _bmm_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), a.stride(2), b.stride(0), b.stride(1),
                          b.stride(2), c.stride(0), c.stride(1), c.stride(2), dot_out_dtype=dot_out_dtype, GROUP_M=8)
        return c

    @staticmethod
    def forward(ctx, a, b, dot_out_dtype=None):
        return _bmm._call(a, b, dot_out_dtype=dot_out_dtype)


bmm = _bmm.apply
