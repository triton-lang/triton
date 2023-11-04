"""
Matrix Multiplication with TMA (Experimental)
================================================
In this tutorial, you will write a very short high-performance multiplication kernel that achieves
performance on parallel with cuBLAS.
"""

# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from torch.testing import assert_close

import triton
import triton.language as tl

if torch.cuda.get_device_capability()[0] < 9:
    import sys
    print("Skipping TMA benchmark for GPU with compute capability < 9")
    sys.exit(0)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=7,
                      num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=7, num_warps=4, num_ctas=2),
        # triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=7, num_warps=4, num_ctas=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, z_ptr,  #
                  M, N, K,  #
                  stride_am, stride_ak,  #
                  stride_bk, stride_bn,  #
                  stride_zm, stride_zn,  #
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr,  #
                  A_ORDER_0: tl.constexpr, A_ORDER_1: tl.constexpr,  #
                  B_ORDER_0: tl.constexpr, B_ORDER_1: tl.constexpr  #
                  ):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    block_offset_m = pid_m * BLOCK_SIZE_M
    block_offset_n = pid_n * BLOCK_SIZE_N

    a_tile_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                   offsets=(block_offset_m, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                   order=(A_ORDER_0, A_ORDER_1))
    b_tile_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                   offsets=(0, block_offset_n), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                   order=(B_ORDER_0, B_ORDER_1))
    z = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    offs_m = block_offset_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_offset_n + tl.arange(0, BLOCK_SIZE_N)
    z_ptrs = z_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_tile_ptr)
        b = tl.load(b_tile_ptr)
        z += tl.dot(a, b)
        a_tile_ptr = tl.advance(a_tile_ptr, [0, BLOCK_SIZE_K])
        b_tile_ptr = tl.advance(b_tile_ptr, [BLOCK_SIZE_K, 0])

    z = z.to(tl.float16)

    tl.store(z_ptrs, z, mask=mask)


def matmul(a, b, a_order, b_order):
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    K, N = b.shape

    z = torch.empty((M, N), device=a.device, dtype=torch.float16)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    matmul_kernel[grid](
        a_ptr=a, b_ptr=b, z_ptr=z,  #
        M=M, N=N, K=K,  #
        stride_am=a.stride(0), stride_ak=a.stride(1),  #
        stride_bk=b.stride(0), stride_bn=b.stride(1),  #
        stride_zm=z.stride(0), stride_zn=z.stride(1),  #
        A_ORDER_0=a_order[0], A_ORDER_1=a_order[1],  #
        B_ORDER_0=b_order[0], B_ORDER_1=b_order[1]  #
    )
    return z


problem_list = [
    [2048, 512, 512, False, True],
    [2048, 1024, 1024, False, False],
    [2048, 2048, 2048, True, False],
    [2048, 4096, 4096, True, True],
]


def test_matmul():
    for case in problem_list:
        M, N, K, TRANS_A, TRANS_B = case
        print(M, N, K, TRANS_A, TRANS_B)
        if (TRANS_A):
            a = torch.randn((K, M), device='cuda', dtype=torch.float16).T
            a_order = [0, 1]
        else:
            a = torch.randn((M, K), device='cuda', dtype=torch.float16)
            a_order = [1, 0]

        if (TRANS_B):
            b = torch.randn((N, K), device='cuda', dtype=torch.float16).T
            b_order = [0, 1]
        else:
            b = torch.randn((K, N), device='cuda', dtype=torch.float16)
            b_order = [1, 0]

        golden = torch.matmul(a, b)
        z = matmul(a, b, a_order, b_order)

        golden = torch.nn.functional.normalize(golden)
        z = torch.nn.functional.normalize(z)
        torch.set_printoptions(profile="full")
        assert_close(z, golden, rtol=1e-2, atol=1e-3, check_dtype=False)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['M', 'N', 'K', 'TRANS_A', 'TRANS_B'],
        x_vals=problem_list,  # different possible values for `x_name`
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['cublas', 'triton'],
        # label name for the lines
        line_names=["cuBLAS", "Triton"],
        # line styles
        styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
        ylabel="TFLOPS",  # label name for the y-axis
        plot_name="matmul-performance",
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, TRANS_A, TRANS_B, provider):
    if (TRANS_A):
        a = torch.randn((K, M), device='cuda', dtype=torch.float16).T
        a_order = [0, 1]
    else:
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        a_order = [1, 0]

    if (TRANS_B):
        b = torch.randn((N, K), device='cuda', dtype=torch.float16).T
        b_order = [0, 1]
    else:
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        b_order = [1, 0]

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), rep=100, quantiles=quantiles,
                                                     fast_flush=False)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, a_order, b_order), rep=100,
                                                     quantiles=quantiles, fast_flush=False)

    def perf(ms):
        return 2 * M * N * K * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


test_matmul()
benchmark.run(show_plots=False, print_data=True)
