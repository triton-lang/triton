import pytest
import torch
import triton
from triton.experimental.gluon import jit
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    constexpr, program_id, BlockedLayout, SwizzledSharedLayout,
    allocate_shared_memory, SliceLayout, DistributedLinearLayout, DotOperandLayout,
    AutoLayout
)
from triton.experimental.gluon.language.amd import AMDMFMALayout, cdna4, cdna3

import os
os.environ["TRITON_CACHE_DIR"] = "/home/sijieli2/gluon_cache"
os.environ["MLIR_ENABLE_DUMP"] = "1"
os.environ["AMDGCN_ENABLE_DUMP"] = "1"

def make_block_layout(x: torch.Tensor, block_shape, num_warps):
    thread_nums, thread_load_bits = 64, 128
    order = (0, 1) if x.stride(0) == 1 else (1, 0)
    def reorder(*origin):
        return tuple([origin[o] for o in order])
    shape0, _ = reorder(*block_shape)
    bits = torch.finfo(x.dtype).bits
    vec = min(thread_load_bits // bits, shape0)
    threads_0 = min(shape0 // vec, thread_nums)
    threads_1 = thread_nums // threads_0
    warps_0 = min(shape0 // vec // threads_0, num_warps)
    warps_1 = num_warps // warps_0

    
    return BlockedLayout(
        size_per_thread=reorder(vec, 1),
        threads_per_warp=reorder(threads_0, threads_1),
        warps_per_cta=reorder(warps_0, warps_1),
        order=order
    )


@jit
def grid(
        row_step, col_step, layout,
        row_end: constexpr, col_end: constexpr,
        row_start: constexpr = 0, col_start: constexpr = 0
    ):
    off_row = gl.arange(row_start, row_end, layout=SliceLayout(1, layout)) * row_step
    off_col = gl.arange(col_start, col_end, layout=SliceLayout(0, layout)) * col_step
    return off_row[:, None] + off_col[None, :]


@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4)
    ],
    key=['M', 'N', 'K'],
)
@triton.heuristics(values={
    'blocked_a': lambda args: make_block_layout(args['a_ptr'], (args['BLOCK_SIZE_M'], args['BLOCK_SIZE_K']), args['num_warps']),
    'blocked_b': lambda args: make_block_layout(args['b_ptr'], (args['BLOCK_SIZE_K'], args['BLOCK_SIZE_N']), args['num_warps'])
})
@jit
def matmul_kernel0(
        a_ptr, b_ptr, c_ptr, 
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: constexpr,
        BLOCK_SIZE_N: constexpr,
        BLOCK_SIZE_K: constexpr,
        num_warps: constexpr,
        blocked_a: constexpr,
        blocked_b: constexpr
    ):
    mfma_layout: constexpr = AMDMFMALayout(version=3, instr_shape=[32, 32],
                            transposed=True, warps_per_cta=[num_warps, 1])
    dot_a_layout: constexpr = DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)
    dot_b_layout: constexpr = DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=8)

    pid_m = program_id(0) * BLOCK_SIZE_M
    pid_n = program_id(1) * BLOCK_SIZE_N

    a_ptr += pid_m * stride_am
    b_ptr += pid_n * stride_bn
    c_ptr += pid_m * stride_cm + pid_n * stride_cn

    offs_a = grid(stride_am, stride_ak, blocked_a, BLOCK_SIZE_M, BLOCK_SIZE_K)
    offs_b = grid(stride_bk, stride_bn, blocked_b, BLOCK_SIZE_K, BLOCK_SIZE_N)
    offs_c = grid(stride_cm, stride_cn, mfma_layout, BLOCK_SIZE_M, BLOCK_SIZE_N)

    accumulator = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout)

    for k in range(0, gl.cdiv(K, BLOCK_SIZE_K)): # num_stages ?
        ga = cdna3.buffer_load(a_ptr, offs_a)
        gb = cdna3.buffer_load(b_ptr, offs_b)

        ra = gl.convert_layout(ga, dot_a_layout)
        rb = gl.convert_layout(gb, dot_b_layout)

        accumulator = cdna3.mfma(ra, rb, accumulator)

        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

    gc = accumulator.to(gl.float16)

    cdna3.buffer_store(gc, c_ptr, offs_c)

@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4)
    ],
    key=['M', 'N', 'K'],
)
@triton.heuristics(values={
    'blocked_a': lambda args: make_block_layout(args['a_ptr'], (args['BLOCK_SIZE_M'], args['BLOCK_SIZE_K']), args['num_warps']),
    'blocked_b': lambda args: make_block_layout(args['b_ptr'], (args['BLOCK_SIZE_K'], args['BLOCK_SIZE_N']), args['num_warps'])
})
@jit
def matmul_kernel1(
        a_ptr, b_ptr, c_ptr, 
        M, N, K,
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        BLOCK_SIZE_M: constexpr,
        BLOCK_SIZE_N: constexpr,
        BLOCK_SIZE_K: constexpr,
        num_warps: constexpr,
        blocked_a: constexpr,
        blocked_b: constexpr
    ):
    mfma_layout: constexpr = AMDMFMALayout(version=3, instr_shape=[32, 32],
                                transposed=True, warps_per_cta=[num_warps//2, 2])

    shared_a_layout: constexpr = SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    shared_b_layout: constexpr = SwizzledSharedLayout(8, 2, 16, order=[0, 1])

    dot_a_layout: constexpr = DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)
    dot_b_layout: constexpr = DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=8)

    pid_m = program_id(0) * BLOCK_SIZE_M  # blockIdx
    pid_n = program_id(1) * BLOCK_SIZE_N

    a_ptr += pid_m * stride_am
    b_ptr += pid_n * stride_bn
    c_ptr += pid_m * stride_cm + pid_n * stride_cn

    offs_a = grid(stride_am, stride_ak, blocked_a, BLOCK_SIZE_M, BLOCK_SIZE_K)
    offs_b = grid(stride_bk, stride_bn, blocked_b, BLOCK_SIZE_K, BLOCK_SIZE_N)
    offs_c = grid(stride_cm, stride_cn, mfma_layout, BLOCK_SIZE_M, BLOCK_SIZE_N)

    smem_a = allocate_shared_memory(a_ptr.dtype.element_ty, [2, BLOCK_SIZE_M, BLOCK_SIZE_K], shared_a_layout)
    smem_b = allocate_shared_memory(b_ptr.dtype.element_ty, [2, BLOCK_SIZE_K, BLOCK_SIZE_M], shared_b_layout)

    accumulator = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout)

    # cdna4.async_copy.buffer_load_to_shared(smem_a0, a_ptr, offs_am)
    # cdna4.async_copy.buffer_load_to_shared(smem_b0, b_ptr, offs_bn)

    ga = cdna3.buffer_load(a_ptr, offs_a)
    gb = cdna3.buffer_load(b_ptr, offs_b)

    smem_a.index(0).store(ga)
    smem_b.index(0).store(gb)
    for k in range(0, gl.cdiv(K, BLOCK_SIZE_K*2)):
        # a = cdna4.async_copy.load_shared_relaxed(smem_a0, ra_layout)
        # b = cdna4.async_copy.load_shared_relaxed(smem_b0, rb_layout)
        # cdna4.async_copy.buffer_load_to_shared(smem_a1, a_ptr, offs_am) 
        # cdna4.async_copy.buffer_load_to_shared(smem_b1, b_ptr, offs_bn)
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk
        
        ga = cdna3.buffer_load(a_ptr, offs_a)
        gb = cdna3.buffer_load(b_ptr, offs_b)

        ra = smem_a.index(0).load(dot_a_layout)
        rb = smem_b.index(0).load(dot_b_layout)

        accumulator = cdna3.mfma(ra, rb, accumulator)  # unroll

        smem_a.index(1).store(ga)
        smem_b.index(1).store(gb)

        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

        gb = cdna3.buffer_load(b_ptr, offs_b)
        ga = cdna3.buffer_load(a_ptr, offs_a)

        ra = smem_a.index(1).load(dot_a_layout)
        rb = smem_b.index(1).load(dot_b_layout)

        accumulator = cdna3.mfma(ra, rb, accumulator)

        smem_a.index(0).store(ga)
        smem_b.index(0).store(gb)

    gc = accumulator.to(gl.float16)

    cdna3.buffer_store(gc, c_ptr, offs_c)

def matmul(a, b):
    M, _ = a.shape
    _, N = b.shape


    c = torch.randn((M,N), dtype=a.dtype, device=a.device)

    grid = lambda META: (M // META["BLOCK_SIZE_M"], N // META["BLOCK_SIZE_N"])

    # blocked_a = make_block_layout(a, (128, 32), 4)
    # blocked_b = make_block_layout(b, (32, 128), 4)
    # print(blocked_a, blocked_b, sep='\n')

    matmul_kernel1[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        # 128, 128, 32, 4
    )
    return c

configs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=["triton"],  # Label name for the lines
        line_names=["Triton"],  # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance-fp16",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
]

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'rocblas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == "__main__":
    M, N, K = 512, 512, 512
    dtype=torch.float16
    device="cuda"

    a = torch.randn((M,K), dtype=dtype, device=device)
    b = torch.randn((K,N), dtype=dtype, device=device)

    triton_out = matmul(a, b)
    torch_out = torch.matmul(a, b)

    torch.testing.assert_close(triton_out, torch_out, rtol=3e-3, atol=3e-3)
    # benchmark.run(show_plots=False, print_data=True)

