import pytest
import torch

import triton
import triton.language as tl
import triton.ops
from triton.testing import get_dram_gbps, get_max_tensorcore_tflops, nvsmi

DEVICE_NAME = {7: 'v100', 8: 'a100'}[torch.cuda.get_device_capability()[0]]

#######################
# Utilities
#######################


def print_perf(cur_ms, cur_util, ref_util):
    # print on the same line cur_ms, cur_util and ref_util with 3 decimal places
    print(f'{cur_ms:.3f} ms \t cur: {cur_util:.3f} \t ref: {ref_util:.3f} \t dif={cur_util - ref_util:.3f}', end='\t')


#######################
# Matrix Multiplication
#######################

sm_clocks = {'v100': 1350, 'a100': 1350}
mem_clocks = {'v100': 877, 'a100': 1215}

matmul_data = {
    'a100': {
        # square
        (512, 512, 512): {'float16': 0.108, 'float32': 0.097, 'int8': 0.05},
        (1024, 1024, 1024): {'float16': 0.355, 'float32': 0.313, 'int8': 0.169},
        (2048, 2048, 2048): {'float16': 0.653, 'float32': 0.532, 'int8': 0.34},
        (8192, 8192, 8192): {'float16': 0.839, 'float32': 0.754, 'int8': 0.51},
        # tall-skinny
        (16, 1024, 1024): {'float16': 0.015, 'float32': 0.009, 'int8': 0.005},
        (16, 4096, 4096): {'float16': 0.080, 'float32': 0.051, 'int8': 0.026},
        (16, 8192, 8192): {'float16': 0.083, 'float32': 0.077, 'int8': 0.043},
        (64, 1024, 1024): {'float16': 0.045, 'float32': 0.023, 'int8': 0.017},
        (64, 4096, 4096): {'float16': 0.170, 'float32': 0.000, 'int8': 0.097},
        (64, 8192, 8192): {'float16': 0.227, 'float32': 0.000, 'int8': 0.174},
        (1024, 64, 1024): {'float16': 0.040, 'float32': 0.046, 'int8': 0.017},
        (4096, 64, 4096): {'float16': 0.160, 'float32': 0.214, 'int8': 0.102},
        (8192, 64, 8192): {'float16': 0.272, 'float32': 0.000, 'int8': 0.177},
        # test EVEN_K==False
        (8192, 8192, 8176): {'float16': 0.828, 'float32': 0.743, 'int8': 0.51},
    }
}


@pytest.mark.parametrize('M, N, K, dtype_str', [(M, N, K, dtype_str)
                                                for M, N, K in matmul_data[DEVICE_NAME].keys()
                                                for dtype_str in ['float16']])
def test_matmul(M, N, K, dtype_str):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if dtype_str in ['float32', 'int8'] and DEVICE_NAME != 'a100':
        pytest.skip('Only test float32 & int8 on a100')
    if (M, N, K) in [(64, 4096, 4096), (64, 8192, 8192), (8192, 64, 8192)] and dtype_str == 'float32':
        pytest.skip('Out of shared memory in float32')
    dtype = {'float16': torch.float16, 'float32': torch.float32, 'int8': torch.int8}[dtype_str]
    torch.manual_seed(0)
    ref_gpu_util = matmul_data[DEVICE_NAME][(M, N, K)][dtype_str]
    cur_sm_clock = nvsmi(['clocks.current.sm'])[0]
    max_gpu_perf = get_max_tensorcore_tflops(dtype, clock_rate=cur_sm_clock * 1e3)
    if dtype == torch.int8:
        a = torch.randint(-128, 127, (M, K), dtype=dtype, device='cuda')
        b = torch.randint(-128, 127, (N, K), dtype=dtype, device='cuda')
        b = b.t()  # only test row-col layout
    else:
        a = torch.randn((M, K), dtype=dtype, device='cuda')
        b = torch.randn((K, N), dtype=dtype, device='cuda')
    fn = lambda: triton.ops.matmul(a, b)
    ms = triton.testing.do_bench_cudagraph(fn)
    cur_gpu_perf = 2. * M * N * K / ms * 1e-9
    cur_gpu_util = cur_gpu_perf / max_gpu_perf
    print_perf(ms, cur_gpu_util, ref_gpu_util)
    triton.testing.assert_close(cur_gpu_util, ref_gpu_util, atol=0.02, rtol=0.01)


#######################
# Element-Wise
#######################


@triton.jit
def _add(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


elementwise_data = {
    'a100': {
        1024 * 16: {'float16': 0.031, 'float32': 0.060},
        1024 * 64: {'float16': 0.120, 'float32': 0.224},
        1024 * 256: {'float16': 0.394, 'float32': 0.691},
        1024 * 1024: {'float16': 1.06, 'float32': 1.453},
        1024 * 16384: {'float16': 0.832, 'float32': 0.862},
        1024 * 65536: {'float16': 0.873, 'float32': 0.882},
        # Non pow 2
        1020 * 100: {'float16': 0.173, 'float32': 0.327},
        10003 * 7007: {'float16': 0.522, 'float32': 0.873},
    }
}


@pytest.mark.parametrize('N', elementwise_data[DEVICE_NAME].keys())
@pytest.mark.parametrize("dtype_str", ['float16', 'bfloat16', 'float32'])
def test_elementwise(N, dtype_str):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    torch.manual_seed(0)
    if dtype_str in ['bfloat16'] and DEVICE_NAME != 'a100':
        pytest.skip('Only test bfloat16 on a100')
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}[dtype_str]
    ref_dtype_str = 'float16' if dtype_str == 'bfloat16' else dtype_str
    ref_gpu_util = elementwise_data[DEVICE_NAME][N][ref_dtype_str]
    max_gpu_perf = get_dram_gbps()
    z = torch.empty((N, ), dtype=dtype, device='cuda')
    x = torch.randn_like(z)
    y = torch.randn_like(z)
    grid = lambda args: (triton.cdiv(N, args['BLOCK_SIZE']), )
    fn = lambda: _add[grid](x, y, z, N, BLOCK_SIZE=1024)
    ms = triton.testing.do_bench_cudagraph(fn)
    cur_gpu_perf = 3. * N * z.element_size() / ms * 1e-6
    cur_gpu_util = cur_gpu_perf / max_gpu_perf
    print_perf(ms, cur_gpu_util, ref_gpu_util)
    triton.testing.assert_close(cur_gpu_util, ref_gpu_util, atol=0.02, rtol=0.01)


#######################
# Flash-Attention
#######################

flash_attention_data = {
    "a100": {
        (4, 48, 4096, 64, True, True, 'forward', 'float16'): 0.542,
        (4, 48, 4096, 64, True, True, 'forward', 'bfloat16'): 0.471,
        (4, 48, 1024, 16, True, True, 'forward', 'float32'): 0.155,
        (4, 48, 4096, 64, True, True, 'backward', 'float16'): 0.232,
        (4, 48, 4096, 64, True, True, 'backward', 'bfloat16'): 0.231,
        (4, 48, 1024, 16, True, True, 'backward', 'float32'): 0.138,
        (4, 48, 4096, 64, True, False, 'forward', 'float16'): 0.306,
        (4, 48, 4096, 64, True, False, 'forward', 'bfloat16'): 0.266,
        (4, 48, 1024, 16, True, False, 'forward', 'float32'): 0.098,
        (4, 48, 4096, 64, True, False, 'backward', 'float16'): 0.157,
        (4, 48, 4096, 64, True, False, 'backward', 'bfloat16'): 0.157,
        (4, 48, 1024, 16, True, False, 'backward', 'float32'): 0.092,
        (4, 48, 4096, 64, False, True, 'forward', 'float16'): 0.541,
        (4, 48, 4096, 64, False, True, 'forward', 'bfloat16'): 0.471,
        (4, 48, 1024, 16, False, True, 'forward', 'float32'): 0.150,
        (4, 48, 4096, 64, False, True, 'backward', 'float16'): 0.291,
        (4, 48, 4096, 64, False, True, 'backward', 'bfloat16'): 0.255,
        (4, 48, 1024, 16, False, True, 'backward', 'float32'): 0.144,
        (4, 48, 4096, 64, False, False, 'forward', 'float16'): 0.306,
        (4, 48, 4096, 64, False, False, 'forward', 'bfloat16'): 0.266,
        (4, 48, 1024, 16, False, False, 'forward', 'float32'): 0.098,
        (4, 48, 4096, 64, False, False, 'backward', 'float16'): 0.159,
        (4, 48, 4096, 64, False, False, 'backward', 'bfloat16'): 0.159,
        (4, 48, 1024, 16, False, False, 'backward', 'float32'): 0.088,
    }
}


@pytest.mark.parametrize("dtype_str", ['float16', 'bfloat16', 'float32'])
@pytest.mark.parametrize("mode", ['forward', 'backward'])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("seq_par", [True, False])
@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [[4, 48, 4096, 64]])
def test_flash_attention(Z, H, N_CTX, D_HEAD, seq_par, causal, mode, dtype_str):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    is_backward = mode == 'backward'
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        pytest.skip("Flash attention only supported for compute capability < 80")
    torch.manual_seed(20)
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}[dtype_str]
    # init data
    if dtype_str == 'float32':
        N_CTX = 1024
        D_HEAD = 16
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2).requires_grad_()
    sm_scale = 0.2
    # benchmark
    fn = lambda: triton.ops.attention(q, k, v, causal, sm_scale, seq_par)
    if is_backward:
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
    ms = triton.testing.do_bench_cudagraph(fn)
    # compute flops
    flops_per_matmul = 2. * Z * H * N_CTX * N_CTX * D_HEAD * 0.5
    total_flops = 2 * flops_per_matmul
    if is_backward:
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    cur_gpu_perf = total_flops / ms * 1e-9
    # maximum flops
    cur_sm_clock = nvsmi(['clocks.current.sm'])[0]
    max_gpu_perf = get_max_tensorcore_tflops(dtype, clock_rate=cur_sm_clock * 1e3)
    cur_gpu_util = cur_gpu_perf / max_gpu_perf
    ref_gpu_util = flash_attention_data[DEVICE_NAME][(Z, H, N_CTX, D_HEAD, seq_par, causal, mode, dtype_str)]
    print_perf(ms, cur_gpu_util, ref_gpu_util)
    triton.testing.assert_close(cur_gpu_util, ref_gpu_util, atol=0.02, rtol=0.01)


#######################
# Reduction
#######################


@triton.jit
def _sum(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # run in a loop to only to make it compute bound.
    for i in range(100):
        x = tl.sum(x, axis=0) + y

    tl.store(output_ptr + offsets, x, mask=mask)


reduction_data = {
    'a100': {
        1024 * 16384: {'float16': 0.016, 'float32': 0.031, 'int16': 0.022, 'int32': 0.048},
        1024 * 65536: {'float16': 0.016, 'float32': 0.032, 'int16': 0.022, 'int32': 0.049},
    }
}


@pytest.mark.parametrize('N', reduction_data[DEVICE_NAME].keys())
@pytest.mark.parametrize("dtype_str", ['float16', 'float32', 'int16', 'int32'])
def test_reductions(N, dtype_str):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    torch.manual_seed(0)
    dtype = {'float16': torch.float16, 'float32': torch.float32, 'int16': torch.int16, 'int32': torch.int32}[dtype_str]
    ref_gpu_util = reduction_data[DEVICE_NAME][N][dtype_str]
    cur_sm_clock = nvsmi(['clocks.current.sm'])[0]
    max_gpu_perf = get_max_tensorcore_tflops(dtype, clock_rate=cur_sm_clock * 1e3)
    z = torch.empty((N, ), dtype=dtype, device='cuda')
    if dtype == torch.float16 or dtype == torch.float32:
        x = torch.randn_like(z)
        y = torch.randn_like(z)
    else:
        info = torch.iinfo(dtype)
        x = torch.randint(info.min, info.max, (N, ), dtype=dtype, device='cuda')
        y = torch.randint(info.min, info.max, (N, ), dtype=dtype, device='cuda')
    grid = lambda args: (triton.cdiv(N, args['BLOCK_SIZE']), )
    fn = lambda: _sum[grid](x, y, z, N, BLOCK_SIZE=1024)
    ms = triton.testing.do_bench_cudagraph(fn)
    cur_gpu_perf = 100. * 2. * N / ms * 1e-9
    cur_gpu_util = cur_gpu_perf / max_gpu_perf
    print_perf(ms, cur_gpu_util, ref_gpu_util)
    triton.testing.assert_close(cur_gpu_util, ref_gpu_util, atol=0.02, rtol=0.01)
