import subprocess
import sys

import pytest
import torch

import triton
import triton.language as tl
import triton.ops
from triton.testing import get_dram_gbps, get_max_tensorcore_tflops

DEVICE_NAME = {7: 'v100', 8: 'a100'}[torch.cuda.get_device_capability()[0]]

#######################
# Utilities
#######################


def print_perf(cur_ms, cur_util, ref_util):
    # print on the same line cur_ms, cur_util and ref_util with 3 decimal places
    print(f'{cur_ms:.3f} ms \t cur: {cur_util:.3f} \t ref: {ref_util:.3f} \t dif={cur_util - ref_util:.3f}', end='\t')


def nvsmi(attrs):
    attrs = ','.join(attrs)
    cmd = ['nvidia-smi', '-i', '0', '--query-gpu=' + attrs, '--format=csv,noheader,nounits']
    out = subprocess.check_output(cmd)
    ret = out.decode(sys.stdout.encoding).split(',')
    ret = [int(x) for x in ret]
    return ret


#######################
# Matrix Multiplication
#######################

sm_clocks = {'v100': 1350, 'a100': 1350}
mem_clocks = {'v100': 877, 'a100': 1215}

matmul_data = {
    'v100': {
        # square
        (512, 512, 512): {'float16': 0.158},
        (1024, 1024, 1024): {'float16': 0.466},
        (2048, 2048, 2048): {'float16': 0.695},
        (4096, 4096, 4096): {'float16': 0.831},
        (8192, 8192, 8192): {'float16': 0.849},
        # tall-skinny
        (16, 1024, 1024): {'float16': 0.0128},
        (16, 4096, 4096): {'float16': 0.0883},
        (16, 8192, 8192): {'float16': 0.101},
        (64, 1024, 1024): {'float16': 0.073},
        (64, 4096, 4096): {'float16': 0.270},
        (64, 8192, 8192): {'float16': 0.459},
        (1024, 64, 1024): {'float16': 0.0692},
        (4096, 64, 4096): {'float16': 0.264},
        (8192, 64, 8192): {'float16': 0.452},
        # Non pow 2 shapes
        (1000, 200, 100): {'float16': 0.084},
        (1000, 200, 700): {'float16': 0.084},
        (994, 136, 402): {'float16': 0.084},
        (995, 135, 409): {'float16': 0.084},
        (99, 1357, 409): {'float16': 0.084},
    },
    # NOTE:
    # A100 in the CI server is slow-ish for some reason.
    # On some other servers, we are getting about 90% peak for 8kx8x8k float16
    'a100': {
        # square
        (512, 512, 512): {'float16': 0.084, 'float32': 0.12, 'int8': 0.05},
        (1024, 1024, 1024): {'float16': 0.332, 'float32': 0.352, 'int8': 0.169},
        (2048, 2048, 2048): {'float16': 0.598, 'float32': 0.522, 'int8': 0.34},
        (4096, 4096, 4096): {'float16': 0.702, 'float32': 0.804, 'int8': 0.46},
        (8192, 8192, 8192): {'float16': 0.829, 'float32': 0.917, 'int8': 0.51},
        # tall-skinny
        (16, 1024, 1024): {'float16': 0.008, 'float32': 0.009, 'int8': 0.005},
        (16, 4096, 4096): {'float16': 0.036, 'float32': 0.038, 'int8': 0.026},
        (16, 8192, 8192): {'float16': 0.056, 'float32': 0.061, 'int8': 0.043},
        (64, 1024, 1024): {'float16': 0.038, 'float32': 0.047, 'int8': 0.017},
        (64, 4096, 4096): {'float16': 0.143, 'float32': 0.162, 'int8': 0.097},
        (64, 8192, 8192): {'float16': 0.232, 'float32': 0.257, 'int8': 0.174},
        (1024, 64, 1024): {'float16': 0.003, 'float32': 0.004, 'int8': 0.017},
        (4096, 64, 4096): {'float16': 0.14, 'float32': 0.122, 'int8': 0.102},
        (8192, 64, 8192): {'float16': 0.207, 'float32': 0.23, 'int8': 0.177},
        # Non pow 2 shapes
        (1000, 200, 100): {'float16': 0.011, 'float32': 0.017, 'int8': 0.05},
        (1000, 200, 700): {'float16': 0.027, 'float32': 0.047, 'int8': 0.05},
        (994, 136, 402): {'float16': 0.015, 'float32': 0.024, 'int8': 0.05},
        (995, 135, 409): {'float16': 0.015, 'float32': 0.025, 'int8': 0.05},
        (99, 1357, 409): {'float16': 0.011, 'float32': 0.036, 'int8': 0.05}
    }
}


@pytest.mark.parametrize('M, N, K, dtype_str',
                         [(M, N, K, dtype_str)
                          for M, N, K in matmul_data[DEVICE_NAME].keys()
                          for dtype_str in ['float16', 'float32']])
def test_matmul(M, N, K, dtype_str):
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
    ms = triton.testing.do_bench(fn, return_mode="min", warmup=100, rep=300)
    cur_gpu_perf = 2. * M * N * K / ms * 1e-9
    cur_gpu_util = cur_gpu_perf / max_gpu_perf
    print_perf(ms, cur_gpu_util, ref_gpu_util)
    triton.testing.assert_close(cur_gpu_util, ref_gpu_util, atol=0.01, rtol=0.05)


#######################
# Element-Wise
#######################


@triton.jit
def _add(x_ptr, y_ptr, output_ptr, n_elements,
         BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


elementwise_data = {
    'v100': {
        1024 * 16: {'float16': 0.0219, 'float32': 0.010},
        1024 * 64: {'float16': 0.0791, 'float32': 0.010},
        1024 * 256: {'float16': 0.243, 'float32': 0.010},
        1024 * 1024: {'float16': 0.530, 'float32': 0.010},
        1024 * 4096: {'float16': 0.796, 'float32': 0.010},
        1024 * 16384: {'float16': 0.905, 'float32': 0.010},
        1024 * 65536: {'float16': 0.939, 'float32': 0.010},
        # Non pow 2
        1020 * 100: {'float16': 0.010, 'float32': 0.010},
        995 * 125: {'float16': 0.010, 'float32': 0.010},
        10003 * 7007: {'float16': 0.010, 'float32': 0.010},
    },
    'a100': {
        1024 * 16: {'float16': 0.010, 'bfloat16': 0.010, 'float32': 0.010},
        1024 * 64: {'float16': 0.040, 'bfloat16': 0.010, 'float32': 0.010},
        1024 * 256: {'float16': 0.132, 'bfloat16': 0.010, 'float32': 0.010},
        1024 * 1024: {'float16': 0.353, 'bfloat16': 0.010, 'float32': 0.010},
        1024 * 4096: {'float16': 0.605, 'bfloat16': 0.010, 'float32': 0.010},
        1024 * 16384: {'float16': 0.758, 'bfloat16': 0.010, 'float32': 0.010},
        1024 * 65536: {'float16': 0.850, 'bfloat16': 0.010, 'float32': 0.010},
        # Non pow 2
        1020 * 100: {'float16': 0.010, 'bfloat16': 0.010, 'float32': 0.010},
        995 * 125: {'float16': 0.010, 'bfloat16': 0.010, 'float32': 0.010},
        10003 * 7007: {'float16': 0.010, 'bfloat16': 0.010, 'float32': 0.010},
    }
}


@pytest.mark.parametrize('N', elementwise_data[DEVICE_NAME].keys())
@pytest.mark.parametrize("dtype_str", ['float16', 'bfloat16', 'float32'])
def test_elementwise(N, dtype_str):
    torch.manual_seed(0)
    if dtype_str in ['bfloat16'] and DEVICE_NAME != 'a100':
        pytest.skip('Only test bfloat16 on a100')
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}[dtype_str]
    ref_gpu_util = elementwise_data[DEVICE_NAME][N][dtype_str]
    max_gpu_perf = get_dram_gbps()
    z = torch.empty((N, ), dtype=dtype, device='cuda')
    x = torch.randn_like(z)
    y = torch.randn_like(z)
    grid = lambda args: (triton.cdiv(N, args['BLOCK_SIZE']), )
    fn = lambda: _add[grid](x, y, z, N, BLOCK_SIZE=1024)
    ms = triton.testing.do_bench(fn, return_mode="min", warmup=100, rep=500)
    cur_gpu_perf = 3. * N * z.element_size() / ms * 1e-6
    cur_gpu_util = cur_gpu_perf / max_gpu_perf
    print_perf(ms, cur_gpu_util, ref_gpu_util)
    triton.testing.assert_close(cur_gpu_util, ref_gpu_util, atol=0.01, rtol=0.05)

#######################
# Flash-Attention
#######################


flash_attention_data = {
    "a100": {
        (4, 48, 4096, 64, False, True, 'forward', 'float16'): 0.39,
        (4, 48, 4096, 64, False, True, 'backward', 'float16'): 0.252,
        (4, 48, 4096, 64, False, True, 'forward', 'bfloat16'): 0.358,
        (4, 48, 4096, 64, False, True, 'backward', 'bfloat16'): 0.243,
        (4, 48, 1024, 16, False, True, 'forward', 'float32'): 0.092,
        (4, 48, 1024, 16, False, True, 'backward', 'float32'): 0.118,
        (4, 48, 4096, 64, True, True, 'forward', 'float16'): 0.39,
        (4, 48, 4096, 64, True, True, 'backward', 'float16'): 0.175,
        (4, 48, 4096, 64, True, True, 'forward', 'bfloat16'): 0.355,
        (4, 48, 4096, 64, True, True, 'backward', 'bfloat16'): 0.168,
        (4, 48, 1024, 16, True, True, 'forward', 'float32'): 0.092,
        (4, 48, 1024, 16, True, True, 'backward', 'float32'): 0.087
    }
}


@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [[4, 48, 4096, 64]])
@pytest.mark.parametrize("seq_par", [True, False])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("mode", ['forward', 'backward'])
@pytest.mark.parametrize("dtype_str", ['float16', 'bfloat16', 'float32'])
def test_flash_attention(Z, H, N_CTX, D_HEAD, seq_par, causal, mode, dtype_str):
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
    ms = triton.testing.do_bench(fn, return_mode="min", warmup=100, rep=500)
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
    ref_gpu_util = flash_attention_data[DEVICE_NAME][(Z, H, N_CTX, D_HEAD, seq_par, mode, dtype_str)]
    print_perf(ms, cur_gpu_util, ref_gpu_util)
    triton.testing.assert_close(cur_gpu_util, ref_gpu_util, atol=0.01, rtol=0.05)
