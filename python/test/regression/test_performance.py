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
    },
    # NOTE:
    # A100 in the CI server is slow-ish for some reason.
    # On some other servers, we are getting about 90% peak for 8kx8x8k float16
    'a100': {
        (512, 512, 512): {'float16': 0.08, 'float32': 0.13, 'int8': 0.05},
        (1024, 1024, 1024): {'float16': 0.33, 'float32': 0.35, 'int8': 0.169},
        (2048, 2048, 2048): {'float16': 0.62, 'float32': 0.57, 'int8': 0.34},
        (4096, 4096, 4096): {'float16': 0.81, 'float32': 0.75, 'int8': 0.46},
        (8192, 8192, 8192): {'float16': 0.77, 'float32': 0.85, 'int8': 0.51},
        # tall-skinny
        (16, 1024, 1024): {'float16': 0.0077, 'float32': 0.0127, 'int8': 0.005},
        (16, 4096, 4096): {'float16': 0.0363, 'float32': 0.0457, 'int8': 0.0259},
        (16, 8192, 8192): {'float16': 0.07, 'float32': 0.0648, 'int8': 0.0431},
        (64, 1024, 1024): {'float16': 0.0271, 'float32': 0.0509, 'int8': 0.0169},
        (64, 4096, 4096): {'float16': 0.16, 'float32': 0.162, 'int8': 0.097},
        (64, 8192, 8192): {'float16': 0.30, 'float32': 0.257, 'int8': 0.174},
        (1024, 64, 1024): {'float16': 0.0263, 'float32': 0.0458, 'int8': 0.017},
        (4096, 64, 4096): {'float16': 0.16, 'float32': 0.177, 'int8': 0.102},
        (8192, 64, 8192): {'float16': 0.25, 'float32': 0.230, 'int8': 0.177},
    }
}


@pytest.mark.parametrize('M, N, K, dtype_str',
                         [(M, N, K, dtype_str)
                          for M, N, K in matmul_data[DEVICE_NAME].keys()
                          for dtype_str in ['float16']])
def test_matmul(M, N, K, dtype_str):
    if dtype_str in ['float32', 'int8'] and DEVICE_NAME != 'a100':
        pytest.skip('Only test float32 & int8 on a100')
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
    ms = triton.testing.do_bench(fn, percentiles=None, warmup=100, rep=300)
    cur_gpu_perf = 2. * M * N * K / ms * 1e-9
    cur_gpu_util = cur_gpu_perf / max_gpu_perf
    torch.testing.assert_allclose(cur_gpu_util, ref_gpu_util, atol=0.01, rtol=0.05)


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
        1024 * 16: 0.0219,
        1024 * 64: 0.0791,
        1024 * 256: 0.243,
        1024 * 1024: 0.530,
        1024 * 4096: 0.796,
        1024 * 16384: 0.905,
        1024 * 65536: 0.939,
    },
    'a100': {
        1024 * 16: 0.008,
        1024 * 64: 0.034,
        1024 * 256: 0.114,
        1024 * 1024: 0.315,
        1024 * 4096: 0.580,
        1024 * 16384: 0.782,
        1024 * 65536: 0.850,
    }
}


@pytest.mark.parametrize('N', elementwise_data[DEVICE_NAME].keys())
def test_elementwise(N):
    torch.manual_seed(0)
    ref_gpu_util = elementwise_data[DEVICE_NAME][N]
    max_gpu_perf = get_dram_gbps()
    z = torch.empty((N, ), dtype=torch.float16, device='cuda')
    x = torch.randn_like(z)
    y = torch.randn_like(z)
    grid = lambda args: (triton.cdiv(N, args['BLOCK_SIZE']), )
    fn = lambda: _add[grid](x, y, z, N, BLOCK_SIZE=1024)
    ms = triton.testing.do_bench(fn, percentiles=None, warmup=100, rep=500)
    cur_gpu_perf = 3. * N * z.element_size() / ms * 1e-6
    cur_gpu_util = cur_gpu_perf / max_gpu_perf
    torch.testing.assert_allclose(cur_gpu_util, ref_gpu_util, atol=0.01, rtol=0.05)

#######################
# Flash-Attention
#######################


flash_attention_data = {
    "a100": {
        (4, 48, 4096, 64, 'forward', 'float16'): 0.37,
        (4, 48, 4096, 64, 'backward', 'float16'): 0.25,
    }
}


@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [[4, 48, 4096, 64]])
@pytest.mark.parametrize("mode", ['forward', 'backward'])
@pytest.mark.parametrize("dtype_str", ['float16'])
def test_flash_attention(Z, H, N_CTX, D_HEAD, mode, dtype_str):
    is_backward = mode == 'backward'
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        pytest.skip("Flash attention only supported for compute capability < 80")
    torch.manual_seed(20)
    dtype = {'float16': torch.float16, 'float32': torch.float32, 'int8': torch.int8}[dtype_str]
    # init data
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2).requires_grad_()
    sm_scale = 0.2
    # benchmark
    fn = lambda: triton.ops.attention(q, k, v, sm_scale)
    if is_backward:
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
    ms = triton.testing.do_bench(fn, percentiles=None, warmup=100, rep=500)
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
    ref_gpu_util = flash_attention_data[DEVICE_NAME][(Z, H, N_CTX, D_HEAD, mode, dtype_str)]
    torch.testing.assert_allclose(cur_gpu_util, ref_gpu_util, atol=0.01, rtol=0.05)
