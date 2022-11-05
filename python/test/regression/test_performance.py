import subprocess
import sys

import pytest
import torch

import triton
import triton.language as tl
from triton.testing import get_dram_gbps, get_max_tensorcore_tflops

DEVICE_NAME = 'v100'

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
        (256, 256, 256): {'float16': 0.027},
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
    'a100': {
        (256, 256, 256): {'float16': 0.010, 'float32': 0.0214, 'int8': 0.006},
        (512, 512, 512): {'float16': 0.061, 'float32': 0.109, 'int8': 0.030},
        (1024, 1024, 1024): {'float16': 0.287, 'float32': 0.331, 'int8': 0.169},
        (2048, 2048, 2048): {'float16': 0.604, 'float32': 0.599, 'int8': 0.385},
        (4096, 4096, 4096): {'float16': 0.842, 'float32': 0.862, 'int8': 0.711},
        (8192, 8192, 8192): {'float16': 0.896, 'float32': 0.932, 'int8': 0.860},
        # tall-skinny
        (16, 1024, 1024): {'float16': 0.0077, 'float32': 0.0127, 'int8': 0.005},
        (16, 4096, 4096): {'float16': 0.0363, 'float32': 0.0457, 'int8': 0.0259},
        (16, 8192, 8192): {'float16': 0.0564, 'float32': 0.0648, 'int8': 0.0431},
        (64, 1024, 1024): {'float16': 0.0271, 'float32': 0.0509, 'int8': 0.0169},
        (64, 4096, 4096): {'float16': 0.141, 'float32': 0.162, 'int8': 0.097},
        (64, 8192, 8192): {'float16': 0.244, 'float32': 0.257, 'int8': 0.174},
        (1024, 64, 1024): {'float16': 0.0263, 'float32': 0.0458, 'int8': 0.017},
        (4096, 64, 4096): {'float16': 0.135, 'float32': 0.177, 'int8': 0.102},
        (8192, 64, 8192): {'float16': 0.216, 'float32': 0.230, 'int8': 0.177},
    }
    #   # deep reductions
    #   (64  , 64  , 16384) : {'a100': 0.},
    #   (64  , 64  , 65536) : {'a100': 0.},
    #   (256 , 256 , 8192 ) : {'a100': 0.},
    #   (256 , 256 , 32768) : {'a100': 0.},
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
    ref_sm_clock = sm_clocks[DEVICE_NAME]
    max_gpu_perf = get_max_tensorcore_tflops(dtype, clock_rate=cur_sm_clock * 1e3)
    assert abs(cur_sm_clock - ref_sm_clock) < 10, f'GPU SMs must run at {ref_sm_clock} MHz'
    if dtype == torch.int8:
        a = torch.randint(-128, 127, (M, K), dtype=dtype, device='cuda')
        b = torch.randint(-128, 127, (N, K), dtype=dtype, device='cuda')
        b = b.t()  # only test row-col layout
    else:
        a = torch.randn((M, K), dtype=dtype, device='cuda')
        b = torch.randn((K, N), dtype=dtype, device='cuda')
    fn = lambda: triton.ops.matmul(a, b)
    ms = triton.testing.do_bench(fn, percentiles=None, warmup=25, rep=1000)
    cur_gpu_perf = 2. * M * N * K / ms * 1e-9
    cur_gpu_util = cur_gpu_perf / max_gpu_perf
    triton.testing.assert_almost_equal(cur_gpu_util, ref_gpu_util, decimal=2)


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
    cur_mem_clock = nvsmi(['clocks.current.memory'])[0]
    ref_mem_clock = mem_clocks[DEVICE_NAME]
    max_gpu_perf = get_dram_gbps()
    assert abs(cur_mem_clock - ref_mem_clock) < 10, f'GPU memory must run at {ref_mem_clock} MHz'
    z = torch.empty((N, ), dtype=torch.float16, device='cuda')
    x = torch.randn_like(z)
    y = torch.randn_like(z)
    grid = lambda args: (triton.cdiv(N, args['BLOCK_SIZE']), )
    fn = lambda: _add[grid](x, y, z, N, BLOCK_SIZE=1024)
    ms = triton.testing.do_bench(fn, percentiles=None, warmup=25, rep=250)
    cur_gpu_perf = 3. * N * z.element_size() / ms * 1e-6
    cur_gpu_util = cur_gpu_perf / max_gpu_perf
    triton.testing.assert_almost_equal(cur_gpu_util, ref_gpu_util, decimal=2)
