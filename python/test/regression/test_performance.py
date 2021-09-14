from numpy import record
import torch
import triton
import subprocess
import sys
import pytest

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
do_bench = lambda fn: triton.testing.do_bench(fn, percentiles=None, warmup=10, rep=250)



#######################
# Matrix Multiplication
#######################

matmul_data = {
  # square
  (256 , 256 , 256  ) : {'v100': 0.026},
  (512 , 512 , 512  ) : {'v100': 0.140},
  (1024, 1024, 1024 ) : {'v100': 0.461},
  (2048, 2048, 2048 ) : {'v100': 0.670},
  (4096, 4096, 4096 ) : {'v100': 0.801},
  (8192, 8192, 8192 ) : {'v100': 0.858},
  # tall-skinny
  (16  , 1024, 1024 ) : {'v100': 0.0126},
  (16  , 4096, 4096 ) : {'v100': 0.0536},
  (16  , 8192, 8192 ) : {'v100': 0.0957},
  (64  , 1024, 1024 ) : {'v100': 0.0486},
  (64  , 4096, 4096 ) : {'v100': 0.204},
  (64  , 8192, 8192 ) : {'v100': 0.337},
  (1024, 64  , 1024 ) : {'v100': 0.0458},
  (4096, 64  , 4096 ) : {'v100': 0.152},
  (8192, 64  , 8192 ) : {'v100': 0.302},
#   # deep reductions
#   (64  , 64  , 16384) : {'v100': 0.},
#   (64  , 64  , 65536) : {'v100': 0.},
#   (256 , 256 , 8192 ) : {'v100': 0.},
#   (256 , 256 , 32768) : {'v100': 0.},
}
@pytest.mark.parametrize('M, N, K', matmul_data.keys())
def test_matmul(M, N, K):
    ref_gpu_util = matmul_data[(M, N, K)]['v100']
    cur_sm_clock = nvsmi(['clocks.current.sm'])[0]
    ref_sm_clock = 1350
    max_gpu_perf = 1e-6*80*8*128*cur_sm_clock
    assert cur_sm_clock == ref_sm_clock, f'GPU SMs must run at {ref_sm_clock} MHz'
    a = torch.randn((M, K), dtype=torch.float16, device='cuda')
    b = torch.randn((K, N), dtype=torch.float16, device='cuda')
    ms = do_bench(lambda: triton.ops.matmul(a, b))
    cur_gpu_perf = 2.*M*N*K/ms * 1e-9
    cur_gpu_util = cur_gpu_perf / max_gpu_perf
    triton.testing.assert_almost_equal(cur_gpu_util, ref_gpu_util, decimal=2)

#######################
# Element-Wise
#######################
import triton.language as tl

@triton.jit
def _add(x_ptr, y_ptr, output_ptr, n_elements, **meta):
    BLOCK_SIZE = meta['BLOCK_SIZE']
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


elementwise_data = {
  1024*16   : {'v100': 0.0219},
  1024*64   : {'v100': 0.0843},
  1024*256  : {'v100': 0.248},
  1024*1024 : {'v100': 0.527},
  1024*4096 : {'v100': 0.776},
  1024*16384: {'v100': 0.880},
  1024*65536: {'v100': 0.911},
}

@pytest.mark.parametrize('N', elementwise_data.keys())
def test_elementwise(N):
    ref_gpu_util = elementwise_data[N]['v100']
    cur_mem_clock = nvsmi(['clocks.current.memory'])[0]
    ref_mem_clock = 877
    max_gpu_perf = 512*2*ref_mem_clock*1e-3
    assert cur_mem_clock == ref_mem_clock, f'GPU memmory must run at {ref_mem_clock} MHz'
    z = torch.empty((N, ), dtype=torch.float16, device='cuda')
    x = torch.randn_like(z)
    y = torch.randn_like(z)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
    ms = do_bench(lambda: _add[grid](x, y, z, N, BLOCK_SIZE=1024))
    cur_gpu_perf = 3.*N*z.element_size()/ms*1e-6
    cur_gpu_util = cur_gpu_perf / max_gpu_perf
    triton.testing.assert_almost_equal(cur_gpu_util, ref_gpu_util, decimal=2)

