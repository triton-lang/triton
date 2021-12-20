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


#######################
# Matrix Multiplication
#######################

matmul_data = {
  # square
  (256 , 256 , 256  ) : {'v100': 0.027},
  (512 , 512 , 512  ) : {'v100': 0.158},
  (1024, 1024, 1024 ) : {'v100': 0.466},
  (2048, 2048, 2048 ) : {'v100': 0.680},
  (4096, 4096, 4096 ) : {'v100': 0.831},
  (8192, 8192, 8192 ) : {'v100': 0.849},
  # tall-skinny
  (16  , 1024, 1024 ) : {'v100': 0.0128},
  (16  , 4096, 4096 ) : {'v100': 0.0883},
  (16  , 8192, 8192 ) : {'v100': 0.101},
  (64  , 1024, 1024 ) : {'v100': 0.073},
  (64  , 4096, 4096 ) : {'v100': 0.270},
  (64  , 8192, 8192 ) : {'v100': 0.360},
  (1024, 64  , 1024 ) : {'v100': 0.0692},
  (4096, 64  , 4096 ) : {'v100': 0.264},
  (8192, 64  , 8192 ) : {'v100': 0.323},
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
    assert abs(cur_sm_clock - ref_sm_clock) < 10, f'GPU SMs must run at {ref_sm_clock} MHz'
    a = torch.randn((M, K), dtype=torch.float16, device='cuda')
    b = torch.randn((K, N), dtype=torch.float16, device='cuda')
    fn = lambda: triton.ops.matmul(a, b)
    ms = triton.testing.do_bench(fn, percentiles=None, warmup=25, rep=1000)
    cur_gpu_perf = 2.*M*N*K/ms * 1e-9
    cur_gpu_util = cur_gpu_perf / max_gpu_perf
    triton.testing.assert_almost_equal(cur_gpu_util, ref_gpu_util, decimal=2)

#######################
# Element-Wise
#######################
import triton.language as tl

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
  1024*16   : {'v100': 0.0219},
  1024*64   : {'v100': 0.0791},
  1024*256  : {'v100': 0.243},
  1024*1024 : {'v100': 0.534},
  1024*4096 : {'v100': 0.796},
  1024*16384: {'v100': 0.905},
  1024*65536: {'v100': 0.939},
}

@pytest.mark.parametrize('N', elementwise_data.keys())
def test_elementwise(N):
    ref_gpu_util = elementwise_data[N]['v100']
    cur_mem_clock = nvsmi(['clocks.current.memory'])[0]
    ref_mem_clock = 877
    max_gpu_perf = 512*2*ref_mem_clock*1e-3
    assert abs(cur_mem_clock - ref_mem_clock) < 10, f'GPU memmory must run at {ref_mem_clock} MHz'
    z = torch.empty((N, ), dtype=torch.float16, device='cuda')
    x = torch.randn_like(z)
    y = torch.randn_like(z)
    grid = lambda args: (triton.cdiv(N, args['BLOCK_SIZE']), )
    fn = lambda: _add[grid](x, y, z, N, BLOCK_SIZE=1024)
    ms = triton.testing.do_bench(fn, percentiles=None, warmup=25, rep=250)
    cur_gpu_perf = 3.*N*z.element_size()/ms*1e-6
    cur_gpu_util = cur_gpu_perf / max_gpu_perf
    triton.testing.assert_almost_equal(cur_gpu_util, ref_gpu_util, decimal=2)

