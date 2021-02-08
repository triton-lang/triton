import pytest
import itertools
import triton
import torch
from common import *

@pytest.mark.parametrize(
    "TM, TN, TK, TZ, NWARP, M, N, K, AT, BT, DTYPE",
    itertools.chain(*[
        [
            # 1 warp
            (16, 16, 16, 1, 1, None, None, None, AT, BT, DTYPE),
            (32, 16, 16, 1, 1, None, None, None, AT, BT, DTYPE),
            (16, 32, 16, 1, 1, None, None, None, AT, BT, DTYPE),
            (16, 16, 32, 1, 1, None, None, None, AT, BT, DTYPE),
            (32, 16, 32, 1, 1, None, None, None, AT, BT, DTYPE),
            (16, 32, 32, 1, 1, None, None, None, AT, BT, DTYPE),
            (16, 16, 64, 1, 1, None, None, None, AT, BT, DTYPE),
            (64, 16, 64, 1, 1, None, None, None, AT, BT, DTYPE),
            (16, 64, 64, 1, 1, None, None, None, AT, BT, DTYPE),
            # 2 warp
            (64, 32, 64, 1, 2, None, None, None, AT, BT, DTYPE),
            (32, 64, 64, 1, 2, None, None, None, AT, BT, DTYPE),
            (64, 32, 16, 1, 2, None, None, None, AT, BT, DTYPE),
            (32, 64, 16, 1, 2, None, None, None, AT, BT, DTYPE),
            (128, 32, 32, 1, 2, None, None, None, AT, BT, DTYPE),
            (32, 128, 32, 1, 2, None, None, None, AT, BT, DTYPE),
            # 4 warp
            (128, 64, 16, 1, 4, None, None, None, AT, BT, DTYPE),
            (64, 128, 16, 1, 4, None, None, None, AT, BT, DTYPE),
            (128, 32, 32, 1, 4, None, None, None, AT, BT, DTYPE),
            (32, 128, 32, 1, 4, None, None, None, AT, BT, DTYPE),
            (128, 32, 64, 1, 4, None, None, None, AT, BT, DTYPE),
            (32, 128, 64, 1, 4, None, None, None, AT, BT, DTYPE),
            # 8 warp
            (128, 256, 16, 1, 8, None, None, None, AT, BT, DTYPE),
            (256, 128, 16, 1, 8, None, None, None, AT, BT, DTYPE),
            (256, 128, 32, 1, 8, None, None, None, AT, BT, DTYPE),
            # split-k
            (64, 64, 16, 2, 4, None, None, None, AT, BT, DTYPE),
            (64, 64, 16, 4, 4, None, None, None, AT, BT, DTYPE),
            (64, 64, 16, 8, 4, None, None, None, AT, BT, DTYPE),
            # variable input
            (128, 128, 32, 1, 4, 256, 256, 256, AT, BT, DTYPE),
            (128, 128, 32, 1, 4, 384, 128, 640, AT, BT, DTYPE),
            (128, 128, 32, 1, 4, 107, 233, 256, AT, BT, DTYPE),
            (128, 128, 32, 1, 4, 107, 233, 311, AT, BT, DTYPE)
        ] for DTYPE in ['float16'] for AT in [False, True] for BT in [False, True]
    ]))
def test_op(TM, TN, TK, TZ, NWARP, M, N, K, AT, BT, DTYPE):
    DTYPE = {'float16': torch.float16, 'float32': torch.float32}[DTYPE]
    torch.manual_seed(0)
    triton.ops._matmul._kernels = dict()
    triton.ops._matmul._CONFIGS = [({'TM': str(TM), 'TN': str(TN), 'TK': str(TK), 'TZ': str(TZ)}, NWARP)]
    if M is None: M = TM
    if N is None: N = TN
    if K is None: K = TK * TZ
    a = torch.randn((K, M) if AT else (M, K), device='cuda', dtype=DTYPE) / K**.5
    b = torch.randn((N, K) if BT else (K, N), device='cuda', dtype=DTYPE) / K**.5
    a = a.t() if AT else a
    b = b.t() if BT else b
    th_c = torch.matmul(a, b)
    tt_c = triton.ops.matmul(a, b)
    assert allclose(th_c, tt_c)

# square benchmarks
square_confs = [
    Benchmark(x_names = ['M', 'N', 'K'],
              x_vals  = [128, 256, 512, 1024, 2048, 3072, 4096, 6144],
              y_name  = 'provider',
              y_vals  = ['torch', 'triton', 'cutlass'],
              y_lines = ['Torch', 'Triton', 'CUTLASS'],
              ylabel  = 'TFLOPS',
              loglog  = False,
              plot_name = f'matmul-square-{AT}{BT}',
              args = {'AT': False, 'BT': False, 'dtype': torch.float16}) \
    for AT in [False, True] for BT in [False, True]
]

@perf_report(square_confs)
def perf_op(M, N, K, AT, BT, dtype, provider, warmup=10, rep=50):
    import os
    a = torch.randn((K, M) if AT else (M, K), device='cuda', dtype=dtype) / K**.5
    b = torch.randn((N, K) if BT else (K, N), device='cuda', dtype=dtype) / K**.5
    if AT: a = a.t()
    if BT: b = b.t()
    num_flops = 2 * M * N * K
    if provider == 'torch':
        torch_ms = do_bench(lambda: torch.matmul(a, b), warmup=warmup, rep=rep)
        torch_tflops = num_flops / torch_ms * 1e-9
        return torch_tflops
    if provider == 'triton':
        triton_ms = do_bench(lambda: triton.ops.matmul(a, b), warmup=warmup, rep=rep)
        triton_tflops = num_flops / triton_ms * 1e-9
        return triton_tflops
    if provider == 'cutlass' and 'CUTLASS_PROFILER' in os.environ:
        import subprocess
        import tempfile
        import pandas as pd
        # run program specified by CUTLASS_PROFILER env variable
        layout_a = 'column' if AT else 'row'
        layout_b = 'column' if BT else 'row'
        # create temporary file name
        fd, fname = tempfile.mkstemp()
        # run program and gets its output
        cmd = [os.environ['CUTLASS_PROFILER'], f'--m={M}', f'--n={N}', f'--k={K}', f'--A=f16:{layout_a}', f'--B=f16:{layout_b}', \
                '--C=f16:column', '--accum=f32', '--operation=gemm', '--verification-enabled=false',  f'--warmup-iterations={warmup}', \
                f'--profiling-iterations={rep}', f'--output={fname}', '--verbose=false']
        # run cmd
        subprocess.run(cmd, stdout=subprocess.PIPE)
        # read CSV output
        df_c = pd.read_csv(f'{fname}.gemm.csv')
        cutlass_tflops = max(df_c['GFLOPs']) / 1e3
        return cutlass_tflops
    return None

if __name__ == '__main__':
    perf_op.run()
