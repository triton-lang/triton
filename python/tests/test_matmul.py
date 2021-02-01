import pytest
import itertools
import triton as tt
import torch as th

@pytest.mark.parametrize("TM, TN, TK, NWARP, M, N, K, AT, BT, DTYPE", itertools.chain(*[
    [
    # 1 warp
    (16, 16, 16, 1, None, None, None, AT, BT, DTYPE),
    (32, 16, 16, 1, None, None, None, AT, BT, DTYPE),
    (16, 32, 16, 1, None, None, None, AT, BT, DTYPE),
    (16, 16, 32, 1, None, None, None, AT, BT, DTYPE),
    (32, 16, 32, 1, None, None, None, AT, BT, DTYPE),
    (16, 32, 32, 1, None, None, None, AT, BT, DTYPE),
    (16, 16, 64, 1, None, None, None, AT, BT, DTYPE),
    (64, 16, 64, 1, None, None, None, AT, BT, DTYPE),
    (16, 64, 64, 1, None, None, None, AT, BT, DTYPE),
    # 2 warp
    (64, 32, 64, 2, None, None, None, AT, BT, DTYPE),
    (32, 64, 64, 2, None, None, None, AT, BT, DTYPE),
    (64, 32, 16, 2, None, None, None, AT, BT, DTYPE),
    (32, 64, 16, 2, None, None, None, AT, BT, DTYPE),
    (128, 32, 32, 2, None, None, None, AT, BT, DTYPE),
    (32, 128, 32, 2, None, None, None, AT, BT, DTYPE),
    # 4 warp
    (128, 64, 16, 4, None, None, None, AT, BT, DTYPE),
    (64, 128, 16, 4, None, None, None, AT, BT, DTYPE),
    (128, 32, 32, 4, None, None, None, AT, BT, DTYPE),
    (32, 128, 32, 4, None, None, None, AT, BT, DTYPE),
    (128, 32, 64, 4, None, None, None, AT, BT, DTYPE),
    (32, 128, 64, 4, None, None, None, AT, BT, DTYPE),
    # 8 warp
    (128, 256, 16, 8, None, None, None, AT, BT, DTYPE),
    (256, 128, 16, 8, None, None, None, AT, BT, DTYPE),
    (256, 128, 32, 8, None, None, None, AT, BT, DTYPE),
    # variable input
    (128, 128, 32, 4, 256,  256,  256 , AT, BT, DTYPE),
    (128, 128, 32, 4, 384,  128,  640 , AT, BT, DTYPE),
    (128, 128, 32, 4, 107,  233,  256 , AT, BT, DTYPE),
    (128, 128, 32, 4, 107,  233,  311 , AT, BT, DTYPE)
    ]
    for DTYPE in ['float16']
    for AT in [False, True]
    for BT in [False, True]
]))
def test_op(TM, TN, TK, NWARP, M, N, K, AT, BT, DTYPE):
    DTYPE = {'float16': th.float16, 'float32': th.float32}[DTYPE]
    th.manual_seed(0)
    tt.ops._matmul.kernel = dict()
    tt.ops._matmul.TM = [TM]
    tt.ops._matmul.TN = [TN]
    tt.ops._matmul.TK = [TK]
    tt.ops._matmul.num_warps = [NWARP]
    if M is None: M = TM
    if N is None: N = TN
    if K is None: K = TK
    a = th.randn((K, M) if AT else (M, K), device='cuda', dtype=DTYPE) / K**.5
    b = th.randn((N, K) if BT else (K, N), device='cuda', dtype=DTYPE) / K**.5
    a = a.t() if AT else a
    b = b.t() if BT else b
    th_c = th.matmul(a, b)
    tt_c = tt.ops.matmul(a, b)
    rtol, atol = {th.float32: (1e-4, 1e-5),
                  th.float16: (1e-2, 1e-3)}[DTYPE]
    assert th.allclose(tt_c, th_c, atol=atol, rtol=rtol)


def do_bench(fn, flops = 0, warmup = 10, rep = 50):
    start_event = th.cuda.Event(enable_timing=True)
    end_event   = th.cuda.Event(enable_timing=True)
    ret = fn()
    for i in range(warmup):
        fn()
    th.cuda.synchronize()
    start_event.record()
    for i in range(rep):
        fn()
    end_event.record()
    th.cuda.synchronize()
    time_ms = start_event.elapsed_time(end_event) / rep
    return time_ms


def perf_op(dtype=th.float16, warmup=10, rep=50):
    import pandas as pd
    import os
    AT, BT = False, False
    has_cutlass = 'CUTLASS_PROFILER' in os.environ
    df = pd.DataFrame(columns=['AT', 'BT', 'N', 'TRITON', 'TORCH', 'CUTLASS'])
    Ns = [128, 256, 512, 1024]
    configs = [(AT, BT, N, N, N) for AT in [False] for BT in [False] for N in Ns]
    for AT, BT, M, N, K in configs:
        a = th.randn((K, M) if AT else (M, K), device='cuda', dtype=dtype) / K**.5
        b = th.randn((N, K) if BT else (K, N), device='cuda', dtype=dtype) / K**.5
        if AT: a = a.t()
        if BT: b = b.t()
        # benchmarks
        torch_ms = do_bench(lambda: th.matmul(a, b), warmup = warmup, rep = rep)
        triton_ms = do_bench(lambda: tt.ops.matmul(a, b), warmup = warmup, rep = rep)
        # store result
        num_flops = 2*M*N*K
        torch_tflops  = num_flops / torch_ms  * 1e-9
        triton_tflops = num_flops / triton_ms * 1e-9
        if 'CUTLASS_PROFILER' in os.environ:
            import subprocess
            # run program specified by CUTLASS_PROFILER env variable
            layout_a = 'column' if AT else 'row'
            layout_b = 'column' if BT else 'row'
            # create temporary file name
            import tempfile
            fd, fname = tempfile.mkstemp()
            # run program and gets its output
            cmd = [os.environ['CUTLASS_PROFILER'], f'--m={M}', f'--n={N}', f'--k={K}', f'--A=f16:{layout_a}', f'--B=f16:{layout_b}', \
                   '--C=f16:column', '--accum=f32', '--operation=gemm', '--verification-enabled=false',  '--warmup-iterations=10', \
                   '--profiling-iterations=50', f'--output={fname}', '--verbose=false']
            # run cmd
            subprocess.run(cmd, stdout=subprocess.PIPE)
            # read CSV output
            df_c = pd.read_csv(f'{fname}.gemm.csv')
            cutlass_tflops = max(df_c['GFLOPs'])/1e3
        else:
            cutlass_tflops = None
        df = df.append({'AT': AT, 'BT': BT, 'N': N, 'TRITON': triton_tflops, 'TORCH': torch_tflops, 'CUTLASS': cutlass_tflops}, ignore_index=True)
    pd.options.display.float_format = lambda x: '{:.2f}'.format(x)
    print(df)