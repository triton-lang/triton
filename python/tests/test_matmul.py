import pytest
import itertools
import triton as tt
import torch as th

@pytest.mark.parametrize("TM, TN, TK, TZ, NWARP, M, N, K, AT, BT, DTYPE", itertools.chain(*[
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
    (128, 128, 32, 1, 4, 256,  256,  256 , AT, BT, DTYPE),
    (128, 128, 32, 1, 4, 384,  128,  640 , AT, BT, DTYPE),
    (128, 128, 32, 1, 4, 107,  233,  256 , AT, BT, DTYPE),
    (128, 128, 32, 1, 4, 107,  233,  311 , AT, BT, DTYPE)
    ]
    for DTYPE in ['float16']
    for AT in [False, True]
    for BT in [False, True]
]))
def test_op(TM, TN, TK, TZ, NWARP, M, N, K, AT, BT, DTYPE):
    DTYPE = {'float16': th.float16, 'float32': th.float32}[DTYPE]
    th.manual_seed(0)
    tt.ops._matmul._kernels = dict()
    tt.ops._matmul._CONFIGS = [({'TM': str(TM) , 'TN': str(TN) , 'TK': str(TK), 'TZ': str(TZ)}, NWARP)]
    if M is None: M = TM
    if N is None: N = TN
    if K is None: K = TK*TZ
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

def time_all(fn, x_names, x_vals, y_name, y_vals, y_lines, ylabel, loglog=True, plot_name='', **kwargs):
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(columns = [x_names[0]] + y_lines)
    for x in x_vals:
        x_args = {x_name: x for x_name in x_names}
        row = [fn(**x_args, **{y_name: y}, **kwargs) for y in y_vals]
        df.loc[len(df)] = [x] + row
    print(df)
    if plot_name:
        df.plot(x=x_names[0], y=y_lines, ylabel=ylabel, xlabel=' = '.join(x_names), title=f'{plot_name}', loglog=loglog)
        plt.savefig(f'{plot_name}.pdf')

def perf_op(M, N, K, AT, BT, dtype, provider, warmup=10, rep=50):
    import os
    a = th.randn((K, M) if AT else (M, K), device='cuda', dtype=dtype) / K**.5
    b = th.randn((N, K) if BT else (K, N), device='cuda', dtype=dtype) / K**.5
    if AT: a = a.t()
    if BT: b = b.t()
    num_flops = 2*M*N*K
    if provider == 'torch':
        torch_ms = do_bench(lambda: th.matmul(a, b), warmup = warmup, rep = rep)
        torch_tflops  = num_flops / torch_ms  * 1e-9
        return torch_tflops
    if provider == 'triton':
        triton_ms = do_bench(lambda: tt.ops.matmul(a, b), warmup = warmup, rep = rep)
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
                '--C=f16:column', '--accum=f32', '--operation=gemm', '--verification-enabled=false',  '--warmup-iterations=10', \
                '--profiling-iterations=50', f'--output={fname}', '--verbose=false']
        # run cmd
        subprocess.run(cmd, stdout=subprocess.PIPE)
        # read CSV output
        df_c = pd.read_csv(f'{fname}.gemm.csv')
        cutlass_tflops = max(df_c['GFLOPs'])/1e3
        return cutlass_tflops
    return None

if __name__ == '__main__':
    # # square
    x_square = [128, 256, 512, 1024, 2048, 3072, 4096, 6144]
    time_all(perf_op, x_names = ['M', 'N', 'K'], x_vals = x_square,  y_name = 'provider' , y_vals = ['torch', 'triton', 'cutlass'], 
            ylabel = 'TFLOPS', y_lines = ['Torch', 'Triton', 'CUTLASS'], AT = False, BT = False, dtype = th.float16, loglog=False, plot_name = 'matmul-square-nn')
    time_all(perf_op, x_names = ['M', 'N', 'K'], x_vals = x_square,  y_name = 'provider' , y_vals = ['torch', 'triton', 'cutlass'], 
            ylabel = 'TFLOPS', y_lines = ['Torch', 'Triton', 'CUTLASS'], AT = False, BT = True, dtype = th.float16, loglog=False, plot_name = 'matmul-square-nt')
    time_all(perf_op, x_names = ['M', 'N', 'K'], x_vals = x_square,  y_name = 'provider' , y_vals = ['torch', 'triton', 'cutlass'], 
            ylabel = 'TFLOPS', y_lines = ['Torch', 'Triton', 'CUTLASS'], AT = True, BT = False, dtype = th.float16, loglog=False, plot_name = 'matmul-square-tn')
    time_all(perf_op, x_names = ['M', 'N', 'K'], x_vals = x_square,  y_name = 'provider' , y_vals = ['torch', 'triton', 'cutlass'], 
            ylabel = 'TFLOPS', y_lines = ['Torch', 'Triton', 'CUTLASS'], AT = True, BT = True, dtype = th.float16, loglog=False, plot_name = 'matmul-square-tt')
    # tall-skinny
    x_tall_skinny = [64, 96, 128, 160, 192, 256, 320, 384, 512, 768, 1024, 1536]
    time_all(perf_op, x_names = ['M'], x_vals = x_tall_skinny,  y_name = 'provider', y_vals = ['torch', 'triton', 'cutlass'], 
            ylabel = 'TFLOPS', y_lines = ['Torch', 'Triton', 'CUTLASS'], AT = False, BT = False, N=2048, K=2048, dtype = th.float16, loglog=False, plot_name = 'matmul-tall-skinny-2k-2k')
    time_all(perf_op, x_names = ['M'], x_vals = x_tall_skinny,  y_name = 'provider', y_vals = ['torch', 'triton', 'cutlass'], 
            ylabel = 'TFLOPS', y_lines = ['Torch', 'Triton', 'CUTLASS'], AT = False, BT = False, N=4096, K=4096, dtype = th.float16, loglog=False, plot_name = 'matmul-tall-skinny-4k-4k')
    time_all(perf_op, x_names = ['M'], x_vals = x_tall_skinny,  y_name = 'provider', y_vals = ['torch', 'triton', 'cutlass'], 
            ylabel = 'TFLOPS', y_lines = ['Torch', 'Triton', 'CUTLASS'], AT = False, BT = False, N=6144, K=6144, dtype = th.float16, loglog=False, plot_name = 'matmul-tall-skinny-6k-6k')