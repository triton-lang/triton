import torch
import triton

# -------------------------------
# Matrix Multiplication
# -------------------------------

nt = {False: 'n', True: 't'}
square_confs = [
    triton.testing.Benchmark(
              x_names = ['M', 'N', 'K'],
              x_vals  = [128, 256, 512, 1024, 2048, 3072, 4096, 6144],
              line_arg  = 'block',
              line_vals  = [16, 32, 64, 128],
              line_names = ['Block16', 'Block32', 'Block64', 'Block128'],
              ylabel  = 'TFLOPS',
              plot_name = f'{op_mode}-{layout_mode}-square-{nt[AT]}{nt[BT]}',
              args = {'layout_mode': layout_mode, 'op_mode': op_mode,
                      'AT': AT, 'BT': BT, 'dtype': torch.float16, 'provider': 'triton'}
    )\
    for AT in [False] for BT in [False] \
    for op_mode in ['dsd']  for layout_mode in ['dense']
]


@triton.testing.perf_report(square_confs)
def bench_matmul(M, N, K, block, layout_mode, op_mode, AT, BT, dtype, provider, warmup=100, rep=1000):
    Z, H = 1, 1
    make_layout = {
        'tril': lambda H, M, N: torch.tril(torch.ones((H, M, N), dtype=torch.int64)),\
        'dense': lambda H, M, N: torch.ones(H, M, N, dtype=torch.int64),
    }[layout_mode]
    # create layout
    shape = {'sdd': (M, N), 'dsd': (K, M) if AT else (M, K), 'dds': (N, K) if BT else (K, N)}[op_mode]
    layout = make_layout(H, shape[0] // block, shape[1] // block)
    # creat inputs
    a = torch.randn((Z, H, K, M) if AT else (Z, H, M, K), dtype=dtype, device='cuda')
    b = torch.randn((Z, H, N, K) if BT else (Z, H, K, N), dtype=dtype, device='cuda')
    # create op
    tflops = lambda ms: num_flops / ms * 1e3
    if provider == 'triton':
        op = triton.ops.blocksparse.matmul(layout, block, op_mode, trans_a=AT, trans_b=BT)
        # inputs
        a = triton.testing.sparsify_tensor(a, layout, block) if op_mode == 'dsd' else a
        b = triton.testing.sparsify_tensor(b, layout, block) if op_mode == 'dds' else b
        mean_ms, min_ms, max_ms = triton.testing.do_bench(lambda: op(a, b), warmup=warmup, rep=rep)
        num_flops = {
            'sdd': 2 * Z * K * float(layout.sum()) * block * block,\
            'dsd': 2 * Z * N * float(layout.sum()) * block * block,\
            'dds': 2 * Z * M * float(layout.sum()) * block * block
        }[op_mode]*1e-12
        return tflops(mean_ms), tflops(min_ms), tflops(max_ms)


# -------------------------------
# Softmax
# -------------------------------

square_confs = [
    triton.testing.Benchmark(
              x_names = ['M', 'N'],
              x_vals  = [128, 256, 512, 1024, 2048, 3072, 4096, 6144],
              line_arg  = 'block',
              line_vals  = [16, 32, 64],
              line_names = ['Block16', 'Block32', 'Block64'],
              ylabel  = 'GBPS',
              plot_name = f'{layout_mode}-square',
              args = {'layout_mode': layout_mode, 'dtype': torch.float16, 'provider': 'triton'}
    )\
    for layout_mode in ['dense', 'tril']
]


@triton.testing.perf_report(square_confs)
def bench_softmax(M, N, block, layout_mode, dtype, provider, warmup=10, rep=50):
    Z, H = 1, 1
    make_layout = {
        'tril': lambda H, M, N: torch.tril(torch.ones((H, M, N), dtype=torch.int64)),
        'dense': lambda H, M, N: torch.ones(H, M, N, dtype=torch.int64),
    }[layout_mode]
    layout = make_layout(H, M // block, N // block)
    a = torch.randn((Z, H, M, N), dtype=dtype, device='cuda')
    if provider == 'triton':
        a = triton.testing.sparsify_tensor(a, layout, block)
        op = triton.ops.blocksparse.softmax(layout, block)
        gbps = lambda ms: (2 * a.numel() * a.element_size() * 1e-9) / (ms * 1e-3)
        mean_ms, min_ms, max_ms = triton.testing.do_bench(lambda: op(a), warmup=warmup, rep=rep)
        return gbps(mean_ms), gbps(min_ms), gbps(max_ms)


bench_matmul.run(print_data=True, show_plots=True)