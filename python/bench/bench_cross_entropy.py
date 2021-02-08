import torch
import triton

confs = [
    triton.testing.Benchmark(
              x_names = ['N'],
              x_vals  = [128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192],
              y_name  = 'provider',
              y_vals  = ['triton', 'torch'],
              y_lines = ['Triton', 'Torch'],
              ylabel  = 'GBPS',
              loglog  = False,
              plot_name = f'{mode}-2048',
              args = {'M': 2048, 'dtype': torch.float16, 'mode': mode}
    )\
    for mode in ['forward', 'backward']
]

@triton.testing.perf_report(confs)
def bench_op(M, N, dtype, mode, provider):
    # create inputs
    x = torch.randn(M, N, dtype=dtype, device='cuda', requires_grad=True)
    idx = 4 + torch.ones(M, dtype=torch.int64, device='cuda')
    num_gb = (2 * x.numel() * x.element_size() * 1e-9)
    # forward pass
    op = {'torch': torch.nn.CrossEntropyLoss(reduction='none'), \
         'triton': triton.ops.cross_entropy}[provider]
    if mode == 'forward':
        ms = triton.testing.do_bench(lambda: op(x, idx))
    if mode == 'backward':
        y = op(x, idx)
        dy = torch.randn_like(y)
        ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True))
    return num_gb / ms * 1e3

if __name__ == '__main__':
    bench_op.run('tmp', False)