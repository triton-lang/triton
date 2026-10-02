import argparse
import time
import torch


def cuda_time(fn, iters=30, warmup=20):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for benchmarking")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    return ms / iters


def bench_case(shape, dtype, training):
    from triton_kernels.batchnorm import batchnorm_forward

    device = "cuda"
    x = torch.randn(*shape, device=device, dtype=dtype)
    if x.ndim == 2:
        C = x.shape[1]
    else:
        C = x.shape[1]
    gamma = torch.randn(C, device=device, dtype=torch.float32)
    beta = torch.randn(C, device=device, dtype=torch.float32)
    running_mean = torch.zeros(C, device=device, dtype=torch.float32)
    running_var = torch.ones(C, device=device, dtype=torch.float32)
    eps = 1e-5
    momentum = 0.1

    def fn_triton():
        y, m, v = batchnorm_forward(
            x, gamma, beta, eps=eps, training=training,
            running_mean=running_mean, running_var=running_var, momentum=momentum, layout="NCHW"
        )
        return y

    def fn_torch():
        y = torch.nn.functional.batch_norm(
            x.float(),
            None if training else running_mean,
            None if training else running_var,
            gamma, beta, training=training, momentum=momentum, eps=eps,
        ).to(x.dtype)
        return y

    t_triton = cuda_time(fn_triton)
    t_torch = cuda_time(fn_torch)
    return t_triton, t_torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    shapes = [
        (64, 128),
        (64, 128, 32, 32),
        (32, 256, 64, 64),
    ]
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    modes = [True, False]  # training, eval

    header = ["shape", "dtype", "mode", "triton_ms", "torch_ms", "speedup(x)"]
    print("\t".join(header))
    for shape in shapes:
        for dtype in dtypes:
            for training in modes:
                try:
                    t_tri, t_ref = bench_case(shape, dtype, training)
                    speedup = t_ref / max(t_tri, 1e-6)
                    print(f"{shape}\t{str(dtype).split('.')[-1]}\t{'train' if training else 'eval'}\t{t_tri:.3f}\t{t_ref:.3f}\t{speedup:.2f}")
                except Exception as e:
                    print(f"{shape}\t{str(dtype).split('.')[-1]}\t{'train' if training else 'eval'}\tERROR\tERROR\t{e}")


if __name__ == "__main__":
    main()


