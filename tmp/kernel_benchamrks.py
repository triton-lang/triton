from typing import Dict

import torch
import triton
from triton.compiler import CompiledKernel
import pandas as pd
import matplotlib.pyplot as plt

import utils

kernels = utils.get_kernels()


def benchmark(m: int = 1024) -> Dict[str, float]:
    torch.cuda.synchronize()
    n = 1024
    rep = 10_000
    benchmark_warmup = 500
    x = torch.randn((m, n), device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)

    res = dict()
    for name, (kernel_short, kernel_long) in kernels.items():
        for kernel_warmup in [True, False]:
            args = (x, 2.0, out, N, x.stride(0), BLOCK_SIZE)
            kwargs = {"num_warps": 4, "warmup": kernel_warmup}
            key_short = f"{name}_short{'_warmup' if kernel_warmup else ''}"
            res[key_short] = utils.do_bench(
                lambda: kernel_short[(M,)](*args, **kwargs),
                rep=rep,
                warmup=benchmark_warmup,
            )

            args = (
                x,
                2.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                out,
                N,
                x.stride(0),
                BLOCK_SIZE,
            )
            kwargs = {"num_warps": 4, "warmup": kernel_warmup}
            key_long = f"{name}_long{'_warmup' if kernel_warmup else ''}"
            res[key_long] = utils.do_bench(
                lambda: kernel_long[(M,)](*args, **kwargs),
                rep=rep,
                warmup=benchmark_warmup,
            )
        # endfor
    # endfor

    kernel_short = kernels["default"][0]
    cache = kernel_short.cache[0]
    key = list(cache.keys())[0]
    kernel = cache[key]

    if utils.IS_TRITON_VERSION_OLD:
        launch_args = (
            M,  # grid_0
            1,  # grid_1
            1,  # grid_2
            4,  # bin.num_warps
            1,  # bin.num_ctas
            1,  # bin.clusterDims[0]
            1,  # bin.clusterDims[1]
            1,  # bin.clusterDims[2]
            0,  # bin.shared
            0,  # stream
            kernel.cu_function,
            CompiledKernel.launch_enter_hook,
            CompiledKernel.launch_exit_hook,
            kernel,
            # Calling args
            x,
            2.0,
            out,
            N,
            x.stride(0),
        )
        c_wrapper = kernel.c_wrapper
        kernel_bench = lambda: c_wrapper(*launch_args)
    else:
        args = (x, 2.0, out, N, x.stride(0))
        launch_metadata = kernel.launch_metadata((M, 1, 1), 0, *args)
        launch_args = (
            M,
            1,
            1,
            0,
            kernel.function,
            kernel.packed_metadata,
            launch_metadata,
            CompiledKernel.launch_enter_hook,
            CompiledKernel.launch_exit_hook,
            *args,
        )
        run = kernel.run
        kernel_bench = lambda: run(*launch_args)

    res["noop"] = utils.do_bench(lambda: None, rep=rep, warmup=benchmark_warmup)
    res["kernel"] = utils.do_bench(
        kernel_bench,
        rep=rep,
        warmup=benchmark_warmup,
    )

    return res


if __name__ == "__main__":
    res = []
    if utils.TRITON_MAJOR == 2:
        num_iterations = 12
    else:
        num_iterations = 16
    for m in range(1, num_iterations):
        print(f"Running for {m}/{num_iterations}")
        m *= 128
        _res = benchmark(m)
        _res["m"] = m
        res.append(_res)

    df = pd.DataFrame(res)
    df = df.set_index("m")
    df *= 1000
    df.to_csv(f"data/kernel_time_triton_{utils.TRITON_MAJOR}.csv")
    plt.figure()

    for name in df.columns:
        if "warmup" not in name:
            plt.plot(df.index, df[name], label=name)

    plt.legend()
    plt.xlabel("Input length")
    plt.ylabel("Run time (us)")
    plt.savefig(f"figures/kernel_time_triton_{utils.TRITON_MAJOR}.png", dpi=200)

    df = df[[_ for _ in df.columns if "warmup" in _]]
    df.to_csv(f"data/kernel_overhead_triton_{utils.TRITON_MAJOR}.csv")
    print(df.describe())
