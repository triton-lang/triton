import low_latency_jit
import pandas as pd
import torch
from torch.autograd import Function
from tqdm import tqdm

import triton
import triton.language as tl
from triton.testing import do_bench

pd.options.display.float_format = "{:,.2f}".format


def get_functions(jit):
    # y = a * x
    @jit
    def fwd_kernel(
        x_ptr,
        a,
        output_ptr,
        N,
        M_STRIDE,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        x_ptr += pid * M_STRIDE
        output_ptr += pid * M_STRIDE
        offsets = tl.arange(0, BLOCK_SIZE_N)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x * a, mask=mask)

    # dx = dy / a
    @jit
    def bwd_kernel(
        dy_ptr,
        a,
        output_ptr,
        N,
        M_STRIDE,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        dy_ptr += pid * M_STRIDE
        output_ptr += pid * M_STRIDE
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        dy = tl.load(dy_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, dy / a, mask=mask)

    # y = a * x
    @jit
    def fwd_kernel_useless(
        x_ptr,
        a,
        b,
        c,
        d,
        e,
        f,
        g,
        h,
        output_ptr,
        N,
        M_STRIDE,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        x_ptr += pid * M_STRIDE
        output_ptr += pid * M_STRIDE
        offsets = tl.arange(0, BLOCK_SIZE_N)
        mul = a * b * c * d * e * f * g * h
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x * mul, mask=mask)

    # dx = dy / a
    @jit
    def bwd_kernel_useless(
        dy_ptr,
        a,
        b,
        c,
        d,
        e,
        f,
        g,
        h,
        output_ptr,
        N,
        M_STRIDE,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        dy_ptr += pid * M_STRIDE
        output_ptr += pid * M_STRIDE
        offsets = tl.arange(0, BLOCK_SIZE)
        mul = a * b * c * d * e * f * g * h
        mask = offsets < N
        dy = tl.load(dy_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, dy / mul, mask=mask)

    class MyMulFunction(Function):
        @staticmethod
        def forward(ctx, x):
            M, N = x.shape
            BLOCK_SIZE = triton.next_power_of_2(N)
            out = torch.empty_like(x)
            fwd_kernel[(M,)](
                x,
                1.0,
                out,
                N,
                x.stride(0),
                BLOCK_SIZE,
                num_warps=4,
                device_type="cuda",
                # device=0,
                # stream=0,
            )
            return out

        @staticmethod
        def backward(ctx, dy):
            M, N = dy.shape
            BLOCK_SIZE = triton.next_power_of_2(N)
            dx = None
            if ctx.needs_input_grad[0]:
                dx = torch.empty_like(dy)
                bwd_kernel[(M,)](
                    dy,
                    1.0,
                    dx,
                    N,
                    dy.stride(0),
                    BLOCK_SIZE,
                    num_warps=4,
                    device_type="cuda",
                    # device=0,
                    # stream=0,
                )
            return dx

    class MyMulFunctionUseless(Function):
        @staticmethod
        def forward(ctx, x):
            M, N = x.shape
            BLOCK_SIZE = triton.next_power_of_2(N)
            out = torch.empty_like(x)
            fwd_kernel_useless[(M,)](
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
                num_warps=4,
                device_type="cuda",
                # device=0,
                # stream=0,
            )
            return out

        @staticmethod
        def backward(ctx, dy):
            M, N = dy.shape
            BLOCK_SIZE = triton.next_power_of_2(N)
            dx = None
            if ctx.needs_input_grad[0]:
                dx = torch.empty_like(dy)
                bwd_kernel_useless[(M,)](
                    dy,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    dx,
                    N,
                    dy.stride(0),
                    BLOCK_SIZE,
                    num_warps=4,
                    device_type="cuda",
                    # device=0,
                    # stream=0,
                )
            return dx

    return (
        MyMulFunction,
        MyMulFunctionUseless,
    )


f1, f2 = get_functions(triton.jit)
f3, f4 = get_functions(low_latency_jit.jit)


def benchmark():
    torch.cuda.synchronize()
    m = 1024
    n = 1024
    x = torch.nn.Parameter(torch.randn((m, n), device="cuda", dtype=torch.float32))
    dy = torch.randn_like(x, dtype=torch.float32)
    return (
        do_bench(lambda: None, rep=10000, warmup=100),
        do_bench(lambda: (x * 2).backward(dy), rep=5000, warmup=500, grad_to_none=x),
        do_bench(
            lambda: f1.apply(x).backward(dy), rep=5000, warmup=500, grad_to_none=x
        ),
        do_bench(
            lambda: f2.apply(x).backward(dy), rep=5000, warmup=500, grad_to_none=x
        ),
        do_bench(
            lambda: f3.apply(x).backward(dy), rep=5000, warmup=500, grad_to_none=x
        ),
        do_bench(lambda: f4.apply(x).backward(dy), rep=5000, warmup=1000),
    )


if __name__ == "__main__":
    res = []
    for _ in tqdm(range(10)):
        res.append(benchmark())

    df = pd.DataFrame(
        res,
        columns=[
            "noop",
            "baseline",
            "short_params",
            "long_params",
            "short_params_optimized",
            "long_params_optimized",
        ],
    )
    df *= 1000
    df.to_csv("res.csv", index=False)

    print(df.describe())
