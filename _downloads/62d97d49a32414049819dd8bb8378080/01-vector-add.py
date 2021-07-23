"""
Vector Addition
=================
In this tutorial, you will write a simple vector addition using Triton and learn about:

- The basic programming model of Triton
- The `triton.jit` decorator, which is used to define Triton kernels.
- The best practices for validating and benchmarking your custom ops against native reference implementations
"""

# %%
# Compute Kernel
# --------------------------

import torch
import triton.language as tl
import triton


@triton.jit
def _add(
    X,  # *Pointer* to first input vector
    Y,  # *Pointer* to second input vector
    Z,  # *Pointer* to output vector
    N,  # Size of the vector
    **meta  # Optional meta-parameters for the kernel
):
    pid = tl.program_id(0)
    # Create an offset for the blocks of pointers to be
    # processed by this program instance
    offsets = pid * meta['BLOCK'] + tl.arange(0, meta['BLOCK'])
    # Create a mask to guard memory operations against
    # out-of-bounds accesses
    mask = offsets < N
    # Load x
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    # Write back x + y
    z = x + y
    tl.store(Z + offsets, z)


# %%
# Let's also declare a helper function that to (1) allocate the output vector
# and (2) enqueueing the above kernel.


def add(x, y):
    z = torch.empty_like(x)
    N = z.shape[0]
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int]
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']), )
    # NOTE:
    #  - each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be index with a launch grid to obtain a callable GPU kernel
    #  - don't forget to pass meta-parameters as keywords arguments
    _add[grid](x, y, z, N, BLOCK=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return z


# %%
# We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
za = x + y
zb = add(x, y)
print(za)
print(zb)
print(f'The maximum difference between torch and triton is ' f'{torch.max(torch.abs(za - zb))}')

# %%
# Seems like we're good to go!

# %%
# Benchmark
# -----------
# We can now benchmark our custom op for vectors of increasing sizes to get a sense of how it does relative to PyTorch.
# To make things easier, Triton has a set of built-in utilities that allow us to concisely plot the performance of your custom ops
# for different problem sizes.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(12, 28, 1)],  # different possible values for `x_name`
        x_log=True,  # x axis is logarithmic
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['torch', 'triton'],  # possible values for `line_arg`
        line_names=["Torch", "Triton"],  # label name for the lines
        ylabel="GB/s",  # label name for the y-axis
        plot_name="vector-add-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={}  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y))
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
# We can now run the decorated function above. Pass `show_plots=True` to see the plots and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data
benchmark.run(print_data=True, show_plots=True)