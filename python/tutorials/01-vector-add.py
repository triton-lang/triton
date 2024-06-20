"""
Vector Addition
===============

In this tutorial, you will write a simple vector addition using Triton.

In doing so, you will learn about:

* The basic programming model of Triton.

* The `triton.jit` decorator, which is used to define Triton kernels.

* The best practices for validating and benchmarking your custom ops against native reference implementations.

"""

# %%
# Compute Kernel
# --------------

import torch

import triton
import triton.language as tl

GPU_BLOCK_SIZE = 1024
CPU_BLOCK_SIZE = 4096
USE_GPU = False


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


# %%
# Let's also declare a helper function to (1) allocate the `z` tensor
# and (2) enqueue the above kernel with appropriate grid/block sizes:


def add(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor, is_cpu):
    if output is None:
        # We need to preallocate the output.
        output = torch.empty_like(x)
        assert x.is_cpu == is_cpu and y.is_cpu == is_cpu and output.is_cpu == is_cpu
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=CPU_BLOCK_SIZE if is_cpu else GPU_BLOCK_SIZE)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


# %%
# We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:
torch.manual_seed(0)
size = 98432

triton.runtime.driver.set_active_to_cpu()
x = torch.rand(size, device='cpu')
y = torch.rand(size, device='cpu')
output_torch_cpu = torch.add(x, y)
output_triton_cpu = add(x, y, None, is_cpu=True)
print(output_torch_cpu)
print(output_triton_cpu)
print(f'The maximum difference between torch-cpu and triton-cpu is '
      f'{torch.max(torch.abs(output_torch_cpu - output_triton_cpu))}')

LINE_VALS = ['triton-cpu-single', 'triton-cpu', 'torch-cpu']
LINE_NAMES = ['TritonCPU 1', 'TritonCPU', 'TorchCPU']
LINE_STYLES = [('blue', '--'), ('blue', '-'), ('green', '-')]

if USE_GPU and triton.runtime.driver.get_active_gpus():
    triton.runtime.driver.set_active_to_gpu()
    x = x.to('cuda')
    y = y.to('cuda')
    output_torch_gpu = x + y
    output_triton_gpu = add(x, y, None, is_cpu=False)
    print(output_torch_gpu)
    print(output_triton_gpu)
    print(f'The maximum difference between torch-gpu and triton-gpu is '
          f'{torch.max(torch.abs(output_torch_gpu - output_triton_gpu))}')

    LINE_VALS += ['triton-gpu', 'torch-gpu']
    LINE_NAMES += ['TritonGPU', 'TorchGPU']
    LINE_STYLES += [('yellow', '-'), ('red', '-')]

# %%
# Seems like we're good to go!

# %%
# Benchmark
# ---------
#
# We can now benchmark our custom op on vectors of increasing sizes to get a sense of how it does relative to PyTorch.
# To make things easier, Triton has a set of built-in utilities that allow us to concisely plot the performance of our custom ops.
# for different problem sizes.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=LINE_VALS,  # Possible values for `line_arg`.
        line_names=LINE_NAMES,  # Label name for the lines.
        styles=LINE_STYLES,  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name=
        # Name for the plot. Used also as a file name for saving the plot.
        f'vector-add-performance (CPU_BLOCK_SIZE={CPU_BLOCK_SIZE}, GPU_BLOCK_SIZE={GPU_BLOCK_SIZE})',
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    import os

    device = 'cpu' if 'cpu' in provider else 'cuda'
    x = torch.rand(size, device=device, dtype=torch.float32)
    y = torch.rand(size, device=device, dtype=torch.float32)

    if device == 'cpu':
        is_cpu = True
        triton.runtime.driver.set_active_to_cpu()
        if 'single' in provider:
            os.environ['TRITON_CPU_SINGLE_CORE'] = '1'
        else:
            os.unsetenv('TRITON_CPU_SINGLE_CORE')
    else:
        is_cpu = False
        triton.runtime.driver.set_active_to_gpu()

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-gpu':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles, is_cpu=is_cpu)
    elif provider == 'triton-gpu':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y, None, False), quantiles=quantiles, is_cpu=is_cpu)
    elif provider == 'torch-cpu':
        # Note that we preallocate the output buffer here to only measure the kernel performance
        # without a large chunk of memory allocation.
        output = torch.empty_like(x)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.add(x, y, out=output), quantiles=quantiles,
                                                     is_cpu=is_cpu)
    elif provider == 'triton-cpu-single':
        output = torch.empty_like(x)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y, output, True), quantiles=quantiles,
                                                     is_cpu=is_cpu)
    elif provider == 'triton-cpu':
        output = torch.empty_like(x)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y, output, True), quantiles=quantiles,
                                                     is_cpu=is_cpu)
    gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=True)
