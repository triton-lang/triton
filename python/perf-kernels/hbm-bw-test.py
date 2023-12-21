"""
Simple test to measure achieved HBM bandwidth.
This kernel moves N bytes of data from one region in HBM to another, using Triton.
"""

# %%
# Compute Kernel
# --------------

import torch

import triton
import triton.language as tl


@triton.jit
def copy_kernel(
    input_ptr,  # *Pointer* to input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements: tl.constexpr,  # Total elements to move.
    BLOCK_SIZE: tl.constexpr,  # Elements to load / store per iteration
    vector_size: tl.constexpr, # Size of the entire vector being moved.
    BOUNDS_CHECK: tl.constexpr, # Whether to use mask for loads.

):
    pid = tl.program_id(axis=0)
    
    lo = pid * n_elements
    hi = lo + n_elements
    for idx in range(lo, hi, BLOCK_SIZE):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        # Create a mask to guard memory operations against out-of-bounds accesses.
        if BOUNDS_CHECK:
            mask = offsets < vector_size
        in_vals = tl.load(input_ptr + offsets, mask=mask if BOUNDS_CHECK else None)
        tl.store(output_ptr + offsets, in_vals, mask=mask if BOUNDS_CHECK else None)


# %%
# Let's also declare a helper function to (1) allocate the `z` tensor
# and (2) enqueue the above kernel with appropriate grid/block sizes:


def copy(x: torch.Tensor, wgs=512, bounds_check=True):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda
    vector_size = output.numel()
    BLOCK_SIZE = 16384
    grid = (wgs, 1, 1)
    BOUNDS_CHECK = bounds_check
    # Each WG will move these many elements
    n_elements = triton.cdiv(vector_size, wgs)
    copy_kernel[grid](
        x, output,
        n_elements, BLOCK_SIZE=BLOCK_SIZE,
        vector_size=vector_size, BOUNDS_CHECK=BOUNDS_CHECK,
        num_warps=4
    )
    return output


torch.manual_seed(0)
size = 2**30
x = torch.rand(size, device='cuda')
output_torch = x
output_triton = copy(x)
print(output_torch)
print(output_triton)
print(
    f'The maximum difference between torch and triton is '
    f'{torch.max(torch.abs(output_torch - output_triton))}'
)

size = 2 ** 30

configs = []
for bounds_check in [True, False]:
    configs.append(triton.testing.Benchmark(
        x_names=['wgs'],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            (2**i) for i in range (0,12)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GiB/s',  # Label name for the y-axis.
        plot_name=f'size={size}-bounds_check={bounds_check}',  # Name for the plot. Used also as a file name for saving the plot.
        args={'size':size, 'bounds_check':bounds_check},  # Values for function arguments not in `x_names` and `y_name`.
    ))

@triton.testing.perf_report(configs)
def benchmark(size, provider, wgs, bounds_check):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.clone(x), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: copy(x, wgs, bounds_check), quantiles=quantiles)
    # 8 because 4 bytes from load, 4 from store.
    gbps = lambda ms: 8 * size / ms * 1e3 / 1024**3
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True)
