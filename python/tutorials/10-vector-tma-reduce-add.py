import torch

import triton
import triton.language as tl
from triton.tools.experimental_descriptor import (
    create_1d_tma_descriptor,
)


def map_dtype_to_triton(dtype: torch.dtype) -> int:
    """
    Maps torch dtype to triton dtype.

    Args:
        dtype (torch.dtype): input dtype.

    Returns:
        tl.dtype: triton dtype.
    """
    if dtype == torch.float16:
        return 0
    elif dtype == torch.bfloat16:
        return 1
    elif dtype == torch.float32:
        return 2
    elif dtype == torch.int32:
        return 3
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    x_desc,
    y_ptr,  # *Pointer* to second input vector.
    y_desc,
    output_desc,
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    # Load x through TMA.
    x = tl._experimental_descriptor_load(
        x_desc, [block_start], [BLOCK_SIZE], x_ptr.dtype.element_ty
    )
    # Store x to through TMA.
    tl._experimental_descriptor_store(output_desc, x, [block_start])
    # Load y through TMA.
    y = tl._experimental_descriptor_load(
        y_desc, [block_start], [BLOCK_SIZE], y_ptr.dtype.element_ty
    )
    # Store y to through TMA reduce add.
    tl._experimental_descriptor_store(output_desc, y, [block_start], store_reduce="add")


def add(x: torch.Tensor, y: torch.Tensor):
    BLOCK_SIZE = 256
    x_desc = create_1d_tma_descriptor(
        x.data_ptr(), size, BLOCK_SIZE, map_dtype_to_triton(x.dtype)
    )
    y_desc = create_1d_tma_descriptor(
        y.data_ptr(), size, BLOCK_SIZE, map_dtype_to_triton(y.dtype)
    )
    output = torch.empty_like(x)
    output_desc = create_1d_tma_descriptor(
        output.data_ptr(), size, BLOCK_SIZE, map_dtype_to_triton(output.dtype)
    )
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, x_desc, y, y_desc, output_desc, BLOCK_SIZE=BLOCK_SIZE)
    return output


torch.manual_seed(0)
size = 98432
x = torch.rand(size, dtype=torch.float32, device="cuda")
y = torch.rand(size, dtype=torch.float32, device="cuda")
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(
    f"The maximum difference between torch and triton is "
    f"{torch.max(torch.abs(output_torch - output_triton))}"
)
assert torch.equal(output_torch, output_triton)

# %%
# Benchmark
# ---------
#
# We can now benchmark our custom op on vectors of increasing sizes to get a sense of how it does relative to PyTorch.
# To make things easier, Triton has a set of built-in utilities that allow us to concisely plot the performance of our custom ops.
# for different problem sizes.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2**i for i in range(12, 28, 1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "torch"],  # Possible values for `line_arg`.
        line_names=["Triton", "Torch"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="vector-add-float32-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y), quantiles=quantiles
        )
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# # %%
# # We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# # `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=True)
