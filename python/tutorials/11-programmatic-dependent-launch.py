"""
Programmatic Dependent Launch
=====================
This script demonstrates the use of programmatic dependent launch ontop of the vector-add example using Triton.

.. code-block:: bash
    python 11-programmatic-dependent-launch.py
"""

import torch
import triton
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_pdl():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


# In this example
@triton.jit
def add_kernel(x_ptr,  #
               y_ptr,  #
               output_ptr,  #
               n_elements,  #
               BLOCK_SIZE: tl.constexpr,  #
               USE_GDC: tl.constexpr,  #
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    if USE_GDC:
        # GDC wait is used to wait for the prior kernel to complete before continuing.
        # If we utilize Programmatic Dependent Launch, we must wait on the prior kernel
        # to complete in case a or b are written to by the prior kernel.
        # This is done to prevent races.
        tl.extra.cuda.gdc_wait()

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    if USE_GDC:
        # GDC launch dependents is used to launch dependent kernels.
        # Once GDC launch it is possible for the next kernel to begin if
        # there are enough resources.
        tl.extra.cuda.gdc_launch_dependents()
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, launch_pdl: bool = True):
    output = torch.empty_like(x)
    assert x.device == y.device and output.device == x.device
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](
        x, y, output, n_elements, BLOCK_SIZE=1024,
        USE_GDC=launch_pdl,  # set constexpr in kernel to use grid dependence control
        launch_pdl=launch_pdl,  # launch kernel with PDL flag set enabled
    )
    return output


def validate(n_elements):
    x = torch.rand(n_elements, device="cuda", dtype=torch.float32)
    y = torch.rand(n_elements, device="cuda", dtype=torch.float32)

    torch_result = x + y
    add_result = add(x, y)

    torch_vs_add = "✅" if torch.allclose(torch_result, add_result, atol=1.0) else "❌"
    print(f"Number of Elements={n_elements} verification naive vs: ", end="")
    print(f"add: {torch_vs_add}")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(23, 28, 1)],
        x_log=False,
        line_arg="provider",
        line_vals=["pdl-fp32", "fp32"],
        line_names=["PDL", "No PDL"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel='GB/s',
        plot_name="pdl-performance",
        args={},
    ))
def benchmark(size, provider):
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]

    fn = lambda: add(x, y, "pdl" in provider)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles, rep=100)

    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":

    if supports_pdl():
        validate(1024)
        benchmark.run(print_data=True, show_plots=True, save_path=".")
    else:
        print("PDL is not supported on this device")
