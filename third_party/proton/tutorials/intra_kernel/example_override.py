"""
Vector Addition with Triton Intra-Kernel Profiling using TTGIR Override

This tutorial demonstrates how to use Triton's TTGIR override mechanism
to enable intra-kernel profiling with Proton. The workflow involves generating,
modifying, and overriding the kernel's intermediate representation to insert
profiling hooks.

Workflow:
1. Generate TTGIR dump files:

   This creates the original TTGIR files in the `ttgir_dump/` directory:

   ../../scripts/dump_ttgir.sh python3 example_override.py --increase-accuracy

2. Insert profiling instrumentation:

   Modify the generated TTGIR files by adding proton.record operators at desired
   profiling points. Example script that adds proton ops in the above ttgir:

   ./insert_proton_records

3. Execute with TTGIR override:

   TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=ttgir_dump python3 example_override.py --increase-accuracy

   - TRITON_ALWAYS_COMPILE=1: Forces recompilation on each run
   - TRITON_KERNEL_OVERRIDE=1: Enables TTGIR override mechanism
   - TRITON_OVERRIDE_DIR=ttgir_dump: Specifies directory containing modified TTGIR files
"""

import argparse

import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton.profiler.mode import Default

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    parser = argparse.ArgumentParser(description="TTGIR override example with Triton intra kernel profiling")
    parser.add_argument(
        "--increase-accuracy",
        action="store_true",
        default=False,
        help="Enable increased-accuracy during profiling (default: False)",
    )
    args = parser.parse_args()

    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    if args.increase_accuracy:
        proton.start(
            "add",
            data="trace",
            backend="instrumentation",
            mode=Default(optimizations="clock32,time_shift"),
        )
    else:
        proton.start("add", data="trace", backend="instrumentation")

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    proton.finalize()
    return output


torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
torch.testing.assert_close(output_torch, output_triton, rtol=1e-3, atol=1e-1)
