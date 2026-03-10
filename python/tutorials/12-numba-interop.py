"""
Launching Triton Kernels from Numba
====================================

This example shows how to call a Triton kernel from inside a
``@numba.njit`` compiled function. This is useful when you want to
orchestrate GPU kernel launches from Numba's compiled CPU code — for
instance, to build a pipeline that mixes Numba CPU logic with Triton
GPU kernels without returning to the Python interpreter between calls.

The key steps are:

1. Define a Triton kernel with ``@triton.jit`` as usual.
2. Call ``kernel.as_numba_kernel(signature, constexprs)`` to compile
   it for a fixed type signature and get a ``NumbaTritonKernel``.
3. Extract its ``.launch`` attribute (an ``@numba.njit`` function)
   into a module-level variable.
4. Call that launch function from inside your own ``@numba.njit`` code.

**Requirements**: ``numba``, NVIDIA GPU, CUDA driver.

**Limitations (v1)**:

- Signature and constexprs must be specified upfront (no dynamic specialization).
- No scratch memory support.
- CUDA stream must be passed as a raw ``uint64`` handle.
- NVIDIA GPUs only.
"""

# %%
# Imports
# -------

import torch
import numba

import triton
import triton.language as tl

# %%
# Triton Kernel
# -------------
# A standard vector-add kernel — nothing special here.


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


# %%
# Compile for Numba
# -----------------
# ``as_numba_kernel`` compiles the Triton kernel for a specific type
# signature, generates a thin C trampoline that calls
# ``cuLaunchKernelEx`` directly, and wraps it in an ``@numba.njit``
# function.
#
# The ``launch`` attribute is the ``@numba.njit``-compiled launcher.
# It must be extracted into a plain variable so that Numba can see it
# as a typed callable when compiling downstream ``@njit`` functions.

numba_add = add_kernel.as_numba_kernel(
    signature={
        'x_ptr': '*fp32',
        'y_ptr': '*fp32',
        'output_ptr': '*fp32',
        'n_elements': 'i32',
    },
    constexprs={'BLOCK_SIZE': 1024},
)

# Extract the @njit launcher into a module-level variable.
launch_add = numba_add.launch

# %%
# Numba Host Function
# -------------------
# Now we can call ``launch_add`` from inside ``@numba.njit``.
# The signature is:
#
# ``launch_add(gridX, gridY, gridZ, stream, x_ptr, y_ptr, output_ptr, n_elements)``
#
# Grid dimensions and stream are explicit; kernel arguments follow in
# the same order as the ``signature`` dict (minus constexprs).


@numba.njit
def add_vectors(x_ptr, y_ptr, output_ptr, n_elements, stream):
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    launch_add(grid, 1, 1, stream, x_ptr, y_ptr, output_ptr, n_elements)


# %%
# Run and Verify
# --------------

def main():
    n = 98432
    x = torch.rand(n, device='cuda', dtype=torch.float32)
    y = torch.rand(n, device='cuda', dtype=torch.float32)
    output = torch.empty_like(x)

    stream = torch.cuda.current_stream().cuda_stream

    # Warm-up (triggers Numba compilation on first call)
    add_vectors(x.data_ptr(), y.data_ptr(), output.data_ptr(), n, stream)
    torch.cuda.synchronize()

    # Verify correctness
    expected = x + y
    max_diff = (output - expected).abs().max().item()
    print(f'n = {n}')
    print(f'max |error| = {max_diff}')
    assert torch.allclose(output, expected), f'FAILED: max diff = {max_diff}'
    print('PASS ✓')

    # %%
    # Benchmark
    # ---------
    # Compare calling the Triton kernel via Numba vs. the normal
    # ``kernel[grid](...)`` Python path.

    import time

    warmup = 10
    repeats = 1000

    # --- Numba path ---
    for _ in range(warmup):
        add_vectors(x.data_ptr(), y.data_ptr(), output.data_ptr(), n, stream)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        add_vectors(x.data_ptr(), y.data_ptr(), output.data_ptr(), n, stream)
    torch.cuda.synchronize()
    numba_us = (time.perf_counter() - t0) / repeats * 1e6
    print(f'Numba  launch: {numba_us:.1f} µs / call')

    # --- Pure-Python Triton path ---
    grid = ((n + 1023) // 1024,)
    for _ in range(warmup):
        add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    torch.cuda.synchronize()
    python_us = (time.perf_counter() - t0) / repeats * 1e6
    print(f'Python launch: {python_us:.1f} µs / call')
    print(f'Speedup: {python_us / numba_us:.1f}x')


if __name__ == '__main__':
    main()
