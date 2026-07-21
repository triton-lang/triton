"""
Simple Gluon stream copy kernel for GFX1250 (Babel style).
- FP16 data input/output
- 4 warps per CTA (4 wave workgroup)
- 128-bit read/write (8 x fp16 elements per lane)
"""

import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
import pytest

# Constants for GFX1250
THREADS_PER_WARP = 32  # GFX1250 warp size
ELEMENTS_PER_THREAD = 8  # 128 bits / 16 bits per fp16 = 8 elements
WARPS_PER_CTA = 4  # 4 warps per CTA.
BLOCK_SIZE = ELEMENTS_PER_THREAD * THREADS_PER_WARP * WARPS_PER_CTA  # 1024 elements per workgroup (4 warps)


@gluon.jit
def stream_unary_kernel(
    a_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: gl.constexpr,
    NUM_WGS: gl.constexpr,
):
    """Grid-stride unary stream: one input load stream and one output store stream."""
    wg_id = gl.program_id(0)
    start_offset = wg_id * BLOCK_SIZE
    grid_stride = NUM_WGS * BLOCK_SIZE
    remaining = gl.maximum(N - start_offset, 0)
    num_iters = gl.cdiv(remaining, grid_stride)

    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [1, 4], [1, 0])
    offs_col = gl.arange(0, BLOCK_SIZE, layout=gl.SliceLayout(0, layout))

    for i in range(0, num_iters):
        offsets = start_offset + i * grid_stride + offs_col
        mask = offsets < N
        a = gl.amd.gfx1250.buffer_load(a_ptr, offsets, mask=mask)
        gl.amd.gfx1250.buffer_store(a, out_ptr, offsets, mask=mask)


@gluon.jit
def stream_binary_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: gl.constexpr,
    NUM_WGS: gl.constexpr,
):
    """Grid-stride binary stream: two input load streams and one output store stream."""
    wg_id = gl.program_id(0)
    start_offset = wg_id * BLOCK_SIZE
    grid_stride = NUM_WGS * BLOCK_SIZE
    remaining = gl.maximum(N - start_offset, 0)
    num_iters = gl.cdiv(remaining, grid_stride)

    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [1, 4], [1, 0])
    offs_col = gl.arange(0, BLOCK_SIZE, layout=gl.SliceLayout(0, layout))

    for i in range(0, num_iters):
        offsets = start_offset + i * grid_stride + offs_col
        mask = offsets < N
        a = gl.amd.gfx1250.buffer_load(a_ptr, offsets, mask=mask)
        b = gl.amd.gfx1250.buffer_load(b_ptr, offsets, mask=mask)
        out = a + b
        gl.amd.gfx1250.buffer_store(out, out_ptr, offsets, mask=mask)


@gluon.jit
def stream_ternary_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: gl.constexpr,
    NUM_WGS: gl.constexpr,
):
    """Grid-stride ternary stream: three input load streams and one output store stream."""
    wg_id = gl.program_id(0)
    start_offset = wg_id * BLOCK_SIZE
    grid_stride = NUM_WGS * BLOCK_SIZE
    remaining = gl.maximum(N - start_offset, 0)
    num_iters = gl.cdiv(remaining, grid_stride)

    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [1, 4], [1, 0])
    offs_col = gl.arange(0, BLOCK_SIZE, layout=gl.SliceLayout(0, layout))

    for i in range(0, num_iters):
        offsets = start_offset + i * grid_stride + offs_col
        mask = offsets < N
        a = gl.amd.gfx1250.buffer_load(a_ptr, offsets, mask=mask)
        b = gl.amd.gfx1250.buffer_load(b_ptr, offsets, mask=mask)
        c = gl.amd.gfx1250.buffer_load(c_ptr, offsets, mask=mask)
        out = a + b + c
        gl.amd.gfx1250.buffer_store(out, out_ptr, offsets, mask=mask)


@gluon.jit
def stream_unary_pipelined_kernel(
    a_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: gl.constexpr,
    NUM_WGS: gl.constexpr,
):
    """Unary stream with next-iteration load issued before current store."""
    wg_id = gl.program_id(0)
    start_offset = wg_id * BLOCK_SIZE
    grid_stride = NUM_WGS * BLOCK_SIZE
    remaining = gl.maximum(N - start_offset, 0)
    num_iters = gl.cdiv(remaining, grid_stride)

    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [1, 4], [1, 0])
    offs_col = gl.arange(0, BLOCK_SIZE, layout=gl.SliceLayout(0, layout))

    offsets = start_offset + offs_col
    mask = offsets < N
    a = gl.amd.gfx1250.buffer_load(a_ptr, offsets, mask=mask)

    for i in range(0, num_iters):
        offsets_curr = start_offset + i * grid_stride + offs_col
        mask_curr = offsets_curr < N
        offsets_next = start_offset + (i + 1) * grid_stride + offs_col
        mask_next = offsets_next < N
        a_next = gl.amd.gfx1250.buffer_load(a_ptr, offsets_next, mask=mask_next)
        gl.amd.gfx1250.buffer_store(a, out_ptr, offsets_curr, mask=mask_curr)
        a = a_next


@gluon.jit
def stream_binary_pipelined_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: gl.constexpr,
    NUM_WGS: gl.constexpr,
):
    """Binary stream with next-iteration loads issued before current store."""
    wg_id = gl.program_id(0)
    start_offset = wg_id * BLOCK_SIZE
    grid_stride = NUM_WGS * BLOCK_SIZE
    remaining = gl.maximum(N - start_offset, 0)
    num_iters = gl.cdiv(remaining, grid_stride)

    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [1, 4], [1, 0])
    offs_col = gl.arange(0, BLOCK_SIZE, layout=gl.SliceLayout(0, layout))

    offsets = start_offset + offs_col
    mask = offsets < N
    a = gl.amd.gfx1250.buffer_load(a_ptr, offsets, mask=mask)
    b = gl.amd.gfx1250.buffer_load(b_ptr, offsets, mask=mask)

    for i in range(0, num_iters):
        offsets_curr = start_offset + i * grid_stride + offs_col
        offsets_next = start_offset + (i + 1) * grid_stride + offs_col
        mask_curr = offsets_curr < N
        mask_next = offsets_next < N
        a_next = gl.amd.gfx1250.buffer_load(a_ptr, offsets_next, mask=mask_next)
        b_next = gl.amd.gfx1250.buffer_load(b_ptr, offsets_next, mask=mask_next)
        gl.amd.gfx1250.buffer_store(a + b, out_ptr, offsets_curr, mask=mask_curr)
        a = a_next
        b = b_next


@gluon.jit
def stream_ternary_pipelined_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: gl.constexpr,
    NUM_WGS: gl.constexpr,
):
    """Ternary stream with next-iteration loads issued before current store."""
    wg_id = gl.program_id(0)
    start_offset = wg_id * BLOCK_SIZE
    grid_stride = NUM_WGS * BLOCK_SIZE
    remaining = gl.maximum(N - start_offset, 0)
    num_iters = gl.cdiv(remaining, grid_stride)

    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [1, 4], [1, 0])
    offs_col = gl.arange(0, BLOCK_SIZE, layout=gl.SliceLayout(0, layout))

    offsets = start_offset + offs_col
    mask = offsets < N
    a = gl.amd.gfx1250.buffer_load(a_ptr, offsets, mask=mask)
    b = gl.amd.gfx1250.buffer_load(b_ptr, offsets, mask=mask)
    c = gl.amd.gfx1250.buffer_load(c_ptr, offsets, mask=mask)

    for i in range(0, num_iters):
        offsets_curr = start_offset + i * grid_stride + offs_col
        offsets_next = start_offset + (i + 1) * grid_stride + offs_col
        mask_curr = offsets_curr < N
        mask_next = offsets_next < N
        a_next = gl.amd.gfx1250.buffer_load(a_ptr, offsets_next, mask=mask_next)
        b_next = gl.amd.gfx1250.buffer_load(b_ptr, offsets_next, mask=mask_next)
        c_next = gl.amd.gfx1250.buffer_load(c_ptr, offsets_next, mask=mask_next)
        gl.amd.gfx1250.buffer_store(a + b + c, out_ptr, offsets_curr, mask=mask_curr)
        a = a_next
        b = b_next
        c = c_next


def reference_torch_nary(inputs):
    ref = inputs[0]
    for tensor in inputs[1:]:
        ref = ref + tensor
    return ref


def run_stream_nary(kernel_fn, N: int, arity: int, check: bool = True, sync: bool = True, num_workgroups: int = 1024,
                    block_size: int = BLOCK_SIZE, num_warps: int = WARPS_PER_CTA):
    """
    Run an N-input memory stream kernel.

    Unary writes `a`, binary writes `a + b`, and ternary writes `a + b + c`.
    The tiny arithmetic keeps all input streams live while leaving the kernel
    dominated by global memory traffic.
    """
    dtype = torch.float16
    torch.random.manual_seed(42)
    inputs = [torch.rand(N, dtype=dtype) for _ in range(arity)]
    out = torch.empty(N, dtype=dtype)

    if check:
        ref = reference_torch_nary(inputs)

    inputs = [tensor.cuda() for tensor in inputs]
    out = out.cuda()

    if arity == 1:
        kernel = kernel_fn[(num_workgroups, )](
            inputs[0],
            out,
            N,
            block_size,
            num_workgroups,
            num_warps=num_warps,
            waves_per_eu=1,
        )
    elif arity == 2:
        kernel = kernel_fn[(num_workgroups, )](
            inputs[0],
            inputs[1],
            out,
            N,
            block_size,
            num_workgroups,
            num_warps=num_warps,
            waves_per_eu=1,
        )
    else:
        kernel = kernel_fn[(num_workgroups, )](
            inputs[0],
            inputs[1],
            inputs[2],
            out,
            N,
            block_size,
            num_workgroups,
            num_warps=num_warps,
            waves_per_eu=1,
        )

    if sync:
        torch.cuda.synchronize()

    if check:
        out_cpu = out.cpu()
        torch.testing.assert_close(out_cpu, ref, rtol=0, atol=0)
        print(f"PASSED: {arity}-input stream of {N} fp16 elements")

    return kernel


def select_kernel(nary: int, pipelined: bool):
    kernels = {
        (1, False): (stream_unary_kernel, "stream_unary_kernel"),
        (2, False): (stream_binary_kernel, "stream_binary_kernel"),
        (3, False): (stream_ternary_kernel, "stream_ternary_kernel"),
        (1, True): (stream_unary_pipelined_kernel, "stream_unary_pipelined_kernel"),
        (2, True): (stream_binary_pipelined_kernel, "stream_binary_pipelined_kernel"),
        (3, True): (stream_ternary_pipelined_kernel, "stream_ternary_pipelined_kernel"),
    }
    try:
        return kernels[(nary, pipelined)]
    except KeyError:
        raise ValueError(f"Unsupported stream configuration: nary={nary}, pipelined={pipelined}") from None


@pytest.mark.parametrize("N", [32768, 500], ids=lambda N: f"N={N}")
@pytest.mark.parametrize("pipelined", [False, True], ids=["non_pipelined", "pipelined"])
@pytest.mark.parametrize("nary", [1, 2, 3], ids=["unary", "binary", "ternary"])
def test_stream_copy(nary, pipelined, N):
    """Test stream copy kernel correctness."""
    kernel_fn, _ = select_kernel(nary, pipelined)
    run_stream_nary(kernel_fn, N, nary, check=True)


if __name__ == "__main__":
    import argparse

    # Handle imports for both pytest (module context) and direct execution
    try:
        from .gfx1250_utils import static_profile
    except ImportError:
        from gfx1250_utils import static_profile

    parser = argparse.ArgumentParser(description="Stream Copy Kernels for GFX1250")
    parser.add_argument("-n", type=int, default=65536, help="Number of elements to copy")
    parser.add_argument("--nary", type=int, choices=[1, 2, 3], default=1, help="Number of input streams")
    parser.add_argument("--pipelined", action="store_true", help="Issue next-iteration loads before the current store")
    parser.add_argument("--no-check", action="store_true",
                        help="Skip stream-copy correctness checks and D2H copy-back for bandwidth runs")
    parser.add_argument("--stream-workgroups", type=int, default=1024, help="Number of workgroups")
    parser.add_argument("--stream-block-size", type=int, default=1024, help="Elements per workgroup")
    args = parser.parse_args()

    kernel_fn, kernel_name = select_kernel(args.nary, args.pipelined)
    print(f"Running {kernel_name} with N={args.n} elements")
    print("Configuration: 4 warps/CTA, 128-bit (8 fp16) read/write per thread")
    print(f"Block size: {args.stream_block_size} elements per workgroup")
    print(f"Grid size: {args.stream_workgroups} workgroups")
    print(f"{args.nary}-input stream copy, pipelined={args.pipelined}, one output stream")
    print()

    kernel = run_stream_nary(kernel_fn, args.n, args.nary, check=not args.no_check,
                             num_workgroups=args.stream_workgroups, block_size=args.stream_block_size)

    print("\nStatic Profile:")
    static_profile(kernel)
