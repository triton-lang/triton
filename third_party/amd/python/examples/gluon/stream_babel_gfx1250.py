"""Semantic FP16 stream kernels using gfx1250 TDM loads."""

import re

import pytest
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
import triton.experimental.gluon.language as gl

# Constants for GFX1250
THREADS_PER_WARP = 32  # GFX1250 warp size
ELEMENTS_PER_THREAD = 8  # 128 bits / 16 bits per fp16 = 8 elements
WARPS_PER_CTA = 4  # 4 warps per CTA.
BLOCK_SIZE = ELEMENTS_PER_THREAD * THREADS_PER_WARP * WARPS_PER_CTA  # 1024 elements per workgroup (4 warps)


@gluon.jit
def stream_tdm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: gl.constexpr,
    NUM_WGS: gl.constexpr,
    NARY: gl.constexpr,
    PIPELINED: gl.constexpr,
):
    """N-input semantic stream with gfx1250 TDM loads staged through LDS."""
    wg_id = gl.program_id(0)
    start_offset = wg_id * BLOCK_SIZE
    grid_stride = NUM_WGS * BLOCK_SIZE
    remaining = gl.maximum(N - start_offset, 0)
    num_iters = gl.cdiv(remaining, grid_stride)

    num_warps: gl.constexpr = gl.num_warps()
    num_buffers: gl.constexpr = 2 if PIPELINED else 1
    blocked_layout: gl.constexpr = gl.BlockedLayout([8], [32], [num_warps], [0])
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_SIZE // 2, 8]], [BLOCK_SIZE], [0])
    offs = gl.arange(0, BLOCK_SIZE, layout=blocked_layout)

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(N, ), strides=(1, ),
                                                       block_shape=(BLOCK_SIZE, ), layout=shared_layout)
    a_buffer = gl.allocate_shared_memory(a_desc.dtype, shape=[num_buffers] + a_desc.block_shape, layout=a_desc.layout)
    if NARY >= 2:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr, shape=(N, ), strides=(1, ),
                                                           block_shape=(BLOCK_SIZE, ), layout=shared_layout)
        b_buffer = gl.allocate_shared_memory(b_desc.dtype, shape=[num_buffers] + b_desc.block_shape,
                                             layout=b_desc.layout)
    if NARY >= 3:
        c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=c_ptr, shape=(N, ), strides=(1, ),
                                                           block_shape=(BLOCK_SIZE, ), layout=shared_layout)
        c_buffer = gl.allocate_shared_memory(c_desc.dtype, shape=[num_buffers] + c_desc.block_shape,
                                             layout=c_desc.layout)

    if PIPELINED:
        gl.amd.gfx1250.tdm.async_load(a_desc, [start_offset], a_buffer.index(0))
        if NARY >= 2:
            gl.amd.gfx1250.tdm.async_load(b_desc, [start_offset], b_buffer.index(0))
        if NARY >= 3:
            gl.amd.gfx1250.tdm.async_load(c_desc, [start_offset], c_buffer.index(0))

        steady_iters = gl.maximum(num_iters - 1, 0)
        for i in range(0, steady_iters):
            current_slot = i % 2
            next_slot = (i + 1) % 2
            current_offset = start_offset + i * grid_stride
            next_offset = current_offset + grid_stride

            gl.amd.gfx1250.tdm.async_load(a_desc, [next_offset], a_buffer.index(next_slot))
            if NARY >= 2:
                gl.amd.gfx1250.tdm.async_load(b_desc, [next_offset], b_buffer.index(next_slot))
            if NARY >= 3:
                gl.amd.gfx1250.tdm.async_load(c_desc, [next_offset], c_buffer.index(next_slot))

            # Leave only the next iteration's TDM loads outstanding.
            gl.amd.gfx1250.tdm.async_wait(NARY)
            value = a_buffer.index(current_slot).load(layout=blocked_layout)
            if NARY >= 2:
                value = value + b_buffer.index(current_slot).load(layout=blocked_layout)
            if NARY >= 3:
                value = value + c_buffer.index(current_slot).load(layout=blocked_layout)

            current_offsets = current_offset + offs
            current_mask = current_offsets < N
            gl.amd.gfx1250.buffer_store(value, out_ptr, current_offsets, mask=current_mask)

        gl.amd.gfx1250.tdm.async_wait(0)
        last_slot = steady_iters % 2
        last_offset = start_offset + steady_iters * grid_stride
        value = a_buffer.index(last_slot).load(layout=blocked_layout)
        if NARY >= 2:
            value = value + b_buffer.index(last_slot).load(layout=blocked_layout)
        if NARY >= 3:
            value = value + c_buffer.index(last_slot).load(layout=blocked_layout)
        last_offsets = last_offset + offs
        last_mask = last_offsets < N
        gl.amd.gfx1250.buffer_store(value, out_ptr, last_offsets, mask=last_mask)
    else:
        for i in range(0, num_iters):
            tile_offset = start_offset + i * grid_stride
            gl.amd.gfx1250.tdm.async_load(a_desc, [tile_offset], a_buffer.index(0))
            if NARY >= 2:
                gl.amd.gfx1250.tdm.async_load(b_desc, [tile_offset], b_buffer.index(0))
            if NARY >= 3:
                gl.amd.gfx1250.tdm.async_load(c_desc, [tile_offset], c_buffer.index(0))
            gl.amd.gfx1250.tdm.async_wait(0)

            value = a_buffer.index(0).load(layout=blocked_layout)
            if NARY >= 2:
                value = value + b_buffer.index(0).load(layout=blocked_layout)
            if NARY >= 3:
                value = value + c_buffer.index(0).load(layout=blocked_layout)
            tile_offsets = tile_offset + offs
            mask = tile_offsets < N
            gl.amd.gfx1250.buffer_store(value, out_ptr, tile_offsets, mask=mask)


def reference_torch_nary(inputs):
    ref = inputs[0]
    for tensor in inputs[1:]:
        ref = ref + tensor
    return ref


def run_stream_nary(N: int, arity: int, pipelined: bool = False, check: bool = True, sync: bool = True,
                    num_workgroups: int = 1024, block_size: int = BLOCK_SIZE, num_warps: int = WARPS_PER_CTA):
    """
    Run an N-input memory stream kernel.

    Unary writes `a`, binary writes `a + b`, and ternary writes `a + b + c`.
    The tiny arithmetic keeps all input streams live while leaving the kernel
    dominated by global memory traffic.
    """
    if N <= 0:
        raise ValueError("N must be positive")
    num_workgroups = min(num_workgroups, triton.cdiv(N, block_size))

    dtype = torch.float16
    torch.random.manual_seed(42)
    inputs = [torch.rand(N, dtype=dtype) for _ in range(arity)]
    out = torch.empty(N, dtype=dtype)

    if check:
        ref = reference_torch_nary(inputs)

    inputs = [tensor.cuda() for tensor in inputs]
    out = out.cuda()

    b_ptr = inputs[1] if arity >= 2 else inputs[0]
    c_ptr = inputs[2] if arity >= 3 else inputs[0]
    kernel = stream_tdm_kernel[(num_workgroups, )](
        inputs[0],
        b_ptr,
        c_ptr,
        out,
        N,
        block_size,
        num_workgroups,
        arity,
        pipelined,
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


def compile_stream_tdm(nary: int, pipelined: bool):
    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp16",
        "out_ptr": "*fp16",
        "N": "i32",
        "BLOCK_SIZE": "constexpr",
        "NUM_WGS": "constexpr",
        "NARY": "constexpr",
        "PIPELINED": "constexpr",
    }
    constexprs = {
        "BLOCK_SIZE": BLOCK_SIZE,
        "NUM_WGS": 1024,
        "NARY": nary,
        "PIPELINED": pipelined,
    }
    with triton.knobs.compilation.scope():
        triton.knobs.compilation.always_compile = True
        return triton.compile(
            gluon._runtime.GluonASTSource(fn=stream_tdm_kernel, signature=signature, constexprs=constexprs),
            target=GPUTarget("hip", "gfx1250", 32),
            options={"num_warps": WARPS_PER_CTA},
        )


@pytest.mark.parametrize("pipelined", [False, True], ids=["non_pipelined", "pipelined"])
@pytest.mark.parametrize("nary", [1, 2, 3], ids=["unary", "binary", "ternary"])
def test_compile_stream_tdm(nary, pipelined):
    """Verify every stream mode uses gfx1250 TDM global-to-LDS loads."""
    kernel = compile_stream_tdm(nary, pipelined)
    actual = len(re.findall(r"\btensor_load_to_lds\b", kernel.asm["amdgcn"]))
    minimum = nary * (2 if pipelined else 1)
    assert actual >= minimum, f"expected at least {minimum} tensor_load_to_lds instructions, got {actual}"


@pytest.mark.parametrize("N", [32768, 500], ids=lambda N: f"N={N}")
@pytest.mark.parametrize("pipelined", [False, True], ids=["non_pipelined", "pipelined"])
@pytest.mark.parametrize("nary", [1, 2, 3], ids=["unary", "binary", "ternary"])
def test_stream_copy(nary, pipelined, N):
    """Test stream copy kernel correctness."""
    run_stream_nary(N, nary, pipelined=pipelined, check=True)


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
    parser.add_argument("--pipelined", action="store_true",
                        help="Double-buffer TDM loads to overlap the next tile with the current tile")
    parser.add_argument("--no-check", action="store_true",
                        help="Skip stream-copy correctness checks and D2H copy-back for bandwidth runs")
    parser.add_argument("--stream-workgroups", type=int, default=1024, help="Number of workgroups")
    parser.add_argument("--stream-block-size", type=int, default=1024, help="Elements per workgroup")
    args = parser.parse_args()

    print(f"Running stream_tdm_kernel with N={args.n} elements")
    print("Configuration: 4 warps/CTA, gfx1250 TDM global-to-LDS loads, buffer stores")
    print(f"Block size: {args.stream_block_size} elements per workgroup")
    print(f"Grid size: {min(args.stream_workgroups, triton.cdiv(args.n, args.stream_block_size))} workgroups")
    print(f"{args.nary}-input stream copy, pipelined={args.pipelined}, "
          f"{2 if args.pipelined else 1} LDS buffer(s) per input, one output stream")
    print()

    kernel = run_stream_nary(args.n, args.nary, pipelined=args.pipelined, check=not args.no_check,
                             num_workgroups=args.stream_workgroups, block_size=args.stream_block_size)

    print("\nStatic Profile:")
    static_profile(kernel)
