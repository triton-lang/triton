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
def stream_binary_double_pipelined_kernel(
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
def stream_ternary_double_pipelined_kernel(
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


@gluon.jit
def stream_load_only_kernel(
    k_ptr,
    k_scale_ptr,
    v_ptr,
    v_scale_ptr,
    DATA_ELEMS_PER_HEAD: gl.constexpr,
    SCALE_ELEMS_PER_HEAD: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    DATA_ROWS: gl.constexpr,
    DATA_COLS: gl.constexpr,
    SCALE_BLOCK_SIZE: gl.constexpr,
    NUM_K_HEADS: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
):
    """
    FA-shaped four-stream TDM load-only diagnostic.

    This streams K, K-scale, V, and V-scale into LDS with no output store and no
    compute. It is a bandwidth diagnostic, not a semantic copy kernel.
    """
    off_h = gl.program_id(0)
    off_z = gl.program_id(2)
    head_batch = NUM_K_HEADS * off_z + off_h

    num_tiles: gl.constexpr = DATA_ELEMS_PER_HEAD // BLOCK_SIZE
    data_base = DATA_ELEMS_PER_HEAD * head_batch
    scale_base = SCALE_ELEMS_PER_HEAD * head_batch

    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k_ptr + data_base,
        shape=(num_tiles * DATA_ROWS, DATA_COLS),
        strides=(DATA_COLS, 1),
        block_shape=(DATA_ROWS, DATA_COLS),
        layout=smem_layout,
    )
    v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v_ptr + data_base,
        shape=(num_tiles * DATA_ROWS, DATA_COLS),
        strides=(DATA_COLS, 1),
        block_shape=(DATA_ROWS, DATA_COLS),
        layout=smem_layout,
    )
    k_scale_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k_scale_ptr + scale_base,
        shape=(num_tiles, SCALE_BLOCK_SIZE),
        strides=(SCALE_BLOCK_SIZE, 1),
        block_shape=(1, SCALE_BLOCK_SIZE),
        layout=smem_layout,
    )
    v_scale_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v_scale_ptr + scale_base,
        shape=(num_tiles, SCALE_BLOCK_SIZE),
        strides=(SCALE_BLOCK_SIZE, 1),
        block_shape=(1, SCALE_BLOCK_SIZE),
        layout=smem_layout,
    )

    k_smem = gl.allocate_shared_memory(k_ptr.dtype.element_ty, [NUM_BUFFERS, DATA_ROWS, DATA_COLS], smem_layout)
    v_smem = gl.allocate_shared_memory(v_ptr.dtype.element_ty, [NUM_BUFFERS, DATA_ROWS, DATA_COLS], smem_layout)
    k_scale_smem = gl.allocate_shared_memory(k_scale_ptr.dtype.element_ty, [NUM_BUFFERS, 1, SCALE_BLOCK_SIZE],
                                             smem_layout)
    v_scale_smem = gl.allocate_shared_memory(v_scale_ptr.dtype.element_ty, [NUM_BUFFERS, 1, SCALE_BLOCK_SIZE],
                                             smem_layout)
    wait_count: gl.constexpr = (NUM_BUFFERS - 1) * 4

    buf = 0
    for i in range(0, num_tiles):
        data_row = i * DATA_ROWS
        gl.amd.gfx1250.tdm.async_load(k_desc, [data_row, 0], k_smem.index(buf), pred=1)
        gl.amd.gfx1250.tdm.async_load(k_scale_desc, [i, 0], k_scale_smem.index(buf), pred=1)
        gl.amd.gfx1250.tdm.async_load(v_desc, [data_row, 0], v_smem.index(buf), pred=1)
        gl.amd.gfx1250.tdm.async_load(v_scale_desc, [i, 0], v_scale_smem.index(buf), pred=1)
        gl.amd.gfx1250.tdm.async_wait(wait_count)
        buf = (buf + 1) % NUM_BUFFERS

    gl.amd.gfx1250.tdm.async_wait(0)


def reference_torch_nary(inputs):
    ref = inputs[0]
    for tensor in inputs[1:]:
        ref = ref + tensor
    return ref


def get_stream_arity(kernel_type: str):
    if "unary" in kernel_type:
        return 1
    if "binary" in kernel_type:
        return 2
    if "ternary" in kernel_type:
        return 3
    raise NotImplementedError(f"Cannot infer stream arity from kernel type: {kernel_type}")


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


def _make_stream_tensor(n: int, dtype: torch.dtype):
    if dtype is torch.uint8:
        return torch.empty(n, dtype=dtype).random_(0, 256).cuda()
    if dtype is torch.float8_e4m3fn:
        return torch.empty(n, dtype=torch.uint8).random_(0, 256).view(dtype).cuda()
    return torch.randn(n, dtype=dtype).cuda()


def run_stream_load_only(batch: int = 64, num_k_heads: int = 16, seqlen_k: int = 8192, head_sz: int = 128,
                         block_n: int = 128):
    """
    Run the standalone FA-shaped load-only stream diagnostic.

    This path intentionally has no correctness check because it performs only
    K/K-scale/V/V-scale TDM loads into LDS.
    """
    if head_sz % 32 != 0:
        raise ValueError("--stream-load-head-size must be divisible by 32 for MXFP scale streams")
    if seqlen_k % block_n != 0:
        raise ValueError("--stream-load-seqlen-k must be divisible by --stream-load-block-n")
    if block_n % 32 != 0:
        raise ValueError("--stream-load-block-n must be divisible by 32 for scale tiles")

    num_buffers = 3
    data_rows = 64
    data_elems_per_head = seqlen_k * head_sz
    scale_elems_per_head = seqlen_k * (head_sz // 32)
    data_block_size = block_n * head_sz
    scale_block_size = block_n * (head_sz // 32)
    if data_block_size % data_rows != 0:
        raise ValueError(f"Derived data tile ({data_block_size} elements) must be divisible by {data_rows}")

    torch.random.manual_seed(42)
    data_n = batch * num_k_heads * data_elems_per_head
    scale_n = batch * num_k_heads * scale_elems_per_head
    k = _make_stream_tensor(data_n, torch.float8_e4m3fn)
    v = _make_stream_tensor(data_n, torch.float8_e4m3fn)
    k_scale = _make_stream_tensor(scale_n, torch.uint8)
    v_scale = _make_stream_tensor(scale_n, torch.uint8)

    kernel = stream_load_only_kernel[(num_k_heads, 1, batch)](
        k,
        k_scale,
        v,
        v_scale,
        data_elems_per_head,
        scale_elems_per_head,
        data_block_size,
        data_rows,
        data_block_size // data_rows,
        scale_block_size,
        num_k_heads,
        num_buffers,
        num_warps=1,
        num_ctas=1,
        waves_per_eu=1,
    )
    torch.cuda.synchronize()
    print(f"PASSED: stream-load-only batch={batch}, heads={num_k_heads}, "
          f"seqlen_k={seqlen_k}, head_sz={head_sz}, block_n={block_n}")
    return kernel


# Test configurations
def generate_test_configs():
    return [
        pytest.param(stream_unary_kernel, 1, 32768, id="unary_N=32768"),
        pytest.param(stream_binary_kernel, 2, 32768, id="binary_N=32768"),
        pytest.param(stream_ternary_kernel, 3, 32768, id="ternary_N=32768"),
        pytest.param(stream_unary_pipelined_kernel, 1, 32768, id="unary_double_N=32768"),
        pytest.param(stream_binary_double_pipelined_kernel, 2, 32768, id="binary_double_N=32768"),
        pytest.param(stream_ternary_double_pipelined_kernel, 3, 32768, id="ternary_double_N=32768"),
        pytest.param(stream_unary_kernel, 1, 500, id="unary_N=500"),
        pytest.param(stream_binary_kernel, 2, 500, id="binary_N=500"),
        pytest.param(stream_ternary_kernel, 3, 500, id="ternary_N=500"),
    ]


@pytest.mark.parametrize("kernel_fn, arity, N", generate_test_configs())
def test_stream_copy(kernel_fn, arity, N):
    """Test stream copy kernel correctness."""
    run_stream_nary(kernel_fn, N, arity, check=True)


def select_kernel(arg_kernel_type):
    if arg_kernel_type == "stream-load-only":
        kernel_fn = stream_load_only_kernel
        kernel_name = "stream_load_only_kernel"
    elif arg_kernel_type in ("copy-ternary", "stream-ternary"):
        kernel_fn = stream_ternary_kernel
        kernel_name = "stream_ternary_kernel"
    elif arg_kernel_type in ("copy-binary", "stream-binary"):
        kernel_fn = stream_binary_kernel
        kernel_name = "stream_binary_kernel"
    elif arg_kernel_type in ("copy-unary", "stream-unary"):
        kernel_fn = stream_unary_kernel
        kernel_name = "stream_unary_kernel"
    elif arg_kernel_type in ("copy-unary-pipelined", "stream-unary-pipelined"):
        kernel_fn = stream_unary_pipelined_kernel
        kernel_name = "stream_unary_pipelined_kernel"
    elif arg_kernel_type == "copy-unary-double-pipelined":
        kernel_fn = stream_unary_pipelined_kernel
        kernel_name = "stream_unary_pipelined_kernel"
    elif arg_kernel_type == "copy-binary-double-pipelined":
        kernel_fn = stream_binary_double_pipelined_kernel
        kernel_name = "stream_binary_double_pipelined_kernel"
    elif arg_kernel_type == "copy-ternary-double-pipelined":
        kernel_fn = stream_ternary_double_pipelined_kernel
        kernel_name = "stream_ternary_double_pipelined_kernel"
    else:
        raise ValueError(f"Unknown kernel type: {arg_kernel_type}")
    return kernel_fn, kernel_name


if __name__ == "__main__":
    import argparse

    # Handle imports for both pytest (module context) and direct execution
    try:
        from .gfx1250_utils import static_profile
    except ImportError:
        from gfx1250_utils import static_profile

    parser = argparse.ArgumentParser(description="Stream Copy Kernel for GFX1250")
    parser.add_argument("-n", type=int, default=65536, help="Number of elements to copy")
    parser.add_argument(
        "--kernel-type", type=str, choices=[
            "copy-unary", "copy-binary", "copy-ternary", "copy-unary-pipelined", "copy-unary-double-pipelined",
            "copy-binary-double-pipelined", "copy-ternary-double-pipelined", "stream-unary", "stream-binary",
            "stream-ternary", "stream-unary-pipelined", "stream-load-only"
        ], default="copy-unary", help="Kernel type to use")
    parser.add_argument("--no-check", action="store_true",
                        help="Skip stream-copy correctness checks and D2H copy-back for bandwidth runs")
    parser.add_argument("--stream-workgroups", type=int, default=1024,
                        help="Workgroups for copy-unary/copy-binary/copy-ternary stream kernels")
    parser.add_argument("--stream-block-size", type=int, default=1024,
                        help="Elements per workgroup for copy-unary/copy-binary/copy-ternary stream kernels")
    parser.add_argument("--stream-load-batch", type=int, default=64,
                        help="Batch size for --kernel-type stream-load-only")
    parser.add_argument("--stream-load-num-k-heads", type=int, default=16,
                        help="KV heads for --kernel-type stream-load-only")
    parser.add_argument("--stream-load-seqlen-k", type=int, default=8192,
                        help="K/V sequence length for --kernel-type stream-load-only")
    parser.add_argument("--stream-load-head-size", type=int, default=128,
                        help="K/V head size for --kernel-type stream-load-only")
    parser.add_argument("--stream-load-block-n", type=int, default=128,
                        help="K/V tile size for --kernel-type stream-load-only")
    args = parser.parse_args()

    kernel_fn, kernel_name = select_kernel(args.kernel_type)
    is_stream_nary = args.kernel_type in (
        "copy-unary",
        "copy-binary",
        "copy-ternary",
        "copy-unary-pipelined",
        "copy-unary-double-pipelined",
        "copy-binary-double-pipelined",
        "copy-ternary-double-pipelined",
        "stream-unary",
        "stream-binary",
        "stream-ternary",
        "stream-unary-pipelined",
    )
    arity = get_stream_arity(args.kernel_type) if is_stream_nary else None
    if is_stream_nary:
        num_workgroups = args.stream_workgroups
    elif args.kernel_type == "stream-load-only":
        num_workgroups = args.stream_load_batch * args.stream_load_num_k_heads
    else:
        raise ValueError(f"Unknown kernel type: {args.kernel_type}")

    if args.kernel_type == "stream-load-only":
        print(f"Running {kernel_name}")
        print("Configuration: 1 warp/CTA, 3 TDM buffers, waves_per_eu=1")
    else:
        print(f"Running {kernel_name} with N={args.n} elements")
        print("Configuration: 4 warps/CTA, 128-bit (8 fp16) read/write per thread")
        print(f"Block size: {args.stream_block_size} elements per workgroup")
    if is_stream_nary:
        print(f"Stream block size: {args.stream_block_size} elements per workgroup")
    print(f"Grid size: {num_workgroups} workgroups")
    if args.kernel_type == "stream-load-only":
        print(f"Stream-load-only: K/K-scale/V/V-scale TDM loads, batch={args.stream_load_batch}, "
              f"heads={args.stream_load_num_k_heads}, seqlen_k={args.stream_load_seqlen_k}, "
              f"head_sz={args.stream_load_head_size}, block_n={args.stream_load_block_n}")
    if arity is not None:
        print(f"{arity}-input stream copy: {args.stream_workgroups} workgroups, one output stream")
    print()

    if args.kernel_type == "stream-load-only":
        kernel = run_stream_load_only(batch=args.stream_load_batch, num_k_heads=args.stream_load_num_k_heads,
                                      seqlen_k=args.stream_load_seqlen_k, head_sz=args.stream_load_head_size,
                                      block_n=args.stream_load_block_n)
    elif arity is not None:
        kernel = run_stream_nary(kernel_fn, args.n, arity, check=not args.no_check,
                                 num_workgroups=args.stream_workgroups, block_size=args.stream_block_size)
    else:
        raise ValueError(f"Unknown kernel type: {args.kernel_type}")

    print("\nStatic Profile:")
    static_profile(kernel)
