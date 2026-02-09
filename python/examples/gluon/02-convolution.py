import torch

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tensor_memory_descriptor,
    get_tmem_reg_layout,
    tcgen05_mma,
    tcgen05_commit,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor, TensorDescriptorIm2Col
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared

from triton.language.core import _aggregate as aggregate


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


# Convolution parameter naming convention:
#   N   = batch size
#   H,W = input spatial dims
#   Ci  = input channels   (part of GEMM-K reduction: K_GEMM = R * S * Ci)
#   Co  = output channels  (maps to GEMM-N dimension: N_GEMM = Co)
#   R,S = filter height, width
#
# GEMM mapping:
#   M_GEMM = N * out_h * out_w   (output spatial positions)
#   N_GEMM = Co                  (output channels)
#   K_GEMM = R * S * Ci          (reduction over filter x input channels)


# Helper class for passing arguments between partitions
@aggregate
class PartitionArgs:
    in_desc: tma.tensor_descriptor_im2col
    weight_desc: tma.tensor_descriptor
    output_ptr: gl.tensor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    acc_bufs: tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    # Convolution parameters
    N: gl.constexpr
    H: gl.constexpr
    W: gl.constexpr
    Ci: gl.constexpr       # input channels
    Co: gl.constexpr       # output channels (GEMM N dimension)
    R: gl.constexpr
    S: gl.constexpr
    out_h: gl.constexpr
    out_w: gl.constexpr
    stride_output_n: gl.constexpr
    stride_output_h: gl.constexpr
    stride_output_w: gl.constexpr
    stride_h: gl.constexpr
    stride_w: gl.constexpr
    pad_h: gl.constexpr
    pad_w: gl.constexpr
    # Tile parameters
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    GROUP_SIZE_M: gl.constexpr
    num_warps: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, in_desc, weight_desc, output_ptr, a_bufs, b_bufs,
                 load_empty_bars, load_ready_bars, acc_bufs, acc_empty_bars, acc_ready_bars,
                 N, H, W, Ci, Co, R, S, out_h, out_w,
                 stride_output_n, stride_output_h, stride_output_w,
                 stride_h, stride_w, pad_h, pad_w,
                 BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps):
        self.in_desc = in_desc
        self.weight_desc = weight_desc
        self.output_ptr = output_ptr
        self.a_bufs = a_bufs
        self.b_bufs = b_bufs
        self.load_empty_bars = load_empty_bars
        self.load_ready_bars = load_ready_bars
        self.acc_bufs = acc_bufs
        self.acc_empty_bars = acc_empty_bars
        self.acc_ready_bars = acc_ready_bars
        self.N = gl.constexpr(N)
        self.H = gl.constexpr(H)
        self.W = gl.constexpr(W)
        self.Ci = gl.constexpr(Ci)
        self.Co = gl.constexpr(Co)
        self.R = gl.constexpr(R)
        self.S = gl.constexpr(S)
        self.out_h = gl.constexpr(out_h)
        self.out_w = gl.constexpr(out_w)
        self.stride_output_n = gl.constexpr(stride_output_n)
        self.stride_output_h = gl.constexpr(stride_output_h)
        self.stride_output_w = gl.constexpr(stride_output_w)
        self.stride_h = gl.constexpr(stride_h)
        self.stride_w = gl.constexpr(stride_w)
        self.pad_h = gl.constexpr(pad_h)
        self.pad_w = gl.constexpr(pad_w)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.GROUP_SIZE_M = gl.constexpr(GROUP_SIZE_M)
        self.num_warps = gl.constexpr(num_warps)


@gluon.jit
def load_partition(p):
    """Load partition: Issues async copies for input (A) and weight (B) tiles"""
    BLOCK_M: gl.constexpr = p.BLOCK_M
    BLOCK_N: gl.constexpr = p.BLOCK_N
    BLOCK_K: gl.constexpr = p.BLOCK_K
    N_GEMM = p.Co  # output channels

    # Compute program ID and tile coordinates
    pid = gl.program_id(axis=0)
    M_GEMM = p.N * p.out_h * p.out_w

    num_pid_m = gl.cdiv(M_GEMM, BLOCK_M)
    num_pid_n = gl.cdiv(N_GEMM, BLOCK_N)
    num_pid_in_group = p.GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * p.GROUP_SIZE_M
    group_size_m = gl.minimum(num_pid_m - first_pid_m, p.GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute base offsets
    offs_m = pid_m * BLOCK_M

    # Offset in the batch dimension
    batch_id = offs_m // (p.out_h * p.out_w)
    m_residual = offs_m % (p.out_h * p.out_w)

    # Offset in the image dimension
    out_y = m_residual // p.out_w
    out_x = m_residual % p.out_w

    num_buffers: gl.constexpr = p.a_bufs.type.shape[0]

    # Number of GEMM-K iterations: K_GEMM = R * S * Ci
    num_k_iter = p.R * p.S * p.Ci // BLOCK_K
    ci_num_blocks = p.Ci // BLOCK_K
    num_rs = p.R * p.S

    for k_iter in range(num_k_iter):
        index = k_iter % num_buffers
        phase = k_iter // num_buffers & 1

        # Decompose k_iter into (r, s, ci_block) indices
        iter_ci = k_iter // num_rs
        remain_rs = k_iter % num_rs
        iter_s = remain_rs % p.S
        iter_r = remain_rs // p.S

        # Wait for buffers to be empty
        empty_bar = p.load_empty_bars.index(index)
        mbarrier.wait(empty_bar, phase ^ 1)

        # Set up ready barrier and issue TMA copies
        ready_bar = p.load_ready_bars.index(index)
        mbarrier.expect(ready_bar, p.in_desc.block_type.nbytes + p.weight_desc.block_type.nbytes)

        # Input tile via TMA im2col
        # Offsets must be i16 for the TMA im2col op
        offset_r = iter_r.to(tl.int16)
        offset_s = iter_s.to(tl.int16)
        tma.async_copy_global_to_shared_im2col(
            p.in_desc,
            [batch_id, out_y - p.pad_h, out_x - p.pad_w, iter_ci * BLOCK_K],
            [offset_r, offset_s],
            ready_bar,
            p.a_bufs.index(index),
        )

        # Weight tile via standard TMA (weight is (Co, R*S*Ci) = (N_gemm, K_gemm))
        k_offset = ((iter_s + iter_r * p.S) * ci_num_blocks + iter_ci) * BLOCK_K
        tma.async_copy_global_to_shared(
            p.weight_desc,
            [pid_n * BLOCK_N, k_offset],
            ready_bar,
            p.b_bufs.index(index),
        )


@gluon.jit
def mma_partition(p):
    """MMA partition: Performs tcgen05 MMA operations"""
    BLOCK_K: gl.constexpr = p.BLOCK_K
    K_GEMM = p.R * p.S * p.Ci  # reduction dimension

    num_buffers: gl.constexpr = p.a_bufs.type.shape[0]

    # Wait for accumulator to be empty
    mbarrier.wait(p.acc_empty_bars.index(0), phase=1)
    acc_buf = p.acc_bufs.index(0)
    use_acc = False

    # Loop over GEMM-K dimension
    for k_iter in range(gl.cdiv(K_GEMM, BLOCK_K)):
        index = k_iter % num_buffers
        load_phase = k_iter // num_buffers & 1

        # Wait for operands to be ready
        mbarrier.wait(p.load_ready_bars.index(index), load_phase)
        
        # Perform MMA: A is [M, K], B in smem is [N, K] so permute to [K, N]
        b_smem = p.b_bufs.index(index)
        tcgen05_mma(p.a_bufs.index(index), b_smem.permute((1, 0)), acc_buf, use_acc=use_acc)
        
        # Signal that buffers are consumed
        tcgen05_commit(p.load_empty_bars.index(index))
        use_acc = True

    # Signal that accumulator is ready
    tcgen05_commit(p.acc_ready_bars.index(0))


@gluon.jit
def epilogue_partition(p):
    """Epilogue partition: Loads from TMEM and stores to global memory"""
    BLOCK_M: gl.constexpr = p.BLOCK_M
    BLOCK_N: gl.constexpr = p.BLOCK_N
    M_GEMM = p.N * p.out_h * p.out_w
    N_GEMM = p.Co  # output channels

    # Compute program ID and tile coordinates
    pid = gl.program_id(axis=0)

    num_pid_m = gl.cdiv(M_GEMM, BLOCK_M)
    num_pid_n = gl.cdiv(N_GEMM, BLOCK_N)
    num_pid_in_group = p.GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * p.GROUP_SIZE_M
    group_size_m = gl.minimum(num_pid_m - first_pid_m, p.GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Layout for output
    c_layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [p.num_warps, 1], [1, 0])
    c_cols_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=c_layout)
    c_rows_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=c_layout)

    # TMEM layout
    tmem_layout: gl.constexpr = TensorMemoryLayout(
        block=(128, BLOCK_N),
        col_stride=1,
    )
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, p.num_warps
    )

    # Wait for accumulator to be ready
    mbarrier.wait(p.acc_ready_bars.index(0), phase=0)

    # Load from TMEM
    acc = p.acc_bufs.index(0).load(acc_reg_layout)
    acc_fp16 = acc.to(gl.float16)
    acc_fp16 = gl.convert_layout(acc_fp16, c_layout)

    # Signal that accumulator is consumed
    mbarrier.arrive(p.acc_empty_bars.index(0), count=1)

    # Compute output addresses
    offs_m_base = pid_m * BLOCK_M
    offs_n_base = pid_n * BLOCK_N
    c_offs_m = offs_m_base + gl.arange(0, BLOCK_M, layout=c_rows_layout)
    c_offs_n = offs_n_base + gl.arange(0, BLOCK_N, layout=c_cols_layout)

    c_batch = c_offs_m // (p.out_h * p.out_w)
    c_rem = c_offs_m % (p.out_h * p.out_w)
    c_out_y = c_rem // p.out_w
    c_out_x = c_rem % p.out_w

    c_offsets = (c_batch[:, None] * p.stride_output_n +
                c_out_y[:, None] * p.stride_output_h +
                c_out_x[:, None] * p.stride_output_w +
                c_offs_n[None, :])

    c_mask = (c_offs_m[:, None] < M_GEMM) & (c_offs_n[None, :] < N_GEMM)

    # Fence before store to order with TMEM load
    fence_async_shared()
    gl.store(p.output_ptr + c_offsets, acc_fp16, mask=c_mask)

    # Invalidate all barriers at the end
    num_buffers: gl.constexpr = p.a_bufs.type.shape[0]
    for i in gl.static_range(num_buffers):
        mbarrier.invalidate(p.load_empty_bars.index(i))
        mbarrier.invalidate(p.load_ready_bars.index(i))
    mbarrier.invalidate(p.acc_empty_bars.index(0))
    mbarrier.invalidate(p.acc_ready_bars.index(0))


@gluon.jit
def implicit_gemm_conv2d_tma_im2col_warp_specialized_kernel(
    in_desc, weight_desc, output,
    N: gl.constexpr, H: gl.constexpr, W: gl.constexpr, Ci: gl.constexpr,
    Co: gl.constexpr, R: gl.constexpr, S: gl.constexpr,
    out_h: gl.constexpr, out_w: gl.constexpr,
    output_stride_n: gl.constexpr, output_stride_h: gl.constexpr, output_stride_w: gl.constexpr,
    stride_h: gl.constexpr, stride_w: gl.constexpr,
    pad_h: gl.constexpr, pad_w: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    num_buffers: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Warp-specialized implicit GEMM convolution kernel with TMA im2col

    GEMM dimensions:
        M = N * out_h * out_w   (output spatial positions)
        N = Co                  (output channels)
        K = R * S * Ci          (reduction over filter x input channels)
    """

    # Allocate shared memory for multi-buffered loads
    # A (input): [BLOCK_M, BLOCK_K] -- im2col tiles
    # B (weight): [BLOCK_N, BLOCK_K] -- matches weight layout (Co, R*S*Ci)
    a_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, BLOCK_K], gl.float16)

    a_bufs = gl.allocate_shared_memory(gl.float16, [num_buffers, BLOCK_M, BLOCK_K], a_smem_layout)
    b_bufs = gl.allocate_shared_memory(gl.float16, [num_buffers, BLOCK_N, BLOCK_K], b_smem_layout)

    # Barriers for load coordination
    load_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    load_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())

    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)

    tmem_layout: gl.constexpr = TensorMemoryLayout(
        block=(128, BLOCK_N),
        col_stride=1,
    )

    acc_bufs = allocate_tensor_memory(gl.float32, [1, BLOCK_M, BLOCK_N], tmem_layout)

    # Barriers for accumulator coordination
    acc_empty_bars = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
    acc_ready_bars = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())

    mbarrier.init(acc_empty_bars.index(0), count=1)
    mbarrier.init(acc_ready_bars.index(0), count=1)

    # Create partition arguments
    p = PartitionArgs(
        in_desc, weight_desc, output, a_bufs, b_bufs,
        load_empty_bars, load_ready_bars, acc_bufs, acc_empty_bars, acc_ready_bars,
        N, H, W, Ci, Co, R, S, out_h, out_w,
        output_stride_n, output_stride_h, output_stride_w,
        stride_h, stride_w, pad_h, pad_w,
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps
    )

    gl.warp_specialize([
        (mma_partition, (p,)),
        (load_partition, (p,)),
        (epilogue_partition, (p,)),
    ], [4, 4], [384, 128])


def implicit_gemm_conv2d_tma_im2col_warp_specialized(input_tensor, weight_tensor, stride=1, padding=0,
                                                     num_buffers=2, num_warps=4):
    """
    Warp-specialized implicit GEMM convolution using TMA im2col.

    Args:
        input_tensor: (N, H, W, Ci) - NHWC layout
        weight_tensor: (Co, R, S, Ci) - output channels first
        num_buffers: Number of pipeline buffers (default: 2)
        num_warps: Number of warps for MMA partition (default: 4)
    """
    if not is_blackwell():
        raise RuntimeError("This kernel requires a Blackwell NVIDIA GPU (SM 10.x)")

    N, H, W, Ci = input_tensor.shape
    Co, R, S, Ci_w = weight_tensor.shape
    assert Ci == Ci_w, "Input and weight channel dimensions must match"

    out_h = (H + 2 * padding - R) // stride + 1
    out_w = (W + 2 * padding - S) // stride + 1

    output = torch.empty((N, out_h, out_w, Co), device=input_tensor.device, dtype=torch.float16)

    # GEMM dimensions: M = N*out_h*out_w, N_gemm = Co, K_gemm = R*S*Ci
    M_GEMM = N * out_h * out_w
    N_GEMM = Co

    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 64
    GROUP_SIZE_M = 4

    grid = (triton.cdiv(M_GEMM, BLOCK_M) * triton.cdiv(N_GEMM, BLOCK_N),)

    # Create TMA im2col descriptor for input (A matrix)
    # Input: [N, H, W, Ci] in NHWC format
    # block_shape = [pixelsPerColumn, channelsPerPixel]
    input_block_shape = [BLOCK_M, BLOCK_K]
    input_layout = gl.NVMMASharedLayout.get_default_for(input_block_shape, gl.float16)

    in_desc = TensorDescriptorIm2Col(
        base=input_tensor,
        shape=list(input_tensor.shape),
        strides=list(input_tensor.stride()),
        block_shape=input_block_shape,
        layout=input_layout,
        padding="zero",
        element_strides=[1, 1, 1, 1],
        pixel_box_lower_corner=[-padding, -padding],
        pixel_box_upper_corner=[-padding, -padding],
    )

    # Create TMA tiled descriptor for weight (B matrix)
    # Weight: (Co, R, S, Ci) -> reshape to (Co, R*S*Ci) = (N_gemm, K_gemm)
    # block_shape = [BLOCK_N, BLOCK_K] tiles N first, K second
    weight_reshaped = weight_tensor.reshape(Co, R * S * Ci)
    weight_block_shape = [BLOCK_N, BLOCK_K]
    weight_layout = gl.NVMMASharedLayout.get_default_for(weight_block_shape, gl.float16)

    weight_desc = TensorDescriptor.from_tensor(weight_reshaped, weight_block_shape, weight_layout)

    implicit_gemm_conv2d_tma_im2col_warp_specialized_kernel[grid](
        in_desc, weight_desc, output,
        N, H, W, Ci,
        Co, R, S,
        out_h, out_w,
        output.stride(0), output.stride(1), output.stride(2),
        stride, stride,
        padding, padding,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_buffers=num_buffers,
        num_warps=num_warps,
    )

    return output


if __name__ == "__main__":
    if not is_blackwell():
        print("This tutorial requires a Blackwell NVIDIA GPU (SM 10.x)")
        print("Exiting...")
        exit(0)

    torch.manual_seed(0)

    # Test parameters
    N, H, W, Ci = 128, 64, 64, 384   # batch, height, width, input channels
    Co, R, S = 384, 3, 3              # output channels, filter height, filter width
    stride = 1
    padding = 1

    print(f"Parameters: N={N}, H={H}, W={W}, Ci={Ci}, Co={Co}, R={R}, S={S}, "
          f"stride={stride}, padding={padding}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")

    # Prepare tensors (PyTorch conv2d uses NCHW for input and (Co, Ci, R, S) for weight)
    x_nchw = torch.randn((N, Ci, H, W), device='cuda', dtype=torch.float16)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()

    w_nchw = torch.randn((Co, Ci, R, S), device='cuda', dtype=torch.float16)
    w_nhwc = w_nchw.permute(0, 2, 3, 1).contiguous()  # (Co, R, S, Ci)

    triton_out = implicit_gemm_conv2d_tma_im2col_warp_specialized(x_nhwc, w_nhwc, stride=stride, padding=padding)

    torch.cuda.synchronize()
    print("Finished first run")

    triton_out = implicit_gemm_conv2d_tma_im2col_warp_specialized(x_nhwc, w_nhwc, stride=stride, padding=padding)

    torch_out = torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride, padding=padding)
    torch_out = torch_out.permute(0, 2, 3, 1)

    print(f"Input Shape (NHWC): {x_nhwc.shape}")
    print(f"Output Shape (NHWC): {triton_out.shape}")

    if torch.allclose(triton_out, torch_out, atol=1e-2, rtol=1e-2):
        print("Match!")
    else:
        print("Mismatch")
        diff = (triton_out - torch_out).abs().max()
        print(f"Max diff: {diff}")

    for _ in range(5):
        implicit_gemm_conv2d_tma_im2col_warp_specialized(x_nhwc, w_nhwc, stride=stride, padding=padding)
        torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride, padding=padding)

    # Benchmark
    print("\n--- Warp-Specialized with Different Configurations ---")

    ms_ws_2buf_4w = triton.testing.do_bench(
        lambda: implicit_gemm_conv2d_tma_im2col_warp_specialized(
            x_nhwc, w_nhwc, stride=stride, padding=padding, num_buffers=2, num_warps=4),
        warmup=100, rep=500
    )

    ms_ws_3buf_4w = triton.testing.do_bench(
        lambda: implicit_gemm_conv2d_tma_im2col_warp_specialized(
            x_nhwc, w_nhwc, stride=stride, padding=padding, num_buffers=3, num_warps=4),
        warmup=100, rep=500
    )

    ms_torch = triton.testing.do_bench(
        lambda: torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride, padding=padding),
        warmup=100, rep=500
    )

    # Calculate TFLOPS
    out_h = (H + 2 * padding - R) // stride + 1
    out_w = (W + 2 * padding - S) // stride + 1
    flops = 2 * N * out_h * out_w * Co * Ci * R * S

    tflops_ws_2buf = flops * 1e-12 / (ms_ws_2buf_4w * 1e-3)
    tflops_ws_3buf = flops * 1e-12 / (ms_ws_3buf_4w * 1e-3)
    tflops_torch = flops * 1e-12 / (ms_torch * 1e-3)

    print(f"\nWarp-Spec (2 buf, 4 warps): {ms_ws_2buf_4w:.3f} ms ({tflops_ws_2buf:.2f} TFLOPS)")
    print(f"Warp-Spec (3 buf, 4 warps): {ms_ws_3buf_4w:.3f} ms ({tflops_ws_3buf:.2f} TFLOPS)")
    print(f"PyTorch:                     {ms_torch:.3f} ms ({tflops_torch:.2f} TFLOPS)")

    best_ws = min(ms_ws_2buf_4w, ms_ws_3buf_4w)
    if best_ws < ms_torch:
        print(f"\nBest Warp-Specialized is {ms_torch / best_ws:.2f}x FASTER than PyTorch")
    else:
        print(f"\nBest Warp-Specialized is {best_ws / ms_torch:.2f}x SLOWER than PyTorch")
