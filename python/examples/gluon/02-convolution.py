import pytest
import torch

import triton
import triton.language as tl

from triton.language.core import _aggregate as aggregate

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor, TensorDescriptorIm2Col
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tensor_memory_descriptor,
    get_tmem_reg_layout,
    tcgen05_mma,
    tcgen05_commit,
)


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def make_tensor_desc(x, shape, strides, block_shape):
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float16)
    return TensorDescriptor(x, shape=shape, strides=strides, block_shape=block_shape, layout=layout)


# ===-----------------------------------------------------------------------===#
# Convolution Configuration
# ===-----------------------------------------------------------------------===#

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


@aggregate
class ConvConfig:
    N: gl.constexpr
    H: gl.constexpr
    W: gl.constexpr
    Ci: gl.constexpr
    Co: gl.constexpr
    R: gl.constexpr
    S: gl.constexpr
    out_h: gl.constexpr
    out_w: gl.constexpr
    stride_h: gl.constexpr
    stride_w: gl.constexpr
    pad_h: gl.constexpr
    pad_w: gl.constexpr
    output_stride_n: gl.constexpr
    output_stride_h: gl.constexpr
    output_stride_w: gl.constexpr

    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    GROUP_SIZE_M: gl.constexpr
    num_buffers: gl.constexpr
    num_warps: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, N, H, W, Ci, Co, R, S, out_h, out_w,
                 stride_h, stride_w, pad_h, pad_w,
                 output_stride_n, output_stride_h, output_stride_w,
                 BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_buffers, num_warps):
        self.N = gl.constexpr(N)
        self.H = gl.constexpr(H)
        self.W = gl.constexpr(W)
        self.Ci = gl.constexpr(Ci)
        self.Co = gl.constexpr(Co)
        self.R = gl.constexpr(R)
        self.S = gl.constexpr(S)
        self.out_h = gl.constexpr(out_h)
        self.out_w = gl.constexpr(out_w)
        self.stride_h = gl.constexpr(stride_h)
        self.stride_w = gl.constexpr(stride_w)
        self.pad_h = gl.constexpr(pad_h)
        self.pad_w = gl.constexpr(pad_w)
        self.output_stride_n = gl.constexpr(output_stride_n)
        self.output_stride_h = gl.constexpr(output_stride_h)
        self.output_stride_w = gl.constexpr(output_stride_w)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.GROUP_SIZE_M = gl.constexpr(GROUP_SIZE_M)
        self.num_buffers = gl.constexpr(num_buffers)
        self.num_warps = gl.constexpr(num_warps)

    @gluon.jit
    def get_program(self, pid):
        """Compute tile coordinates from program ID with grouped ordering."""
        M_GEMM = self.N * self.out_h * self.out_w
        N_GEMM = self.Co

        num_pid_m = gl.cdiv(M_GEMM, self.BLOCK_M)
        num_pid_n = gl.cdiv(N_GEMM, self.BLOCK_N)
        num_pid_in_group = self.GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * self.GROUP_SIZE_M
        group_size_m = gl.minimum(num_pid_m - first_pid_m, self.GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        return ConvProgram(self, pid_m, pid_n)


@aggregate
class ConvProgram:
    config: ConvConfig
    pid_m: gl.tensor
    pid_n: gl.tensor

    @gluon.constexpr_function
    def __init__(self, config, pid_m, pid_n):
        self.config = config
        self.pid_m = pid_m
        self.pid_n = pid_n

    @gluon.jit
    def get_m_offsets(self):
        """Decompose M-tile offset into (batch, out_y, out_x)."""
        offs_m = self.pid_m * self.config.BLOCK_M
        batch_id = offs_m // (self.config.out_h * self.config.out_w)
        m_residual = offs_m % (self.config.out_h * self.config.out_w)
        out_y = m_residual // self.config.out_w
        out_x = m_residual % self.config.out_w
        return batch_id, out_y, out_x


# ===-----------------------------------------------------------------------===#
# Partition Arguments
# ===-----------------------------------------------------------------------===#


@aggregate
class PartitionArgs:
    config: ConvConfig
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

    @gluon.constexpr_function
    def __init__(self, config, in_desc, weight_desc, output_ptr,
                 a_bufs, b_bufs, load_empty_bars, load_ready_bars,
                 acc_bufs, acc_empty_bars, acc_ready_bars):
        self.config = config
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


# ===-----------------------------------------------------------------------===#
# Warp-Specialized Partitions
# ===-----------------------------------------------------------------------===#


@gluon.jit
def load_partition(p):
    """Load partition: issues TMA copies for input (im2col) and weight tiles."""
    config = p.config
    BLOCK_M: gl.constexpr = config.BLOCK_M
    BLOCK_N: gl.constexpr = config.BLOCK_N
    BLOCK_K: gl.constexpr = config.BLOCK_K

    pid = gl.program_id(axis=0)
    prog = config.get_program(pid)
    batch_id, out_y, out_x = prog.get_m_offsets()

    num_buffers: gl.constexpr = p.a_bufs.type.shape[0]
    ci_num_blocks = config.Ci // BLOCK_K
    num_rs = config.R * config.S
    num_k_iter = num_rs * ci_num_blocks

    for k_iter in range(num_k_iter):
        index = k_iter % num_buffers
        phase = k_iter // num_buffers & 1

        # Decompose k_iter into (r, s, ci_block) indices
        iter_ci = k_iter // num_rs
        remain_rs = k_iter % num_rs
        iter_s = remain_rs % config.S
        iter_r = remain_rs // config.S

        # Wait for buffers to be consumed
        mbarrier.wait(p.load_empty_bars.index(index), phase ^ 1)

        # Set up ready barrier and issue both TMA copies
        ready_bar = p.load_ready_bars.index(index)
        mbarrier.expect(ready_bar, p.in_desc.block_type.nbytes + p.weight_desc.block_type.nbytes)

        # Input tile via TMA im2col (offsets must be i16)
        tma.async_copy_global_to_shared_im2col(
            p.in_desc,
            [batch_id, out_y * config.stride_h - config.pad_h, out_x * config.stride_w - config.pad_w, iter_ci * BLOCK_K],
            [iter_r.to(tl.int16), iter_s.to(tl.int16)],
            ready_bar,
            p.a_bufs.index(index),
        )

        # Weight tile via standard TMA: weight is (Co, R*S*Ci) = (N_GEMM, K_GEMM)
        k_offset = ((iter_s + iter_r * config.S) * ci_num_blocks + iter_ci) * BLOCK_K
        tma.async_copy_global_to_shared(
            p.weight_desc,
            [prog.pid_n * BLOCK_N, k_offset],
            ready_bar,
            p.b_bufs.index(index),
        )


@gluon.jit
def mma_partition(p):
    """MMA partition: performs tcgen05 matrix-multiply-accumulate operations."""
    config = p.config
    BLOCK_K: gl.constexpr = config.BLOCK_K
    K_GEMM = config.R * config.S * config.Ci

    num_buffers: gl.constexpr = p.a_bufs.type.shape[0]

    # Wait for accumulator buffer
    mbarrier.wait(p.acc_empty_bars.index(0), phase=1)
    acc_buf = p.acc_bufs.index(0)
    use_acc = False

    for k_iter in range(gl.cdiv(K_GEMM, BLOCK_K)):
        index = k_iter % num_buffers
        load_phase = k_iter // num_buffers & 1

        # Wait for operands to be ready
        mbarrier.wait(p.load_ready_bars.index(index), load_phase)

        # MMA: A is [M, K], B in smem is [N, K] so permute to [K, N]
        tcgen05_mma(p.a_bufs.index(index), p.b_bufs.index(index).permute((1, 0)),
                    acc_buf, use_acc=use_acc)

        # Signal that load buffers are consumed
        tcgen05_commit(p.load_empty_bars.index(index))
        use_acc = True

    # Signal that accumulator is ready
    tcgen05_commit(p.acc_ready_bars.index(0))


@gluon.jit
def epilogue_partition(p):
    """Epilogue partition: loads accumulator from TMEM and stores to global memory."""
    config = p.config
    BLOCK_M: gl.constexpr = config.BLOCK_M
    BLOCK_N: gl.constexpr = config.BLOCK_N
    M_GEMM = config.N * config.out_h * config.out_w
    N_GEMM = config.Co

    pid = gl.program_id(axis=0)
    prog = config.get_program(pid)

    # Register layouts
    c_layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [config.num_warps, 1], [1, 0])
    c_cols_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=c_layout)
    c_rows_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=c_layout)

    tmem_layout: gl.constexpr = TensorMemoryLayout(block=(128, BLOCK_N), col_stride=1)
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, config.num_warps)

    # Wait for accumulator to be ready
    mbarrier.wait(p.acc_ready_bars.index(0), phase=0)

    # Load from TMEM and convert to fp16
    acc = p.acc_bufs.index(0).load(acc_reg_layout)
    result = gl.convert_layout(acc.to(gl.float16), c_layout)

    # Signal that accumulator is consumed
    mbarrier.arrive(p.acc_empty_bars.index(0), count=1)

    # Compute output addresses: decompose M offsets into (batch, out_y, out_x)
    offs_m = prog.pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=c_rows_layout)
    offs_n = prog.pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=c_cols_layout)

    c_batch = offs_m // (config.out_h * config.out_w)
    c_rem = offs_m % (config.out_h * config.out_w)
    c_out_y = c_rem // config.out_w
    c_out_x = c_rem % config.out_w

    c_offsets = (c_batch[:, None] * config.output_stride_n +
                 c_out_y[:, None] * config.output_stride_h +
                 c_out_x[:, None] * config.output_stride_w +
                 offs_n[None, :])
    c_mask = (offs_m[:, None] < M_GEMM) & (offs_n[None, :] < N_GEMM)

    fence_async_shared()
    gl.store(p.output_ptr + c_offsets, result, mask=c_mask)

    # Clean up barriers
    num_buffers: gl.constexpr = p.a_bufs.type.shape[0]
    for i in gl.static_range(num_buffers):
        mbarrier.invalidate(p.load_empty_bars.index(i))
        mbarrier.invalidate(p.load_ready_bars.index(i))
    mbarrier.invalidate(p.acc_empty_bars.index(0))
    mbarrier.invalidate(p.acc_ready_bars.index(0))


# ===-----------------------------------------------------------------------===#
# Kernel Entry Point
# ===-----------------------------------------------------------------------===#


@gluon.jit
def conv2d_im2col_kernel(
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
    """Warp-specialized implicit GEMM convolution kernel using TMA im2col.

    GEMM dimensions:
        M = N * out_h * out_w   (output spatial positions)
        N = Co                  (output channels)
        K = R * S * Ci          (reduction over filter x input channels)
    """
    config = ConvConfig(
        N, H, W, Ci, Co, R, S, out_h, out_w,
        stride_h, stride_w, pad_h, pad_w,
        output_stride_n, output_stride_h, output_stride_w,
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_buffers, num_warps,
    )

    # Allocate shared memory for multi-buffered loads
    a_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, BLOCK_K], gl.float16)

    a_bufs = gl.allocate_shared_memory(gl.float16, [num_buffers, BLOCK_M, BLOCK_K], a_smem_layout)
    b_bufs = gl.allocate_shared_memory(gl.float16, [num_buffers, BLOCK_N, BLOCK_K], b_smem_layout)

    # Barriers for load pipeline
    load_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    load_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)

    # Allocate tensor memory for accumulator
    tmem_layout: gl.constexpr = TensorMemoryLayout(block=(128, BLOCK_N), col_stride=1)
    acc_bufs = allocate_tensor_memory(gl.float32, [1, BLOCK_M, BLOCK_N], tmem_layout)

    # Barriers for accumulator pipeline
    acc_empty_bars = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
    acc_ready_bars = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
    mbarrier.init(acc_empty_bars.index(0), count=1)
    mbarrier.init(acc_ready_bars.index(0), count=1)

    p = PartitionArgs(
        config, in_desc, weight_desc, output,
        a_bufs, b_bufs, load_empty_bars, load_ready_bars,
        acc_bufs, acc_empty_bars, acc_ready_bars,
    )

    gl.warp_specialize([
        (mma_partition, (p,)),
        (load_partition, (p,)),
        (epilogue_partition, (p,)),
    ], [4, 4], [384, 128])


# ===-----------------------------------------------------------------------===#
# Host-Side Entry Point
# ===-----------------------------------------------------------------------===#


def conv2d_im2col(input_tensor, weight_tensor, stride=1, padding=0, num_buffers=2, num_warps=4):
    """
    Warp-specialized implicit GEMM convolution using TMA im2col.

    Args:
        input_tensor: (N, H, W, Ci) - NHWC layout
        weight_tensor: (Co, R, S, Ci) - output channels first
        stride: convolution stride (default: 1)
        padding: convolution padding (default: 0)
        num_buffers: number of pipeline buffers (default: 2)
        num_warps: number of warps for MMA partition (default: 4)
    """
    N, H, W, Ci = input_tensor.shape
    Co, R, S, Ci_w = weight_tensor.shape
    assert Ci == Ci_w, "Input and weight channel dimensions must match"

    out_h = (H + 2 * padding - R) // stride + 1
    out_w = (W + 2 * padding - S) // stride + 1

    output = torch.empty((N, out_h, out_w, Co), device=input_tensor.device, dtype=torch.float16)

    M_GEMM = N * out_h * out_w
    N_GEMM = Co

    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 64
    GROUP_SIZE_M = 4

    grid = (triton.cdiv(M_GEMM, BLOCK_M) * triton.cdiv(N_GEMM, BLOCK_N),)

    # TMA im2col descriptor for input: [N, H, W, Ci] in NHWC
    #
    # The pixel_box defines the access boundary per batch:
    #   Lower = pixel_box_lower_corner + offsets
    #   Upper = [H, W] + pixel_box_upper_corner + offsets
    # With element_strides = [1, stride, stride, 1], TMA steps by `stride`
    # in H/W between output pixels. The window must contain exactly out_h * out_w
    # pixels per batch:
    #   pixels_h = floor((window_h - 1) / stride) + 1 = out_h
    #   => window_h = (out_h - 1) * stride + 1
    #   => upper_h = (out_h - 1) * stride + 1 - H - padding
    upper_h = (out_h - 1) * stride + 1 - H - padding
    upper_w = (out_w - 1) * stride + 1 - W - padding

    input_block_shape = [BLOCK_M, BLOCK_K]
    input_layout = gl.NVMMASharedLayout.get_default_for(input_block_shape, gl.float16)
    in_desc = TensorDescriptorIm2Col(
        base=input_tensor,
        shape=list(input_tensor.shape),
        strides=list(input_tensor.stride()),
        block_shape=input_block_shape,
        layout=input_layout,
        padding="zero",
        element_strides=[1, stride, stride, 1],
        pixel_box_lower_corner=[-padding, -padding],
        pixel_box_upper_corner=[upper_h, upper_w],
    )

    # TMA tiled descriptor for weight: (Co, R*S*Ci) = (N_GEMM, K_GEMM)
    weight_reshaped = weight_tensor.reshape(Co, R * S * Ci)
    weight_block_shape = [BLOCK_N, BLOCK_K]
    weight_layout = gl.NVMMASharedLayout.get_default_for(weight_block_shape, gl.float16)
    weight_desc = TensorDescriptor.from_tensor(weight_reshaped, weight_block_shape, weight_layout)

    conv2d_im2col_kernel[grid](
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


# ===-----------------------------------------------------------------------===#
# Unit Tests
# ===-----------------------------------------------------------------------===#


@pytest.mark.parametrize("N", [1, 128])
@pytest.mark.parametrize("H,W", [(64, 64)])
@pytest.mark.parametrize("Ci,Co", [(384, 384)])
@pytest.mark.parametrize("R,S", [(3, 3), (4, 4), (5, 5)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU (SM 10.x)")
def test_op(N, H, W, Ci, Co, R, S, stride, padding):
    torch.manual_seed(0)

    x_nchw = torch.randn((N, Ci, H, W), device="cuda", dtype=torch.float16)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()

    w_nchw = torch.randn((Co, Ci, R, S), device="cuda", dtype=torch.float16)
    w_nhwc = w_nchw.permute(0, 2, 3, 1).contiguous()  # (Co, R, S, Ci)

    triton_out = conv2d_im2col(x_nhwc, w_nhwc, stride=stride, padding=padding)

    torch_out = torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride, padding=padding)
    torch_out = torch_out.permute(0, 2, 3, 1)  # NCHW -> NHWC

    torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)


# ===-----------------------------------------------------------------------===#
# Benchmarking
# ===-----------------------------------------------------------------------===#

BATCH = [128]
CHANNELS = [(384, 384)]
SPATIAL = [(64, 64)]
FILTER = [(3, 3)]
STRIDE = [1]
PADDING = [1]

bench_configs = []
for N, (Ci, Co), (H, W), (R, S), stride_val, pad_val in [
    (N, ch, sp, f, s, p)
    for N in BATCH
    for ch in CHANNELS
    for sp in SPATIAL
    for f in FILTER
    for s in STRIDE
    for p in PADDING
]:
    config = triton.testing.Benchmark(
        x_names=["num_buffers"],
        x_vals=[2, 3],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name=f"Conv2d N={N} Ci={Ci} Co={Co} H={H} W={W} R={R} S={S} stride={stride_val} pad={pad_val}",
        args={
            "N": N, "H": H, "W": W, "Ci": Ci, "Co": Co,
            "R": R, "S": S, "stride_val": stride_val, "pad_val": pad_val,
        },
    )
    bench_configs.append(config)


@triton.testing.perf_report(bench_configs)
def bench(N, H, W, Ci, Co, R, S, stride_val, pad_val, num_buffers, provider):
    torch.manual_seed(0)

    x_nchw = torch.randn((N, Ci, H, W), device="cuda", dtype=torch.float16)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()
    w_nchw = torch.randn((Co, Ci, R, S), device="cuda", dtype=torch.float16)
    w_nhwc = w_nchw.permute(0, 2, 3, 1).contiguous()

    if provider == "triton":
        fn = lambda: conv2d_im2col(x_nhwc, w_nhwc, stride=stride_val, padding=pad_val, num_buffers=num_buffers)
    elif provider == "torch":
        fn = lambda: torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride_val, padding=pad_val)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    ms = triton.testing.do_bench(fn)

    out_h = (H + 2 * pad_val - R) // stride_val + 1
    out_w = (W + 2 * pad_val - S) // stride_val + 1
    flops = 2.0 * N * out_h * out_w * Co * Ci * R * S
    return flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    bench.run(save_path=".", print_data=True)
