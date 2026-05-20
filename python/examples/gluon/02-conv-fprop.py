import importlib.util
import sys
from pathlib import Path

import pytest
import torch

import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor, TensorDescriptorIm2Col
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tensor_memory_descriptor,
    tcgen05_mma,
    tcgen05_commit,
    tcgen05_mma_barrier_count,
)


def _load_conv_common():
    module_name = "triton_examples_gluon_conv_common"
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    module_path = Path(__file__).with_name("02-conv-common.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load shared conv helpers from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_conv_common = _load_conv_common()

# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#

Counter = _conv_common.Counter
GL_GEMM_DTYPE = _conv_common.GL_GEMM_DTYPE
PersistentTileScheduler = _conv_common.PersistentTileScheduler
TORCH_GEMM_DTYPE = _conv_common.TORCH_GEMM_DTYPE
get_operand_cga_layout = _conv_common.get_operand_cga_layout
get_transposed_cga_layout = _conv_common.get_transposed_cga_layout
invalidate_mbarrier_ring = _conv_common.invalidate_mbarrier_ring
is_blackwell = _conv_common.is_blackwell
maybe_pad_ci_for_tma = _conv_common.maybe_pad_channel_dims_for_tma
normalize_2d = _conv_common.normalize_2d
validate_2cta_m_split = _conv_common.validate_2cta_m_split

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


@gluon.aggregate
class ConvConfig:
    N: gl.tensor
    H: gl.tensor
    W: gl.tensor
    Ci: gl.tensor
    Co: gl.tensor
    R: gl.tensor
    S: gl.tensor
    out_h: gl.tensor
    out_w: gl.tensor
    stride_h: gl.tensor
    stride_w: gl.tensor
    pad_h: gl.tensor
    pad_w: gl.tensor
    output_stride_n: gl.tensor
    output_stride_h: gl.tensor
    output_stride_w: gl.tensor
    M_GEMM: gl.tensor

    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    GROUP_SIZE_M: gl.constexpr
    num_buffers: gl.constexpr
    num_warps: gl.constexpr

    @gluon.jit
    def get_program(self, pid):
        """Compute tile coordinates from program ID with grouped ordering."""
        M_GEMM = self.M_GEMM
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

    @gluon.jit
    def get_num_tiles(self):
        return gl.cdiv(self.M_GEMM, self.BLOCK_M) * gl.cdiv(self.Co, self.BLOCK_N)

    @gluon.jit
    def get_num_k_iterations(self):
        return self.R * self.S * gl.cdiv(self.Ci, self.BLOCK_K)


@gluon.aggregate
class ConvProgram:
    config: ConvConfig
    pid_m: gl.tensor
    pid_n: gl.tensor

    @gluon.jit
    def get_m_offsets(self):
        """Decompose M-tile offset into (batch, out_y, out_x)."""
        offs_m = self.pid_m * self.config.BLOCK_M
        config = self.config
        out_x = offs_m % config.out_w
        out_y = (offs_m // config.out_w) % config.out_h
        batch_id = (offs_m // config.out_w) // config.out_h
        return batch_id, out_y, out_x


# ===-----------------------------------------------------------------------===#
# Partition Arguments
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class PartitionArgs:
    config: ConvConfig
    in_desc: tma.tensor_descriptor_im2col
    weight_desc: tma.tensor_descriptor
    output_desc: tma.tensor_descriptor
    output_ptr: gl.tensor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    acc_bufs: tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor


# ===-----------------------------------------------------------------------===#
# Warp-Specialized Partitions
# ===-----------------------------------------------------------------------===#


@gluon.jit
def load_partition(p):
    """Load partition: iterate over this CTA's assigned output tiles."""
    config = p.config
    BLOCK_K: gl.constexpr = config.BLOCK_K

    empty_bars = p.load_empty_bars
    ready_bars = p.load_ready_bars
    state = Counter.create(1, empty_bars.shape[0])

    num_rs = config.R * config.S
    num_k_iter = config.get_num_k_iterations()

    scheduler = PersistentTileScheduler.initialize(config.get_num_tiles())
    for idx in range(scheduler.get_num_tiles()):
        prog = config.get_program(scheduler.get_tile_id(idx))
        batch_id, out_y, out_x = prog.get_m_offsets()

        for k_iter in range(num_k_iter):
            iter_ci = k_iter // num_rs
            remain_rs = k_iter % num_rs
            iter_s = remain_rs % config.S
            iter_r = remain_rs // config.S

            a_stage = p.a_bufs.index(state.index)
            b_stage = p.b_bufs.index(state.index)
            ready_bar = ready_bars.index(state.index)
            mbarrier.wait(empty_bars.index(state.index), state.phase, deps=[a_stage, b_stage])
            mbarrier.expect(ready_bar, p.in_desc.nbytes_per_cta + p.weight_desc.nbytes_per_cta)

            tma.async_load_im2col(
                p.in_desc,
                [
                    batch_id,
                    out_y * config.stride_h - config.pad_h,
                    out_x * config.stride_w - config.pad_w,
                    iter_ci * BLOCK_K,
                ],
                [iter_r.to(tl.int16), iter_s.to(tl.int16)],
                ready_bar,
                a_stage,
            )

            k_offset = (iter_r * config.S + iter_s) * config.Ci + iter_ci * BLOCK_K
            tma.async_load(
                p.weight_desc,
                [prog.pid_n * config.BLOCK_N, k_offset],
                ready_bar,
                b_stage,
            )
            state = state.next()


@gluon.jit
def mma_partition(p):
    """MMA partition: accumulate over all tiles assigned to this CTA."""
    config = p.config
    TWO_CTAS: gl.constexpr = gl.num_ctas() > 1

    num_k_iter = config.get_num_k_iterations()
    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])

    scheduler = PersistentTileScheduler.initialize(config.get_num_tiles())
    for _ in range(scheduler.get_num_tiles()):
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)
        use_acc = False

        for _k_iter in range(num_k_iter):
            mbarrier.wait(p.load_ready_bars.index(load_state.index), load_state.phase)
            tcgen05_mma(
                p.a_bufs.index(load_state.index),
                p.b_bufs.index(load_state.index).permute((1, 0)),
                acc_buf,
                use_acc=use_acc,
                multicast=TWO_CTAS,
                mbarriers=[p.load_empty_bars.index(load_state.index)],
            )
            load_state = load_state.next()
            use_acc = True

        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()


@gluon.jit
def epilogue_partition(p):
    """Epilogue partition: store all tiles assigned to this CTA."""
    config = p.config
    BLOCK_M: gl.constexpr = config.BLOCK_M
    BLOCK_N: gl.constexpr = config.BLOCK_N
    TWO_CTAS: gl.constexpr = gl.num_ctas() > 1

    acc_state = Counter.create(0, p.acc_empty_bars.shape[0])
    scheduler = PersistentTileScheduler.initialize(config.get_num_tiles())

    if TWO_CTAS:
        EPILOGUE_BLOCK_N: gl.constexpr = p.output_desc.block_shape[1]
        gl.static_assert(BLOCK_N % EPILOGUE_BLOCK_N == 0)
        SUBTILE_FACTOR: gl.constexpr = BLOCK_N // EPILOGUE_BLOCK_N
        SUBTILE_STAGES: gl.constexpr = 1 if SUBTILE_FACTOR == 1 else 2
        acc_smems = gl.allocate_shared_memory(
            p.output_desc.dtype,
            [SUBTILE_STAGES, BLOCK_M, EPILOGUE_BLOCK_N],
            p.output_desc.layout,
        )
        sub_state = Counter.create(0, SUBTILE_STAGES)
        for idx in range(scheduler.get_num_tiles()):
            prog = config.get_program(scheduler.get_tile_id(idx))
            off_m = prog.pid_m * BLOCK_M
            off_n = prog.pid_n * BLOCK_N

            mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
            acc_buf = p.acc_bufs.index(acc_state.index)
            for s in gl.static_range(SUBTILE_FACTOR):
                acc_sub = acc_buf.slice(EPILOGUE_BLOCK_N * s, EPILOGUE_BLOCK_N)
                acc_smem = acc_smems.index(sub_state.index)
                acc_tile = acc_sub.load().to(p.output_desc.dtype)
                tma.store_wait(pendings=SUBTILE_STAGES - 1)
                acc_smem.store(acc_tile)
                tma.async_copy_shared_to_global(p.output_desc, [off_m, off_n + EPILOGUE_BLOCK_N * s], acc_smem)
                sub_state = sub_state.next()

            mbarrier.arrive(p.acc_empty_bars.index(acc_state.index), count=1)
            acc_state = acc_state.next()
        tma.store_wait(0)
    else:
        M_GEMM = config.M_GEMM
        N_GEMM = config.Co
        for idx in range(scheduler.get_num_tiles()):
            prog = config.get_program(scheduler.get_tile_id(idx))

            mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
            acc = p.acc_bufs.index(acc_state.index).load()
            result = gl.convert_layout(acc.to(GL_GEMM_DTYPE), gl.CoalescedLayout())
            mbarrier.arrive(p.acc_empty_bars.index(acc_state.index), count=1)
            acc_state = acc_state.next()

            offs_m = prog.pid_m * BLOCK_M + gl.arange(0, BLOCK_M)
            offs_n = prog.pid_n * BLOCK_N + gl.arange(0, BLOCK_N)

            c_out_x = offs_m % config.out_w
            c_out_y = (offs_m // config.out_w) % config.out_h
            c_batch = (offs_m // config.out_w) // config.out_h

            c_offsets = (c_batch[:, None] * config.output_stride_n + c_out_y[:, None] * config.output_stride_h +
                         c_out_x[:, None] * config.output_stride_w + offs_n[None, :])
            c_mask = (offs_m[:, None] < M_GEMM) & (offs_n[None, :] < N_GEMM)
            gl.store(p.output_ptr + c_offsets, result, mask=c_mask)

    invalidate_mbarrier_ring(p.load_empty_bars)
    invalidate_mbarrier_ring(p.load_ready_bars)
    invalidate_mbarrier_ring(p.acc_empty_bars)
    invalidate_mbarrier_ring(p.acc_ready_bars)


# ===-----------------------------------------------------------------------===#
# Kernel Entry Point
# ===-----------------------------------------------------------------------===#


@gluon.jit(do_not_specialize=[
    "N",
    "H",
    "W",
    "R",
    "S",
    "pad_h",
    "pad_w",
])
def conv2d_fprop_kernel(
    in_desc,
    weight_desc,
    output_desc,
    output,
    N,
    H,
    W,
    Ci,
    Co,
    R,
    S,
    out_h,
    out_w,
    output_stride_n,
    output_stride_h,
    output_stride_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    num_buffers: gl.constexpr,
    num_acc_buffers: gl.constexpr,
    EPILOGUE_BLOCK_N: gl.constexpr,
    CGA_LAYOUT: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Warp-specialized forward convolution kernel."""
    TWO_CTAS: gl.constexpr = gl.num_ctas() > 1
    M_GEMM = N * out_h * out_w
    config = ConvConfig(
        N,
        H,
        W,
        Ci,
        Co,
        R,
        S,
        gl.to_tensor(out_h),
        gl.to_tensor(out_w),
        gl.to_tensor(stride_h),
        gl.to_tensor(stride_w),
        pad_h,
        pad_w,
        gl.to_tensor(output_stride_n),
        gl.to_tensor(output_stride_h),
        gl.to_tensor(output_stride_w),
        M_GEMM,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_SIZE_M,
        num_buffers,
        num_warps,
    )

    a_cga_layout: gl.constexpr = get_operand_cga_layout(CGA_LAYOUT, 0)
    b_cga_layout: gl.constexpr = get_transposed_cga_layout(get_operand_cga_layout(CGA_LAYOUT, 1))
    a_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], GL_GEMM_DTYPE,
                                                                       cga_layout=a_cga_layout)
    b_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, BLOCK_K], GL_GEMM_DTYPE,
                                                                       cga_layout=b_cga_layout)

    a_bufs = gl.allocate_shared_memory(GL_GEMM_DTYPE, [num_buffers, BLOCK_M, BLOCK_K], a_smem_layout)
    b_bufs = gl.allocate_shared_memory(GL_GEMM_DTYPE, [num_buffers, BLOCK_N, BLOCK_K], b_smem_layout)

    mma_barrier_count: gl.constexpr = tcgen05_mma_barrier_count(
        [a_bufs.index(0), b_bufs.index(0).permute((1, 0))],
        multicast=TWO_CTAS,
        two_ctas=TWO_CTAS,
    )
    load_empty_bars = mbarrier.allocate_mbarrier(batch=num_buffers)
    load_ready_bars = mbarrier.allocate_mbarrier(batch=num_buffers, two_ctas=TWO_CTAS)
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=mma_barrier_count)
        mbarrier.init(load_ready_bars.index(i), count=1)

    TMEM_BLOCK_M: gl.constexpr = 64 if BLOCK_M == 64 else 128
    tmem_layout: gl.constexpr = TensorMemoryLayout(
        block=(TMEM_BLOCK_M, BLOCK_N),
        col_stride=1,
        cga_layout=CGA_LAYOUT,
        two_ctas=TWO_CTAS,
    )
    # Smaller tiles can profit from a double-buffered accumulator ring, but
    # large 256x256 tiles exceed Blackwell's TMEM budget unless the ring depth
    # is reduced to 1.
    acc_bufs = allocate_tensor_memory(gl.float32, [num_acc_buffers, BLOCK_M, BLOCK_N], tmem_layout)

    acc_empty_bars = mbarrier.allocate_mbarrier(batch=num_acc_buffers, two_ctas=TWO_CTAS)
    acc_ready_bars = mbarrier.allocate_mbarrier(batch=num_acc_buffers)
    for i in gl.static_range(num_acc_buffers):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    p = PartitionArgs(
        config,
        in_desc,
        weight_desc,
        output_desc,
        output,
        a_bufs,
        b_bufs,
        load_empty_bars,
        load_ready_bars,
        acc_bufs,
        acc_empty_bars,
        acc_ready_bars,
    )

    gl.warp_specialize([
        (epilogue_partition, (p, )),
        (mma_partition, (p, )),
        (load_partition, (p, )),
    ], [1, 1], [24, 24])


def conv2d_fprop_get_configs(pre_hook=None, include_2cta=False, block_n_values=(128, )):
    configs = [
        triton.Config(
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_SIZE_M": group_size_m,
                "num_buffers": num_buffers,
                "num_acc_buffers": num_acc_buffers,
                "EPILOGUE_BLOCK_N": block_n,
                "CGA_LAYOUT": cga_layout,
            },
            num_warps=num_warps,
            num_ctas=2**len(cga_layout),
            pre_hook=pre_hook,
        )
        for block_m in (64, 128)
        for block_n in (8, 32, 128, 256)
        for block_k in (64, )
        for group_size_m in (4, )
        for num_buffers in (3, 4, 5)
        for num_acc_buffers in (2, )
        for cga_layout in ((), )
        for num_warps in (4, )
    ]
    if include_2cta:
        configs.extend([
            triton.Config(
                {
                    "BLOCK_M": 256,
                    "BLOCK_N": block_n,
                    "BLOCK_K": block_k,
                    "GROUP_SIZE_M": 4,
                    "num_buffers": num_buffers,
                    "num_acc_buffers": 2,
                    "EPILOGUE_BLOCK_N": epilogue_block_n,
                    "CGA_LAYOUT": ((1, 0), ),
                },
                num_warps=4,
                num_ctas=2,
                pre_hook=pre_hook,
            )
            for block_n in block_n_values
            for block_k in (64, 128)
            for epilogue_block_n in (block_n, )
            for num_buffers in (3, 4, 5)
        ])
    return configs


def conv2d_fprop_tma_set_block_size_hook(nargs):
    in_block_shape = [nargs["BLOCK_M"], nargs["BLOCK_K"]]
    weight_block_shape = [nargs["BLOCK_N"], nargs["BLOCK_K"]]
    output_block_shape = [nargs["BLOCK_M"], nargs["EPILOGUE_BLOCK_N"]]
    cga_layout = nargs["CGA_LAYOUT"]
    validate_2cta_m_split(cga_layout)

    nargs["in_desc"].block_shape = in_block_shape
    nargs["in_desc"].layout = gl.NVMMASharedLayout.get_default_for(in_block_shape, GL_GEMM_DTYPE,
                                                                   cga_layout=get_operand_cga_layout(cga_layout, 0))

    nargs["weight_desc"].block_shape = weight_block_shape
    weight_cga_layout = get_transposed_cga_layout(get_operand_cga_layout(cga_layout, 1))
    nargs["weight_desc"].layout = gl.NVMMASharedLayout.get_default_for(weight_block_shape, GL_GEMM_DTYPE,
                                                                       cga_layout=weight_cga_layout)

    nargs["output_desc"].block_shape = output_block_shape
    nargs["output_desc"].layout = gl.NVMMASharedLayout.get_default_for(output_block_shape, GL_GEMM_DTYPE,
                                                                       cga_layout=cga_layout)


# Key on the effective implicit-GEMM/convolution geometry instead of the full
# raw input shape. `out_h/out_w` already encode the impact of H/W/padding on
# the launch shape, so keeping all of them would only fragment the autotune
# cache without exposing meaningfully different tile choices.
conv2d_fprop_autotuned_kernel = triton.autotune(
    configs=conv2d_fprop_get_configs(pre_hook=conv2d_fprop_tma_set_block_size_hook),
    key=["out_h", "out_w", "stride_h", "stride_w"],
)(conv2d_fprop_kernel)

# The 2CTA epilogue stores through TMA, so it is only safe when every output
# tile is fully covered. This mixed autotune compares the normal single-CTA
# configs with 2CTA configs, and the host calls it only for full-tile shapes.
conv2d_fprop_autotuned_kernel_with_2cta = triton.autotune(
    configs=conv2d_fprop_get_configs(pre_hook=conv2d_fprop_tma_set_block_size_hook, include_2cta=True),
    key=["N", "Co", "out_h", "out_w", "stride_h", "stride_w"],
)(conv2d_fprop_kernel)

conv2d_fprop_autotuned_kernel_with_2cta_256 = triton.autotune(
    configs=conv2d_fprop_get_configs(
        pre_hook=conv2d_fprop_tma_set_block_size_hook,
        include_2cta=True,
        block_n_values=(128, 256),
    ),
    key=["N", "Co", "out_h", "out_w", "stride_h", "stride_w"],
)(conv2d_fprop_kernel)

# ===-----------------------------------------------------------------------===#
# Host-Side Entry Point
# ===-----------------------------------------------------------------------===#


def _prepare_conv_fprop_inputs(input_tensor, weight_tensor, stride, padding):
    N, H, W, Ci = input_tensor.shape
    Co, R, S, Ci_w = weight_tensor.shape
    assert Ci == Ci_w, "Input and weight channel dimensions must match"
    if input_tensor.dtype != TORCH_GEMM_DTYPE or weight_tensor.dtype != TORCH_GEMM_DTYPE:
        raise ValueError(
            f"conv2d_fprop expects bfloat16 input/weight tensors, got {input_tensor.dtype} and {weight_tensor.dtype}")
    stride_h, stride_w = normalize_2d(stride, "stride")
    pad_h, pad_w = normalize_2d(padding, "padding")
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError(f"stride must be positive, got {(stride_h, stride_w)}")
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"padding must be non-negative, got {(pad_h, pad_w)}")

    # The Hopper/Blackwell TMA path requires outer strides to be 16-byte aligned.
    # For NHWC/OHWI bf16 tensors this means padding the channel dimension for
    # narrow inputs such as RGB (Ci=3).
    input_tensor, weight_tensor = maybe_pad_ci_for_tma(input_tensor, weight_tensor)
    N, H, W, Ci = input_tensor.shape
    Co, R, S, Ci_w = weight_tensor.shape

    out_h = (H + 2 * pad_h - R) // stride_h + 1
    out_w = (W + 2 * pad_w - S) // stride_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Invalid convolution geometry: computed output size "
                         f"({out_h}, {out_w}) from H={H}, W={W}, R={R}, S={S}, "
                         f"stride={(stride_h, stride_w)}, padding={(pad_h, pad_w)}.")

    output = torch.empty((N, out_h, out_w, Co), device=input_tensor.device, dtype=TORCH_GEMM_DTYPE)
    return input_tensor, weight_tensor, output, N, H, W, Ci, Co, R, S, out_h, out_w, stride_h, stride_w, pad_h, pad_w


def _make_conv_fprop_descriptors(input_tensor, weight_tensor, output_matrix, out_h, out_w, stride_h, stride_w, pad_h,
                                 pad_w, input_block_shape, weight_block_shape, output_block_shape, cga_layout=()):
    # TMA im2col descriptor for input: [N, H, W, Ci] in NHWC
    #
    # The pixel_box defines the access boundary per batch:
    #   Lower = pixel_box_lower_corner + offsets
    #   Upper = [H, W] + pixel_box_upper_corner + offsets
    # With element_strides = [1, stride_h, stride_w, 1], TMA steps by the
    # per-dimension convolution stride between output pixels. The window must
    # contain exactly out_h * out_w pixels per batch:
    #   pixels_h = floor((window_h - 1) / stride_h) + 1 = out_h
    #   pixels_w = floor((window_w - 1) / stride_w) + 1 = out_w
    #   => window_h = (out_h - 1) * stride_h + 1
    #   => window_w = (out_w - 1) * stride_w + 1
    _, H, W, _ = input_tensor.shape
    upper_h = (out_h - 1) * stride_h + 1 - H - pad_h
    upper_w = (out_w - 1) * stride_w + 1 - W - pad_w
    Co, R, S, Ci = weight_tensor.shape

    validate_2cta_m_split(cga_layout)
    input_layout = gl.NVMMASharedLayout.get_default_for(input_block_shape, GL_GEMM_DTYPE,
                                                        cga_layout=get_operand_cga_layout(cga_layout, 0))
    in_desc = TensorDescriptorIm2Col.from_tensor(
        input_tensor,
        input_block_shape,
        input_layout,
        padding="zero",
        element_strides=[1, stride_h, stride_w, 1],
        pixel_box_lower_corner=[-pad_h, -pad_w],
        pixel_box_upper_corner=[upper_h, upper_w],
    )

    # TMA tiled descriptor for weight: (Co, R*S*Ci) = (N_GEMM, K_GEMM)
    weight_reshaped = weight_tensor.reshape(Co, R * S * Ci)
    weight_cga_layout = get_transposed_cga_layout(get_operand_cga_layout(cga_layout, 1))
    weight_layout = gl.NVMMASharedLayout.get_default_for(weight_block_shape, GL_GEMM_DTYPE,
                                                         cga_layout=weight_cga_layout)
    weight_desc = TensorDescriptor.from_tensor(weight_reshaped, weight_block_shape, weight_layout)

    output_layout = gl.NVMMASharedLayout.get_default_for(output_block_shape, GL_GEMM_DTYPE, cga_layout=cga_layout)
    output_desc = TensorDescriptor.from_tensor(output_matrix, output_block_shape, output_layout)
    return in_desc, weight_desc, output_desc


def _make_grid(num_sms, M_GEMM, N_GEMM):

    def grid(meta):
        num_tiles = triton.cdiv(M_GEMM, meta["BLOCK_M"]) * triton.cdiv(N_GEMM, meta["BLOCK_N"])
        return (min(num_sms, num_tiles), )

    return grid


def _launch_conv(
    kernel,
    grid,
    *,
    in_desc,
    weight_desc,
    output_desc,
    output,
    N,
    H,
    W,
    Ci,
    Co,
    R,
    S,
    out_h,
    out_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    kernel_meta=None,
):
    if kernel_meta is None:
        kernel_meta = {}

    kernel[grid](
        in_desc,
        weight_desc,
        output_desc,
        output,
        N,
        H,
        W,
        Ci,
        Co,
        R,
        S,
        out_h,
        out_w,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        **kernel_meta,
    )


def _supports_2cta_fprop_autotune(M_GEMM, Co, block_n):
    return M_GEMM % 256 == 0 and Co % block_n == 0


def conv2d_fprop(input_tensor, weight_tensor, stride=1, padding=0, **kwargs):
    """Production fprop entrypoint.

    Selects the best kernel configuration with Triton autotuning for the given
    convolution shape.
    """
    input_tensor, weight_tensor, output, N, H, W, Ci, Co, R, S, out_h, out_w, stride_h, stride_w, pad_h, pad_w = \
        _prepare_conv_fprop_inputs(input_tensor, weight_tensor, stride, padding)

    M_GEMM = N * out_h * out_w
    N_GEMM = Co
    num_sms = torch.cuda.get_device_properties(input_tensor.device).multi_processor_count
    output_matrix = output.view(-1, Co)

    dummy_block_shape = [1, 1]
    in_desc, weight_desc, output_desc = _make_conv_fprop_descriptors(
        input_tensor,
        weight_tensor,
        output_matrix,
        out_h,
        out_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dummy_block_shape,
        dummy_block_shape,
        dummy_block_shape,
    )

    kernel = conv2d_fprop_autotuned_kernel
    if _supports_2cta_fprop_autotune(M_GEMM, Co, 256):
        kernel = conv2d_fprop_autotuned_kernel_with_2cta_256
    elif _supports_2cta_fprop_autotune(M_GEMM, Co, 128):
        kernel = conv2d_fprop_autotuned_kernel_with_2cta

    _launch_conv(
        kernel,
        _make_grid(num_sms, M_GEMM, N_GEMM),
        in_desc=in_desc,
        weight_desc=weight_desc,
        output_desc=output_desc,
        output=output,
        N=N,
        H=H,
        W=W,
        Ci=Ci,
        Co=Co,
        R=R,
        S=S,
        out_h=out_h,
        out_w=out_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
    )

    return output


conv2d_fprop_persistent = conv2d_fprop


def _make_conv2d_fprop_fixed_kernel_meta(num_buffers, num_warps, *, use_2cta=False):
    if use_2cta:
        cga_layout = ((1, 0), )
        return {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_SIZE_M": 4,
            "num_buffers": 4 if num_buffers is None else min(num_buffers, 4),
            "num_acc_buffers": 2,
            "EPILOGUE_BLOCK_N": 128,
            "CGA_LAYOUT": cga_layout,
            "num_warps": num_warps,
            "num_ctas": 2**len(cga_layout),
        }

    return {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "GROUP_SIZE_M": 4,
        "num_buffers": 3 if num_buffers is None else num_buffers,
        "num_acc_buffers": 2,
        "EPILOGUE_BLOCK_N": 128,
        "CGA_LAYOUT": (),
        "num_warps": num_warps,
    }


def _run_conv2d_fprop_fixed(input_tensor, weight_tensor, stride=1, padding=0, num_buffers=None, num_warps=4, *,
                            use_2cta=False):
    """Fixed-config fprop entrypoint used for CI and debugging.

    Runs the kernel with a fixed supported tile shape instead of autotuning.
    """
    input_tensor, weight_tensor, output, N, H, W, Ci, Co, R, S, out_h, out_w, stride_h, stride_w, pad_h, pad_w = \
        _prepare_conv_fprop_inputs(input_tensor, weight_tensor, stride, padding)

    kernel_meta = _make_conv2d_fprop_fixed_kernel_meta(num_buffers, num_warps, use_2cta=use_2cta)
    BLOCK_M = kernel_meta["BLOCK_M"]
    BLOCK_N = kernel_meta["BLOCK_N"]
    BLOCK_K = kernel_meta["BLOCK_K"]
    EPILOGUE_BLOCK_N = kernel_meta["EPILOGUE_BLOCK_N"]

    M_GEMM = N * out_h * out_w
    N_GEMM = Co
    num_sms = torch.cuda.get_device_properties(input_tensor.device).multi_processor_count
    output_matrix = output.view(-1, Co)

    in_desc, weight_desc, output_desc = _make_conv_fprop_descriptors(
        input_tensor,
        weight_tensor,
        output_matrix,
        out_h,
        out_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        [BLOCK_M, BLOCK_K],
        [BLOCK_N, BLOCK_K],
        [BLOCK_M, EPILOGUE_BLOCK_N],
        kernel_meta["CGA_LAYOUT"],
    )

    _launch_conv(
        conv2d_fprop_kernel,
        _make_grid(num_sms, M_GEMM, N_GEMM),
        in_desc=in_desc,
        weight_desc=weight_desc,
        output_desc=output_desc,
        output=output,
        N=N,
        H=H,
        W=W,
        Ci=Ci,
        Co=Co,
        R=R,
        S=S,
        out_h=out_h,
        out_w=out_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        kernel_meta=kernel_meta,
    )

    return output


def conv2d_fprop_fixed(input_tensor, weight_tensor, stride=1, padding=0, num_buffers=None, num_warps=4):
    """Fixed-config fprop entrypoint used for CI and debugging.

    Uses the regular fprop kernel with a 2CTA M-split tile when the output
    matrix shape is fully covered by TMA-store tiles. Small or partial-tile
    cases use the same kernel in single-CTA mode with scalar stores.
    """
    stride_h, stride_w = normalize_2d(stride, "stride")
    pad_h, pad_w = normalize_2d(padding, "padding")
    N, H, W, Ci = input_tensor.shape
    Co, R, S, Ci_w = weight_tensor.shape
    use_2cta = False
    if Ci == Ci_w:
        out_h = (H + 2 * pad_h - R) // stride_h + 1
        out_w = (W + 2 * pad_w - S) // stride_w + 1
        if out_h > 0 and out_w > 0:
            use_2cta = (N * out_h * out_w) % 256 == 0 and Co % 128 == 0

    return _run_conv2d_fprop_fixed(
        input_tensor,
        weight_tensor,
        stride=stride,
        padding=padding,
        num_buffers=num_buffers,
        num_warps=num_warps,
        use_2cta=use_2cta,
    )


# ===-----------------------------------------------------------------------===#
# Unit Tests
# ===-----------------------------------------------------------------------===#


def _assert_conv_fprop_correct(fprop_fn, N, Ci, H, W, Co, R, S, stride, padding, **kwargs):
    """Run fprop_fn on NHWC tensors and compare against torch.nn.functional.conv2d."""
    torch.manual_seed(0)
    x_nchw = torch.randn((N, Ci, H, W), device="cuda", dtype=TORCH_GEMM_DTYPE)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()
    w_nchw = torch.randn((Co, Ci, R, S), device="cuda", dtype=TORCH_GEMM_DTYPE)
    w_nhwc = w_nchw.permute(0, 2, 3, 1).contiguous()

    triton_out = fprop_fn(x_nhwc, w_nhwc, stride=stride, padding=padding, **kwargs)
    torch_out = torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride, padding=padding)
    torch_out = torch_out.permute(0, 2, 3, 1)
    torch.testing.assert_close(triton_out, torch_out, atol=5e-2, rtol=5e-2)


FPROP_2CTA_PARAMS = [
    pytest.param(conv2d_fprop_fixed, 1, 384, 32, 32, 512, 3, 3, 1, 1, id="2cta_n1_ci384_co512_r3s3"),
    pytest.param(conv2d_fprop_fixed, 128, 384, 8, 8, 512, 3, 3, 1, 1, id="2cta_n128_ci384_co512_r3s3"),
    pytest.param(conv2d_fprop_fixed, 1, 416, 32, 32, 512, 3, 3, 1, 1, id="2cta_padded_ci416_co512"),
]


@pytest.mark.parametrize("fprop_fn,N,Ci,H,W,Co,R,S,stride,padding", [
    *FPROP_2CTA_PARAMS, *[(conv2d_fprop_fixed, N, Ci, 64, 64, Co, R, S, stride, padding)
                          for N in (1, 128)
                          for Ci, Co in ((384, 384), (416, 416))
                          for R, S in ((3, 3), (4, 4), (5, 5))
                          for stride in (1, 2)
                          for padding in (0, 1)],
    (conv2d_fprop_fixed, 1, 96, 1, 8, 128, 1, 2, (1, 2), 0),  # asymmetric stride
    (conv2d_fprop_fixed, 16, 5, 32, 32, 96, 3, 3, 1, 1),  # padded channels
])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU (SM 10.x)")
def test_op(fprop_fn, N, Ci, H, W, Co, R, S, stride, padding):
    _assert_conv_fprop_correct(fprop_fn, N, Ci, H, W, Co, R, S, stride, padding)


# ===-----------------------------------------------------------------------===#
# Benchmarking
# ===-----------------------------------------------------------------------===#

BATCH = [128]
CHANNELS = [(384, 384)]
SPATIAL = [(64, 64)]
FILTER = [(3, 3)]
STRIDE = [1]
PADDING = [1]


def _make_bench_inputs(N, H, W, Ci, Co, R, S):
    torch.manual_seed(0)

    x_nchw = torch.randn((N, Ci, H, W), device="cuda", dtype=TORCH_GEMM_DTYPE)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()
    w_nchw = torch.randn((Co, Ci, R, S), device="cuda", dtype=TORCH_GEMM_DTYPE)
    w_nhwc = w_nchw.permute(0, 2, 3, 1).contiguous()
    return x_nchw, x_nhwc, w_nchw, w_nhwc


def _benchmark_tflops(fn, *, N, H, W, Ci, Co, R, S, stride_val, pad_val):
    ms = triton.testing.do_bench(fn)
    out_h = (H + 2 * pad_val - R) // stride_val + 1
    out_w = (W + 2 * pad_val - S) // stride_val + 1
    flops = 2.0 * N * out_h * out_w * Co * Ci * R * S
    return flops * 1e-12 / (ms * 1e-3)


bench_configs = []
for N, (Ci, Co), (H, W), (R, S), stride_val, pad_val in [(N, ch, sp, f, s, p)
                                                         for N in BATCH
                                                         for ch in CHANNELS
                                                         for sp in SPATIAL
                                                         for f in FILTER
                                                         for s in STRIDE
                                                         for p in PADDING]:
    bench_configs.append(
        triton.testing.Benchmark(
            x_names=["kernel"],
            x_vals=["autotuned"],
            line_arg="provider",
            line_vals=["gluon", "torch"],
            line_names=["Gluon (autotuned)", "PyTorch"],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name=f"Conv2d N={N} Ci={Ci} Co={Co} H={H} W={W} R={R} S={S} stride={stride_val} pad={pad_val}",
            args={
                "N": N,
                "H": H,
                "W": W,
                "Ci": Ci,
                "Co": Co,
                "R": R,
                "S": S,
                "stride_val": stride_val,
                "pad_val": pad_val,
            },
        ))


@triton.testing.perf_report(bench_configs)
def bench(N, H, W, Ci, Co, R, S, stride_val, pad_val, kernel, provider):
    x_nchw, x_nhwc, w_nchw, w_nhwc = _make_bench_inputs(N, H, W, Ci, Co, R, S)

    if provider == "gluon":
        fn = lambda: conv2d_fprop(x_nhwc, w_nhwc, stride=stride_val, padding=pad_val)
    elif provider == "torch":
        fn = lambda: torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride_val, padding=pad_val)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return _benchmark_tflops(
        fn,
        N=N,
        H=H,
        W=W,
        Ci=Ci,
        Co=Co,
        R=R,
        S=S,
        stride_val=stride_val,
        pad_val=pad_val,
    )


if __name__ == "__main__":
    bench.run(save_path=".", print_data=True)
