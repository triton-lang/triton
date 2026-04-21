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
init_mbarrier_ring = _conv_common.init_mbarrier_ring
invalidate_mbarrier_ring = _conv_common.invalidate_mbarrier_ring
is_blackwell = _conv_common.is_blackwell
maybe_pad_ci_for_tma = _conv_common.maybe_pad_channel_dims_for_tma
normalize_2d = _conv_common.normalize_2d

# ===-----------------------------------------------------------------------===#
# Wgrad GEMM mapping
# ===-----------------------------------------------------------------------===#
#
# grad_W[Co, R*S*Ci] = grad_out[M, Co]^T  @  im2col(input)[M, R*S*Ci]
#
# where M = N * out_h * out_w   (spatial positions — reduction dimension)
#
# MMA tiling:
#   BLOCK_M = tile over Co           (rows of grad_weight)
#   BLOCK_N = tile over Ci per (r,s) (cols of grad_weight)
#   BLOCK_K = tile over spatial       (reduction)
#
# Logical tile space: cdiv(Co, BLOCK_M) * R * S * cdiv(Ci, BLOCK_N), optionally
# multiplied by split-K. The launch uses a persistent scheduler and runs only
# `min(num_sms, logical_tiles)` CTAs.
#
# Loads per K iteration:
#   A = grad_out tile: TMA tiled on (M_spatial, Co),
#       block [BLOCK_K, BLOCK_M] — permuted to [M, K] in kernel.
#   B = im2col(input) tile: TMA im2col on [N,H,W,Ci], block [BLOCK_K, BLOCK_N]
#       Already [K, N], no kernel permute.
#
# MMA: acc[BLOCK_M, BLOCK_N] += A.permute(1,0) @ B

# ===-----------------------------------------------------------------------===#
# Wgrad Configuration
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class WgradConfig:
    N: gl.tensor
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
    K_GEMM: gl.tensor
    M_spatial: gl.tensor

    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    SPLIT_K: gl.constexpr
    num_buffers: gl.constexpr
    num_warps: gl.constexpr

    @gluon.jit
    def get_num_output_tiles(self):
        co_num_blocks = gl.cdiv(self.Co, self.BLOCK_M)
        ci_num_blocks = gl.cdiv(self.Ci, self.BLOCK_N)
        return co_num_blocks * self.R * self.S * ci_num_blocks

    @gluon.jit
    def get_num_k_iterations(self):
        return gl.cdiv(self.M_spatial, self.BLOCK_K)

    @gluon.jit
    def get_active_split_k(self):
        total_k_iters = self.get_num_k_iterations()
        k_iters_per_split = gl.cdiv(total_k_iters, self.SPLIT_K)
        return gl.cdiv(total_k_iters, k_iters_per_split)

    @gluon.jit
    def get_num_tiles(self):
        return self.get_num_output_tiles() * self.get_active_split_k()

    @gluon.jit
    def get_program(self, pid):
        active_split_k = self.get_active_split_k()
        split_k_idx = pid % active_split_k
        tile_id = pid // active_split_k

        ci_num_blocks = gl.cdiv(self.Ci, self.BLOCK_N)
        co_num_blocks = gl.cdiv(self.Co, self.BLOCK_M)
        pid_co = tile_id % co_num_blocks
        pid_n = tile_id // co_num_blocks

        ci_block = pid_n % ci_num_blocks
        rs_idx = pid_n // ci_num_blocks
        iter_r = rs_idx // self.S
        iter_s = rs_idx % self.S

        total_k_iters = self.get_num_k_iterations()
        k_iters_per_split = gl.cdiv(total_k_iters, active_split_k)
        k_start = split_k_idx * k_iters_per_split
        remaining_k_iters = total_k_iters - k_start
        zero = gl.to_tensor(0)
        k_iters_this_split = gl.where(
            remaining_k_iters > 0,
            gl.minimum(k_iters_per_split, remaining_k_iters),
            zero,
        )

        return WgradProgram(self, pid_co, ci_block, iter_r, iter_s, split_k_idx, k_start, k_iters_this_split)


@gluon.aggregate
class WgradProgram:
    config: WgradConfig
    pid_co: gl.tensor
    ci_block: gl.tensor
    iter_r: gl.tensor
    iter_s: gl.tensor
    split_k_idx: gl.tensor
    k_start: gl.tensor
    k_iters_this_split: gl.tensor

    @gluon.jit
    def get_co_offset(self):
        return self.pid_co * self.config.BLOCK_M

    @gluon.jit
    def get_ci_offset(self):
        return self.ci_block * self.config.BLOCK_N

    @gluon.jit
    def get_spatial_offsets(self, local_k):
        m_global = (self.k_start + local_k) * self.config.BLOCK_K
        spatial_per_batch = self.config.out_h * self.config.out_w
        m_in_batch = m_global % spatial_per_batch
        batch = m_global // spatial_per_batch
        out_x = m_in_batch % self.config.out_w
        out_y = m_in_batch // self.config.out_w
        return m_global, batch, out_y, out_x

    @gluon.jit
    def get_weight_k_offset(self):
        return (self.iter_r * self.config.S + self.iter_s) * self.config.Ci + self.get_ci_offset()


# ===-----------------------------------------------------------------------===#
# Partition Arguments
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class PartitionArgs:
    config: WgradConfig
    in_desc: tma.tensor_descriptor_im2col
    grad_out_desc: tma.tensor_descriptor
    grad_weight_ptr: gl.tensor
    grad_weight_stride_0: gl.tensor
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
    """Load partition: iterate over the persistent wgrad work items assigned to this CTA."""
    config = p.config

    empty_bars = p.load_empty_bars
    ready_bars = p.load_ready_bars
    state = Counter.create(1, empty_bars.shape[0])
    scheduler = PersistentTileScheduler.initialize(config.get_num_tiles())

    for idx in range(scheduler.get_num_tiles()):
        prog = config.get_program(scheduler.get_tile_id(idx))
        co_offset = prog.get_co_offset()
        ci_offset = prog.get_ci_offset()

        for local_k in range(prog.k_iters_this_split):
            m_global, batch, out_y, out_x = prog.get_spatial_offsets(local_k)
            ready_bar = ready_bars.index(state.index)
            mbarrier.wait(empty_bars.index(state.index), state.phase)
            mbarrier.expect(ready_bar, p.grad_out_desc.block_type.nbytes + p.in_desc.block_type.nbytes)

            # A = grad_output: (M_spatial, Co), block [BLOCK_K, BLOCK_M]
            tma.async_load(
                p.grad_out_desc,
                [m_global, co_offset],
                ready_bar,
                p.a_bufs.index(state.index),
            )

            # B = im2col(input): [N, H, W, Ci], block [BLOCK_K, BLOCK_N]
            tma.async_load_im2col(
                p.in_desc,
                [
                    batch,
                    out_y * config.stride_h - config.pad_h,
                    out_x * config.stride_w - config.pad_w,
                    ci_offset,
                ],
                [prog.iter_r.to(tl.int16), prog.iter_s.to(tl.int16)],
                ready_bar,
                p.b_bufs.index(state.index),
            )
            state = state.next()


@gluon.jit
def mma_partition(p):
    """MMA partition: accumulate all split-K work items assigned to this CTA."""
    config = p.config
    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])
    scheduler = PersistentTileScheduler.initialize(config.get_num_tiles())

    for idx in range(scheduler.get_num_tiles()):
        prog = config.get_program(scheduler.get_tile_id(idx))

        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)
        use_acc = False

        for _local_k in range(prog.k_iters_this_split):
            mbarrier.wait(p.load_ready_bars.index(load_state.index), load_state.phase)
            tcgen05_mma(
                p.a_bufs.index(load_state.index).permute((1, 0)),
                p.b_bufs.index(load_state.index),
                acc_buf,
                use_acc=use_acc,
            )
            tcgen05_commit(p.load_empty_bars.index(load_state.index))
            load_state = load_state.next()
            use_acc = True

        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()


@gluon.jit
def epilogue_partition(p):
    """Epilogue partition: store the persistent wgrad work items assigned to this CTA."""
    config = p.config
    active_split_k = config.get_active_split_k()
    BLOCK_M: gl.constexpr = config.BLOCK_M
    BLOCK_N: gl.constexpr = config.BLOCK_N
    acc_state = Counter.create(0, p.acc_empty_bars.shape[0])
    scheduler = PersistentTileScheduler.initialize(config.get_num_tiles())

    for idx in range(scheduler.get_num_tiles()):
        prog = config.get_program(scheduler.get_tile_id(idx))
        co_offset = prog.get_co_offset()
        ci_offset = prog.get_ci_offset()
        weight_k_offset = prog.get_weight_k_offset()

        mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
        acc = p.acc_bufs.index(acc_state.index).load()
        result = gl.convert_layout(acc, gl.CoalescedLayout())
        mbarrier.arrive(p.acc_empty_bars.index(acc_state.index), count=1)
        acc_state = acc_state.next()

        split_co_offset = gl.where(active_split_k > 1, prog.split_k_idx * config.Co, gl.to_tensor(0))
        offs_m = co_offset + gl.arange(0, BLOCK_M)
        offs_n = weight_k_offset + gl.arange(0, BLOCK_N)

        ci_valid = (ci_offset + gl.arange(0, BLOCK_N)) < config.Ci
        mask = (offs_m[:, None] < config.Co) & (offs_n[None, :] < config.K_GEMM) & ci_valid[None, :]
        store_rows = split_co_offset + offs_m
        offsets = store_rows[:, None] * p.grad_weight_stride_0 + offs_n[None, :]
        gl.store(p.grad_weight_ptr + offsets, result, mask=mask)

    invalidate_mbarrier_ring(p.load_empty_bars)
    invalidate_mbarrier_ring(p.load_ready_bars)
    invalidate_mbarrier_ring(p.acc_empty_bars)
    invalidate_mbarrier_ring(p.acc_ready_bars)


# ===-----------------------------------------------------------------------===#
# Kernel Entry Point
# ===-----------------------------------------------------------------------===#


@gluon.jit(do_not_specialize=[
    "N",
    "R",
    "S",
    "out_h",
    "out_w",
    "stride_h",
    "stride_w",
    "pad_h",
    "pad_w",
])
def conv2d_wgrad_kernel(
    in_desc,
    grad_out_desc,
    grad_weight,
    N,
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
    K_GEMM,
    grad_weight_stride_0,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SPLIT_K: gl.constexpr,
    num_buffers: gl.constexpr,
    num_acc_buffers: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Warp-specialized wgrad kernel: grad_W = grad_out^T @ im2col(input).

    GEMM dimensions (per CTA):
        M = Co tile                  (output rows)
        N = Ci tile at fixed (r,s)   (output cols)
        K = N_batch * out_h * out_w  (spatial reduction, split across SPLIT_K CTAs)
    """
    M_spatial = N * out_h * out_w
    config = WgradConfig(
        N,
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
        K_GEMM,
        M_spatial,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        SPLIT_K,
        num_buffers,
        num_warps,
    )

    # a_bufs: grad_output tiles [BLOCK_K, BLOCK_M] (spatial × Co)
    # TMA loads from (M_spatial, Co), permuted to [BLOCK_M, BLOCK_K] at MMA call.
    a_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_M], GL_GEMM_DTYPE)
    # b_bufs: im2col input tiles [BLOCK_K, BLOCK_N] (spatial × Ci)
    b_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], GL_GEMM_DTYPE)

    a_bufs = gl.allocate_shared_memory(GL_GEMM_DTYPE, [num_buffers, BLOCK_K, BLOCK_M], a_smem_layout)
    b_bufs = gl.allocate_shared_memory(GL_GEMM_DTYPE, [num_buffers, BLOCK_K, BLOCK_N], b_smem_layout)

    load_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    load_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    init_mbarrier_ring(load_empty_bars)
    init_mbarrier_ring(load_ready_bars)

    TMEM_BLOCK_M: gl.constexpr = 64 if BLOCK_M == 64 else 128
    tmem_layout: gl.constexpr = TensorMemoryLayout(block=(TMEM_BLOCK_M, BLOCK_N), col_stride=1)
    acc_bufs = allocate_tensor_memory(gl.float32, [num_acc_buffers, BLOCK_M, BLOCK_N], tmem_layout)

    acc_empty_bars = gl.allocate_shared_memory(gl.int64, [num_acc_buffers, 1], mbarrier.MBarrierLayout())
    acc_ready_bars = gl.allocate_shared_memory(gl.int64, [num_acc_buffers, 1], mbarrier.MBarrierLayout())
    init_mbarrier_ring(acc_empty_bars)
    init_mbarrier_ring(acc_ready_bars)

    p = PartitionArgs(
        config,
        in_desc,
        grad_out_desc,
        grad_weight,
        gl.to_tensor(grad_weight_stride_0),
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


# ===-----------------------------------------------------------------------===#
# Autotuning
# ===-----------------------------------------------------------------------===#


def conv2d_wgrad_get_configs(pre_hook=None):
    return [
        triton.Config(
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "SPLIT_K": split_k,
                "num_buffers": num_buffers,
                "num_acc_buffers": num_acc_buffers,
            },
            num_warps=num_warps,
            pre_hook=pre_hook,
        )
        for block_m in (64, 128)
        for block_n in (64, 128, 256)
        for block_k in (64, )
        for split_k in (1, 2, 4, 8, 16, 32)
        for num_buffers in (3, 4)
        for num_acc_buffers in (2, )
        for num_warps in (4, )
    ]


# ===-----------------------------------------------------------------------===#
# Host-Side Entry Point
# ===-----------------------------------------------------------------------===#


def _prepare_wgrad_problem(input_nhwc, grad_output_nhwc, R, S, stride, padding):
    """Validate inputs, pad channels, and return derived quantities."""
    if input_nhwc.dtype != TORCH_GEMM_DTYPE or grad_output_nhwc.dtype != TORCH_GEMM_DTYPE:
        raise ValueError(
            f"conv2d_wgrad expects bfloat16 input and grad-output tensors, got {input_nhwc.dtype} and {grad_output_nhwc.dtype}"
        )

    stride_h, stride_w = normalize_2d(stride, "stride")
    pad_h, pad_w = normalize_2d(padding, "padding")
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError(f"stride must be positive, got {(stride_h, stride_w)}")
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"padding must be non-negative, got {(pad_h, pad_w)}")

    N, H, W, Ci_orig = input_nhwc.shape
    N2, out_h, out_w, Co = grad_output_nhwc.shape
    assert N == N2, "Batch size mismatch"

    expected_out_h = (H + 2 * pad_h - R) // stride_h + 1
    expected_out_w = (W + 2 * pad_w - S) // stride_w + 1
    if out_h != expected_out_h or out_w != expected_out_w:
        raise ValueError("Grad-output shape mismatch: expected "
                         f"({N}, {expected_out_h}, {expected_out_w}, {Co}) from input/filter geometry, got "
                         f"({N2}, {out_h}, {out_w}, {Co}).")
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Invalid convolution geometry for wgrad")

    input_nhwc = maybe_pad_ci_for_tma(input_nhwc)
    Ci = input_nhwc.shape[-1]
    K_GEMM = R * S * Ci

    return input_nhwc, grad_output_nhwc, Ci_orig, N, Ci, Co, out_h, out_w, stride_h, stride_w, pad_h, pad_w, K_GEMM


def _allocate_wgrad_output(device, Co, K_GEMM):
    return torch.zeros((Co, K_GEMM), device=device, dtype=torch.float32)


def _make_wgrad_descriptors(input_nhwc, grad_output_nhwc, Co, out_h, out_w, stride_h, stride_w, pad_h, pad_w,
                            input_block_shape, grad_out_block_shape):
    """Create TMA descriptors for wgrad im2col and grad_output."""
    # TMA im2col descriptor for the activation tensor [N, H, W, Ci] in NHWC.
    _, H, W, _ = input_nhwc.shape
    upper_h = (out_h - 1) * stride_h + 1 - H - pad_h
    upper_w = (out_w - 1) * stride_w + 1 - W - pad_w

    input_layout = gl.NVMMASharedLayout.get_default_for(input_block_shape, GL_GEMM_DTYPE)
    in_desc = TensorDescriptorIm2Col(
        base=input_nhwc,
        shape=list(input_nhwc.shape),
        strides=list(input_nhwc.stride()),
        block_shape=input_block_shape,
        layout=input_layout,
        padding="zero",
        element_strides=[1, stride_h, stride_w, 1],
        pixel_box_lower_corner=[-pad_h, -pad_w],
        pixel_box_upper_corner=[upper_h, upper_w],
    )

    # TMA tiled descriptor for grad_output reshaped as (M_spatial, Co).
    M_spatial = input_nhwc.shape[0] * out_h * out_w
    grad_out_2d = grad_output_nhwc.reshape(M_spatial, Co)
    grad_out_layout = gl.NVMMASharedLayout.get_default_for(grad_out_block_shape, GL_GEMM_DTYPE)
    grad_out_desc = TensorDescriptor.from_tensor(grad_out_2d, grad_out_block_shape, grad_out_layout)

    return in_desc, grad_out_desc


def _make_grid(num_sms, M_spatial, Co, Ci, R, S):

    def grid(meta):
        co_blocks = triton.cdiv(Co, meta["BLOCK_M"])
        ci_blocks = triton.cdiv(Ci, meta["BLOCK_N"])
        total_k_iters = triton.cdiv(M_spatial, meta["BLOCK_K"])
        k_iters_per_split = triton.cdiv(total_k_iters, meta["SPLIT_K"])
        active_split_k = triton.cdiv(total_k_iters, k_iters_per_split)
        total_tiles = co_blocks * R * S * ci_blocks * active_split_k
        return (min(num_sms, total_tiles), )

    return grid


def _get_active_split_k(M_spatial, BLOCK_K, SPLIT_K):
    total_k_iters = triton.cdiv(M_spatial, BLOCK_K)
    k_iters_per_split = triton.cdiv(total_k_iters, SPLIT_K)
    return triton.cdiv(total_k_iters, k_iters_per_split)


def _get_safe_wgrad_active_split_k(M_spatial, Co, K_GEMM, kernel_meta):
    active_split_k = _get_active_split_k(M_spatial, kernel_meta["BLOCK_K"], kernel_meta["SPLIT_K"])
    if active_split_k > 1:
        # The split-K workspace is indexed as row * stride + col inside the kernel.
        # Very large workspaces can exceed the addressing range supported by the generated code.
        workspace_elems = active_split_k * Co * K_GEMM
        if workspace_elems > (2**31 - 1):
            raise ValueError("wgrad split-K workspace exceeds safe indexing range: "
                             f"active_split_k={active_split_k}, Co={Co}, K_GEMM={K_GEMM}")
    return active_split_k


def _allocate_wgrad_split_k_workspace(device, active_split_k, Co, K_GEMM):
    return torch.empty((active_split_k * Co, K_GEMM), device=device, dtype=torch.float32)


_wgrad_autotune_cache = {}


def _make_wgrad_autotune_key(
    device,
    num_sms,
    N,
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
):
    return (
        torch.cuda.get_device_capability(device),
        num_sms,
        N,
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
    )


def _make_wgrad_runner(
    input_nhwc,
    grad_output_nhwc,
    grad_weight_flat,
    *,
    N,
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
    K_GEMM,
    num_sms,
    kernel_meta,
):
    M_spatial = N * out_h * out_w
    active_split_k = _get_safe_wgrad_active_split_k(M_spatial, Co, K_GEMM, kernel_meta)
    uses_split_k_workspace = active_split_k > 1
    launch_output = grad_weight_flat
    if uses_split_k_workspace:
        launch_output = _allocate_wgrad_split_k_workspace(input_nhwc.device, active_split_k, Co, K_GEMM)

    in_desc, grad_out_desc = _make_wgrad_descriptors(
        input_nhwc,
        grad_output_nhwc,
        Co,
        out_h,
        out_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        [kernel_meta["BLOCK_K"], kernel_meta["BLOCK_N"]],
        [kernel_meta["BLOCK_K"], kernel_meta["BLOCK_M"]],
    )
    grid = _make_grid(num_sms, M_spatial, Co, Ci, R, S)

    def run():
        _launch_wgrad(
            conv2d_wgrad_kernel,
            grid,
            in_desc=in_desc,
            grad_out_desc=grad_out_desc,
            grad_weight=launch_output,
            N=N,
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
            K_GEMM=K_GEMM,
            kernel_meta=kernel_meta,
        )
        if uses_split_k_workspace:
            _reduce_wgrad_split_k_partials(launch_output, grad_weight_flat, Co, K_GEMM, active_split_k)

    return run


def _benchmark_wgrad_config(
    input_nhwc,
    grad_output_nhwc,
    *,
    N,
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
    K_GEMM,
    num_sms,
    kernel_meta,
):
    try:
        grad_weight_flat = torch.empty((Co, K_GEMM), device=input_nhwc.device, dtype=torch.float32)
        run = _make_wgrad_runner(
            input_nhwc,
            grad_output_nhwc,
            grad_weight_flat,
            N=N,
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
            K_GEMM=K_GEMM,
            num_sms=num_sms,
            kernel_meta=kernel_meta,
        )
        run()
        torch.cuda.synchronize()
        return triton.testing.do_bench(run)
    except Exception:
        return float("inf")


def _select_wgrad_kernel_meta(
    input_nhwc,
    grad_output_nhwc,
    *,
    N,
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
    K_GEMM,
    num_sms,
):
    cache_key = _make_wgrad_autotune_key(input_nhwc.device, num_sms, N, Ci, Co, R, S, out_h, out_w, stride_h, stride_w,
                                         pad_h, pad_w)
    cached = _wgrad_autotune_cache.get(cache_key)
    if cached is not None:
        return dict(cached)

    best_ms = float("inf")
    best_kernel_meta = None
    for config in conv2d_wgrad_get_configs():
        kernel_meta = config.all_kwargs()
        ms = _benchmark_wgrad_config(
            input_nhwc,
            grad_output_nhwc,
            N=N,
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
            K_GEMM=K_GEMM,
            num_sms=num_sms,
            kernel_meta=kernel_meta,
        )
        if ms < best_ms:
            best_ms = ms
            best_kernel_meta = dict(kernel_meta)

    if best_kernel_meta is None:
        raise RuntimeError("Failed to autotune conv2d_wgrad: no valid kernel configurations.")

    _wgrad_autotune_cache[cache_key] = dict(best_kernel_meta)
    return dict(best_kernel_meta)


@triton.jit
def reduce_split_k_partials_kernel(
    partial_ptr,
    grad_weight_ptr,
    partial_stride_0,
    grad_weight_stride_0,
    Co,
    K_GEMM,
    ACTIVE_SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < Co) & (offs_n[None, :] < K_GEMM)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for split_k_idx in range(ACTIVE_SPLIT_K):
        partial_rows = split_k_idx * Co + offs_m
        partial_offsets = partial_rows[:, None] * partial_stride_0 + offs_n[None, :]
        acc += tl.load(partial_ptr + partial_offsets, mask=mask, other=0.0)

    grad_weight_offsets = offs_m[:, None] * grad_weight_stride_0 + offs_n[None, :]
    tl.store(grad_weight_ptr + grad_weight_offsets, acc, mask=mask)


def _reduce_wgrad_split_k_partials(partials, grad_weight_flat, Co, K_GEMM, active_split_k):
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(Co, BLOCK_M), triton.cdiv(K_GEMM, BLOCK_N))
    reduce_split_k_partials_kernel[grid](
        partials,
        grad_weight_flat,
        partials.stride(0),
        grad_weight_flat.stride(0),
        Co,
        K_GEMM,
        ACTIVE_SPLIT_K=active_split_k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )


def _launch_wgrad(
    kernel,
    grid,
    *,
    in_desc,
    grad_out_desc,
    grad_weight,
    N,
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
    K_GEMM,
    kernel_meta=None,
):
    if kernel_meta is None:
        kernel_meta = {}

    kernel[grid](
        in_desc,
        grad_out_desc,
        grad_weight,
        N,
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
        K_GEMM,
        grad_weight.stride(0),
        **kernel_meta,
    )


def _finalize_wgrad_output(grad_weight_flat, Co, R, S, Ci, Ci_orig):
    result = grad_weight_flat.reshape(Co, R, S, Ci).to(TORCH_GEMM_DTYPE)
    if Ci != Ci_orig:
        result = result[:, :, :, :Ci_orig].contiguous()
    return result


def conv2d_wgrad(input_nhwc, grad_output_nhwc, R, S, stride=1, padding=0):
    """Production wgrad entrypoint.

    Selects the best kernel configuration with host-side autotuning, then runs
    deterministic two-pass split-K when reduction is needed.
    """
    (input_nhwc, grad_output_nhwc, Ci_orig, N, Ci, Co,
     out_h, out_w, stride_h, stride_w, pad_h, pad_w, K_GEMM) = \
        _prepare_wgrad_problem(input_nhwc, grad_output_nhwc, R, S, stride, padding)
    grad_weight_flat = _allocate_wgrad_output(input_nhwc.device, Co, K_GEMM)

    num_sms = torch.cuda.get_device_properties(input_nhwc.device).multi_processor_count

    kernel_meta = _select_wgrad_kernel_meta(
        input_nhwc,
        grad_output_nhwc,
        N=N,
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
        K_GEMM=K_GEMM,
        num_sms=num_sms,
    )
    run = _make_wgrad_runner(
        input_nhwc,
        grad_output_nhwc,
        grad_weight_flat,
        N=N,
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
        K_GEMM=K_GEMM,
        num_sms=num_sms,
        kernel_meta=kernel_meta,
    )
    run()

    return _finalize_wgrad_output(grad_weight_flat, Co, R, S, Ci, Ci_orig)


def _make_wgrad_fixed_kernel_meta(SPLIT_K, num_buffers, num_warps):
    # Keep the fixed path on a tile shape that is also covered by autotune configs.
    return {
        "BLOCK_M": 128,
        "BLOCK_N": 256,
        "BLOCK_K": 64,
        "SPLIT_K": SPLIT_K,
        "num_buffers": num_buffers,
        "num_acc_buffers": 2,
        "num_warps": num_warps,
    }


def conv2d_wgrad_fixed(input_nhwc, grad_output_nhwc, R, S, stride=1, padding=0, num_buffers=2, num_warps=4, SPLIT_K=1):
    """Fixed-config wgrad entrypoint used for CI and debugging.

    Runs the kernel with a fixed supported tile shape instead of autotuning,
    while still using deterministic two-pass split-K when reduction is needed.
    """
    (input_nhwc, grad_output_nhwc, Ci_orig, N, Ci, Co,
     out_h, out_w, stride_h, stride_w, pad_h, pad_w, K_GEMM) = \
        _prepare_wgrad_problem(input_nhwc, grad_output_nhwc, R, S, stride, padding)
    grad_weight_flat = _allocate_wgrad_output(input_nhwc.device, Co, K_GEMM)

    num_sms = torch.cuda.get_device_properties(input_nhwc.device).multi_processor_count
    kernel_meta = _make_wgrad_fixed_kernel_meta(SPLIT_K, num_buffers, num_warps)
    run = _make_wgrad_runner(
        input_nhwc,
        grad_output_nhwc,
        grad_weight_flat,
        N=N,
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
        K_GEMM=K_GEMM,
        num_sms=num_sms,
        kernel_meta=kernel_meta,
    )
    run()

    return _finalize_wgrad_output(grad_weight_flat, Co, R, S, Ci, Ci_orig)


# ===-----------------------------------------------------------------------===#
# Unit Tests
# ===-----------------------------------------------------------------------===#


def _assert_wgrad_correct(wgrad_fn, N, Ci, H, W, Co, R, S, stride, padding, **kwargs):
    """Run wgrad_fn and compare against PyTorch autograd reference."""
    torch.manual_seed(0)
    stride_h, stride_w = normalize_2d(stride, "stride")
    pad_h, pad_w = normalize_2d(padding, "padding")

    x_nchw = torch.randn((N, Ci, H, W), device="cuda", dtype=TORCH_GEMM_DTYPE)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()

    out_h = (H + 2 * pad_h - R) // stride_h + 1
    out_w = (W + 2 * pad_w - S) // stride_w + 1

    grad_out_nchw = torch.randn((N, Co, out_h, out_w), device="cuda", dtype=TORCH_GEMM_DTYPE)
    grad_out_nhwc = grad_out_nchw.permute(0, 2, 3, 1).contiguous()

    w_nchw = torch.randn((Co, Ci, R, S), device="cuda", dtype=TORCH_GEMM_DTYPE)
    w_ref = w_nchw.detach().requires_grad_(True)
    out_ref = torch.nn.functional.conv2d(x_nchw, w_ref, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    out_ref.backward(grad_out_nchw)
    ref_grad_w_nhwc = w_ref.grad.permute(0, 2, 3, 1).contiguous()

    triton_grad_w = wgrad_fn(x_nhwc, grad_out_nhwc, R, S, stride=stride, padding=padding, **kwargs)
    torch.testing.assert_close(triton_grad_w, ref_grad_w_nhwc, atol=1, rtol=0.01)


@pytest.mark.parametrize("wgrad_fn,N,Ci,H,W,Co,R,S,stride,padding", [
    *[(conv2d_wgrad_fixed, N, Ci, H, W, Co, R, S, stride, padding)
      for N in (1, 128)
      for H, W in ((64, 64), (64, 32))
      for Ci, Co in ((128, 128), (384, 384), (128, 384))
      for R, S in ((1, 1), (2, 2), (3, 3), (1, 3))
      for stride in (1, 2, 3)
      for padding in (0, 1)], (conv2d_wgrad_fixed, 16, 5, 32, 32, 96, 3, 3, 1, 1),  # padded channels
    (conv2d_wgrad_fixed, 16, 96, 1, 8, 128, 1, 2, (1, 2), 0),  # asymmetric stride
    (conv2d_wgrad_fixed, 16, 512, 2, 2, 768, 2, 2, (2, 2), 0),  # small spatial
])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU (SM 10.x)")
def test_op(wgrad_fn, N, Ci, H, W, Co, R, S, stride, padding):
    _assert_wgrad_correct(wgrad_fn, N, Ci, H, W, Co, R, S, stride, padding)


# ===-----------------------------------------------------------------------===#
# Benchmarking
# ===-----------------------------------------------------------------------===#

BATCH = [128]
CHANNELS = [(384, 384)]
SPATIAL = [(64, 64)]
FILTER = [(3, 3)]
STRIDE = [1]
PADDING = [1]


def _make_bench_inputs(N, H, W, Ci, Co, R, S, stride_val, pad_val):
    torch.manual_seed(0)
    out_h = (H + 2 * pad_val - R) // stride_val + 1
    out_w = (W + 2 * pad_val - S) // stride_val + 1
    x_nchw = torch.randn((N, Ci, H, W), device="cuda", dtype=TORCH_GEMM_DTYPE)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()
    grad_out_nchw = torch.randn((N, Co, out_h, out_w), device="cuda", dtype=TORCH_GEMM_DTYPE)
    grad_out_nhwc = grad_out_nchw.permute(0, 2, 3, 1).contiguous()
    return x_nchw, x_nhwc, grad_out_nchw, grad_out_nhwc


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
            plot_name=f"Wgrad N={N} Ci={Ci} Co={Co} H={H} W={W} R={R} S={S} stride={stride_val} pad={pad_val}",
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
    x_nchw, x_nhwc, grad_out_nchw, grad_out_nhwc = \
        _make_bench_inputs(N, H, W, Ci, Co, R, S, stride_val, pad_val)

    if provider == "gluon":
        fn = lambda: conv2d_wgrad(x_nhwc, grad_out_nhwc, R, S, stride=stride_val, padding=pad_val)
    elif provider == "torch":
        w_nchw = torch.randn((Co, Ci, R, S), device="cuda", dtype=TORCH_GEMM_DTYPE)
        fn = lambda: torch.ops.aten.convolution_backward(
            grad_out_nchw,
            x_nchw,
            w_nchw,
            bias_sizes=None,
            stride=[stride_val, stride_val],
            padding=[pad_val, pad_val],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
            output_mask=[False, True, False],
        )
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
