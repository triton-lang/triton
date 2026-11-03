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
ensure_tma_compatible_strides = _conv_common.ensure_tma_compatible_strides
init_mbarrier_ring = _conv_common.init_mbarrier_ring
invalidate_mbarrier_ring = _conv_common.invalidate_mbarrier_ring
is_blackwell = _conv_common.is_blackwell
maybe_pad_ci_for_tma = _conv_common.maybe_pad_channel_dims_for_tma
normalize_2d = _conv_common.normalize_2d

# ===-----------------------------------------------------------------------===#
# Dgrad GEMM mapping
# ===-----------------------------------------------------------------------===#
#
# Dgrad as forward conv on grad_Y with rotated weight:
#   grad_X[M, Ci] = im2col(grad_Y)[M, R_eff*S_eff*Co]  @  W_rot[R_eff*S_eff*Co, Ci]^T
#
# For stride > 1, the host decomposes dgrad into up to stride_h * stride_w
# subproblems. Each subproblem fixes (sub_a, sub_b, r0, s0, R_eff, S_eff,
# offset_a, offset_b), builds a grad_Y im2col descriptor, and launches the
# persistent kernel once.

# Per subproblem launch, the logical tile space is:
#   cdiv(M_GEMM, BLOCK_M) * cdiv(Ci, BLOCK_N) * active_split_k
# where M_GEMM = N * H_sub * W_sub and
#   total_k_iters = R_eff * S_eff * cdiv(Co, BLOCK_K).
#
# The epilogue scatters results back to the full output tensor at
#   h = sub_a + c_out_y * stride_h
#   w = sub_b + c_out_x * stride_w
#
# If active_split_k > 1, the kernel stores fp32 partials to a workspace and a
# separate reduction kernel accumulates them into the final output. The launch
# uses a persistent scheduler and runs only `min(num_sms, logical_tiles)` CTAs.


@gluon.aggregate
class DgradConfig:
    N: gl.tensor
    Co: gl.tensor
    Ci: gl.tensor
    R_eff: gl.tensor
    S_eff: gl.tensor
    H_sub: gl.tensor
    W_sub: gl.tensor
    pad_h: gl.tensor
    pad_w: gl.tensor
    output_stride_n: gl.tensor
    output_stride_h: gl.tensor
    output_stride_w: gl.tensor
    M_GEMM: gl.tensor
    sub_a: gl.tensor
    sub_b: gl.tensor
    conv_stride_h: gl.tensor
    conv_stride_w: gl.tensor
    r0: gl.tensor
    s0: gl.tensor
    S_orig: gl.tensor
    H_full: gl.tensor
    W_full: gl.tensor

    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    GROUP_SIZE_M: gl.constexpr
    SPLIT_K: gl.constexpr
    num_buffers: gl.constexpr
    num_warps: gl.constexpr

    @gluon.jit
    def get_num_output_tiles(self):
        return gl.cdiv(self.M_GEMM, self.BLOCK_M) * gl.cdiv(self.Ci, self.BLOCK_N)

    @gluon.jit
    def get_num_k_iterations(self):
        return self.R_eff * self.S_eff * gl.cdiv(self.Co, self.BLOCK_K)

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

        num_pid_m = gl.cdiv(self.M_GEMM, self.BLOCK_M)
        num_pid_n = gl.cdiv(self.Ci, self.BLOCK_N)

        num_pid_in_group = self.GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * self.GROUP_SIZE_M
        group_size_m = gl.minimum(num_pid_m - first_pid_m, self.GROUP_SIZE_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

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

        return DgradProgram(self, pid_m, pid_n, split_k_idx, k_start, k_iters_this_split)


@gluon.aggregate
class DgradProgram:
    config: DgradConfig
    pid_m: gl.tensor
    pid_n: gl.tensor
    split_k_idx: gl.tensor
    k_start: gl.tensor
    k_iters_this_split: gl.tensor

    @gluon.jit
    def get_m_offsets(self):
        offs_m = self.pid_m * self.config.BLOCK_M
        config = self.config
        out_x = offs_m % config.W_sub
        out_y = (offs_m // config.W_sub) % config.H_sub
        batch_id = (offs_m // config.W_sub) // config.H_sub
        return batch_id, out_y, out_x

    @gluon.jit
    def get_ci_offset(self):
        return self.pid_n * self.config.BLOCK_N

    @gluon.jit
    def get_k_iteration(self, local_k):
        k_iter = self.k_start + local_k
        num_rs = self.config.R_eff * self.config.S_eff
        iter_co = k_iter // num_rs
        remain_rs = k_iter % num_rs
        iter_s = remain_rs % self.config.S_eff
        iter_r = remain_rs // self.config.S_eff
        return iter_co, iter_r, iter_s

    @gluon.jit
    def get_weight_k_offset(self, local_k):
        iter_co, iter_r, iter_s = self.get_k_iteration(local_k)
        actual_r = self.config.r0 + iter_r * self.config.conv_stride_h
        actual_s = self.config.s0 + iter_s * self.config.conv_stride_w
        k_offset = (actual_r * self.config.S_orig + actual_s) * self.config.Co + iter_co * self.config.BLOCK_K
        return iter_co, iter_r, iter_s, k_offset


@gluon.aggregate
class PartitionArgs:
    config: DgradConfig
    grad_y_desc: tma.tensor_descriptor_im2col
    weight_desc: tma.tensor_descriptor
    output_ptr: gl.tensor
    store_split_k_partials: gl.constexpr
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
    """Load partition: iterate over the persistent dgrad work items assigned to this CTA."""
    config = p.config
    BLOCK_K: gl.constexpr = config.BLOCK_K

    empty_bars = p.load_empty_bars
    ready_bars = p.load_ready_bars
    state = Counter.create(1, empty_bars.shape[0])
    scheduler = PersistentTileScheduler.initialize(config.get_num_tiles())

    for idx in range(scheduler.get_num_tiles()):
        prog = config.get_program(scheduler.get_tile_id(idx))
        batch_id, out_y, out_x = prog.get_m_offsets()
        ci_offset = prog.get_ci_offset()

        for local_k in range(prog.k_iters_this_split):
            iter_co, iter_r, iter_s, weight_k_offset = prog.get_weight_k_offset(local_k)
            ready_bar = ready_bars.index(state.index)
            mbarrier.wait(empty_bars.index(state.index), state.phase)
            mbarrier.expect(ready_bar, p.grad_y_desc.block_type.nbytes + p.weight_desc.block_type.nbytes)

            tma.async_load_im2col(
                p.grad_y_desc,
                [
                    batch_id,
                    out_y - config.pad_h,
                    out_x - config.pad_w,
                    iter_co * BLOCK_K,
                ],
                [iter_r.to(tl.int16), iter_s.to(tl.int16)],
                ready_bar,
                p.a_bufs.index(state.index),
            )

            tma.async_load(
                p.weight_desc,
                [ci_offset, weight_k_offset],
                ready_bar,
                p.b_bufs.index(state.index),
            )
            state = state.next()


@gluon.jit
def mma_partition(p):
    """MMA partition: accumulate all split-K dgrad work items assigned to this CTA."""
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
                p.a_bufs.index(load_state.index),
                p.b_bufs.index(load_state.index).permute((1, 0)),
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
    """Epilogue partition: store the persistent dgrad work items assigned to this CTA."""
    config = p.config
    BLOCK_M: gl.constexpr = config.BLOCK_M
    BLOCK_N: gl.constexpr = config.BLOCK_N
    M_GEMM = config.M_GEMM
    N_GEMM = config.Ci
    acc_state = Counter.create(0, p.acc_empty_bars.shape[0])
    scheduler = PersistentTileScheduler.initialize(config.get_num_tiles())

    for idx in range(scheduler.get_num_tiles()):
        prog = config.get_program(scheduler.get_tile_id(idx))

        mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
        acc = p.acc_bufs.index(acc_state.index).load()
        mbarrier.arrive(p.acc_empty_bars.index(acc_state.index), count=1)
        acc_state = acc_state.next()

        offs_m = prog.pid_m * BLOCK_M + gl.arange(0, BLOCK_M)
        offs_n = prog.get_ci_offset() + gl.arange(0, BLOCK_N)

        c_out_x = offs_m % config.W_sub
        c_out_y = (offs_m // config.W_sub) % config.H_sub
        c_batch = (offs_m // config.W_sub) // config.H_sub

        h = config.sub_a + c_out_y * config.conv_stride_h
        w = config.sub_b + c_out_x * config.conv_stride_w

        c_offsets = (c_batch[:, None] * config.output_stride_n + h[:, None] * config.output_stride_h +
                     w[:, None] * config.output_stride_w + offs_n[None, :])
        c_mask = ((offs_m[:, None] < M_GEMM) & (offs_n[None, :] < N_GEMM) & (h[:, None] < config.H_full) &
                  (w[:, None] < config.W_full))

        result = gl.convert_layout(acc, gl.CoalescedLayout())
        if p.store_split_k_partials:
            split_batch = prog.split_k_idx * config.N + c_batch
            split_offsets = (split_batch[:, None] * config.output_stride_n + h[:, None] * config.output_stride_h +
                             w[:, None] * config.output_stride_w + offs_n[None, :])
            gl.store(p.output_ptr + split_offsets, result, mask=c_mask)
        else:
            gl.store(p.output_ptr + c_offsets, result.to(GL_GEMM_DTYPE), mask=c_mask)

    invalidate_mbarrier_ring(p.load_empty_bars)
    invalidate_mbarrier_ring(p.load_ready_bars)
    invalidate_mbarrier_ring(p.acc_empty_bars)
    invalidate_mbarrier_ring(p.acc_ready_bars)


# ===-----------------------------------------------------------------------===#
# Kernel Entry Point
# ===-----------------------------------------------------------------------===#


@gluon.jit(do_not_specialize=[
    "N",
    "S_orig",
    "H_sub",
    "W_sub",
    "H_full",
    "W_full",
    "conv_stride_h",
    "conv_stride_w",
    "sub_a",
    "sub_b",
    "r0",
    "s0",
    "R_eff",
    "S_eff",
    "pad_h",
    "pad_w",
])
def conv2d_dgrad_kernel(
    grad_y_desc,
    weight_desc,
    output,
    N,
    Co,
    Ci,
    S_orig,
    H_sub,
    W_sub,
    H_full,
    W_full,
    output_stride_n,
    output_stride_h,
    output_stride_w,
    conv_stride_h,
    conv_stride_w,
    sub_a,
    sub_b,
    r0,
    s0,
    R_eff,
    S_eff,
    pad_h,
    pad_w,
    STORE_SPLIT_K_PARTIALS: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    SPLIT_K: gl.constexpr,
    num_buffers: gl.constexpr,
    num_acc_buffers: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Warp-specialized dgrad kernel.

    Logical tile space = cdiv(M_sub, BLOCK_M) * cdiv(Ci, BLOCK_N), optionally
    multiplied by split-K. Sub-problem parameters (sub_a/b, r0/s0, R_eff/S_eff,
    pad) are per-launch constants.
    """
    M_GEMM = N * H_sub * W_sub
    config = DgradConfig(
        N,
        Co,
        Ci,
        R_eff,
        S_eff,
        gl.to_tensor(H_sub),
        gl.to_tensor(W_sub),
        pad_h,
        pad_w,
        gl.to_tensor(output_stride_n),
        gl.to_tensor(output_stride_h),
        gl.to_tensor(output_stride_w),
        M_GEMM,
        sub_a,
        sub_b,
        gl.to_tensor(conv_stride_h),
        gl.to_tensor(conv_stride_w),
        r0,
        s0,
        S_orig,
        H_full,
        W_full,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_SIZE_M,
        SPLIT_K,
        num_buffers,
        num_warps,
    )

    a_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], GL_GEMM_DTYPE)
    b_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, BLOCK_K], GL_GEMM_DTYPE)

    a_bufs = gl.allocate_shared_memory(GL_GEMM_DTYPE, [num_buffers, BLOCK_M, BLOCK_K], a_smem_layout)
    b_bufs = gl.allocate_shared_memory(GL_GEMM_DTYPE, [num_buffers, BLOCK_N, BLOCK_K], b_smem_layout)

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
        grad_y_desc,
        weight_desc,
        output,
        STORE_SPLIT_K_PARTIALS,
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


def conv2d_dgrad_get_configs():
    return [
        triton.Config(
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_SIZE_M": group_size_m,
                "SPLIT_K": split_k,
                "num_buffers": num_buffers,
                "num_acc_buffers": num_acc_buffers,
            },
            num_warps=num_warps,
        )
        for block_m in (64, 128)
        for block_n in (64, 128, 256)
        for block_k in (64, )
        for group_size_m in (4, )
        for split_k in (1, 2, 4, 8)
        for num_buffers in (3, 4, 5)
        for num_acc_buffers in (2, )
        for num_warps in (4, )
    ]


# ===-----------------------------------------------------------------------===#
# Host-Side Entry Point
# ===-----------------------------------------------------------------------===#


def _make_dgrad_subproblem_specs(R, S, stride_h, stride_w, pad_h, pad_w):
    p_h_prime = R - 1 - pad_h
    p_w_prime = S - 1 - pad_w

    subproblem_specs = []
    for a in range(stride_h):
        for b in range(stride_w):
            r0 = ((p_h_prime - a) % stride_h + stride_h) % stride_h
            s0 = ((p_w_prime - b) % stride_w + stride_w) % stride_w
            R_eff = (R - r0 + stride_h - 1) // stride_h
            S_eff = (S - s0 + stride_w - 1) // stride_w
            if R_eff <= 0 or S_eff <= 0:
                continue
            offset_a = (a + r0 - p_h_prime) // stride_h
            offset_b = (b + s0 - p_w_prime) // stride_w
            subproblem_specs.append((a, b, r0, s0, R_eff, S_eff, offset_a, offset_b))

    return subproblem_specs


def _prepare_dgrad_inputs(grad_output_nhwc, weight_nhwc, H_in, W_in, stride, padding):
    """Validate inputs, pad channels, and compute sub-problem decomposition."""
    if grad_output_nhwc.dtype != TORCH_GEMM_DTYPE or weight_nhwc.dtype != TORCH_GEMM_DTYPE:
        raise ValueError(
            f"conv2d_dgrad expects bfloat16 grad-output and weight tensors, got {grad_output_nhwc.dtype} and {weight_nhwc.dtype}"
        )

    stride_h, stride_w = normalize_2d(stride, "stride")
    pad_h, pad_w = normalize_2d(padding, "padding")
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError(f"stride must be positive, got {(stride_h, stride_w)}")
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"padding must be non-negative, got {(pad_h, pad_w)}")

    N, out_h, out_w, Co = grad_output_nhwc.shape
    Co_w, R, S, Ci = weight_nhwc.shape
    if Co != Co_w:
        raise ValueError(f"Channel dimension mismatch: grad-output has {Co}, weight has {Co_w}")

    expected_out_h = (H_in + 2 * pad_h - R) // stride_h + 1
    expected_out_w = (W_in + 2 * pad_w - S) // stride_w + 1
    if out_h != expected_out_h or out_w != expected_out_w:
        raise ValueError("Grad-output shape mismatch: expected "
                         f"({N}, {expected_out_h}, {expected_out_w}, {Co}) from input/filter geometry, got "
                         f"({N}, {out_h}, {out_w}, {Co}).")
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Invalid convolution geometry for dgrad")

    grad_output_nhwc = maybe_pad_ci_for_tma(grad_output_nhwc)
    grad_output_nhwc = ensure_tma_compatible_strides(grad_output_nhwc)
    Co_padded = grad_output_nhwc.shape[-1]
    if Co_padded != Co:
        w_padded = weight_nhwc.new_zeros((Co_padded, R, S, Ci))
        w_padded[:Co] = weight_nhwc
        weight_nhwc = w_padded.contiguous()
        Co = Co_padded

    W_rot = weight_nhwc.flip(1, 2).permute(3, 1, 2, 0).contiguous()  # (Ci, R, S, Co)
    W_rot_flat = W_rot.reshape(Ci, R * S * Co)

    H_sub = (H_in + stride_h - 1) // stride_h
    W_sub = (W_in + stride_w - 1) // stride_w
    subproblem_specs = _make_dgrad_subproblem_specs(R, S, stride_h, stride_w, pad_h, pad_w)
    if not subproblem_specs:
        raise ValueError("No valid dgrad sub-problems were generated")

    return (
        grad_output_nhwc,
        W_rot_flat,
        N,
        Co,
        Ci,
        S,
        out_h,
        out_w,
        H_in,
        W_in,
        H_sub,
        W_sub,
        stride_h,
        stride_w,
        subproblem_specs,
    )


def _make_dgrad_weight_descriptor(W_rot_flat, weight_block_shape):
    weight_layout = gl.NVMMASharedLayout.get_default_for(weight_block_shape, GL_GEMM_DTYPE)
    weight_desc = TensorDescriptor.from_tensor(W_rot_flat, weight_block_shape, weight_layout)
    return weight_desc


def _make_dgrad_grad_y_descriptor(grad_output_nhwc, H_sub, W_sub, out_h, out_w, offset_a, offset_b, input_block_shape):
    lower_h = offset_a
    lower_w = offset_b
    upper_h = H_sub + offset_a - out_h
    upper_w = W_sub + offset_b - out_w

    input_layout = gl.NVMMASharedLayout.get_default_for(input_block_shape, GL_GEMM_DTYPE)
    return TensorDescriptorIm2Col(
        base=grad_output_nhwc,
        shape=list(grad_output_nhwc.shape),
        strides=list(grad_output_nhwc.stride()),
        block_shape=input_block_shape,
        layout=input_layout,
        padding="zero",
        element_strides=[1, 1, 1, 1],
        pixel_box_lower_corner=[lower_h, lower_w],
        pixel_box_upper_corner=[upper_h, upper_w],
    )


def _make_grid(num_sms, M_GEMM, Ci, Co, R_eff, S_eff):

    def grid(meta):
        total_mn_tiles = triton.cdiv(M_GEMM, meta["BLOCK_M"]) * triton.cdiv(Ci, meta["BLOCK_N"])
        total_k_iters = R_eff * S_eff * triton.cdiv(Co, meta["BLOCK_K"])
        k_iters_per_split = triton.cdiv(total_k_iters, meta["SPLIT_K"])
        active_split_k = triton.cdiv(total_k_iters, k_iters_per_split)
        total_tiles = total_mn_tiles * active_split_k
        return (min(num_sms, total_tiles), )

    return grid


def _get_active_split_k(total_k_iters, SPLIT_K):
    k_iters_per_split = triton.cdiv(total_k_iters, SPLIT_K)
    return triton.cdiv(total_k_iters, k_iters_per_split)


def _get_dgrad_subproblem_active_split_k(Co, R_eff, S_eff, BLOCK_K, SPLIT_K):
    total_k_iters = R_eff * S_eff * triton.cdiv(Co, BLOCK_K)
    return _get_active_split_k(total_k_iters, SPLIT_K)


def _get_max_active_split_k(Co, subproblem_specs, BLOCK_K, SPLIT_K):
    return max(
        _get_dgrad_subproblem_active_split_k(Co, R_eff, S_eff, BLOCK_K, SPLIT_K)
        for _, _, _, _, R_eff, S_eff, _, _ in subproblem_specs)


def _get_safe_dgrad_max_active_split_k(Co, subproblem_specs, N, H_in, W_in, Ci, kernel_meta):
    """Return max active split-K across subproblems, or raise if workspace would be too large to index safely."""
    max_active_split_k = _get_max_active_split_k(Co, subproblem_specs, kernel_meta["BLOCK_K"], kernel_meta["SPLIT_K"])
    if max_active_split_k > 1:
        # Workspace shape: (active_split_k * N, H_in, W_in, Ci); indexed in kernels by batch/row offsets.
        # Very large workspaces can exceed the addressing range supported by the generated code.
        workspace_elems = max_active_split_k * N * H_in * W_in * Ci
        if workspace_elems > (2**31 - 1):
            raise ValueError("dgrad split-K workspace exceeds safe indexing range: "
                             f"active_split_k={max_active_split_k}, N={N}, H_in={H_in}, W_in={W_in}, Ci={Ci}")
    return max_active_split_k


def _allocate_dgrad_split_k_workspace(device, active_split_k, N, H_in, W_in, Ci):
    return torch.zeros((active_split_k * N, H_in, W_in, Ci), device=device, dtype=torch.float32)


_dgrad_autotune_cache = {}


def _make_dgrad_autotune_key(
    device,
    num_sms,
    N,
    Co,
    Ci,
    S,
    out_h,
    out_w,
    H_in,
    W_in,
    H_sub,
    W_sub,
    stride_h,
    stride_w,
    subproblem_specs,
):
    return (
        torch.cuda.get_device_capability(device),
        num_sms,
        N,
        Co,
        Ci,
        S,
        out_h,
        out_w,
        H_in,
        W_in,
        H_sub,
        W_sub,
        stride_h,
        stride_w,
        tuple(subproblem_specs),
    )


@triton.jit
def reduce_dgrad_split_k_partials_kernel(
    partial_ptr,
    output_ptr,
    partial_stride_n,
    partial_stride_h,
    partial_stride_w,
    output_stride_n,
    output_stride_h,
    output_stride_w,
    N,
    H,
    W,
    Ci,
    ACTIVE_SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    M = N * H * W
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    w_idx = offs_m % W
    h_idx = (offs_m // W) % H
    batch_idx = offs_m // (H * W)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < Ci)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for split_k_idx in range(ACTIVE_SPLIT_K):
        partial_batch = split_k_idx * N + batch_idx
        partial_offsets = (partial_batch[:, None] * partial_stride_n + h_idx[:, None] * partial_stride_h +
                           w_idx[:, None] * partial_stride_w + offs_n[None, :])
        acc += tl.load(partial_ptr + partial_offsets, mask=mask, other=0.0)

    output_offsets = (batch_idx[:, None] * output_stride_n + h_idx[:, None] * output_stride_h +
                      w_idx[:, None] * output_stride_w + offs_n[None, :])
    tl.store(output_ptr + output_offsets, acc, mask=mask)


def _reduce_dgrad_split_k_partials(partials, output, N, H, W, Ci, active_split_k):
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(N * H * W, BLOCK_M), triton.cdiv(Ci, BLOCK_N))
    reduce_dgrad_split_k_partials_kernel[grid](
        partials,
        output,
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        N,
        H,
        W,
        Ci,
        ACTIVE_SPLIT_K=active_split_k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )


def _launch_dgrad_subproblems(
    kernel,
    num_sms,
    *,
    grad_output_nhwc,
    W_rot_flat,
    output,
    N,
    Co,
    Ci,
    S,
    out_h,
    out_w,
    H_in,
    W_in,
    H_sub,
    W_sub,
    stride_h,
    stride_w,
    subproblem_specs,
    weight_block_shape,
    input_block_shape,
    kernel_meta=None,
):
    if kernel_meta is None:
        kernel_meta = {}
    kernel_meta.setdefault("STORE_SPLIT_K_PARTIALS", False)

    M_GEMM = N * H_sub * W_sub
    weight_desc = _make_dgrad_weight_descriptor(W_rot_flat, weight_block_shape)

    for a, b, r0_val, s0_val, R_eff_val, S_eff_val, offset_a, offset_b in subproblem_specs:
        grad_y_desc = _make_dgrad_grad_y_descriptor(
            grad_output_nhwc,
            H_sub,
            W_sub,
            out_h,
            out_w,
            offset_a,
            offset_b,
            input_block_shape,
        )

        kernel[_make_grid(num_sms, M_GEMM, Ci, Co, R_eff_val, S_eff_val)](
            grad_y_desc=grad_y_desc,
            weight_desc=weight_desc,
            output=output,
            N=N,
            Co=Co,
            Ci=Ci,
            S_orig=S,
            H_sub=H_sub,
            W_sub=W_sub,
            H_full=H_in,
            W_full=W_in,
            output_stride_n=output.stride(0),
            output_stride_h=output.stride(1),
            output_stride_w=output.stride(2),
            conv_stride_h=stride_h,
            conv_stride_w=stride_w,
            sub_a=a,
            sub_b=b,
            r0=r0_val,
            s0=s0_val,
            R_eff=R_eff_val,
            S_eff=S_eff_val,
            pad_h=-offset_a,
            pad_w=-offset_b,
            **kernel_meta,
        )


def _allocate_dgrad_output(device, N, H_in, W_in, Ci, split_k=1):
    if split_k == 1:
        return torch.empty((N, H_in, W_in, Ci), device=device, dtype=TORCH_GEMM_DTYPE)
    return torch.zeros((N, H_in, W_in, Ci), device=device, dtype=torch.float32)


def _finalize_dgrad_output(output):
    return output.to(TORCH_GEMM_DTYPE)


def _make_dgrad_runner(
    grad_output_nhwc,
    W_rot_flat,
    *,
    N,
    Co,
    Ci,
    S,
    out_h,
    out_w,
    H_in,
    W_in,
    H_sub,
    W_sub,
    stride_h,
    stride_w,
    subproblem_specs,
    num_sms,
    kernel_meta,
):
    max_active_split_k = _get_safe_dgrad_max_active_split_k(Co, subproblem_specs, N, H_in, W_in, Ci, kernel_meta)
    uses_split_k_workspace = max_active_split_k > 1
    output = _allocate_dgrad_output(
        grad_output_nhwc.device,
        N,
        H_in,
        W_in,
        Ci,
        split_k=max_active_split_k if uses_split_k_workspace else 1,
    )
    launch_output = output
    if uses_split_k_workspace:
        launch_output = _allocate_dgrad_split_k_workspace(grad_output_nhwc.device, max_active_split_k, N, H_in, W_in,
                                                          Ci)

    def run():
        _launch_dgrad_subproblems(
            conv2d_dgrad_kernel,
            num_sms,
            grad_output_nhwc=grad_output_nhwc,
            W_rot_flat=W_rot_flat,
            output=launch_output,
            N=N,
            Co=Co,
            Ci=Ci,
            S=S,
            out_h=out_h,
            out_w=out_w,
            H_in=H_in,
            W_in=W_in,
            H_sub=H_sub,
            W_sub=W_sub,
            stride_h=stride_h,
            stride_w=stride_w,
            subproblem_specs=subproblem_specs,
            weight_block_shape=[kernel_meta["BLOCK_N"], kernel_meta["BLOCK_K"]],
            input_block_shape=[kernel_meta["BLOCK_M"], kernel_meta["BLOCK_K"]],
            kernel_meta={
                **kernel_meta,
                "STORE_SPLIT_K_PARTIALS": uses_split_k_workspace,
            },
        )
        if uses_split_k_workspace:
            _reduce_dgrad_split_k_partials(launch_output, output, N, H_in, W_in, Ci, max_active_split_k)

    return run, output


def _benchmark_dgrad_config(
    grad_output_nhwc,
    W_rot_flat,
    *,
    N,
    Co,
    Ci,
    S,
    out_h,
    out_w,
    H_in,
    W_in,
    H_sub,
    W_sub,
    stride_h,
    stride_w,
    subproblem_specs,
    num_sms,
    kernel_meta,
):
    try:
        run, _ = _make_dgrad_runner(
            grad_output_nhwc,
            W_rot_flat,
            N=N,
            Co=Co,
            Ci=Ci,
            S=S,
            out_h=out_h,
            out_w=out_w,
            H_in=H_in,
            W_in=W_in,
            H_sub=H_sub,
            W_sub=W_sub,
            stride_h=stride_h,
            stride_w=stride_w,
            subproblem_specs=subproblem_specs,
            num_sms=num_sms,
            kernel_meta=kernel_meta,
        )
        run()
        torch.cuda.synchronize()
        return triton.testing.do_bench(run)
    except Exception:
        return float("inf")


def _select_dgrad_kernel_meta(
    grad_output_nhwc,
    W_rot_flat,
    *,
    N,
    Co,
    Ci,
    S,
    out_h,
    out_w,
    H_in,
    W_in,
    H_sub,
    W_sub,
    stride_h,
    stride_w,
    subproblem_specs,
    num_sms,
):
    cache_key = _make_dgrad_autotune_key(
        grad_output_nhwc.device,
        num_sms,
        N,
        Co,
        Ci,
        S,
        out_h,
        out_w,
        H_in,
        W_in,
        H_sub,
        W_sub,
        stride_h,
        stride_w,
        subproblem_specs,
    )
    cached = _dgrad_autotune_cache.get(cache_key)
    if cached is not None:
        return dict(cached)

    best_ms = float("inf")
    best_kernel_meta = None
    for config in conv2d_dgrad_get_configs():
        kernel_meta = config.all_kwargs()
        ms = _benchmark_dgrad_config(
            grad_output_nhwc,
            W_rot_flat,
            N=N,
            Co=Co,
            Ci=Ci,
            S=S,
            out_h=out_h,
            out_w=out_w,
            H_in=H_in,
            W_in=W_in,
            H_sub=H_sub,
            W_sub=W_sub,
            stride_h=stride_h,
            stride_w=stride_w,
            subproblem_specs=subproblem_specs,
            num_sms=num_sms,
            kernel_meta=kernel_meta,
        )
        if ms < best_ms:
            best_ms = ms
            best_kernel_meta = dict(kernel_meta)

    if best_kernel_meta is None:
        raise RuntimeError("Failed to autotune conv2d_dgrad: no valid kernel configurations.")

    _dgrad_autotune_cache[cache_key] = dict(best_kernel_meta)
    return dict(best_kernel_meta)


def conv2d_dgrad(grad_output_nhwc, weight_nhwc, H_in, W_in, stride=1, padding=0):
    """Production dgrad entrypoint.

    Selects the best kernel configuration with host-side autotuning, then runs
    deterministic two-pass split-K when reduction is needed.
    """
    (grad_output_nhwc, W_rot_flat, N, Co, Ci, S,
     out_h, out_w, H_in, W_in, H_sub, W_sub,
     stride_h, stride_w, subproblem_specs) = \
        _prepare_dgrad_inputs(grad_output_nhwc, weight_nhwc, H_in, W_in, stride, padding)

    device = grad_output_nhwc.device
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    kernel_meta = _select_dgrad_kernel_meta(
        grad_output_nhwc,
        W_rot_flat,
        N=N,
        Co=Co,
        Ci=Ci,
        S=S,
        out_h=out_h,
        out_w=out_w,
        H_in=H_in,
        W_in=W_in,
        H_sub=H_sub,
        W_sub=W_sub,
        stride_h=stride_h,
        stride_w=stride_w,
        subproblem_specs=subproblem_specs,
        num_sms=num_sms,
    )
    run, output = _make_dgrad_runner(
        grad_output_nhwc,
        W_rot_flat,
        N=N,
        Co=Co,
        Ci=Ci,
        S=S,
        out_h=out_h,
        out_w=out_w,
        H_in=H_in,
        W_in=W_in,
        H_sub=H_sub,
        W_sub=W_sub,
        stride_h=stride_h,
        stride_w=stride_w,
        subproblem_specs=subproblem_specs,
        num_sms=num_sms,
        kernel_meta=kernel_meta,
    )
    run()

    return _finalize_dgrad_output(output)


def _make_dgrad_fixed_kernel_meta(SPLIT_K, num_buffers, num_warps):
    # Keep the fixed path on a tile shape that is also covered by autotune configs.
    return {
        "BLOCK_M": 128,
        "BLOCK_N": 256,
        "BLOCK_K": 64,
        "GROUP_SIZE_M": 4,
        "SPLIT_K": SPLIT_K,
        "num_buffers": num_buffers,
        "num_acc_buffers": 2,
        "num_warps": num_warps,
    }


def conv2d_dgrad_fixed(grad_output_nhwc, weight_nhwc, H_in, W_in, stride=1, padding=0, num_buffers=2, num_warps=4,
                       SPLIT_K=1):
    """Fixed-config dgrad entrypoint used for CI and debugging.

    Runs the kernel with a fixed supported tile shape instead of autotuning,
    while still using deterministic two-pass split-K when reduction is needed.
    """
    (grad_output_nhwc, W_rot_flat, N, Co, Ci, S,
     out_h, out_w, H_in, W_in, H_sub, W_sub,
     stride_h, stride_w, subproblem_specs) = \
        _prepare_dgrad_inputs(grad_output_nhwc, weight_nhwc, H_in, W_in, stride, padding)

    device = grad_output_nhwc.device
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    kernel_meta = _make_dgrad_fixed_kernel_meta(SPLIT_K, num_buffers, num_warps)
    run, output = _make_dgrad_runner(
        grad_output_nhwc,
        W_rot_flat,
        N=N,
        Co=Co,
        Ci=Ci,
        S=S,
        out_h=out_h,
        out_w=out_w,
        H_in=H_in,
        W_in=W_in,
        H_sub=H_sub,
        W_sub=W_sub,
        stride_h=stride_h,
        stride_w=stride_w,
        subproblem_specs=subproblem_specs,
        num_sms=num_sms,
        kernel_meta=kernel_meta,
    )
    run()

    return _finalize_dgrad_output(output)


# ===-----------------------------------------------------------------------===#
# Unit Tests
# ===-----------------------------------------------------------------------===#


def _assert_dgrad_correct(dgrad_fn, N, Ci, H, W, Co, R, S, stride, padding, **kwargs):
    """Run dgrad and compare against PyTorch autograd reference."""
    torch.manual_seed(0)
    stride_h, stride_w = normalize_2d(stride, "stride")
    pad_h, pad_w = normalize_2d(padding, "padding")

    out_h = (H + 2 * pad_h - R) // stride_h + 1
    out_w = (W + 2 * pad_w - S) // stride_w + 1

    grad_out_nchw = torch.randn((N, Co, out_h, out_w), device="cuda", dtype=TORCH_GEMM_DTYPE)
    grad_out_nhwc = grad_out_nchw.permute(0, 2, 3, 1).contiguous()

    w_nchw = torch.randn((Co, Ci, R, S), device="cuda", dtype=TORCH_GEMM_DTYPE)
    w_nhwc = w_nchw.permute(0, 2, 3, 1).contiguous()

    triton_dgrad = dgrad_fn(grad_out_nhwc, w_nhwc, H, W, stride=stride, padding=padding, **kwargs)

    ref_dgrad = torch.ops.aten.convolution_backward(
        grad_out_nchw,
        torch.randn((N, Ci, H, W), device="cuda", dtype=TORCH_GEMM_DTYPE),
        w_nchw,
        bias_sizes=None,
        stride=[stride_h, stride_w],
        padding=[pad_h, pad_w],
        dilation=[1, 1],
        transposed=False,
        output_padding=[0, 0],
        groups=1,
        output_mask=[True, False, False],
    )[0]
    ref_dgrad_nhwc = ref_dgrad.permute(0, 2, 3, 1).contiguous()

    torch.testing.assert_close(triton_dgrad, ref_dgrad_nhwc, atol=1e-2, rtol=1e-2)


DGRAD_SHAPE_PARAMS = [
    *[(N, Ci, 64, 64, Co, R, S, stride, padding)
      for N in (1, 128)
      for Ci, Co in ((384, 384), (128, 128))
      for R, S in ((2, 2), (3, 3))
      for stride in (1, 2)
      for padding in (0, 1)],
    (16, 5, 32, 32, 96, 3, 3, 1, 1),
    (16, 512, 2, 2, 768, 2, 2, 2, 0),
    (16, 96, 1, 8, 128, 1, 2, (1, 2), (0, 0)),
    (16, 128, 1, 4, 192, 1, 2, (1, 2), (0, 0)),
    (16, 160, 1, 2, 256, 1, 2, (1, 2), (0, 0)),
]


@pytest.mark.parametrize(
    "dgrad_fn,N,Ci,H,W,Co,R,S,stride,padding",
    [(conv2d_dgrad_fixed, *shape) for shape in DGRAD_SHAPE_PARAMS],
)
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU (SM 10.x)")
def test_op(dgrad_fn, N, Ci, H, W, Co, R, S, stride, padding):
    _assert_dgrad_correct(dgrad_fn, N, Ci, H, W, Co, R, S, stride, padding)


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
    grad_out_nchw = torch.randn((N, Co, out_h, out_w), device="cuda", dtype=TORCH_GEMM_DTYPE)
    grad_out_nhwc = grad_out_nchw.permute(0, 2, 3, 1).contiguous()
    w_nchw = torch.randn((Co, Ci, R, S), device="cuda", dtype=TORCH_GEMM_DTYPE)
    w_nhwc = w_nchw.permute(0, 2, 3, 1).contiguous()
    return x_nchw, grad_out_nchw, grad_out_nhwc, w_nchw, w_nhwc


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
            plot_name=f"Dgrad N={N} Ci={Ci} Co={Co} H={H} W={W} R={R} S={S} stride={stride_val} pad={pad_val}",
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
    x_nchw, grad_out_nchw, grad_out_nhwc, w_nchw, w_nhwc = \
        _make_bench_inputs(N, H, W, Ci, Co, R, S, stride_val, pad_val)

    if provider == "gluon":
        fn = lambda: conv2d_dgrad(grad_out_nhwc, w_nhwc, H, W, stride=stride_val, padding=pad_val)
    elif provider == "torch":
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
            output_mask=[True, False, False],
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
