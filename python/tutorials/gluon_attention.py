import functools

import triton
import triton.language as tl

from triton.language.core import builtin, tensor_descriptor
from triton.language.core import _aggregate as aggregate
from triton.tools.tensor_descriptor import TensorDescriptor

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._core import _unwrap_if_constexpr
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tensor_memory_descriptor,
    tma,
    mbarrier,
)

# ===-----------------------------------------------------------------------===#
# Constexpr Utilities
# ===-----------------------------------------------------------------------===#


def constexpr_fn(fn):

    def recursively_unwrap(c):
        c = _unwrap_if_constexpr(c)
        if hasattr(c, "__iter__"):
            c = [recursively_unwrap(x) for x in c]
        return c

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args = [recursively_unwrap(arg) for arg in args]
        kwargs = {k: recursively_unwrap(v) for k, v in kwargs.items()}
        return fn(*args, **kwargs)

    return wrapper


# ===-----------------------------------------------------------------------===#
# Layout Utilities
# ===-----------------------------------------------------------------------===#


def get_tmem_32x32b_layout(M, N, shape, num_warps):
    assert len(shape) == 2, "expected a 2D tensor"
    assert num_warps in [4, 8], "expected 4 or 8 warps"

    blocks_per_tile = [shape[0] // M, shape[1] // N]
    num_blocks = blocks_per_tile[0] * blocks_per_tile[1]

    num_warp_groups = num_warps // 4
    if M == 64:
        threads_per_warp = [16, 2]
        if num_blocks == 1:
            size_per_thread = [1, N // (num_warp_groups * 2)]
            warps_per_cta = [4, num_warp_groups]
        else:
            size_per_thread = [1, N // 2]
            warps_per_cta = [4 * min(blocks_per_tile[0], num_warp_groups)]
            warps_per_cta.append(triton.cdiv(num_warp_groups, warps_per_cta[0] // 4))
    else:
        if shape[0] > 128:
            size_per_thread = [1, N]
            threads_per_warp = [32, 1]
            warps_per_cta = [4 * num_warp_groups, 1]
        else:
            size_per_thread = [1, N // num_warp_groups]
            threads_per_warp = [32, 1]
            warps_per_cta = [4, num_warp_groups]
    return gl.BlockedLayout(
        size_per_thread=size_per_thread,
        threads_per_warp=threads_per_warp,
        warps_per_cta=warps_per_cta,
        order=[0, 1],
    )


def get_mma_instr_shape(shape, element_ty):
    m = 128 if shape[0] >= 128 else 64
    n = 256 if shape[1] >= 256 else shape[1]
    k = 256 // element_ty.primitive_bitwidth
    return (m, n, k)


@builtin
@constexpr_fn
def get_1d_blocked(num_warps, num_threads, _builder=None):
    return gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[num_threads],
        warps_per_cta=[num_warps],
        order=[0],
    )


@builtin
@constexpr_fn
def get_nvmma_layout(shape, element_ty, order=[1, 0], fp4_padded=False, _builder=None):
    packing_factor = 2 if fp4_padded else 1

    contig_dim_size = shape[order[0]] * packing_factor * element_ty.primitive_bitwidth // 8
    if contig_dim_size >= 128 and contig_dim_size % 128 == 0:
        swizzle_byte_width = 128
    elif contig_dim_size >= 64 and contig_dim_size % 64 == 0:
        swizzle_byte_width = 64
    elif contig_dim_size >= 32 and contig_dim_size % 32 == 0:
        swizzle_byte_width = 32
    else:
        swizzle_byte_width = 0

    flatten_outer_dim = 1
    for i in range(1, len(shape)):
        flatten_outer_dim *= shape[order[i]]
    if len(shape) < 2 or flatten_outer_dim < 8:
        swizzle_byte_width = 0
    transposed = order[0] == 0

    return gl.NVMMASharedLayout(
        swizzle_byte_width=swizzle_byte_width,
        element_bitwidth=element_ty.primitive_bitwidth,
        rank=len(shape),
        transposed=transposed,
        fp4_padded=fp4_padded,
    )


# ===-----------------------------------------------------------------------===#
# Data Abstractions
# ===-----------------------------------------------------------------------===#


@aggregate
class TensorMemoryVariable:
    desc: tensor_memory_descriptor
    reg_layout: gl.constexpr

    @builtin
    @constexpr_fn
    def allocate_mma(num_buffers, shape, element_ty, num_warps, _builder=None):
        instr_shape = get_mma_instr_shape(shape, element_ty)
        tmem_layout = TensorMemoryLayout([instr_shape[0], instr_shape[1]], unpacked=True)
        reg_layout = get_tmem_32x32b_layout(instr_shape[0], instr_shape[1], shape, num_warps)

        desc = allocate_tensor_memory(element_ty, [num_buffers] + list(shape), tmem_layout, _builder=_builder)
        return TensorMemoryVariable(desc, gl.constexpr(reg_layout))

    def __init__(self, desc: tensor_memory_descriptor, reg_layout):
        self.desc = desc
        self.reg_layout = reg_layout

    @gluon.jit
    def store(self, idx, value):
        self.desc.subslice(idx).store(value)


@aggregate
class Barrier:
    handle: gl.shared_memory_descriptor

    @gluon.jit
    def create():
        handle = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(handle, count=1)
        return Barrier(handle)

    def __init__(self, handle):
        self.handle = handle

    @gluon.jit
    def expect(self, size: gl.constexpr):
        mbarrier.expect(self.handle, size)

    @gluon.jit
    def wait(self, phase):
        mbarrier.wait(self.handle, phase)

    @gluon.jit
    def dealloc(self):
        mbarrier.invalidate(self.handle)


@aggregate
class MMAOperandLoader:
    smem: gl.shared_memory_descriptor
    desc: tensor_descriptor
    barrier: Barrier
    phase: gl.tensor

    @gluon.jit
    def create(desc):
        shape: gl.constexpr = desc.block_type.shape
        smem_layout: gl.constexpr = get_nvmma_layout(shape, desc.dtype)

        smem = gl.allocate_shared_memory(desc.dtype, shape, smem_layout)
        barrier = Barrier.create()
        phase = 0
        return MMAOperandLoader(smem, desc, barrier, phase)

    def __init__(self, smem, desc, barrier, phase):
        self.smem = smem
        self.desc = desc
        self.barrier = barrier
        self.phase = phase

    @gluon.jit
    def release(self):
        self.barrier.dealloc()
        self.smem._keep_alive()

    @gluon.jit
    def issue(self, coord):
        size: gl.constexpr = get_load_size_bytes(self.desc)
        tma.async_copy_global_to_local(self.desc, coord, self.barrier.handle, self.smem)
        self.barrier.expect(size)

    @gluon.jit
    def wait(self):
        self.barrier.wait(self.phase)
        self.phase ^= 1
        return self


@aggregate
class MMAv5:
    barrier: Barrier

    @gluon.jit
    def create():
        barrier = Barrier.create()
        x = MMAv5(barrier)
        return x

    def __init__(self, barrier):
        self.barrier = barrier

    @gluon.jit
    def release(self):
        self.barrier.dealloc()


# ===-----------------------------------------------------------------------===#
# _attn_fwd
# ===-----------------------------------------------------------------------===#


@builtin
@constexpr_fn
def get_load_size_bytes(desc, _builder=None):
    size = desc.dtype.primitive_bitwidth // 8
    for dim in desc.block_type.shape:
        size *= dim
    return size


@gluon.jit
def async_load_tensor_desc(desc, smem, barrier):
    size: gl.constexpr = get_load_size_bytes(desc)
    coord: gl.constexpr = [0] * len(desc.shape)
    tma.async_copy_global_to_local(desc, coord, barrier.handle, smem)
    barrier.expect(size)


@gluon.jit
def load_tensor_desc_to_smem(desc, smem):
    barrier = Barrier.create()
    async_load_tensor_desc(desc, smem, barrier)
    barrier.wait(0)
    barrier.dealloc()


@gluon.jit
def _attn_fwd_inner(m_i, l_i,  #
                    q_smem, acc,  #
                    desc_k, desc_v,  #
                    qk_scale, offset_y, start_m,  #
                    offs_m, offs_n,  #
                    dtype: gl.constexpr,  #
                    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,  #
                    STAGE: gl.constexpr, N_CTX: gl.constexpr):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, N_CTX

    k_loader = MMAOperandLoader.create(desc_k)
    v_loader = MMAOperandLoader.create(desc_v)

    mma = MMAv5.create()

    offsetkv_y = offset_y + lo
    for start_n in range(lo, hi, BLOCK_N):
        k_loader.issue([offsetkv_y, 0])
        k_loader = k_loader.wait()

    k_loader.release()
    v_loader.release()
    mma.release()

    return m_i, l_i


@gluon.jit
def _attn_fwd(sm_scale, M, Z, H, N_CTX,  #
              desc_q, desc_k, desc_v, desc_o,  #
              HEAD_DIM: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,  #
              STAGE: gl.constexpr, dtype: gl.constexpr,  #
              num_warps: gl.constexpr, threads_per_warp: gl.constexpr):
    tl.static_assert(BLOCK_N <= HEAD_DIM, "BLOCK_N must be less than or equal to HEAD_DIM")

    start_m = gl.program_id(0)
    off_hz = gl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    # qo_offset_y = offset_y + start_m * BLOCK_M

    layout_1d: gl.constexpr = get_1d_blocked(num_warps, threads_per_warp)
    offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=layout_1d)
    offs_n = gl.arange(0, BLOCK_N, layout=layout_1d)

    qk_scale = sm_scale
    qk_scale *= 1.44269504

    m_i = gl.zeros([BLOCK_M], dtype=tl.float32, layout=layout_1d) - float("inf")
    l_i = gl.zeros([BLOCK_M], dtype=tl.float32, layout=layout_1d) + 1.0

    q_smem_layout: gl.constexpr = get_nvmma_layout(desc_q.block_type.shape, desc_q.dtype)
    q_smem = gl.allocate_shared_memory(desc_q.dtype, desc_q.block_type.shape, q_smem_layout)
    load_tensor_desc_to_smem(desc_q, q_smem)

    acc = TensorMemoryVariable.allocate_mma(1, [BLOCK_M, BLOCK_N], gl.float32, num_warps)
    acc.store(0, gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=acc.reg_layout))

    if STAGE & 1:
        m_i, l_i = _attn_fwd_inner(m_i, l_i, q_smem, acc, desc_k, desc_v,  #
                                   qk_scale, offset_y, start_m, offs_m, offs_n,  #
                                   dtype, BLOCK_M, BLOCK_N, HEAD_DIM, 4 - STAGE, N_CTX)
    if STAGE & 2:
        m_i, l_i = _attn_fwd_inner(m_i, l_i, q_smem, acc, desc_k, desc_v,  #
                                   qk_scale, offset_y, start_m, offs_m, offs_n,  #
                                   dtype, BLOCK_M, BLOCK_N, HEAD_DIM, 2, N_CTX)

    q_smem._keep_alive()


def attention_forward(q, k, v, causal, sm_scale):
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert q.dtype == torch.float16

    stage = 3 if causal else 1
    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    y_dim = q.shape[0] * q.shape[1] * q.shape[2]

    BLOCK_M = 128
    BLOCK_N = min(HEAD_DIM_K, 128)

    desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM_K])
    desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM_K])
    desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM_K])
    desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM_K])

    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)

    kernel = _attn_fwd.warmup(  #
        sm_scale, M, q.shape[0], q.shape[1], q.shape[2],  #
        desc_q, desc_k, desc_v, desc_o,  #
        HEAD_DIM_K, BLOCK_M, BLOCK_N,  #
        stage, gl.float16,  #
        num_warps=4, threads_per_warp=32,  #
        grid=grid,  #
    )
    print(kernel.asm["ttgir"])


if __name__ == "__main__":
    import torch

    BATCH = 4
    H = 32
    N_CTX = 16 * 1024
    HEAD_DIM = 128
    causal = True

    dtype = torch.float16
    device = "cuda"

    sm_scale = 1.3
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)

    attention_forward(q, k, v, causal, sm_scale)
