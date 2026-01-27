# ruff: noqa: E402
import hip

# Needed for internal dev flow for now; will remove later
hip.hip.hipInit(0)

import torch
import pytest
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm
from triton.experimental.gluon.language.amd.gfx1250 import async_copy as cp
from triton.language.core import _aggregate as aggregate
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

# Handle imports for both pytest (module context) and direct execution
try:
    from .gfx1250_utils import static_profile, composition
except ImportError:
    from gfx1250_utils import static_profile, composition


@gluon.constexpr_function
def get_scale_blocked_layout(num_warps: gl.constexpr):
    return gl.BlockedLayout([1, 8], [1, 32], [num_warps // 2, 2], [1, 0])


@gluon.constexpr_function
def get_wmma_layout(num_warps, packed, scale_preshuffle):
    assert (num_warps in (4, 8))
    if scale_preshuffle:
        reg_bases = [[0, 1], [1, 0]]
        tiles_per_warp = 2
    else:
        reg_bases = []
        tiles_per_warp = 1

    # [NUM_WARPS // 2, 2]
    if num_warps == 4:
        warp_bases = [[0, tiles_per_warp], [tiles_per_warp, 0]]
    else:
        warp_bases = [[0, tiles_per_warp], [0, tiles_per_warp * 2], [tiles_per_warp, 0]]

    instr_shape = [16, 16, 64] if packed else [16, 16, 128]

    return gl.amd.AMDWMMALayout(3, True, warp_bases, reg_bases, instr_shape)


@aggregate
class MXFPGEMMConfig:
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    DTYPE_A: gl.constexpr
    DTYPE_B: gl.constexpr
    DIV_FACTOR_A: gl.constexpr
    DIV_FACTOR_B: gl.constexpr
    NUM_BUFFERS: gl.constexpr
    TRANSPOSE_B: gl.constexpr
    WITH_A_SCALE: gl.constexpr
    NUM_LOADS_IN_BATCH: gl.constexpr
    NUM_SUBTILES: gl.constexpr  # (M, N, K)
    NUM_WARPS: gl.constexpr

    # Layouts
    shared_layout_a: gl.constexpr
    dot_layout_a: gl.constexpr

    shared_layout_b: gl.constexpr
    dot_layout_b: gl.constexpr

    shared_layout_a_scale: gl.constexpr
    layout_a_scale: gl.constexpr

    shared_layout_b_scale: gl.constexpr
    layout_b_scale: gl.constexpr

    acc_layout: gl.constexpr

    # Scales
    SCALE_PRESHUFFLE: gl.constexpr
    PRESHUFFLE_FACTOR: gl.constexpr
    SCALE_KWIDTH: gl.constexpr
    BLOCK_M_PRESHUFFLED: gl.constexpr
    BLOCK_N_PRESHUFFLED: gl.constexpr
    BLOCK_K_SCALE_PRESHUFFLED: gl.constexpr
    SCALE_BLOCK: gl.constexpr
    ASYNC_COPY_SCALE: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, BLOCK_M, BLOCK_N, BLOCK_K, DTYPE_A, DTYPE_B, SCALE_BLOCK, NUM_BUFFERS, TRANSPOSE_B, WITH_A_SCALE,
                 SCALE_PRESHUFFLE, NUM_WARPS, ASYNC_COPY_SCALE=False, NUM_SUBTILES=(1, 1, 1)):
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.DTYPE_A = gl.constexpr(DTYPE_A)
        self.DTYPE_B = gl.constexpr(DTYPE_B)
        self.NUM_BUFFERS = gl.constexpr(NUM_BUFFERS)
        self.TRANSPOSE_B = gl.constexpr(TRANSPOSE_B)
        self.WITH_A_SCALE = gl.constexpr(WITH_A_SCALE)
        self.SCALE_PRESHUFFLE = gl.constexpr(SCALE_PRESHUFFLE)
        self.SCALE_BLOCK = gl.constexpr(SCALE_BLOCK)
        self.DIV_FACTOR_A = gl.constexpr(2 if DTYPE_A == "e2m1" else 1)
        self.DIV_FACTOR_B = gl.constexpr(2 if DTYPE_B == "e2m1" else 1)
        self.NUM_LOADS_IN_BATCH = gl.constexpr(4 if WITH_A_SCALE else 3)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        self.ASYNC_COPY_SCALE = gl.constexpr(ASYNC_COPY_SCALE)
        self.NUM_SUBTILES = gl.constexpr(NUM_SUBTILES)

        NUM_SUBTILES_M = self.NUM_SUBTILES[0]
        NUM_SUBTILES_N = self.NUM_SUBTILES[1]
        NUM_SUBTILES_K = self.NUM_SUBTILES[2]

        BLOCK_K_SCALE = BLOCK_K // SCALE_BLOCK
        self.SCALE_KWIDTH = gl.constexpr(4 if BLOCK_K_SCALE >= 4 else BLOCK_K_SCALE)
        self.PRESHUFFLE_FACTOR = gl.constexpr(128 if SCALE_PRESHUFFLE else 1)

        self.BLOCK_M_PRESHUFFLED = gl.constexpr(BLOCK_M // self.PRESHUFFLE_FACTOR)
        self.BLOCK_N_PRESHUFFLED = gl.constexpr(BLOCK_N // self.PRESHUFFLE_FACTOR)
        self.BLOCK_K_SCALE_PRESHUFFLED = gl.constexpr(BLOCK_K_SCALE * self.PRESHUFFLE_FACTOR)

        WMMA_LAYOUT: gl.constexpr = get_wmma_layout(NUM_WARPS, False, SCALE_PRESHUFFLE)
        WMMA_LAYOUT_PACKED: gl.constexpr = get_wmma_layout(NUM_WARPS, True, SCALE_PRESHUFFLE)

        self.dot_layout_a = gl.constexpr(
            gl.DotOperandLayout(operand_index=0, parent=WMMA_LAYOUT_PACKED if DTYPE_A == "e2m1" else WMMA_LAYOUT,
                                k_width=16))
        self.dot_layout_b = gl.constexpr(
            gl.DotOperandLayout(operand_index=1, parent=WMMA_LAYOUT_PACKED if DTYPE_B == "e2m1" else WMMA_LAYOUT,
                                k_width=16))
        self.layout_a_scale = gl.constexpr(
            gl.amd.gfx1250.get_wmma_scale_layout(self.dot_layout_a,
                                                 [BLOCK_M // NUM_SUBTILES_M, BLOCK_K_SCALE // NUM_SUBTILES_K]))
        self.layout_b_scale = gl.constexpr(
            gl.amd.gfx1250.get_wmma_scale_layout(self.dot_layout_b,
                                                 [BLOCK_N // NUM_SUBTILES_N, BLOCK_K_SCALE // NUM_SUBTILES_K]))
        self.acc_layout = gl.constexpr(WMMA_LAYOUT)

        BLOCK_K_PACKED_A = BLOCK_K // self.DIV_FACTOR_A // NUM_SUBTILES_K
        BLOCK_K_PACKED_B = BLOCK_K // self.DIV_FACTOR_B // NUM_SUBTILES_K
        self.shared_layout_a = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for([[BLOCK_K_PACKED_A, 16]],
                                                    [BLOCK_M // NUM_SUBTILES_M, BLOCK_K_PACKED_A], [1, 0]))
        if TRANSPOSE_B:
            self.shared_layout_b = gl.constexpr(
                gl.PaddedSharedLayout.with_identity_for([[BLOCK_K_PACKED_B, 16]],
                                                        [BLOCK_N // NUM_SUBTILES_N, BLOCK_K_PACKED_B], [1, 0]))
        else:
            self.shared_layout_b = gl.constexpr(
                gl.PaddedSharedLayout.with_identity_for([[BLOCK_N // NUM_SUBTILES_N, 16]],
                                                        [BLOCK_K_PACKED_B, BLOCK_N // NUM_SUBTILES_N], [1, 0]))

        self.shared_layout_a_scale = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for(
                [[256, 16]],
                [self.BLOCK_M_PRESHUFFLED // NUM_SUBTILES_M, self.BLOCK_K_SCALE_PRESHUFFLED // NUM_SUBTILES_K], [1, 0]))
        self.shared_layout_b_scale = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for(
                [[256, 16]],
                [self.BLOCK_N_PRESHUFFLED // NUM_SUBTILES_N, self.BLOCK_K_SCALE_PRESHUFFLED // NUM_SUBTILES_K], [1, 0]))


@aggregate
class ScaleAsyncCopyDescriptor:
    cfg: MXFPGEMMConfig
    op_idx: gl.constexpr
    ptr: gl.tensor
    offs: gl.tensor
    step_nonk: gl.tensor
    step_k: gl.tensor
    dtype: gl.constexpr
    block_shape: gl.constexpr
    layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, cfg: MXFPGEMMConfig, op_idx, ptr, offs, step_nonk, step_k, layout):
        self.cfg = cfg
        self.op_idx = gl.constexpr(op_idx)
        self.ptr = ptr
        self.offs = offs
        self.step_nonk = step_nonk
        self.step_k = step_k
        BLOCK_NONK = cfg.BLOCK_M_PRESHUFFLED if op_idx == 0 else cfg.BLOCK_N_PRESHUFFLED
        self.dtype = gl.constexpr(ptr.dtype.element_ty)
        self.block_shape = gl.constexpr(
            [BLOCK_NONK // cfg.NUM_SUBTILES[op_idx], cfg.BLOCK_K_SCALE_PRESHUFFLED // cfg.NUM_SUBTILES[2]])
        self.layout = gl.constexpr(layout)

    @gluon.jit
    def initialize(cfg: MXFPGEMMConfig, op_idx: gl.constexpr, ptr, off, stride, layout):
        gl.static_assert(op_idx == 0 or op_idx == 1)
        if op_idx == 0:
            BLOCK_NONK: gl.constexpr = cfg.BLOCK_M_PRESHUFFLED // cfg.NUM_SUBTILES[op_idx]
        else:
            BLOCK_NONK: gl.constexpr = cfg.BLOCK_N_PRESHUFFLED // cfg.NUM_SUBTILES[op_idx]
        BLOCK_K: gl.constexpr = cfg.BLOCK_K_SCALE_PRESHUFFLED // cfg.NUM_SUBTILES[2]

        blocked_layout: gl.constexpr = get_scale_blocked_layout(cfg.NUM_WARPS)
        offs_non_k = gl.arange(0, BLOCK_NONK, gl.SliceLayout(1, blocked_layout))
        offs_k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, blocked_layout))
        offs = off + offs_non_k[:, None] * stride + offs_k[None, :]
        step_nonk = BLOCK_NONK * stride
        step_k = BLOCK_K

        return ScaleAsyncCopyDescriptor(cfg, op_idx, ptr, offs, step_nonk, step_k, layout)

    @gluon.jit
    def issue_async_load(self, idx: int, buffer, pred=1):
        NUM_SUBTILES_NONK: gl.constexpr = self.cfg.NUM_SUBTILES[self.op_idx]
        if pred:
            cp.global_to_shared(
                buffer, self.ptr + (idx % NUM_SUBTILES_NONK) * self.step_nonk +
                (idx // NUM_SUBTILES_NONK) * self.step_k + self.offs)
            cp.commit_group()


@aggregate
class MXFPGEMMProgramBase:

    @gluon.constexpr_function
    def __init__(self):
        pass

    @gluon.jit
    def issue_loads(self, load_idx, a_buffer, b_buffer, a_scale_buffer, b_scale_buffer, pred=1):
        cfg = self.cfg
        NUM_SUBTILES_K = cfg.NUM_SUBTILES[2]
        BLOCK_K_PACKED_A: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_A // NUM_SUBTILES_K
        BLOCK_K_PACKED_B: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_B // NUM_SUBTILES_K

        gl.amd.gfx1250.tdm.async_load(self.a_desc,  #
                                      [0, load_idx * BLOCK_K_PACKED_A],  #
                                      a_buffer.index((load_idx // NUM_SUBTILES_K) % cfg.NUM_BUFFERS),  #
                                      pred=pred)
        if cfg.TRANSPOSE_B:
            gl.amd.gfx1250.tdm.async_load(self.b_desc,  #
                                          [0, load_idx * BLOCK_K_PACKED_B],  #
                                          b_buffer.index((load_idx // NUM_SUBTILES_K) % cfg.NUM_BUFFERS),  #
                                          pred=pred)
        else:
            gl.amd.gfx1250.tdm.async_load(self.b_desc,  #
                                          [load_idx * BLOCK_K_PACKED_B, 0],  #
                                          b_buffer.index((load_idx // NUM_SUBTILES_K) % cfg.NUM_BUFFERS),  #
                                          pred=pred)
        if cfg.WITH_A_SCALE:
            gl.amd.gfx1250.tdm.async_load(self.a_scale_desc,  #
                                          [0, load_idx * cfg.BLOCK_K_SCALE_PRESHUFFLED // NUM_SUBTILES_K],  #
                                          a_scale_buffer.index((load_idx // NUM_SUBTILES_K) % cfg.NUM_BUFFERS),  #
                                          pred=pred)
        gl.amd.gfx1250.tdm.async_load(self.b_scale_desc,  #
                                      [0, load_idx * cfg.BLOCK_K_SCALE_PRESHUFFLED // NUM_SUBTILES_K],  #
                                      b_scale_buffer.index((load_idx // NUM_SUBTILES_K) % cfg.NUM_BUFFERS),  #
                                      pred=pred)

        return load_idx + 1

    @gluon.jit
    def issue_local_loads(self, wmma_idx, a_buffer, b_buffer, a_scale_buffer, b_scale_buffer):
        cfg = self.cfg
        NUM_SUBTILES_K: gl.constexpr = cfg.NUM_SUBTILES[2]
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK // NUM_SUBTILES_K
        a = a_buffer.index(wmma_idx % cfg.NUM_BUFFERS).load(layout=cfg.dot_layout_a)
        if cfg.TRANSPOSE_B:
            b = b_buffer.index(wmma_idx % cfg.NUM_BUFFERS).permute([1, 0]).load(layout=cfg.dot_layout_b)
        else:
            b = b_buffer.index(wmma_idx % cfg.NUM_BUFFERS).load(layout=cfg.dot_layout_b)
        if cfg.WITH_A_SCALE:
            a_scale_buffer_slice = a_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
        b_scale_buffer_slice = b_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
        if cfg.SCALE_PRESHUFFLE:
            if cfg.WITH_A_SCALE:
                a_scale_buffer_slice = a_scale_buffer_slice.reshape((
                    cfg.BLOCK_M_PRESHUFFLED,  #
                    BLOCK_K_SCALE // cfg.SCALE_KWIDTH,  #
                    cfg.PRESHUFFLE_FACTOR // 4,  #
                    4,  #
                    cfg.SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((cfg.BLOCK_M, BLOCK_K_SCALE))
            b_scale_buffer_slice = b_scale_buffer_slice.reshape((
                cfg.BLOCK_N_PRESHUFFLED,  #
                BLOCK_K_SCALE // cfg.SCALE_KWIDTH,  #
                cfg.PRESHUFFLE_FACTOR // 4,  #
                4,  #
                cfg.SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((cfg.BLOCK_N, BLOCK_K_SCALE))
        if cfg.WITH_A_SCALE:
            scale_a = a_scale_buffer_slice.load(layout=cfg.layout_a_scale)
        else:
            # Use a placeholder to make compiler happy
            scale_a = gl.constexpr(0)
        scale_b = b_scale_buffer_slice.load(layout=cfg.layout_b_scale)
        return a, b, scale_a, scale_b


@composition
@aggregate
class MXFPGEMMPipelinedProgram:
    base: MXFPGEMMProgramBase

    cfg: MXFPGEMMConfig
    a_buffer: gl.shared_memory_descriptor
    b_buffer: gl.shared_memory_descriptor
    a_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    b_scale_buffer: gl.shared_memory_descriptor

    a_desc: tdm.tensor_descriptor
    b_desc: tdm.tensor_descriptor
    a_scale_desc: tdm.tensor_descriptor | gl.constexpr
    b_scale_desc: tdm.tensor_descriptor

    c_ptr: gl.tensor
    c_offs: gl.tensor
    c_mask: gl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg: MXFPGEMMConfig, a_buffer, b_buffer, a_scale_buffer, b_scale_buffer, a_desc, b_desc,
                 a_scale_desc, b_scale_desc, c_ptr, c_offs, c_mask):
        self.cfg = cfg
        self.a_buffer = a_buffer
        self.b_buffer = b_buffer
        # Have to use constexpr to workaround a compiler issue with optional scale
        if cfg.WITH_A_SCALE:
            self.a_scale_buffer = a_scale_buffer
        else:
            self.a_scale_buffer = gl.constexpr(a_scale_buffer)
        self.b_scale_buffer = b_scale_buffer
        self.a_desc = a_desc
        self.b_desc = b_desc
        if cfg.WITH_A_SCALE:
            self.a_scale_desc = a_scale_desc
        else:
            self.a_scale_desc = gl.constexpr(a_scale_desc)
        self.b_scale_desc = b_scale_desc
        self.c_ptr = c_ptr
        self.c_offs = c_offs
        self.c_mask = c_mask

        self.base = MXFPGEMMProgramBase()

    @gluon.jit
    def initialize(cfg: MXFPGEMMConfig, a_desc, b_desc, a_scale_desc, b_scale_desc, c_ptr, c_offs, c_mask):
        NUM_BUFFERS: gl.constexpr = cfg.NUM_BUFFERS
        a_buffer = gl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape,
                                             layout=a_desc.layout)
        b_buffer = gl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape,
                                             layout=b_desc.layout)
        if cfg.WITH_A_SCALE:
            a_scale_buffer = gl.allocate_shared_memory(a_scale_desc.dtype,
                                                       shape=[NUM_BUFFERS] + a_scale_desc.block_shape,
                                                       layout=a_scale_desc.layout)
        else:
            a_scale_buffer = gl.constexpr(0)

        b_scale_buffer = gl.allocate_shared_memory(b_scale_desc.dtype, shape=[NUM_BUFFERS] + b_scale_desc.block_shape,
                                                   layout=b_scale_desc.layout)

        return MXFPGEMMPipelinedProgram(cfg, a_buffer, b_buffer, a_scale_buffer, b_scale_buffer, a_desc, b_desc,
                                        a_scale_desc, b_scale_desc, c_ptr, c_offs, c_mask)

    @gluon.jit
    def pipeline(self, K):
        cfg = self.cfg
        load_idx = 0
        wmma_idx = 0

        # prologue
        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            load_idx = self.issue_loads(load_idx, self.a_buffer, self.b_buffer, self.a_scale_buffer,
                                        self.b_scale_buffer)

        accumulator = gl.zeros((cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=self.cfg.acc_layout)
        loop_ub = gl.cdiv(K, cfg.BLOCK_K)
        gl.assume(loop_ub > 0)
        epilogue_lb = loop_ub - (cfg.NUM_BUFFERS - 1)
        for i in range(0, loop_ub):
            pred = i - epilogue_lb
            pred = (pred >> 31) & 1
            load_idx = self.issue_loads(load_idx, self.a_buffer, self.b_buffer, self.a_scale_buffer,
                                        self.b_scale_buffer, pred=pred)

            gl.amd.gfx1250.tdm.async_wait((cfg.NUM_BUFFERS - 1) * self.cfg.NUM_LOADS_IN_BATCH)

            a, b, scale_a, scale_b = self.issue_local_loads(wmma_idx, self.a_buffer, self.b_buffer, self.a_scale_buffer,
                                                            self.b_scale_buffer)
            wmma_idx += 1
            accumulator = gl.amd.gfx1250.wmma_scaled(a, scale_a, cfg.DTYPE_A, b, scale_b, cfg.DTYPE_B, accumulator)

        gl.amd.gfx1250.buffer_store(accumulator, self.c_ptr, self.c_offs, mask=self.c_mask)

    @gluon.jit
    def warp_pipeline(self, K):
        cfg = self.cfg
        load_idx = 0
        wmma_idx = 0

        # prologue
        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            load_idx = self.issue_loads(load_idx, self.a_buffer, self.b_buffer, self.a_scale_buffer,
                                        self.b_scale_buffer)

        accumulator = gl.zeros((cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=self.cfg.acc_layout)
        loop_ub = gl.cdiv(K, cfg.BLOCK_K) - (cfg.NUM_BUFFERS - 1)
        gl.amd.gfx1250.tdm.async_wait((cfg.NUM_BUFFERS - 2) * self.cfg.NUM_LOADS_IN_BATCH)
        gl.assume(loop_ub >= 0)
        for _ in range(0, loop_ub):
            with gl.amd.warp_pipeline_stage("lds", priority=1):
                a, b, scale_a, scale_b = self.issue_local_loads(wmma_idx, self.a_buffer, self.b_buffer,
                                                                self.a_scale_buffer, self.b_scale_buffer)
                wmma_idx += 1

            gl.amd.gfx1250.tdm.async_wait((cfg.NUM_BUFFERS - 3) * self.cfg.NUM_LOADS_IN_BATCH)
            with gl.amd.warp_pipeline_stage("tdm+wmma", priority=0):
                load_idx = self.issue_loads(load_idx, self.a_buffer, self.b_buffer, self.a_scale_buffer,
                                            self.b_scale_buffer)
                accumulator = gl.amd.gfx1250.wmma_scaled(a, scale_a, cfg.DTYPE_A, b, scale_b, cfg.DTYPE_B, accumulator)

        # epilogue
        for i in gl.static_range(cfg.NUM_BUFFERS - 1):
            gl.amd.gfx1250.tdm.async_wait((cfg.NUM_BUFFERS - 1 - i) * self.cfg.NUM_LOADS_IN_BATCH)
            a, b, scale_a, scale_b = self.issue_local_loads(wmma_idx, self.a_buffer, self.b_buffer, self.a_scale_buffer,
                                                            self.b_scale_buffer)
            wmma_idx += 1
            accumulator = gl.amd.gfx1250.wmma_scaled(a, scale_a, cfg.DTYPE_A, b, scale_b, cfg.DTYPE_B, accumulator)

        gl.amd.gfx1250.buffer_store(accumulator, self.c_ptr, self.c_offs, mask=self.c_mask)


@composition
@aggregate
class MXFPGEMMSliceKProgram:
    base: MXFPGEMMProgramBase

    cfg: MXFPGEMMConfig
    a_buffer0: gl.shared_memory_descriptor
    a_buffer1: gl.shared_memory_descriptor
    b_buffer0: gl.shared_memory_descriptor
    b_buffer1: gl.shared_memory_descriptor
    a_scale_buffer0: gl.shared_memory_descriptor | gl.constexpr
    a_scale_buffer1: gl.shared_memory_descriptor | gl.constexpr
    b_scale_buffer0: gl.shared_memory_descriptor
    b_scale_buffer1: gl.shared_memory_descriptor

    a_desc: tdm.tensor_descriptor
    b_desc: tdm.tensor_descriptor
    a_scale_desc: tdm.tensor_descriptor | gl.constexpr
    b_scale_desc: tdm.tensor_descriptor

    c_ptr: gl.tensor
    c_offs: gl.tensor
    c_mask: gl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg: MXFPGEMMConfig, a_buffer0, a_buffer1, b_buffer0, b_buffer1, a_scale_buffer0,
                 a_scale_buffer1, b_scale_buffer0, b_scale_buffer1, a_desc, b_desc, a_scale_desc, b_scale_desc, c_ptr,
                 c_offs, c_mask):
        self.cfg = cfg
        self.a_buffer0 = a_buffer0
        self.a_buffer1 = a_buffer1
        self.b_buffer0 = b_buffer0
        self.b_buffer1 = b_buffer1
        # Have to use constexpr to workaround a compiler issue with optional scale
        if cfg.WITH_A_SCALE:
            self.a_scale_buffer0 = a_scale_buffer0
            self.a_scale_buffer1 = a_scale_buffer1
        else:
            self.a_scale_buffer0 = gl.constexpr(a_scale_buffer0)
            self.a_scale_buffer1 = gl.constexpr(a_scale_buffer1)
        self.b_scale_buffer0 = b_scale_buffer0
        self.b_scale_buffer1 = b_scale_buffer1
        self.a_desc = a_desc
        self.b_desc = b_desc
        if cfg.WITH_A_SCALE:
            self.a_scale_desc = a_scale_desc
        else:
            self.a_scale_desc = gl.constexpr(a_scale_desc)
        self.b_scale_desc = b_scale_desc
        self.c_ptr = c_ptr
        self.c_offs = c_offs
        self.c_mask = c_mask

        self.base = MXFPGEMMProgramBase()

    @gluon.jit
    def initialize(cfg: MXFPGEMMConfig, a_desc, b_desc, a_scale_desc, b_scale_desc, c_ptr, c_offs, c_mask):
        NUM_BUFFERS: gl.constexpr = cfg.NUM_BUFFERS
        a_buffer0 = gl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape,
                                              layout=a_desc.layout)
        a_buffer1 = gl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape,
                                              layout=a_desc.layout)
        b_buffer0 = gl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape,
                                              layout=b_desc.layout)
        b_buffer1 = gl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape,
                                              layout=b_desc.layout)
        if cfg.WITH_A_SCALE:
            a_scale_buffer0 = gl.allocate_shared_memory(a_scale_desc.dtype,
                                                        shape=[NUM_BUFFERS] + a_scale_desc.block_shape,
                                                        layout=a_scale_desc.layout)
            a_scale_buffer1 = gl.allocate_shared_memory(a_scale_desc.dtype,
                                                        shape=[NUM_BUFFERS] + a_scale_desc.block_shape,
                                                        layout=a_scale_desc.layout)
        else:
            a_scale_buffer0 = gl.constexpr(0)
            a_scale_buffer1 = gl.constexpr(0)

        b_scale_buffer0 = gl.allocate_shared_memory(b_scale_desc.dtype, shape=[NUM_BUFFERS] + b_scale_desc.block_shape,
                                                    layout=b_scale_desc.layout)
        b_scale_buffer1 = gl.allocate_shared_memory(b_scale_desc.dtype, shape=[NUM_BUFFERS] + b_scale_desc.block_shape,
                                                    layout=b_scale_desc.layout)

        return MXFPGEMMSliceKProgram(cfg, a_buffer0, a_buffer1, b_buffer0, b_buffer1, a_scale_buffer0, a_scale_buffer1,
                                     b_scale_buffer0, b_scale_buffer1, a_desc, b_desc, a_scale_desc, b_scale_desc,
                                     c_ptr, c_offs, c_mask)

    @gluon.jit
    def pipeline(self, K):
        cfg = self.cfg
        load_idx = 0
        wmma_idx = 0

        NUM_SUBTILES_K: gl.constexpr = cfg.NUM_SUBTILES[2]

        # prologue
        # iter 0
        load_idx = self.issue_loads(load_idx, self.a_buffer0, self.b_buffer0, self.a_scale_buffer0,
                                    self.b_scale_buffer0)
        load_idx = self.issue_loads(load_idx, self.a_buffer1, self.b_buffer1, self.a_scale_buffer1,
                                    self.b_scale_buffer1)

        # iter 1
        load_idx = self.issue_loads(load_idx, self.a_buffer0, self.b_buffer0, self.a_scale_buffer0,
                                    self.b_scale_buffer0)
        load_idx = self.issue_loads(load_idx, self.a_buffer1, self.b_buffer1, self.a_scale_buffer1,
                                    self.b_scale_buffer1)
        # iter 0
        gl.amd.gfx1250.tdm.async_wait((cfg.NUM_BUFFERS - 1) * cfg.NUM_LOADS_IN_BATCH * NUM_SUBTILES_K)
        a0, b0, scale_a0, scale_b0 = self.issue_local_loads(wmma_idx, self.a_buffer0, self.b_buffer0,
                                                            self.a_scale_buffer0, self.b_scale_buffer0)

        accumulator = gl.zeros((cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=self.cfg.acc_layout)
        loop_ub = gl.cdiv(K, cfg.BLOCK_K) - 1
        for _ in range(0, loop_ub - 1):
            # iter i
            accumulator = gl.amd.gfx1250.wmma_scaled(a0, scale_a0, cfg.DTYPE_A, b0, scale_b0, cfg.DTYPE_B, accumulator)

            # iter i
            a1, b1, scale_a1, scale_b1 = self.issue_local_loads(wmma_idx, self.a_buffer1, self.b_buffer1,
                                                                self.a_scale_buffer1, self.b_scale_buffer1)
            wmma_idx += 1

            # iter i + 2
            load_idx = self.issue_loads(load_idx, self.a_buffer0, self.b_buffer0, self.a_scale_buffer0,
                                        self.b_scale_buffer0)
            load_idx = self.issue_loads(load_idx, self.a_buffer1, self.b_buffer1, self.a_scale_buffer1,
                                        self.b_scale_buffer1)

            # iter i
            accumulator = gl.amd.gfx1250.wmma_scaled(a1, scale_a1, cfg.DTYPE_A, b1, scale_b1, cfg.DTYPE_B, accumulator)

            # iter i + 1
            gl.amd.gfx1250.tdm.async_wait((cfg.NUM_BUFFERS - 1) * cfg.NUM_LOADS_IN_BATCH * NUM_SUBTILES_K)
            a0, b0, scale_a0, scale_b0 = self.issue_local_loads(wmma_idx, self.a_buffer0, self.b_buffer0,
                                                                self.a_scale_buffer0, self.b_scale_buffer0)

        # epilogue
        # iter end - 2
        accumulator = gl.amd.gfx1250.wmma_scaled(a0, scale_a0, cfg.DTYPE_A, b0, scale_b0, cfg.DTYPE_B, accumulator)

        # iter end - 2
        a1, b1, scale_a1, scale_b1 = self.issue_local_loads(wmma_idx, self.a_buffer1, self.b_buffer1,
                                                            self.a_scale_buffer1, self.b_scale_buffer1)
        wmma_idx += 1

        # iter end - 2
        accumulator = gl.amd.gfx1250.wmma_scaled(a1, scale_a1, cfg.DTYPE_A, b1, scale_b1, cfg.DTYPE_B, accumulator)
        # iter end - 1
        gl.amd.gfx1250.tdm.async_wait(0)
        a0, b0, scale_a0, scale_b0 = self.issue_local_loads(wmma_idx, self.a_buffer0, self.b_buffer0,
                                                            self.a_scale_buffer0, self.b_scale_buffer0)
        # iter end - 1
        accumulator = gl.amd.gfx1250.wmma_scaled(a0, scale_a0, cfg.DTYPE_A, b0, scale_b0, cfg.DTYPE_B, accumulator)

        # iter end - 1
        a1, b1, scale_a1, scale_b1 = self.issue_local_loads(wmma_idx, self.a_buffer1, self.b_buffer1,
                                                            self.a_scale_buffer1, self.b_scale_buffer1)
        wmma_idx += 1

        accumulator = gl.amd.gfx1250.wmma_scaled(a1, scale_a1, cfg.DTYPE_A, b1, scale_b1, cfg.DTYPE_B, accumulator)

        gl.amd.gfx1250.buffer_store(accumulator, self.c_ptr, self.c_offs, mask=self.c_mask)

    @gluon.jit
    def warp_pipeline(self, K):
        cfg = self.cfg
        load_idx = 0
        wmma_idx = 0
        gl.static_assert(cfg.NUM_BUFFERS == 3)

        NUM_SUBTILES_K: gl.constexpr = cfg.NUM_SUBTILES[2]

        # prologue
        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            load_idx = self.issue_loads(load_idx, self.a_buffer0, self.b_buffer0, self.a_scale_buffer0,
                                        self.b_scale_buffer0)
            load_idx = self.issue_loads(load_idx, self.a_buffer1, self.b_buffer1, self.a_scale_buffer1,
                                        self.b_scale_buffer1)

        accumulator = gl.zeros((cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=self.cfg.acc_layout)
        loop_ub = gl.cdiv(K, cfg.BLOCK_K) - (cfg.NUM_BUFFERS - 1)
        gl.assume(loop_ub >= 0)
        # wait for the first prefetch
        gl.amd.gfx1250.tdm.async_wait((cfg.NUM_BUFFERS - 2) * self.cfg.NUM_LOADS_IN_BATCH * NUM_SUBTILES_K)
        for _ in range(0, loop_ub):
            with gl.amd.warp_pipeline_stage("lds0", priority=1):
                a0, b0, scale_a0, scale_b0 = self.issue_local_loads(wmma_idx, self.a_buffer0, self.b_buffer0,
                                                                    self.a_scale_buffer0, self.b_scale_buffer0)

            gl.amd.gfx1250.tdm.async_wait((cfg.NUM_BUFFERS - 3) * self.cfg.NUM_LOADS_IN_BATCH * NUM_SUBTILES_K)
            with gl.amd.warp_pipeline_stage("tdm+wmma+lds1", priority=0):
                load_idx = self.issue_loads(load_idx, self.a_buffer0, self.b_buffer0, self.a_scale_buffer0,
                                            self.b_scale_buffer0)
                load_idx = self.issue_loads(load_idx, self.a_buffer1, self.b_buffer1, self.a_scale_buffer1,
                                            self.b_scale_buffer1)
                accumulator = gl.amd.gfx1250.wmma_scaled(a0, scale_a0, cfg.DTYPE_A, b0, scale_b0, cfg.DTYPE_B,
                                                         accumulator)
                a1, b1, scale_a1, scale_b1 = self.issue_local_loads(wmma_idx, self.a_buffer1, self.b_buffer1,
                                                                    self.a_scale_buffer1, self.b_scale_buffer1)
                wmma_idx += 1
                accumulator = gl.amd.gfx1250.wmma_scaled(a1, scale_a1, cfg.DTYPE_A, b1, scale_b1, cfg.DTYPE_B,
                                                         accumulator)

        # epilogue
        for i in gl.static_range(cfg.NUM_BUFFERS - 1):
            gl.amd.gfx1250.tdm.async_wait((cfg.NUM_BUFFERS - 1 - i) * self.cfg.NUM_LOADS_IN_BATCH * NUM_SUBTILES_K)
            a0, b0, scale_a0, scale_b0 = self.issue_local_loads(wmma_idx, self.a_buffer0, self.b_buffer0,
                                                                self.a_scale_buffer0, self.b_scale_buffer0)
            accumulator = gl.amd.gfx1250.wmma_scaled(a0, scale_a0, cfg.DTYPE_A, b0, scale_b0, cfg.DTYPE_B, accumulator)

            a1, b1, scale_a1, scale_b1 = self.issue_local_loads(wmma_idx, self.a_buffer1, self.b_buffer1,
                                                                self.a_scale_buffer1, self.b_scale_buffer1)
            accumulator = gl.amd.gfx1250.wmma_scaled(a1, scale_a1, cfg.DTYPE_A, b1, scale_b1, cfg.DTYPE_B, accumulator)
            wmma_idx += 1

        gl.amd.gfx1250.buffer_store(accumulator, self.c_ptr, self.c_offs, mask=self.c_mask)


@aggregate
class MXFPGEMMSliceNKProgram:
    cfg: MXFPGEMMConfig
    a_buffer0: gl.shared_memory_descriptor
    a_buffer1: gl.shared_memory_descriptor
    b_buffer00: gl.shared_memory_descriptor
    b_buffer01: gl.shared_memory_descriptor
    b_buffer10: gl.shared_memory_descriptor
    b_buffer11: gl.shared_memory_descriptor
    a_scale_buffer0: gl.shared_memory_descriptor | gl.constexpr
    a_scale_buffer1: gl.shared_memory_descriptor | gl.constexpr
    b_scale_buffer00: gl.shared_memory_descriptor
    b_scale_buffer01: gl.shared_memory_descriptor
    b_scale_buffer10: gl.shared_memory_descriptor
    b_scale_buffer11: gl.shared_memory_descriptor

    a_desc: tdm.tensor_descriptor
    b_desc: tdm.tensor_descriptor
    a_scale_desc: tdm.tensor_descriptor | ScaleAsyncCopyDescriptor | gl.constexpr
    b_scale_desc: tdm.tensor_descriptor | ScaleAsyncCopyDescriptor

    c_ptr: gl.tensor
    c_offs: gl.tensor
    c_mask: gl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg: MXFPGEMMConfig, a_buffer0, a_buffer1, b_buffer00, b_buffer01, b_buffer10, b_buffer11,
                 a_scale_buffer0, a_scale_buffer1, b_scale_buffer00, b_scale_buffer01, b_scale_buffer10,
                 b_scale_buffer11, a_desc, b_desc, a_scale_desc, b_scale_desc, c_ptr, c_offs, c_mask):
        self.cfg = cfg
        self.a_buffer0 = a_buffer0
        self.a_buffer1 = a_buffer1
        self.b_buffer00 = b_buffer00
        self.b_buffer01 = b_buffer01
        self.b_buffer10 = b_buffer10
        self.b_buffer11 = b_buffer11
        if cfg.WITH_A_SCALE:
            self.a_scale_buffer0 = a_scale_buffer0
            self.a_scale_buffer1 = a_scale_buffer1
        else:
            self.a_scale_buffer0 = gl.constexpr(a_scale_buffer0)
            self.a_scale_buffer1 = gl.constexpr(a_scale_buffer1)

        self.b_scale_buffer00 = b_scale_buffer00
        self.b_scale_buffer01 = b_scale_buffer01
        self.b_scale_buffer10 = b_scale_buffer10
        self.b_scale_buffer11 = b_scale_buffer11
        self.a_desc = a_desc
        self.b_desc = b_desc
        if cfg.WITH_A_SCALE:
            self.a_scale_desc = a_scale_desc
        else:
            self.a_scale_desc = gl.constexpr(a_scale_desc)
        self.b_scale_desc = b_scale_desc
        self.c_ptr = c_ptr
        self.c_offs = c_offs
        self.c_mask = c_mask

    @gluon.jit
    def initialize(cfg: MXFPGEMMConfig, a_desc, b_desc, a_scale_desc, b_scale_desc, c_ptr, c_offs, c_mask):
        NUM_BUFFERS: gl.constexpr = cfg.NUM_BUFFERS
        a_buffer0 = gl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape,
                                              layout=a_desc.layout)
        a_buffer1 = gl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape,
                                              layout=a_desc.layout)
        b_buffer00 = gl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape,
                                               layout=b_desc.layout)
        b_buffer01 = gl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape,
                                               layout=b_desc.layout)
        b_buffer10 = gl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape,
                                               layout=b_desc.layout)
        b_buffer11 = gl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape,
                                               layout=b_desc.layout)
        if cfg.WITH_A_SCALE:
            a_scale_buffer0 = gl.allocate_shared_memory(a_scale_desc.dtype,
                                                        shape=[NUM_BUFFERS] + a_scale_desc.block_shape,
                                                        layout=a_scale_desc.layout)
            a_scale_buffer1 = gl.allocate_shared_memory(a_scale_desc.dtype,
                                                        shape=[NUM_BUFFERS] + a_scale_desc.block_shape,
                                                        layout=a_scale_desc.layout)
        else:
            a_scale_buffer0 = gl.constexpr(0)
            a_scale_buffer1 = gl.constexpr(0)

        b_scale_buffer00 = gl.allocate_shared_memory(b_scale_desc.dtype, shape=[NUM_BUFFERS] + b_scale_desc.block_shape,
                                                     layout=b_scale_desc.layout)
        b_scale_buffer01 = gl.allocate_shared_memory(b_scale_desc.dtype, shape=[NUM_BUFFERS] + b_scale_desc.block_shape,
                                                     layout=b_scale_desc.layout)
        b_scale_buffer10 = gl.allocate_shared_memory(b_scale_desc.dtype, shape=[NUM_BUFFERS] + b_scale_desc.block_shape,
                                                     layout=b_scale_desc.layout)
        b_scale_buffer11 = gl.allocate_shared_memory(b_scale_desc.dtype, shape=[NUM_BUFFERS] + b_scale_desc.block_shape,
                                                     layout=b_scale_desc.layout)

        return MXFPGEMMSliceNKProgram(cfg, a_buffer0, a_buffer1, b_buffer00, b_buffer01, b_buffer10, b_buffer11,
                                      a_scale_buffer0, a_scale_buffer1, b_scale_buffer00, b_scale_buffer01,
                                      b_scale_buffer10, b_scale_buffer11, a_desc, b_desc, a_scale_desc, b_scale_desc,
                                      c_ptr, c_offs, c_mask)

    @gluon.jit
    def issue_subtile_local_loads(self, wmma_idx, subtile_start, a_buffer, b_buffer, a_scale_buffer, b_scale_buffer,
                                  SUBTILE_LEN: gl.constexpr):
        cfg = self.cfg
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
        SUBTILE_LEN_SCALE: gl.constexpr = SUBTILE_LEN // cfg.SCALE_BLOCK
        a = a_buffer.index(wmma_idx % cfg.NUM_BUFFERS).slice(subtile_start // cfg.DIV_FACTOR_A,
                                                             SUBTILE_LEN // cfg.DIV_FACTOR_A,
                                                             1).load(layout=cfg.dot_layout_a)
        if cfg.TRANSPOSE_B:
            b = b_buffer.index(wmma_idx % cfg.NUM_BUFFERS).slice(subtile_start // cfg.DIV_FACTOR_B,
                                                                 SUBTILE_LEN // cfg.DIV_FACTOR_B,
                                                                 1).permute([1, 0]).load(layout=cfg.dot_layout_b)
        else:
            b = b_buffer.index(wmma_idx % cfg.NUM_BUFFERS).slice(subtile_start // cfg.DIV_FACTOR_B,
                                                                 SUBTILE_LEN // cfg.DIV_FACTOR_B,
                                                                 0).load(layout=cfg.dot_layout_b)
        if cfg.WITH_A_SCALE:
            a_scale_buffer_slice = a_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
        b_scale_buffer_slice = b_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
        if cfg.SCALE_PRESHUFFLE:
            if cfg.WITH_A_SCALE:
                a_scale_buffer_slice = a_scale_buffer_slice \
                    .reshape((cfg.BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE // cfg.SCALE_KWIDTH, cfg.PRESHUFFLE_FACTOR // 4, 4, cfg.SCALE_KWIDTH)) \
                    .permute((0, 3, 2, 1, 4)) \
                    .reshape((cfg.BLOCK_M, BLOCK_K_SCALE))
            b_scale_buffer_slice = b_scale_buffer_slice \
                .reshape((cfg.BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE // cfg.SCALE_KWIDTH, cfg.PRESHUFFLE_FACTOR // 4, 4, cfg.SCALE_KWIDTH)) \
                .permute((0, 3, 2, 1, 4)) \
                .reshape((cfg.BLOCK_N, BLOCK_K_SCALE))
        if cfg.WITH_A_SCALE:
            a_scale_buffer_slice = a_scale_buffer_slice.slice(subtile_start // cfg.SCALE_BLOCK, SUBTILE_LEN_SCALE, 1)
            scale_a = a_scale_buffer_slice.load(layout=cfg.layout_a_scale)
        else:
            scale_a = gl.constexpr(0)

        b_scale_buffer_slice = b_scale_buffer_slice.slice(subtile_start // cfg.SCALE_BLOCK, SUBTILE_LEN_SCALE, 1)
        scale_b = b_scale_buffer_slice.load(layout=cfg.layout_b_scale)

        return a, b, scale_a, scale_b

    @gluon.jit
    def issue_local_load_a(self, wmma_idx, a_buffer, a_scale_buffer):
        cfg = self.cfg
        NUM_SUBTILES_M: gl.constexpr = cfg.NUM_SUBTILES[0]
        NUM_SUBTILES_K: gl.constexpr = cfg.NUM_SUBTILES[2]
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK // NUM_SUBTILES_K
        a = a_buffer.index(wmma_idx % cfg.NUM_BUFFERS).load(layout=cfg.dot_layout_a)
        if cfg.WITH_A_SCALE:
            a_scale_buffer_slice = a_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
            if cfg.SCALE_PRESHUFFLE:
                a_scale_buffer_slice = a_scale_buffer_slice.reshape((
                    cfg.BLOCK_M_PRESHUFFLED // NUM_SUBTILES_M,  #
                    BLOCK_K_SCALE // cfg.SCALE_KWIDTH,  #
                    cfg.PRESHUFFLE_FACTOR // 4,  #
                    4,  #
                    cfg.SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((cfg.BLOCK_M // NUM_SUBTILES_M, BLOCK_K_SCALE))
            scale_a = a_scale_buffer_slice.load(layout=cfg.layout_a_scale)
        else:
            scale_a = gl.constexpr(0)
        return a, scale_a

    @gluon.jit
    def issue_local_load_b(self, wmma_idx, b_buffer, b_scale_buffer):
        cfg = self.cfg
        NUM_SUBTILES_N: gl.constexpr = cfg.NUM_SUBTILES[1]
        NUM_SUBTILES_K: gl.constexpr = cfg.NUM_SUBTILES[2]
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK // NUM_SUBTILES_K
        if cfg.TRANSPOSE_B:
            b = b_buffer.index(wmma_idx % cfg.NUM_BUFFERS).permute([1, 0]).load(layout=cfg.dot_layout_b)
        else:
            b = b_buffer.index(wmma_idx % cfg.NUM_BUFFERS).load(layout=cfg.dot_layout_b)
        b_scale_buffer_slice = b_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
        if cfg.SCALE_PRESHUFFLE:
            b_scale_buffer_slice = b_scale_buffer_slice.reshape((
                cfg.BLOCK_N_PRESHUFFLED // NUM_SUBTILES_N,  #
                BLOCK_K_SCALE // cfg.SCALE_KWIDTH,  #
                cfg.PRESHUFFLE_FACTOR // 4,  #
                4,  #
                cfg.SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((cfg.BLOCK_N // NUM_SUBTILES_N, BLOCK_K_SCALE))
        scale_b = b_scale_buffer_slice.load(layout=cfg.layout_b_scale)
        return b, scale_b

    @gluon.jit
    def issue_load_a(self, load_idx, a_buffer, a_scale_buffer, pred=1):
        cfg = self.cfg
        NUM_SUBTILES_K: gl.constexpr = cfg.NUM_SUBTILES[2]
        BLOCK_K: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_A // NUM_SUBTILES_K
        gl.amd.gfx1250.tdm.async_load(self.a_desc,  #
                                      [0, load_idx * BLOCK_K],  #
                                      a_buffer.index((load_idx // NUM_SUBTILES_K) % cfg.NUM_BUFFERS),  #
                                      pred=pred)
        if cfg.WITH_A_SCALE:
            a_scale_buffer_slice = a_scale_buffer.index((load_idx // NUM_SUBTILES_K) % cfg.NUM_BUFFERS)
            if cfg.ASYNC_COPY_SCALE:
                self.a_scale_desc.issue_async_load(load_idx, a_scale_buffer_slice, pred=pred)
            else:
                gl.amd.gfx1250.tdm.async_load(self.a_scale_desc,  #
                                              [0, load_idx * cfg.BLOCK_K_SCALE_PRESHUFFLED // NUM_SUBTILES_K],  #
                                              a_scale_buffer_slice,  #
                                              pred=pred)
        return load_idx + 1

    @gluon.jit
    def issue_load_b(self, load_idx, b_buffer, b_scale_buffer, pred=1):
        cfg = self.cfg
        NUM_SUBTILES_N: gl.constexpr = cfg.NUM_SUBTILES[1]
        NUM_SUBTILES_K: gl.constexpr = cfg.NUM_SUBTILES[2]
        NUM_SUBTILES_NK: gl.constexpr = cfg.NUM_SUBTILES[1] * cfg.NUM_SUBTILES[2]
        BLOCK_N: gl.constexpr = cfg.BLOCK_N // NUM_SUBTILES_N
        BLOCK_K: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_B // NUM_SUBTILES_K
        if cfg.TRANSPOSE_B:
            gl.amd.gfx1250.tdm.async_load(self.b_desc,  #
                                          [(load_idx % NUM_SUBTILES_N) * BLOCK_N,
                                           (load_idx // NUM_SUBTILES_N) * BLOCK_K],  #
                                          b_buffer.index((load_idx // NUM_SUBTILES_NK) % cfg.NUM_BUFFERS),  #
                                          pred=pred)
        else:
            gl.amd.gfx1250.tdm.async_load(self.b_desc,  #
                                          [(load_idx // NUM_SUBTILES_K) * BLOCK_K,
                                           (load_idx % NUM_SUBTILES_N) * BLOCK_N],  #
                                          b_buffer.index((load_idx // NUM_SUBTILES_NK) % cfg.NUM_BUFFERS),  #
                                          pred=pred)
        b_scale_buffer_slice = b_scale_buffer.index((load_idx // NUM_SUBTILES_NK) % cfg.NUM_BUFFERS)
        if cfg.ASYNC_COPY_SCALE:
            self.b_scale_desc.issue_async_load(load_idx, b_scale_buffer_slice, pred=pred)
        else:
            gl.amd.gfx1250.tdm.async_load(
                self.b_scale_desc,  #
                [(load_idx % NUM_SUBTILES_N) * (cfg.BLOCK_N_PRESHUFFLED // NUM_SUBTILES_N),  #
                 (load_idx // NUM_SUBTILES_N) * cfg.BLOCK_K_SCALE_PRESHUFFLED // NUM_SUBTILES_K],  #
                b_scale_buffer_slice,  #
                pred=pred)
        return load_idx + 1

    @gluon.jit
    def async_wait(self, waitcnt_a: int, waitcnt_b: int):
        cfg = self.cfg
        if cfg.ASYNC_COPY_SCALE:
            gl.amd.gfx1250.tdm.async_wait(int(waitcnt_a + waitcnt_b))
            cp.wait_group(waitcnt_b if cfg.WITH_A_SCALE else (waitcnt_a + waitcnt_b))
        else:
            gl.amd.gfx1250.tdm.async_wait((waitcnt_a + waitcnt_b) * 2 \
                                           if cfg.WITH_A_SCALE \
                                           else (waitcnt_a + waitcnt_b * 2))

    @gluon.jit
    def pipeline(self, K):
        cfg = self.cfg
        load_a_idx = 0
        load_b_idx = 0
        wmma_idx = 0

        # prologue
        # iter 0
        load_a_idx = self.issue_load_a(load_a_idx, self.a_buffer0, self.a_scale_buffer0)
        load_b_idx = self.issue_load_b(load_b_idx, self.b_buffer00, self.b_scale_buffer00)
        load_b_idx = self.issue_load_b(load_b_idx, self.b_buffer01, self.b_scale_buffer01)
        load_a_idx = self.issue_load_a(load_a_idx, self.a_buffer1, self.a_scale_buffer1)
        load_b_idx = self.issue_load_b(load_b_idx, self.b_buffer10, self.b_scale_buffer10)
        load_b_idx = self.issue_load_b(load_b_idx, self.b_buffer11, self.b_scale_buffer11)

        self.async_wait(1, 3)
        a0, scale_a0 = self.issue_local_load_a(wmma_idx, self.a_buffer0, self.a_scale_buffer0)
        b00, scale_b00 = self.issue_local_load_b(wmma_idx, self.b_buffer00, self.b_scale_buffer00)

        c0 = gl.zeros((cfg.BLOCK_M // cfg.NUM_SUBTILES[0], cfg.BLOCK_N // cfg.NUM_SUBTILES[1]), dtype=gl.float32,
                      layout=cfg.acc_layout)
        c1 = gl.zeros((cfg.BLOCK_M // cfg.NUM_SUBTILES[0], cfg.BLOCK_N // cfg.NUM_SUBTILES[1]), dtype=gl.float32,
                      layout=cfg.acc_layout)

        loop_ub = gl.cdiv(K, cfg.BLOCK_K)
        epilogue_lb = loop_ub - (cfg.NUM_BUFFERS - 1)
        gl.assume(loop_ub > 0)
        for i in range(0, loop_ub):
            pred = i - epilogue_lb
            pred = (pred >> 31) & 1

            # iter i + 1
            load_a_idx = self.issue_load_a(load_a_idx, self.a_buffer0, self.a_scale_buffer0, pred=pred)
            load_b_idx = self.issue_load_b(load_b_idx, self.b_buffer00, self.b_scale_buffer00, pred=pred)

            self.async_wait(2, 3)
            # iter i
            c0 = gl.amd.gfx1250.wmma_scaled(a0, scale_a0, cfg.DTYPE_A, b00, scale_b00, cfg.DTYPE_B, c0)
            b01, scale_b01 = self.issue_local_load_b(wmma_idx, self.b_buffer01, self.b_scale_buffer01)

            # iter i + 1
            load_b_idx = self.issue_load_b(load_b_idx, self.b_buffer01, self.b_scale_buffer01, pred=pred)
            self.async_wait(1, 3)
            # iter i
            c1 = gl.amd.gfx1250.wmma_scaled(a0, scale_a0, cfg.DTYPE_A, b01, scale_b01, cfg.DTYPE_B, c1)
            a1, scale_a1 = self.issue_local_load_a(wmma_idx, self.a_buffer1, self.a_scale_buffer1)
            b10, scale_b10 = self.issue_local_load_b(wmma_idx, self.b_buffer10, self.b_scale_buffer10)

            # iter i + 1
            load_a_idx = self.issue_load_a(load_a_idx, self.a_buffer1, self.a_scale_buffer1, pred=pred)
            load_b_idx = self.issue_load_b(load_b_idx, self.b_buffer10, self.b_scale_buffer10, pred=pred)

            self.async_wait(2, 3)
            # iter i
            c0 = gl.amd.gfx1250.wmma_scaled(a1, scale_a1, cfg.DTYPE_A, b10, scale_b10, cfg.DTYPE_B, c0)
            b11, scale_b11 = self.issue_local_load_b(wmma_idx, self.b_buffer11, self.b_scale_buffer11)

            # iter i + 1
            load_b_idx = self.issue_load_b(load_b_idx, self.b_buffer11, self.b_scale_buffer11, pred=pred)
            # iter i + 1
            self.async_wait(1, 3)
            wmma_idx += 1
            # iter i
            c1 = gl.amd.gfx1250.wmma_scaled(a1, scale_a1, cfg.DTYPE_A, b11, scale_b11, cfg.DTYPE_B, c1)
            a0, scale_a0 = self.issue_local_load_a(wmma_idx, self.a_buffer0, self.a_scale_buffer0)
            b00, scale_b00 = self.issue_local_load_b(wmma_idx, self.b_buffer00, self.b_scale_buffer00)

        accumulator = gl.join(c0, c1)
        accumulator = accumulator.permute(0, 2, 1).reshape((cfg.BLOCK_M, cfg.BLOCK_N))
        accumulator = gl.convert_layout(accumulator, cfg.acc_layout, assert_trivial=True)

        gl.amd.gfx1250.buffer_store(accumulator, self.c_ptr, self.c_offs, mask=self.c_mask)


@gluon.jit
def create_tensor_descriptor(cfg: MXFPGEMMConfig, a_ptr, a_offs, b_ptr, b_offs, a_scale_ptr, a_scale_offs, b_scale_ptr,
                             b_scale_offs, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_scale):
    SCALE_BLOCK: gl.constexpr = cfg.SCALE_BLOCK
    PRESHUFFLE_FACTOR: gl.constexpr = cfg.PRESHUFFLE_FACTOR
    NUM_SUBTILES_M: gl.constexpr = cfg.NUM_SUBTILES[0]
    NUM_SUBTILES_N: gl.constexpr = cfg.NUM_SUBTILES[1]
    NUM_SUBTILES_K: gl.constexpr = cfg.NUM_SUBTILES[2]
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + a_offs,  #
        shape=(M, K // cfg.DIV_FACTOR_A),  #
        strides=(stride_am, stride_ak),  #
        block_shape=(cfg.BLOCK_M // NUM_SUBTILES_M, cfg.BLOCK_K // cfg.DIV_FACTOR_A // NUM_SUBTILES_K),  #
        layout=cfg.shared_layout_a)

    if cfg.TRANSPOSE_B:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr + b_offs,  #
            shape=(N, K // cfg.DIV_FACTOR_B),  #
            strides=(stride_bn, stride_bk),  #
            block_shape=(cfg.BLOCK_N // NUM_SUBTILES_N, cfg.BLOCK_K // cfg.DIV_FACTOR_B // NUM_SUBTILES_K),  #
            layout=cfg.shared_layout_b)
    else:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr + b_offs,  #
            shape=(K // cfg.DIV_FACTOR_B, N),  #
            strides=(stride_bk, stride_bn),  #
            block_shape=(cfg.BLOCK_K // cfg.DIV_FACTOR_B // NUM_SUBTILES_K, cfg.BLOCK_N // NUM_SUBTILES_N),  #
            layout=cfg.shared_layout_b)

    if cfg.ASYNC_COPY_SCALE:
        if cfg.WITH_A_SCALE:
            a_scale_desc = ScaleAsyncCopyDescriptor.initialize(cfg, 0, a_scale_ptr, a_scale_offs, stride_scale,
                                                               cfg.shared_layout_a_scale)
        else:
            a_scale_desc = gl.constexpr(0)
        b_scale_desc = ScaleAsyncCopyDescriptor.initialize(cfg, 1, b_scale_ptr, b_scale_offs, stride_scale,
                                                           cfg.shared_layout_b_scale)
    else:
        if cfg.WITH_A_SCALE:
            a_scale_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=a_scale_ptr + a_scale_offs,  #
                shape=(M // PRESHUFFLE_FACTOR, K // SCALE_BLOCK * PRESHUFFLE_FACTOR),  #
                strides=(stride_scale, 1),  #
                block_shape=(cfg.BLOCK_M_PRESHUFFLED // NUM_SUBTILES_M,
                             cfg.BLOCK_K_SCALE_PRESHUFFLED // NUM_SUBTILES_K),  #
                layout=cfg.shared_layout_a_scale)
        else:
            # Use a placeholder to make compiler happy
            a_scale_desc = gl.constexpr(0)

        b_scale_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_scale_ptr + b_scale_offs,  #
            shape=(N // PRESHUFFLE_FACTOR, K // SCALE_BLOCK * PRESHUFFLE_FACTOR),  #
            strides=(stride_scale, 1),  #
            block_shape=(cfg.BLOCK_N_PRESHUFFLED // NUM_SUBTILES_N, cfg.BLOCK_K_SCALE_PRESHUFFLED // NUM_SUBTILES_K),  #
            layout=cfg.shared_layout_b_scale)

    return a_desc, b_desc, a_scale_desc, b_scale_desc


@gluon.jit
def mxgemm_tdm_pipelined_kernel(a_ptr, b_ptr, c_ptr, a_scale, b_scale, M, N, K, stride_am, stride_ak, stride_bk,
                                stride_bn, stride_cm, stride_cn, stride_scale, DTYPE_A: gl.constexpr,
                                DTYPE_B: gl.constexpr, SCALE_BLOCK: gl.constexpr, BLOCK_M: gl.constexpr,
                                BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
                                TRANSPOSE_B: gl.constexpr, NUM_BUFFERS: gl.constexpr, SCALE_PRESHUFFLE: gl.constexpr,
                                ASYNC_COPY_SCALE: gl.constexpr, WITH_A_SCALE: gl.constexpr, SCHEDULE: gl.constexpr,
                                NUM_WARPS: gl.constexpr, PINGPONG: gl.constexpr):

    if PINGPONG:
        gl.static_assert(NUM_WARPS == 8 and (SCHEDULE == 'baseline' or SCHEDULE == 'sliceK'))

    if SCHEDULE == 'sliceNK':
        NUM_SUBTILES: gl.constexpr = (1, 2, 2)
    elif SCHEDULE == 'sliceK':
        NUM_SUBTILES: gl.constexpr = (1, 1, 2)
    else:
        gl.static_assert(SCHEDULE == 'baseline')
        NUM_SUBTILES: gl.constexpr = (1, 1, 1)

    cfg = MXFPGEMMConfig(BLOCK_M, BLOCK_N, BLOCK_K, DTYPE_A, DTYPE_B, SCALE_BLOCK, NUM_BUFFERS, TRANSPOSE_B,
                         WITH_A_SCALE, SCALE_PRESHUFFLE, NUM_WARPS, ASYNC_COPY_SCALE, NUM_SUBTILES)

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_offs = pid_m * BLOCK_M * stride_am
    b_offs = pid_n * BLOCK_N * stride_bn
    a_scale_offs = pid_m * cfg.BLOCK_M_PRESHUFFLED * stride_scale
    b_scale_offs = pid_n * cfg.BLOCK_N_PRESHUFFLED * stride_scale
    a_desc, b_desc, a_scale_desc, b_scale_desc = create_tensor_descriptor(cfg, a_ptr, a_offs, b_ptr, b_offs, a_scale,
                                                                          a_scale_offs, b_scale, b_scale_offs, M, N, K,
                                                                          stride_am, stride_ak, stride_bk, stride_bn,
                                                                          stride_scale)

    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.acc_layout))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, cfg.acc_layout))
    c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if SCHEDULE == 'sliceNK':
        pgm = MXFPGEMMSliceNKProgram.initialize(cfg, a_desc, b_desc, a_scale_desc, b_scale_desc, c_ptr, c_offs, c_mask)
    elif SCHEDULE == 'sliceK':
        pgm = MXFPGEMMSliceKProgram.initialize(cfg, a_desc, b_desc, a_scale_desc, b_scale_desc, c_ptr, c_offs, c_mask)
    else:
        gl.static_assert(SCHEDULE == 'baseline')
        pgm = MXFPGEMMPipelinedProgram.initialize(cfg, a_desc, b_desc, a_scale_desc, b_scale_desc, c_ptr, c_offs,
                                                  c_mask)

    if PINGPONG:
        pgm.warp_pipeline(K)
    else:
        pgm.pipeline(K)


def torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, M, N, K):
    if a_scale is None:
        a_scale_f32 = torch.full((M, K), 1.0, dtype=torch.float32)
    else:
        a_scale_f32 = a_scale.to(torch.float32).repeat_interleave(scale_block, dim=1)[:M, :K]
    b_scale_f32 = b_scale.to(torch.float32).repeat_interleave(scale_block, dim=1).T.contiguous()[:K, :N]

    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)

    return torch.matmul(a_f32 * a_scale_f32, b_f32 * b_scale_f32).to(torch.float32)


def init_data(dtype, d0: int, d1: int):
    if dtype == 'float4':
        return MXFP4Tensor(size=(d0, d1)).random()
    elif dtype == "float8_e5m2":
        return torch.randint(20, 40, (d0, d1), dtype=torch.uint8).view(torch.float8_e5m2)
    elif dtype == "float8_e4m3":
        return torch.randint(20, 40, (d0, d1), dtype=torch.uint8).view(torch.float8_e4m3fn)
    else:
        raise NotImplementedError(f"NYI: unsupported dtype: {dtype}")


def pack_scale(x):
    if x is None:
        return x
    NON_K, K_SCALE = x.shape
    preshuffle_factor = 128
    num_chunk_m = NON_K // preshuffle_factor
    SCALE_KWIDTH = 4 if K_SCALE >= 4 else K_SCALE
    num_chunk_k = K_SCALE // SCALE_KWIDTH

    x = x.view(num_chunk_m, 4, preshuffle_factor // 4, num_chunk_k, SCALE_KWIDTH)
    x = x.permute(0, 3, 2, 1, 4).contiguous()
    return x.view(NON_K // preshuffle_factor, K_SCALE * preshuffle_factor)


@pytest.mark.parametrize(
    "DTYPE_A, DTYPE_B",
    [['float8_e5m2', 'float4'], ['float4', 'float8_e4m3'], ['float8_e4m3', 'float8_e5m2'], ['float4', 'float4']])
@pytest.mark.parametrize("M,N,K", [(512, 512, 512)])
@pytest.mark.parametrize("SCHEDULE,BLOCK_M,BLOCK_N,BLOCK_K", [('baseline', 256, 256, 128), ('sliceK', 128, 256, 256),
                                                              ('sliceK', 128, 128, 256), ('sliceNK', 128, 256, 256)])
@pytest.mark.parametrize("TRANSPOSE_B", [True])
@pytest.mark.parametrize("SCALE_PRESHUFFLE", [True])
@pytest.mark.parametrize("WITH_A_SCALE", [True, False])
@pytest.mark.parametrize("ASYNC_COPY_SCALE", [False])
@pytest.mark.parametrize("NUM_BUFFERS", [2, 3])
@pytest.mark.parametrize("GROUP_SIZE_M", [8])
@pytest.mark.parametrize("PINGPONG", [True, False])
def test_runtime_mxgemm_tdm_8warps_pipeline(DTYPE_A, DTYPE_B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B,
                                            NUM_BUFFERS, SCALE_PRESHUFFLE, WITH_A_SCALE, SCHEDULE, ASYNC_COPY_SCALE,
                                            GROUP_SIZE_M, PINGPONG):
    SCALE_BLOCK = 32
    numWarps = 8
    numCtas = 1

    torch.manual_seed(0)

    def is_fp8(dtype):
        return dtype in ['float8_e5m2', 'float8_e4m3']

    if SCHEDULE == 'sliceK' and NUM_BUFFERS == 3:
        problem_size = BLOCK_M * BLOCK_N * BLOCK_K
        if is_fp8(DTYPE_A) and is_fp8(DTYPE_B):
            if problem_size >= 128 * 256 * 256:
                pytest.skip(
                    'Large block size will exceed lds limit with fp8 inputs and 3 buffers. Please use 128x128x256')

    if SCHEDULE == 'sliceNK' and (PINGPONG or NUM_BUFFERS == 3):
        pytest.skip('NYI: Skipping pingpong or 3 buffers in sliceNK schedule')

    if PINGPONG and NUM_BUFFERS != 3:
        pytest.skip('Pingpong only supports 3 buffers')

    a = init_data(DTYPE_A, M, K)
    b = init_data(DTYPE_B, K, N)
    a_scale_size = (M, (K + SCALE_BLOCK - 1) // SCALE_BLOCK)
    b_scale_size = (N, (K + SCALE_BLOCK - 1) // SCALE_BLOCK)
    if WITH_A_SCALE:
        a_scale = MXScaleTensor(size=a_scale_size).random(low=1.0, high=32.0)
    else:
        a_scale = None
    b_scale = MXScaleTensor(size=b_scale_size).random(low=1.0, high=32.0)

    c_ref = torch_gemm_mxfp(a, b, a_scale, b_scale, SCALE_BLOCK, M, N, K)

    if WITH_A_SCALE:
        a_scale = a_scale.data
    b_scale = b_scale.data

    if SCALE_PRESHUFFLE:
        a_scale = pack_scale(a_scale)
        b_scale = pack_scale(b_scale)

    # mxfp4 input needs packed along the k dim, i.e., two mxfp4 are packed in one uint8
    if DTYPE_A in ['float4', 'float6_e2m3', 'float6_e3m2']:
        a = a.to_packed_tensor(dim=1)
    if DTYPE_B in ['float4', 'float6_e2m3', 'float6_e3m2']:
        b = b.to_packed_tensor(dim=0)

    c_d = torch.zeros(M, N, dtype=torch.float32).cuda()
    a_d = a.data.contiguous().cuda()
    if TRANSPOSE_B:
        b_d = b.data.T.contiguous().cuda()
    else:
        b_d = b.data.contiguous().cuda()
    if WITH_A_SCALE:
        a_scale_d = a_scale.cuda()
    else:
        a_scale_d = None
    b_scale_d = b_scale.cuda()

    stride_am, stride_ak = a_d.stride(0), a_d.stride(1)
    if TRANSPOSE_B:
        stride_bk, stride_bn = b_d.stride(1), b_d.stride(0)
    else:
        stride_bk, stride_bn = b_d.stride(0), b_d.stride(1)
    stride_cm, stride_cn = c_d.stride(0), c_d.stride(1)
    stride_scale = b_scale_d.stride(0)

    numBlocks = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = [numBlocks, 1, 1]

    dtype_converter = {'float8_e5m2': "e5m2", "float8_e4m3": "e4m3", "float4": "e2m1"}

    k = mxgemm_tdm_pipelined_kernel[grid](a_d, b_d, c_d, a_scale_d, b_scale_d, M, N, K, stride_am, stride_ak, stride_bk,
                                          stride_bn, stride_cm, stride_cn, stride_scale, dtype_converter[DTYPE_A],
                                          dtype_converter[DTYPE_B], SCALE_BLOCK, BLOCK_M, BLOCK_N, BLOCK_K,
                                          GROUP_SIZE_M, TRANSPOSE_B, NUM_BUFFERS, SCALE_PRESHUFFLE, ASYNC_COPY_SCALE,
                                          WITH_A_SCALE, SCHEDULE, numWarps, PINGPONG, num_warps=numWarps,
                                          num_ctas=numCtas, waves_per_eu=(numWarps // 4))
    static_profile(k)

    if TRANSPOSE_B:
        assert 'ds_load_u8' not in k.asm['amdgcn']

    torch.testing.assert_close(c_d.cpu(), c_ref.cpu(), rtol=1e-5, atol=1e-8)
    print('✅Pass')


@pytest.mark.parametrize(
    "DTYPE_A, DTYPE_B",
    [['float8_e5m2', 'float4'], ['float4', 'float8_e4m3'], ['float8_e4m3', 'float8_e5m2'], ['float4', 'float4']])
@pytest.mark.parametrize("M,N,K", [(128, 128, 128), (256, 256, 512)])
@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(64, 64, 64), (128, 128, 128), (256, 256, 256)])
@pytest.mark.parametrize("TRANSPOSE_B", [True, False])
@pytest.mark.parametrize("NUM_BUFFERS", [2, 4])
@pytest.mark.parametrize("SCALE_PRESHUFFLE", [True, False])
@pytest.mark.parametrize("WITH_A_SCALE", [True, False])
@pytest.mark.parametrize("SCHEDULE", ['sliceNK', 'sliceK', 'baseline'])
@pytest.mark.parametrize("ASYNC_COPY_SCALE", [True, False])
@pytest.mark.parametrize("GROUP_SIZE_M", [8])
def test_runtime_mxgemm_tdm_pipelined(DTYPE_A, DTYPE_B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, NUM_BUFFERS,
                                      SCALE_PRESHUFFLE, WITH_A_SCALE, SCHEDULE, ASYNC_COPY_SCALE, GROUP_SIZE_M):
    SCALE_BLOCK = 32
    numWarps = 4
    numCtas = 1

    if SCALE_PRESHUFFLE:
        if BLOCK_M < 128 or BLOCK_N < 128 or (SCHEDULE != "baseline" and BLOCK_K < 256):
            pytest.skip("Skipping block sizes too small for preshuffling")

    if not WITH_A_SCALE and DTYPE_A == "float4":
        pytest.skip("Skip fp4 x mxfp gemm to reduce test cases.")

    if SCHEDULE != 'baseline' and not (SCALE_PRESHUFFLE and TRANSPOSE_B):
        pytest.skip('Only test with SCALE_PRESHUFFLE and TRANSPOSE_B in sliceK and sliceNK schedules')

    if NUM_BUFFERS == 4 and BLOCK_M >= 256:
        pytest.skip("Large block size with 4 buffers will exceed lds limit")

    if SCHEDULE == 'sliceNK':
        if BLOCK_K < 256 or BLOCK_N < 256:
            pytest.skip('BLOCK_K and BLOCK_N are too small for sliceNK schedule')
        if M < 256:
            pytest.skip('Skip small problem size to reduce test cases')
    else:
        if ASYNC_COPY_SCALE:
            pytest.skip('Only use ASYNC_COPY_SCALE in sliceNK schedule')

    if SCHEDULE != 'baseline' and NUM_BUFFERS != 2:
        pytest.skip('Only test 2 buffers in sliceK and sliceNK schedules')

    if ASYNC_COPY_SCALE and (M < BLOCK_M or N < BLOCK_N or K < BLOCK_K):
        pytest.skip('NYI: Skipping small problem sizes for async copy scale')

    torch.manual_seed(0)

    a = init_data(DTYPE_A, M, K)
    b = init_data(DTYPE_B, K, N)
    a_scale_size = (M, (K + SCALE_BLOCK - 1) // SCALE_BLOCK)
    b_scale_size = (N, (K + SCALE_BLOCK - 1) // SCALE_BLOCK)
    if WITH_A_SCALE:
        a_scale = MXScaleTensor(size=a_scale_size).random(low=1.0, high=32.0)
    else:
        a_scale = None
    b_scale = MXScaleTensor(size=b_scale_size).random(low=1.0, high=32.0)

    c_ref = torch_gemm_mxfp(a, b, a_scale, b_scale, SCALE_BLOCK, M, N, K)

    if WITH_A_SCALE:
        a_scale = a_scale.data
    b_scale = b_scale.data

    if SCALE_PRESHUFFLE:
        a_scale = pack_scale(a_scale)
        b_scale = pack_scale(b_scale)

    # mxfp4 input needs packed along the k dim, i.e., two mxfp4 are packed in one uint8
    if DTYPE_A in ['float4', 'float6_e2m3', 'float6_e3m2']:
        a = a.to_packed_tensor(dim=1)
    if DTYPE_B in ['float4', 'float6_e2m3', 'float6_e3m2']:
        b = b.to_packed_tensor(dim=0)

    c_d = torch.zeros(M, N, dtype=torch.float32).cuda()
    a_d = a.data.contiguous().cuda()
    if TRANSPOSE_B:
        b_d = b.data.T.contiguous().cuda()
    else:
        b_d = b.data.contiguous().cuda()
    if WITH_A_SCALE:
        a_scale_d = a_scale.cuda()
    else:
        a_scale_d = None
    b_scale_d = b_scale.cuda()

    stride_am, stride_ak = a_d.stride(0), a_d.stride(1)
    if TRANSPOSE_B:
        stride_bk, stride_bn = b_d.stride(1), b_d.stride(0)
    else:
        stride_bk, stride_bn = b_d.stride(0), b_d.stride(1)
    stride_cm, stride_cn = c_d.stride(0), c_d.stride(1)
    stride_scale = b_scale_d.stride(0)

    numBlocks = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = [numBlocks, 1, 1]

    dtype_converter = {'float8_e5m2': "e5m2", "float8_e4m3": "e4m3", "float4": "e2m1"}

    k = mxgemm_tdm_pipelined_kernel[grid](a_d, b_d, c_d, a_scale_d, b_scale_d, M, N, K, stride_am, stride_ak, stride_bk,
                                          stride_bn, stride_cm, stride_cn, stride_scale, dtype_converter[DTYPE_A],
                                          dtype_converter[DTYPE_B], SCALE_BLOCK, BLOCK_M, BLOCK_N, BLOCK_K,
                                          GROUP_SIZE_M, TRANSPOSE_B, NUM_BUFFERS, SCALE_PRESHUFFLE, ASYNC_COPY_SCALE,
                                          WITH_A_SCALE, SCHEDULE, NUM_WARPS=numWarps, PINGPONG=False,
                                          num_warps=numWarps, num_ctas=numCtas, waves_per_eu=numWarps // 4)
    static_profile(k)

    if TRANSPOSE_B:
        assert 'ds_load_u8' not in k.asm['amdgcn']

    torch.testing.assert_close(c_d.cpu(), c_ref.cpu(), rtol=1e-5, atol=1e-8)
    print('✅Pass')


if __name__ == '__main__':
    import argparse

    supported_dtypes = ['float8_e4m3', 'float8_e5m2', 'float4']

    parser = argparse.ArgumentParser()
    parser.add_argument('-M', type=int, default=8192, help='problem M size')
    parser.add_argument('-N', type=int, default=8192, help='problem N size')
    parser.add_argument('-K', type=int, default=1024, help='problem K size')
    parser.add_argument('-BM', type=int, default=256, help='BLOCK_M')
    parser.add_argument('-BN', type=int, default=256, help='BLOCK_N')
    parser.add_argument('-BK', type=int, default=128, help='BLOCK_K')
    parser.add_argument('--num_warps', type=int, default=4, choices=[4, 8])
    parser.add_argument('--num_buffers', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--group_size_m', type=int, default=8, choices=[1, 2, 4, 8])
    parser.add_argument('--scale_preshuffled', action='store_true')
    parser.add_argument('--with_a_scale', action='store_true')
    parser.add_argument('--async_copy_scale', action='store_true')
    parser.add_argument('--schedule', type=str, choices=['sliceNK', 'sliceK', 'baseline'], default='sliceNK')
    parser.add_argument('--dtype_a', type=str, default='float8_e4m3', choices=supported_dtypes)
    parser.add_argument('--dtype_b', type=str, default='float8_e4m3', choices=supported_dtypes)
    parser.add_argument('--pingpong', action='store_true')

    args = parser.parse_args()

    if args.pingpong:
        assert (args.num_warps == 8 and (args.schedule == 'baseline' or args.schedule == 'sliceK'))

    if args.num_warps == 8:
        assert (args.num_buffers == 3 and not args.async_copy_scale)
        test_runtime_mxgemm_tdm_8warps_pipeline(args.dtype_a, args.dtype_b,  #
                                                args.M, args.N, args.K,  #
                                                args.BM, args.BN, args.BK,  #
                                                TRANSPOSE_B=True,  #
                                                NUM_BUFFERS=args.num_buffers,  #
                                                SCALE_PRESHUFFLE=args.scale_preshuffled,  #
                                                WITH_A_SCALE=args.with_a_scale,  #
                                                SCHEDULE=args.schedule,  #
                                                ASYNC_COPY_SCALE=False,  #
                                                GROUP_SIZE_M=args.group_size_m,  #
                                                PINGPONG=args.pingpong)
    else:
        assert (args.num_buffers in (2, 4))
        test_runtime_mxgemm_tdm_pipelined(args.dtype_a, args.dtype_b,  #
                                          args.M, args.N, args.K,  #
                                          args.BM, args.BN, args.BK,  #
                                          TRANSPOSE_B=True,  #
                                          NUM_BUFFERS=args.num_buffers,  #
                                          SCALE_PRESHUFFLE=args.scale_preshuffled,  #
                                          WITH_A_SCALE=args.with_a_scale,  #
                                          SCHEDULE=args.schedule,  #
                                          ASYNC_COPY_SCALE=args.async_copy_scale,  #
                                          GROUP_SIZE_M=args.group_size_m)
