import torch
import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
import triton.experimental.gluon.language.nvidia.blackwell as blackwell
import triton.experimental.gluon.language.nvidia.blackwell.tma as tma
from triton.experimental.gluon.language.nvidia.blackwell import float2
import triton.experimental.gluon.language.nvidia.hopper.mbarrier as mbarrier
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.tensor import (
    FP4,
    convert_layout,
    wrap_torch_tensor,
)
from triton_kernels.tensor_details.layout import (
    BlackwellMX4ValuePackedShuffledLayout,
    make_default_matmul_mxfp4_w_scale_layout,
)
from triton_kernels.testing import alloc_rand

# ===-----------------------------------------------------------------------===#
# Device Code
# ===-----------------------------------------------------------------------===#

REDUCTION_N_VALUE = 2
REDUCTION_N = gl.constexpr(REDUCTION_N_VALUE)
BLOCK_M_VALUE = 128
BLOCK_M = gl.constexpr(BLOCK_M_VALUE)
BLOCK_N_VALUE = 256
BLOCK_N = gl.constexpr(BLOCK_N_VALUE)
BLOCK_K_VALUE = 128
BLOCK_K = gl.constexpr(BLOCK_K_VALUE)
K_TILES_VALUE = 3
K_TILES = gl.constexpr(K_TILES_VALUE)
OUT_PACKED_N_VALUE = 64
OUT_PACKED_N = gl.constexpr(OUT_PACKED_N_VALUE)
X_NUM_BUFS_VALUE = 4
X_NUM_BUFS = gl.constexpr(X_NUM_BUFS_VALUE)
W_NUM_BUFS_VALUE = 2
W_NUM_BUFS = gl.constexpr(W_NUM_BUFS_VALUE)
W_SCALE_NUM_BUFS_VALUE = 4
W_SCALE_NUM_BUFS = gl.constexpr(W_SCALE_NUM_BUFS_VALUE)
ACC_NUM_BUFS_VALUE = 1
ACC_NUM_BUFS = gl.constexpr(ACC_NUM_BUFS_VALUE)
NUM_WARPS_VALUE = 8
LOAD_ACTIVATION_WARPS_VALUE = 4
LOAD_ACTIVATION_WARPS = gl.constexpr(LOAD_ACTIVATION_WARPS_VALUE)
LOAD_WEIGHT_WARPS_VALUE = 1
LOAD_WEIGHT_WARPS = gl.constexpr(LOAD_WEIGHT_WARPS_VALUE)
LOAD_WEIGHT_SCALES_WARPS_VALUE = 1
LOAD_WEIGHT_SCALES_WARPS = gl.constexpr(LOAD_WEIGHT_SCALES_WARPS_VALUE)
REPLAY_WARPS_VALUE = 4
REPLAY_WARPS = gl.constexpr(REPLAY_WARPS_VALUE)
MMA_WARPS_VALUE = 4
MMA_WARPS = gl.constexpr(MMA_WARPS_VALUE)
LOAD_ACTIVATION_REGS_VALUE = 32
LOAD_ACTIVATION_REGS = gl.constexpr(LOAD_ACTIVATION_REGS_VALUE)
LOAD_WEIGHT_REGS_VALUE = 24
LOAD_WEIGHT_REGS = gl.constexpr(LOAD_WEIGHT_REGS_VALUE)
LOAD_WEIGHT_SCALES_REGS_VALUE = 24
LOAD_WEIGHT_SCALES_REGS = gl.constexpr(LOAD_WEIGHT_SCALES_REGS_VALUE)
REPLAY_REGS_VALUE = 80
REPLAY_REGS = gl.constexpr(REPLAY_REGS_VALUE)
MMA_REGS_VALUE = 80
MMA_REGS = gl.constexpr(MMA_REGS_VALUE)
MXFP_BLOCK_SIZE_VALUE = 32
SCALE_SIZE_OUTER_VALUE = 128
SCALE_SIZE_OUTER = gl.constexpr(SCALE_SIZE_OUTER_VALUE)
SCALE_SIZE_INNER_VALUE = 4
SCALE_SIZE_INNER = gl.constexpr(SCALE_SIZE_INNER_VALUE)
MMA_BLOCK_COL_VALUE = 128
MMA_BLOCK_COL = gl.constexpr(MMA_BLOCK_COL_VALUE)
MXFP_BLOCK_SIZE = gl.constexpr(MXFP_BLOCK_SIZE_VALUE)


@gluon.jit
def advance(idx: gl.tensor, phase: gl.tensor, num_bufs: gl.constexpr) -> tuple[gl.tensor, gl.tensor]:
    next_idx = idx + 1
    wrap = next_idx == num_bufs
    return gl.where(wrap, 0, next_idx), gl.where(wrap, phase ^ 1, phase)


@gluon.jit
def unswizzle_mx_scale(
    smem,
    SIZE_OUTER: gl.constexpr,
    SIZE_INNER: gl.constexpr,
    MXFP_BLOCK_SIZE: gl.constexpr,
):
    rows: gl.constexpr = smem.shape[1]
    cols: gl.constexpr = smem.shape[2] * smem.shape[3] * smem.shape[4]
    tiles: gl.constexpr = cols // (SIZE_OUTER * SIZE_INNER)
    smem = smem.reshape((rows, tiles, MXFP_BLOCK_SIZE, SIZE_OUTER // MXFP_BLOCK_SIZE, SIZE_INNER))
    smem = smem.permute((0, 3, 2, 1, 4))
    return smem.reshape((rows * SIZE_OUTER, cols // SIZE_OUTER))


@gluon.jit
def alloc_barrier_ring(num_bufs: gl.constexpr, count: gl.constexpr = 1):
    bars = mbarrier.allocate_mbarrier(batch=num_bufs)
    for i in gl.static_range(num_bufs):
        mbarrier.init(bars.index(i), count=count)
    return bars


@gluon.jit
def alloc_ring_barriers(num_bufs: gl.constexpr, consumer_count: gl.constexpr = 1):
    return (
        alloc_barrier_ring(num_bufs, count=consumer_count),
        alloc_barrier_ring(num_bufs),
    )


@gluon.jit
def pack_e4m3x2(values):
    return gl.inline_asm_elementwise(
        """
        {
            .reg .f32 lane<2>;
            mov.b64 {lane0, lane1}, $1;
            cvt.rn.satfinite.e4m3x2.f32 $0, lane1, lane0;
        }
        """,
        "=h,l",
        [values.value],
        dtype=gl.int16,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def pack_u16x2(x0, x1):
    return gl.inline_asm_elementwise(
        """
        mov.b32 $0, { $1, $2 };
        """,
        "=r,h,h",
        [x0, x1],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def fp4_prmt_shuffle_elements(x):
    return gl.inline_asm_elementwise(
        asm="""
        {
          .reg .b32 lo;
          .reg .b32 hi;
          and.b32 lo, $2, 0x0f0f0f0f;
          and.b32 hi, $2, 0xf0f0f0f0;
          shl.b32 lo, lo, 2;
          shr.u32 hi, hi, 2;
          prmt.b32 $0, lo, hi, 0x5140;
          prmt.b32 $1, lo, hi, 0x7362;
        }
        """,
        constraints="=r,=r,r",
        args=[x],
        dtype=(gl.uint32, gl.uint32),
        is_pure=True,
        pack=1,
    )


@gluon.jit
def pack_fp8x4(values):
    lhs, rhs = gl.split(values.reshape((values.shape[0], values.shape[1] // 2, 2)))
    return pack_u16x2(lhs, rhs)


@gluon.aggregate
class PartitionArgs:
    x_desc: tma.tensor_descriptor
    w_desc: tma.tensor_descriptor
    scale_desc: tma.tensor_descriptor

    out_ptr: gl.tensor

    x_bufs: gl.shared_memory_descriptor
    x_empty_bars: gl.shared_memory_descriptor
    x_ready_bars: gl.shared_memory_descriptor

    w_bufs: gl.shared_memory_descriptor
    w_empty_bars: gl.shared_memory_descriptor
    w_ready_bars: gl.shared_memory_descriptor

    w_scale_bufs: gl.shared_memory_descriptor
    w_scale_empty_bars: gl.shared_memory_descriptor
    w_scale_ready_bars: gl.shared_memory_descriptor

    x_scale_tmem: blackwell.tensor_memory_descriptor
    w_scale_tmem: blackwell.tensor_memory_descriptor
    dense_replay_tmem: blackwell.tensor_memory_descriptor
    replay_tmem: blackwell.tensor_memory_descriptor
    replay_empty_bars: gl.shared_memory_descriptor
    replay_full_bars: gl.shared_memory_descriptor
    acc_bufs: blackwell.tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    dense_copy_done_bar: gl.shared_memory_descriptor
    unpack_sync_bar: gl.shared_memory_descriptor


@gluon.jit
def load_activations(p: PartitionArgs):
    offs_layout: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]),
    )
    tile_x_bytes: gl.constexpr = p.x_desc.block_type.nbytes * BLOCK_M

    idx = 0
    phase = 1
    issued = 0

    offs_m = gl.arange(0, BLOCK_M, layout=offs_layout)

    mask_m = offs_m < 1
    offs_x_m = gl.where(mask_m, offs_m, p.x_desc.shape[0])

    for ki in range(K_TILES):
        off_k_x = ki * BLOCK_K
        empty_bar = p.x_empty_bars.index(idx)
        ready_bar = p.x_ready_bars.index(idx)
        x_buf = p.x_bufs.index(idx)

        mbarrier.wait(empty_bar, phase, pred=issued >= X_NUM_BUFS)
        mbarrier.expect(ready_bar, tile_x_bytes)
        tma.async_gather(
            p.x_desc,
            offs_x_m,
            off_k_x,
            ready_bar,
            x_buf,
        )

        idx, phase = advance(idx, phase, X_NUM_BUFS)
        issued += 1


@gluon.jit
def load_weights(p: PartitionArgs):
    TILE_W_BYTES: gl.constexpr = p.w_desc.nbytes_per_cta

    idx = 0
    phase = 1
    issued = 0

    pid_n = gl.program_id(0)

    for ki in range(0, K_TILES, 2):
        w_empty_bar = p.w_empty_bars.index(idx)
        w_ready_bar = p.w_ready_bars.index(idx)
        w_buf = p.w_bufs.index(idx)

        mbarrier.wait(w_empty_bar, phase, pred=issued >= W_NUM_BUFS)
        mbarrier.expect(w_ready_bar, TILE_W_BYTES)
        tma.async_copy_global_to_shared(
            p.w_desc,
            [0, ki // 2, pid_n, 0, 0],
            w_ready_bar,
            w_buf,
        )

        idx, phase = advance(idx, phase, W_NUM_BUFS)
        issued += 1


@gluon.jit
def load_weight_scales(p: PartitionArgs):
    TILE_SCALE_BYTES: gl.constexpr = p.scale_desc.nbytes_per_cta

    idx = 0
    phase = 1
    issued = 0

    scale_idx = gl.program_id(0) * (BLOCK_N // SCALE_SIZE_OUTER)
    for ki in range(K_TILES):
        off_k_scale = ki * BLOCK_K // (MXFP_BLOCK_SIZE * SCALE_SIZE_INNER)

        scale_empty_bar = p.w_scale_empty_bars.index(idx)
        scale_ready_bar = p.w_scale_ready_bars.index(idx)
        scale_buf = p.w_scale_bufs.index(idx)

        mbarrier.wait(scale_empty_bar, phase, pred=issued >= W_SCALE_NUM_BUFS)
        mbarrier.expect(scale_ready_bar, TILE_SCALE_BYTES)
        tma.async_copy_global_to_shared(
            p.scale_desc,
            [0, scale_idx, off_k_scale, 0, 0],
            scale_ready_bar,
            scale_buf,
        )

        idx, phase = advance(idx, phase, W_SCALE_NUM_BUFS)
        issued += 1


@gluon.jit
def mma_partition(p: PartitionArgs):
    # Consumers.
    x_idx = 0
    x_phase = 0
    w_idx = 0
    w_phase = 0
    scale_idx = 0
    scale_phase = 0
    replay_idx = 0
    # Producer.
    mma_idx = 0
    mma_phase = 1
    replay_full_phase = 0
    unpack_sync_phase = 0

    k_second_u8_layout: gl.constexpr = blackwell.TensorMemoryLayout(
        (MMA_BLOCK_COL, BLOCK_K),
        col_stride=1,
        fp4_padded=True,
    )

    acc_empty_bar = p.acc_empty_bars.index(mma_idx)
    acc_ready_bar = p.acc_ready_bars.index(mma_idx)
    acc_buf = p.acc_bufs.index(mma_idx)
    mbarrier.wait(acc_empty_bar, mma_phase)

    use_acc = False
    for _ in range(K_TILES // 2):
        scale_ready_bar = p.w_scale_ready_bars.index(scale_idx)
        scale_empty_bar = p.w_scale_empty_bars.index(scale_idx)
        scale_buf = p.w_scale_bufs.index(scale_idx)
        mbarrier.wait(scale_ready_bar, scale_phase)
        blackwell.tcgen05_copy(
            unswizzle_mx_scale(scale_buf, SCALE_SIZE_OUTER, SCALE_SIZE_INNER, MXFP_BLOCK_SIZE),
            p.w_scale_tmem,
        )

        x_ready_bar = p.x_ready_bars.index(x_idx)
        x_empty_bar = p.x_empty_bars.index(x_idx)
        x_buf = p.x_bufs.index(x_idx)
        mbarrier.wait(x_ready_bar, x_phase)

        w_ready_bar = p.w_ready_bars.index(w_idx)
        w_empty_bar = p.w_empty_bars.index(w_idx)
        w_buf = p.w_bufs.index(w_idx)
        mbarrier.wait(w_ready_bar, w_phase)

        w_packed_pair = w_buf.reshape((BLOCK_N, BLOCK_K))
        fp4_padded_layout: gl.constexpr = gl.NVMMASharedLayout(
            swizzle_byte_width=128,
            element_bitwidth=8,
            rank=2,
            fp4_padded=True,
        )
        w_pair_first = w_packed_pair._reinterpret(
            gl.uint8,
            (BLOCK_N, BLOCK_K // 2),
            fp4_padded_layout,
        )

        blackwell.tcgen05_mma_scaled(
            w_pair_first,
            x_buf.permute((1, 0)),
            acc_buf,
            p.w_scale_tmem,
            p.x_scale_tmem,
            a_type="e2m1",
            b_type="e4m3",
            use_acc=use_acc,
            mbarriers=[x_empty_bar, scale_empty_bar, w_empty_bar],
        )
        use_acc = True

        replay_full_bar = p.replay_full_bars.index(replay_idx)
        mbarrier.wait(replay_full_bar, replay_full_phase)
        mbarrier.arrive(p.unpack_sync_bar)
        mbarrier.wait(p.unpack_sync_bar, unpack_sync_phase)
        unpack_sync_phase = unpack_sync_phase ^ 1

        w_idx, w_phase = advance(w_idx, w_phase, W_NUM_BUFS)
        scale_idx, scale_phase = advance(scale_idx, scale_phase, W_SCALE_NUM_BUFS)
        x_idx, x_phase = advance(x_idx, x_phase, X_NUM_BUFS)

        scale_ready_bar = p.w_scale_ready_bars.index(scale_idx)
        scale_empty_bar = p.w_scale_empty_bars.index(scale_idx)
        scale_buf = p.w_scale_bufs.index(scale_idx)
        mbarrier.wait(scale_ready_bar, scale_phase)
        blackwell.tcgen05_copy(
            unswizzle_mx_scale(scale_buf, SCALE_SIZE_OUTER, SCALE_SIZE_INNER, MXFP_BLOCK_SIZE),
            p.w_scale_tmem,
        )

        x_ready_bar = p.x_ready_bars.index(x_idx)
        x_empty_bar = p.x_empty_bars.index(x_idx)
        x_buf = p.x_bufs.index(x_idx)
        mbarrier.wait(x_ready_bar, x_phase)

        replay_tmem = p.replay_tmem.index(replay_idx)
        replay_empty_bar = p.replay_empty_bars.index(replay_idx)
        blackwell.tcgen05_mma_scaled(
            replay_tmem._reinterpret(gl.uint8, (BLOCK_N, BLOCK_K), k_second_u8_layout),
            x_buf.permute((1, 0)),
            acc_buf,
            p.w_scale_tmem,
            p.x_scale_tmem,
            a_type="e2m1",
            b_type="e4m3",
            use_acc=use_acc,
            mbarriers=[x_empty_bar, scale_empty_bar, replay_empty_bar],
        )

        replay_idx, replay_full_phase = advance(replay_idx, replay_full_phase, 2)
        x_idx, x_phase = advance(x_idx, x_phase, X_NUM_BUFS)
        scale_idx, scale_phase = advance(scale_idx, scale_phase, W_SCALE_NUM_BUFS)

    scale_ready_bar = p.w_scale_ready_bars.index(scale_idx)
    scale_empty_bar = p.w_scale_empty_bars.index(scale_idx)
    scale_buf = p.w_scale_bufs.index(scale_idx)
    mbarrier.wait(scale_ready_bar, scale_phase)
    blackwell.tcgen05_copy(
        unswizzle_mx_scale(scale_buf, SCALE_SIZE_OUTER, SCALE_SIZE_INNER, MXFP_BLOCK_SIZE),
        p.w_scale_tmem,
    )

    x_ready_bar = p.x_ready_bars.index(x_idx)
    x_empty_bar = p.x_empty_bars.index(x_idx)
    x_buf = p.x_bufs.index(x_idx)
    mbarrier.wait(x_ready_bar, x_phase)

    w_ready_bar = p.w_ready_bars.index(w_idx)
    w_empty_bar = p.w_empty_bars.index(w_idx)
    w_buf = p.w_bufs.index(w_idx)
    mbarrier.wait(w_ready_bar, w_phase)

    w_packed_pair = w_buf.reshape((BLOCK_N, BLOCK_K))
    fp4_padded_layout: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=128,
        element_bitwidth=8,
        rank=2,
        fp4_padded=True,
    )
    w_pair_first = w_packed_pair._reinterpret(
        gl.uint8,
        (BLOCK_N, BLOCK_K // 2),
        fp4_padded_layout,
    )

    blackwell.tcgen05_mma_scaled(
        w_pair_first,
        x_buf.permute((1, 0)),
        acc_buf,
        p.w_scale_tmem,
        p.x_scale_tmem,
        a_type="e2m1",
        b_type="e4m3",
        use_acc=use_acc,
        mbarriers=[x_empty_bar, scale_empty_bar, w_empty_bar],
    )

    blackwell.tcgen05_commit(w_empty_bar)
    w_idx, w_phase = advance(w_idx, w_phase, W_NUM_BUFS)
    scale_idx, scale_phase = advance(scale_idx, scale_phase, W_SCALE_NUM_BUFS)
    x_idx, x_phase = advance(x_idx, x_phase, X_NUM_BUFS)

    blackwell.tcgen05_commit(acc_ready_bar)
    mma_idx, mma_phase = advance(mma_idx, mma_phase, ACC_NUM_BUFS)


@gluon.jit
def replay_partition(p: PartitionArgs):
    w_idx = 0
    w_phase = 0
    replay_idx = 0
    replay_empty_phase = 1
    dense_copy_phase = 0

    dense_pair_layout: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=128,
        element_bitwidth=32,
        rank=2,
        fp4_padded=False,
    )
    for _ in range(K_TILES // 2):
        w_ready_bar = p.w_ready_bars.index(w_idx)
        w_empty_bar = p.w_empty_bars.index(w_idx)
        w_buf = p.w_bufs.index(w_idx)
        mbarrier.wait(w_ready_bar, w_phase)

        replay_tmem = p.replay_tmem.index(replay_idx)
        replay_empty_bar = p.replay_empty_bars.index(replay_idx)
        replay_full_bar = p.replay_full_bars.index(replay_idx)
        mbarrier.wait(replay_empty_bar, replay_empty_phase)
        dense_replay_tmem = p.dense_replay_tmem
        dense_pair_words = w_buf.reshape((BLOCK_N, BLOCK_K))._reinterpret(
            gl.uint32,
            (BLOCK_N, BLOCK_K // 4),
            dense_pair_layout,
        )
        blackwell.tcgen05_copy(dense_pair_words, dense_replay_tmem)
        blackwell.tcgen05_commit(p.dense_copy_done_bar)
        mbarrier.wait(p.dense_copy_done_bar, dense_copy_phase)
        dense_copy_phase = dense_copy_phase ^ 1

        blackwell.fence_async_shared()
        mbarrier.arrive(w_empty_bar)
        w_idx, w_phase = advance(w_idx, w_phase, W_NUM_BUFS)

        replay_cols: gl.constexpr = BLOCK_K // 4
        replay_segments: gl.constexpr = BLOCK_K // 16
        dense_frag = dense_replay_tmem.slice(0, replay_cols)
        replay_frag = replay_tmem.slice(0, replay_cols)
        dense_words = dense_frag.load()
        dense_words = dense_words.reshape((BLOCK_N, replay_segments, 2, 2)).permute((0, 1, 3, 2))
        _, dense_words = gl.split(dense_words)
        dense_words = dense_words.reshape((1, 1, 1, BLOCK_N, replay_segments, 2))
        lo, hi = fp4_prmt_shuffle_elements(dense_words)
        k_second = gl.join(lo, hi).reshape([BLOCK_N, replay_cols])
        replay_frag.store(gl.convert_layout(k_second, replay_frag.get_reg_layout()))
        mbarrier.arrive(replay_full_bar)
        replay_idx, replay_empty_phase = advance(replay_idx, replay_empty_phase, 2)

    w_idx, w_phase = advance(w_idx, w_phase, W_NUM_BUFS)


@gluon.jit
def epilogue_partition(p: PartitionArgs):
    idx = 0
    phase = 0

    out_recip = 0.25

    num_warps: gl.constexpr = gl.num_warps()
    split_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4],
        [1, 32],
        [num_warps, 1],
        [1, 0],
    )
    store_layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_M // gl.num_warps(), 2],
        [1, 32],
        [gl.num_warps(), 1],
        [1, 0],
    )

    out_off_n_packed = gl.program_id(0) * (BLOCK_N // REDUCTION_N // 4)
    acc_empty_bar = p.acc_empty_bars.index(idx)
    acc_ready_bar = p.acc_ready_bars.index(idx)
    acc_buf = p.acc_bufs.index(idx)

    mbarrier.wait(acc_ready_bar, phase)
    acc_regs = acc_buf.load().permute((1, 0))
    mbarrier.arrive(acc_empty_bar)
    idx, phase = advance(idx, phase, ACC_NUM_BUFS)
    acc = gl.convert_layout(acc_regs, split_layout)
    acc_packed = float2.pack(acc, axis=1)

    packed_fp8 = gl.convert_layout(
        pack_e4m3x2(acc_packed * float2.full_like(acc_packed, out_recip)),
        store_layout,
    )
    values = pack_fp8x4(packed_fp8)
    layout: gl.constexpr = values.type.layout
    offs_m = gl.arange(0, values.shape[0], layout=gl.SliceLayout(1, layout))
    offs_n = out_off_n_packed + gl.arange(0, values.shape[1], layout=gl.SliceLayout(0, layout))
    mask_m = gl.expand_dims(offs_m < 1, 1)
    mask_n = gl.expand_dims(offs_n < OUT_PACKED_N, 0)
    ptrs = p.out_ptr.cast(gl.pointer_type(gl.int32), bitcast=True)
    ptrs = ptrs + gl.expand_dims(offs_m, 1) * OUT_PACKED_N
    ptrs = ptrs + gl.expand_dims(offs_n, 0)
    gl.store(ptrs, values, mask=mask_m & mask_n)


@gluon.jit
def ws_matmul_kernel(
    x_desc: tma.tensor_descriptor,
    w_desc: tma.tensor_descriptor,
    scale_desc: tma.tensor_descriptor,
    out_ptr: gl.tensor,
):
    gl.static_assert(gl.num_ctas() == 1, "standalone repro is 1CTA-only")

    scale_k: gl.constexpr = BLOCK_K // MXFP_BLOCK_SIZE
    x_scale_layout: gl.constexpr = blackwell.TensorMemoryScalesLayout()
    w_scale_layout: gl.constexpr = blackwell.TensorMemoryScalesLayout()
    acc_layout: gl.constexpr = blackwell.TensorMemoryLayout(
        [MMA_BLOCK_COL, BLOCK_M],
        col_stride=1,
    )
    replay_layout: gl.constexpr = blackwell.TensorMemoryLayout(
        (MMA_BLOCK_COL, BLOCK_K // 4),
        col_stride=1,
    )
    dense_replay_layout: gl.constexpr = blackwell.TensorMemoryLayout(
        (MMA_BLOCK_COL, BLOCK_K // 4),
        col_stride=1,
    )

    x_bufs = gl.allocate_shared_memory(
        x_desc.dtype,
        [X_NUM_BUFS, BLOCK_M, x_desc.block_type.shape[1]],
        x_desc.layout,
    )
    x_empty_bars, x_ready_bars = alloc_ring_barriers(X_NUM_BUFS)

    w_bufs = gl.allocate_shared_memory(
        w_desc.dtype,
        [W_NUM_BUFS] + w_desc.block_type.shape,
        w_desc.layout,
    )
    w_empty_bars, w_ready_bars = alloc_ring_barriers(W_NUM_BUFS, consumer_count=2)

    w_scale_bufs = gl.allocate_shared_memory(
        scale_desc.dtype,
        [W_SCALE_NUM_BUFS] + scale_desc.block_type.shape,
        scale_desc.layout,
    )
    w_scale_empty_bars, w_scale_ready_bars = alloc_ring_barriers(W_SCALE_NUM_BUFS)

    x_scale_tmem = blackwell.allocate_tensor_memory(gl.uint8, [BLOCK_M, scale_k], x_scale_layout)
    w_scale_tmem = blackwell.allocate_tensor_memory(gl.uint8, [BLOCK_N, scale_k], w_scale_layout)
    dense_replay_tmem = blackwell.allocate_tensor_memory(
        gl.uint32,
        [BLOCK_N, BLOCK_K // 4],
        dense_replay_layout,
    )
    replay_tmem = blackwell.allocate_tensor_memory(
        gl.uint32,
        [2, BLOCK_N, BLOCK_K // 4],
        replay_layout,
    )
    replay_empty_bars = alloc_barrier_ring(2)
    replay_full_bars = alloc_barrier_ring(2)

    acc_tmem = blackwell.allocate_tensor_memory(
        gl.float32,
        [ACC_NUM_BUFS, BLOCK_N, BLOCK_M],
        acc_layout,
    )
    acc_empty_bars, acc_ready_bars = alloc_ring_barriers(ACC_NUM_BUFS)

    dense_copy_done_bar = mbarrier.allocate_mbarrier()
    mbarrier.init(dense_copy_done_bar, count=1)
    unpack_sync_bar = mbarrier.allocate_mbarrier()
    mbarrier.init(unpack_sync_bar, count=1)

    x_scale_tmem.store(
        gl.full((BLOCK_M, scale_k), 127, dtype=gl.uint8, layout=x_scale_tmem.get_reg_layout())
    )

    p = PartitionArgs(
        x_desc=x_desc,
        w_desc=w_desc,
        scale_desc=scale_desc,
        out_ptr=out_ptr,
        #
        x_bufs=x_bufs,
        x_empty_bars=x_empty_bars,
        x_ready_bars=x_ready_bars,
        #
        w_bufs=w_bufs,
        w_empty_bars=w_empty_bars,
        w_ready_bars=w_ready_bars,
        #
        w_scale_bufs=w_scale_bufs,
        w_scale_empty_bars=w_scale_empty_bars,
        w_scale_ready_bars=w_scale_ready_bars,
        #
        x_scale_tmem=x_scale_tmem,
        w_scale_tmem=w_scale_tmem,
        dense_replay_tmem=dense_replay_tmem,
        replay_tmem=replay_tmem,
        replay_empty_bars=replay_empty_bars,
        replay_full_bars=replay_full_bars,
        acc_bufs=acc_tmem,
        acc_empty_bars=acc_empty_bars,
        acc_ready_bars=acc_ready_bars,
        dense_copy_done_bar=dense_copy_done_bar,
        unpack_sync_bar=unpack_sync_bar,
    )

    gl.warp_specialize(
        [
            (epilogue_partition, (p,)),
            (load_activations, (p,)),
            (load_weights, (p,)),
            (load_weight_scales, (p,)),
            (replay_partition, (p,)),
            (mma_partition, (p,)),
        ],
        [
            LOAD_ACTIVATION_WARPS,
            LOAD_WEIGHT_WARPS,
            LOAD_WEIGHT_SCALES_WARPS,
            REPLAY_WARPS,
            MMA_WARPS,
        ],
        [
            LOAD_ACTIVATION_REGS,
            LOAD_WEIGHT_REGS,
            LOAD_WEIGHT_SCALES_REGS,
            REPLAY_REGS,
            MMA_REGS,
        ],
    )

# ===-----------------------------------------------------------------------===#
# Benchmark and Testing Helpers
# ===-----------------------------------------------------------------------===#

def run_repro(max_launches: int = 1000):
    torch.cuda.set_device(0)
    device = "cuda:0"
    torch.manual_seed(0)

    x = alloc_rand((1, 384), device=device, dtype=torch.float8_e4m3fn)
    w_data = torch.randn((1, 384, 512), device=device, dtype=torch.bfloat16)
    w_data, w_scale = downcast_to_mxfp(w_data, FP4, axis=1)  # type: ignore[arg-type]
    w = convert_layout(
        wrap_torch_tensor(w_data, dtype=FP4),
        BlackwellMX4ValuePackedShuffledLayout(block_k=BLOCK_K_VALUE, block_n=BLOCK_N_VALUE),
    )
    w_scale = convert_layout(
        wrap_torch_tensor(w_scale),
        make_default_matmul_mxfp4_w_scale_layout(mx_axis=1, num_warps=NUM_WARPS_VALUE),
    )

    out = torch.zeros((1, 256), dtype=torch.float8_e4m3fn, device=device)
    x_desc = TensorDescriptor(
        x,
        list(x.shape),
        list(x.stride()),
        [1, BLOCK_K_VALUE],
        gl.NVMMASharedLayout(swizzle_byte_width=BLOCK_K_VALUE, element_bitwidth=8, rank=2),
    )
    w_ptr = w.storage.data
    w_strides = list(w_ptr.stride())
    w_block_shape = w.storage.layout.swizzle_block_shape([1, BLOCK_K_VALUE * 2, BLOCK_N_VALUE])
    w_block_shape[w_strides.index(1)] //= 2
    w_desc = TensorDescriptor(
        w_ptr,
        list(w_ptr.shape),
        w_strides,
        w_block_shape,
        gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8, rank=5, fp4_padded=False),
    )
    scale_ptr = w_scale.storage.data
    scale_desc = TensorDescriptor(
        scale_ptr,
        list(scale_ptr.shape),
        list(scale_ptr.stride()),
        [
            1,
            BLOCK_N_VALUE // SCALE_SIZE_OUTER_VALUE,
            BLOCK_K_VALUE // MXFP_BLOCK_SIZE_VALUE // SCALE_SIZE_INNER_VALUE,
            2,
            256,
        ],
        gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5),
    )
    kernel = ws_matmul_kernel[(2,)]

    def run_kernel():
        kernel(
            x_desc=x_desc,
            w_desc=w_desc,
            scale_desc=scale_desc,
            out_ptr=out,
            num_warps=NUM_WARPS_VALUE,
            num_ctas=1,
            maxnreg=None,
        )

    run_kernel()
    torch.cuda.synchronize()
    expected = out.clone()

    for launch in range(1, max_launches + 1):
        out.zero_()
        run_kernel()
        torch.cuda.synchronize()
        if not torch.equal(out, expected):
            maxdiff = (out.to(torch.float32) - expected.to(torch.float32)).abs().max().item()
            print(f"FAIL launch={launch} maxdiff={maxdiff}")
            return 1
    print(f"PASS launch={max_launches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_repro())
