from dataclasses import dataclass
import torch
import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
import triton.experimental.gluon.language.nvidia.blackwell as blackwell
import triton.experimental.gluon.language.nvidia.blackwell.tma as tma
from triton.experimental.gluon.language.nvidia.blackwell import float2
import triton.experimental.gluon.language.nvidia.hopper.mbarrier as mbarrier
import triton.language.extra.libdevice as libdevice
from triton_kernels.matmul import (
    FlexCtx,
    FnSpecs,
    FusedActivation,
    PrecisionConfig,
)
from triton_kernels.numerics import InFlexData, OutFlexData
from triton_kernels.numerics_details.mxfp import MXFP_BLOCK_SIZE, downcast_to_mxfp
from triton_kernels.swiglu import swiglu_fn
from triton_kernels.tensor import (
    FP4,
    RaggedTensorMetadata,
    Tensor,
    convert_layout,
    make_ragged_tensor_metadata,
    wrap_torch_tensor,
)
from triton_kernels.tensor_details.dtype import UINT8
from triton_kernels.tensor_details.layout import (
    BlackwellMX4ValuePackedShuffledLayout,
    make_default_matmul_mxfp4_w_scale_layout,
)
from triton_kernels.testing import alloc_rand

# ===-----------------------------------------------------------------------===#
# Device Code
# ===-----------------------------------------------------------------------===#


@gluon.jit
def advance(idx: gl.tensor, phase: gl.tensor, num_bufs: gl.constexpr) -> tuple[gl.tensor, gl.tensor]:
    next_idx = idx + 1
    wrap = next_idx == num_bufs
    return gl.where(wrap, 0, next_idx), gl.where(wrap, phase ^ 1, phase)


@gluon.jit
def unpack_block_schedule(schedule: gl.tensor) -> tuple[gl.tensor, gl.tensor]:
    return schedule & 0xFFFF, schedule >> 16


@gluon.jit
def apply_block_schedule(
    block_id: gl.tensor,
    GRID_N: gl.constexpr,
    slice_offsets: gl.tensor,
    block_schedule: gl.tensor,
) -> tuple[gl.tensor, gl.tensor, gl.tensor, gl.tensor]:
    schedule_pid_m = block_id // GRID_N
    pid_n = block_id % GRID_N

    slice_idx, pid_m = unpack_block_schedule(gl.load(block_schedule + schedule_pid_m))
    slice_offset = gl.load(slice_offsets + slice_idx)

    return pid_m, pid_n, slice_idx, slice_offset


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
def replay_mma_wait_relaxed_cluster(bar, phase, x):
    return gl.inline_asm_elementwise(
        """
        {
            .reg .pred complete;
            .reg .b64 bar64;
            cvt.u64.u32 bar64, $1;
            cvta.shared.u64 bar64, bar64;
        waitLoop:
            mbarrier.try_wait.parity.relaxed.cluster.b64 complete, [bar64], $2;
            @!complete bra.uni waitLoop;
            mov.b32 $0, $3;
        }
        """,
        "=r,r,r,r",
        [bar.to_i32(), phase, x],
        dtype=gl.int32,
        is_pure=False,
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
    out_desc: tma.tensor_descriptor
    x_scale_ptr: gl.tensor | gl.constexpr
    w_scale_ptr: gl.tensor | gl.constexpr
    out_scale_ptr: gl.tensor

    out_ptr: gl.tensor
    bias_ptr: gl.tensor
    bias_stride: gl.tensor
    gather_indx_ptr: gl.tensor
    x_slice_sizes: gl.tensor
    x_slice_offs: gl.tensor
    x_block_schedule: gl.tensor

    x_bufs: gl.shared_memory_descriptor
    x_empty_bars: gl.shared_memory_descriptor
    x_ready_bars: gl.shared_memory_descriptor
    X_NUM_BUFS: gl.constexpr

    w_bufs: gl.shared_memory_descriptor
    w_empty_bars: gl.shared_memory_descriptor
    w_ready_bars: gl.shared_memory_descriptor
    W_NUM_BUFS: gl.constexpr

    w_scale_bufs: gl.shared_memory_descriptor
    w_scale_empty_bars: gl.shared_memory_descriptor
    w_scale_ready_bars: gl.shared_memory_descriptor
    W_SCALE_NUM_BUFS: gl.constexpr

    x_scale_tmem: blackwell.tensor_memory_descriptor
    w_scale_tmem: blackwell.tensor_memory_descriptor
    dense_replay_tmem: blackwell.tensor_memory_descriptor
    replay_tmem: blackwell.tensor_memory_descriptor
    replay_empty_bars: gl.shared_memory_descriptor
    replay_full_bars: gl.shared_memory_descriptor
    acc_bufs: blackwell.tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    ACC_NUM_BUFS: gl.constexpr
    dense_copy_done_bar: gl.shared_memory_descriptor
    unpack_sync_bar: gl.shared_memory_descriptor

    grid_m: gl.tensor
    GRID_N: gl.constexpr
    K_TILES: gl.constexpr
    SCALE_FLAT_N: gl.constexpr
    SCALE_BLOCK_N_DIV: gl.constexpr
    num_blocks: gl.tensor

    NUM_SMS: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    SCALE_SIZE_OUTER: gl.constexpr
    SCALE_SIZE_INNER: gl.constexpr
    MXFP_BLOCK_SIZE: gl.constexpr
    MMA_BLOCK_COL: gl.constexpr

    SWIGLU_ALPHA: gl.constexpr
    SWIGLU_LIMIT: gl.constexpr
    REDUCTION_N: gl.constexpr
    FLEXPOINT_SATURATE_INF: gl.constexpr

    @gluon.jit
    def apply_block_schedule(self, block_id: gl.tensor) -> tuple[gl.tensor, gl.tensor, gl.tensor, gl.tensor]:
        return apply_block_schedule(
            block_id=block_id,
            GRID_N=self.GRID_N,
            slice_offsets=self.x_slice_offs,
            block_schedule=self.x_block_schedule,
        )


@gluon.jit
def issue_activation_tile(
    p: PartitionArgs,
    idx,
    phase,
    issued,
    offs_x_m,
    off_k_x,
    tile_x_bytes: gl.constexpr,
):
    empty_bar = p.x_empty_bars.index(idx)
    ready_bar = p.x_ready_bars.index(idx)
    x_buf = p.x_bufs.index(idx)

    mbarrier.wait(empty_bar, phase, pred=issued >= p.X_NUM_BUFS)
    mbarrier.expect(ready_bar, tile_x_bytes)
    tma.async_gather(
        p.x_desc,
        offs_x_m,
        off_k_x,
        ready_bar,
        x_buf,
    )

    idx, phase = advance(idx, phase, p.X_NUM_BUFS)
    return idx, phase, issued + 1


@gluon.jit
def load_activations(p: PartitionArgs):
    offs_layout: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]),
    )
    tile_x_bytes: gl.constexpr = p.x_desc.block_type.nbytes * p.BLOCK_M

    idx = 0
    phase = 1
    issued = 0

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, _, slice_idx, slice_offset = p.apply_block_schedule(block_id)
        off_m = pid_m * p.BLOCK_M
        offs_m = off_m + gl.arange(0, p.BLOCK_M, layout=offs_layout)

        shape_m = gl.load(p.x_slice_sizes + slice_idx)
        mask_m = offs_m < shape_m
        offs_x_m = gl.load(
            p.gather_indx_ptr + slice_offset + offs_m,
            mask=mask_m,
            other=p.x_desc.shape[0],
        )

        for ki in range(p.K_TILES):
            off_k_x = ki * p.BLOCK_K
            idx, phase, issued = issue_activation_tile(p, idx, phase, issued, offs_x_m, off_k_x, tile_x_bytes)


@gluon.jit
def load_weights(p: PartitionArgs):
    TILE_W_BYTES: gl.constexpr = p.w_desc.nbytes_per_cta

    idx = 0
    phase = 1
    issued = 0

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        _, pid_n, slice_idx, _ = p.apply_block_schedule(block_id)

        for ki in range(0, p.K_TILES, 2):
            w_empty_bar = p.w_empty_bars.index(idx)
            w_ready_bar = p.w_ready_bars.index(idx)
            w_buf = p.w_bufs.index(idx)

            mbarrier.wait(w_empty_bar, phase, pred=issued >= p.W_NUM_BUFS)
            mbarrier.expect(w_ready_bar, TILE_W_BYTES)
            tma.async_copy_global_to_shared(
                p.w_desc,
                [slice_idx, ki // 2, pid_n, 0, 0],
                w_ready_bar,
                w_buf,
            )

            idx, phase = advance(idx, phase, p.W_NUM_BUFS)
            issued += 1


@gluon.jit
def load_weight_scales(p: PartitionArgs):
    TILE_SCALE_BYTES: gl.constexpr = p.scale_desc.nbytes_per_cta

    idx = 0
    phase = 1
    issued = 0

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        _, pid_n, slice_idx, _ = p.apply_block_schedule(block_id)

        scale_idx = slice_idx * p.SCALE_FLAT_N + pid_n * p.SCALE_BLOCK_N_DIV
        for ki in range(p.K_TILES):
            off_k_scale = ki * p.BLOCK_K // (p.MXFP_BLOCK_SIZE * p.SCALE_SIZE_INNER)

            scale_empty_bar = p.w_scale_empty_bars.index(idx)
            scale_ready_bar = p.w_scale_ready_bars.index(idx)
            scale_buf = p.w_scale_bufs.index(idx)

            mbarrier.wait(scale_empty_bar, phase, pred=issued >= p.W_SCALE_NUM_BUFS)
            mbarrier.expect(scale_ready_bar, TILE_SCALE_BYTES)
            tma.async_copy_global_to_shared(
                p.scale_desc,
                [0, scale_idx, off_k_scale, 0, 0],
                scale_ready_bar,
                scale_buf,
            )

            idx, phase = advance(idx, phase, p.W_SCALE_NUM_BUFS)
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
        (p.MMA_BLOCK_COL, p.BLOCK_K),
        col_stride=1,
        fp4_padded=True,
    )

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        acc_empty_bar = p.acc_empty_bars.index(mma_idx)
        acc_ready_bar = p.acc_ready_bars.index(mma_idx)
        acc_buf = p.acc_bufs.index(mma_idx)
        mbarrier.wait(acc_empty_bar, mma_phase)

        use_acc = False
        for _ in range(p.K_TILES // 2):
            # First K tile. Wait for scales, weight, and act tile.
            scale_ready_bar = p.w_scale_ready_bars.index(scale_idx)
            scale_empty_bar = p.w_scale_empty_bars.index(scale_idx)
            scale_buf = p.w_scale_bufs.index(scale_idx)
            mbarrier.wait(scale_ready_bar, scale_phase)
            blackwell.tcgen05_copy(
                unswizzle_mx_scale(scale_buf, p.SCALE_SIZE_OUTER, p.SCALE_SIZE_INNER, p.MXFP_BLOCK_SIZE),
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

            # Keep the single MMA issuer from releasing an input ring slot
            # before every MMA warp has observed the ready epoch. MMAv5 lowering
            # also inserts one local barrier before the issuer block, but ptxas
            # lowers that as a single deferred barrier and it is not sufficient
            # for the repeated-run ring protocol below.
            gl.barrier()

            # w_packed_pair 512x128xi8, logical 512x256xfp4
            w_packed_pair = w_buf.reshape((p.BLOCK_N, p.BLOCK_K))
            fp4_padded_layout: gl.constexpr = gl.NVMMASharedLayout(
                swizzle_byte_width=128,
                element_bitwidth=8,
                rank=2,
                fp4_padded=True,
            )

            w_pair_first = w_packed_pair._reinterpret(
                gl.uint8,
                (p.BLOCK_N, p.BLOCK_K // 2),
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

            w_idx, w_phase = advance(w_idx, w_phase, p.W_NUM_BUFS)
            scale_idx, scale_phase = advance(scale_idx, scale_phase, p.W_SCALE_NUM_BUFS)
            x_idx, x_phase = advance(x_idx, x_phase, p.X_NUM_BUFS)

            # Second K tile. Wait for scales and act tile.
            scale_ready_bar = p.w_scale_ready_bars.index(scale_idx)
            scale_empty_bar = p.w_scale_empty_bars.index(scale_idx)
            scale_buf = p.w_scale_bufs.index(scale_idx)
            mbarrier.wait(scale_ready_bar, scale_phase)
            blackwell.tcgen05_copy(
                unswizzle_mx_scale(scale_buf, p.SCALE_SIZE_OUTER, p.SCALE_SIZE_INNER, p.MXFP_BLOCK_SIZE),
                p.w_scale_tmem,
            )

            x_ready_bar = p.x_ready_bars.index(x_idx)
            x_empty_bar = p.x_empty_bars.index(x_idx)
            x_buf = p.x_bufs.index(x_idx)
            mbarrier.wait(x_ready_bar, x_phase)
            # This second-tile post-wait convergence is correctness-critical;
            # without it, the issuer can release the next empty epoch early.
            gl.barrier()

            replay_tmem = p.replay_tmem.index(replay_idx)
            replay_empty_bar = p.replay_empty_bars.index(replay_idx)
            blackwell.tcgen05_mma_scaled(
                replay_tmem._reinterpret(gl.uint8, (p.BLOCK_N, p.BLOCK_K), k_second_u8_layout),
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
            x_idx, x_phase = advance(x_idx, x_phase, p.X_NUM_BUFS)
            scale_idx, scale_phase = advance(scale_idx, scale_phase, p.W_SCALE_NUM_BUFS)

        # First K tile. Wait for scales, weight, and act tile.
        scale_ready_bar = p.w_scale_ready_bars.index(scale_idx)
        scale_empty_bar = p.w_scale_empty_bars.index(scale_idx)
        scale_buf = p.w_scale_bufs.index(scale_idx)
        mbarrier.wait(scale_ready_bar, scale_phase)
        blackwell.tcgen05_copy(
            unswizzle_mx_scale(scale_buf, p.SCALE_SIZE_OUTER, p.SCALE_SIZE_INNER, p.MXFP_BLOCK_SIZE),
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

        # Preserve the same post-wait convergence for the odd-tail path.
        gl.barrier()

        # w_packed_pair 512x128xi8, logical 512x256xfp4
        w_packed_pair = w_buf.reshape((p.BLOCK_N, p.BLOCK_K))
        fp4_padded_layout: gl.constexpr = gl.NVMMASharedLayout(
            swizzle_byte_width=128,
            element_bitwidth=8,
            rank=2,
            fp4_padded=True,
        )

        w_pair_first = w_packed_pair._reinterpret(
            gl.uint8,
            (p.BLOCK_N, p.BLOCK_K // 2),
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

        blackwell.tcgen05_commit(w_empty_bar)
        w_idx, w_phase = advance(w_idx, w_phase, p.W_NUM_BUFS)

        scale_idx, scale_phase = advance(scale_idx, scale_phase, p.W_SCALE_NUM_BUFS)
        x_idx, x_phase = advance(x_idx, x_phase, p.X_NUM_BUFS)

        blackwell.tcgen05_commit(acc_ready_bar)
        mma_idx, mma_phase = advance(mma_idx, mma_phase, p.ACC_NUM_BUFS)


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
    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
            for _ in range(p.K_TILES // 2):
                w_ready_bar = p.w_ready_bars.index(w_idx)
                w_empty_bar = p.w_empty_bars.index(w_idx)
                w_buf = p.w_bufs.index(w_idx)
                mbarrier.wait(w_ready_bar, w_phase)

                replay_tmem = p.replay_tmem.index(replay_idx)
                replay_empty_bar = p.replay_empty_bars.index(replay_idx)
                replay_full_bar = p.replay_full_bars.index(replay_idx)
                mbarrier.wait(replay_empty_bar, replay_empty_phase)
                dense_replay_tmem = p.dense_replay_tmem
                dense_pair_words = w_buf.reshape((p.BLOCK_N, p.BLOCK_K))._reinterpret(
                    gl.uint32,
                    (p.BLOCK_N, p.BLOCK_K // 4),
                    dense_pair_layout,
                )
                blackwell.tcgen05_copy(dense_pair_words, dense_replay_tmem)
                blackwell.tcgen05_commit(p.dense_copy_done_bar)
                replay_mma_wait_relaxed_cluster(p.dense_copy_done_bar, dense_copy_phase, dense_copy_phase)
                dense_copy_phase = dense_copy_phase ^ 1

                # Once the async copy finishes, shared memory is no longer
                # needed by the replay side. Release the slot before the
                # TMEM load/unpack work so the next TMA can overlap it.
                blackwell.fence_async_shared()
                mbarrier.arrive(w_empty_bar)
                w_idx, w_phase = advance(w_idx, w_phase, p.W_NUM_BUFS)

                replay_cols: gl.constexpr = p.BLOCK_K // 4
                replay_segments: gl.constexpr = p.BLOCK_K // 16
                dense_frag = dense_replay_tmem.slice(0, replay_cols)
                replay_frag = replay_tmem.slice(0, replay_cols)
                dense_words = dense_frag.load()
                dense_words = dense_words.reshape((p.BLOCK_N, replay_segments, 2, 2)).permute((0, 1, 3, 2))
                _, dense_words = gl.split(dense_words)
                dense_words = dense_words.reshape((1, 1, 1, p.BLOCK_N, replay_segments, 2))
                lo, hi = fp4_prmt_shuffle_elements(dense_words)
                k_second = gl.join(lo, hi).reshape([p.BLOCK_N, replay_cols])
                replay_frag.store(gl.convert_layout(k_second, replay_frag.get_reg_layout()))
                mbarrier.arrive(replay_full_bar)
                replay_idx, replay_empty_phase = advance(replay_idx, replay_empty_phase, 2)

            # The tail tile has no replay work, but it still occupies one
            # packed-weight ring slot. Keep this partition's cursor aligned
            # with the MMA partition before the next output block.
            w_idx, w_phase = advance(w_idx, w_phase, p.W_NUM_BUFS)


@gluon.jit
def store_packed_out(
    p: PartitionArgs,
    packed_out,
    off_m,
    out_off_n_packed,
    shape_m,
    slice_offset,
):
    values = pack_fp8x4(packed_out)
    layout: gl.constexpr = values.type.layout
    offs_m = off_m + gl.arange(0, values.shape[0], layout=gl.SliceLayout(1, layout))
    offs_n = out_off_n_packed + gl.arange(0, values.shape[1], layout=gl.SliceLayout(0, layout))
    mask_m = gl.expand_dims(offs_m < shape_m, 1)
    mask_n = gl.expand_dims(offs_n < (p.out_desc.shape[1] + 3) // 4, 0)
    mask = mask_m & mask_n
    ptrs = p.out_ptr.cast(gl.pointer_type(gl.int32), bitcast=True)
    ptrs = ptrs + gl.expand_dims(slice_offset + offs_m, 1) * (p.out_desc.strides[0] // 4)
    ptrs = ptrs + gl.expand_dims(offs_n, 0) * p.out_desc.strides[1]
    gl.store(ptrs, values, mask=mask)


@gluon.jit
def _swiglu_step1(acc_packed, limit):
    gelu, linear = float2.unpack2(acc_packed)
    gelu = gl.minimum(gelu.to(gl.float32), limit)
    linear = gl.minimum(gl.maximum(linear.to(gl.float32), -limit), limit)
    return gelu, linear


@gluon.jit
def _swiglu_step2(gelu, linear, alpha):
    den = 1.0 + libdevice.exp(-alpha * gelu)
    activated = gelu / den
    activated_packed = float2.pack(activated, axis=1)
    linear_packed = float2.pack(linear, axis=1)
    return float2.fma(activated_packed, linear_packed, activated_packed)


@gluon.jit
def pack_fp8_out_fragment(out_packed, out_recip):
    scaled_out_packed = out_packed * float2.full_like(out_packed, out_recip)
    return pack_e4m3x2(scaled_out_packed)


@gluon.jit
def get_store_layout(p: PartitionArgs):
    return gl.BlockedLayout(
        [p.BLOCK_M // gl.num_warps(), 2],
        [1, 32],
        [gl.num_warps(), 1],
        [1, 0],
    )


@gluon.jit
def epilogue_direct_store(
    p: PartitionArgs,
    acc_packed,
    out_recip,
    off_m,
    out_off_n_packed,
    shape_m,
    slice_offset,
    store_layout: gl.constexpr,
):
    gelu, linear = _swiglu_step1(acc_packed, p.SWIGLU_LIMIT)
    out_packed = _swiglu_step2(gelu, linear, p.SWIGLU_ALPHA)
    packed_fp8 = gl.convert_layout(pack_fp8_out_fragment(out_packed, out_recip), store_layout)
    store_packed_out(
        p,
        packed_fp8,
        off_m,
        out_off_n_packed,
        shape_m,
        slice_offset,
    )


@gluon.jit
def apply_bias_and_scale(
    p: PartitionArgs,
    idx,
    phase,
    pid_n,
    slice_idx,
    split_layout: gl.constexpr,
    bias_layout: gl.constexpr,
    acc_scale,
):
    off_n = pid_n * p.BLOCK_N
    acc_empty_bar = p.acc_empty_bars.index(idx)
    acc_ready_bar = p.acc_ready_bars.index(idx)
    acc_buf = p.acc_bufs.index(idx)

    offs_bias_n = off_n + gl.arange(0, p.BLOCK_N, layout=bias_layout)
    bias = gl.convert_layout(
        gl.expand_dims(gl.load(p.bias_ptr + slice_idx * p.bias_stride + offs_bias_n), axis=0),
        split_layout,
    )
    mbarrier.wait(acc_ready_bar, phase)
    acc_regs = acc_buf.load().permute((1, 0))
    mbarrier.arrive(acc_empty_bar)
    idx, phase = advance(idx, phase, p.ACC_NUM_BUFS)
    acc = gl.convert_layout(acc_regs, split_layout)
    acc_packed = float2.pack(acc, axis=1)
    bias_packed = float2.pack(bias, axis=1)
    bias_packed = float2.Float2Tensor(gl.convert_layout(bias_packed.value, acc_packed.value.type.layout))
    acc_packed = float2.fma(acc_packed, float2.full_like(acc_packed, acc_scale), bias_packed)
    return idx, phase, acc_packed


@gluon.jit
def epilogue_partition(p: PartitionArgs):
    idx = 0
    phase = 0

    x_scale = 1.0 if p.x_scale_ptr is None else gl.load(p.x_scale_ptr)
    w_scale = 1.0 if p.w_scale_ptr is None else gl.load(p.w_scale_ptr)
    acc_scale = x_scale * w_scale
    out_recip = 1.0 / gl.load(p.out_scale_ptr)

    num_warps: gl.constexpr = gl.num_warps()
    warps_n: gl.constexpr = 1
    split_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4],
        [1, 32],
        [num_warps // warps_n, warps_n],
        [1, 0],
    )
    bias_layout: gl.constexpr = gl.SliceLayout(0, split_layout)
    store_layout: gl.constexpr = get_store_layout(p)

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, pid_n, slice_idx, slice_offset = p.apply_block_schedule(block_id)
        off_m = pid_m * p.BLOCK_M
        shape_m = gl.load(p.x_slice_sizes + slice_idx)
        out_off_n_packed = pid_n * (p.BLOCK_N // p.REDUCTION_N // 4)
        idx, phase, acc_packed = apply_bias_and_scale(
            p,
            idx,
            phase,
            pid_n,
            slice_idx,
            split_layout,
            bias_layout,
            acc_scale,
        )
        epilogue_direct_store(
            p,
            acc_packed,
            out_recip,
            off_m,
            out_off_n_packed,
            shape_m,
            slice_offset,
            store_layout,
        )


@gluon.jit
def ws_matmul_kernel(
    x_desc: tma.tensor_descriptor,
    w_desc: tma.tensor_descriptor,
    scale_desc: tma.tensor_descriptor,
    out_desc: tma.tensor_descriptor,
    out_ptr: gl.tensor,
    #
    bias_ptr: gl.tensor,
    bias_stride: gl.tensor,
    #
    gather_indx_ptr: gl.tensor,
    #
    x_slice_sizes: gl.tensor,
    x_slice_offs: gl.tensor,
    x_block_offs: gl.tensor,
    x_block_schedule: gl.tensor,
    #
    x_scale_ptr: gl.tensor,
    w_scale_ptr: gl.tensor,
    out_scale_ptr: gl.tensor,
    #
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    NUM_SLICES: gl.constexpr,
    #
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    REDUCTION_N: gl.constexpr,
    #
    FLEXPOINT_SATURATE_INF: gl.constexpr,
    #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_SMS: gl.constexpr,
    X_NUM_BUFS: gl.constexpr,
    W_NUM_BUFS: gl.constexpr,
    W_SCALE_NUM_BUFS: gl.constexpr,
    ACC_NUM_BUFS: gl.constexpr,
    #
    LOAD_ACTIVATION_WARPS: gl.constexpr,
    LOAD_WEIGHT_WARPS: gl.constexpr,
    LOAD_WEIGHT_SCALES_WARPS: gl.constexpr,
    REPLAY_WARPS: gl.constexpr,
    MMA_WARPS: gl.constexpr,
    #
    LOAD_ACTIVATION_REGS: gl.constexpr,
    LOAD_WEIGHT_REGS: gl.constexpr,
    LOAD_WEIGHT_SCALES_REGS: gl.constexpr,
    REPLAY_REGS: gl.constexpr,
    MMA_REGS: gl.constexpr,
    #
    SCALE_SIZE_OUTER: gl.constexpr,
    SCALE_SIZE_INNER: gl.constexpr,
    MXFP_BLOCK_SIZE: gl.constexpr,
):
    gl.static_assert(gl.num_ctas() == 1, "standalone repro is 1CTA-only")

    grid_m = gl.load(x_block_offs + NUM_SLICES)
    grid_n: gl.constexpr = triton.cdiv(N, BLOCK_N)
    k_tiles: gl.constexpr = triton.cdiv(K, BLOCK_K)
    scale_flat_n: gl.constexpr = N // SCALE_SIZE_OUTER
    scale_block_n_div: gl.constexpr = BLOCK_N // SCALE_SIZE_OUTER
    num_blocks = grid_m * grid_n

    scale_k: gl.constexpr = BLOCK_K // MXFP_BLOCK_SIZE
    x_scale_layout: gl.constexpr = blackwell.TensorMemoryScalesLayout()
    w_scale_layout: gl.constexpr = blackwell.TensorMemoryScalesLayout()
    MMA_BLOCK_COL: gl.constexpr = min(128, BLOCK_N)
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
    replay_tmem = blackwell.allocate_tensor_memory(gl.uint32, [2, BLOCK_N, BLOCK_K // 4], replay_layout)
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

    x_scale_tmem.store(gl.full((BLOCK_M, scale_k), 127, dtype=gl.uint8, layout=x_scale_tmem.get_reg_layout()))

    p = PartitionArgs(
        x_desc=x_desc,
        w_desc=w_desc,
        scale_desc=scale_desc,
        out_desc=out_desc,
        x_scale_ptr=x_scale_ptr,
        w_scale_ptr=w_scale_ptr,
        out_scale_ptr=out_scale_ptr,
        #
        out_ptr=out_ptr,
        bias_ptr=bias_ptr,
        bias_stride=bias_stride,
        gather_indx_ptr=gather_indx_ptr,
        x_slice_sizes=x_slice_sizes,
        x_slice_offs=x_slice_offs,
        x_block_schedule=x_block_schedule,
        #
        x_bufs=x_bufs,
        x_empty_bars=x_empty_bars,
        x_ready_bars=x_ready_bars,
        X_NUM_BUFS=X_NUM_BUFS,
        #
        w_bufs=w_bufs,
        w_empty_bars=w_empty_bars,
        w_ready_bars=w_ready_bars,
        W_NUM_BUFS=W_NUM_BUFS,
        #
        w_scale_bufs=w_scale_bufs,
        w_scale_empty_bars=w_scale_empty_bars,
        w_scale_ready_bars=w_scale_ready_bars,
        W_SCALE_NUM_BUFS=W_SCALE_NUM_BUFS,
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
        ACC_NUM_BUFS=ACC_NUM_BUFS,
        dense_copy_done_bar=dense_copy_done_bar,
        unpack_sync_bar=unpack_sync_bar,
        #
        grid_m=grid_m,
        GRID_N=grid_n,
        K_TILES=k_tiles,
        SCALE_FLAT_N=scale_flat_n,
        SCALE_BLOCK_N_DIV=scale_block_n_div,
        num_blocks=num_blocks,
        #
        NUM_SMS=NUM_SMS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SCALE_SIZE_OUTER=SCALE_SIZE_OUTER,
        SCALE_SIZE_INNER=SCALE_SIZE_INNER,
        MXFP_BLOCK_SIZE=MXFP_BLOCK_SIZE,
        MMA_BLOCK_COL=MMA_BLOCK_COL,
        #
        SWIGLU_ALPHA=SWIGLU_ALPHA,
        SWIGLU_LIMIT=SWIGLU_LIMIT,
        REDUCTION_N=REDUCTION_N,
        FLEXPOINT_SATURATE_INF=FLEXPOINT_SATURATE_INF,
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
        [LOAD_ACTIVATION_WARPS, LOAD_WEIGHT_WARPS, LOAD_WEIGHT_SCALES_WARPS, REPLAY_WARPS, MMA_WARPS],
        [LOAD_ACTIVATION_REGS, LOAD_WEIGHT_REGS, LOAD_WEIGHT_SCALES_REGS, REPLAY_REGS, MMA_REGS],
    )


# ===-----------------------------------------------------------------------===#
# Host Code
# ===-----------------------------------------------------------------------===#


def make_tensor_descriptor(
    t: torch.Tensor | Tensor,
    block_shape: tuple[int, ...],
    *,
    layout_block_shape: tuple[int, ...] | None = None,
    cga_layout: tuple[tuple[int, ...], ...] = (),
):
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    ptr = t if isinstance(t, torch.Tensor) else t.storage.data
    shape = list(ptr.shape)
    strides = list(ptr.stride())
    desc_block_shape = list(block_shape)
    layout_shape = list(layout_block_shape or block_shape)

    if isinstance(t, Tensor) and t.dtype == FP4:
        assert isinstance(t.storage.layout, BlackwellMX4ValuePackedShuffledLayout)
        assert layout_block_shape is None
        desc_block_shape = t.storage.layout.swizzle_block_shape(desc_block_shape)
        desc_block_shape[strides.index(1)] //= 2
        layout_shape = desc_block_shape

    rank = len(layout_shape)
    if t.dtype == FP4:
        assert rank == 5
        layout = gl.NVMMASharedLayout(
            swizzle_byte_width=128,
            element_bitwidth=8,
            rank=rank,
            fp4_padded=False,
            cga_layout=cga_layout,
        )
    elif t.dtype == UINT8:
        assert rank == 5
        layout = gl.NVMMASharedLayout(
            swizzle_byte_width=0,
            element_bitwidth=8,
            rank=rank,
            cga_layout=cga_layout,
        )
    else:
        assert t.dtype == torch.float8_e4m3fn
        layout = gl.NVMMASharedLayout(
            swizzle_byte_width=layout_shape[-1],
            element_bitwidth=8,
            rank=rank,
            cga_layout=cga_layout,
        )
    return TensorDescriptor(ptr, shape, strides, desc_block_shape, layout)


@dataclass(frozen=True, slots=True)
class KernelConfig:
    BLOCK_M: int = 128
    BLOCK_N: int = 256
    BLOCK_K: int = 128

    X_NUM_BUFS: int = 4
    W_NUM_BUFS: int = 2
    W_SCALE_NUM_BUFS: int = 4
    ACC_NUM_BUFS: int = 1

    NUM_WARPS: int = 8
    LOAD_ACTIVATION_WARPS: int = 4
    LOAD_WEIGHT_WARPS: int = 1
    LOAD_WEIGHT_SCALES_WARPS: int = 1
    REPLAY_WARPS: int = 4
    MMA_WARPS: int = 4

    LOAD_ACTIVATION_REGS: int = 32
    LOAD_WEIGHT_REGS: int = 24
    LOAD_WEIGHT_SCALES_REGS: int = 24
    REPLAY_REGS: int = 80
    MMA_REGS: int = 80
    MAXNREG: int = None
    OCCUPANCY: int = 1

    MXFP_BLOCK_SIZE: int = 32
    SCALE_SIZE_OUTER: int = 128
    SCALE_SIZE_INNER: int = 4

def matmul(
    a: torch.Tensor,
    b: torch.Tensor | Tensor,
    bias: torch.Tensor,
    a_ragged_metadata: RaggedTensorMetadata,
    gather_indx: torch.Tensor,
    precision_config: PrecisionConfig,
    c: torch.Tensor,
    fused_activation: FusedActivation,
):
    specs = fused_activation.specs
    assert specs.name == "swiglu"
    reduction_n = specs.reduction_n
    swiglu_alpha, swiglu_limit = fused_activation.fn_args

    b_mx_scales = precision_config.b_mx_scale

    out_dtype = precision_config.out_dtype
    assert out_dtype is not None

    assert c.ndim == 2

    flex_ctx = precision_config.flex_ctx

    assert a.ndim == 2
    _, k = a.shape
    _, _, n = b.shape
    m = gather_indx.shape[0]

    p = KernelConfig()
    assert isinstance(b, Tensor)
    assert isinstance(b.storage.layout, BlackwellMX4ValuePackedShuffledLayout)
    assert b.storage.layout.block_k == p.BLOCK_K
    assert b.storage.layout.block_n == p.BLOCK_N
    x_block_offs = a_ragged_metadata.block_offs(p.BLOCK_M)
    x_block_schedule = a_ragged_metadata.block_schedule(p.BLOCK_M)
    expected_grid_m = a_ragged_metadata.n_blocks(a_ragged_metadata.n_slices, m, p.BLOCK_M)
    grid_n = triton.cdiv(n, p.BLOCK_N)
    sms = torch.cuda.get_device_properties(bias.device).multi_processor_count
    sms *= p.OCCUPANCY
    launch_grid = max(1, min(sms, expected_grid_m * grid_n))
    grid = (launch_grid,)

    x_desc = make_tensor_descriptor(
        a,
        (1, p.BLOCK_K),
        layout_block_shape=(p.BLOCK_M, p.BLOCK_K),
    )
    w_desc = make_tensor_descriptor(
        b,
        (1, p.BLOCK_K * 2, p.BLOCK_N),
        # Sharded weight tiles use the physical [1, 1, 1, N, K/2] MX4 shuffled block layout.
    )
    scale_desc = make_tensor_descriptor(
        b_mx_scales,
        (
            1,
            p.BLOCK_N // p.SCALE_SIZE_OUTER,
            p.BLOCK_K // p.MXFP_BLOCK_SIZE // p.SCALE_SIZE_INNER,
            2,
            256,
        ),
        # Weight scale tiles use the physical [1, N//128, K//(32*4), 2, 256] layout.
    )
    # The output descriptor is only used for shape/stride metadata during
    # direct stores, so cap the layout width to a legal FP8 swizzle size.
    out_desc = make_tensor_descriptor(c, (p.BLOCK_M, min(p.BLOCK_N // reduction_n, 128)))

    ws_matmul_kernel[grid](
        x_desc=x_desc,
        w_desc=w_desc,
        scale_desc=scale_desc,
        out_desc=out_desc,
        out_ptr=c,
        #
        bias_ptr=bias,
        bias_stride=bias.stride(0),
        #
        gather_indx_ptr=gather_indx,
        #
        x_slice_sizes=a_ragged_metadata.slice_sizes,
        x_slice_offs=a_ragged_metadata.slice_offs,
        x_block_offs=x_block_offs,
        x_block_schedule=x_block_schedule,
        #
        x_scale_ptr=flex_ctx.lhs_data.scale,
        w_scale_ptr=flex_ctx.rhs_data.scale,
        out_scale_ptr=flex_ctx.out_data.expected_scale,
        #
        M=m,
        N=n,
        K=k,
        NUM_SLICES=a_ragged_metadata.n_slices,
        #
        SWIGLU_ALPHA=swiglu_alpha,
        SWIGLU_LIMIT=swiglu_limit,
        REDUCTION_N=reduction_n,
        #
        FLEXPOINT_SATURATE_INF=precision_config.flexpoint_saturate_inf,
        #
        BLOCK_M=p.BLOCK_M,
        BLOCK_N=p.BLOCK_N,
        BLOCK_K=p.BLOCK_K,
        NUM_SMS=launch_grid,
        X_NUM_BUFS=p.X_NUM_BUFS,
        W_NUM_BUFS=p.W_NUM_BUFS,
        W_SCALE_NUM_BUFS=p.W_SCALE_NUM_BUFS,
        ACC_NUM_BUFS=p.ACC_NUM_BUFS,
        #
        LOAD_ACTIVATION_WARPS=p.LOAD_ACTIVATION_WARPS,
        LOAD_WEIGHT_WARPS=p.LOAD_WEIGHT_WARPS,
        LOAD_WEIGHT_SCALES_WARPS=p.LOAD_WEIGHT_SCALES_WARPS,
        REPLAY_WARPS=p.REPLAY_WARPS,
        MMA_WARPS=p.MMA_WARPS,
        #
        LOAD_ACTIVATION_REGS=p.LOAD_ACTIVATION_REGS,
        LOAD_WEIGHT_REGS=p.LOAD_WEIGHT_REGS,
        LOAD_WEIGHT_SCALES_REGS=p.LOAD_WEIGHT_SCALES_REGS,
        REPLAY_REGS=p.REPLAY_REGS,
        MMA_REGS=p.MMA_REGS,
        #
        SCALE_SIZE_OUTER=p.SCALE_SIZE_OUTER,
        SCALE_SIZE_INNER=p.SCALE_SIZE_INNER,
        MXFP_BLOCK_SIZE=p.MXFP_BLOCK_SIZE,
        #
        num_warps=p.NUM_WARPS,
        num_ctas=1,
        maxnreg=p.MAXNREG,
    )

    return c


# ===-----------------------------------------------------------------------===#
# Benchmark and Testing Helpers
# ===-----------------------------------------------------------------------===#


def alloc_randn(shape: tuple[int, ...], dtype: torch.dtype, device: str) -> torch.Tensor:
    if dtype.itemsize == 1:
        return alloc_rand(shape, device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=dtype)


def alloc_randn_fp4(shape: tuple[int, ...], device: str, p: KernelConfig | None) -> tuple[Tensor, Tensor]:
    if p is not None:
        block_k, block_n, num_warps = p.BLOCK_K, p.BLOCK_N, p.NUM_WARPS
    else:
        block_k, block_n, num_warps = 128, 256, 8

    data = alloc_randn(shape, torch.bfloat16, device)
    data, scale = downcast_to_mxfp(data, FP4, axis=1)  # type: ignore[arg-type]

    data_layout = BlackwellMX4ValuePackedShuffledLayout(block_k=block_k, block_n=block_n)
    scale_layout = make_default_matmul_mxfp4_w_scale_layout(mx_axis=1, num_warps=num_warps)

    data = convert_layout(wrap_torch_tensor(data, dtype=FP4), data_layout)
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout)
    return data, scale


def make_prod_like_logits(
    batch_size: int,
    num_experts: int,
    experts_per_token: int,
    device: str,
    dtype: torch.dtype = torch.float16,
    *,
    zipf_alpha: float = 1.10,
    num_clusters: int = 16,
    cluster_boost: float = 1.25,
    gumbel_scale: float = 0.75,
    batch_hot_experts: int = 4,
    batch_hot_boost: float = 0.6,
) -> torch.Tensor:
    ranks = torch.arange(1, num_experts + 1, device=device, dtype=torch.float32)
    ranked_probs = ranks.pow(-zipf_alpha)
    ranked_probs /= ranked_probs.sum()

    perm = torch.randperm(num_experts, device=device)
    expert_probs = torch.empty_like(ranked_probs)
    expert_probs[perm] = ranked_probs

    logits = expert_probs.clamp_min(1e-12).log()[None, :].expand(batch_size, -1).clone()

    cluster_size = min(num_experts, max(2 * experts_per_token, num_experts // 16))
    cluster_experts = torch.stack(
        [torch.multinomial(expert_probs, cluster_size, replacement=False) for _ in range(num_clusters)]
    )
    token_cluster = torch.randint(num_clusters, (batch_size,), device=device)
    rows = torch.arange(batch_size, device=device)[:, None]
    logits[rows, cluster_experts[token_cluster]] += cluster_boost

    if batch_hot_experts > 0:
        hot = torch.multinomial(expert_probs, batch_hot_experts, replacement=False)
        logits[:, hot] += batch_hot_boost

    noise = -torch.empty_like(logits).exponential_().log()
    logits += gumbel_scale * noise

    return logits.to(dtype)


FROZEN_SLICE_SIZES = [2, 11, 1, 2, 1, 0, 3, 7, 5, 13, 1, 0, 1, 0, 3, 4]

# Frozen from the original failing top-k/routing setup. Only the first
# sum(FROZEN_SLICE_SIZES)=54 rows are actually scheduled locally; keeping the
# larger gather length is part of the trigger.
FROZEN_GATHER_INDX = [
    306, 151, 194, 322, 378, 425, 123, 193, 201, 330, 410, 481, 5, 18, 52, 85,
    147, 152, 180, 205, 210, 237, 249, 410, 464, 486, 500, 42, 83, 86, 335, 305,
    198, 299, 118, 361, 81, 26, 453, 4, 429, 306, 66, 170, 12, 131, 260, 344,
    377, 379, 436, 342, 139, 83, 135, 187, 277, 366, 377, 383, 444, 452, 460, 1,
    3, 34, 39, 40, 61, 62, 68, 71, 72, 74, 87, 110, 120, 122, 125, 129,
    157, 171, 178, 182, 186, 191, 197, 202, 208, 212, 213, 215, 223, 231, 233, 241,
    252, 274, 275, 276, 284, 294, 297, 315, 323, 325, 326, 328, 337, 343, 359, 362,
    369, 370, 374, 381, 385, 393, 394, 409, 413, 416, 423, 448, 470, 475, 477, 493,
    504, 511, 2, 29, 55, 133, 146, 186, 209, 265, 278, 305, 316, 417, 458, 498,
    503, 26, 59, 112, 113, 114, 117, 121, 124, 152, 155, 190, 211, 227, 248, 256,
    266, 269, 271, 273, 280, 281, 288, 298, 299, 303, 310, 329, 332, 333, 340, 341,
    346, 354, 360, 365, 371, 380, 382, 384, 401, 411, 430, 438, 442, 446, 447, 449,
    453, 463, 473, 484, 505, 507, 509, 510, 17, 21, 22, 27, 38, 45, 47, 49,
    58, 70, 77, 80, 82, 95, 97, 98, 99, 106, 107, 108, 119, 127, 128, 137,
    140, 144, 164, 169, 176, 198, 214, 224, 226, 229, 250, 257, 264, 270, 286, 291,
    302, 317, 324, 336, 339, 345, 352, 356, 364, 373, 386, 389, 396, 397, 400, 404,
    414, 419, 433, 434, 451, 465, 476, 488, 506, 13, 14, 15, 20, 24, 28, 31,
    32, 36, 43, 53, 56, 67, 75, 84, 96, 103, 105, 115, 126, 132, 142, 143,
    145, 148, 149, 150, 153, 159, 161, 168, 173, 174, 177, 184, 189, 196, 200, 203,
    217, 219, 220, 221, 238, 244, 253, 259, 272, 277, 279, 282, 285, 289, 296, 307,
    311, 312, 320, 331, 334, 338, 348, 349, 350, 351, 353, 355, 368, 372, 375, 376,
    387, 398, 399, 402, 406, 420, 421, 427, 428, 431, 437, 454, 459, 466, 469, 474,
    478, 479, 480, 487, 489, 497, 508, 25, 48, 60, 76, 89, 91, 92, 93, 94,
    100, 104, 109, 116, 130, 136, 138, 141, 154, 163, 170, 172, 175, 188, 192, 194,
    201, 206, 207, 210, 225, 228, 232, 234, 236, 239, 240, 242, 245, 246, 247, 251,
    258, 263, 267, 283, 287, 290, 292, 293, 295, 300, 304, 308, 318, 319, 321, 327,
    330, 335, 347, 357, 358, 367, 378, 388, 390, 391, 392, 395, 403, 407, 408, 412,
    415, 422, 426, 432, 435, 440, 441, 443, 445, 450, 455, 456, 457, 462, 467, 471,
    472, 481, 482, 483, 485, 486, 491, 494, 496, 499, 501, 502, 0, 6, 7, 8,
    9, 10, 11, 16, 19, 23, 30, 33, 35, 37, 41, 44, 46, 50, 51, 54,
    57, 63, 64, 65, 66, 69, 73, 78, 79, 88, 90, 101, 102, 111, 118, 123,
    131, 134, 135, 147, 156, 158, 160, 162, 166, 167, 179, 181, 183, 185, 187, 193,
    195, 199, 204, 205, 216, 218, 222, 230, 235, 237, 243, 254, 255, 260, 261, 262,
    268, 301, 306, 309, 313, 314, 322, 342, 344, 361, 363, 366, 377, 379, 383, 393,
    394, 405, 410, 411, 418, 424, 425, 436, 444, 452, 460, 461, 464, 468, 477, 484,
    490, 492, 495, 500, 504, 505, 507, 509, 18, 52, 85, 147, 152, 180, 205, 210,
    237, 249, 410, 464, 486, 500, 42, 83, 86, 335, 305, 198, 299, 118, 361, 81,
    26, 453, 4, 429, 306, 66, 170, 12, 131, 260, 344, 377, 379, 436, 342, 139,
    83, 135, 187, 277, 366, 377, 383, 444, 452, 460, 1, 3, 34, 39, 40, 61,
    62, 68, 71, 72, 74, 87, 110, 120, 122, 125, 129, 157, 171, 178, 182, 186,
    191, 197, 202, 208, 212, 213, 215, 223, 231, 233, 241, 252, 274, 275, 276, 284,
    294, 297, 315, 323, 325, 326, 328, 337, 343, 359, 362, 369, 370, 374, 381, 385,
    393, 394, 409, 413, 416, 423, 448, 470, 475, 477, 493, 504, 511, 2, 29, 55,
    133, 146, 186, 209, 265, 278, 305, 316, 417, 458, 498, 503, 26, 59, 112, 113,
    114, 117, 121, 124, 152, 155, 190, 211, 227, 248, 256, 266, 269, 271, 273, 280,
    281, 288, 298, 299, 303, 310, 329, 332, 333, 340, 341, 346, 354, 360, 365, 371,
    380, 382, 384, 401, 411, 430, 438, 442, 446, 447, 449, 453, 463, 473, 484, 505,
    507, 509, 510, 17, 21, 22, 27, 38, 45, 47, 49, 58, 70, 77, 80, 82,
    95, 97, 98, 99, 106, 107, 108, 119, 127, 128, 137, 140, 144, 164, 169, 176,
    198, 214, 224, 226, 229, 250, 257, 264, 270, 286, 291, 302, 317, 324, 336, 339,
    345, 352, 356, 364, 373, 386, 389, 396, 397, 400, 404, 414, 419, 433, 434, 451,
    465, 476, 488, 506, 13, 14, 15, 20, 24, 28, 31, 32, 36, 43, 53, 56,
    67, 75, 84, 96, 103, 105, 115, 126, 132, 142, 143, 145, 148, 149, 150, 153,
    159, 161, 168, 173, 174, 177, 184, 189, 196, 200, 203, 217, 219, 220, 221, 238,
    244, 253, 259, 272, 277, 279, 282, 285, 289, 296, 307, 311, 312, 320, 331, 334,
    338, 348, 349, 350, 351, 353, 355, 368, 372, 375, 376, 387, 398, 399, 402, 406,
    420, 421, 427, 428, 431, 437, 454, 459, 466, 469, 474, 478, 479, 480, 487, 489,
    497, 508, 25, 48, 60, 76, 89, 91, 92, 93, 94, 100, 104, 109, 116, 130,
    136, 138, 141, 154, 163, 170, 172, 175, 188, 192, 194, 201, 206, 207, 210, 225,
    228, 232, 234, 236, 239, 240, 242, 245, 246, 247, 251, 258, 263, 267, 283, 287,
    290, 292, 293, 295, 300, 304, 308, 318, 319, 321, 327, 330, 335, 347, 357, 358,
    367, 378, 388, 390, 391, 392, 395, 403, 407, 408, 412, 415, 422, 426, 432, 435,
    440, 441, 443, 445, 450, 455, 456, 457, 462, 467, 471, 472, 481, 482, 483, 485,
    486, 491, 494, 496, 499, 501, 502, 0, 6, 7, 8, 9, 10, 11, 16, 19,
    23, 30, 33, 35, 37, 41, 44, 46, 50, 51, 54, 57, 63, 64, 65, 66,
    69, 73, 78, 79, 88, 90, 101, 102, 111, 118, 123, 131, 134, 135, 147, 156,
    158, 160, 162, 166, 167, 179, 181, 183, 185, 187, 193, 195, 199, 204, 205, 216,
    218, 222, 230, 235, 237, 243, 254, 255, 260, 261, 262, 268, 301, 306, 309, 313,
    314, 322, 342, 344, 361, 363, 366, 377, 379, 383, 393, 394, 405, 410, 411, 418,
    424, 425, 436, 444, 452, 460, 461, 464, 468, 477, 484, 490, 492, 495, 500, 504,
    505, 507, 509, 126, 168, 490, 41, 142, 23, 31, 63, 100, 108, 120, 130, 144,
    150, 165, 176, 200, 215, 217, 221, 252, 287, 310, 328, 340, 351, 353, 402, 415,
    421, 430, 446, 495, 185, 47, 119, 192, 510, 161, 172, 280, 19, 290, 405, 456,
    297, 412, 398, 40, 271, 173, 241, 376, 355, 284, 44, 64, 234, 351, 439, 466,
    296, 322, 437,
]

def run_repro(max_launches: int = 1000):
    torch.cuda.set_device(0)
    device = "cuda:0"
    torch.manual_seed(0)

    hidden_size = 17 * 128
    intermediate_size = 5760
    batch_size = 512

    p = KernelConfig()
    n_expts_local = len(FROZEN_SLICE_SIZES)

    # Preserve the original x/w/bias RNG stream while using frozen routing data.
    torch.randint(0, 8, size=(), device=device)
    _ = make_prod_like_logits(batch_size, 128, 4, device)

    local_expts_hist = torch.tensor(FROZEN_SLICE_SIZES, dtype=torch.int32, device=device)
    gather_indx = torch.tensor(FROZEN_GATHER_INDX, dtype=torch.int32, device=device)
    ragged_metadata = make_ragged_tensor_metadata(local_expts_hist, gather_indx.numel())
    ragged_metadata.expected_slice_size = 16

    x = alloc_randn((batch_size, hidden_size), dtype=torch.float8_e4m3fn, device=device)
    w, w_scale = alloc_randn_fp4((n_expts_local, hidden_size, intermediate_size), device=device, p=p)
    bias = alloc_randn((n_expts_local, intermediate_size), dtype=torch.float32, device=device)

    fused_activation = FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (1.1, 1.3),
    )
    precision = PrecisionConfig(
        flexpoint_saturate_inf=True,
        b_mx_scale=w_scale,
        b_microblock_size=MXFP_BLOCK_SIZE.value,
        out_dtype=torch.float8_e4m3fn,
        flex_ctx=FlexCtx(
            lhs_data=InFlexData(dtype=torch.float8_e4m3fn, scale=torch.tensor([1.0], device=device)),
            rhs_data=InFlexData(),
            out_data=OutFlexData(dtype=torch.float8_e4m3fn, expected_scale=torch.tensor([4.0], device=device)),
        ),
    )
    out_shape = (gather_indx.shape[0], bias.shape[1] // fused_activation.specs.reduction_n)
    out = torch.zeros(out_shape, dtype=torch.float8_e4m3fn, device=device)

    matmul(
        a=x,
        b=w,
        bias=bias,
        a_ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        precision_config=precision,
        c=out,
        fused_activation=fused_activation,
    )
    torch.cuda.synchronize()
    expected = out.clone()

    for launch in range(1, max_launches + 1):
        out.zero_()
        matmul(
            a=x,
            b=w,
            bias=bias,
            a_ragged_metadata=ragged_metadata,
            gather_indx=gather_indx,
            precision_config=precision,
            c=out,
            fused_activation=fused_activation,
        )
        torch.cuda.synchronize()
        if not torch.equal(out, expected):
            maxdiff = (out.to(torch.float32) - expected.to(torch.float32)).abs().max().item()
            print(f"FAIL launch={launch} maxdiff={maxdiff}")
            return 1
    print(f"PASS launch={max_launches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_repro())
