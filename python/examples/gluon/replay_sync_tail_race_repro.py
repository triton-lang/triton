import torch
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
)
from triton_kernels.testing import alloc_rand

# ===-----------------------------------------------------------------------===#
# Device Code
# ===-----------------------------------------------------------------------===#

REDUCTION_N = gl.constexpr(2)
BLOCK_M = gl.constexpr(128)
BLOCK_N = gl.constexpr(256)
BLOCK_K = gl.constexpr(128)
K_TILES = gl.constexpr(3)
OUT_PACKED_N = gl.constexpr(64)
NUM_WARPS = 8
LOAD_WEIGHT_WARPS = gl.constexpr(1)
REPLAY_WARPS = gl.constexpr(4)
MMA_WARPS = gl.constexpr(4)
LOAD_WEIGHT_REGS = gl.constexpr(24)
REPLAY_REGS = gl.constexpr(80)
MMA_REGS = gl.constexpr(80)
MMA_BLOCK_COL = gl.constexpr(128)
MXFP_BLOCK_SIZE = gl.constexpr(32)

@gluon.jit
def alloc_barrier(count: gl.constexpr = 1):
    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=count)
    return bar

@gluon.aggregate
class PartitionArgs:
    x_desc: tma.tensor_descriptor
    w_desc: tma.tensor_descriptor

    out_ptr: gl.tensor

    x_buf: gl.shared_memory_descriptor
    x_empty_bar: gl.shared_memory_descriptor
    x_ready_bar: gl.shared_memory_descriptor

    w_buf: gl.shared_memory_descriptor
    w_empty_bar: gl.shared_memory_descriptor
    w_ready_bar: gl.shared_memory_descriptor

    replay_tmem: blackwell.tensor_memory_descriptor
    replay_empty_bar: gl.shared_memory_descriptor
    replay_full_bar: gl.shared_memory_descriptor
    acc_buf: blackwell.tensor_memory_descriptor
    acc_empty_bar: gl.shared_memory_descriptor
    acc_ready_bar: gl.shared_memory_descriptor


@gluon.jit
def load_activations(p: PartitionArgs):
    offs_layout: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]),
    )

    phase = 1

    offs_m = gl.arange(0, BLOCK_M, layout=offs_layout)

    mask_m = offs_m < 1
    offs_x_m = gl.where(mask_m, offs_m, p.x_desc.shape[0])

    for ki in range(K_TILES):
        off_k_x = ki * BLOCK_K
        mbarrier.wait(p.x_empty_bar, phase, pred=ki != 0)
        mbarrier.expect(p.x_ready_bar, p.x_desc.block_type.nbytes * BLOCK_M)
        tma.async_gather(
            p.x_desc,
            offs_x_m,
            off_k_x,
            p.x_ready_bar,
            p.x_buf,
        )

        phase = phase ^ 1


@gluon.jit
def load_weights(p: PartitionArgs):
    phase = 1

    pid_n = gl.program_id(0)

    for ki in range(0, K_TILES, 2):
        mbarrier.wait(p.w_empty_bar, phase, pred=ki != 0)
        mbarrier.expect(p.w_ready_bar, p.w_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(
            p.w_desc,
            [0, ki // 2, pid_n, 0, 0],
            p.w_ready_bar,
            p.w_buf,
        )

        phase = phase ^ 1

@gluon.jit
def mma_partition(p: PartitionArgs):
    scale_mma_bar = alloc_barrier()
    scale_k: gl.constexpr = BLOCK_K // MXFP_BLOCK_SIZE
    x_scale_tmem = blackwell.allocate_tensor_memory(
        gl.uint8,
        [BLOCK_M, scale_k],
        blackwell.TensorMemoryScalesLayout(),
    )
    w_scale_tmem = blackwell.allocate_tensor_memory(
        gl.uint8,
        [BLOCK_N, scale_k],
        blackwell.TensorMemoryScalesLayout(),
    )
    x_scale_tmem.store(
        gl.full((BLOCK_M, scale_k), 127, dtype=gl.uint8, layout=x_scale_tmem.get_reg_layout())
    )
    w_scale_tmem.store(
        gl.full((BLOCK_N, scale_k), 127, dtype=gl.uint8, layout=w_scale_tmem.get_reg_layout())
    )
    k_second_u8_layout: gl.constexpr = blackwell.TensorMemoryLayout(
        (MMA_BLOCK_COL, BLOCK_K),
        col_stride=1,
        fp4_padded=True,
    )

    mbarrier.wait(p.acc_empty_bar, 1)

    mbarrier.wait(p.x_ready_bar, 0)
    mbarrier.wait(p.w_ready_bar, 0)

    w_packed_pair = p.w_buf.reshape((BLOCK_N, BLOCK_K))
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
        p.x_buf.permute((1, 0)),
        p.acc_buf,
        w_scale_tmem,
        x_scale_tmem,
        a_type="e2m1",
        b_type="e4m3",
        use_acc=False,
        mbarriers=[p.x_empty_bar, scale_mma_bar, p.w_empty_bar],
    )

    mbarrier.wait(p.replay_full_bar, 0)

    mbarrier.wait(p.x_ready_bar, 1)
    blackwell.tcgen05_mma_scaled(
        p.replay_tmem._reinterpret(gl.uint8, (BLOCK_N, BLOCK_K), k_second_u8_layout),
        p.x_buf.permute((1, 0)),
        p.acc_buf,
        w_scale_tmem,
        x_scale_tmem,
        a_type="e2m1",
        b_type="e4m3",
        use_acc=True,
        mbarriers=[p.x_empty_bar, scale_mma_bar, p.replay_empty_bar],
    )

    mbarrier.wait(p.x_ready_bar, 0)
    mbarrier.wait(p.w_ready_bar, 1)

    w_packed_pair = p.w_buf.reshape((BLOCK_N, BLOCK_K))
    w_pair_first = w_packed_pair._reinterpret(
        gl.uint8,
        (BLOCK_N, BLOCK_K // 2),
        fp4_padded_layout,
    )

    blackwell.tcgen05_mma_scaled(
        w_pair_first,
        p.x_buf.permute((1, 0)),
        p.acc_buf,
        w_scale_tmem,
        x_scale_tmem,
        a_type="e2m1",
        b_type="e4m3",
        use_acc=True,
        mbarriers=[p.x_empty_bar, scale_mma_bar, p.w_empty_bar],
    )

    blackwell.tcgen05_commit(p.w_empty_bar)
    blackwell.tcgen05_commit(p.acc_ready_bar)


@gluon.jit
def replay_partition(p: PartitionArgs):
    dense_copy_done_bar = alloc_barrier()
    dense_replay_tmem = blackwell.allocate_tensor_memory(
        gl.uint32,
        [BLOCK_N, BLOCK_K // 4],
        blackwell.TensorMemoryLayout((MMA_BLOCK_COL, BLOCK_K // 4), col_stride=1),
    )
    dense_pair_layout: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=128,
        element_bitwidth=32,
        rank=2,
        fp4_padded=False,
    )
    mbarrier.wait(p.w_ready_bar, 0)
    mbarrier.wait(p.replay_empty_bar, 1)
    dense_pair_words = p.w_buf.reshape((BLOCK_N, BLOCK_K))._reinterpret(
        gl.uint32,
        (BLOCK_N, BLOCK_K // 4),
        dense_pair_layout,
    )
    blackwell.tcgen05_copy(dense_pair_words, dense_replay_tmem)
    blackwell.tcgen05_commit(dense_copy_done_bar)
    mbarrier.wait(dense_copy_done_bar, 0)

    blackwell.fence_async_shared()
    mbarrier.arrive(p.w_empty_bar)

    replay_cols: gl.constexpr = BLOCK_K // 4
    replay_segments: gl.constexpr = BLOCK_K // 16
    dense_frag = dense_replay_tmem.slice(0, replay_cols)
    replay_frag = p.replay_tmem.slice(0, replay_cols)
    dense_words = dense_frag.load()
    dense_words = dense_words.reshape((BLOCK_N, replay_segments, 2, 2)).permute((0, 1, 3, 2))
    _, dense_words = gl.split(dense_words)
    dense_words = dense_words.reshape((1, 1, 1, BLOCK_N, replay_segments, 2))
    lo, hi = gl.inline_asm_elementwise(
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
        args=[dense_words],
        dtype=(gl.uint32, gl.uint32),
        is_pure=True,
        pack=1,
    )
    k_second = gl.join(lo, hi).reshape([BLOCK_N, replay_cols])
    replay_frag.store(gl.convert_layout(k_second, replay_frag.get_reg_layout()))
    mbarrier.arrive(p.replay_full_bar)


@gluon.jit
def epilogue_partition(p: PartitionArgs):
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
    mbarrier.wait(p.acc_ready_bar, 0)
    acc_regs = p.acc_buf.load().permute((1, 0))
    mbarrier.arrive(p.acc_empty_bar)
    acc = gl.convert_layout(acc_regs, split_layout)
    acc_packed = float2.pack(acc, axis=1)
    packed_fp8 = gl.convert_layout(
        gl.inline_asm_elementwise(
            """
            {
                .reg .f32 lane<2>;
                mov.b64 {lane0, lane1}, $1;
                cvt.rn.satfinite.e4m3x2.f32 $0, lane1, lane0;
            }
            """,
            "=h,l",
            [(acc_packed * float2.full_like(acc_packed, out_recip)).value],
            dtype=gl.int16,
            is_pure=True,
            pack=1,
        ),
        store_layout,
    )
    lhs, rhs = gl.split(packed_fp8.reshape((packed_fp8.shape[0], packed_fp8.shape[1] // 2, 2)))
    values = gl.inline_asm_elementwise(
        """
        mov.b32 $0, { $1, $2 };
        """,
        "=r,h,h",
        [lhs, rhs],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )
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

    acc_layout: gl.constexpr = blackwell.TensorMemoryLayout(
        [MMA_BLOCK_COL, BLOCK_M],
        col_stride=1,
    )
    replay_layout: gl.constexpr = blackwell.TensorMemoryLayout(
        (MMA_BLOCK_COL, BLOCK_K // 4),
        col_stride=1,
    )
    x_buf = gl.allocate_shared_memory(
        x_desc.dtype,
        [BLOCK_M, x_desc.block_type.shape[1]],
        x_desc.layout,
    )
    x_empty_bar = alloc_barrier()
    x_ready_bar = alloc_barrier()

    w_buf = gl.allocate_shared_memory(
        w_desc.dtype,
        w_desc.block_type.shape,
        w_desc.layout,
    )
    w_empty_bar = alloc_barrier(count=2)
    w_ready_bar = alloc_barrier()

    replay_tmem = blackwell.allocate_tensor_memory(
        gl.uint32,
        [BLOCK_N, BLOCK_K // 4],
        replay_layout,
    )
    replay_empty_bar = alloc_barrier()
    replay_full_bar = alloc_barrier()

    acc_buf = blackwell.allocate_tensor_memory(
        gl.float32,
        [BLOCK_N, BLOCK_M],
        acc_layout,
    )
    acc_empty_bar = alloc_barrier()
    acc_ready_bar = alloc_barrier()

    p = PartitionArgs(
        x_desc=x_desc,
        w_desc=w_desc,
        out_ptr=out_ptr,
        #
        x_buf=x_buf,
        x_empty_bar=x_empty_bar,
        x_ready_bar=x_ready_bar,
        #
        w_buf=w_buf,
        w_empty_bar=w_empty_bar,
        w_ready_bar=w_ready_bar,
        #
        replay_tmem=replay_tmem,
        replay_empty_bar=replay_empty_bar,
        replay_full_bar=replay_full_bar,
        acc_buf=acc_buf,
        acc_empty_bar=acc_empty_bar,
        acc_ready_bar=acc_ready_bar,
    )

    scale_buf = gl.allocate_shared_memory(
        scale_desc.dtype,
        scale_desc.block_type.shape,
        scale_desc.layout,
    )
    scale_ready_bar = alloc_barrier()
    tma.async_copy_global_to_shared(
        scale_desc,
        [0, 0, 0, 0, 0],
        scale_ready_bar,
        scale_buf,
    )
    gl.warp_specialize(
        [
            (load_activations, (p,)),
            (load_weights, (p,)),
            (replay_partition, (p,)),
            (mma_partition, (p,)),
        ],
        [
            LOAD_WEIGHT_WARPS,
            REPLAY_WARPS,
            MMA_WARPS,
        ],
        [
            LOAD_WEIGHT_REGS,
            REPLAY_REGS,
            MMA_REGS,
        ],
    )
    epilogue_partition(p)

# ===-----------------------------------------------------------------------===#
# Benchmark and Testing Helpers
# ===-----------------------------------------------------------------------===#

def run_repro(max_launches: int = 1000):
    torch.cuda.set_device(0)
    device = "cuda:0"
    torch.manual_seed(0)

    x = alloc_rand((1, 384), device=device, dtype=torch.float8_e4m3fn)
    w_data = torch.randn((1, 384, 512), device=device, dtype=torch.bfloat16)
    w_data, _ = downcast_to_mxfp(w_data, FP4, axis=1)  # type: ignore[arg-type]
    w = convert_layout(
        wrap_torch_tensor(w_data, dtype=FP4),
        BlackwellMX4ValuePackedShuffledLayout(block_k=128, block_n=256),
    )
    scale_data = torch.zeros((1, 4, 3, 2, 256), dtype=torch.uint8, device=device)

    out = torch.zeros((1, 256), dtype=torch.float8_e4m3fn, device=device)
    x_desc = TensorDescriptor(
        x,
        list(x.shape),
        list(x.stride()),
        [1, 128],
        gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8, rank=2),
    )
    w_ptr = w.storage.data
    w_strides = list(w_ptr.stride())
    w_block_shape = w.storage.layout.swizzle_block_shape([1, 256, 256])
    w_block_shape[w_strides.index(1)] //= 2
    w_desc = TensorDescriptor(
        w_ptr,
        list(w_ptr.shape),
        w_strides,
        w_block_shape,
        gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8, rank=5, fp4_padded=False),
    )
    scale_desc = TensorDescriptor(
        scale_data,
        list(scale_data.shape),
        list(scale_data.stride()),
        [
            1,
            2,
            1,
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
            num_warps=NUM_WARPS,
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
