import torch
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
import triton.experimental.gluon.language.nvidia.blackwell as blackwell
import triton.experimental.gluon.language.nvidia.blackwell.tma as tma
from triton.experimental.gluon.language.nvidia.blackwell import float2
import triton.experimental.gluon.language.nvidia.hopper.mbarrier as mbarrier

BLOCK_M = gl.constexpr(16)
BLOCK_N = gl.constexpr(128)
BLOCK_K = gl.constexpr(128)

@gluon.jit
def alloc_barrier():
    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=1)
    return bar

@gluon.jit
def ws_matmul_kernel(
    x_desc: tma.tensor_descriptor,
    out_ptr: gl.tensor,
):
    x_buf = gl.allocate_shared_memory(
        gl.float8e4nv,
        [BLOCK_M, BLOCK_K],
        gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8, rank=2),
    )
    x_ready_bar = alloc_barrier()

    replay_tmem = blackwell.allocate_tensor_memory(
        gl.uint32,
        [BLOCK_N, BLOCK_K // 4],
        blackwell.TensorMemoryLayout((128, BLOCK_K // 4), col_stride=1),
    )
    acc_buf = blackwell.allocate_tensor_memory(
        gl.float32,
        [BLOCK_N, BLOCK_M],
        blackwell.TensorMemoryLayout([128, BLOCK_M], col_stride=1),
    )
    acc_ready_bar = alloc_barrier()

    offs_layout: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]),
    )
    offs_m = gl.arange(0, BLOCK_M, layout=offs_layout)
    offs_x_m = gl.where(offs_m < 1, offs_m, 1)

    mbarrier.expect(x_ready_bar, BLOCK_M * BLOCK_K)
    tma.async_gather(x_desc, offs_x_m, 0, x_ready_bar, x_buf)

    replay_tmem.store(
        gl.full((BLOCK_N, BLOCK_K // 4), 1, dtype=gl.uint32, layout=replay_tmem.get_reg_layout())
    )

    scale_k: gl.constexpr = BLOCK_K // 32
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
        (128, BLOCK_K),
        col_stride=1,
        fp4_padded=True,
    )

    mbarrier.wait(x_ready_bar, 0)
    blackwell.tcgen05_mma_scaled(
        replay_tmem._reinterpret(gl.uint8, (BLOCK_N, BLOCK_K), k_second_u8_layout),
        x_buf.permute((1, 0)),
        acc_buf,
        w_scale_tmem,
        x_scale_tmem,
        a_type="e2m1",
        b_type="e4m3",
        use_acc=False,
        mbarriers=[x_ready_bar, x_ready_bar, x_ready_bar],
    )
    blackwell.tcgen05_commit(acc_ready_bar)

    split_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4],
        [1, 32],
        [gl.num_warps(), 1],
        [1, 0],
    )
    mbarrier.wait(acc_ready_bar, 0)
    acc_regs = acc_buf.load().permute((1, 0))
    acc = gl.convert_layout(acc_regs, split_layout)
    acc_packed = float2.pack(acc, axis=1)
    packed_fp8 = gl.inline_asm_elementwise(
        """
        {
            .reg .f32 lane<2>;
            mov.b64 {lane0, lane1}, $1;
            cvt.rn.satfinite.e4m3x2.f32 $0, lane1, lane0;
        }
        """,
        "=h,l",
        [acc_packed.value],
        dtype=gl.int16,
        is_pure=True,
        pack=1,
    )
    layout: gl.constexpr = packed_fp8.type.layout
    out_m = gl.arange(0, packed_fp8.shape[0], layout=gl.SliceLayout(1, layout))
    out_n = gl.arange(0, packed_fp8.shape[1], layout=gl.SliceLayout(0, layout))
    ptrs = out_ptr.cast(gl.pointer_type(gl.int16), bitcast=True)
    ptrs = ptrs + gl.expand_dims(out_m, 1) * (BLOCK_N // 2)
    ptrs = ptrs + gl.expand_dims(out_n, 0)
    gl.store(ptrs, packed_fp8)

def run_repro():
    x = torch.ones((1, 128), device="cuda:0", dtype=torch.float8_e4m3fn)
    out = torch.empty((16, 32), dtype=torch.int16, device="cuda:0")
    x_desc = TensorDescriptor(
        x,
        [1, 128],
        [128, 1],
        [1, 128],
        gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8, rank=2),
    )
    kernel = ws_matmul_kernel[(1,)]

    def run_kernel():
        kernel(
            x_desc=x_desc,
            out_ptr=out,
            num_warps=4,
        )

    run_kernel()
    expected = out.clone()

    run_kernel()
    if not torch.equal(out, expected):
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_repro())
