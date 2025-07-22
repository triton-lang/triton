import pytest
import torch
import tempfile

import triton
from triton.backends.compiler import GPUTarget
from triton._internal_testing import is_blackwell

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_32x32b_reg_layout,
    mbarrier,
    tcgen05_mma,
    tcgen05_commit,
)


@pytest.mark.skipif(not is_blackwell(), reason="Requires compute capability == 10")
def test_tmem_copy_2d():
    device = "cuda"

    smem_h = 256
    num_cols = smem_h * 4 // 32

    copy_ops = """
%93 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
ttng.init_barrier %93, 1 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
%tmem_alloc = ttng.tmem_alloc {{tensor_memory_offset = 0 : i32}}: () -> !ttg.memdesc<128x{num_cols}xi32, #tmem, #ttng.tensor_memory, mutable>
ttng.tmem_copy %17, %tmem_alloc, %93 : (!ttg.memdesc<{smem_h}x4xi32, #shared, #ttg.shared_memory>, !ttg.memdesc<128x{num_cols}xi32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>) -> ()

%c0_i32 = arith.constant 0 : i32
ttng.wait_barrier %93, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
    """.format(num_cols=num_cols, smem_h=smem_h)

    ir_body = """

    %cst = arith.constant dense<4> : tensor<{smem_h}x1xi32, #blocked>
    %0 = tt.make_range {{end = {smem_h} : i32, start = 0 : i32}} : tensor<{smem_h}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
    %1 = tt.make_range {{end = 4 : i32, start = 0 : i32}} : tensor<4xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
    %2 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<{smem_h}x4x!tt.ptr<i32>, #blocked>
    %3 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x{num_cols}x!tt.ptr<i32>, #blocked>

    %4 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{smem_h}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>> -> tensor<{smem_h}x1xi32, #blocked>
    %5 = arith.muli %4, %cst : tensor<{smem_h}x1xi32, #blocked>
    %6 = tt.expand_dims %1 {{axis = 0 : i32}} : tensor<4xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>> -> tensor<1x4xi32, #blocked>
    %7 = tt.broadcast %6 : tensor<1x4xi32, #blocked> -> tensor<{smem_h}x4xi32, #blocked>
    %8 = tt.broadcast %5 : tensor<{smem_h}x1xi32, #blocked> -> tensor<{smem_h}x4xi32, #blocked>
    %9 = arith.addi %8, %7 : tensor<{smem_h}x4xi32, #blocked>
    %10 = tt.addptr %2, %9 : tensor<{smem_h}x4x!tt.ptr<i32>, #blocked>, tensor<{smem_h}x4xi32, #blocked>

    %01 = tt.make_range {{end = 128 : i32, start = 0 : i32}} : tensor<128xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
    %41 = tt.expand_dims %01 {{axis = 1 : i32}} : tensor<128xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>> -> tensor<128x1xi32, #blocked>
    %cst1 = arith.constant dense<{num_cols}> : tensor<128x1xi32, #blocked>
    %51 = arith.muli %41, %cst1 : tensor<128x1xi32, #blocked>
    %31 = tt.make_range {{end = {num_cols} : i32, start = 0 : i32}} : tensor<{num_cols}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
    %21 = tt.expand_dims %31 {{axis = 0 : i32}} : tensor<{num_cols}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>> -> tensor<1x{num_cols}xi32, #blocked>
    %71 = tt.broadcast %21 : tensor<1x{num_cols}xi32, #blocked> -> tensor<128x{num_cols}xi32, #blocked>
    %81 = tt.broadcast %51 : tensor<128x1xi32, #blocked> -> tensor<128x{num_cols}xi32, #blocked>
    %91 = arith.addi %81, %71 : tensor<128x{num_cols}xi32, #blocked>
    %14 = tt.addptr %3, %91 : tensor<128x{num_cols}x!tt.ptr<i32>, #blocked>, tensor<128x{num_cols}xi32, #blocked>

    %11 = tt.load %10 : tensor<{smem_h}x4x!tt.ptr<i32>, #blocked>
    %17 = ttg.local_alloc %11 : (tensor<{smem_h}x4xi32, #blocked>) -> !ttg.memdesc<{smem_h}x4xi32, #shared, #ttg.shared_memory>
    {copy_ops}
    %22 = ttng.tmem_load %tmem_alloc : !ttg.memdesc<128x{num_cols}xi32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x{num_cols}xi32, #blocked>
    tt.store %14, %22 : tensor<128x{num_cols}x!tt.ptr<i32>, #blocked>

    tt.return

    """.format(copy_ops=copy_ops, num_cols=num_cols, smem_h=smem_h)

    ir = """
    #blocked = #ttg.blocked<{sizePerThread=[1, 4], threadsPerWarp=[32, 1], warpsPerCTA=[4, 1], order=[0, 1]}>
    #shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
    #shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
    #tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 32, unpacked = false>
    module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    tt.func public @kernel_0d1d(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
    """ + ir_body + """
    }
    }
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name, target=GPUTarget("cuda", 100, 32))

    x = torch.randint(size=(smem_h, 4), low=-100, high=100, dtype=torch.int32).to(device)
    z_tri = torch.zeros(size=(128, num_cols), dtype=torch.int32).to(device)
    kernel[(1, 1, 1)](x, z_tri)

    num_rep_m = smem_h // 32

    for m in range(num_rep_m):
        col_offset = m * 4
        for i in range(4):
            # Copied values are duplicated across warps
            assert torch.equal(x[m * 32:(m + 1) * 32], z_tri[32 * i:32 * (i + 1), col_offset:(col_offset + 4)])


def test_tmem_subslice_block_m_64():

    @gluon.jit
    def kernel(s_ptr, o_ptr, out_ptr, acc_ptr):
        BLOCK_M: ttgl.constexpr = 64
        N: ttgl.constexpr = 128
        BLOCK_N: ttgl.constexpr = 64

        tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, BLOCK_N), unpacked=True)
        s_tmem = allocate_tensor_memory(ttgl.float32, (BLOCK_M, N), layout=tmem_layout)
        # o_tmem = allocate_tensor_memory(ttgl.float32, (BLOCK_M, N), layout=tmem_layout)

        layout: ttgl.constexpr = get_tmem_32x32b_reg_layout(BLOCK_M, BLOCK_N, (BLOCK_M, N), num_warps=4)

        offsets = ttgl.arange(0, BLOCK_M)[:, None] * N + ttgl.arange(0, N)[None, :]
        offsets = ttgl.convert_layout(offsets, layout)
        s = ttgl.load(s_ptr + offsets)
        o = ttgl.load(o_ptr + offsets)

        s_tmem.store(s)
        # o_tmem.store(o)

        p_tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, BLOCK_N), unpacked=False)
        p_tmem = s_tmem.slice(0, N // 2)._reinterpret(ttgl.float16, [BLOCK_M, N], p_tmem_layout)
        # p_tmem.store(ttgl.full((BLOCK_M, N), 0.0, dtype=ttgl.float16, layout=layout))

        d1_tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, 1), unpacked=True)
        d1_layout: ttgl.constexpr = get_tmem_32x32b_reg_layout(BLOCK_M, 1, (BLOCK_M, 1), num_warps=4)

        m_tmem = s_tmem.slice(N // 4, 1)._reinterpret(ttgl.float32, [BLOCK_M, 1], d1_tmem_layout)
        m_tmem.store(ttgl.full((BLOCK_M, 1), 2.0, dtype=ttgl.float32, layout=d1_layout))
        # l_tmem = s_tmem.slice(N // 4 + 1, 1)._reinterpret(ttgl.float32, [BLOCK_M, 1], d1_tmem_layout)
        # l_tmem.store(ttgl.full((BLOCK_M, 1), 3.0, dtype=ttgl.float32, layout=d1_layout))
        # a_tmem = s_tmem.slice(N // 4 + 2, 1)._reinterpret(ttgl.float32, [BLOCK_M, 1], d1_tmem_layout)
        # a_tmem.store(ttgl.full((BLOCK_M, 1), 4.0, dtype=ttgl.float32, layout=d1_layout))

        s = s_tmem.load(layout)


        pa_tmem = p_tmem
        pb_tmem = p_tmem.slice(N // 2, N)

        # TMEM[0:16]  = [p0, p1, o0]
        # TMEM[16:32] = [p1, p0, o1]
        p = offsets.to(ttgl.float16)
        pa_tmem.store(p)
        p0, p1 = p.reshape((BLOCK_M, 2, N // 2)).permute(0, 2, 1).split()
        p10 = ttgl.join(p0, p1).permute(0, 2, 1).reshape((BLOCK_M, N))
        pb_tmem.store(ttgl.convert_layout(p10, layout, assert_trivial=True))

        # shared_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=16, rank=2)

        # lhs = ttgl.allocate_shared_memory(ttgl.float16, (BLOCK_M, N), shared_layout)
        # rhs = ttgl.allocate_shared_memory(ttgl.float16, (N, N), shared_layout)

        # lhs.store(ttgl.full((BLOCK_M, N), 1.0, dtype=ttgl.float16, layout=layout))
        # rhs.store(ttgl.full((N, N), 1.0, dtype=ttgl.float16, layout=layout))

        bar = ttgl.allocate_shared_memory(ttgl.int64, [1], ttgl.constexpr(mbarrier.MBarrierLayout()))
        mbarrier.init(bar, count=1)

        # o0_tmem = o_tmem.slice(0, N // 2)
        # o1_tmem = o_tmem.slice(N // 2, N // 2)

        # rhs0 = rhs.slice(0, N // 2, dim=1)
        # rhs1 = rhs.slice(N // 2, N // 2, dim=1)

        # tcgen05_mma(pa_tmem.slice(0, N // 2), rhs0.slice(0, N // 2), o0_tmem)
        # tcgen05_mma(pb_tmem.slice(0, N // 2), rhs0.slice(N // 2, N // 2), o0_tmem)
        # tcgen05_mma(pa_tmem.slice(N // 2, N // 2), rhs1.slice(0, N // 2), o1_tmem)
        # tcgen05_mma(pb_tmem.slice(N // 2, N // 2), rhs1.slice(N // 2, N // 2), o1_tmem)

        # tcgen05_commit(bar)
        # mbarrier.wait(bar, 0)
        # mbarrier.invalidate(bar)

        # o = o_tmem.load(layout)
        # ttgl.store(acc_ptr + offsets, o)

        ttgl.store(out_ptr + offsets, s)

    torch.manual_seed(0)
    s = torch.randn((64, 128), dtype=torch.float32, device="cuda")
    o = torch.randn((64, 128), dtype=torch.float32, device="cuda")

    out_tri = torch.empty_like(s)
    acc_tri = torch.empty_like(o)
    compiled = kernel[(1, )](s, o, out_tri, acc_tri)

    ttgir = compiled.asm["ttgir"]
    # Check that we have two 64x128xf32 allocations.
    # print(ttgir)
    # assert ttgir.count("ttng.tmem_alloc") == 2
    # assert ttgir.count("ttng.tmem_alloc : () -> !ttg.memdesc<64x128xf32") == 2

    # Check that we allocated only 128 columns of TMEM.
    llir = compiled.asm["llir"]
    # assert llir.count("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$1], 128")

    # Given TMEM[0:32] is the slice of TMEM for warpgroup 0, the expected layout
    # of S is
    #
    #   TMEM[0:16]  = S[0:16, 0:64]
    #   TMEM[16:32] = S[0:16, 64:128]
    #
    # When slicing S to obtain P, we expect it to overlap with the left half,
    # i.e. S[0:16, 0:32] and S[0:16, 64:96].
    out_ref = s
    # out_ref[:, 0:32] = 0.0
    # out_ref[:, 64:96] = 0.0

    # Given S = [s0, s1, s2, s3], they are arranged like
    #
    #   TMEM[0:16]  = [s0, s1]
    #   TMEM[16:32] = [s2, s3]
    #
    # Thus slicing S at  N//4 will obtain an offset to the beginning of s1.
    out_ref[:, 32] = 2.0
    # out_ref[:, 33] = 3.0
    # out_ref[:, 34] = 4.0

    torch.set_printoptions(threshold=10000)
    print(out_tri[3, :])
    print(out_ref[3, :])

    torch.testing.assert_close(out_ref, out_tri, atol=0, rtol=0)

    # print(acc_tri)


test_tmem_subslice_block_m_64()
