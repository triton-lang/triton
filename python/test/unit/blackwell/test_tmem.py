import pytest
import torch
import tempfile

import triton
from triton.backends.compiler import GPUTarget

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


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 10, reason="Requires compute capability == 10")
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


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 10, reason="Requires compute capability == 10")
def test_tmem_subslice_block_m_64():

    @gluon.jit
    def kernel(s_ptr, out_ptr):
        BLOCK_M: ttgl.constexpr = 64
        N: ttgl.constexpr = 128
        BLOCK_N: ttgl.constexpr = 64

        tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, BLOCK_N), unpacked=True)
        s_tmem = allocate_tensor_memory(ttgl.float32, (BLOCK_M, N), layout=tmem_layout)
        o_tmem = allocate_tensor_memory(ttgl.float32, (BLOCK_M, N), layout=tmem_layout)

        layout: ttgl.constexpr = get_tmem_32x32b_reg_layout(BLOCK_M, BLOCK_N, (BLOCK_M, N), num_warps=4)

        offsets = ttgl.arange(0, BLOCK_M)[:, None] * N + ttgl.arange(0, N)[None, :]
        offsets = ttgl.set_auto_layout(offsets, layout)
        s = ttgl.load(s_ptr + offsets)

        s_tmem.store(s)
        o_tmem.store(s)

        p_tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, BLOCK_N), unpacked=False)
        p_tmem = s_tmem.slice(0, N // 2)._reinterpret(ttgl.float16, [BLOCK_M, N], p_tmem_layout)
        p_tmem.store(ttgl.full((BLOCK_M, N), 0.0, dtype=ttgl.float16, layout=layout))

        d1_tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, 1), unpacked=True)
        d1_layout: ttgl.constexpr = get_tmem_32x32b_reg_layout(BLOCK_M, 1, (BLOCK_M, 1), num_warps=4)

        m_tmem = s_tmem.slice(N // 4, 1)._reinterpret(ttgl.float32, [BLOCK_M, 1], d1_tmem_layout)
        m_tmem.store(ttgl.full((BLOCK_M, 1), 2.0, dtype=ttgl.float32, layout=d1_layout))
        l_tmem = s_tmem.slice(N // 4 + 1, 1)._reinterpret(ttgl.float32, [BLOCK_M, 1], d1_tmem_layout)
        l_tmem.store(ttgl.full((BLOCK_M, 1), 3.0, dtype=ttgl.float32, layout=d1_layout))
        a_tmem = s_tmem.slice(N // 4 + 2, 1)._reinterpret(ttgl.float32, [BLOCK_M, 1], d1_tmem_layout)
        a_tmem.store(ttgl.full((BLOCK_M, 1), 4.0, dtype=ttgl.float32, layout=d1_layout))

        s = s_tmem.load(layout)

        ttgl.store(out_ptr + offsets, s)

    torch.manual_seed(0)
    s = torch.randn((64, 128), dtype=torch.float32, device="cuda")

    out_tri = torch.empty_like(s)
    compiled = kernel[(1, )](s, out_tri)

    ttgir = compiled.asm["ttgir"]
    # Check that we have two 64x128xf32 allocations.
    assert ttgir.count("ttng.tmem_alloc") == 2
    assert ttgir.count("ttng.tmem_alloc : () -> !ttg.memdesc<64x128xf32") == 2

    # Check that we allocated only 128 columns of TMEM.
    llir = compiled.asm["llir"]
    assert llir.count("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$1], 128")

    # Given TMEM[0:32] is the slice of TMEM for warpgroup 0, the expected layout
    # of S is
    #
    #   TMEM[0:16]  = S[0:16, 0:64]
    #   TMEM[16:32] = S[0:16, 64:128]
    #
    # When slicing S to obtain P, we expect it to overlap with the left half,
    # i.e. S[0:16, 0:32] and S[0:16, 64:96].
    out_ref = s
    out_ref[:, 0:32] = 0.0
    out_ref[:, 64:96] = 0.0

    # Given S = [s0, s1, s2, s3], they are arranged like
    #
    #   TMEM[0:16]  = [s0, s1]
    #   TMEM[16:32] = [s2, s3]
    #
    # Thus slicing S at  N//4 will obtain an offset to the beginning of s1.
    out_ref[:, 32] = 2.0
    out_ref[:, 33] = 3.0
    out_ref[:, 34] = 4.0

    torch.testing.assert_close(out_ref, out_tri, atol=0, rtol=0)


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 10, reason="Requires compute capability == 10")
def test_block_m_64_mma():

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, d_ptr):
        BLOCK_M: ttgl.constexpr = 64
        N: ttgl.constexpr = 128
        BLOCK_N: ttgl.constexpr = 64

        a_offsets = ttgl.arange(0, BLOCK_M)[:, None] * N + ttgl.arange(0, N)[None, :]
        b_offsets = ttgl.arange(0, N)[:, None] * N + ttgl.arange(0, N)[None, :]

        a_layout: ttgl.constexpr = get_tmem_32x32b_reg_layout(BLOCK_M, BLOCK_N, (BLOCK_M, N), num_warps=4)
        b_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
        a_offsets = ttgl.set_auto_layout(a_offsets, a_layout)
        b_offsets = ttgl.set_auto_layout(b_offsets, b_layout)

        a = ttgl.load(a_ptr + a_offsets)
        b = ttgl.load(b_ptr + b_offsets)
        c = ttgl.load(c_ptr + a_offsets)

        a_tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, BLOCK_N), unpacked=False)
        acc_tmem_layout: ttgl.constexpr = TensorMemoryLayout((BLOCK_M, BLOCK_N), unpacked=True)
        al_tmem = allocate_tensor_memory(ttgl.float16, (BLOCK_M, N), layout=a_tmem_layout)
        ar_tmem = allocate_tensor_memory(ttgl.float16, (BLOCK_M, N), layout=a_tmem_layout)
        acc_tmem = allocate_tensor_memory(ttgl.float32, (BLOCK_M, N), layout=acc_tmem_layout)

        a0, a1 = a.reshape((BLOCK_M, 2, N // 2)).permute(0, 2, 1).split()

        al = ttgl.join(a0, a1).permute(0, 2, 1).reshape((BLOCK_M, N))
        ar = ttgl.join(a1, a0).permute(0, 2, 1).reshape((BLOCK_M, N))

        al_tmem.store(ttgl.convert_layout(al, a_layout, assert_trivial=True))
        ar_tmem.store(ttgl.convert_layout(ar, a_layout, assert_trivial=True))

        b_shared_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=16, rank=2)
        b_shared = ttgl.allocate_shared_memory(ttgl.float16, [N, N], layout=b_shared_layout)
        b_shared.store(b)

        acc_tmem.store(c)

        bar = ttgl.allocate_shared_memory(ttgl.int64, [1], ttgl.constexpr(mbarrier.MBarrierLayout()))
        mbarrier.init(bar, count=1)

        # This is a manually tiled MMA where LHS is in TMEM with blockM=64,
        # where we circumvent the limitation that LHS and accumulator need to
        # share the same TMEM rows by storing the LHS twice.
        #
        # TMEM      al   ar   c
        # [0, 16)   a0   a1   c0
        # [16, 32)  a1   a0   c1
        #
        # d0 = a0 @ b00 + a1 @ b10 + c0
        # d1 = a0 @ b10 + a1 @ b11 + c1

        N2: ttgl.constexpr = N // 2
        c0 = acc_tmem.slice(0, N2)
        c1 = acc_tmem.slice(N2, N2)

        tcgen05_mma(al_tmem.slice(0, N2), b_shared.slice(0, N2, dim=0).slice(0, N2, dim=1), c0)
        tcgen05_mma(ar_tmem.slice(0, N2), b_shared.slice(N2, N2, dim=0).slice(0, N2, dim=1), c0)
        tcgen05_mma(ar_tmem.slice(N2, N2), b_shared.slice(0, N2, dim=0).slice(N2, N2, dim=1), c1)
        tcgen05_mma(al_tmem.slice(N2, N2), b_shared.slice(N2, N2, dim=0).slice(N2, N2, dim=1), c1)

        tcgen05_commit(bar)
        mbarrier.wait(bar, 0)
        mbarrier.invalidate(bar)

        d = acc_tmem.load(a_layout)
        ttgl.store(d_ptr + a_offsets, d)

    torch.manual_seed(0)
    a = torch.randn((64, 128), dtype=torch.float16, device="cuda")
    b = torch.randn((128, 128), dtype=torch.float16, device="cuda")
    c = torch.randn((64, 128), dtype=torch.float32, device="cuda")

    d_tri = torch.empty_like(c)
    compiled = kernel[(1, )](a, b, c, d_tri)

    ttgir = compiled.asm["ttgir"]
    assert ttgir.count("ttng.tmem_alloc") == 3
    assert ttgir.count("ttng.tmem_alloc : () -> !ttg.memdesc<64x128xf32") == 1
    assert ttgir.count("ttng.tmem_alloc : () -> !ttg.memdesc<64x128xf16") == 2

    llir = compiled.asm["llir"]
    assert llir.count("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$1], 128")

    d_ref = a @ b + c
    torch.testing.assert_close(d_ref, d_tri, rtol=0.08, atol=0)
