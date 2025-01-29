import pytest
import torch
import tempfile

import triton
from triton.backends.compiler import GPUTarget


def test_tmem_copy_2d():
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] == 10:
        pytest.skip("Test requires Blackwell target.")

    device = "cuda"

    smem_h = 256
    num_cols = smem_h * 4 // 32

    copy_ops = """
%93 = ttg.local_alloc  : () -> !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
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
    #shared = #ttg.shared<{vec=1, perPhase=1, maxPhase=1, order=[1, 0]}>
    #shared1 = #ttg.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
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
