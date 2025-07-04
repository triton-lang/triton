import pytest
import torch
import triton
from triton._internal_testing import is_hip

num_ctas_list = [1]

GPU_DIALECT = "ttg"

if is_hip():
    THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size
else:
    THREADS_PER_WARP = 32


@pytest.mark.parametrize("M, N, M_tile_size, N_tile_size",
                         [[128, 128, 64, 64], [128, 128, 64, 32], [128, 64, 64, 32], [256, 128, 64, 64]])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_split_subview_split(dtype, M, N, M_tile_size, N_tile_size, device='cuda'):
    if not is_hip():
        pytest.skip("concat op is AMD specific instruction.")

    num_repeats_M = int(M / M_tile_size)
    num_repeats_N = int(N / N_tile_size)

    ir = f"""
    #blocked = #ttg.blocked<{{sizePerThread=[1, 8], threadsPerWarp=[16, 4], warpsPerCTA=[4, 1], order=[1, 0], CTAsPerCGA=[1, 1], CTASplitNum=[1, 1], CTAOrder=[0, 1]}}>
    #shared = #ttg.swizzled_shared<{{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}}>
    #smem = #ttg.shared_memory

    module attributes {{"ttg.num-ctas" = 1, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = {str(64)} : i32}} {{
    tt.func public @kernel(%arg0: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}) {{
        %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #blocked>
        %cst_n = arith.constant dense<{N_tile_size}> : tensor<{M_tile_size}x1xi32, #blocked>
        %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
        %1 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
        %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<{M}x{N}x!tt.ptr<f16>, #blocked>
        %4 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>> -> tensor<{M}x1xi32, #blocked>
        %5 = arith.muli %4, %cst : tensor<{M}x1xi32, #blocked>
        %6 = tt.expand_dims %1 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>> -> tensor<1x{N}xi32, #blocked>
        %7 = tt.broadcast %6 : tensor<1x{N}xi32, #blocked> -> tensor<{M}x{N}xi32, #blocked>
        %8 = tt.broadcast %5 : tensor<{M}x1xi32, #blocked> -> tensor<{M}x{N}xi32, #blocked>
        %9 = arith.addi %8, %7 : tensor<{M}x{N}xi32, #blocked>
        %ptrs = tt.addptr %2, %9 : tensor<{M}x{N}x!tt.ptr<f16>, #blocked>, tensor<{M}x{N}xi32, #blocked>
        %11 = tt.load %ptrs {{cache = 1 : i32, evict = 1 : i32, isVolatile = false}} : tensor<{M}x{N}x!tt.ptr<f16>, #blocked>

        %c0_i32 = arith.constant 0 : i32

        %12 = ttg.local_alloc : () -> !ttg.memdesc<1x{M}x{N}xf16, #shared, #smem, mutable>
        %13 = ttg.memdesc_subview %12[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x{M}x{N}xf16, #shared, #smem, mutable> -> !ttg.memdesc<{M}x{N}xf16, #shared, #smem, mutable>
        ttg.local_store %11, %13 : tensor<{M}x{N}xf16, #blocked> -> !ttg.memdesc<{M}x{N}xf16, #shared, #smem, mutable>

    """

    for m in range(num_repeats_M):
        for n in range(num_repeats_N):
            linear_idx = n + m * num_repeats_N
            m_offset = m * M_tile_size
            n_offset = n * N_tile_size
            ir += f"""
        %off0_{m}_{n} = arith.constant {m_offset} : i32
        %off1_{m}_{n} = arith.constant {n_offset} : i32

        %view{linear_idx} = ttg.memdesc_subview %13[%off0_{m}_{n}, %off1_{m}_{n}] : !ttg.memdesc<{M}x{N}xf16, #shared, #smem, mutable> -> !ttg.memdesc<{M_tile_size}x{N_tile_size}xf16, #shared, #smem, mutable, {M}x{N}>
        %data{linear_idx} = ttg.local_load %view{linear_idx} : !ttg.memdesc<{M_tile_size}x{N_tile_size}xf16, #shared, #smem, mutable, {M}x{N}> -> tensor<{M_tile_size}x{N_tile_size}xf16, #blocked>
        %inc{linear_idx} = arith.constant dense<{linear_idx}.0> : tensor<{M_tile_size}x{N_tile_size}xf16, #blocked>

        %res{linear_idx} = arith.addf %data{linear_idx}, %inc{linear_idx} : tensor<{M_tile_size}x{N_tile_size}xf16, #blocked>
        """

    total_num_repeats = num_repeats_M * num_repeats_N
    concat_op = "%concat = amdgpu.concat " + ', '.join([f'%res{i}' for i in range(total_num_repeats)])
    concat_op += ' : ' + ', '.join(
        [f'tensor<{M_tile_size}x{N_tile_size}xf16, #blocked>'
         for _ in range(total_num_repeats)]) + f' -> tensor<{M}x{N}xf16, #blocked>'
    ir += f"""
        {concat_op}
        tt.store %ptrs, %concat : tensor<{M}x{N}x!tt.ptr<f16>, #blocked>
        tt.return
    }}
    }}
    """

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)

    triton_result = torch.zeros((M, N), device=device, dtype=torch.float16)
    kernel[(1, 1, 1)](triton_result.data_ptr())

    rows = []
    for m in range(num_repeats_M):
        columns = []
        for n in range(num_repeats_N):
            linear_idx = n + m * num_repeats_N
            tile = float(linear_idx) * torch.ones((M_tile_size, N_tile_size), device=device, dtype=torch.float16)
            columns.append(tile)
        rows.append(torch.cat(columns, dim=1))
    expected_result = torch.cat(rows, dim=0)

    test_result = torch.equal(triton_result, expected_result)
    assert test_result
