import torch
import pytest
import pathlib
import triton
import triton.language as tl

from triton._internal_testing import is_cuda

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10:
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


@pytest.mark.skipif(not is_cuda(), reason="warp specialization is only supported on NVIDIA")
def test_warp_specialize_basic_ir(tmp_path: pathlib.Path):
    ir = """
    tt.func @kernel(%arg0: !tt.ptr<i32>) {
      %c42_i32 = arith.constant 42 : i32
      gpu.barrier
      ttg.warp_specialize(%arg0)
      default {
        tt.store %arg0, %c42_i32 : !tt.ptr<i32>
        gpu.barrier
        ttg.warp_yield
      }
      partition0(%arg1: !tt.ptr<i32>) num_warps(1) {
        %c5555_i32 = arith.constant 5555 : i32
        %c1_i32 = arith.constant 1 : i32
        gpu.barrier
        %ptr = tt.addptr %arg1, %c1_i32 : !tt.ptr<i32>, i32
        tt.store %ptr, %c5555_i32 : !tt.ptr<i32>
        ttg.warp_return
      } : (!tt.ptr<i32>) -> ()
      tt.return
    }
    """

    temp_file = tmp_path / "test_warp_specialize_basic_ir.ttir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    input = torch.empty(2, dtype=torch.int32, device='cuda')
    kernel[(1, 1, 1)](input)
    assert input[0] == 42
    assert input[1] == 5555


@pytest.mark.skipif(not is_cuda(), reason="warp specialization is only supported on NVIDIA")
def test_warpgroup_reduction(tmp_path: pathlib.Path):

    def template(i, num_warps, in_ptr, out_ptr):
        return f"""
          %range = tt.make_range {{end = {(i+1)*256} : i32, start = {i*256} : i32}} : tensor<256xi32, #blocked{num_warps}>
          %splatted = tt.splat {in_ptr} : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #blocked{num_warps}>
          %ptrs = tt.addptr %splatted, %range : tensor<256x!tt.ptr<i32>, #blocked{num_warps}>, tensor<256xi32, #blocked{num_warps}>
          %input = tt.load %ptrs : tensor<256x!tt.ptr<i32>, #blocked{num_warps}>
          %result = "tt.reduce"(%input) ({{
          ^bb0(%lhs: i32, %rhs: i32):
            %result = arith.addi %lhs, %rhs : i32
            tt.reduce.return %result : i32
          }}) {{axis = 0 : i32}} : (tensor<256xi32, #blocked{num_warps}>) -> i32
          %offset = arith.constant {i} : i32
          %output = tt.addptr {out_ptr}, %offset : !tt.ptr<i32>, i32
          tt.store %output, %result : !tt.ptr<i32>
        """

    ir = """
    #blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    #blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
    #blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

    module attributes {"ttg.num-warps" = 4 : i32} {

    tt.func @kernel(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
      ttg.warp_specialize(%arg0, %arg1)
      default {
      """ + template(0, 4, "%arg0", "%arg1") + """
        ttg.warp_yield
      }
      partition0(%arg2: !tt.ptr<i32>, %arg3: !tt.ptr<i32>) num_warps(4) {
      """ + template(1, 4, "%arg2", "%arg3") + """
        ttg.warp_return
      }
      partition1(%arg4: !tt.ptr<i32>, %arg5: !tt.ptr<i32>) num_warps(2) {
      """ + template(2, 2, "%arg4", "%arg5") + """
        ttg.warp_return
      }
      partition2(%arg6: !tt.ptr<i32>, %arg7: !tt.ptr<i32>) num_warps(1) {
      """ + template(3, 1, "%arg6", "%arg7") + """
        ttg.warp_return
      } : (!tt.ptr<i32>, !tt.ptr<i32>) -> ()
      tt.return
    }

    }
    """

    temp_file = tmp_path / "test_warpgroup_reduction.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    input = torch.arange(1024, dtype=torch.int32, device='cuda')
    output = torch.empty(4, dtype=torch.int32, device='cuda')
    kernel[(1, 1, 1)](input, output)
    assert output[0] == torch.arange(0, 256).sum()
    assert output[1] == torch.arange(256, 512).sum()
    assert output[2] == torch.arange(512, 768).sum()
    assert output[3] == torch.arange(768, 1024).sum()


@triton.jit
def _compute_pid(tile_id, num_pid_n, num_pid_m, GROUP_SIZE_M):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def matmul_tma_ws_kernel(  #
        a_ptr, b_ptr, c_ptr,  #
        a_stride0, a_stride1,  #
        b_stride0, b_stride1,  #
        c_stride0, c_stride1,  #
        M, N, K,  #
        num_stages: tl.constexpr,  #
        BLOCK_SIZE_M: tl.constexpr,  #
        BLOCK_SIZE_N: tl.constexpr,  #
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        USE_FP8: tl.constexpr,  #
):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[a_stride0, a_stride1],
                                       block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[b_stride0, b_stride1],
                                       block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[c_stride0, c_stride1],
                                       block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = _compute_pid(pid, num_pid_n, num_pid_m, GROUP_SIZE_M)

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    off_am = pid_m * BLOCK_SIZE_M
    off_bn = pid_n * BLOCK_SIZE_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(k_tiles, warp_specialize=True, num_stages=num_stages):
        off_k = k * BLOCK_SIZE_K
        a = a_desc.load((off_am, off_k))
        b = b_desc.load((off_bn, off_k))
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(tl.float8e4nv if USE_FP8 else tl.float16)
    c_desc.store((off_am, off_bn), c)


def exceeds_smem_capacity(num_stages, BLOCK_M, BLOCK_N, BLOCK_K, use_fp8):
    return (num_stages * BLOCK_K * (BLOCK_M + BLOCK_N) + BLOCK_M * BLOCK_N) * (1 if use_fp8 else 2) > 228 * 1024


@pytest.mark.parametrize("M, N, K", [(32, 32, 32), (8192, 8192, 512)])
@pytest.mark.parametrize("BLOCK_SIZE_M", [128])
@pytest.mark.parametrize("BLOCK_SIZE_N", [128, 256])
@pytest.mark.parametrize("BLOCK_SIZE_K", [64, 128])
@pytest.mark.parametrize("num_stages", [2, 3, 4])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 10, reason="Requires compute capability >= 10")
def test_warp_specialize_tma_matmul(M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps, use_fp8):
    if exceeds_smem_capacity(num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, use_fp8=use_fp8):
        pytest.skip("uses too much shared memory")
    dtype = torch.float8_e4m3fn if use_fp8 else torch.float16

    GROUP_SIZE_M = 8

    device = "cuda"
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device).to(dtype)
    B = torch.randn((N, K), dtype=torch.float16, device=device).to(dtype)
    C = torch.randn((M, N), dtype=torch.float16, device=device).to(dtype)

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    kernel = matmul_tma_ws_kernel[grid](A, B, C, *A.stride(), *B.stride(), *C.stride(), M, N, K, num_stages,
                                        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps=num_warps,
                                        USE_FP8=use_fp8)
    ttgir = kernel.asm["ttgir"]
    assert "ttng.tc_gen5_mma" in ttgir
    assert "ttg.warp_specialize" in ttgir

    ref_out = torch.empty((M, N), dtype=dtype, device=device)
    cublas.matmul(A, B, ref_out)
    torch.testing.assert_close(ref_out.to(torch.float16), C.to(torch.float16), atol=0.03, rtol=0.03)


@triton.jit
def matmul_tma_persistent_ws_kernel(  #
        a_ptr, b_ptr, c_ptr,  #
        a_stride0, a_stride1,  #
        b_stride0, b_stride1,  #
        c_stride0, c_stride1,  #
        M, N, K,  #
        num_stages: tl.constexpr,  #
        BLOCK_SIZE_M: tl.constexpr,  #
        BLOCK_SIZE_N: tl.constexpr,  #
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        NUM_SMS: tl.constexpr,  #
        USE_FP8: tl.constexpr,  #
):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[a_stride0, a_stride1],
                                       block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[b_stride0, b_stride1],
                                       block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[c_stride0, c_stride1],
                                       block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=True, num_stages=num_stages):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_n, num_pid_m, GROUP_SIZE_M)

        off_am = pid_m * BLOCK_SIZE_M
        off_bn = pid_n * BLOCK_SIZE_N
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            off_k = ki * BLOCK_SIZE_K
            a = a_desc.load((off_am, off_k))
            b = b_desc.load((off_bn, off_k))
            accumulator = tl.dot(a, b.T, accumulator)

        c = accumulator.to(tl.float8e4nv if USE_FP8 else tl.float16)
        c_desc.store((off_am, off_bn), c)


@pytest.mark.parametrize("M, N, K", [(32, 32, 32), (8192, 8192, 512)])
@pytest.mark.parametrize("BLOCK_SIZE_M", [128])
@pytest.mark.parametrize("BLOCK_SIZE_N", [128, 256])
@pytest.mark.parametrize("BLOCK_SIZE_K", [64, 128])
@pytest.mark.parametrize("num_stages", [2, 3, 4])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 10, reason="Requires compute capability >= 10")
def test_warp_specialize_tma_matmul_persistent(M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps,
                                               use_fp8):
    if exceeds_smem_capacity(num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, use_fp8):
        pytest.skip("uses too much shared memory")
    dtype = torch.float8_e4m3fn if use_fp8 else torch.float16

    GROUP_SIZE_M = 8
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    device = "cuda"
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device).to(dtype)
    B = torch.randn((N, K), dtype=torch.float16, device=device).to(dtype)
    C = torch.randn((M, N), dtype=torch.float16, device=device).to(dtype)

    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    def grid(META):
        return (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )

    kernel = matmul_tma_persistent_ws_kernel[grid](A, B, C, *A.stride(), *B.stride(), *C.stride(), M, N, K, num_stages,
                                                   BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, NUM_SMS,
                                                   num_warps=num_warps, USE_FP8=use_fp8)
    ttgir = kernel.asm["ttgir"]
    assert "ttng.tc_gen5_mma" in ttgir
    assert "ttg.warp_specialize" in ttgir

    ref_out = torch.empty((M, N), dtype=dtype, device=device)
    cublas.matmul(A, B, ref_out)
    torch.testing.assert_close(ref_out.to(torch.float16), C.to(torch.float16), atol=0.03, rtol=0.03)
