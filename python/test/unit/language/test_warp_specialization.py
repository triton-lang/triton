import torch
import pytest
import pathlib
import triton
import triton.language as tl

from triton._internal_testing import is_hip, is_hopper, is_blackwell
from triton.tools.tensor_descriptor import TensorDescriptor

if not is_hip() and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in [9, 10]:
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


def is_hopper_or_blackwell():
    return is_hopper() or is_blackwell()


@pytest.mark.skipif(is_hip(), reason="warp specialization is not supported on hip devices")
@pytest.mark.skipif(not is_hopper_or_blackwell(), reason="Requires Hopper or Blackwell")
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


@pytest.mark.skipif(is_hip(), reason="warp specialization is not supported on hip devices")
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_warp_specialize_tmem_ir(tmp_path: pathlib.Path):
    ir = """
    #blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
    #shared = #ttg.swizzled_shared<{vec=1, perPhase=1, maxPhase=1, order=[1, 0]}>
    #tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>

    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

    tt.func @test_tmem_ws(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
      %cst = arith.constant dense<64> : tensor<128x64xi32, #blocked>
      %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
      %4 = tt.broadcast %1 {axis = 1 : i32} : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
      %5 = tt.broadcast %3 {axis = 0 : i32} : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
      %6 = arith.muli %4, %cst : tensor<128x64xi32, #blocked>
      %7 = arith.addi %6, %5 : tensor<128x64xi32, #blocked>
      %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x64x!tt.ptr<f32>, #blocked>
      %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x64x!tt.ptr<f32>, #blocked>

      %ptrs_in = tt.addptr %8, %7 : tensor<128x64x!tt.ptr<f32>, #blocked>, tensor<128x64xi32, #blocked>
      %ptrs_out = tt.addptr %9, %7 : tensor<128x64x!tt.ptr<f32>, #blocked>, tensor<128x64xi32, #blocked>

      %v_init = tt.load %ptrs_in : tensor<128x64x!tt.ptr<f32>, #blocked>

      %v_shared = ttg.local_alloc %v_init : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #shared, #ttg.shared_memory>
      %v = ttg.local_load %v_shared : !ttg.memdesc<128x64xf32, #shared, #ttg.shared_memory> -> tensor<128x64xf32, #blocked>

      %tmem_in = ttng.tmem_alloc %v : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory>
      %tmem_out = ttng.tmem_alloc : () -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>

      ttg.warp_specialize(%tmem_in, %tmem_out)
      default {
        ttg.warp_yield
      }
      partition0(%in: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory>, %out: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(1) {
        ttg.warp_return
      }
      partition1(%in: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory>, %out: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(2) {
        ttg.warp_return
      }
      partition2(%in: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory>, %out: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(4) {
        %x = ttng.tmem_load %in : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory> -> tensor<128x64xf32, #blocked>
        %true = arith.constant true
        ttng.tmem_store %x, %out, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
        ttg.warp_return
      } : (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>) -> ()

      %result = ttng.tmem_load %tmem_out : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
      tt.store %ptrs_out, %result : tensor<128x64x!tt.ptr<f32>, #blocked>
      tt.return
    }

    }
    """

    temp_file = tmp_path / "test_warp_specialize_tmem_ir.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))
    input = torch.arange(128 * 64, dtype=torch.float32, device='cuda').reshape(128, 64)
    output = torch.empty_like(input)
    kernel[(1, 1, 1)](input, output)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


@pytest.mark.skipif(is_hip(), reason="warp specialization is not supported on hip devices")
@pytest.mark.skipif(not is_hopper_or_blackwell(), reason="Requires Hopper or Blackwell")
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
@pytest.mark.parametrize("num_stages", [2, 3])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.skipif(is_hip(), reason="warp specialization is not supported on hip devices")
@pytest.mark.skipif(not is_hopper_or_blackwell(), reason="Requires Hopper or Blackwell")
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
    if is_blackwell():
        assert "ttng.tc_gen5_mma" in ttgir
        assert "ttng.async_tma_copy_global_to_local" in ttgir
    else:
        assert "ttng.warp_group_dot" in ttgir
        assert "ttng.async_tma_copy_global_to_local" in ttgir
    if is_hopper() and num_warps == 8:
        assert "ttg.warp_specialize" not in ttgir
    else:
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
        FLATTEN: tl.constexpr,  #
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

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=FLATTEN, warp_specialize=True,
                            num_stages=num_stages):
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
@pytest.mark.parametrize("num_stages", [2, 3])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.parametrize("flatten", [False, True] if is_blackwell() else [True])
@pytest.mark.skipif(is_hip(), reason="warp specialization is not supported on hip devices")
@pytest.mark.skipif(not is_hopper_or_blackwell(), reason="Requires Hopper or Blackwell")
def test_warp_specialize_tma_matmul_persistent(M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps,
                                               use_fp8, flatten):
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
                                                   num_warps=num_warps, USE_FP8=use_fp8, FLATTEN=flatten
                                                   and is_blackwell())
    ttgir = kernel.asm["ttgir"]
    if is_blackwell():
        assert "ttng.tc_gen5_mma" in ttgir
        assert "ttng.async_tma_copy_global_to_local" in ttgir
    else:
        assert "ttng.warp_group_dot" in ttgir
        assert "ttng.async_tma_copy_global_to_local" in ttgir
    if is_hopper() and num_warps == 8:
        assert "ttg.warp_specialize" not in ttgir
    else:
        assert "ttg.warp_specialize" in ttgir

    ref_out = torch.empty((M, N), dtype=dtype, device=device)
    cublas.matmul(A, B, ref_out)
    torch.testing.assert_close(ref_out.to(torch.float16), C.to(torch.float16), atol=0.03, rtol=0.03)


@triton.jit
def attention_inner_loop_kernel(  #
        desc_q, desc_k, desc_v,  #
        desc_acc, l_i_ptr, m_i_ptr,  #
        M, N, qk_scale,  #
        BLOCK_M: tl.constexpr,  #
        HEAD_DIM: tl.constexpr,  #
        warp_specialize: tl.constexpr  #
):
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    off_m = tl.program_id(0) * BLOCK_M
    q = desc_q.load([off_m, 0])

    for start_n in tl.range(0, N, HEAD_DIM, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, HEAD_DIM)
        k = desc_k.load([start_n, 0]).T

        qk = tl.dot(q, k)

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = desc_v.load([start_n, 0])
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    desc_acc.store([off_m, 0], acc.to(q.dtype))
    tl.store(l_i_ptr + off_m + tl.arange(0, BLOCK_M), l_i)
    tl.store(m_i_ptr + off_m + tl.arange(0, BLOCK_M), m_i)


@pytest.mark.parametrize("M, N", [(8192, 8192), (1024, 1024)])
@pytest.mark.parametrize("BLOCK_M", [64, 128])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("num_stages", [2, 3])
@pytest.mark.parametrize("disable_acc_multibuf", [False, True])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.skipif(is_hip(), reason="warp specialization is not supported on hip devices")
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_warp_specialize_attention_forward(M, N, BLOCK_M, HEAD_DIM, num_stages, disable_acc_multibuf, num_warps,
                                           use_fp8):
    if BLOCK_M == 128 and HEAD_DIM == 128 and not use_fp8:
        # These configurations currently use too much shared memory.
        if (num_warps, num_stages) in [(4, 4), (8, 4), (8, 3)]:
            pytest.skip("uses too much shared memory")

    dtype = torch.float8_e4m3fn if use_fp8 else torch.float16

    torch.manual_seed(42)
    q = torch.randn((M, HEAD_DIM), device="cuda").to(dtype)
    k = torch.randn((N, HEAD_DIM), device="cuda").to(dtype)
    v = torch.randn((N, HEAD_DIM), device="cuda").to(dtype)

    acc_ref = torch.empty((M, HEAD_DIM), dtype=dtype, device="cuda")
    l_i_ref = torch.empty((M, ), dtype=dtype, device="cuda")
    m_i_ref = torch.empty((M, ), dtype=dtype, device="cuda")
    acc = torch.empty((M, HEAD_DIM), dtype=dtype, device="cuda")
    l_i = torch.empty((M, ), dtype=dtype, device="cuda")
    m_i = torch.empty((M, ), dtype=dtype, device="cuda")

    desc_q = TensorDescriptor(q, shape=[M, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])
    desc_k = TensorDescriptor(k, shape=[N, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = TensorDescriptor(v, shape=[N, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])
    desc_acc_ref = TensorDescriptor(acc_ref, shape=[M, HEAD_DIM], strides=[HEAD_DIM, 1],
                                    block_shape=[BLOCK_M, HEAD_DIM])
    desc_acc = TensorDescriptor(acc, shape=[M, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])

    attention_inner_loop_kernel[(M // BLOCK_M, )](desc_q, desc_k, desc_v, desc_acc_ref, l_i_ref, m_i_ref, M, N, 0.5,
                                                  BLOCK_M, HEAD_DIM, False, num_stages=num_stages, num_warps=num_warps)
    attention_inner_loop_kernel[(M // BLOCK_M, )](desc_q, desc_k, desc_v, desc_acc, l_i, m_i, M, N, 0.5, BLOCK_M,
                                                  HEAD_DIM, True, num_stages=num_stages, num_warps=num_warps)

    torch.testing.assert_close(acc.to(torch.float32), acc_ref.to(torch.float32), atol=0, rtol=0)
    torch.testing.assert_close(l_i.to(torch.float32), l_i_ref.to(torch.float32), atol=0, rtol=0)
    torch.testing.assert_close(m_i.to(torch.float32), m_i_ref.to(torch.float32), atol=0, rtol=0)


@triton.jit
def attention_persistent_inner_loop_kernel(  #
        desc_q, desc_k, desc_v,  #
        desc_acc, l_i_ptr, m_i_ptr,  #
        M, N, qk_scale,  #
        BLOCK_M: tl.constexpr,  #
        HEAD_DIM: tl.constexpr,  #
        warp_specialize: tl.constexpr,  #
        num_stages: tl.constexpr):
    prog_id = tl.program_id(0)
    num_sm = tl.num_programs(0)
    num_tiles = tl.cdiv(M, BLOCK_M)

    tiles_per_sm = num_tiles // num_sm
    if prog_id < num_tiles % num_sm:
        tiles_per_sm += 1

    tile_idx = prog_id
    for _ in tl.range(0, tiles_per_sm, warp_specialize=warp_specialize, num_stages=num_stages):
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        off_m = tile_idx * BLOCK_M
        q = desc_q.load([off_m, 0])

        for start_n in tl.range(0, N, HEAD_DIM):
            start_n = tl.multiple_of(start_n, HEAD_DIM)
            k = desc_k.load([start_n, 0]).T

            qk = tl.dot(q, k)

            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            alpha = tl.math.exp2(m_i - m_ij)
            l_ij = tl.sum(p, 1)
            acc = acc * alpha[:, None]

            v = desc_v.load([start_n, 0])
            p = p.to(v.dtype)
            acc = tl.dot(p, v, acc)

            l_i = l_i * alpha + l_ij
            m_i = m_ij

        desc_acc.store([off_m, 0], acc.to(q.dtype))
        tl.store(l_i_ptr + off_m + tl.arange(0, BLOCK_M), l_i)
        tl.store(m_i_ptr + off_m + tl.arange(0, BLOCK_M), m_i)

        tile_idx += num_sm


@pytest.mark.parametrize("M, N", [(8192, 8192), (1024, 1024)])
@pytest.mark.parametrize("BLOCK_M", [64, 128])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("num_stages", [2, 3])
@pytest.mark.parametrize("disable_acc_multibuf", [False, True])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_warp_specialize_attention_persistent_forward(M, N, BLOCK_M, HEAD_DIM, num_stages, disable_acc_multibuf,
                                                      num_warps, use_fp8):
    if BLOCK_M == 128 and HEAD_DIM == 128 and not use_fp8:
        # These configurations currently use too much shared memory.
        if (num_warps, num_stages) in [(4, 4), (8, 4), (8, 3), (4, 3)]:
            pytest.skip("uses too much shared memory")

    dtype = torch.float8_e4m3fn if use_fp8 else torch.float16

    torch.manual_seed(42)

    q = torch.randn((M, HEAD_DIM), device="cuda").to(dtype)
    k = torch.randn((N, HEAD_DIM), device="cuda").to(dtype)
    v = torch.randn((N, HEAD_DIM), device="cuda").to(dtype)

    acc_ref = torch.empty((M, HEAD_DIM), dtype=dtype, device="cuda")
    l_i_ref = torch.empty((M, ), dtype=dtype, device="cuda")
    m_i_ref = torch.empty((M, ), dtype=dtype, device="cuda")
    acc = torch.empty((M, HEAD_DIM), dtype=dtype, device="cuda")
    l_i = torch.empty((M, ), dtype=dtype, device="cuda")
    m_i = torch.empty((M, ), dtype=dtype, device="cuda")

    desc_q = TensorDescriptor(q, shape=[M, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])
    desc_k = TensorDescriptor(k, shape=[N, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = TensorDescriptor(v, shape=[N, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])
    desc_acc_ref = TensorDescriptor(acc_ref, shape=[M, HEAD_DIM], strides=[HEAD_DIM, 1],
                                    block_shape=[BLOCK_M, HEAD_DIM])
    desc_acc = TensorDescriptor(acc, shape=[M, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])

    NUM_SM = 4
    attention_persistent_inner_loop_kernel[(NUM_SM, )](desc_q, desc_k, desc_v, desc_acc, l_i, m_i, M, N, 0.5, BLOCK_M,
                                                       HEAD_DIM, True, num_stages=num_stages, num_warps=num_warps)
    attention_inner_loop_kernel[(M // BLOCK_M, )](desc_q, desc_k, desc_v, desc_acc_ref, l_i_ref, m_i_ref, M, N, 0.5,
                                                  BLOCK_M, HEAD_DIM, False, num_stages=num_stages, num_warps=num_warps)

    torch.testing.assert_close(acc.to(torch.float32), acc_ref.to(torch.float32), atol=0, rtol=0)
    torch.testing.assert_close(l_i.to(torch.float32), l_i_ref.to(torch.float32), atol=0, rtol=0)
    torch.testing.assert_close(m_i.to(torch.float32), m_i_ref.to(torch.float32), atol=0, rtol=0)


@triton.jit
def grouped_matmul_tma_kernel(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    gm,
    gn,
    gk,
    g_lds,
    group_size,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    dtype = tl.float16
    num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
    num_tiles = num_m_tiles * num_n_tiles
    start_pid = tl.program_id(axis=0)

    for g in tl.range(group_size, warp_specialize=True):
        lda = tl.load(g_lds + g * 3)
        ldb = tl.load(g_lds + g * 3 + 1)
        ldc = tl.load(g_lds + g * 3 + 2)

        a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(dtype))
        b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(dtype))
        c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(dtype))

        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[gm, gk],
            strides=[lda, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )

        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[gn, gk],
            strides=[ldb, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )
        c_desc = tl.make_tensor_descriptor(
            c_ptr,
            shape=[gm, gn],
            strides=[ldc, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        for tile_idx in tl.range(start_pid, num_tiles, NUM_SM):
            tile_m_idx = tile_idx // num_n_tiles
            tile_n_idx = tile_idx % num_n_tiles
            offs_am = tile_m_idx * BLOCK_SIZE_M
            offs_bn = tile_n_idx * BLOCK_SIZE_N

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(gk, BLOCK_SIZE_K)):
                a = a_desc.load([offs_am, kk * BLOCK_SIZE_K])
                b = b_desc.load([offs_bn, kk * BLOCK_SIZE_K])
                accumulator += tl.dot(a, b.T)

            offs_cm = tile_m_idx * BLOCK_SIZE_M
            offs_cn = tile_n_idx * BLOCK_SIZE_N

            c = accumulator.to(dtype)
            c_desc.store([offs_cm, offs_cn], c)


def group_gemm_tma_fn(group_A, group_B):
    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_lds = []
    group_C = []
    M, K = group_A[0].shape
    N, _ = group_B[0].shape

    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        C = torch.empty((M, N), device="cuda", dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    d_a_ptrs = torch.tensor(A_addrs, device="cuda")
    d_b_ptrs = torch.tensor(B_addrs, device="cuda")
    d_c_ptrs = torch.tensor(C_addrs, device="cuda")
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device="cuda")

    def alloc_fn(size: int, _, __):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (META['NUM_SM'], )
    out = grouped_matmul_tma_kernel[grid](d_a_ptrs, d_b_ptrs, d_c_ptrs, M, N, K, d_g_lds, group_size, BLOCK_SIZE_M=128,
                                          BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, NUM_SM=4, num_stages=3)
    assert "ttg.warp_specialize" in out.asm["ttgir"]
    return group_C


@pytest.mark.parametrize("M", [128, 256, 512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("N", [256, 512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("K", [128, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("group_size", [4, 8, 16])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_grouped_gemm(M, N, K, group_size):
    torch.manual_seed(42)
    group_A = []
    group_B = []
    group_B_T = []

    for i in range(group_size):
        A = torch.rand((M, K), device="cuda", dtype=torch.float16)
        B = torch.rand((K, N), device="cuda", dtype=torch.float16)
        B_T = B.T.contiguous()
        group_A.append(A)
        group_B.append(B)
        group_B_T.append(B_T)

    ref_out = [torch.matmul(a, b) for a, b in zip(group_A, group_B)]

    tri_tma_out = group_gemm_tma_fn(group_A, group_B_T)
    for i in range(group_size):
        assert torch.allclose(ref_out[i], tri_tma_out[i], atol=1e-2, rtol=1e-2)
