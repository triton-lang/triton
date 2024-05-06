import numpy as np
import pytest
import torch
import tempfile

import triton
import triton.language as tl


def test_descriptor_load_ttgir():
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] == 9:
        pytest.skip("Test requires Hopper target.")
        return
    device = "cuda"
    SIZE = 128

    ir = f"""
    #blocked = #triton_gpu.blocked<{{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}}>
    #shared = #triton_gpu.shared<{{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}}>
    module attributes {{"triton_gpu.target" = "cuda:90", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32}} {{
      tt.func public @kernel(%arg0: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i8> {{tt.divisibility = 16 : i32}}) attributes {{noinline = false}} {{
        %c0_i32 = arith.constant 0 : i32
        %0 = tt.make_range {{end = {SIZE} : i32, start = 0 : i32}} : tensor<{SIZE}xi32, #blocked>
        %1 = triton_gpu.local_alloc  : () -> !tt.memdesc<{SIZE}xf32, #shared, mutable>
        %2 = triton_gpu.local_alloc  : () -> !tt.memdesc<1xi64, #shared, mutable>
        triton_nvidia_gpu.init_barrier %2, 1 : <1xi64, #shared, mutable>
        %true = arith.constant 1 : i1
        triton_nvidia_gpu.async_tma_copy_global_to_local %arg1[%c0_i32] %1, %2, %true : <i8>, <1xi64, #shared, mutable> -> <{SIZE}xf32, #shared, mutable>
        triton_nvidia_gpu.wait_barrier %2, %c0_i32 : <1xi64, #shared, mutable>
        %3 = triton_gpu.local_load %1 : !tt.memdesc<{SIZE}xf32, #shared, mutable> -> tensor<{SIZE}xf32, #blocked>
        %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<{SIZE}x!tt.ptr<f32>, #blocked>
        %5 = tt.addptr %4, %0 : tensor<{SIZE}x!tt.ptr<f32>, #blocked>, tensor<{SIZE}xi32, #blocked>
        tt.store %5, %3 : tensor<{SIZE}x!tt.ptr<f32>, #blocked>
        tt.return
      }}
    }}
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)

    x = torch.randn(SIZE, dtype=torch.float32, device=device)
    desc = np.empty(SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_1d_tma_descriptor(x.data_ptr(), SIZE, SIZE, x.element_size(), desc)
    desc = torch.tensor(desc, device=device)
    z_tri = torch.empty_like(x)
    kernel[(1, 1, 1)](z_tri, desc)
    assert torch.equal(x, z_tri)


def test_experimetal_descriptor_load():
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] == 9:
        pytest.skip("Test requires Hopper target.")
        return
    device = "cuda"
    SIZE = 128

    @triton.jit
    def kernel(Z, desc, SIZE: tl.constexpr):
        off_desc = 0
        off = tl.arange(0, SIZE)
        x = tl._experimental_descriptor_load(desc, [off_desc], [SIZE], Z.dtype.element_ty)
        tl.store(Z + off, x)

    x = torch.randn(SIZE, dtype=torch.float32, device=device)
    desc = np.empty(SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_1d_tma_descriptor(x.data_ptr(), SIZE, SIZE, x.element_size(), desc)
    desc = torch.tensor(desc, device=device)
    z_tri = torch.empty_like(x)
    kernel[(1, )](z_tri, desc, SIZE=SIZE, num_warps=4)
    assert torch.equal(x, z_tri)


@triton.jit
def matmul_kernel_tma(a_desc_ptr, b_desc_ptr, c_ptr,  #
                      M, N, K,  #
                      stride_cm, stride_cn,  #
                      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], tl.float16)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], tl.float16)
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator)


@pytest.mark.parametrize("num_stages", [1, 4])
def test_experimental_tma_matmul(num_stages):
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] == 9:
        pytest.skip("Test requires Hopper target.")
        return
    device = "cuda"
    M, N, K = 8192, 8192, 1024
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((K, N), dtype=torch.float16, device=device)
    C = torch.empty((M, N), dtype=torch.float32, device=device)
    TMA_SIZE = 128
    desc_a = np.empty(TMA_SIZE, dtype=np.int8)
    desc_b = np.empty(TMA_SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(A.data_ptr(), M, K, BLOCK_M, BLOCK_K, A.element_size(),
                                                              desc_a)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(B.data_ptr(), K, N, BLOCK_K, BLOCK_N, B.element_size(),
                                                              desc_b)

    desc_a = torch.tensor(desc_a, device=device)
    desc_b = torch.tensor(desc_b, device=device)
    matmul_kernel_tma[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)](desc_a,
                                                                                 desc_b, C, M, N, K, C.stride(0),
                                                                                 C.stride(1), BLOCK_M, BLOCK_N, BLOCK_K,
                                                                                 num_warps=4, num_stages=num_stages)
    ref_out = torch.matmul(A.to(torch.float32), B.to(torch.float32))
    torch.testing.assert_close(ref_out, C, rtol=1e-3, atol=1e-3)
