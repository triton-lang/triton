import pytest
from typing import Optional
import torch
import triton.testing_autows as utils
import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()


def num_sms():
    return torch.cuda.get_device_properties("cuda").multi_processor_count


@triton.jit
def grouped_matmul_tma_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    gm, gn, gk,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # is the output FP8 or FP16
    FP8: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8 else tl.float16
    num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
    num_tiles = num_m_tiles * num_n_tiles
    start_pid = tl.program_id(axis=0)

    for g in range(group_size):
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
    g_sizes = []
    g_lds = []
    group_C = []
    M, K = group_A[0].shape
    N, _ = group_B[0].shape

    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        C = torch.empty((M, N), device=DEVICE, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (META['NUM_SM'], )
    out = grouped_matmul_tma_kernel[grid](d_a_ptrs, d_b_ptrs, d_c_ptrs, M, N, K, d_g_lds, group_size,
                                          BLOCK_SIZE_M=128,
                                          BLOCK_SIZE_N=128,
                                          BLOCK_SIZE_K=128,
                                          FP8=torch.float8_e4m3fn == group_A[0].dtype, NUM_SM=num_sms(),
                                          num_stages=3,
                                          enable_warp_specialization=True)
    # print(out.asm["ttgir"], flush=True)
    return group_C


@pytest.mark.parametrize("M", [128, 256, 512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("N", [256, 512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("K", [128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("group_size", [1, 4, 8, 16])
def test_grouped_gemm(M, N, K, group_size):
    if not utils.is_sm10x():
        pytest.skip()  # Some test cases fail accuracy check on Hopper

    group_A = []
    group_B = []
    group_B_T = []

    for i in range(group_size):
        A = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
        B = torch.rand((K, N), device=DEVICE, dtype=torch.float16)
        B_T = B.T.contiguous()
        group_A.append(A)
        group_B.append(B)
        group_B_T.append(B_T)

    ref_out = [torch.matmul(a, b) for a, b in zip(group_A, group_B)]

    tri_tma_out = group_gemm_tma_fn(group_A, group_B_T)
    for i in range(group_size):
        assert torch.allclose(ref_out[i], tri_tma_out[i], atol=1e-2, rtol=0)
