import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_interpreter, numpy_random, to_triton, requires_tma, unwrap_tensor, tma_dtypes
from triton.tools.tensor_descriptor import TensorDescriptor


@requires_tma
@pytest.mark.interpreter()
@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("num_ctas", [1, 2])
@pytest.mark.parametrize("M_BLOCK,N_BLOCK", [(2, 16), (8, 16), (8, 32), (8, 128), (512, 32), (1, 1024)])
def test_host_tensor_descriptor_load(dtype_str, num_ctas, M_BLOCK, N_BLOCK):

    @triton.jit(debug=True)
    def kernel(out_ptr, desc, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        assert desc.shape[0] == M
        assert desc.shape[1] == N
        assert desc.strides[0] == N
        assert desc.strides[1] == 1
        assert desc.block_shape == [M_BLOCK, N_BLOCK]
        block = desc.load([M_BLOCK, 2 * N_BLOCK])
        idx = tl.arange(0, M_BLOCK)[:, None] * N_BLOCK + tl.arange(0, N_BLOCK)[None, :]
        tl.store(out_ptr + idx, block)

    M, N = M_BLOCK * 3, N_BLOCK * 4
    inp = to_triton(numpy_random((M, N), dtype_str), device="cuda", dst_type=dtype_str)
    out = inp.new_empty((M_BLOCK, N_BLOCK))

    inp_desc = TensorDescriptor(inp, shape=inp.shape, strides=inp.stride(), block_shape=[M_BLOCK, N_BLOCK])
    kernel[(1, )](out, inp_desc, M, N, M_BLOCK, N_BLOCK, num_ctas=num_ctas)

    expect = unwrap_tensor(inp)[1 * M_BLOCK:2 * M_BLOCK, 2 * N_BLOCK:3 * N_BLOCK]
    torch.testing.assert_close(expect, unwrap_tensor(out))


@triton.jit
def matmul_kernel_host_tensor_descriptor(a_desc, b_desc, c_desc):
    K = a_desc.shape[1]
    BLOCK_M: tl.constexpr = a_desc.block_shape[0]
    BLOCK_K: tl.constexpr = a_desc.block_shape[1]
    BLOCK_N: tl.constexpr = b_desc.block_shape[1]

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k = 0

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_k, offs_bn])
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_K
    accumulator = accumulator.to(a_desc.dtype)
    c_desc.store([offs_am, offs_bn], accumulator)


@requires_tma
@pytest.mark.interpreter()
@pytest.mark.parametrize("num_ctas", [1, 2])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K, num_stages", [
    (128, 128, 16, 1),
    (512, 64, 32, 2),
    (64, 512, 32, 2),
    (128, 128, 16, 4),
    (64, 128, 32, 4),
    (32, 32, 32, 4),
    (256, 128, 32, 4),
])
def test_host_tensor_descriptor_matmul(num_stages, num_ctas, BLOCK_M, BLOCK_N, BLOCK_K):
    device = "cuda"
    if is_interpreter():
        M, N, K = BLOCK_M, BLOCK_N, BLOCK_K
    else:
        M, N, K = 1024, 512, 256
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((K, N), dtype=torch.float16, device=device)
    C = torch.empty((M, N), dtype=torch.float16, device=device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), 1)

    A_desc = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_M, BLOCK_K])
    B_desc = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_K, BLOCK_N])
    C_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_M, BLOCK_N])

    kernel = matmul_kernel_host_tensor_descriptor[grid](
        A_desc,
        B_desc,
        C_desc,  #
        num_warps=8,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )
    ref_out = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.float16)
    torch.testing.assert_close(ref_out, C, rtol=1e-3, atol=1e-3)
    if is_interpreter():
        return

    if BLOCK_M >= 64 * num_ctas and BLOCK_N >= 64 and torch.cuda.get_device_capability()[0] == 9:
        # TODO: The use of stmatrix for Blackwell is currently not supported.
        # Only a subset of TMEM and stmatrix layout pairs are compatible, for example 16x256bx2 and m8n8x4.
        assert "stmatrix.sync.aligned.m8n8.x4.shared.b16" in kernel.asm["ptx"]


@requires_tma
def test_specialization_after_host_tensordesc():

    @triton.jit
    def kernel(a, b):
        pass

    device = "cuda"
    A = torch.randn(1024, device=device)
    desc = TensorDescriptor.from_tensor(A, [128])
    h = kernel.warmup(desc, 16, grid=(1, ))
    assert ", %arg3: i32 {tt.divisibility = 16 : i32}" in h.asm["ttir"]
