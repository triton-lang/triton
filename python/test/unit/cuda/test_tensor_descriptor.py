import pytest
import torch
import numpy as np

import triton
from triton.compiler.errors import CompilationError
import triton.language as tl
from triton._internal_testing import is_interpreter, numpy_random, to_triton, requires_tma, unwrap_tensor, tma_dtypes, to_numpy
from triton.tools.tensor_descriptor import TensorDescriptor
from typing import Optional

SUPPORTED_REDUCE_DTYPES = {
    "add": {tl.uint32, tl.int32, tl.uint64, tl.float32, tl.float16, tl.bfloat16},
    "min": {tl.uint32, tl.int32, tl.uint64, tl.int64, tl.float16, tl.bfloat16},
    "max": {tl.uint32, tl.int32, tl.uint64, tl.int64, tl.float16, tl.bfloat16},
    "and": {tl.uint32, tl.int32, tl.uint64, tl.int64},
    "or": {tl.uint32, tl.int32, tl.uint64, tl.int64},
    "xor": {tl.uint32, tl.int32, tl.uint64, tl.int64},
}


def min_op(a, b):
    out = np.minimum(to_numpy(a), to_numpy(b))
    return unwrap_tensor(to_triton(out, device=a.device))


def max_op(a, b):
    out = np.maximum(to_numpy(a), to_numpy(b))
    return unwrap_tensor(to_triton(out, device=a.device))


REDUCE_OP = {
    "add": lambda a, b: unwrap_tensor(a) + unwrap_tensor(b),
    "min": min_op,
    "max": max_op,
    "and": lambda a, b: torch.bitwise_and(unwrap_tensor(a), unwrap_tensor(b)),
    "or": lambda a, b: torch.bitwise_or(unwrap_tensor(a), unwrap_tensor(b)),
    "xor": lambda a, b: torch.bitwise_xor(unwrap_tensor(a), unwrap_tensor(b)),
}


@requires_tma
# TODO: interpreter support
# @pytest.mark.interpreter
@pytest.mark.parametrize("kind", ["add", "min", "max", "and", "or", "xor"])
@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("num_ctas", [1, 2])
@pytest.mark.parametrize("descriptor", ["host", "device"])
@pytest.mark.parametrize("M_BLOCK,N_BLOCK", [(2, 16), (8, 16), (8, 32), (8, 128)])
def test_tensor_descriptor_reduce(kind, descriptor, dtype_str, num_ctas, M_BLOCK, N_BLOCK):

    @triton.jit(debug=True)
    def kernel(out_desc, out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr, kind: tl.constexpr):
        moffset = tl.program_id(0) * M_BLOCK
        noffset = tl.program_id(1) * N_BLOCK

        midx = moffset + tl.arange(0, M_BLOCK)[:, None]
        nidx = noffset + tl.arange(0, N_BLOCK)[None, :]
        idx = midx * N + nidx

        val = tl.load(a_ptr + idx)

        if out_desc is None:
            desc = tl.make_tensor_descriptor(
                out_ptr,
                shape=[M, N],
                strides=[N, 1],
                block_shape=[M_BLOCK, N_BLOCK],
            )
        else:
            desc = out_desc

        assert desc.shape[0] == M
        assert desc.shape[1] == N
        assert desc.strides[0] == N
        assert desc.strides[1] == 1
        assert desc.block_shape == [M_BLOCK, N_BLOCK]
        if kind == "add":
            desc.atomic_add([moffset, noffset], val)
        elif kind == "min":
            desc.atomic_min([moffset, noffset], val)
        elif kind == "max":
            desc.atomic_max([moffset, noffset], val)
        elif kind == "and":
            desc.atomic_and([moffset, noffset], val)
        elif kind == "or":
            desc.atomic_or([moffset, noffset], val)
        else:
            tl.static_assert(kind == "xor")
            desc.atomic_xor([moffset, noffset], val)

    M, N = 32, 128
    rs = np.random.RandomState(seed=17)
    inp = to_triton(numpy_random((M, N), dtype_str, rs), device="cuda", dst_type=dtype_str)
    out = to_triton(numpy_random((M, N), dtype_str, rs), device="cuda", dst_type=dtype_str)

    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK

    if descriptor == "host":
        out_desc = TensorDescriptor.from_tensor(out, [M_BLOCK, N_BLOCK])
    else:

        def alloc_fn(size: int, align: int, stream: Optional[int]):
            assert size == 128 * (grid_m * grid_n) * num_ctas
            assert align == 128
            assert stream == 0
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)
        out_desc = None

    supported = getattr(tl, dtype_str) in SUPPORTED_REDUCE_DTYPES[kind]
    if not supported:
        with pytest.raises(CompilationError):
            kernel[(grid_m, grid_n)](out_desc, out, inp, M, N, M_BLOCK, N_BLOCK, kind, num_ctas=num_ctas)
        return

    expect = REDUCE_OP[kind](inp, out)
    kernel[(grid_m, grid_n)](out_desc, out, inp, M, N, M_BLOCK, N_BLOCK, kind, num_ctas=num_ctas)
    torch.testing.assert_close(expect, unwrap_tensor(out), check_dtype=False)


@triton.jit
def tma_gather_rows_kernel(out_ptr, in_ptr, idx_ptr, y, X: tl.constexpr, Y: tl.constexpr, BLOCK_X: tl.constexpr,
                           BLOCK_Y: tl.constexpr):
    idx = tl.load(idx_ptr + tl.arange(0, BLOCK_X))
    desc = tl.make_tensor_descriptor(in_ptr, [X, Y], [Y, 1], [1, BLOCK_Y])
    out = desc.gather(idx, y)
    tl.store(out_ptr + tl.arange(0, BLOCK_X)[:, None] * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :], out)


def torch_gather_rows(input, idx, y, block_y):
    out = torch.empty(0, device=input.device, dtype=input.dtype)
    for i in idx:
        x = input[i][y:y + block_y]
        out = torch.cat((out, x.reshape(1, x.shape[0])), dim=0)
    return out


@pytest.mark.interpreter
@pytest.mark.parametrize("X, Y", [(128, 128), (64, 256)])
@pytest.mark.parametrize("BLOCK_X, BLOCK_Y", [(32, 32), (64, 128), (16, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int8])
@pytest.mark.parametrize("y", [0, 32, 48])
@pytest.mark.skipif(not is_interpreter() and torch.cuda.get_device_capability()[0] != 10,
                    reason="TMA Gather only works on cloud Blackwell Chips")
def test_tma_gather(X, Y, BLOCK_X, BLOCK_Y, dtype, y, device):
    if BLOCK_X > X or y + BLOCK_Y > Y:
        pytest.skip()

    torch.manual_seed(42)
    if dtype != torch.int8:
        input = torch.rand((X, Y), dtype=dtype, device=device)
    else:
        input = torch.arange(X * Y, dtype=dtype, device=device).reshape(X, Y)
    output = torch.empty((BLOCK_X, BLOCK_Y), dtype=dtype, device=device)

    idx = torch.randint(BLOCK_X, (BLOCK_X, ), dtype=torch.int32, device=device)

    def alloc_fn(size: int, align: int, steam):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)

    tma_gather_rows_kernel[(1, )](output, input, idx, y, X, Y, BLOCK_X, BLOCK_Y)

    ref = torch_gather_rows(input, idx, y, BLOCK_Y)
    torch.testing.assert_close(ref, output, atol=0, rtol=0)


@triton.jit
def tma_gather_dot_pipeline(  #
        a_ptr, b_ptr, output_ptr,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        K: tl.constexpr,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
):
    a_desc = tl.make_tensor_descriptor(a_ptr, [BLOCK_M, K], [K, 1], [1, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, [K, BLOCK_N], [BLOCK_N, 1], [1, BLOCK_N])

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    for k in range(0, K, BLOCK_K):
        a = a_desc.gather(tl.arange(0, BLOCK_M), k)
        b = b_desc.gather(tl.arange(0, BLOCK_K) + k, 0)
        accumulator = tl.dot(a, b, acc=accumulator)

    offs_cm = tl.arange(0, BLOCK_M)
    offs_cn = tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(output_ptrs, accumulator)


@pytest.mark.interpreter
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(16, 16, 16)])
@pytest.mark.parametrize("K", [128])
@pytest.mark.skipif(not is_interpreter() and torch.cuda.get_device_capability()[0] != 10,
                    reason="TMA Gather only works on cloud Blackwell Chips")
def test_tma_gather_dot_pipeline(BLOCK_M, BLOCK_N, BLOCK_K, K, device):

    def alloc_fn(size: int, align: int, steam):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)

    a = torch.arange(BLOCK_M * K, device=device).reshape(BLOCK_M, K).float()
    b = torch.arange(K * BLOCK_N, device=device).reshape(K, BLOCK_N).float()

    c = a @ b

    output = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.float32, device=device)
    if not is_interpreter():
        kernel = tma_gather_dot_pipeline.warmup(a, b, output, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                                output.stride(0), output.stride(1), K, BLOCK_M, BLOCK_N, BLOCK_K,
                                                grid=(1, ))
        assert kernel.asm["ttgir"].count("ttng.async_tma_gather") == 6
    tma_gather_dot_pipeline[(1, 1, 1)](a, b, output, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                       output.stride(0), output.stride(1), K, BLOCK_M, BLOCK_N, BLOCK_K)

    torch.testing.assert_close(c, output)


def torch_scatter_rows(input, idx, y, block_y, X, Y):
    out = torch.zeros((X, Y), dtype=input.dtype, device=input.device)
    for i, j in enumerate(idx):
        out[j][y:y + block_y] = input[i]
    return out


@triton.jit
def tma_scatter_rows_kernel(out_ptr, in_ptr, idx_ptr, y, X: tl.constexpr, Y: tl.constexpr, BLOCK_X: tl.constexpr,
                            BLOCK_Y: tl.constexpr):
    idx = tl.load(idx_ptr + tl.arange(0, BLOCK_X))
    data = tl.load(in_ptr + tl.arange(0, BLOCK_X)[:, None] * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :])
    desc = tl.make_tensor_descriptor(out_ptr, [X, Y], [Y, 1], [1, BLOCK_Y])
    desc.scatter(data, idx, y)


@pytest.mark.interpreter
@pytest.mark.parametrize("X, Y", [(128, 128), (64, 256)])
@pytest.mark.parametrize("BLOCK_X, BLOCK_Y", [(32, 32), (64, 128), (16, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int8])
@pytest.mark.parametrize("y", [0, 32, 48])
@pytest.mark.skipif(not is_interpreter() and torch.cuda.get_device_capability()[0] != 10,
                    reason="TMA Gather only works on cloud Blackwell Chips")
def test_tma_scatter(X, Y, BLOCK_X, BLOCK_Y, dtype, y):
    if BLOCK_X > X or y + BLOCK_Y > Y:
        pytest.skip()

    torch.manual_seed(42)
    input = torch.arange(BLOCK_X * BLOCK_Y, dtype=dtype, device='cuda').reshape(BLOCK_X, BLOCK_Y)
    output = torch.zeros((X, Y), dtype=dtype, device='cuda')

    idx = torch.randperm(BLOCK_X, dtype=torch.int32, device='cuda')

    def alloc_fn(size: int, align: int, steam):
        return torch.empty(size, dtype=torch.int8, device='cuda')

    triton.set_allocator(alloc_fn)

    tma_scatter_rows_kernel[(1, )](output, input, idx, y, X, Y, BLOCK_X, BLOCK_Y)

    ref = torch_scatter_rows(input, idx, y, BLOCK_Y, X, Y)
    torch.testing.assert_close(ref, output, atol=0, rtol=0)


@requires_tma
@pytest.mark.interpreter()
@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("num_ctas", [1, 2])
@pytest.mark.parametrize("M_BLOCK,N_BLOCK", [(2, 16), (8, 16), (8, 32), (8, 128)])
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
    (256, 64, 32, 2),
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
