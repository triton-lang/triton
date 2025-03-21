import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_interpreter, numpy_random, to_triton, requires_tma, unwrap_tensor, tma_dtypes
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
from typing import Optional


@requires_tma
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", tma_dtypes)
def test_tensor_descriptor_load(dtype_str):

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[M_BLOCK, N_BLOCK],
        )

        assert desc.shape[0] == M
        assert desc.shape[1] == N
        assert desc.strides[0] == N
        assert desc.strides[1] == 1
        assert desc.block_shape == [M_BLOCK, N_BLOCK]
        block = desc.load([M_BLOCK, 2 * N_BLOCK])
        idx = tl.arange(0, M_BLOCK)[:, None] * N_BLOCK + tl.arange(0, N_BLOCK)[None, :]
        tl.store(out_ptr + idx, block)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert size == 128
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    M, N = 32, 128
    inp = to_triton(numpy_random((M, N), dtype_str), device="cuda", dst_type=dtype_str)

    M_BLOCK = 8
    N_BLOCK = 32
    out = inp.new_empty((M_BLOCK, N_BLOCK))

    kernel[(1, )](out, inp, M, N, M_BLOCK, N_BLOCK)

    expect = unwrap_tensor(inp)[1 * M_BLOCK:2 * M_BLOCK, 2 * N_BLOCK:3 * N_BLOCK]
    torch.testing.assert_close(expect, unwrap_tensor(out))


@requires_tma
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", tma_dtypes)
def test_tensor_descriptor_store(dtype_str):

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        moffset = tl.program_id(0) * M_BLOCK
        noffset = tl.program_id(1) * N_BLOCK

        midx = moffset + tl.arange(0, M_BLOCK)[:, None]
        nidx = noffset + tl.arange(0, N_BLOCK)[None, :]
        idx = midx * N + nidx

        val = tl.load(a_ptr + idx)

        desc = tl.make_tensor_descriptor(
            out_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[M_BLOCK, N_BLOCK],
        )

        assert desc.shape[0] == M
        assert desc.shape[1] == N
        assert desc.strides[0] == N
        assert desc.strides[1] == 1
        assert desc.block_shape == [M_BLOCK, N_BLOCK]
        desc.store([moffset, noffset], val)

    M, N = 32, 128
    inp = to_triton(numpy_random((M, N), dtype_str), device="cuda", dst_type=dtype_str)

    M_BLOCK = 8
    N_BLOCK = 32
    out = inp.new_empty((M, N))

    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert size == 128 * (grid_m * grid_n)
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    kernel[(grid_m, grid_n)](out, inp, M, N, M_BLOCK, N_BLOCK)

    torch.testing.assert_close(unwrap_tensor(inp), unwrap_tensor(out))


@requires_tma
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("K_BLOCK", [16, 32, 64, 128])
def test_tensor_descriptor_load3d(dtype_str, K_BLOCK):

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, K, stride_m, stride_n, stride_k, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr,
               K_BLOCK: tl.constexpr):
        desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, N, K],
            strides=[stride_m, stride_n, stride_k],
            block_shape=[M_BLOCK, N_BLOCK, K_BLOCK],
        )

        pid_m, pid_n, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        offs = pid_m * M_BLOCK, pid_n * N_BLOCK, pid_k * K_BLOCK

        block = desc.load(offs)

        idx_m = offs[0] + tl.arange(0, M_BLOCK)[:, None, None]
        idx_n = offs[1] + tl.arange(0, N_BLOCK)[None, :, None]
        idx_k = offs[2] + tl.arange(0, K_BLOCK)[None, None, :]
        idx = idx_m * N * K + idx_n * K + idx_k
        mask = (idx_m < M) & (idx_n < N) & (idx_k < K)
        tl.store(out_ptr + idx, block, mask)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    inp = to_triton(numpy_random((10, 64, 128), dtype_str), device="cuda", dst_type=dtype_str)
    inp.data = inp.data[:, :50, :119]

    if K_BLOCK * inp.element_size() < 32:
        return pytest.skip("Invalid last dim size")

    M_BLOCK, N_BLOCK = 8, 8
    out = inp.new_empty(inp.shape)

    grid = tuple(triton.cdiv(size, block) for size, block in zip(inp.shape, (M_BLOCK, N_BLOCK, K_BLOCK)))
    kernel[grid](out, inp, *inp.shape, *inp.stride(), M_BLOCK, N_BLOCK, K_BLOCK)

    actual = unwrap_tensor(out)
    expect = unwrap_tensor(inp)
    torch.testing.assert_close(expect, actual)


@requires_tma
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("K_BLOCK", [16, 32, 64, 128])
def test_tensor_descriptor_store3d(dtype_str, K_BLOCK):

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, K, stride_m, stride_n, stride_k, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr,
               K_BLOCK: tl.constexpr):
        desc = tl.make_tensor_descriptor(
            out_ptr,
            shape=[M, N, K],
            strides=[stride_m, stride_n, stride_k],
            block_shape=[M_BLOCK, N_BLOCK, K_BLOCK],
        )

        pid_m, pid_n, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        offs = pid_m * M_BLOCK, pid_n * N_BLOCK, pid_k * K_BLOCK

        idx_m = offs[0] + tl.arange(0, M_BLOCK)[:, None, None]
        idx_n = offs[1] + tl.arange(0, N_BLOCK)[None, :, None]
        idx_k = offs[2] + tl.arange(0, K_BLOCK)[None, None, :]
        idx = idx_m * N * K + idx_n * K + idx_k
        mask = (idx_m < M) & (idx_n < N) & (idx_k < K)
        block = tl.load(a_ptr + idx, mask)

        desc.store(offs, block)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    inp = to_triton(numpy_random((10, 50, 119), dtype_str), device="cuda", dst_type=dtype_str)

    if K_BLOCK * inp.element_size() < 32:
        return pytest.skip("Invalid last dim size")

    M_BLOCK, N_BLOCK = 8, 8
    out = inp.new_empty((10, 64, 128))

    grid = tuple(triton.cdiv(size, block) for size, block in zip(inp.shape, (M_BLOCK, N_BLOCK, K_BLOCK)))
    kernel[grid](out, inp, *inp.shape, *out.stride(), M_BLOCK, N_BLOCK, K_BLOCK)

    expect = unwrap_tensor(inp)
    actual = unwrap_tensor(out)[:, :50, :119]
    torch.testing.assert_close(expect, actual)


@requires_tma
@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("ndim", [2, 3, 4, 5])
@pytest.mark.parametrize("INNER_BLOCK", [16, 32, 64, 128])
def test_tensor_descriptor_load_nd(dtype_str, ndim, INNER_BLOCK):

    @triton.jit
    def kernel(out_ptr, a_ptr, shape, strides, BLOCK_SHAPE):
        desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=shape,
            strides=strides,
            block_shape=BLOCK_SHAPE,
        )
        ndim: tl.constexpr = len(BLOCK_SHAPE)

        offs = (0, ) * ndim
        block = desc.load(offs)

        idx = tl.full(BLOCK_SHAPE, 0, tl.int32)
        stride = 1
        for k in tl.static_range(ndim - 1, -1, -1):
            arange = tl.arange(0, BLOCK_SHAPE[k])
            for _ in tl.static_range(k):
                arange = tl.expand_dims(arange, 0)
            for _ in tl.static_range(k + 1, ndim):
                arange = tl.expand_dims(arange, -1)

            idx += arange * stride
            stride *= BLOCK_SHAPE[k]

        tl.store(out_ptr + idx, block)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    alloc_shape = (1, 1, 3, 7, INNER_BLOCK)[-ndim:]
    inp = to_triton(numpy_random(alloc_shape, dtype_str), device="cuda", dst_type=dtype_str)
    inp.data = inp.data[..., :INNER_BLOCK - 3]

    if INNER_BLOCK * inp.element_size() < 32:
        return pytest.skip("Invalid last dim size")

    BLOCK_SHAPE = (2, 2, 4, 8, INNER_BLOCK)[-ndim:]
    out = inp.new_empty(BLOCK_SHAPE)

    constexpr_block_shape = tuple(tl.constexpr(v) for v in BLOCK_SHAPE)
    kernel[(1, )](out, inp, inp.shape, inp.stride(), constexpr_block_shape)

    # Check in-bounds
    actual = unwrap_tensor(out)
    expect = unwrap_tensor(inp)
    idx = [slice(None, s) for s in inp.shape]
    torch.testing.assert_close(expect, actual[idx])

    # Check out-of-bounds
    actual[idx].zero_()
    expect = expect.new_zeros(BLOCK_SHAPE)
    torch.testing.assert_close(expect, actual)


@requires_tma
@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("ndim", [2, 3, 4, 5])
@pytest.mark.parametrize("INNER_BLOCK", [16, 32, 64, 128])
def test_tensor_descriptor_store_nd(dtype_str, ndim, INNER_BLOCK):

    @triton.jit
    def kernel(out_ptr, a_ptr, shape, strides, BLOCK_SHAPE):
        desc = tl.make_tensor_descriptor(
            out_ptr,
            shape=shape,
            strides=strides,
            block_shape=BLOCK_SHAPE,
        )
        ndim: tl.constexpr = len(BLOCK_SHAPE)

        idx = tl.full(BLOCK_SHAPE, 0, tl.int32)
        stride = 1
        for k in tl.static_range(ndim - 1, -1, -1):
            arange = tl.arange(0, BLOCK_SHAPE[k])
            for _ in tl.static_range(k):
                arange = tl.expand_dims(arange, 0)
            for _ in tl.static_range(k + 1, ndim):
                arange = tl.expand_dims(arange, -1)

            idx += arange * stride
            stride *= BLOCK_SHAPE[k]

        block = tl.load(a_ptr + idx)

        offs = (0, ) * ndim
        desc.store(offs, block)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    BLOCK_SHAPE = (2, 2, 4, 8, INNER_BLOCK)[-ndim:]
    inp = to_triton(numpy_random(BLOCK_SHAPE, dtype_str), device="cuda", dst_type=dtype_str)

    if INNER_BLOCK * inp.element_size() < 32:
        return pytest.skip("Invalid last dim size")

    out = inp.new_empty(BLOCK_SHAPE)
    out.data.fill_(-1)

    desc_shape = (1, 1, 3, 7, INNER_BLOCK)[-ndim:]
    constexpr_block_shape = tuple(tl.constexpr(v) for v in BLOCK_SHAPE)
    kernel[(1, )](out, inp, desc_shape, out.stride(), constexpr_block_shape)

    # Check in-bounds
    actual = unwrap_tensor(out)
    expect = unwrap_tensor(inp)
    idx = [slice(None, s) for s in desc_shape]
    torch.testing.assert_close(expect[idx], actual[idx])

    # Check out-of-bounds
    actual[idx].fill_(-1)
    expect = expect.new_full(BLOCK_SHAPE, -1)
    torch.testing.assert_close(expect, actual)


@triton.jit(noinline=True)
def tensor_descriptor_in_function_helper(out_ptr, in_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    in_desc = tl.make_tensor_descriptor(
        in_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK
    value = in_desc.load([moffset, noffset])
    out_desc.store([moffset, noffset], value.abs())


@requires_tma
@pytest.mark.interpreter
def test_tensor_descriptor_in_function():

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        tensor_descriptor_in_function_helper(out_ptr, a_ptr, M, N, M_BLOCK, N_BLOCK)

    M, N = 32, 128
    inp = torch.randn((M, N), device="cuda")

    M_BLOCK = 8
    N_BLOCK = 32
    out = inp.new_empty((M, N))

    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert size == 2 * 128 * (grid_m * grid_n)
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    expect = inp.abs()
    kernel[(grid_m, grid_n)](out, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(expect, out)


@triton.jit(noinline=True)
def tensor_descriptor_return_helper(ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    return tl.make_tensor_descriptor(
        ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )


@requires_tma
@pytest.mark.interpreter
def test_tensor_descriptor_return_value():

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        in_desc = tensor_descriptor_return_helper(a_ptr, M, N, M_BLOCK, N_BLOCK)
        out_desc = tensor_descriptor_return_helper(out_ptr, M, N, M_BLOCK, N_BLOCK)
        moffset = tl.program_id(0) * M_BLOCK
        noffset = tl.program_id(1) * N_BLOCK
        value = in_desc.load([moffset, noffset])
        out_desc.store([moffset, noffset], value.abs())

    M, N = 32, 128
    inp = torch.randn((M, N), device="cuda")

    M_BLOCK = 8
    N_BLOCK = 32
    out = inp.new_zeros((M, N))

    def alloc_fn(size: int, align: int, stream: Optional[int]) -> torch.Tensor:
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    expect = inp.abs()
    kernel[(M // M_BLOCK, N // N_BLOCK)](out, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(expect, out)


@triton.jit
def matmul_kernel_make_tensor_desciptor(a_ptr, b_ptr, c_ptr,  #
                                        M, N, K,  #
                                        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                        BLOCK_SIZE_K: tl.constexpr,  #
                                        ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_k, offs_bn])
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    accumulator = accumulator.to(a_desc.dtype)
    c_desc.store([offs_am, offs_bn], accumulator)


@requires_tma
@pytest.mark.interpreter
@pytest.mark.parametrize("num_stages", [1, 4])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(32, 32, 32), (128, 64, 64), (128, 128, 64), (128, 256, 64)])
def testmake_tensor_descriptor_matmul(num_stages, BLOCK_M, BLOCK_N, BLOCK_K):
    device = "cuda"
    if is_interpreter():
        M, N, K = BLOCK_M, BLOCK_N, BLOCK_K
    else:
        M, N, K = 8192, 8192, 1024
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((K, N), dtype=torch.float16, device=device)
    C = torch.empty((M, N), dtype=torch.float16, device=device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert size == 3 * 128 * grid[0]
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    kernel = matmul_kernel_make_tensor_desciptor[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=8,
        num_stages=num_stages,
    )
    ref_out = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.float16)
    torch.testing.assert_close(ref_out, C, rtol=1e-3, atol=1e-3)
    if is_interpreter():
        return

    assert "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned" in kernel.asm["ptx"]
    if BLOCK_M >= 64 and BLOCK_N >= 64 and torch.cuda.get_device_capability()[0] == 9:
        # TODO: The use of stmatrix for Blackwell is currently not supported.
        # Only a subset of TMEM and stmatrix layout pairs are compatible, for example 16x256bx2 and m8n8x4.
        assert "stmatrix.sync.aligned.m8n8.x4.shared.b16" in kernel.asm["ptx"]


@triton.jit
def kernel_make_tensor_desciptor_loop_carried(a_ptr, M, N, MBLOCK: tl.constexpr, NBLOCK: tl.constexpr):
    # Test that descriptors work with
    pid = tl.program_id(0)
    moffset = MBLOCK * pid

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[MBLOCK, NBLOCK],
    )

    for i in range(0, N, NBLOCK):
        assert isinstance(a_desc, tl.tensor_descriptor)
        if i % (3 * NBLOCK) == 0:
            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[M, N],
                strides=[N, 1],
                block_shape=[MBLOCK, NBLOCK],
            )
            assert isinstance(a_desc, tl.tensor_descriptor)
        assert isinstance(a_desc, tl.tensor_descriptor)
        a = a_desc.load([moffset, i])
        a_desc.store([moffset, i], a + 10)

    n = 0
    while n < N:
        assert isinstance(a_desc, tl.tensor_descriptor)
        if n % (3 * NBLOCK) == 0:
            assert isinstance(a_desc, tl.tensor_descriptor)
            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[M, N],
                strides=[N, 1],
                block_shape=[MBLOCK, NBLOCK],
            )
        assert isinstance(a_desc, tl.tensor_descriptor)
        a = a_desc.load([moffset, n])
        a_desc.store([moffset, n], a + 5)

        n += NBLOCK


@requires_tma
@pytest.mark.interpreter
def test_make_tensor_descriptor_loop_carried():
    device = "cuda"
    M, N = 64, 512
    torch.manual_seed(42)
    A = torch.randn((M, N), dtype=torch.float32, device=device)
    MBLOCK, NBLOCK = 8, 128
    grid = (triton.cdiv(M, MBLOCK), )

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert size == 128 * grid[0]
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    ref_out = A + 15
    kernel = kernel_make_tensor_desciptor_loop_carried[grid](
        A,
        M,
        N,
        MBLOCK,
        NBLOCK,
    )
    torch.testing.assert_close(ref_out, A)
    if not is_interpreter():
        assert "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned" in kernel.asm[
            "ptx"]


@triton.jit
def batched_gemm_2d_tma_kernel(a_ptr, b_ptr, c_ptr,  #
                               B, M, N, K,  #
                               dtype: tl.constexpr,  #
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
                               NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(axis=0)
    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles_per_batch = num_tiles_m * num_tiles_n
    num_tiles = B * num_tiles_per_batch

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    tile_m = 0
    tile_n = 0
    tile_b = 0

    offs_m = 0
    offs_n = 0
    offs_b = 0

    a_desc = tl.make_tensor_descriptor(a_ptr + offs_b * (M * K), [M, K], [K, 1], [BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr + offs_b * (N * K), [N, K], [K, 1], [BLOCK_N, BLOCK_K])
    c_desc = tl.make_tensor_descriptor(c_ptr + offs_b * (M * N), [M, N], [N, 1], [BLOCK_M, BLOCK_N])

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            tile_b = tile_id // num_tiles_per_batch
            tile_m = (tile_id // num_tiles_n) % num_tiles_m
            tile_n = tile_id % num_tiles_n

            offs_b = tile_b
            offs_m = tile_m * BLOCK_M
            offs_n = tile_n * BLOCK_N

            a_desc = tl.make_tensor_descriptor(a_ptr + offs_b * (M * K), [M, K], [K, 1], [BLOCK_M, BLOCK_K])
            b_desc = tl.make_tensor_descriptor(b_ptr + offs_b * (N * K), [N, K], [K, 1], [BLOCK_N, BLOCK_K])
            c_desc = tl.make_tensor_descriptor(c_ptr + offs_b * (M * N), [M, N], [N, 1], [BLOCK_M, BLOCK_N])

        offs_k = ki * BLOCK_K

        a = a_desc.load([offs_m, offs_k])
        b = b_desc.load([offs_n, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            c = accumulator.to(dtype)

            c_desc.store([offs_m, offs_n], c)
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


@requires_tma
@pytest.mark.interpreter
def test_tensor_descriptor_batched_gemm_2d_tma():
    device = "cuda"
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
    if is_interpreter():
        B, M, N, K = 2, BLOCK_M, BLOCK_N, BLOCK_K
    else:
        B, M, N, K = 2, 1024, 1024, 128
    NUM_SMS = 96
    num_stages = 3

    grid = (min(NUM_SMS, B * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), )

    a = torch.randn((B, M, K), device=device, dtype=torch.float16)
    b = torch.randn((B, N, K), device=device, dtype=torch.float16)
    c = torch.empty((B, M, N), device=device, dtype=torch.float16)

    expect = torch.bmm(a, b.mT)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        # TODO: should only need num_stages * 3 descriptors per SM
        assert size == 128 * 3 * (num_stages + 1) * grid[0]
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    batched_gemm_2d_tma_kernel[grid](
        a, b, c,  #
        B, M, N, K,  #
        tl.float16,  #
        BLOCK_M, BLOCK_N, BLOCK_K,  #
        NUM_SMS,  #
        num_stages=num_stages, num_warps=8)
    torch.cuda.synchronize()

    torch.testing.assert_close(c, expect, rtol=1e-3, atol=1e-3)


@triton.jit
def batched_gemm_3d_tma_kernel(a_ptr, b_ptr, c_ptr,  #
                               B, M, N, K,  #
                               dtype: tl.constexpr,  #
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
                               NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(axis=0)
    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles_per_batch = num_tiles_m * num_tiles_n
    num_tiles = B * num_tiles_per_batch

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    tile_m = 0
    tile_n = 0
    tile_b = 0

    offs_m = 0
    offs_n = 0
    offs_b = 0

    a_desc = tl.make_tensor_descriptor(a_ptr, [B, M, K], [K * M, K, 1], [1, BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, [B, N, K], [N * K, K, 1], [1, BLOCK_N, BLOCK_K])
    c_desc = tl.make_tensor_descriptor(c_ptr, [B, M, N], [M * N, N, 1], [1, BLOCK_M, BLOCK_N])

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            tile_b = tile_id // num_tiles_per_batch
            tile_m = (tile_id // num_tiles_n) % num_tiles_m
            tile_n = tile_id % num_tiles_n

            offs_b = tile_b
            offs_m = tile_m * BLOCK_M
            offs_n = tile_n * BLOCK_N

        offs_k = ki * BLOCK_K

        a = a_desc.load([offs_b, offs_m, offs_k]).reshape([BLOCK_M, BLOCK_K])
        b = b_desc.load([offs_b, offs_n, offs_k]).reshape([BLOCK_N, BLOCK_K])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            c = accumulator.to(dtype)

            c_desc.store([offs_b, offs_m, offs_n], c.reshape((1, BLOCK_M, BLOCK_N)))
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


@requires_tma
@pytest.mark.interpreter
def test_tensor_descriptor_batched_gemm_3d_tma():
    device = "cuda"
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
    if is_interpreter():
        B, M, N, K = 2, BLOCK_M, BLOCK_N, BLOCK_K
    else:
        B, M, N, K = 2, 1024, 1024, 128
    NUM_SMS = 96
    num_stages = 3

    grid = (min(NUM_SMS, B * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), )

    a = torch.randn((B, M, K), device=device, dtype=torch.float16)
    b = torch.randn((B, N, K), device=device, dtype=torch.float16)
    c = torch.empty((B, M, N), device=device, dtype=torch.float16)

    expect = torch.bmm(a, b.mT)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        # TODO: should only need num_stages * 3 descriptors per SM
        assert size == 128 * 3 * grid[0]
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    h = batched_gemm_3d_tma_kernel[grid](
        a, b, c,  #
        B, M, N, K,  #
        tl.float16,  #
        BLOCK_M, BLOCK_N, BLOCK_K,  #
        NUM_SMS,  #
        num_stages=num_stages, num_warps=8)
    torch.cuda.synchronize()

    if not is_interpreter():
        capability = torch.cuda.get_device_capability(0)[0]
        dot_op = {9: "warp_group_dot", 10: "tc_gen5_mma"}
        assert dot_op[capability] in h.asm["ttgir"]

    torch.testing.assert_close(c, expect, rtol=1e-3, atol=1e-3)


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
@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("ndim", [3, 4, 5])
@pytest.mark.parametrize("INNER_BLOCK", [16, 32, 64, 128])
def test_tensor_descriptor_rank_reducing_load(dtype_str, ndim, INNER_BLOCK):

    @triton.jit
    def kernel(out_ptr, a_ptr, shape, strides, BLOCK_SHAPE):
        desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=shape,
            strides=strides,
            block_shape=BLOCK_SHAPE,
        )
        ndim: tl.constexpr = len(BLOCK_SHAPE)

        offs = (0, ) * ndim
        M_BLOCK: tl.constexpr = BLOCK_SHAPE[-2]
        N_BLOCK: tl.constexpr = BLOCK_SHAPE[-1]
        block = desc.load(offs).reshape(M_BLOCK, N_BLOCK)

        idx = tl.arange(0, M_BLOCK)[:, None] * strides[-2] + tl.arange(0, N_BLOCK)[None, :]
        tl.store(out_ptr + idx, block)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    alloc_shape = (1, 1, 1, 7, INNER_BLOCK)[-ndim:]
    inp = to_triton(numpy_random(alloc_shape, dtype_str), device="cuda", dst_type=dtype_str)
    inp.data = inp.data[..., :INNER_BLOCK - 3]

    if INNER_BLOCK * inp.element_size() < 32:
        return pytest.skip("Invalid last dim size")

    BLOCK_SHAPE = (1, 1, 1, 8, INNER_BLOCK)[-ndim:]
    out = inp.new_empty(BLOCK_SHAPE)

    constexpr_block_shape = tuple(tl.constexpr(v) for v in BLOCK_SHAPE)
    kernel[(1, )](out, inp, inp.shape, inp.stride(), constexpr_block_shape)

    # Check in-bounds
    actual = unwrap_tensor(out)
    expect = unwrap_tensor(inp)
    idx = [slice(None, s) for s in inp.shape]
    torch.testing.assert_close(expect, actual[idx])

    # Check out-of-bounds
    actual[idx].zero_()
    expect = expect.new_zeros(BLOCK_SHAPE)
    torch.testing.assert_close(expect, actual)


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit()
def matmul_kernel_rank_reducing(a_ptr, b_ptr, c_ptr,  #
                                M, N, K,  #
                                BLOCK_SIZE_M: tl.constexpr,  #
                                BLOCK_SIZE_N: tl.constexpr,  #
                                BLOCK_SIZE_K: tl.constexpr,  #
                                NUM_SMS: tl.constexpr):  #
    # Matmul using TMA and device-side descriptor creation
    GROUP_SIZE_M: tl.constexpr = 8
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[1, M, K],
        strides=[M * K, K, 1],
        block_shape=[1, BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[1, N, K],
        strides=[N * K, K, 1],
        block_shape=[1, BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[1, M, N],
        strides=[M * N, N, 1],
        block_shape=[1, BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([0, offs_am, offs_k]).reshape(BLOCK_SIZE_M, BLOCK_SIZE_K)
            b = b_desc.load([0, offs_bn, offs_k]).reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        c = accumulator.to(dtype).reshape(1, BLOCK_SIZE_M, BLOCK_SIZE_N)
        c_desc.store([0, offs_cm, offs_cn], c)


@requires_tma
@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16", "float32"])
def test_tensor_descriptor_rank_reducing_matmul(dtype_str):
    NUM_SMS = 4
    M, N, K = 256, 256, 64
    A = to_triton(numpy_random((1, M, K), dtype_str), device="cuda", dst_type=dtype_str)
    B = to_triton(numpy_random((1, N, K), dtype_str), device="cuda", dst_type=dtype_str)
    C = A.new_empty(1, M, N)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)
    matmul_kernel_rank_reducing[(NUM_SMS, )](
        A,
        B,
        C,
        M,
        N,
        K,
        NUM_SMS=4,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
    )

    actual = unwrap_tensor(C)
    expect = torch.matmul(A, B.mT)
    torch.testing.assert_close(expect, actual, atol=1e-1, rtol=1e-4)


def f8_to_f16(x, dtype):

    @triton.jit
    def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(Y + offs, x, mask=mask)

    ret = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']), )
    dtype = getattr(tl, dtype)
    kernel[grid](ret, triton.reinterpret(x, dtype), ret.numel(), BLOCK_SIZE=1024)
    return ret


@triton.jit
def mxfp8_mxfp4_matmul_tma(  #
        a_ptr, b_ptr, output_ptr,  #
        a_scale, b_scale,  #
        M, N, K,  #
        stride_scale,  #
        stride_am, stride_ak,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr):  #
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_bn_tma = pid_n * BLOCK_N
    offs_ak = tl.arange(0, BLOCK_K)
    offs_scale_k = tl.arange(0, BLOCK_K // 32)
    a_scale_ptr = a_scale + offs_am[:, None] * stride_scale + offs_scale_k[None, :]
    b_scale_ptr = b_scale + offs_bn[:, None] * stride_scale + offs_scale_k[None, :]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    offs_bk = 0

    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K // 2],
        strides=[K // 2, 1],
        block_shape=[BLOCK_N, BLOCK_K // 2],
    )

    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = tl.load(a_ptrs)
        b = b_desc.load([offs_bn_tma, offs_bk])

        scale_a = tl.load(a_scale_ptr)
        scale_b = tl.load(b_scale_ptr)
        accumulator = tl.dot_scaled(a, scale_a, "e5m2", b.T, scale_b, "e2m1", accumulator)
        a_ptrs += BLOCK_K * stride_ak

        offs_bk += b_desc.block_shape[-1]
        a_scale_ptr += BLOCK_K // 32
        b_scale_ptr += BLOCK_K // 32

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=c_mask)


@requires_tma
@pytest.mark.parametrize("M, N, K", [(1024, 512, 256), (128, 256, 256), (8192, 8192, 8192)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 128, 128), (128, 128, 256), (128, 256, 128),
                                                       (128, 256, 256)])
@pytest.mark.parametrize("NUM_STAGES", [1, 3])
def test_mxfp8_mxfp4_matmul_tma(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, device):
    if BLOCK_N == 256 and BLOCK_K == 256:
        NUM_STAGES = min(NUM_STAGES, 2)

    if BLOCK_K < K and torch.cuda.get_device_capability(0)[0] != 10:
        pytest.skip("Currently broken on hopper")

    a = torch.randint(20, 40, (M, K), dtype=torch.uint8).view(torch.float8_e5m2).to(device)

    dtype_src_str = "float8e5"

    b_mxfp4 = MXFP4Tensor(size=(N, K), device=device).random()
    b = b_mxfp4.to_packed_tensor(dim=1)
    b_ref = b_mxfp4.to(torch.float32).T

    a_scale_mxfp4 = MXScaleTensor(size=(M, (K + 32 - 1) // 32), device=device).random(high=64.0)
    b_scale_mxfp4 = MXScaleTensor(size=(N, (K + 32 - 1) // 32), device=device).random(high=64.0)
    a_scale = a_scale_mxfp4.data
    b_scale = b_scale_mxfp4.data

    a_scale_ref = a_scale_mxfp4.to(torch.float32).repeat_interleave(32, dim=1)[:M, :K]
    b_scale_ref = b_scale_mxfp4.to(torch.float32).repeat_interleave(32, dim=1).T.contiguous()[:K, :N]

    output = a.new_empty((M, N), dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    mxfp8_mxfp4_matmul_tma[grid](a, b, output, a_scale, b_scale, M, N, K, a_scale.stride(0), a.stride(0), a.stride(1),
                                 output.stride(0), output.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES=NUM_STAGES)

    a_ref = f8_to_f16(a.view(torch.float8_e5m2), dtype_src_str).to(torch.float32)
    ref_out = torch.matmul(a_ref * a_scale_ref, b_ref * b_scale_ref)

    torch.testing.assert_close(ref_out, output, atol=1e-3, rtol=1e-3)
