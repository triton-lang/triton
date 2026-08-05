import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import supports_clc
from triton.tools.tensor_descriptor import TensorDescriptor

requires_clc = pytest.mark.skipif(not supports_clc(), reason="CLC requires NVIDIA SM100+")


@triton.jit
def _clc_count_kernel(counts, seen_x, seen_y, seen_z, GRID_X: tl.constexpr, GRID_Y: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    linear_pid = (pid_z * GRID_Y + pid_y) * GRID_X + pid_x
    tl.atomic_add(counts + linear_pid, 1)
    tl.store(seen_x + linear_pid, pid_x)
    tl.store(seen_y + linear_pid, pid_y)
    tl.store(seen_z + linear_pid, pid_z)


def _run_count_grid(grid, num_ctas, device):
    grid_x = grid[0]
    grid_y = grid[1] if len(grid) > 1 else 1
    grid_z = grid[2] if len(grid) > 2 else 1
    total = grid_x * grid_y * grid_z
    counts = torch.zeros(total, dtype=torch.int32, device=device)
    seen_x = torch.full((total, ), -1, dtype=torch.int32, device=device)
    seen_y = torch.full((total, ), -1, dtype=torch.int32, device=device)
    seen_z = torch.full((total, ), -1, dtype=torch.int32, device=device)
    compiled = _clc_count_kernel[grid](
        counts,
        seen_x,
        seen_y,
        seen_z,
        GRID_X=grid_x,
        GRID_Y=grid_y,
        num_warps=4,
        num_ctas=num_ctas,
        clc=True,
    )
    return compiled, counts, seen_x, seen_y, seen_z


@requires_clc
@pytest.mark.parametrize("num_ctas", [1])
def test_clc_no_pending_work(num_ctas, device):
    _, counts, seen_x, seen_y, seen_z = _run_count_grid((1, ), num_ctas, device)
    assert counts.item() == 1
    assert seen_x.item() == seen_y.item() == seen_z.item() == 0


@requires_clc
@pytest.mark.parametrize("num_ctas", [1])
@pytest.mark.parametrize("grid_rank", [1, 3])
def test_clc_exactly_once_program_id_multicta(num_ctas, grid_rank, device):
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    # Larger than the architectural resident-CTA limit, so pending launches
    # remain available for cancellation.
    grid_x = 64 * max(1, num_sms // num_ctas)
    grid = (grid_x, ) if grid_rank == 1 else (grid_x, 2, 2)
    compiled, counts, seen_x, seen_y, seen_z = _run_count_grid(grid, num_ctas, device)

    assert "clusterlaunchcontrol.try_cancel" in compiled.asm["ptx"]
    assert torch.all(counts == 1)
    grid_y = grid[1] if grid_rank == 3 else 1
    linear = torch.arange(counts.numel(), dtype=torch.int32, device=device)
    expected_x = linear % grid_x
    expected_y = (linear // grid_x) % grid_y
    expected_z = linear // (grid_x * grid_y)
    assert torch.equal(seen_x, expected_x)
    assert torch.equal(seen_y, expected_y)
    assert torch.equal(seen_z, expected_z)


@triton.jit
def _clc_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = tile_id % num_pid_m
    pid_n = tile_id // num_pid_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in tl.range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _run_pointer_matmul(M, N, K, block_m, block_n, block_k, num_ctas, device):
    torch.manual_seed(42)
    a = torch.randn((M, K), dtype=torch.float16, device=device)
    b = torch.randn((K, N), dtype=torch.float16, device=device)
    c = torch.empty((M, N), dtype=torch.float16, device=device)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n), )
    _clc_matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
        num_ctas=num_ctas,
        clc=True,
    )
    expected = torch.matmul(a.float(), b.float()).half()
    torch.testing.assert_close(c, expected, atol=3e-2, rtol=3e-2)
    return grid


@requires_clc
@pytest.mark.parametrize("num_ctas", [1])
@pytest.mark.parametrize("M,N,K", [(512, 512, 64), (257, 383, 70)])
def test_clc_pointer_matmul_num_ctas(num_ctas, M, N, K, device):
    _run_pointer_matmul(M, N, K, 128, 128, 32, num_ctas, device)


@triton.jit
def _clc_device_descriptor_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                         BLOCK_K: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, [K, N], [N, 1], [BLOCK_K, BLOCK_N])
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [BLOCK_M, BLOCK_N])

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for offs_k in range(0, K, BLOCK_K):
        a = a_desc.load([offs_m, offs_k])
        b = b_desc.load([offs_k, offs_n])
        accumulator = tl.dot(a, b, accumulator)
    c_desc.store([offs_m, offs_n], accumulator.to(a_desc.dtype))


@triton.jit
def _clc_host_descriptor_matmul_kernel(a_desc, b_desc, c_desc):
    K = a_desc.shape[1]
    BLOCK_M: tl.constexpr = a_desc.block_shape[0]
    BLOCK_K: tl.constexpr = a_desc.block_shape[1]
    BLOCK_N: tl.constexpr = b_desc.block_shape[1]

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for offs_k in range(0, K, BLOCK_K):
        a = a_desc.load([offs_m, offs_k])
        b = b_desc.load([offs_k, offs_n])
        accumulator = tl.dot(a, b, accumulator)
    c_desc.store([offs_m, offs_n], accumulator.to(a_desc.dtype))


@requires_clc
@pytest.mark.parametrize("descriptor", ["device", "host"])
@pytest.mark.parametrize("num_ctas", [1])
def test_clc_descriptor_matmul_num_ctas(descriptor, num_ctas, device):
    M, N, K = 512, 512, 64
    block_m, block_n, block_k = 128, 128, 32
    torch.manual_seed(42)
    a = torch.randn((M, K), dtype=torch.float16, device=device)
    b = torch.randn((K, N), dtype=torch.float16, device=device)
    c = torch.empty((M, N), dtype=torch.float16, device=device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n), 1)

    if descriptor == "device":

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device=device)

        triton.set_allocator(alloc_fn)
        _clc_device_descriptor_matmul_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=8,
            num_stages=2,
            num_ctas=num_ctas,
            clc=True,
        )
    else:
        a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k])
        b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n])
        c_desc = TensorDescriptor.from_tensor(c, [block_m, block_n])
        _clc_host_descriptor_matmul_kernel[grid](
            a_desc,
            b_desc,
            c_desc,
            num_warps=8,
            num_stages=2,
            num_ctas=num_ctas,
            clc=True,
        )

    expected = torch.matmul(a.float(), b.float()).half()
    torch.testing.assert_close(c, expected, atol=1e-2, rtol=1e-2)


@triton.jit
def _clc_descriptor_copy_kernel(src_desc, dst_desc):
    pid = tl.program_id(0)
    block = src_desc.load([pid * 2, 0])
    dst_desc.store([pid * 2, 0], block)


@requires_clc
@pytest.mark.parametrize("num_ctas", [1])
def test_clc_multicta_loop_reuse_consan(num_ctas, device, fresh_knobs):
    triton.knobs.compilation.instrumentation_mode = "consan"
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    grid_size = 64 * max(1, num_sms // num_ctas)
    src = torch.randn((grid_size * 2, 16), dtype=torch.float16, device=device)
    dst = torch.empty_like(src)
    src_desc = TensorDescriptor.from_tensor(src, [2, 16])
    dst_desc = TensorDescriptor.from_tensor(dst, [2, 16])

    _clc_descriptor_copy_kernel[(grid_size, )](
        src_desc,
        dst_desc,
        num_warps=4,
        num_stages=2,
        num_ctas=num_ctas,
        clc=True,
    )

    torch.testing.assert_close(dst, src, atol=0, rtol=0)
