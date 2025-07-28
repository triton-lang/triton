import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

# Gluon is a language that gives the user control over low level detail of the GPU kernel implementation.
# It means that the user can (and has to) control the memory layout of the tensors, the thread block size, the number of threads per block, etc.

# Blocked layout defines a mapping between the elements of the tensor and the threads in the GPU block.
# If the layout is smaller than the tensor size, it will get tiled over the tensor (hence the name "blocked").
# The layout is defined by the number of elements per thread, the number of threads per warp, the number of warps per CTA, and the order of the dimensions.
# Each of these parameters needs to have a rank of the tensor that it is applied to.
# For example, if the tensor is 2D, the layout can be defined as:
# - size_per_thread: [2, 4] - each thread holds 8 elements of the tensor, distributed across 2x4 block
# - threads_per_warp: [16, 2] - each warp holds 32 threads, distributed across 16x2 block. Number of total threads per warp is defined by the hardware. 32 on H100.
# - warps_per_cta: [1, 4] - each CTA holds 4 warps, distributed across 1x4 block. Number of total warps per CTA depends on your launch configuration.
# - order: [1, 0] - defines the order of the distribution over all the other parameters. [1, 0] means the elements/threads/warps are distributed over the second dimension, then the first dimension.

# Let's start with a simple kernel that copies a tensor from source to destination, breaking down the problem among GPU blocks, each block copying the SIZE elements.
@gluon.jit
def copy_kernel1d(src, dst, block_layout: ttgl.constexpr, BLOCK_SIZE: ttgl.constexpr, SIZE: ttgl.constexpr):
    pid = ttgl.program_id(0)

    offs = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE, layout=block_layout)
    data = ttgl.load(src + offs, mask=offs < SIZE)
    ttgl.store(dst + offs, data, mask=offs < SIZE)

def test_copy_kernel1d():
    # 4 GB tensor (1G * 4 bytes)
    TENSOR_SIZE = 1024 * 1024 * 1024
    # Elements to copy per block:
    # 128 elements per thread
    # 32 threads per warp
    # 4 warps per CTA
    BLOCK_SIZE = 128 * 32 * 4 * 1
    src = torch.randn(TENSOR_SIZE, device="cuda")
    dst = torch.zeros_like(src, device="cuda")
    # Naive block layout defining how the elements are distributed among the threads in the GPU block.
    # Since the layout is smaller than the tensor size, it will get tiled over the tensor.
    # Amount of tiling is BLOCK_SIZE / LAYOUT_SIZE = 128, hence the 128 elements per thread mentioned above.
    block_layout = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])
    grid = ((TENSOR_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    fn = lambda: copy_kernel1d[grid](src, dst, block_layout, BLOCK_SIZE, TENSOR_SIZE)
    ms = triton.testing.do_bench_cudagraph(fn)
    print(f"copy_kernel1: {ms} ms")
    assert torch.equal(src, dst)

# 6.23 ms on H100
test_copy_kernel1d()

# Power of blocked layouts starts to show when we start to access the tensor in a potentially
# non-contiguous way. Let's look at the case of a 2D copy kernel. For example, when copying 
# from a global memory to registers, we want "neighboring" elements to be loaded by neighboring
# threads so that the memory access pattern is contiguous and reads can be coalesced.
# NVIDIA GPUs (and Triton and Gluon) assume row-major layout; the fastest-changing index is the right-most dimension.
# Let's look at the case of a 2D copy kernel.
@gluon.jit
def copy_kernel2d(
    src, dst, block_layout: ttgl.constexpr, 
    BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
    M: ttgl.constexpr, N: ttgl.constexpr,
    ):
    pid_m = ttgl.program_id(0)
    pid_n = ttgl.program_id(1)

    # Create slice layout for the offsets in the M and N dimensions.
    # Slicing works by "dropping" the `dim` dimension from the parent layout.
    # Such layout can be then broadcasted along the dropped dimension.
    offs_m = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(dim=1, parent=block_layout))
    offs_n = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(dim=0, parent=block_layout))
    # Note the broadcasting operators along the sliced dimensions.
    data = ttgl.load(src + offs_m[:, None] * N + offs_n[None, :], mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    ttgl.store(dst + offs_m[:, None] * N + offs_n[None, :], data, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def test_copy_kernel2d(block_layout):
    M = 1024 * 1024
    N = 1024
    src = torch.randn((M, N), device="cuda")
    dst = torch.zeros_like(src, device="cuda")
    BLOCK_M = 128
    BLOCK_N = 128
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    fn = lambda: copy_kernel2d[grid](src, dst, block_layout, BLOCK_M, BLOCK_N, M, N)
    ms = triton.testing.do_bench_cudagraph(fn)
    assert torch.equal(src, dst)
    return ms

def test_copy_kernel2d_v1():
    # This block layout is not optimal for the memory access pattern of the kernel.
    # Two elements from each row are loaded by the same thread, however the subsequent threads
    # in the warp load the elements from the next row.
    block_layout = ttgl.BlockedLayout(size_per_thread=[1, 2], threads_per_warp=[32, 1], warps_per_cta=[1, 4], order=[1, 0])
    ms = test_copy_kernel2d(block_layout)
    print(f"copy_kernel2d_v1: {ms} ms")

def test_copy_kernel2d_v2():
    # This block layout is optimal for the memory access pattern of the kernel.
    # Subsequent threads in the warp load the elements from the same row.
    block_layout = ttgl.BlockedLayout(size_per_thread=[1, 2], threads_per_warp=[1, 32], warps_per_cta=[1, 4], order=[1, 0])
    ms = test_copy_kernel2d(block_layout)
    print(f"copy_kernel2d_v2: {ms} ms")

# 28.63 ms on H100
test_copy_kernel2d_v1()
# 6.33 ms on H100
test_copy_kernel2d_v2()

# TODO: Layout conversion
# TODO: Shared memory
