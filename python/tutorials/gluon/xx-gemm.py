"""
GEMM in Gluon
=============

In this tutorial, we will start by writing a simple GEMM (General Matrix
Multiply) kernel using Gluon. We will then incrementally optimize and develop
the kernel into a more efficient version.

This tutorial will teach you how to bootstrap a basic kernel in Gluon and how to
leverage features of Gluon's programming model, such as direct access to memory
and low-level hardware instructions, to build more efficient kernels.

Gluon, like Triton, is a tile-based programming model. A tile describes an
N-dimensional array whose elements are distributed over a set of GPU warps and
threads. In Gluon, these tiles can reside in registers or in memory.

Tiles that live in registers are referred to as "tensors". Each element of the
tensor is mapped to a GPU thread and exists inside the register file of that
thread. The "layout" of a tensor describes how the logical elements of the
tensor are mapped to virtual registers within GPU threads.

Tiles that live in memory are referred to as "memory descriptors" or "memdescs"
for short. Memdescs describe a contiguous region of memory where the elements of
a tile can be found, but the order in which the elements are stored is called
the "layout" of the memdesc. The layout describes how the logical elements of
the tile are mapped to a linear offset into the contiguous memory region.

Gluon also directly exposes asynchronous operations, such as async TMA copies,
async MMA operations, and mbarrier objects. In this way, writing Gluon is a lot
like writing CUDA C++, where the user is responsible for directly managing
memory, the layout of data in registers and in memory, how the data is accessed
and moved, and for synchronizing asynchronous operations in the program.
"""

# %%
# Declare out imports.

import triton
import triton.language as tl
import torch
import pytest

# %%
# GEMM is defined as `D = alpha * A @ B + beta * C`, where `alpha` and `beta`
# are scalars and `A`, `B`, `C`, and `D` are matrices. GEMM comprises of a
# standard matrix multiplication `A @ B` plus a scaling and addition operation
# fused into a single kernel. Let the shape of `A` be `(M, K)` and the shape of
# `B` be `(K, N)`. Thus, the shapes of `C` and `D` are `(M, N)`.
#
# Let's start by writing GEMM in Triton to understand the overall structure of
# the kernel, in particular how the work is partitioned and distributed. You
# will see that this does not change between Triton and Gluon. Both Triton and
# Gluon implement the same SPMD (Single Program Multiple Data) programming
# model: the kernel is comprised of a single "program" that is replicated across
# all the data that comprise the overall problem.
#
# A Triton or Gluon program maps directly to a GPU thread block or CTA
# (Cooperative Thread Array). A program executes entirely within a single SM
# (Streaming Multiprocessor), has access to the same shared memory, tensor
# memory, and register pool. Programs abstract over processing data in
# individual warps and threads through the tile layouts described above.
#
# We have to leverage the mathematical properties of GEMM to figure out
# how to partition the work into programs. Multiplying a matrix by a scalar and
# adding two matrices are elementwise operations: each element of the output
# matrix is a function only of the corresponding elements of the input matrices.
# In other words, they are embarrassingly parallel.
#
# Matmul on the other hand computes each output element as the dot product of a
# row of the LHS (Left-Hand Side) matrix and a column of the RHS (Right-Hand
# Side) matrix. Let's divide the output `D` into blocks of size
# `(BLOCK_M, BLOCK_N)` and say that each program computes a single block of the
# output. This yields the classic blocked matrix multiplication algorithm. We
# also fuse in scale by `alpha` and the elementwise addition with `beta * C`.
#
# Let's set up the kernel to accept the base pointers for all the matrices as
# well as the hyperparameters `BLOCK_M` and `BLOCK_N`. The NVIDIA GPU tensor
# core performs a block MMA operation, so we need to break up the rows of `A`
# and the columns of `B` into `(BLOCK_M, BLOCK_K)` and `(BLOCK_K, BLOCK_N)`
# tiles respectively.


@triton.jit
def gemm_triton_kernel(  #
        a_ptr, b_ptr, c_ptr, d_ptr,  #
        alpha, beta, M, N, K,  #
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_dm, stride_dn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # The output matrix has shape `(M, N)` divide into a grid of blocks
    # of shape (BLOCK_M, BLOCK_N). Each program computes a single block of the
    # output. We will use a 2D grid where the program ID directly corresponds to
    # the index of the output block.
    #
    # The dimensions of the launch grid ensure that we don't schedule a program
    # that has no work. However, programs along the edges of the grid may only
    # need to compute a partial block. We will use masked loads and stores to
    # guard against out-of-bounds accesses.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Translate the block indices into the start offsets of the elements. This
    # is the offset of the tiles in C and D, the starting row for A and the
    # starting column for B.
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    # Span over the rows and columns of the output block. This turns the offsets
    # to the first element into offsets over the set of rows and columns.
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    # Mask out the indices that are out-of-bounds along M and N. Setting the
    # OOB indices to zero will cause them to load garbage data, but it won't
    # matter since the outputs will be masked out later.
    offs_m = tl.where(offs_m < M, offs_m, 0)
    offs_n = tl.where(offs_n < N, offs_n, 0)

    # Tell the triton compiler that the offsets are aligned to BLOCK_M and
    # BLOCK_N respectively. This allows the compiler to coalesce the loads and
    # generate more efficient load patterns.
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    # Span over the first full tile along K for A and B.
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Initialize the accumulator to zero. For now, we can always use a float32
    # accumulator.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over the length of the row and column of blocks by `BLOCK_K`.
    for k in range(0, K, BLOCK_K):
        # Mask along the K dimension in A and B to guard against out-of-bounds
        # accesses.
        a_mask = k + offs_k[None, :] < K
        b_mask = k + offs_k[:, None] < K

        # Load the tiles of A and B for this k. Set masked values to zero.
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Perform the block MMA operation of the tiles of A and B into the
        # current accumulator.
        acc = tl.dot(a_tile, b_tile, acc)

        # Move the pointers to the next tile along K.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Obtain the pointers to the tiles of C and D.
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    d_ptrs = d_ptr + (offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn)

    # Compute the mask for C and D.
    cd_mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]

    # Load the tile of C and compute the output tile. C is always float32.
    c_tile = tl.load(c_ptrs, mask=cd_mask, other=0.0)

    # Perform the scale and addition in float32, otherwise there is too much
    # precision loss.
    d_tile = alpha * acc.to(tl.float16) + beta * c_tile

    # Write the output tile. It is implicitly downcasted from float32 to the
    # output dtype.
    tl.store(d_ptrs, d_tile, mask=cd_mask)


# %%
# Now let's define the launch function and interface.


def gemm_triton(A, B, C, alpha, beta):
    M, K = A.shape
    K, N = B.shape

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

    # Define the grid over D.
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    D = torch.empty((M, N), device=A.device, dtype=A.dtype)
    gemm_triton_kernel[grid](  #
        A, B, C, D,  #
        alpha, beta, M, N, K,  #
        *A.stride(), *B.stride(), *C.stride(), *D.stride(),  #
        BLOCK_M, BLOCK_N, BLOCK_K)
    return D


# %%
# Let's define a test and benchmark harness for matmul. We will use these across
# the different implementations we will explore in this tutorial. We are going
# to use cuBLAS as the reference implementation since PyTorch does not support
# float8 dtypes.

try:
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
except ImportError:
    cublas = None
assert cublas is not None, "this tutorial requires a CUDA device"


def cublas_gemm(a, b, c, alpha, beta):
    assert a.shape[1] == b.shape[1], "the second dimension of A and B must match"
    d = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=a.dtype)
    cublas.gemm(a, b, c, d, alpha, beta)
    return d


@pytest.mark.parametrize("M, N, K", [(64, 64, 64), (1000, 2000, 4000)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float8_e4m3fn], ids=repr)
@pytest.mark.parametrize("impl", [gemm_triton], ids=lambda f: f.__name__)
def test_impl(M, N, K, dtype, impl, device="cuda"):
    A = torch.ones(M, K, dtype=torch.float16, device=device).to(dtype)
    B = torch.ones(K, N, dtype=torch.float16, device=device).to(dtype)
    C = torch.ones(M, N, dtype=torch.float16, device=device)
    B = B.T.contiguous()
    alpha = 0.7
    beta = 0.5

    D_tri = impl(A, B.T, C, alpha, beta)
    D_ref = cublas_gemm(A, B, C, alpha, beta)
    torch.testing.assert_close(D_ref.to(torch.float16), D_tri.to(torch.float16), atol=1e-2, rtol=1e-3)
