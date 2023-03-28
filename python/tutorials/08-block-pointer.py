"""
Block Pointer
=====================
This tutorial will guide you through writing a matrix multiplication algorithm that utilizes block pointer semantics.
These semantics are more friendly for Triton to optimize and can result in better performance on specific hardware.

"""

# %%
# Motivations
# -----------
# In the previous matrix multiplication tutorial, we constructed blocks of values by de-referencing blocks of pointers,
# i.e., `load(block<pointer_type<element_type>>) -> block<element_type>`, which involved loading blocks of elements from
# memory. This approach allowed for flexibility in using hardware-managed cache and implementing complex data
# structures, such as tensors of trees or unstructured look-up tables.
#
# However, the drawback of this approach is that it relies heavily on complex optimization passes by the compiler to
# optimize memory access patterns. This can result in brittle code that may suffer from performance degradation when the
# optimizer fails to perform adequately. Additionally, as memory controllers specialize to accommodate dense spatial
# data structures commonly used in machine learning workloads, this problem is likely to worsen.
#
# To address this issue, we will use block pointers `pointer_type<block<element_type>>` and load them into
# `block<element_type>`, in which way gives better friendliness for the compiler to optimize memory access patterns.
#
# Let's start with the previous matrix multiplication example and demonstrate how to rewrite it to utilize block pointer
# semantics.

# %%
# Make a Block Pointer
# --------------------
#
#

# %%
# Final Result
# ------------

import torch

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
        # by to get the element one row down (A has M rows)
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse
    # See above `L2 Cache Optimizations` section for details
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create block pointers for the first blocks of A and B
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # See above `Make a Block Pointer` section for details
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                    order=(1, 0))

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # Load with boundary checks, no need to calculate the mask manually
        # For better performance, you may remove some axis from the boundary
        # check, if you can guarantee that the access is always in-bound in
        # that axis.
        # See above `Load/Store a Block Pointer` section for details
        a = tl.load(a_block_ptr, boundary_check=(1, ))
        b = tl.load(b_block_ptr, boundary_check=(0, ))
        # We accumulate along the K dimension
        accumulator += tl.dot(a, b)
        # Advance the block pointer to the next K block
        # See above `Advance a Block Pointer` section for details
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
    # you can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION:
        accumulator = ACTIVATION(accumulator)
    c = accumulator.to(tl.float16)

    # ----------------------------------------------------------------
    # Write back the block of the output matrix C with boundary checks
    # See above `Load/Store a Block Pointer` section for details
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# We can now create a convenience wrapper function that only takes two input tensors
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel
def matmul(a, b, activation=None):
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    assert (
            K % 32 == 0
    ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation,
    )
    return c


# %%
# Unit Test
# ---------
#
# Still we can test our matrix multiplication with block pointers against a native torch implementation (i.e., cuBLAS).

torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
triton_output = matmul(a, b, activation=None)
torch_output = torch.matmul(a, b)
print(f"triton_output={triton_output}")
print(f"torch_output={torch_output}")
torch.testing.assert_allclose(triton_output, torch_output)
