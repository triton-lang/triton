"""
Matrix Multiplication
======================
In this tutorial, you will write a 25-lines high-performance matrix multiplication kernel that outperforms CUTLASS and falls just short of matching cuBLAS's performance.
You will specifically learn about:

- The block-level matrix multiplication operator `@`
- Multi-dimensional pointer arithmetic
- Program re-ordering for improved L2 cache hit rate 
- Automatic performance tuning
"""

# %%
# Motivations
# -------------
# Matrix multiplications are a key building block of most modern high-performance computing systems.
# They are notoriously hard to optimize, hence their implementation is typically done by hardware vendors themselves as part of so-called "kernel libraries" (e.g., cuBLAS).
# Unfortunately, these libraries are often proprietary and cannot be customized to accomodate the needs of modern deep learning workloads (e.g., mixture of experts, fused activation functions, etc.).
# For this reason, this tutorial will show you how to implement efficient matrix multiplications yourself with Triton, in a way that is easy to customize and extend.
#
# Roughly speaking, the kernel that we will write will implement the following blocked algorithm:
#
#  .. code-block:: python
#
#    # do in parallel
#    for m in range(0, M, MB):
#      # do in parallel
#      for n in range(0, N, NB):
#        acc = zeros((MB, NB), dtype=float32)
#        for k in range(0, K, KB):
#          acc += dot(A[m : m+MB, k : k+KB],  B[k : k+KB, n : n+NB])
#        C[m : m+MB, n : n+NB] = acc;
#
# where each iteration of the doubly-nested for-loop corresponds to a Triton program instance.

# %%
# Compute Kernel
# ----------------
#
# The above algorithm is actually fairly straightforward to implement in Triton.
# The main difficulty comes from the 2D pointer arithmetic that must be done to specify the memory locations of the tiles of :code:`A` and :code:`B` that we need to read in the inner loop.
#
# Pointer Arithmetics
# ~~~~~~~~~~~~~~~~~~~~
#
# For a row-major 2D tensor :code:`X`, the memory location of :code:`X[i, j]` is given by :code:`&X[i, j] = i + X.stride(0) + j`.
# Therefore, blocks of pointers for :code:`A[m : m+MB, k:k+KB]` and :code:`B[k : k+KB, n : n+NB]` can be defined in pseudo-code as:
#
#  .. code-block:: python
#
#    &A[m : m+MB, k:k+KB] =  A + (m : m+MB)[:, None]*A.stride(0) + (k : k+KB)[newaxis, :];
#    &B[k : k+KB, n:n+NB] =  B + (k : k+KB)[:, None]*B.stride(0) + (n : n+NB)[newaxis, :];
#
# Which means that, at initialization (i.e., :code:`k = 0`), pointers for blocks of A and B can be initialized in Triton as:
#
#  .. code-block:: python
#    :force:
#
#    range_m = program_id_m * MB + 0 ... MB;
#    range_n = program_id_n * NB + 0 ... NB;
#    range_k = triton.arange(0, KB);
#    // pointer for A operand
#    pa = A + (range_m[:, None] * stride_a_0 + range_k[None, :] * 1);
#    // pointer for B operand
#    pb = B + (range_k[:, None] * stride_b_0 + range_n[None, :] * 1);
#
# These pointers can then be updated in the inner loop as:
#
#  .. code-block:: C
#
#    pa += KB * 1;
#    pb += KB * ldb;
#
#
# L2 Cache Optimizations
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# As mentioned above, each program instance computes an :code:`[MB, NB]` block of :code:`C`.
# However, the order in which these blocks are computer matters, since it affects the L2 cache hit rate of our program.
# This means that a naive row-major ordering:
#
#  .. code-block:: C
#
#    int program_id = get_program_id(0);
#    int grid_m = (M + MB - 1) / MB;
#    int grid_n = (N + NB - 1) / NB;
#    int program_id_m = program_id / grid_n;
#    int program_id_n = program_id % grid_n;
#
# is unlikely to result in optimal performance.
#
# One possible solution is to launch blocks in an order that promotes data reuse.
# This can be done by 'super-grouping' blocks in groups of :code:`GROUP_SIZE` before switching to the next column:
#
#  .. code-block:: C
#
#    int program_id = get_program_id(0);
#    int width = GROUP_SIZE * grid_n;
#    int group_id = pid / width;
#    // we need to handle the case where M % (GROUP_SIZE*BM) != 0
#    int group_size = min(grid_m - group_id * GROUP_SIZE, GROUP_SIZE);
#    int pid_m = group_id * GROUP_SIZE + (pid % group_size);
#    int pid_n = (pid % width) / (group_size);
#
# In practice, this can improve the performance of our matrix multiplication kernel by >10\% on some hardware architecture (e.g., 220 to 245 TFLOPS on A100).
#

# %%
# Final Result
# -------------
#
# In order to use Triton's built-in auto-tuner in the above kernel, we need to define a list of :code:`triton.Config` objects. that can be constructed as follows:

import torch
import triton


@triton.jit()
def relu(x):
    return max(x, 0)


@triton.jit(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
def _matmul(A, B, C, M, N, K, lda, ldb, ldc, **META):
    # extract meta-parameters
    BLOCK_M = META['BLOCK_M']
    BLOCK_N = META['BLOCK_N']
    BLOCK_K = META['BLOCK_K']
    GROUP_M = META['GROUP_M']
    # matrix multiplication
    pid = triton.program_id(0)
    grid_m = (M + BLOCK_M - 1) / BLOCK_M
    grid_n = (N + BLOCK_N - 1) / BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid / width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) / (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + triton.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + triton.arange(0, BLOCK_N)
    rk = triton.arange(0, BLOCK_K)
    A = A + (rm[:, None] * lda + rk[None, :] * 1)
    B = B + (rk[:, None] * ldb + rn[None, :] * 1)
    acc = triton.zeros((BLOCK_M, BLOCK_N), dtype=triton.float32)
    for k in range(K, 0, -BLOCK_K):
        a = triton.load(A)
        b = triton.load(B)
        acc += triton.dot(a, b)
        A += BLOCK_K * 1
        B += BLOCK_K * ldb
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + triton.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + triton.arange(0, BLOCK_N)
    C = C + (rm[:, None] * ldc + rn[None, :] * 1)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    triton.store(C, acc.to(triton.float16), mask=mask)


M, N, K = 512, 512, 512
a = torch.randn((M, K), device='cuda', dtype=torch.float16)
b = torch.randn((K, N), device='cuda', dtype=torch.float16)
c = torch.empty((M, N), device='cuda', dtype=torch.float16)
grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
_matmul[grid](a, b, c, M, N, K, a.stride(0), b.stride(0), c.stride(0))

# %%
# Autograd Function
# ~~~~~~~~~~~~~~~~~~
#
# Now we are ready to expose our auto-tuned kernel as a `torch.autograd.Function`.
# To do so, we just need to define a `forward` function that takes a two tensors as input and returns a tensor as output.


class _dot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        assert a.is_contiguous() and b.is_contiguous(), "inputs must be contiguous"
        M, K, N = a.shape[0], a.shape[1], b.shape[1]
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
        _matmul[grid](a, b, c, M, N, K, a.stride(0), b.stride(0), c.stride(0), EVEN_K=(K % 64 == 0))
        return c


dot = _dot.apply

# %%
# Unit Test
# -----------
#
# We can test our custom matrix multiplication operation against cuBLAS (i.e., :code:`torch.matmul`).
# Note that we need to modify the :code`atol` and :code:`rtol` parameters of `torch.allclose` to account for the fact that we are comparing FP16 tensors.

#torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
c_0 = dot(a, b)
c_1 = torch.matmul(a, b)
print(c_0)
print(c_1)
print(triton.testing.allclose(c_0, c_1))

# %%
# Benchmark
# --------------
#
# Installing The CUTLASS Bindings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The cuBLAS library (used by :code:`torch.matmul`) uses handwritten assembly-level optimizations that cannot be replicated using publicly available tools.
# For this reason, we will instead compare the performance of our kernel against `CUTLASS <https://github.com/NVIDIA/cutlass/>`_ , a highly optimized CUDA library for matrix multiplication written by NVIDIA themselves._
# To install CUTLASS, you need a recent version of cmake:
#
#  .. code-block:: bash
#
#    cd /path/to/cutlass/
#    git clone https://github.com/NVIDIA/cutlass.git
#    cd cutlass
#    mkdir build
#    cd build
#    wget https://github.com/Kitware/CMake/releases/download/v3.19.4/cmake-3.19.4-Linux-x86_64.tar.gz
#    tar xzvf *.tar.gz
#
# You can then install CUTLASS as follows for V100
#
#  .. code-block:: bash
#
#    ./cmake-3.19.4-Linux-x86_64/bin/cmake ../ -DCUTLASS_NVCC_ARCHS_ENABLED=70 -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_f16_s884gemm_f16_*_align8
#    make -j8 install
#
# Or as follows for A100:
#
#  .. code-block:: bash
#
#    ./cmake-3.19.4-Linux-x86_64/bin/cmake ../ -DCUTLASS_NVCC_ARCHS_ENABLED=80 -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_f16_s16816gemm_*align8
#    make -j8 install
#
# Where you can change CUTLASS_LIBRARY_KERNELS as you desire. Here, we are only interested in FP16 tensor core performance.
# Triton comes with some basic Python bindings for benchmarking CUTLASS. These will be compiled when the environment variables :code:`CUTLASS_INCLUDE_DIR` and :code:`CUTLASS_LIBRARY_DIR` are set during the installation process.
# To re-install Triton with the updated CUTLASS bindings, run the following command:
#
# .. code-block:: bash
#
#    export CUTLASS_INCLUDE_DIR=/tmp/cutlass/build/install/include/
#    export CUTLASS_LIBRARY_DIR=/tmp/cutlass/build/install/lib/
#    pip uninstall -y triton
#    pip install -e "git+https://github.com/ptillet/triton.git#egg=triton&subdirectory=python"
#
# Which we can test as follows:

import triton
c_2 = triton.testing.cutlass_matmul(a, b)
print(c_2)
print(torch.allclose(c_0, c_2, rtol=1e-3, atol=1e-3))

# %%
# Note that this wrapper for CUTLASS was written for benchmarking purposes and is probably not production-ready.
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can now compare the performance of our kernel against CUTLASS. Here we focus on square matrices, but feel free to arrange the script as you wish to compare any other matrix shape.#


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # argument names to use as an x-axis for the plot
        x_vals=[8192],  # different possible values for `x_name`
        y_name='provider',  # argument name whose value corresponds to a different line in the plot
        y_vals=['cublas', 'triton'],  # possible keys for `y_name`
        y_lines=["cuBLAS", "Triton"],  # label name for the lines
        ylabel="TFLOPS",  # label name for the y-axis
        plot_name="matmul-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={}
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: dot(a, b))
    if provider == 'cutlass':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton.testing.cutlass_matmul(a, b))
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(print_data=True)

# %%
# As we can see, the performance of our kernel is pretty good. It is in fact faster than CUTLASS, and therefore probably comparable to the absolute best CUDA code an expert could write.