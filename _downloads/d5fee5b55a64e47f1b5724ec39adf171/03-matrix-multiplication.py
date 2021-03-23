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
#          acc += A[m : m+MB, k : k+KB] @  B[k : k+KB, n : n+NB]
#        C[m : m+MB, n : n+NB] = acc;
#
# where each iteration of the doubly-nested for-loops corresponds to a Triton program instance.

# %%
# Compute Kernel
# ----------------
#
# The above algorithm is actually fairly straightforward to implement in Triton, as we can simply use the :code:`@` operator for block-level matrix multiplication.
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
#    &A[m : m+MB, k:k+KB] =  A + (m : m+MB)[:, newaxis]*A.stride(0) + (k : k+KB)[newaxis, :];
#    &B[k : k+KB, n:n+NB] =  B + (k : k+KB)[:, newaxis]*B.stride(0) + (n : n+NB)[newaxis, :];
#
# Which means that, at initialization (i.e., :code:`k = 0`), pointers for blocks of A and B can be initialized in Triton as:
#
#  .. code-block:: C
#    :force:
#
#    int rm[MB] = program_id_m * MB + 0 ... MB;
#    int rn[NB] = program_id_n * NB + 0 ... NB;
#    int rk[KB] = 0 ... KB;
#    TYPE *pa[MB, KB] = A + (rm[:, newaxis] * stride_a_0 + rk [newaxis, :] * 1);
#    TYPE *pb[KB, NB] = B + (rk[:, newaxis] * stride_b_0 + rn [newaxis, :] * 1);
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
# Final Result
# ~~~~~~~~~~~~~~
#
# We are now ready to put all these pieces together and write our Triton kernel for matrix multiplication.
# Note that we rematerialize :code:`rm` and :code:`rn:` after the inner loop to decrease register pressure.
# This is an optimization that provides an additional 5% performance improvement and cannot be currently done by the Triton compiler.
#
#  .. code-block:: C
#    :force:
#
#    #define MAX_GROUP_SIZE 8
#
#    __global__ void dot(TYPE* A, TYPE* B, TYPE* C,
#                       int M, int N, int K,
#                       int stride_a_0, int stride_b_0, int stride_c_0) {
#      // prologue
#      int pid = get_program_id(0);
#      int grid_m = (M + MB - 1) / MB;
#      int grid_n = (N + NB - 1) / NB;
#      // re-order program ID for better L2 performance
#      int width = MAX_GROUP_SIZE * grid_n;
#      int group_id = pid / width;
#      int group_size = min(grid_m - group_id * MAX_GROUP_SIZE, MAX_GROUP_SIZE);
#      int pid_m = group_id * MAX_GROUP_SIZE + (pid % group_size);
#      int pid_n = (pid % width) / (group_size);
#      // pointers to operands
#      // note the parentheses here; they force the offset
#      // computation to happen in typeof(stride_a_0) = int32 rather than
#      // typeof(A) = int64
#      int rm[MB] = pid_m * MB + 0 ... MB;
#      int rn[NB] = pid_n * NB + 0 ... NB;
#      int rk[KB] = 0 ... KB;
#      TYPE *pa[MB, KB] = A + (rk [newaxis, :] * 1 + rm[:, newaxis] * stride_a_0);
#      TYPE *pb[KB, NB] = B + (rk[:, newaxis] * stride_b_0 + rn [newaxis, :] * 1);
#      // reduction loop
#      float acc[MB, NB] = 0;
#      for (int k = K; k > 0; k -= KB) {
#        acc += (*pa) @ (*pb);
#        pa += KB * 1;
#        pb += KB * stride_b_0;
#      }
#      // pointers to output
#      // here we rematerialize `rm` and `rn` so that they are not live through
#      // the above reduction loop. In the future, the compiler should be able to
#      // do this automatically.
#      rm = pid_m * MB + 0 ... MB;
#      rn = pid_n * NB + 0 ... NB;
#      TYPE *pc[MB, NB] = C + (rm[:, newaxis] * stride_c_0 + rn[newaxis, :]);
#      // we write back using *?() operator. `acc` gets casted to `float32` implicitly.
#      *? (rm[:, newaxis] < M && rn [newaxis, :] < N) pc = acc;
#    }
#
# Where :code:`TYPE` is the data-type of the input matrices and :code:`MB`, :code:`NB`, :code:`KB` are the block sizes defined in the above pseudo-code.
# Good values for these block sizes are hard to find, hence we will introduce the auto-tuner in the next section of this tutorial.
# If :code:`TYPE` is :code:`half`, then tensor cores will be used automatically provided that :code:`MB`, :code:`NB` and :code:`KB` are multiples of 16.
#

# %%
# Torch Bindings
# ----------------
#
# Auto-Tuning
# ~~~~~~~~~~~~~~
#
# In order to use Triton's built-in auto-tuner in the above kernel, we need to define a list of :code:`triton.config` objects. that can be constructed as follows:

import torch
import triton

autotune_configs = [
    triton.config(defines={"MB": "128", "NB": "128", "KB": "32"}, num_warps=4),
    triton.config(defines={'MB': '64', 'NB': '128', 'KB': '32'}, num_warps=4),
    triton.config(defines={'MB': '128', 'NB': '64', 'KB': '32'}, num_warps=4),
    triton.config(defines={'MB': '64', 'NB': '64', 'KB': '64'}, num_warps=4),
    triton.config(defines={'MB': '32', 'NB': '128', 'KB': '64'}, num_warps=4),
    triton.config(defines={'MB': '128', 'NB': '32', 'KB': '64'}, num_warps=4),
    triton.config(defines={'MB': '64', 'NB': '32', 'KB': '64'}, num_warps=2),
    triton.config(defines={'MB': '32', 'NB': '64', 'KB': '64'}, num_warps=2)
]

# %%
# we also need to define a list of :code:`string` (i.e., "autotuning key") that specifies the set of argument names whose change in value will trigger the auto-tuner to kick in.
# Here, we want to re-tune our kernel only when the shape of input matrices changes.

autotune_key = ["M", "N", "K"]

# %%
# We can now create an auto-tuned kernel by passing the `autotune_configs` and `autotune_key` lists to the constructor of the :code:`triton.kernel` class.

src = """
#define MAX_GROUP_SIZE 8

__global__ void dot(TYPE* A, TYPE* B, TYPE* C, 
                   int M, int N, int K, 
                   int lda, int ldb, int ldc) {
  int pid = get_program_id(0);
  int grid_m = (M + MB - 1) / MB;
  int grid_n = (N + NB - 1) / NB;
  int width = MAX_GROUP_SIZE * grid_n;
  int group_id = pid / width;
  int group_size = min(grid_m - group_id * MAX_GROUP_SIZE, MAX_GROUP_SIZE);
  int pid_m = group_id * MAX_GROUP_SIZE + (pid % group_size);
  int pid_n = (pid % width) / (group_size);
  int rm[MB] = pid_m * MB + 0 ... MB;
  int rn[NB] = pid_n * NB + 0 ... NB;
  int rk[KB] = 0 ... KB;
  TYPE *pa[MB, KB] = A + (rk [newaxis, :] * 1 + rm[:, newaxis] * lda);
  TYPE *pb[KB, NB] = B + (rk[:, newaxis] * ldb + rn [newaxis, :] * 1);
  float acc[MB, NB] = 0;
  for (int k = K; k > 0; k -= KB) {
    acc += (*pa) @ (*pb);
    pa += KB * 1;
    pb += KB * ldb;
  }
  rm = pid_m * MB + 0 ... MB;
  rn = pid_n * NB + 0 ... NB;
  TYPE *pc[MB, NB] = C + (rm[:, newaxis] * ldc + rn[newaxis, :]);
  *? (rm[:, newaxis] < M && rn [newaxis, :] < N) pc = acc;
}
"""


def make_kernel(device, dtype):
    key = (device, dtype)
    cache = make_kernel.cache
    if key not in cache:
        defines = {'TYPE': dtype}
        cache[key] = triton.kernel(src, device=device, defines=defines, autotune_vals=autotune_configs, autotune_key=autotune_key)
    return cache[key]


make_kernel.cache = dict()

# %%
# Autograd Function
# ~~~~~~~~~~~~~~~~~~
#
# Now we are ready to expose our auto-tuned kernel as a `torch.autograd.Function`.
# To do so, we just need to define a `forward` function that takes a two tensors as input and returns a tensor as output.


class _dot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        M, Ka = a.shape
        Kb, N = b.shape
        assert Ka == Kb, "incompatible dimensions"
        assert a.is_contiguous() and b.is_contiguous(), "inputs must be contiguous"
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        kernel = make_kernel(a.device, a.dtype)
        grid = lambda opt: (triton.cdiv(M, opt.MB) * triton.cdiv(N, opt.NB), )
        kernel(a.data_ptr(), b.data_ptr(), c.data_ptr(), \
               M, N, Ka, \
               a.stride(0), b.stride(0), c.stride(0), \
               grid=grid)
        return c


dot = _dot.apply

# %%
# Unit Test
# -----------
#
# We can test our custom matrix multiplication operation against cuBLAS (i.e., :code:`torch.matmul`).
# Note that we need to modify the :code`atol` and :code:`rtol` parameters of `torch.allclose` to account for the fact that we are comparing FP16 tensors.

a = torch.rand((512, 768), device='cuda', dtype=torch.float16)
b = torch.rand((768, 896), device='cuda', dtype=torch.float16)
c_0 = dot(a, b)
c_1 = torch.matmul(a, b)
print(c_0)
print(c_1)
print(torch.allclose(c_0, c_1, rtol=1e-3, atol=1e-3))

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
#    export CUTLASS_LIBRARY_DIR=/tmp/cutlass/build/install/lib/a
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
        x_vals=[256 * i for i in range(2, 33)],  # different possible values for `x_name`
        y_name='provider',  # argument name whose value corresponds to a different line in the plot
        y_vals=['torch', 'triton', 'cutlass'],  # possible keys for `y_name`
        y_lines=["Torch", "Triton", 'CUTLASS'],  # label name for the lines
        ylabel="TFLOPS",  # label name for the y-axis
        plot_name="matmul-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={}
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: dot(a, b))
    if provider == 'cutlass':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton.testing.cutlass_matmul(a, b))
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True)

# %%
# As we can see, the performance of our kernel is pretty good. It is in fact faster than CUTLASS, and therefore probably comparable to the absolute best CUDA code an expert could write.