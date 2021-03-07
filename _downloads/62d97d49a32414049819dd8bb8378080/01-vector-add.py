"""
Vector Addition
=================
In this tutorial, you will write a simple vector addition using Triton and learn about:

- The basic syntax of the Triton programming language
- The best practices for creating PyTorch custom operators using the :code:`triton.kernel` Python API
- The best practices for validating and benchmarking custom ops against native reference implementations
"""

# %%
# Compute Kernel
# --------------------------
#
# Each compute kernel is declared using the :code:`__global__` attribute, and executed many times in parallel
# on different chunks of data (See the `Single Program, Multiple Data <(https://en.wikipedia.org/wiki/SPMD>`_)
# programming model for more details).
#
#  .. code-block:: C
#
#    __global__ void add(float* z, float* x, float* y, int N){
#        // The `get_program_id(i)` returns the i-th coordinate
#        // of the program in the overaching SPMD context
#        // (a.k.a launch grid). This is what allows us to process
#        // different chunks of data in parallel.
#        // For those similar with CUDA, `get_program_id({0,1,2})`
#        // is similar to blockIdx.{x,y,z}
#        int pid = get_program_id(0);
#        // In Triton, arrays are first-class citizen. In other words,
#        // they are primitives data-types and are -- contrary to C and
#        // CUDA -- not implemented as pointers to contiguous chunks of
#        // memory.
#        // In the few lines below, we create an array of `BLOCK` pointers
#        // whose memory values are, e.g.:
#        // [z + pid*BLOCK + 0, z + pid*BLOCK + 1, ..., z + pid*BLOCK + BLOCK - 1]
#        // Note: here BLOCK is expected to be a pre-processor macro defined at compile-time
#        int offset[BLOCK] = pid * BLOCK + 0 ... BLOCK;
#        float* pz [BLOCK] = z + offset;
#        float* px [BLOCK] = x + offset;
#        float* py [BLOCK] = y + offset;
#        // Simple element-wise control-flow for load/store operations can
#        // be achieved using the the ternary operator `cond ? val_true : val_false`
#        // or the conditional dereferencing operator `*?(cond)ptr
#        // Here, we make sure that we do not access memory out-of-bounds when we
#        // write-back `z`
#        bool check[BLOCK] = offset < N;
#        *?(check)pz = *?(check)px + *?(check)py;
#    }
#
# The existence of arrays as a primitive data-type for Triton comes with a number of advantages that are highlighted in the `MAPL'2019 Triton paper <http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf>`_.

# %%
# Torch bindings
# --------------------------
# The only thing that matters when it comes to Triton and Torch is the :code:`triton.kernel` class. This allows you to transform the above C-like function into a callable python object that can be used to modify :code:`torch.tensor` objects. To create a :code:`triton.kernel`, you only need three things:
#
# - :code:`source: string`: the source-code of the kernel you want to create
# - :code:`device: torch.device`: the device you want to compile this code for
# - :code:`defines: dict`: the set of macros that you want the pre-processor to `#define` for you

import torch
import triton

# source-code for Triton compute kernel
# here we just copy-paste the above code without the extensive comments.
# you may prefer to store it in a .c file and load it from there instead.
_src = """
__global__ void add(float* z, float* x, float* y, int N){
    // program id
    int pid = get_program_id(0);
    // create arrays of pointers
    int offset[BLOCK] = pid * BLOCK + 0 ... BLOCK;
    float* pz[BLOCK] = z + offset;
    float* px[BLOCK] = x + offset;
    float* py[BLOCK] = y + offset;
    // bounds checking
    bool check[BLOCK] = offset < N;
    // write-back
    *?(check)pz = *?(check)px + *?(check)py;
}
    """


# This function returns a callable `triton.kernel` object created from the above source code.
# For portability, we maintain a cache of kernels for different `torch.device`
# We compile the kernel with -DBLOCK=1024
def make_add_kernel(device):
    cache = make_add_kernel.cache
    if device not in cache:
        defines = {'BLOCK': 1024}
        cache[device] = triton.kernel(_src, device=device, defines=defines)
    return cache[device]


make_add_kernel.cache = dict()


# This is a standard torch custom autograd Function;
# The only difference is that we can now use the above kernel in the `forward` and `backward` functions.`
class _add(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        # constraints of the op
        assert x.dtype == torch.float32
        # *allocate output*
        z = torch.empty_like(x)
        # *create launch grid*:
        # this is a function which takes compilation parameters `opt`
        # as input and returns a tuple of int (i.e., launch grid) for the kernel.
        # triton.cdiv is a shortcut for ceil division:
        # triton.cdiv(a, b) = (a + b - 1) // b
        N = z.shape[0]
        grid = lambda opt: (triton.cdiv(N, opt.BLOCK), )
        # *launch kernel*:
        # pointer to the data of torch tensors can be retrieved with
        # the `.data_ptr()` method
        kernel = make_add_kernel(z.device)
        kernel(z.data_ptr(), x.data_ptr(), y.data_ptr(), N, grid=grid)
        return z


# Just like we standard PyTorch ops We use the :code:`.apply` method to create a callable object for our function
add = _add.apply

# %%
# We can now use the above function to compute the sum of two `torch.tensor` objects:

# %%
# Unit Test
# --------------------------
#
# Of course, the first thing that we should check is that whether kernel is correct. This is pretty easy to test, as shown below:

torch.manual_seed(0)
x = torch.rand(98432, device='cuda')
y = torch.rand(98432, device='cuda')
za = x + y
zb = add(x, y)
print(za)
print(zb)
print(f'The maximum difference between torch and triton is ' f'{torch.max(torch.abs(za - zb))}')

# %%
# Seems like we're good to go!

# %%
# Benchmarking
# --------------------------
# We can now benchmark our custom op for vectors of increasing sizes to get a sense of how it does relative to PyTorch.

import matplotlib.pyplot as plt

# There are three tensors of 4N bytes each. So the bandwidth of a given kernel
# is 12N / time_ms * 1e-6 GB/s
gbps = lambda N, ms: 12 * N / ms * 1e-6
# We want to benchmark small and large vector alike
sizes = [2**i for i in range(12, 25, 1)]
triton_bw = []
torch_bw = []
for N in sizes:
    x = torch.rand(N, device='cuda', dtype=torch.float32)
    y = torch.rand(N, device='cuda', dtype=torch.float32)
    # Triton provide a do_bench utility function that can be used to benchmark
    # arbitrary workloads. It supports a `warmup` parameter that is used to stabilize
    # GPU clock speeds as well as a `rep` parameter that controls the number of times
    # the benchmark is repeated. Importantly, we set `clear_l2 = True` to make sure
    # that the L2 cache does not contain any element of x before each kernel call when
    # N is small.
    do_bench = lambda fn: gbps(N, triton.testing.do_bench(fn, warmup=10, rep=100, clear_l2=True))
    triton_bw += [do_bench(lambda: add(x, y))]
    torch_bw += [do_bench(lambda: x + y)]
# We plot the results as a semi-log
plt.semilogx(sizes, triton_bw, label='Triton')
plt.semilogx(sizes, torch_bw, label='Torch')
plt.legend()
plt.show()

# %%
# Seems like our simple element-wise operation operates at peak bandwidth. While this is a fairly low bar for a custom GPU programming language, this is a good start before we move to more advanced operations.