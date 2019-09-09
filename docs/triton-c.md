# The Triton-C Programming Language

## <span style="color:darkred"> Table of Contents </span>
1. [Motivations](#motivations)
    1. [Issues  of C/C++ for Linear Algebra](#issues-c-c++)
    2. [Design Philosophy of Triton-C](#design-philosophy)
2. [Vector Addition](#vector-addition)
    1. [Differences with CUDA](#differences-with-cuda)
    2. [Advantages over CUDA](#advantages-over-cuda)
    	1. [Vectorization](#vectorization)
    	2. [Parameterization](#parameterization)
    	3. [Auto-Tuning](#auto-tuning)
3. [Matrix Transposition](#matrix-transposition)
4. [Matrix Multiplication](#matrix-multiplication)


## <span style="color:darkred"> Motivations </span> <a name="motivations"></a>

## <span style="color:darkblue"> Issues of C/C++ for Linear Algebra </span> <a name="issues-c-c++"></a>

In C and C++, arrays and pointers have similar semantics. Indeed, there is no way to manipulate statically shaped multi-dimensional arrays (beyond initialization) as a whole without resorting to third-party libraries:

```c
float x[16][8] = {3.14};
float y[16][8] = {5.17};
// z = x + y
float z[16][8];
for(int i = 0; i < 16; i ++)
  for(int j = 0; j < 8; j++)
    z[i][j] = x[i][j] + y[i][j];
```

This issue can be somewhat mitigated using templates metaprogramming in C++:

```c
matrix<float, 16, 8> x = {3.14};
matrix<float, 16, 8> y = {5.17};
matrix<float, 16, 8> z = x + y;
```

This is better, but there are still some important issues with this approach:

- The syntax could be better, especially when it comes to broadcasting and reshaping.

- Data-flow information for array operations does not propagate beyond the program's AST, thereby making it difficult for compilers to optimize moderately complicated array programs (i.e., Matrix-Multiplication). This can be worked around using heavy metaprogramming techniques (see [CUTLASS](https://github.com/NVIDIA/cutlass)), but even then programmers still have to allocate and synchronize shared memory manually and endure prohibitively long compilation procedures not easily amenable to auto-tuning.

For these reasons, most Deep-Learning frameworks still rely heavily on highly optimized subroutines (e.g., BLAS), which makes the development of novel custom primitives time-consuming for experts and challenging for others. This is where Triton comes into play.

## <span style="color:darkblue"> Design Philosophy of Triton-C </span> <a name="design-philosophy"></a>

The purpose of Triton is to bring native support for efficient numerical multi-dimensional array operations into a standard procedural languages. We achieve this through:

* **Triton-C**: Syntactic and semantical extensions to the C language. In particular, native support for reshaping, broadcasting, matrix-multiplication, transposition, etc. This is the object of this tutorial.

* **Triton-IR**: An LLVM-like IR for array operations, as well as various (automatic memory coalescing, automatic vectorization, shared memory allocation/synchronization, tensor core instruction selection, etc.). Although our system generates Triton-IR programs from Triton-C source-code, this is beyond the scope of this tutorial. More information can be found [here](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf).

Anyway, the Triton-C code corresponding to the above matrix addition operation can be written and extended as follows:
```c
float x[16, 8] = 3.14;
float y[16, 8] = 5.17;
// float z[8, 8] = x + y; // doesn't compile -- incompatible shapes!
float z[16, 8] = x + y;
float u[16] = z[:, +]; // sum along the second axis
float v[16, 32] = u[:, newaxis]; // broadcasting along the second axis
```

Of course, we can do much more than additions, reduction and broadcasting. The purpose of this tutorial is to walk you through all the features of Triton-C, and eventually show you how it can be used to build auto-tuned matrix-multiplication kernels on par with state-of-the-art CUDA-C implementation in less than an afternoon.

_Note: You might be thinking that this is exactly what [MLIR](https://github.com/tensorflow/mlir) was made for... and you're right! You can think of Triton-IR as a dialect for MLIR, and Triton-C as a frontend for it. If you're interested in making this a thing, let me know._

## <span style="color:darkred">  Vector Addition </span> <a name="vector-addition"></a>

### <span style="color:darkblue">  Differences with CUDA </span> <a name="differences-with-cuda"></a>

Let's start it off by looking at a simple example. Vector addition, in its most trivial Triton-C implementation, can be written as follows:

```c
// launched on a grid of (N / 32) programs of 1 thread each
__global__  void add(int N, float *a, float *b, float* c) {
	int id = get_program_id(0);
	int off[32] = id * 32 + (0 ... 32)
	*(c + off) = *(a + off) + *(b + off)
}
```
For reference, here is an equivalent CUDA kernel (nvcc will generate the same PTX code as triton-jit on the above code):

```c
// launched on a grid of (N / 32) programs of 32 threads each
__global__ void add(int N, float *a, float *b, float *c) {
    int off = blockIdx.x * 32 + threadIdx.x;
    c[off] = a[off] + b[off];
}
```

As you can see, there are three main differences between our Triton-C kernel and the equivalent CUDA-C:

- **The programming model is different**. 
While Triton-C and CUDA-C both use a Single-Program, Multiple-Data (SPMD) programming model, each Triton-C kernel is single-threaded. 
	Therefore, `get_program_id({0, 1, 2})` is equivalent to `blockIdx.{x, y, z}`, but there is no such thing as `blockDim` and `threadIdx`.
    
- **The semantics of arrays is different** 
In the above Triton-C kernel, `off` is an array of 32 consecutive integers:  `int off[32] = {id * 32 + 0, id * 32 + 1, ..., id * 32 + 31}`. 
	As a result, the statement: `c + off` implicitly broadcast `c` and creates an array of 32 pointers. This could also be done explicitly as follows:
```
float* c_broadcast[32] = c;
float* c_ptr[32] = c_broadcast + off; // c_ptr = c + off
```

- **The semantics of the subscript operator is different**. 
n C/CUDA-C, subscripting can be used to offset and dereference a pointer, but in Triton-C it can only be used to index and broadcast an array (think NumPy).

### <span style="color:darkblue"> Advantages over CUDA </span> <a name="advantages-over-cuda"></a>

At this point, the advantages of Triton-C over CUDA may not be obvious. But they should become clearer and clearer as this tutorial progresses. First and foremost, the purpose of this subsection is to show how Triton can be used to optimize vector additions by automatically taking care of load/store vectorization, code parameterization and auto-tuning -- all of which require nontrivial implementation efforts in CUDA.

#### <span style="color:purple"> Vectorization </span> <a name="vectorization"></a>

On some hardware architectures, vectorizing load/store operations can lead to better memory utilization and, in turn, noticeable performance gains. In general, 128-bit memory transactions are favored, leading to the following CUDA kernel:
```c
// launched on a grid of (N / 128) programs of 32 threads each
__global__ void add(int N, float4 *a, float4 *b, float4 *c) {
    int off = blockIdx.x * 32 + threadIdx.x;
    c[off] = a[off] + b[off];
}
```
Or, for half-precision inputs:
```c
// launched on a grid of (N / 256) programs of 32 threads each
__global__ void add(int N, half8 *a, half8 *b, half8 *c) {
    int off = blockIdx.x * 32 + threadIdx.x;
    c[off] = a[off] + b[off];
}
```

Now this is a bit annoying, because as a programmer you have to keep track of not only the ideal vector size for each data-type (which might change in future GPU architectures), but also of how many elements are processed in each thread-block -- and adjust the grid size of the kernel accordingly! Not to mention that you may want to tune the thread-block size as well.

In Triton-C, this is not a problem as the compiler will figure out automatically when and where vectorization should be used, without any change in the source-code necessary.

#### <span style="color:purple"> Parameterization </span> <a name="parameterization"></a>

Specifically, the Triton compiler would refuse to 4-way vectorize our above compute kernel because it would require the array `int off[32]` to be distributed over 8 threads, which is less than a warp. Fortunately, it turns out that this problem can be easily solved using preprocessor directrives to _parameterize_ our kernel:
```c
// launched on a grid of (N / SIZE) programs of 1 thread each
__global__  void add(int N, TYPE* a, TYPE* b, TYPE* c) {
	int id = get_program_id(0);
	int off[SIZE] = id * SIZE + (0 ... SIZE)
	*(c + off) = *(a + off) + *(b + off)
}
// Not vectorized when compiled with -DSIZE=32 -DTYPE=float
// 4-Vectorized when compiled with -DSIZE=128 -DTYPE=float
// 8-Vectorized when compiled with -DSIZE=256 -DTYPE=half
```
Now, `TYPE` and `SIZE` are preprocessors macros which can be specified at compile-time, thereby giving the Triton compiler enough information to vectorize when beneficial without requiring any additional code modification.


#### <span style="color:purple"> Auto-Tuning </span> <a name="auto-tuning"></a>

As it turns out, different input vector lengths `N`  may require different values of `SIZE` to perform optimally. Fortunately, the Triton preprocessor also accepts lists of possible definitions for macros, in which case an auto-tuning procedure will be launched every-time new input sizes are encountered. For example, compiling the above kernel with the option`-DSIZE=[32, 64, 128, 256] -DTYPE=float`
will result in the parameter `SIZE` being automatically tuned every time a new value of `N` is encountered.

_Note: Tuning our reference CUDA kernel would be much more cumbersome, as template metaprogramming would have to be used to ensure that proper vector types would be used_


## <span style="color:darkred">  Matrix Transposition </span> <a name="matrix-transposition"></a>


## <span style="color:darkred">  Matrix Multiplication </span> <a name="matrix-transposition"></a>

## <span style="color:darkred">  Next Steps</span> <a name="matrix-transposition"></a>
