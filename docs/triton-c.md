# The Triton-C Programming Language

## Table of Contents
1. [Motivations](#motivations)
2. [Vector Addition](#vector-addition)
    1. [Differences over CUDA](#differences-with-cuda)
    2. [Advantages over CUDA](#advantages-over-cuda)
    	1. [Vectorization](#vectorization)
    	2. [Parameterization](#parameterization)
    	3. [Auto-Tuning](#auto-tuning)
3. [Matrix Transposition](#matrix-transposition)
4. [Matrix Multiplication](#matrix-multiplication)


## Motivations <a name="motivations"></a>

The semantics of arrays in C/C++ is similar to that of pointers. In other way, there is no way to manipulate statically shaped multi-dimensional arrays (beyond initialization) as a whole without resorting to third-party libraries, as shown below:

```c
float x[16][8] = {3.14};
float y[16][8] = {5.17};
// z = x + y
float z[16][8];
for(int i = 0; i < 16; i ++)
  for(int j = 0; j < 8; j++)
    z[i][j] = x[i][j] + y[i][j];
```

As mentioned above, this issue can be mitigated through the use of third-party libraries:

```c
matrix<float, 16, 8> x = {3.14};
matrix<float, 16, 8> y = {5.17};
matrix<float, 16, 8> z = x + y;
```

Here, we have a simple one-liner that will tell your C++ compiler to generate the above nested loop and check that the shapes match. This is better, but there are still some important issues with this approach:

- The syntax could be better.

- The compiler will now see a bunch of nested loops. Don't get me wrong, compilers have gotten really good at optimizing these (especially using polyhedral compilation), but they're still not at the point where they can automatically distribute them between CUDA threads and achieve performance on par with expert-tuned code. 

Triton-C addresses these issues by (a) adding syntax and semantics for numerical array operations to the C language; and (b) relying on an LLVM-like IR -- Triton-IR -- which supports array operations natively.  The set of optimizations done by Triton-JIT on Triton-IR is beyond the scope of this tutorial, but you can learn more about it [there](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf).

The above code then becomes the following Triton-C:
```c
float x[16, 8] = 3.14;
float y[16, 8] = 5.17;
// float z[8, 8] = x + y // doesn't compile -- incompatible shapes!
float z[16, 8] = x + y;
```

Of course, we can do much more than additions: matrix-multiplication, transposition, numpy-style broadcasting ... all of these array operations are built into Triton-C.

_Note: You might be thinking that this is exactly what [MLIR](https://github.com/tensorflow/mlir) was made for... and you're right! You can think of Triton-IR as a dialect for MLIR, and Triton-C as a frontend for it. If you're interested in making this a thing, let me know._

## Vector Addition <a name="vector-addition"></a>

### Differences with CUDA <a name="differences-with-cuda"></a>

Let's look at a really simple example to get started. Vector addition in its most trivial Triton-C implementation is written as follows:

```c
// launched on a grid of (N / 32) programs of 1 thread each
__global__  void add(int N, float *a, float *b, float* c) {
	int id = get_program_id(0);
	int off[32] = id * 32 + (0 ... 32)
	*(c + off) = *(a + off) + *(b + off)
}
```
For reference, here is an equivalent CUDA kernel (the resulting PTX code will be identical):

```c
// launched on a grid of (N / 32) programs of 32 threads each
__global__ void add(int N, float *a, float *b, float *c) {
    int off = blockIdx.x * 32 + threadIdx.x;
    c[off] = a[off] + b[off];
}
```

There are two main differences between the Triton-C kernel and the CUDA-C kernel on this simple example:

- **The programming model is different**.
While Triton-C and CUDA-C both use a Single-Program, Multiple-Data (SPMD) programming model, each Triton-C kernel is single-threaded (and automatically parallelized). Therefore, `get_program_id({0, 1, 2})` is equivalent to `blockIdx.{x, y, z}` and there is no such thing as `blockDim` and `threadIdx`.
    
- **The semantics of arrays is different**
In the above Triton-C kernel, `off` is an array of 32 consecutive integers:  `int off[32] = {id * 32 + 0, id * 32 + 1, ..., id * 32 + 31}`.

    As a result, the statement: `c + off` implicitly broadcast `c` and creates an array of 32 pointers. This could also be done explicitly as follows:
```
float* c_broadcast[32] = c;
float* c_ptr[32] = c_broadcast + off; // c_ptr = c + off
```

- **The semantics of the subscript operator is different**. 
In C/CUDA-C, subscripting can be used to offset and dereference a pointer, but in Triton-C it can only be used to index and broadcast an array (think NumPy).

### Advantages over CUDA <a name="advantages-over-cuda"></a>

The above example does not exactly show any practical benefits for Triton, but its advantages over CUDA should become more and more obvious as this tutorial progresses. In this subsection, we show how Triton can be used to optimize vector additions by automatically taking care of load/store vectorization and auto-tuning.

#### Vectorization <a name="vectorization"></a>

On some hardware architectures, vectorizing I/O operations can lead to better memory utilization and, in turn, noticeable performance gains. In general, 128-bit memory transactions are favored, leading to the following kernel:
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

Now this is a bit annoying, because as a programmer you have to keep track of not only the ideal vector size for each data-type (which might change in future GPU architectures), but also of how many elements are processed in each thread-block -- and adjust the grid size of the kernel accordingly!

In Triton-C, this is not a problem as the compiler will figure out automatically when vectorization should or should not be used, without any change in the source-code necessary.

#### Parameterization <a name="parameterization"></a>

It turns out that the Triton compiler would refuse to vectorize our code because then our array of 32 pointers would have to be distributed over 8 threads, which is less than a warp. Fortunately, it turns out that this problem can be easily solved using preprocessor directrives:
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


#### Auto-Tuning <a name="auto-tuning"></a>

So now we have this parameter, `SIZE`, whose optimal value depends not only on the data-type that is being used but also on the size of the input vectors `N`. Fortunately, the Triton preprocessor also accepts a list of possible definitions for macros, in which case an auto-tuning procedure will be launched every-time new input sizes are encountered.

In other words, compiling the above kernel with the option`-DSIZE=[32, 64, 128, 256] -DTYPE=float`
will result in the parameter `SIZE` being automatically tuned every time a new value of `N` is encountered.

_Note: Tuning our reference CUDA kernel would be much more cumbersome, as template metaprogramming would have to be used to ensure that proper vector types would be used_
