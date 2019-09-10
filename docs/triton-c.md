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
    1. [Compute Kernel](#trans-compute-kernel)
    2. [Conditional Dereferencing](#conditional-dereferencing)
4. [Matrix Multiplication](#matrix-multiplication)
    1. [Compute Kernel](#matmul-compute-kernel)
    2. [Optimizations](#optimizations)
    	1. [Pre-Fetching](#pre-fetching)
    	1. [Rematerialization](#rematerialization)
    3. [Fused Transpositions](#fused-trans)
  
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

_Note: You might be thinking that this is exactly what [MLIR](https://github.com/tensorflow/mlir) was made for... and you're right! You can think of Triton-IR  conceptually as a dialect for MLIR, and Triton-C as a frontend for it -- although not yet implemented as such. If you're interested in making this a thing, let me know._

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

Transpositions are (relatively) hard to efficiently write in CUDA as naive implementations typically suffer from _uncoalesced_ memory operations when writing back the transposed matrix to DRAM.  Of course, this can be fixed by using shared memory as shown [here](https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/), but this comes at the cost of simplicity and -- more importantly -- interferes with auto-tuning.

### <span style="color:darkblue"> Compute Kernel </span> <a name="trans-compute-kernel"></a>

In Triton, however, kernels are single-threaded and the compiler automatically detects if and when data should be temporarily stashed to shared memory in order to enable shared memory stores/loads. Therefore, an optimal Triton kernel for this operation would look like:

```c
// launched on a grid of (M / TM) x (N / TN) programs of 1 thread each
__global__ void transpose(TYPE * X, TYPE * Y,  int M, int N, int ldx, int ldy) {
// extract program ID
  int pidm = get_program_id(0); //(1)
  int pidn = get_program_id(1); //(2)
  // create 1D range along the two matrix's axes
  int rm[TM] = pidm * TM + 0 ... TM; //(3)
  int rn[TN] = pidn * TN + 0 ... TN; //(4)
  // create 2D array of pointers
  TYPE* px[TM, TN] = X + rm[:, newaxis] + rn[newaxis, :] * ldx; //(5)
  TYPE* py[TN, TM] = Y + rm[newaxis, :] * ldy + rn[:, newaxis]; //(6)
  // write back using the transposition operator '^'
  *py = ^(*px); //(7)
}
```

At a high level, this kernel loads a `TM x TN` tile from the input matrix `X`, transposes it and writes the resulting `TN x TM` tile to the output matrix `Y`. As a result, transposition of the full input matrix is achieved by launching a grid of `(M / TM) x (N / TN)` programs decomposed as follows:

- Statements (1) and (2) extract the coordinates the program in the above 2D launch grid. For example, the program producing the output tile `Y[TN:2TN-1, 2TN:3TN-1]` holds the values:
```
pidm = 2
pidn = 1
``` 

- Statements (3) and (4) construct the ranges of indices:
```
rm = [pidm*TM + 0, pidm*TM + 1, ..., pidm*TM + (TM - 1)]
rn = [pidn*TN + 0, pidn*TN + 1, ..., pidn*TN + (TN - 1)]
```

which will be used in statements (5) and (6) to construct tiles of pointers

- Statements (5) constructs the following array of pointers `px` using numpy-style broadcasting semantics:
```
│ X + (pidm*TM + 0)       + (pidn*TN + 0)*ldx,  ...,  ...,  X + (pidm*TM + 0)      +  (pidn*TN + TN - 1)*ldx) │
│      ⋮                                                                                       ⋮             │
│      ⋮                                                                                       ⋮             │
│ X + (pidm*TM + TM - 1)  + (pidn*TN + 0)*ldx,  ...,  ...,  X + (pidm*TM + TM - 1) +  (pidn*TN + TN - 1)*ldx) │
```
- Statement (6) constructs the following array of pointers `py` using numpy-style broadcasting semantics:
```
│ Y + (pidn*TN + 0)       + (pidm*TM + 0)*ldy,  ...,  ...,  Y + (pidn*TN + 0)      +  (pidm*TM + TM - 1)*ldy) │
│      ⋮                                                                                       ⋮             │
│      ⋮                                                                                       ⋮             │
│ Y + (pidn*TN + TN - 1)  + (pidn*TN + 0)*ldy,  ...,  ...,  Y + (pidn*TN + TN - 1) +  (pidm*TM + TM - 1)*ldy) │
```
- Statement (7) element-wise dereferences the above array of pointers `*px`, transposes it using the unary transposition operator `^`, and writes it back at the location specified by `py`.

### <span style="color:darkblue"> Conditional Dereferencing </span> <a name="conditional-dereferencing"></a>

You might have noticed that the above code will fail when `M` and `N` are not multiples of `TM` and `TN` respectively. Fortunately, the above kernel can be slightly modified to handle thie situation, as shown below:
```
// launched on a grid of ((M + TM - 1) / TM) x ((N + TN - 1) / TN) programs
__global__ void transpose(TYPE * X, TYPE * Y,  int M, int N, int ldx, int ldy) {
   // ...
   // create bounds-checking mask
   bool checkx[TM, TN] = (rm[:, newaxis] < M) && (rn[newaxis, :] < N); //(7a)
   bool checky[TN, TM] = (rm[newaxis, :] < M) && (rn[:, newaxis] < N); //(7b)
   // conditional write-back using the conditional dereferencing operatior '*?()'
   *?(checky)py = ^(*?(checkx)px); //(7)
}
```

Here, statements (7a) creates an array of booleans `checkx[TM, TN]` such that `checkx(i, j) = True` if and only if `px(i, j)` should be dereferenced. Statement (7b) does the same for `py`. Then, both `px` and `py` can be conditionally dereferenced using Triton-C's conditional dereferencing operator `*?(predicate) pointer`.


## <span style="color:darkred">  Matrix Multiplication </span> <a name="matrix-multiplication"></a>

The purpose of this section is to present a Triton-C implementation of matrix multiplication that achieves performance competitive with the best existing hand-written CUDA-C kernels (see [CUTLASS](https://github.com/NVIDIA/cutlass)). We will also see how pre-processors macros can be leveraged to fuse transposition operations as well as to provide support for auto-tuning and FP16 Tensor Cores.

_Note: Bounds-checking is ommitted for the sake of clarity. This feature can be easily added into our kernel, but may result in a slight performance hit because LLVM and PTXAS have issues dealing with conditionals and predicates inside loops._

### <span style="color:darkblue"> Compute Kernel </span> <a name="matmul-compute-kernel"></a>

Matrix multiplications of the form `C = A x B` can be implemented in Triton-C fairly concisely, as shown below:

```c
// launched on a grid of (M / TM) x (N / TN) programs of 1 thread each
__global__ void dot(TYPE * A, TYPE * B, TYPE * C,  int M, int N, int K,
        	        int lda __multipleof(8),  int ldb __multipleof(8),  int ldc __multipleof(8)) {
  // prologue
  int pm = get_program_id(0); //(1)
  int pn = get_program_id(1); //(2)
  int rm[TM] = pm * TM + 0 ... TM; //(3)
  int rn[TN] = pn * TN + 0 ... TN; //(4)
  int rk[TK] = 0 ... TK; //(5)
  // initialize accumulator 
  float c[TM, TN] = 0; //(6)
  // pointers to operands
  TYPE* pa[TM, TK] = A + rk[newaxis, :] * 1 + rm[:, newaxis] * lda; //(7)
  TYPE* pb[TK, TN] = B + rk[:, newaxis] * ldb + rn[newaxis, :] * 1; //(8)
  // reduction loop
  for(int k = K; k > 0; k-= TK){
    // fetch operands
    TYPE a[TM, TK] = *pa; //(9) 
    TYPE b[TK, TN] = *pb; //(10)
    // matrix-multiply accumulate
    c += a @ b; //(11)
    // increment pointers
    pa = pa + TK * 1; //(12)
    pb = pb + TK * ldb; //(13)
  }
  // epilogue
  TYPE* pc[TM, TN] = C + rn[newaxis, :] + rm[:, newaxis] * ldc; //(14)
  *pc = c; //(15)
}
```
Here, each kernel instance produces a `TM x TN` tile of the output matrix C as follows:

- Statements (1) - (2) fetch the id of the current program instance.
- Statements (3) - (4) construct ranges of indices to process for the vertical and horizontal axes of the output matrix `C`
- Statement (5) constructs a range of indices along the reduction axis: `rk = [0, 1, ..., TK - 1]`
- Statement (6) initialize a `TM x TN` array of accumulators to hold the result of `A[rm, :] x B[:, rn]`
- Statements (7) - (8) initializes arrays of pointers `pa` and `pb` to the operands `A` and `B` using logic similar to that of the above transposition kernel
- Statements (9) - (10) load tiles of operands by dereferencing `pa` and `pb`
- Statement (11) performs updates the accumulator array using Triton-C's matrix multiplication operator '@'
- Statements (12) - (13) updates `pa` and `pb`
- Statement (14) creates an array of pointers `pc` to the result matrix `C`
- Statement (15) writes back the accumulator to `C`

Internally, the Triton compiler will perform quite a few optimizations that will ensure good performance for this kernel:

- Automatic coalescing of load/store operations
- Automatic vectorization of load/store operations
- Stashing `a` and `b` to shared memory
- Automatic allocation of shared memory
- Automatic synchronization of shared memory
- Automatic padding of shared memory to avoid bank conflicts
- Automatic usage of tensor cores when TYPE = half and TK % 4 = 0

### <span style="color:darkblue"> Optimizations </span> <a name="optimizations"></a>

Nonetheless, there are two important optimizations that the Triton compiler does not do at the moment yet are critical to achieve peak performance: pre-fetching and rematerialization. In this subsection we describe how these optimizations can be done manually by  modifying the above source-code.

#### <span style="color:purple"> Pre-Fetching </span> <a name="pre-fetching"></a>

The purpose of pre-fetching is to overlap the update of the accumulator `c` with the memory loads for the next tiles that will need to be multiplied. This can be done by modifying the above reduction loop as follows:

```
// pre-fetch operands
TYPE a[TM, TK] = *pa; //(9) 
TYPE b[TK, TN] = *pb; //(10)
for(int k = K; k > 0; k-= TK){
   c += a @ b;
   pa = pa + TK * 1;
   pb = pb + TK * ldb;
   // don't prefetch last iteration
   bool check = k > TK;
   // pre-fetch operands
   a = check ? *pa : 0;
   b = check ? *pb : 0;
 }
```

Note that the Triton-C compiler will now also be able to use double-buffering techniques to make sure that the array `a` can be used and updated at the same time without any memory hazard.

#### <span style="color:purple"> Rematerialization </span> <a name="rematerialization"></a>

[Rematerialization](https://en.wikipedia.org/wiki/Rematerialization) is a compiler optimization which consists in recomputing some values instead of storing and reloading them from (register) memory, so as to decrease register pressure in the compute kernel. Although LLVM does this automatically to some extent, it fails to find good heuristics for the above kernel -- thereby requiring some source code modification to achieve optimal performance. Fortunately, only `rm` and `rn` need to be rematerialized, leading to the  following epilogue:

```c
// epilogue
int rcm[TM] = pm * TM + 0 ... TM;
int rcn[TN] = pn * TN + 0 ... TN;
TYPE* pc[TM, TN] = C + rcn[newaxis, :] + rcm[:, newaxis] * ldc;
*pc = c; 
```

### <span style="color:darkblue"> Fused Transpositions </span> <a name="fused-trans"></a>

It is common for optimized matrix-multiplication implementations (e.g., BLAS) to provide variants in which one or both operands are transposed. This is also what is done in the [PyTriton](https://github.com/ptillet/triton/blob/master/python/triton/ops/dot.py) implementation of matrix-multiplication. Fortunately, this can be done by using pre-processors macros for tile shapes and broadcasting directives, leading to the following kernel:

```c
void dot(TYPE * A, TYPE * B, TYPE * C,
         int M, int N, int K,
         int lda __multipleof(8),  int ldb __multipleof(8),  int ldc __multipleof(8)) {
  // prologue
  int pm = get_program_id(0);
  int pn = get_program_id(1);
  int rm[TM] = pm * TM + 0 ... TM;
  int rn[TN] = pn * TN + 0 ... TN;
  int rk[TK] = 0 ... TK;
  float c[TM, TN] = 0;
  // pointers to operands
  TYPE* pa[SHAPE_A] = A + rk[BROADCAST_AK] * STRIDE_AK + rm[BROADCAST_AM] * STRIDE_AM;
  TYPE* pb[SHAPE_B] = B + rk[BROADCAST_BK] * STRIDE_BK + rn[BROADCAST_BN] * STRIDE_BN;
  // prefetches operands
  TYPE a[SHAPE_A] = (*pa);
  TYPE b[SHAPE_B] = (*pb);
  // reduction loop
  for(int k = K; k > 0; k-= TK){
    c += USE_A @ USE_B;
    pa = pa + TK * STRIDE_AK;
    pb = pb + TK * STRIDE_BK;
    a = *pa;
    b = *pb;
  }
  // epilogue
  int rcm[TM] =  pm * TM + 0 ... TM;
  int rcn[TN] =  pn * TN + 0 ... TN;
  TYPE* pc[TM, TN] = C + rcn[newaxis, :] + rcm[:, newaxis] * ldc;
  *pc = c;
}
```

All matrix multiplications variants can then be retrieved using the following compilation option:
```c
// A is not transposed
-DUSE_A=a -DSTRIDE_AK=1-DSTRIDE_AM=lda -DBROADCAST_AK=newaxis,: -DBROADCAST_AN=:,newaxis -DSHAPE_A=TM,TK
// A is transposed
-DUSE_A=^a -DSTRIDE_AK=lda-DSTRIDE_AM=1 -DBROADCAST_AK=:,newaxis -DBROADCAST_AN=newaxis,: -DSHAPE_A=TK,TM
// B is not transpose
-DUSE_B=b -DSTRIDE_BK=ldb-DSTRIDE_BN=1 -DBROADCAST_BK=:,newaxis -DBROADCAST_BN=newaxis,: -DSHAPE_B=TK,TN
// B is transpose
-DUSE_B=^b -DSTRIDE_BK=1-DSTRIDE_BN=ldb -DBROADCAST_BK=newaxis,: -DBROADCAST_BN=:,newaxis -DSHAPE_B=TN,TK
```


