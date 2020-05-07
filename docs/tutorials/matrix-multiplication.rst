*********************
Matrix Multiplication
*********************

The purpose of this section is to present a Triton-C implementation of matrix multiplication that achieves performance competitive with the best existing hand-written CUDA kernels (see `CUTLASS <https://github.com/NVIDIA/cutlass>`_). We will also see how pre-processors macros can be leveraged to fuse transposition operations as well as to provide support for auto-tuning and FP16 Tensor Cores.

*Note: Bounds-checking is ommitted throughout for the sake of clarity. This feature can be easily added into our kernel, but may result in a slight performance hit because LLVM and PTXAS have issues dealing with conditionals and predicates inside loops.*

==============
Compute Kernel
==============

Matrix multiplications of the form `C = A x B` can be implemented in Triton-C fairly concisely, as shown below:

.. code-block:: C

    // Triton-C
    // launched on a grid of (M / TM) x (N / TN) programs
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

Here, each kernel instance produces a :code:`TM x TN` tile of the output matrix C as follows:

- Statements (1) - (2) fetch the id of the current program instance.
- Statements (3) - (4) construct ranges of indices to process for the vertical and horizontal axes of the output matrix :code:`C`
- Statement (5) constructs a range of indices along the reduction axis: :code:`rk = [0, 1, ..., TK - 1]`
- Statement (6) initialize a :code:`TM x TN` array of accumulators to hold the result of :code:`A[rm, :] x B[:, rn]`
- Statements (7) - (8) initializes arrays of pointers :code:`pa` and :code:`pb` to the operands :code:`A` and :code:`B` using logic similar to that of the above transposition kernel
- Statements (9) - (10) load tiles of operands by dereferencing :code:`pa` and :code:`pb`
- Statement (11) performs updates the accumulator array using Triton-C's matrix multiplication operator :code:'@'
- Statements (12) - (13) updates :code:`pa` and :code:`pb`
- Statement (14) creates an array of pointers `pc` to the result matrix :code:`C`
- Statement (15) writes back the accumulator to :code:`C`

Internally, the Triton compiler will perform quite a few optimizations that will ensure good performance for this kernel:

- Automatic coalescing of load/store operations
- Automatic vectorization of load/store operations
- Stashing `a` and `b` to shared memory
- Automatic allocation of shared memory
- Automatic synchronization of shared memory
- Automatic padding of shared memory to avoid bank conflicts
- Automatic usage of tensor cores when TYPE = half and TK % 4 = 0


==============
Optimizations
==============

Nonetheless, there are two important optimizations that the Triton compiler does not do automatically at the moment yet are critical to achieve peak performance: pre-fetching and rematerialization. In this subsection we describe how these optimizations can be done manually by  modifying the above source-code.

-------------
Pre-Fetching
-------------

The purpose of pre-fetching is to overlap the update of the accumulator `c` with the memory loads for the next tiles that will need to be multiplied. This can be done by modifying the above reduction loop as follows:


.. code-block:: C

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


Note that the Triton-C compiler will now also be able to use double-buffering techniques to make sure that the array `a` can be used and updated at the same time without any memory hazard.

-----------------
Rematerialization
-----------------

`Rematerialization <https://en.wikipedia.org/wiki/Rematerialization>`_ is a compiler optimization which consists in recomputing some values instead of storing and reloading them from (register) memory, so as to decrease register pressure in the compute kernel. Although LLVM does this automatically to some extent, it fails to find good heuristics for the above kernel -- thereby requiring some source code modification to achieve optimal performance. Fortunately, only :code:`rm` and :code:`rn` need to be rematerialized, leading to the  following epilogue:

.. code-block:: C

    // epilogue
    int rcm[TM] = pm * TM + 0 ... TM;
    int rcn[TN] = pn * TN + 0 ... TN;
    TYPE* pc[TM, TN] = C + rcn[newaxis, :] + rcm[:, newaxis] * ldc;
    *pc = c;


------------------------------------
Fused Transpositions and Auto-Tuning
------------------------------------

It is common for optimized matrix-multiplication implementations (e.g., BLAS) to provide variants in which one or both operands are transposed. Fortunately, this can be done by using pre-processors macros for tile shapes and broadcasting directives, leading to the following kernel:

.. code-block:: C

    // Triton-C
    // launched on a grid of (M / TM) x (N / TN) programs
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


All matrix multiplications variants can then be retrieved using the following compilation option:

.. code-block:: C

    // A is not transposed
    -DUSE_A=a -DSTRIDE_AK=1-DSTRIDE_AM=lda
    -DBROADCAST_AK=newaxis,: -DBROADCAST_AN=:,newaxis -DSHAPE_A=TM,TK
    // A is transposed
    -DUSE_A=^a -DSTRIDE_AK=lda-DSTRIDE_AM=1
    -DBROADCAST_AK=:,newaxis -DBROADCAST_AN=newaxis,: -DSHAPE_A=TK,TM
    // B is not transpose
    -DUSE_B=b -DSTRIDE_BK=ldb-DSTRIDE_BN=1
    -DBROADCAST_BK=:,newaxis -DBROADCAST_BN=newaxis,: -DSHAPE_B=TK,TN
    // B is transpose
    -DUSE_B=^b -DSTRIDE_BK=1-DSTRIDE_BN=ldb
    -DBROADCAST_BK=newaxis,: -DBROADCAST_BN=:,newaxis -DSHAPE_B=TN,TK


Auto-tuning can also be handled using pre-processor macros:

.. code-block:: C

    // Auto-tuning TM and TN in {32, 64, 128}; TK in {8, 16}
    -DTM=[32, 64, 128] -DTN=[32, 64, 128] -DTK=[8, 16]

A runnable version of this kernel is available `here <https://github.com/ptillet/triton/tree/master/python/examples/tutorials/mat_mul.py>`_.
