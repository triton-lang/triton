*********************
Matrix Transpositions
*********************


Transpositions are (relatively) hard to efficiently write in CUDA because naive implementations typically suffer from *uncoalesced* memory operations when writing back the transposed matrix to DRAM.

Of course, this can be fixed by using shared memory as shown `here <https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc>`_, but this comes at the cost of simplicity interferes with auto-tuning.

==============
Compute Kernel
==============

In Triton, however, kernels are single-threaded and the compiler automatically detects if and when data should be temporarily stashed to shared memory. Therefore, an optimal Triton kernel for this operation would look like:

.. code-block:: C

    // launched on a grid of (M / TM) x (N / TN) programs of 1 thread each
    __global__ void transpose(TYPE * X, TYPE * Y,
                              int M, int N, int ldx, int ldy) {
    // extract program ID
      int pidm = get_program_id(0); //(1)
      int pidn = get_program_id(1); //(2)
      // create 1D range along the two matrix's axes
      int rm[TM] = pidm * TM + 0 ... TM; //(3)
      int rn[TN] = pidn * TN + 0 ... TN; //(4)
      // create 2D array of pointers
      TYPE* px[TM, TN] = X + rm[:, newaxis] * ldx + rn[newaxis, :]; //(5)
      TYPE* py[TN, TM] = Y + rm[newaxis, :] + rn[:, newaxis] * ldy; //(6)
      // write back using the transposition operator '^'
      *py = ^(*px); //(7)
    }

At a high level, this kernel loads a :code:`TM x TN` tile from the input matrix :code:`X`, transposes it and writes the resulting :code:`TN x TM` tile to the output matrix :code:`Y`. Eventually, transposition of the full input matrix is achieved by launching a grid of :code:`(M / TM) x (N / TN)` programs decomposed as follows:

- Statements (1) and (2) extract the coordinates the program in the above 2D launch grid. For example, the program producing the output tile `Y[TN:2TN-1, 2TN:3TN-1]` holds the values:

  .. code-block:: C

    pidm = 2
    pidn = 1


- Statements (3) and (4) construct the ranges of indices:

  .. code-block:: C

    rm = [pidm*TM + 0, pidm*TM + 1, ..., pidm*TM + (TM - 1)]
    rn = [pidn*TN + 0, pidn*TN + 1, ..., pidn*TN + (TN - 1)]


which will be used in statements (5) and (6) to construct tiles of pointers

- Statements (5) constructs the following array of pointers `px` using numpy-style broadcasting semantics:

::

    │ X + (pidm*TM + 0)       + (pidn*TN + 0)*ldx,  ...,  ...,  X + (pidm*TM + 0)      +  (pidn*TN + TN - 1)*ldx) │
    │      ⋮                                                                                       ⋮              │
    │      ⋮                                                                                       ⋮              │
    │ X + (pidm*TM + TM - 1)  + (pidn*TN + 0)*ldx,  ...,  ...,  X + (pidm*TM + TM - 1) +  (pidn*TN + TN - 1)*ldx) │


- Statement (6) constructs the following array of pointers `py` using numpy-style broadcasting semantics:

::

    │ Y + (pidn*TN + 0)       + (pidm*TM + 0)*ldy,  ...,  ...,  Y + (pidn*TN + 0)      +  (pidm*TM + TM - 1)*ldy) │
    │      ⋮                                                                                       ⋮              │
    │      ⋮                                                                                       ⋮              │
    │ Y + (pidn*TN + TN - 1)  + (pidn*TN + 0)*ldy,  ...,  ...,  Y + (pidn*TN + TN - 1) +  (pidm*TM + TM - 1)*ldy) │

- Statement (7) element-wise dereferences the above array of pointers `*px`, transposes it using the unary transposition operator `^`, and writes it back at the location specified by `py`.


==================================
A Note on Numpy-style Broadcasting
==================================

The construction statements (5) and (6) are a little subtle. To help understand them, consider the following numpy example.

First, we create a row vector of numbers 0 to 11, which we reshape into a 4x3 matrix.

.. code-block:: python

  import numpy as np

  vec = np.linspace(0,11,12)
  mat = vec.reshape((4,3))

Imagine that we would like to process this in two 2x3 tiles (i.e. tile 0 will consider the top half, and tile 1 will consider the bottom).

::

  [[ 0,  1,  2],
   [ 3,  4,  5],
   [ 6,  7,  8],
   [ 9, 10, 11]]

Given `pidm=0`, `pidn=0`, `TM=2`, `TN=3`, we would like for tile 0 to have the values:

::

    [ 0,  1,  2],
    [ 3,  4,  5],

We construct ranges `rm` and `rn` as:
::

    rm = [0, 1]
    rn = [0, 1, 2]

Using numpy-style broadcasting, we can add these together to create a matrix:

::

    rm[:, np.newaxis] + rn[np.newaxis, :]

           rn -> [0, 1, 2]
     rm -> [0., [[0, 1, 2],
            1.]  [1, 2, 3]]

The bottom row is incorrect. Notice that `rm` indexes the rows of the matrix; we need to offset it so that each element gives the index
of the start of that row. For instance, to access row 1 column 0, we need to access location 3. To access row 2 column 0, we need
to access location 6. To translate from row N, column 0, we need to multiply N by the number of columns in each row (the leading dimension).
In this case this is 3, so what we really need is:

::

    ldx = 3
    px  = rm[:, np.newaxis] * ldx + rn[np.newaxis,:]

`newaxis` is built into Triton, and pointer arrays can be constructed in just the same way (as in this example).

==========================
The __multipleof attribute
==========================

The memory loads and store in our transposition kernel are not vectorizable by default, since `X + ldx` (and `Y + ldy`) may be misaligned when `ldx` (and `ldy`) are not multiples of e.g., 4. This is unfortunate because tensor dimensions can be easily made into  nice powers of two in Deep Learning, due to batch-sizes and layer width being flexible.

For this reason, Triton provides a __multipleof(N) attributes for variables that are guaranteed to always be multiple of N. In the case of Matrix Transpositions, vector loads can be enabled by modifying the function's signature as follows:

.. code-block:: C

  __global__ void transpose(TYPE * X, TYPE * Y,  int M, int N,
                            int ldx __multipleof(8),
                            int ldy __multipleof(8)) {
  // ...
  }


==========================
Bounds Checking
==========================


You might have noticed that the above code will fail when `M` and `N` are not multiples of `TM` and `TN` respectively. Fortunately, the above kernel can be slightly modified to handle this situation, as shown below:

.. code-block:: C

    // launched on a grid of ((M + TM - 1) / TM) x ((N + TN - 1) / TN) programs
    __global__ void transpose(TYPE * X, TYPE * Y,  int M, int N, int ldx, int ldy) {
       // ...
       // create bounds-checking mask
       bool checkx[TM, TN] = (rm[:, newaxis] < M) && (rn[newaxis, :] < N); //(7a)
       bool checky[TN, TM] = (rm[newaxis, :] < M) && (rn[:, newaxis] < N); //(7b)
       // conditional write-back using the conditional dereferencing operatior '*?()'
       *?(checky)py = ^(*?(checkx)px); //(7)
    }


Here, statements (7a) creates an array of booleans :code:`checkx[TM, TN]` such that :code:`checkx(i, j) = True` if and only if `px(i, j)` should be dereferenced. Statement (7b) does the same for `py`. Both `px` and `py` are then conditionally dereferenced using Triton-C's conditional dereferencing operator :code:`*?(predicate) pointer`.

A runnable version of this kernel is available `here <https://github.com/ptillet/triton/tree/master/python/examples/tutorials/mat_transpose.py>`_.
