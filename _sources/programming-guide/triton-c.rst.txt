=======================
The Triton-C Language
=======================

In the introduction, we stressed the importance of blocked algorithms and described their core principles in pseudo-code. To facilitate their implementation on modern GPU hardware, we present Triton-C, a single-threaded imperative kernel language in which block variables are first-class citizen.  This language may be used either directly by developers familiar with C, or as an intermediate language for existing (and future) transcompilers. In this chapter, we describe its differences with C, its Numpy-like semantics and its "Single-Program, Multiple-Data" (SPMD) programming model.

-------------------
Differences with C
-------------------

The syntax of Triton-C is based on that of ANSI C, but was modified and extended to accomodate the semantics and programming model described in the next two  subsections. These changes fall into the following categories:

+++++++++++
Extensions
+++++++++++

**Variable declarations**: Triton adds special-purpose syntax for multi-dimensional array declarations (e.g., :code:`int block[16, 16]`), which purposely differs from that of nested arrays (i.e., arrays of pointers) found in ANSI C (e.g., :code:`int block[16][16]`). Block dimensions must be constant but can also be made parametric with the use of pre-processor macros. One-dimensional blocks of integers may be initialized using ellipses (e.g., :code:`int range[16] = 0 ... 16`).

**Primitive types**: Triton-C supports the following primitive data-types: :code:`bool`, :code:`uint8`, :code:`uint16`, :code:`uint32`, :code:`uint64`, :code:`int8`, :code:`int16`, :code:`int32`, :code:`int64`, :code:`half`, :code:`float`, :code:`double`.

**Operators and built-in function**: The usual C operators were extended to support element-wise array operations (:code:`+`, :code:`-`, :code:`&&`, :code:`*`, etc.) and complex array operations(:code:`@` for matrix multiplication). Additionally, some built-in functions were added for concurrency (:code:`get_program_id`, :code:`atomic_add`).

**Slicing and broadcasting**: Multi-dimensional blocks can be broadcast along any particular dimension using numpy-like slicing syntax (e.g., :code:`int array[8, 8] = range[:, newaxis]` for stacking columns). Note that, as of now, slicing blocks to retrieve sub-blocks (or scalars) is forbidden as it is incompatible with the automatic parallelization methods used by our JIT. Reductions can be achieved using a syntax similar to slicing (e.g., :code:`array[+]` for summing an array, or :code:`array[:, max]` for row-wise maximum). Currently supported reduction operators are :code:`+`, :code:`min`, :code:`max`.

**Masked pointer dereferencement**: Block-level operations in Triton-C are "atomic", in the sense that they execute either completely or not at all. Basic element-wise control-flow for block-level operations can nonetheless be achieved using ternary operators and the *masked pointer dereferencement* operator exemplified below:

.. code-block:: C

  // create mask
  bool mask[16, 16] = ...;
  // conditional addition
  float x[16, 16] = mask ? a + b : 0;
  // conditional load
  float y[16] 16] = mask ? *ptr : 0;
  // conditional store
  *?(mask)ptr = y;
  \end{lstlisting}


+++++++++++++
Restrictions
+++++++++++++

The Triton project is still in its infancy. As such, there are quite a few features of ANSI C that are not supported:

**Non-kernel functions**: Right now, all function definitions must be kernels, i.e. be preceded with the :code:`__global__` attribute. We are aware that this is a severe limitations, and the reason why it exists is because our automatic parallelization engine would not be capable of handling array parameter arguments.

**Non-primitive types**: Non-primitive types defined with :code:`struct` and :code:`union` are currently not supported, again because it is unclear at this point how these constructs would hook into our block-level data-flow analysis passes.

**While loops**: We just haven't had time to implement those yet.

----------------
Semantics
----------------

The existence of built-in **blocked** types, variable and operations in Triton-C offers two main benefits. First, it simplifies the structure of blocked programs by hiding important details pertaining to concurrent programming such as memory coalescing, cache management and specialized tensor instrinsics. Second, it opens the door for compilers to perform these optimizations automatically. However, it also means that programs have some kind of *block-level semantics* that does not exist in C. Though some aspects of it (e.g., the :code:`@` operator) are pretty intuitive, one in particular might be puzzling to some GPU programmers: broadcasting semantics.

+++++++++++++++++++++++
Broadcasting Semantics
+++++++++++++++++++++++


Block variables in Triton are strongly typed, meaning that certain instructions statically require their operands to satisfy strict shape constraints. For example, a scalar may not be added to an array unless it is first appropriately broadcast. *Broadcasting semantics* (first introduced in `Numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_) provides two formal rules for performing these conversions automatically in the case of binary operators: (1) the shape of the lowest-dimension operand is left-padded with ones until both operands have the same dimensionality; and (2) the content of both operands is replicated as many times as needed until their shape is identical. An error is emitted if this cannot be done.

.. code-block:: C

  int a[16], b[32, 16], c[16, 1];
  // a is first reshaped to [1, 16]
  // and then broadcast to [32, 16]
  int x_1[32, 16] = a[newaxis, :] + b;
  // Same as above but implicitly
  int x_2[32, 16] = a + b;
  // a is first reshaped to [1, 16]
  // a is broadcast to [16, 16]
  // c is broadcast to [16, 16]
  int y[16, 16] = a + c;

------------------
Programming Model
------------------

As discussed in the `CUDA documentation <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>`_, The execution of CUDA  code on GPUs is supported by an `SPMD <https://en.wikipedia.org/wiki/SPMD>`_ programming model in which each kernel instance is associated with an identifiable *thread-block*, itself decomposed into *warps* of 32 *threads*. The Triton programming model is similar, but each kernel is *single-threaded* -- though automatically parallelized -- and associated with a global :code:`program id` which varies from instance to instance. This approach leads to simpler kernels in which CUDA-like concurrency primitives (shared memory synchronization, inter-thread communication, etc.) do not exist. The global program ids associated with each  kernel instance can be queried using the :code:`get_program_id(axis)` built-in function where :code:`0 <= axis <= 2`. This is, for example, useful to create e.g., blocks of pointers as shown in the tutorials.

