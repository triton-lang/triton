====================================================
Putting It All Together
====================================================

In the previous tutorial, we saw how to write tensor-core-friendly matrix multiplication code competitive with cuBLAS in 20 lines of Triton code. Here, we will see how to wrap it into an automatically differentiable PyTorch functions for easy integration in your Deep Learning pipeline.

-----------------
PyTriton Function
-----------------

The PyTriton API provides a :code:`triton.function` class which automatically handles the interaction with automatic differentiation in whichever framework was detected. Therefore, every differentiable custom operation written with PyTriton should inherit from this class

.. code-block:: python

    import triton

    # Entry point
    class _dot(torch.autograd.Function):

      @staticmethod
      # Forward Pass
      def forward(ctx, *args):
        #...

      @staticmethod
      # Backward Pass
      def backward(ctx, dy):
        #...

-----------------
PyTriton Kernels
-----------------


PyTriton also provides a :code:`triton.kernel` class which automatically takes care of interaction with the Triton-JIT as well as the generation and compilation of C++ framework bindings code. For our dot operation we create a kernel from the Triton code shown at the end of the previous tutorial.

.. code-block:: python

    src = """
    __global__ void dot(TYPE * A, TYPE * B, TYPE * C,
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
   """

   kernel = triton.kernel(src)


At this point, `kernel` is a callable object which takes the same signature as the :code:`dot` function in our source code, except that pointers are treated as tensors: :code:`[tensor, tensor, tensor, int, int, int, int, int, int]`.

-----------------------
Using PyTriton Kernels
-----------------------


However, in practice only A, B are provided by the user, and all the other :code:`int` arguments should be derived from these operands only. Hence, we create a helper function that extracts shapes from the :code:`A` and :code:`B` tensors, and then returns the results of a call to :code:`kernel`:

.. code:: python

  @staticmethod
  def _call(a, b, transpose_a, transpose_b):
    # extract shapes
    shape_a = a.shape
    shape_b = b.shape
    M, Ka = shape_a[0], shape_a[1]
    Kb, N = shape_b[0], shape_b[1]
    # transpose shapes
    if transpose_a:
      M, Ka = Ka, M
    if transpose_b:
      Kb, N = N, Kb
    # contiguous dimensions
    lda = M if transpose_a else Ka
    ldb = Kb if transpose_b else N
    ldc = N
    # data-type
    dtype = a.dtype
    # allocate output
    c = triton.empty([M, N], dtype = dtype)
    # launch grid
    grid = lambda opt: [triton.cdiv(M, opt.d('TM')), triton.cdiv(N, opt.d('TN'))]
    # pre-processor definitions
    defines = {# tile sizes
              'TYPE'        : dtype,
              'AT'          : transpose_a,
              'BT'          : transpose_b,
              'TM'          : [32, 64, 128]
              'TN'          : [32, 64, 128]
              'TK'          : [8]
              # handle A transposition
              'USE_A'       : '^a'         if transpose_a else 'a',
              'STRIDE_AK'   : 'lda'        if transpose_a else '1',
              'STRIDE_AM'   : '1'          if transpose_a else 'lda',
              'BROADCAST_AK': ':, newaxis' if transpose_a else 'newaxis, :',
              'BROADCAST_AM': 'newaxis, :' if transpose_a else ':, newaxis',
              'SHAPE_A'     : 'TK, TM'     if transpose_a else 'TM, TK',
              # handle B transposition
              'USE_B'       : '^b'         if transpose_b else 'b',
              'STRIDE_BK'   : '1'          if transpose_b else 'ldb',
              'STRIDE_BN'   : 'ldb'        if transpose_b else '1',
              'BROADCAST_BK': 'newaxis, :' if transpose_b else ':, newaxis',
              'BROADCAST_BN': ':, newaxis' if transpose_b else 'newaxis, :',
              'SHAPE_B'     : 'TN, TK'     if transpose_b else 'TK, TN'}
    return _dot.kernel(a, b, c, M, N, Ka, lda, ldb, ldc,
                       grid=grid, num_warps=4, defines=defines)


--------------------------------------------
Automatic Differentiation
--------------------------------------------

At this point, our custom operation only takes two tensor arguments and transposition information, which is good. However, it is still not compatible with PyTorch's or TensorFlow's automatic differentiation engine, and a small amount of additional effort is needed.


Creating custom operations for Triton and PyTorch is very similar; programmers have to provide two static methods :code:`forward` and :code:`backward` that take a context as their first input:

.. code:: python

  @staticmethod
  def forward(ctx, a, b, transpose_a = False, transpose_b = False):
    ctx.save_for_backward(a, b)
    ctx.t_a = transpose_a
    ctx.t_b = transpose_b
    return _dot._call(a, b, transpose_a, transpose_b)

  @staticmethod
  def backward(ctx, dy):
    a, b = ctx.saved_tensors
    t_a, t_b = ctx.t_a, ctx.t_b
    if not t_a and not t_b:
      da = _dot._call(dy, b, False, True)
      db = _dot._call(a, dy, True, False)
    elif not t_a and t_b:
      da = _dot._call(dy, b, False, False)
      db = _dot._call(dy, a, True, False)
    elif t_a and not t_b:
      da = _dot._call(b, dy, False, True)
      db = _dot._call(a, dy, False, False)
    elif t_a and t_b:
      da = _dot._call(b, dy, True, True)
      db = _dot._call(dy, a, True, True)
    else:
      assert False
    return da, db, None, None, None, None, None, None, None


A callable operation can be created using the :code:`apply` method of the :code:`torch.autograd.Function` class.

.. code:: python

  dot = _dot.apply


And that's it! In just ~100 lines of pure python, we have written a fully functional matrix multiplication that will not only work with automatic differentiation but also provide performance very close to cuBLAS. And it's all open-source~