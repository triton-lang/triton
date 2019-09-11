# The PyTriton API


## <span style="color:darkred"> Table of Contents </span>

1. [Motivations](#motivations)
2. [Triton Functions](#pytriton-function)
    1. [Creation of Triton Kernels](#creation-triton-kernels)
    2. [Usage of Triton Kernels](#usage-triton-kernels)
3. [Integration with Automatic Differentiation](#autodiff)
    1. [Basics](#autodiff-basics)
    2. [Convenience](#autodiff-convenience)


## <span style="color:darkred"> Motivations </span> <a name="motivations"></a>


The purpose of PyTriton is to provide an API for easily executing Triton-C kernels from PyTorch and Tensorflow. One of the main advantages of PyTriton is that it is framework agnostic: any custom op written using this API will be transparently compatible with both Tensorflow and PyTorch without any additional effort required, as will be shown in this tutorial. 

Consider for example the following piece of code:

```python
import numpy as np
import triton

def run_tf():
  M, N, K = 128, 128, 128
  a = tf.placeholder(tf.float32, shape=[M, K])
  b = tf.placeholder(tf.float32, shape=[N, K])
  c = triton.ops.dot(a, b, transpose_a = False, transpose_b = True)
  da, db = tf.gradients(c, [a, b])
  # Run
  ha = np.random.rand(M, K).astype(np.float32)
  hb = np.random.rand(K, N).astype(np.float32)
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  result = sess.run([da], feed_dict = {a: ha, b: hb})

def run_torch():
  M, N, K = 128, 128, 128
  a = torch.randn(M, K).cuda()
  b = torch.randn(K, N).cuda()
  a.requires_grad_(True)
  b.requires_grad_(True)
  c = triton.ops.dot(a, b, False, True)
  c.backward()
  da = a.grad.clone()
  db = b.grad.clone()

## Run on tensorflow
# import tensorflow as tf
# run_tf()

## Run on pytorch
# import torch
# run_torch()
```

PyTriton works by detecting which frameworks are imported and automatically generating and just-in-time compiling C++ binding code for them. Specifically, the following chain of events is triggered when a Triton operation is executed:

1. The imported frameworks are detected
2. C++ binding code for Tensorflow or PyTorch is generated, compiled and cached.
3. The corresponding custom-op is automatically loaded from the generated .so file, and a framework-agnostic wrapper is created.
4. The wrapper is called and a tf.tensor or a torch.tensor is returned. In the case of Tensorflow, the gradient is also registered at this point if applicable


The remainder of this tutorial will show you how to re-implement the above `triton.ops.dot` operation from scratch.

## <span style="color:darkred"> PyTriton Functions </span> <a name="pytriton-function"></a>

The PyTriton API provides a `triton.function` class which automatically handles the interaction with automatic differentiation in whichever framework was detected. Therefore, every differentiable custom operation written with PyTriton should inherit from this class

```python
import triton

class _dot(triton.function):

```

### <span style="color:darkblue">Creation of Triton Kernels </span> <a name="creation-triton-kernel"></a>


PyTriton also provides a `triton.kernel` class which automatically takes care of interaction with the Triton-JIT as well as the generation and compilation of C++ framework bindings code. For our dot operation we create a kernel from the Triton-C code derived at the end of the [previous tutorial](https://github.com/ptillet/triton/blob/master/docs/triton-c.md)

``` 
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

}
"""

  kernel = triton.kernel(src, ['C'])
```

Note that the second argument to `triton.kernel` constructors indicates which of the operands our kernel function should return. Here, we only return `C`.

At this point, `kernel` is a callable object which takes the same signature as the `dot` function in our source code, except that pointers are treated as tensors: 
```
[tensor, tensor, tensor, int, int, int, int, int, int]
```

### <span style="color:darkblue">Usage of Triton Kernels </span> <a name="usage-triton-kernels"></a>

However, in practice only A, B are provided by the user, and all the other `int` arguments should be derived from these operands only. Hence, we create a helper function that extracts shapes from the `A` and `B` tensors, and then returns the results of a call to `kernel`:

```python
  @staticmethod
  def _call(a, b, transpose_a, transpose_b):
    # extract shapes
    shape_a = triton.shape(a)
    shape_b = triton.shape(b)
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
    # compute
    grid = lambda opt: [triton.cdiv(M, opt.d('TM')), triton.cdiv(N, opt.d('TN'))]
    # macros -- not necessary but makes kernel source-code simpler
    macros = {# handle A transposition
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
    return _dot.kernel(a, b, c, M, N, Ka, lda, ldb, ldc, grid,           
                  AT = transpose_a, BT = transpose_b, TYPE = dtype, 
                  TM = [32, 64, 128], TN = [32, 64, 128], TK = [8], **macros)

```

While this code should be mostly self-explanatory, there are a few of noteworthy things worth pointing out

- `triton.shape` provides a framework-agnostic way to retrieve the shape of a tensor

- `triton.empty` creates an empty tensor of the specified dimensions

- `grid` corresponds to the grid with which our Triton kernel will be launched. Because in our case this grid depends on parametric tile variables, it is supplied as a function of compilation options `opt`, whose compile-time definition can be retrieved using `opt.d(name)`. Here, `opt.d('TM')` and `opt.d('TN')` retrieve the first and second tile dimension our kernel was compiled with. We also provide a helper `triton.cdiv` for ceil divisions.

- `macros` provides a list of preprocessor definitions to compile the kernel with. Alternatively, these can also be supplied as named argument to the `_dot.kernel`. We recall that lists can be supplied to the preprocessor, in which case an auto-tuning procedure will be triggered. Here, the value of `TM` and `TN` are both tuned between 32, 64 and 128.

## <span style="color:darkred"> Compatibility with Automatic Differentiation</span> <a name="autodiff"></a>

At this point, our custom operation only takes two tensor arguments and transposition information, which is good. However, it is still not compatible with PyTorch's or TensorFlow's automatic differentiation engine, and a small amount of additional effort is needed.

### <span style="color:darkblue"> Basics </span> <a name="autodiff-basics"></a>

PyTriton binds to Tensorflow's and PyTorch's automatic differentiation framework using a single, common API inspired by PyTorch. It consists of two static methods `forward` and `backward` that take a context as their first input:

```
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
```

### <span style="color:darkblue">Convenience </span> <a name="autodiff-convenience"></a>

Still like for PyTorch, a callable operation can be created using the `apply` method of our `triton.function` class. We wrap it as a module variable for convenience:

```python
dot = _dot.apply
```
And that's it! Our custom op is now created and ready to be used with both PyTorch and Tensorflow.
