import triton

class _dot(triton.function):

  src = """
void dot(TYPE * A, TYPE * B, TYPE * C,
         int M, int N, int K,
         int lda __multipleof(8),
         int ldb __multipleof(8),
         int ldc) {
  // prologue
  int ridx = get_program_id(0);
  int ridy = get_program_id(1);
  int rxa[TM] = ridx * TM + 0 ... TM;
  int ryb[TN] = ridy * TN + 0 ... TN;
  int rka[TK] = 0 ... TK;
  int rkb[TK] = 0 ... TK;
  float c[TM, TN] = 0;
  // pointers to operands
  TYPE* pa[SHAPE_A] = A + rka[BROADCAST_AK] * STRIDE_AK + rxa[BROADCAST_AM] * STRIDE_AM;
  TYPE* pb[SHAPE_B] = B + rkb[BROADCAST_BK] * STRIDE_BK + ryb[BROADCAST_BN] * STRIDE_BN;
  // prefetches operands
  TYPE a[SHAPE_A] = *pa;
  TYPE b[SHAPE_B] = *pb;
  // reduction loop
  for(int k = K; k > 0; k-= TK){
    c += USE_A @ USE_B;
    pa = pa + TK * STRIDE_AK;
    pb = pb + TK * STRIDE_BK;
    a = *pa;
    b = *pb;
  }
  // epilogue
  int rxc[TM] =  ridx * TM + 0 ... TM;
  int ryc[TN] =  ridy * TN + 0 ... TN;
  TYPE* pc[TM, TN] = C + ryc[newaxis, :] + rxc[:, newaxis] * ldc;
  bool checkc[TM, TN] = (rxc < M)[:, newaxis] && (ryc < N)[newaxis, :];
  *?(checkc) pc = c;
}
"""

  kernel = triton.kernel(src, ['C'])

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
                  TM = [64, 128], TN = [64, 128], TK = [8], **macros)

  @staticmethod
  def forward(ctx, a, b, transpose_a = False, transpose_b = False):
    ctx.save_for_backward(a, b, transpose_a, transpose_b)
    return _dot._call(a, b, transpose_a, transpose_b)

  @staticmethod
  def backward(ctx, dy):
    a, b, t_a, t_b = ctx.saved_tensors
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
    return [da, db, None, None, None, None, None, None, None]
  
dot = _dot.apply