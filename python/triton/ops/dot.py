import triton

class _dot(triton.function):

  src = """
void dot(TYPE * A __noalias __readonly __aligned(16),
         TYPE * B __noalias __readonly __aligned(16),
         TYPE * C,
         float alpha,
         int M, int N, int K,
         int lda __multipleof(8),
         int ldb __multipleof(8),
         int ldc) {
    // prologue
      int ridx = get_program_id(0);
      int ridy = get_program_id(1);
      int rm[TM] = ridx * TM + 0 ... TM;
      int rn[TN] = ridy * TN + 0 ... TN;
      int rk[TK] = 0 ... TK;

      // pointers to operands
      TYPE* pa[SHAPE_A] = A + rk[BROADCAST_AK] * STRIDE_AK + rm[BROADCAST_AM] * STRIDE_AM;
      TYPE* pb[SHAPE_B] = B + rk[BROADCAST_BK] * STRIDE_BK + rn[BROADCAST_BN] * STRIDE_BN;

      // prefetches operands
      bool checka[SHAPE_A] = rk[BROADCAST_AK] < K;
      bool checkb[SHAPE_B] = rk[BROADCAST_BK] < K;
      TYPE a[SHAPE_A] = checka ? *pa : 0;
      TYPE b[SHAPE_B] = checkb ? *pb : 0;

      // reduction loop
      float c[TM, TN] = 0;
      for(int k = K; k > 0; k -= TK){
        c += USE_A @ USE_B;
        bool checka[SHAPE_A] = k > TK;
        bool checkb[SHAPE_B] = k > TK;
        pa += TK * STRIDE_AK;
        pb += TK * STRIDE_BK;
        a = *?(checka)pa;
        b = *?(checkb)pb;
      }
      //c = c * alpha;

      // epilogue
      int rxm[TM] = get_program_id(0) * TM + 0 ... TM;
      int rxn[TN] = get_program_id(1) * TN + 0 ... TN;
      TYPE* pc[TM, TN] = C + rxm[:, newaxis] * ldc + rxn[newaxis, :];
      bool checkc[TM, TN] = (rxm[:, newaxis] < M) && (rxn[newaxis, :] < N);
      *?(checkc)pc = (TYPE[TM, TN])c;
}
"""
  kernel = triton.kernel(src, ['C'])

  @staticmethod
  def _call(a, b, transpose_a, transpose_b, bench):
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
    _dot.kernel(a, b, c, 1., M, N, Ka, lda, ldb, ldc, 
                grid, bench=bench,           
                AT = transpose_a, BT = transpose_b, TYPE = dtype, 
                TM = [64], TN = [128], TK = [8], **macros)
    return c

  @staticmethod
  def forward(ctx, a, b, transpose_a = False, transpose_b = False, bench = 0):
    ctx.save_for_backward(a, b)
    ctx.t_a = transpose_a
    ctx.t_b = transpose_b
    ctx.bench = bench
    return _dot._call(a, b, transpose_a, transpose_b, bench)

  @staticmethod
  def backward(ctx, dy):
    a, b = ctx.saved_tensors
    t_a, t_b = ctx.t_a, ctx.t_b
    bench = ctx.bench
    if not t_a and not t_b:
      da = _dot._call(dy, b, False, True, bench)
      db = _dot._call(a, dy, True, False, bench)
    elif not t_a and t_b:
      da = _dot._call(dy, b, False, False, bench)
      db = _dot._call(dy, a, True, False, bench)
    elif t_a and not t_b:
      da = _dot._call(b, dy, False, True, bench)
      db = _dot._call(a, dy, False, False, bench)
    elif t_a and t_b:
      da = _dot._call(b, dy, True, True, bench)
      db = _dot._call(dy, a, True, True, bench)
    else:
      assert False
    return da, db, None, None, None

dot = _dot.apply