import triton
import numpy as np 

class _conv(triton.function):

  src = """
void convnd(A_TYPE *A,
          B_TYPE *B,
          float *C,
          int M, int N, int K,
          int AH, int AW,
          int BH, int BW,
          int CH, int CW,
          int NC,
          int lda_n, int lda_c, int lda_d, int lda_h, int lda_w,
          int ldb_c, int ldb_t, int ldb_r, int ldb_s, int ldb_k,
          int ldc_n, int ldc_k, int ldc_m, int ldc_p, int ldc_q,
          int pad_h, int pad_w,
          int stride_h, int stride_w,
          int upsample_h, int upsample_w,
          int off_uh, int off_uw,
          int off_uah, int off_uaw,
          int off_uch, int off_ucw,
          int* ADELTA, int* ADIFF){

  // range of indices along the reduction axis
  int rxa[TM] = get_program_id(0) * TM + 0 ... TM;
  int ryb[TN] = get_program_id(1) * TN + 0 ... TN;
  int rk[TK] = 0 ... TK;

  // initialize accumulator
  float c[TM, TN] = 0;

  // pointers for A
  int rabh[TM] = rxa / CW;
  int raw[TM] = rxa % CW;
  int rab[TM] = rabh / CH;
  int rah[TM] = rabh % CH;
  rah = rah * UPAW - off_uah;
  raw = raw * UPAH - off_uaw;
  int ram[TM] = rab*lda_n + rah*lda_h + raw*lda_w;
  int rak[TK] = *(ADELTA + rk);
  A_TYPE* pa[TM, TK] = A + ram[:, newaxis] + rak[newaxis, :];

  // pointers for B
  int rbk[TK] = rk;
  int rbn[TN] = ryb;
  B_TYPE* pb[TK, TN] = B + rbn[newaxis, :] * ldb_k + rbk[:, newaxis] * ldb_c;

  // pointers for A look-up table
  int rklut[TK] = rk % LUT_SIZE;
  int* padiff[TK] = ADIFF + rklut;
  int* padelta[TK] = ADELTA + TK + rklut + off_uw * LUT_SIZE + off_uh * LUT_SIZE * upsample_w;
  int adiff[TK] = *padiff;
  int adelta[TK] = *padelta;

  // reduction loop
  A_TYPE a[TM, TK] = *pa;
  B_TYPE b[TK, TN] = *pb;
  for(int k = K; k > 0; k = k - TK){
    c += a @ b;
    pa += adelta[newaxis, :];
    pb += TK * ldb_c;
    // increment A look-up table
    padelta = padelta + adiff;
    adelta = *padelta;
    padiff = padiff + adiff;
    adiff = *padiff;
    // pre-fetches
    bool checka[TM, TK] = k > TK;
    bool checkb[TK, TN] = k > TK;
    a = checka ? *pa : 0;
    b = checkb ? *pb : 0;
  }

  // write back
  int rxc[TM] = get_program_id(0) * TM + 0 ... TM;
  int rc1[TN] = get_program_id(1) * TN + 0 ... TN;
  int rcn[TM] = rxc / (CH*CW);
  int rcpq[TM] = rxc % (CH*CW);
  int rcp[TM] = rcpq / CW;
  int rcq[TM] = rcpq % CW;
  rcp = rcp * upsample_h + off_uch;
  rcq = rcq * upsample_w + off_ucw;
  int rc0[TM] = rcn * ldc_n + rcp * ldc_p + rcq * ldc_q;
  float* pc[TM, TN]  = C + rc1[newaxis, :]*ldc_k + rc0[:, newaxis];
  bool checkc0[TM] = rxc < M;
  bool checkc1[TN] = rc1 < N;
  bool checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];
  *?(checkc)pc = c;
}
"""
  kernel = triton.kernel(src, ['C'])

  @staticmethod
  def _unpack(idx, order, shape_b):
    _123 = idx  // shape_b[order[0]]
    _0   = idx   % shape_b[order[0]]
    _23  = _123 // shape_b[order[1]]
    _1   = _123  % shape_b[order[1]]
    _3   = _23  // shape_b[order[2]]
    _2   = _23   % shape_b[order[2]]
    return _0, _1, _2, _3

  @staticmethod
  def _roundup(x, div):
    return (x + div - 1) // div * div

  @staticmethod
  def _delta_a(upsample_d, upsample_h, upsample_w, 
               bc, bd, bh, bw,
               ac, ad, ah, aw,
               stride_a, shape_b,
               TK):
    # Parse the axes so that the reduction is done 
    # from the innermost dimension outward
    order = sorted([bc, bd, bh, bw], reverse = True)
    c, d, h, w = [order.index(x) for x in [bc, bd, bh, bw]]
    # Size of the lookup table is the product of the 3 innermost dimensions
    K = _conv._roundup(TK, shape_b[order[0]] * shape_b[order[1]] * shape_b[order[2]])
    # Allocate temporary arrays
    ud = np.arange(upsample_d, dtype=np.int32)[:, np.newaxis, np.newaxis, np.newaxis]
    uh = np.arange(upsample_h, dtype=np.int32)[np.newaxis, :, np.newaxis, np.newaxis]
    uw = np.arange(upsample_w, dtype=np.int32)[np.newaxis, np.newaxis, :, np.newaxis]
    k  = np.arange(K         , dtype=np.int32)[np.newaxis, np.newaxis, np.newaxis, :]
    # Find reduction indices at the current and next reduction indices
    currentk = _conv._unpack(k     , order, shape_b)
    nextk    = _conv._unpack(k + TK, order, shape_b)
    # Compute memory stride
    result = 0
    result += (nextk[c] - currentk[c]) * stride_a[ac]
    result += (nextk[d] - currentk[d]) * stride_a[ad]
    result += (nextk[h] - currentk[h]) * stride_a[ah]
    result += (nextk[w] - currentk[w]) * stride_a[aw]
    # Initial k
    ki = np.arange(TK       , dtype=np.int32)[np.newaxis, np.newaxis, np.newaxis, :]
    currentk = _conv._unpack(ki, order, shape_b)
    resulti = 0
    resulti += currentk[c] * stride_a[ac]
    resulti += currentk[d] * stride_a[ad]
    resulti += currentk[h] * stride_a[ah]
    resulti += currentk[w] * stride_a[aw]
    return np.concatenate((resulti, result), axis=-1)

  @staticmethod
  def _extract_strides(shape):
    rank = len(shape)
    ret = [1] * rank
    for i in range(rank - 1, 0, -1):
      ret[i-1] = ret[i] * shape[i]
    return ret


  @staticmethod
  def _call(a, b, 
            pad_d, pad_h, pad_w, 
            stride_d, stride_h, stride_w,
            upsample_d, upsample_h, upsample_w,
            a_layout, b_layout, c_layout):
    # input shapes
    shape_a = list(triton.shape(a))
    shape_b = list(triton.shape(b))
    dim = len(shape_a) - 2
    # indices
    an, ac, ad, ah, aw = [a_layout.find(x) for x in 'ncdhw']
    bk, bc, bd, bh, bw = [b_layout.find(x) for x in 'kctrs']
    cn, ck, cd, ch, cw = [c_layout.find(x) for x in 'nkdhw']
    # extract shapes
    if dim == 2:
      shape_a.insert(ad, 1)
    if dim == 2:
      shape_b.insert(bd, 1)
    # output shape
    shape_c = [0] * 5
    shape_c[cn] = shape_a[an]
    shape_c[ck] = shape_b[bk]
    shape_c[cd] = (shape_a[ad]*upsample_d - shape_b[bd] + 1 + 2*pad_d + stride_d - 1) // stride_d
    shape_c[ch] = (shape_a[ah]*upsample_h - shape_b[bh] + 1 + 2*pad_h + stride_h - 1) // stride_h
    shape_c[cw] = (shape_a[aw]*upsample_w - shape_b[bw] + 1 + 2*pad_w + stride_w - 1) // stride_w
    # strides
    stride_a = _conv._extract_strides(shape_a)
    stride_b = _conv._extract_strides(shape_b)
    stride_c = _conv._extract_strides(shape_c)
    # tiling parameters
    TM = [32]
    TN = [32]
    TK = 8
    # pointer deltas for a
    delta_a = _conv._delta_a(upsample_d, upsample_h, upsample_w, 
                             bc, bd, bh, bw,
                             ac, ad, ah, aw,
                             stride_a, shape_b,
                             TK)
    delta_a = triton.fw.torch.from_numpy(delta_a).cuda()
    # delta increments for a
    inc_a = np.arange(delta_a.shape[-1] - TK, dtype=np.int32)
    inc_a = ((inc_a + TK) % inc_a.size) - inc_a
    inc_a = triton.fw.torch.from_numpy(inc_a).cuda()
    # allocate output
    if dim == 2:
      shape_c.pop(cd)
    c = triton.empty(shape_c, dtype=a.dtype)
    if dim == 2:
      shape_c.insert(cd, 1)
    # execute kernel
    trans_b = False
    is_wgrad = False
    is_blut = False 
    macros = {  
               'UPAR':         'stride_h'       if is_wgrad else '1',
               'UPAS':         'stride_w'       if is_wgrad else '1',
               'UPAH':         ''               if is_wgrad else 'stride_h',
               'UPAW':         ''               if is_wgrad else 'stride_w',
               'LUT_SIZE':     delta_a.shape[-1],
               'TM': TM, 'TN': TN, 'TK': TK,
               'A_TYPE': 'float', 'B_TYPE': 'float'
              }
    MATMUL_M = shape_c[cn] * shape_c[cd] * shape_c[ch] * shape_c[cw]
    MATMUL_N = shape_c[ck]
    MATMUL_K = shape_b[bc] * shape_b[bd] * shape_b[bh] * shape_b[bw]
    _conv.kernel(a, b, c, 
                 # matrix multiplication shapes
                 MATMUL_M, MATMUL_N, MATMUL_K,
                 # shapes for a 
                 shape_a[ah], shape_a[aw], 
                 # shapes for b
                 shape_b[bh], shape_b[bw], 
                 # chapes for c
                 shape_c[ch], shape_c[cw], shape_c[cn],
                 # strides for a
                 stride_a[an], stride_a[ac], stride_a[ad + 0], stride_a[ad + 1], stride_a[ad + 2],
                 # strides for b
                 stride_b[bc], stride_b[bd + 0], stride_b[bd + 1], stride_b[bd + 2], stride_b[bk],
                 # strides for c
                 stride_c[cn], stride_c[ck], stride_c[cd], stride_c[cd + 1], stride_c[cd + 2],
                 # padding
                 pad_h, pad_w,
                 # striding 
                 stride_h, stride_w, 
                 # upsampling
                 upsample_h, upsample_w, 
                 0, 0, 0, 0, 0, 0,
                 # look-up table
                 delta_a, inc_a,
                 lambda opt: [triton.cdiv(MATMUL_M, opt.d('TM')), triton.cdiv(MATMUL_N, opt.d('TN'))],
                 **macros)
    return c
  
  @staticmethod
  def forward(ctx, x, w, 
              pad_d = 0, pad_h = 0, pad_w = 0, 
              stride_d = 1, stride_h = 1, stride_w = 1, 
              upsample_d = 1, upsample_h = 1, upsample_w = 1,
              layout_a = 'ncdhw', layout_b = 'ktrsc', layout_c = 'nkdhw'):
    # save for backward
    ctx.save_for_backward(x, w)
    ctx.pad_d = pad_d
    ctx.pad_h = pad_h
    ctx.pad_w = pad_w
    ctx.stride_d = stride_d
    ctx.stride_h = stride_h
    ctx.stride_w = stride_w
    ctx.upsample_d = upsample_d
    ctx.upsample_h = upsample_h
    ctx.upsample_w = upsample_w
    ctx.layout_a = layout_a
    ctx.layout_b = layout_b
    ctx.layout_c = layout_c
    # return
    return _conv._call(x, w, 
                    pad_d, pad_h, pad_w, 
                    stride_d, stride_h, stride_w, 
                    upsample_d, upsample_h, upsample_w, 
                    layout_a, layout_b, layout_c)

  @staticmethod
  def backward(ctx, dy):
    x, w = ctx.saved_tensors
    pad_d = ctx.pad_d
    pad_h = ctx.pad_h
    pad_w = ctx.pad_w
    stride_d = ctx.stride_d
    stride_h = ctx.stride_h
    stride_w = ctx.stride_w
    upsample_d = ctx.upsample_d
    upsample_h = ctx.upsample_h
    upsample_w = ctx.upsample_w
    layout_a = ctx.layout_a
    layout_b = ctx.layout_b
    layout_c = ctx.layout_c

    # TODO: Deal with this
    dx_pad_d = 1
    dx_pad_h = 1
    dx_pad_w = 1
    dx = _conv.call(dy, w, 
                    dw_pad_d, dw_pad_h, dw_pad_w,
                    upsample_w, upsample_h, upsample_w,
                    stride_d, stride_h, stride_w,
                    'ncdhw', 'cktrs', 'nkdhw')
    


    ret = [None] * 14
    ret[0] = None
    ret[1] = dw
    return None, 

conv = _conv.apply