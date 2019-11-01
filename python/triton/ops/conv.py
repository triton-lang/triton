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
          int* a_delta, int* inc_a){

  // range of indices along the reduction axis
  int rka[TK] = 0 ... TK;
  int rkb[TK] = 0 ... TK;

  // initialize accumulator
  float c[TM, TN] = 0;

  // pointers for A
  int rxa[TM] = get_program_id(0) * TM + 0 ... TM;
  int rabh[TM] = rxa / CW;
  int raw[TM] = rxa % CW;
  int rab[TM] = rabh / CH;
  int rah[TM] = rabh % CH;
  rah = rah * UPAW - off_uah;
  raw = raw * UPAH - off_uaw;
  int racr[TK] = rka / BW;
  int ras[TK] = rka % BW;
  int rac[TK] = racr / BH;
  int rar[TK] = racr % BH;
  rar = UPAR * rar;
  ras = UPAS * ras;
  int ra0[TM] = rab*lda_n + rah*lda_h + raw*lda_w;
  int ra1[TK] = rac*lda_c + rar*lda_h + ras*lda_w;
  A_TYPE* pa[TM, TK] = A + ra0[:, newaxis] + ra1[newaxis, :];

  // pointers for B
  int rbn[TN] = get_program_id(1) * TN + 0 ... TN;
  B_TYPE* pb[TK, TN] = B + rbn[newaxis, :] * ldb_k + rkb[:, newaxis] * ldb_s;

  // pointers for A look-up table
  int offda[TK] = rka % LUT_SIZE;
  int* pincd[TK] = inc_a + offda;
  int* pda[TK]  = a_delta + offda + off_uw * LUT_SIZE + off_uh * LUT_SIZE * upsample_w;
  int da[TK] = *pda;
  int incd[TK] = *pincd;

  // reduction loop
  A_TYPE a[TM, TK] = *pa;
  B_TYPE b[TK, TN] = *pb;
  for(int k = K; k > 0; k = k - TK){
    c += a @ b;
    pa += da[newaxis, :];
    pb += TK * ldb_s;
    // increment A look-up table
    pda = pda + incd;
    da = *pda;
    pincd = pincd + incd;
    incd = *pincd;
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
  def _unpack(idx, D, H, W):
    cdh = idx // W
    w = idx % W
    cd = cdh // H
    h = cdh % H
    c = cd // D
    d = cd % D
    return c, d, h, w

  @staticmethod
  def _delta_a(upsample_d, upsample_h, upsample_w, depth, TK,
               T, R, S, stride_a):
    ud = np.arange(upsample_d, dtype=np.int32)[:, np.newaxis, np.newaxis, np.newaxis]
    uh = np.arange(upsample_h, dtype=np.int32)[np.newaxis, :, np.newaxis, np.newaxis]
    uw = np.arange(upsample_w, dtype=np.int32)[np.newaxis, np.newaxis, :, np.newaxis]
    ctrs = np.arange(depth, dtype=np.int32)[np.newaxis, np.newaxis, np.newaxis, :]
    c, t, r, s = _conv._unpack(ctrs, T, R, S)
    nextc, nextt, nextr, nexts = _conv._unpack(ctrs + TK, T, R, S)
    cdiff = nextc - c
    tdiff = nextt - t
    rdiff = nextr - r
    sdiff = nexts - s
    return cdiff*stride_a[1] + tdiff*stride_a[2] + rdiff*stride_a[3] + sdiff*stride_a[4]

  @staticmethod
  def _extract_strides(shape):
    rank = len(shape)
    ret = [1] * rank
    for i in range(rank - 1, 0, -1):
      ret[i-1] = ret[i] * shape[i]
    return ret


  @staticmethod
  def _call(a, b, 
            upsample_d, upsample_h, upsample_w, 
            pad_d, pad_h, pad_w, 
            stride_d, stride_h, stride_w, 
            mode):
    # input shapes
    shape_a = list(triton.shape(a))
    shape_b = list(triton.shape(b))
    # add depth
    shape_a.insert(2, 1)
    shape_b.insert(1, 1)
    NB, NC, AD, AH, AW = shape_a
    NC, BD, BH, BW, NF = shape_b
    # output shape
    CD = (AD*upsample_d - BD + 1 + 2*pad_d + stride_d - 1) // stride_d
    CH = (AH*upsample_h - BH + 1 + 2*pad_h + stride_h - 1) // stride_h
    CW = (AW*upsample_w - BW + 1 + 2*pad_w + stride_w - 1) // stride_w
    shape_c = [NB, NF, CD, CH, CW]
    # strides
    stride_a = _conv._extract_strides(shape_a)
    stride_b = _conv._extract_strides(shape_b)
    stride_c = _conv._extract_strides(shape_c)
    # look-up tables
    TK = 8
    FS = BD * BH * BW 
    depth = (TK + FS - 1)//FS * FS
    delta_a = _conv._delta_a(upsample_d, upsample_h, upsample_w, 
                             depth, TK, BD, BH, BW, stride_a)
    delta_a = triton.fw.torch.from_numpy(delta_a).cuda()
    inc_a = np.arange(depth, dtype=np.int32)
    inc_a = ((inc_a + TK) % depth) - inc_a
    inc_a = triton.fw.torch.from_numpy(inc_a).cuda()

    trans_b = False
    is_wgrad = False
    is_blut = False 
    macros = {  
               'UPAR':         'stride_h'       if is_wgrad else '1',
               'UPAS':         'stride_w'       if is_wgrad else '1',
               'UPAH':         ''               if is_wgrad else 'stride_h',
               'UPAW':         ''               if is_wgrad else 'stride_w',
               'LUT_SIZE':      depth,
               'TM': [32],
               'TN': [32],
               'TK': TK,
               'A_TYPE': 'float',
               'B_TYPE': 'float'
              }
    
    shape_c.pop(2)
    c = triton.empty(shape_c, dtype=a.dtype)
    grid = lambda opt: [triton.cdiv(NB*CD*CH*CW, opt.d('TM')), triton.cdiv(NF, opt.d('TN'))]
    print(stride_c)
    print(stride_b)
    _conv.kernel(a, b, c, NB*CD*CH*CW, NF, NC*BD*BH*BW, AH, AW, BH, BW, CH, CW, NC,
                 stride_a[0], stride_a[1], stride_a[2], stride_a[3], stride_a[4],
                 stride_b[0], stride_b[1], stride_b[2], stride_b[3], stride_b[4],
                 stride_c[0], stride_c[1], stride_c[2], stride_c[3], stride_c[4],
                 pad_h, pad_w, stride_h, stride_w, upsample_h, upsample_w, 
                 0, 0, 0, 0, 0, 0,
                 delta_a, inc_a,
                 grid, **macros)
    return c
  
  @staticmethod
  def forward(ctx, input, weight):
    return _conv._call(input, weight, 1, 1, 1, 0, 0, 0, 1, 1, 1, '')

conv = _conv.apply