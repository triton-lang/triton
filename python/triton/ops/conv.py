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
          int* a_delta){

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
  rar = FLIPR rar;
  ras = FLIPS ras;
  rar = UPAR * rar;
  ras = UPAS * ras;
  int ra0[TM] = rab*lda_n + rah*lda_h + raw*lda_w;
  int ra1[TK] = rac*lda_c + rar*lda_h + ras*lda_w;
  A_TYPE* pa[TM, TK] = A + ra0[:, newaxis] + ra1[newaxis, :];

  // pointers for B
  int rb0[TN] = get_program_id(1) * TN + 0 ... TN;
#ifdef B_LUT
  int rbcr[TK] = rkb / BW;
  int rbs[TK] = rkb % BW;
  int rbc[TK] = rbcr / BH;
  int rbr[TK] = rbcr % BH;
  rbr = rbr * upsample_h + off_uh;
  rbs = rbs * upsample_w + off_uw;
  int rb1[TK] = rbc*ldb_c + rbr*ldb_r + rbs*ldb_s;
#else
  int rb1[TK] = rkb * STRIDE_B0;
#endif
  B_TYPE* pb [B_SHAPE] = B + rb1[BROADCAST_B1] * STRIDE_B1 + rb0[BROADCAST_B0] * STRIDE_B0 * ldb_k;

  // pointers for A look-up table
  int offda[TK] = rka % LUT_SIZE;
  int* pincd[TK] = a_delta + offda;
  int* pda[TK]  = a_delta + LUT_SIZE + offda + off_uw * LUT_SIZE + off_uh * LUT_SIZE * upsample_w;
  int da[TK] = *pda;
  int incd[TK] = *pincd;

  // pointers for B look-up table
  int offdb[TK] = rkb % LUT_SIZE;
#ifdef B_LUT
  int* pdb[TK] = b_delta + offdb + off_uw * LUT_SIZE + off_uh * LUT_SIZE * upsample_w;
  int db[TK] = *pdb;
#endif

  // reduction loop
  A_TYPE a[TM, TK] = *pa;
  B_TYPE b[B_SHAPE] = *pb;
  for(int k = K; k > 0; k = k - TK){
    c += a @ USE_B;
    pa = pa + da[newaxis, :];
    pb = pb + INC_PB;
    // increment A look-up table
    pda = pda + incd;
    da = *pda;
    pincd = pincd + incd;
    incd = *pincd;
    // increment B look-up table
#ifdef B_LUT
    pdb = pdb + INC_PDB;
    db = *pdb;
#endif
    // pre-fetches
    a = *pa;
    b = *pb;
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
  *pc = c;
}
"""
  kernel = triton.kernel(src, ['C'])

  @staticmethod
  def _unpack(idx, D, H, W):
    c = idx // (D*H*W)
    dhw = idx % (D*H*W)
    dh = dhw // W
    w = dhw % W
    d = dh // H
    h = dh % H
    return c, d, h, w

  @staticmethod
  def _delta_a(upsample_d, upsample_h, upsample_w, depth, TK,
               T, R, S, stride_a):
    ud = np.arange(upsample_d)[:, np.newaxis, np.newaxis, np.newaxis]
    uh = np.arange(upsample_h)[np.newaxis, :, np.newaxis, np.newaxis]
    uw = np.arange(upsample_w)[np.newaxis, np.newaxis, :, np.newaxis]
    ctrs = np.arange(depth)[np.newaxis, np.newaxis, np.newaxis, :]
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

    trans_b = False
    is_wgrad = False
    is_blut = False 
    macros = { 
               'B_SHAPE':      'TN, TK'       if trans_b else 'TK, TN',
               'BROADCAST_B0': ':, newaxis'   if trans_b else 'newaxis, :',
               'BROADCAST_B1': 'newaxis, :'   if trans_b else ':, newaxis',
               'STRIDE_B0':    'ldb_s'          if trans_b else '1',
               'STRIDE_B1':    '1'              if trans_b else 'ldb_s',
               'USE_B':        '^b'             if trans_b else 'b',
               'FLIPR':        ''               if trans_b else 'BH - 1 -',
               'FLIPS':        ''               if trans_b else 'BW - 1 -',
               'UPAR':         'stride_h'       if is_wgrad else '1',
               'UPAS':         'stride_w'       if is_wgrad else '1',
               'UPAH':         ''               if is_wgrad else 'stride_h',
               'UPAW':         ''               if is_wgrad else 'stride_w',
               'REDAX0':       'NC'             if trans_b else 'BH',
               'REDAX1':       'BH'             if trans_b else 'BW',
               'REDAX2':       'BW'             if trans_b else 'NC',
               'AX0':          'c'              if trans_b else 'r',
               'AX1':          'r'              if trans_b else 's',
               'AX2':          's'              if trans_b else 'c',
               'INC_PB':       'db[newaxis, :]' if is_blut else 'TK',
               'INC_PDB':      'incd'           if trans_b else 'TK',
               'LUT_SIZE':      depth,
               'TM': [32],
               'TN': [32],
               'TK': TK,
               'A_TYPE': 'float',
               'B_TYPE': 'float'
              }
    
    shape_c.pop(2)
    print(shape_c)
    c = triton.empty(shape_c, dtype=a.dtype)
    _conv.kernel(a, b, c, CD*CH*CW, NF, NC*BD*BH*BW, AH, AW, BH, BW, CH, CW, NC,
                 stride_a[0], stride_a[1], stride_a[2], stride_a[3], stride_a[4],
                 stride_b[0], stride_b[1], stride_b[2], stride_b[3], stride_b[4],
                 stride_c[0], stride_c[1], stride_c[2], stride_c[3], stride_c[4],
                 pad_h, pad_w, stride_h, stride_w, upsample_h, upsample_w, 
                 0, 0, 0, 0, 0, 0,
                 delta_a,
                 lambda opt: (1, 1, 1), **macros)
    return c
  
  @staticmethod
  def forward(ctx, input, weight):
    _conv._call(input, weight, 1, 1, 1, 0, 0, 0, 1, 1, 1, '')

conv = _conv.apply