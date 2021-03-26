__global__ void conv(TYPE *A __noalias __readonly,
                     TYPE *B __noalias __readonly,
                     TYPE *C __noalias,
                     float alpha,
                     // equivalent matmul
                     int M, int N, int K,
                     // convolution properties
                     int pad_h, int pad_w, int stride_h, int stride_w,
                     // pointer increment
                     int *ADELTA,
                     // memory strides
                     int lda_z, int lda_ci, int lda_h, int lda_w,
                     int ldb_ci, int ldb_r, int ldb_s, int ldb_co,
                     int ldc_z, int ldc_co, int ldc_p, int ldc_q)
{
  // prologue
  int ridx = get_program_id(0);
  int ridy = get_program_id(1);
  int ridz = get_program_id(2);
  int gridx = M / TM;
  int gridy = N / TN;
  int rid = ridx + ridy * gridx;
  ridx = rid / gridy;
  ridy = rid % gridy;
  int rm[TM] = ridx * TM + 0 ... TM;
  int rn[TN] = ridy * TN + 0 ... TN;
  // reduction splitting
  K = K / TZ;
  int rk[TK] = ridz * K + 0 ... TK;

  // unpack aggregate rows
  // m = (z, p, q)
  int rq[TM] = rm % QQ;
  int rzp[TM] = rm / QQ;
  int rp[TM] = rzp % PP;
  int rz[TM] = rzp / PP;
  // unpack aggregate reduction
  // k = (ci, r, s)
  int rs[TK] = rk % SS;
  int rcir[TK] = rk / SS;
  int rr[TK] = rcir % RR;
  int rci[TK] = rcir / RR;

  // padding / striding
  int rh_0[TM] = rp * stride_h - pad_h;
  int rw_0[TM] = rq * stride_w - pad_w;
  int rh[TM, TK] = rh_0[:, newaxis] + rr [newaxis, :];
  int rw[TM, TK] = rw_0[:, newaxis] + rs [newaxis, :];

  // pointers to lhs
  int offa[TM, TK] = rz[:, newaxis] * lda_z + rci [newaxis, :] * lda_ci +
                     rh * lda_h + rw * 1;
  TYPE *pa[TM, TK] = A + offa;
  int *padelta[TK] = ADELTA + rk;
  // pointers to rhs
  int offb[TK, TN] = rci[:, newaxis] * ldb_ci + rr[:, newaxis] * ldb_r +
                      rs[:, newaxis] * ldb_s  + rn [newaxis, :] * 1;
  TYPE *pb[TK, TN] = B + offb;

  // prefetches operands
  bool checkam[TM, TK] = rm[:, newaxis] < M;
  bool checka[TM, TK] = checkam && rh >= 0 && rh < HH && rw >= 0 && rw < WW;
  bool checkb[TK, TN] = rk[:, newaxis] < K;
  TYPE a[TM, TK] = checka ? *pa : 0;
  TYPE b[TK, TN] = checkb ? *pb : 0;
  int total = 0;

  // reduction loop
  float acc[TM, TN] = 0;
  for (int k = K; k > 0; k -= TK)
  {
    acc += a @b;
    // increment A
    int adelta[TK] = *padelta;
    padelta += TK;
    pa += adelta [newaxis, :];
    // bounds-checking A
    rk += TK;
    rs = rk % SS;
    rcir = rk / SS;
    rr = rcir % RR;
    rh = rh_0[:, newaxis] + rr [newaxis, :];
    rw = rw_0[:, newaxis] + rs [newaxis, :];
    bool checka[TM, TK] = checkam && rh >= 0 && rh < HH && rw >= 0 && rw < WW;
    // increment B
    pb += TK * ldb_s;
    // bounds-checking B
    bool checkb[TK, TN] = k > TK;
    a = checka ? *pa : 0;
    b = *? (checkb)pb;
  }
  acc = acc * alpha;
  TYPE c[TM, TN] = acc;

  // epilogue
  rm = ridx * TM + 0 ... TM;
  rn = ridy * TN + 0 ... TN;
  rq = rm % QQ;
  rzp = rm / QQ;
  rp = rzp % PP;
  rz = rzp / PP;
  int offc[TM, TN] = rz[:, newaxis] * ldc_z + rn [newaxis, :] * ldc_co +
                     rp[:, newaxis] * ldc_p + rq[:, newaxis] * 1;
  TYPE *pc[TM, TN] = C + offc;
  bool checkc[TM, TN] = rm[:, newaxis] < M && rn [newaxis, :] < N;

#if (TZ == 1)
  *? (checkc)pc = c;
#else
  // accumulate partial result using spin-locks
  int *plock = locks + rid;
  int *pcount = plock + get_num_programs(0) * get_num_programs(1);
  for (int repeat = 1; repeat == 1; repeat = atomic_cas(plock, 0, 1))
    ;
  int count = *pcount;
  if (count == 0)
    *? (checkc)pc = c;
  else
    *? (checkc)pc = c + *? (checkc)pc;
  atomic_xchg(pcount, (count + 1) % TZ);
  atomic_xchg(plock, 0);
#endif
}