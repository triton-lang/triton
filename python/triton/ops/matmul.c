#define STM 1
#define STN 1

__global__ void matmul(TYPE *A __noalias __readonly,
                       TYPE *B __noalias __readonly,
                       TYPE *C __noalias,
                       float alpha,
                       int M, int N, int K,
                       int lda, int ldb, int ldc,
                       int *locks) {
  // prologue
  int pid = get_program_id(0);
  int pidz = get_program_id(2);
  int gridm = (M + TM - 1) / TM;
  int gridn = (N + TN - 1) / TN;

  // swizzle for better L2 performance
  int width = STM * gridn;
  int stm = pid / width;
  int RSTM = min(gridm - stm * STM, STM);
  int stn = (pid % width) / (RSTM * STN);
  int RSTN = min(gridn - stn * STN, STN);
  int laneid = pid % (RSTM * RSTN);
  int lanem = laneid / RSTN;
  int lanen = laneid % RSTN;
  int pidm = stm * STM + lanem;
  int pidn = stn * STN + lanen;
  int rm[TM] = pidm * TM + 0 ... TM;
  int rn[TN] = pidn * TN + 0 ... TN;

  // split-k for better parrallelism
  K = K / SPLITK;
  int rk[TK] = 0 ... TK;
  // pointers to operands
  int offa[TM, TK] = (pidz * K + rk [newaxis, :]) * STRIDE_AK + rm[:, newaxis] * STRIDE_AM;
  int offb[TK, TN] = (pidz * K + rk[:, newaxis]) * STRIDE_BK + rn [newaxis, :] * STRIDE_BN;
  TYPE *pa[TM, TK] = A + offa;
  TYPE *pb[TK, TN] = B + offb;

  // prefetches operands
  bool checka[TM, TK] = rk [newaxis, :] < K;
  bool checkb[TK, TN] = rk[:, newaxis] < K;
  TYPE a[TM, TK] = checka ? *pa : 0;
  TYPE b[TK, TN] = checkb ? *pb : 0;
  pa += TK * STRIDE_AK;
  pb += TK * STRIDE_BK;

  // reduction loop
  float acc[TM, TN] = 0;
  for (int k = K; k > 0; k -= TK) {
#if (IS_TK_DIV_K == 1)
    bool checkk[TK] = k > TK;
#else
    bool checkk[TK] = rk < k - TK;
#endif
    bool checka[TM, TK] = checkk [newaxis, :];
    bool checkb[TK, TN] = checkk[:, newaxis];
    acc += a @b;
#if (IS_TK_DIV_K == 1)
    a = *? (checka)pa;
    b = *? (checkb)pb;
#else
    a = checka ? *pa : 0;
    b = checkb ? *pb : 0;
#endif
    pa += TK * STRIDE_AK;
    pb += TK * STRIDE_BK;
  }
  acc = acc * alpha;
  TYPE c[TM, TN] = acc;

  // epilogue
  int rcm[TM] = pidm * TM + 0 ... TM;
  int rcn[TN] = pidn * TN + 0 ... TN;
  int offc[TM, TN] = rcm[:, newaxis] * ldc + rcn [newaxis, :];
  TYPE *pc[TM, TN] = C + offc;
  bool checkc[TM, TN] = rcm[:, newaxis] < M && rcn [newaxis, :] < N;
#if (SPLITK == 1)
  *? (checkc)pc = c;
#else
  // accumulate partial result using spin-locks
  int *plock = locks + pid;
  int *pcount = plock + get_num_programs(0);
  for (int repeat = 1; repeat == 1; repeat = atomic_cas(plock, 0, 1))
    ;
  int count = *pcount;
  if (count == 0)
    *? (checkc)pc = c;
  else
    *? (checkc)pc = c + *? (checkc)pc;
  atomic_xchg(pcount, (count + 1) % SPLITK);
  atomic_xchg(plock, 0);
#endif
}