#define STM 8
#define STN 8

__global__ void matmul(TYPE *A __noalias __readonly __aligned(16),
                       TYPE *B __noalias __readonly __aligned(16),
                       TYPE *C __noalias __aligned(16),
                       float alpha,
                       int M,
                       int N,
                       int K,
                       int lda __multipleof(LDA_POW2_DIV),
                       int ldb __multipleof(LDB_POW2_DIV),
                       int ldc __multipleof(LDC_POW2_DIV),
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
  int rk[TK] = 0 ... TK;
  K = K / 1;
  // reduction loop
  float acc[TM, TN] = 0;
  int offa[TM, TK] = (pidz * TK + rk [newaxis, :]) * STRIDE_AK + rm[:, newaxis] * STRIDE_AM;
  int offb[TK, TN] = (pidz * TK + rk[:, newaxis]) * STRIDE_BK + rn [newaxis, :] * STRIDE_BN;
  TYPE *pa[TM, TK] = A + offa;
  TYPE *pb[TK, TN] = B + offb;
  for (int k = K; k > 0; k -= TK * SPLITK) {
    acc += (*pa) @(*pb);
    pa += TK * SPLITK * STRIDE_AK;
    pb += TK * SPLITK * STRIDE_BK;
  }
  // epilogue
  int offc[TM, TN] = rm[:, newaxis] * ldc + rn [newaxis, :];
  TYPE *pc[TM, TN] = C + offc;
  bool checkc[TM, TN] = rm[:, newaxis] < M && rn [newaxis, :] < N;
  TYPE c[TM, TN] = acc * alpha;
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