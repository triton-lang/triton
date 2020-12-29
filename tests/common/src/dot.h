namespace src {

    const char *dot =
R"(
#define STM 4
#define STN 4

__global__ void dot(TYPE * A __noalias __readonly __aligned(16),
                    TYPE * B __noalias __readonly __aligned(16),
                    TYPE * C __noalias __aligned(16),
                    float alpha,
                    int M __retune,
                    int N __retune,
                    int K __retune __multipleof(16),
                    int lda __multipleof(8),
                    int ldb __multipleof(8),
                    int ldc __multipleof(8),
                    int* locks) {
      // prologue
      int pid = get_program_id(0);
      int pidz = get_program_id(2);
      int gridm = M / TM;
      int gridn = N / TN;
      int pidn = pid % gridn;
      int pidm = pid / gridn;
      /*
      int stgridm = (gridm + STM - 1) / STM;
      int stgridn = (gridn + STN - 1) / STN;
      int stid = pid / (STM * STN);
      int laneid = pid % (STM * STN);
      int stm = stid / stgridn;
      int stn = stid % stgridn;
      int lanem = laneid / STN;
      int lanen = laneid % STN;
      int pidm = stm*STM + lanem;
      int pidn = stn*STN + lanen;
      */

      int rm[TM] = pidm * TM + 0 ... TM;
      int rn[TN] = pidn * TN + 0 ... TN;

      // reduction splitting
      K           = K / TZ;
      int rk[TK]  = pidz * K + 0 ... TK;

      // pointers to operands
      int offa[SHAPE_A] = rk[BROADCAST_AK] * STRIDE_AK + rm[BROADCAST_AM] * STRIDE_AM;
      int offb[SHAPE_B] = rk[BROADCAST_BK] * STRIDE_BK + rn[BROADCAST_BN] * STRIDE_BN;
      TYPE* pa[SHAPE_A] = A + offa;
      TYPE* pb[SHAPE_B] = B + offb;

      // prefetches operands
      bool checka[SHAPE_A] = rk[BROADCAST_AK] < K;
      bool checkb[SHAPE_B] = rk[BROADCAST_BK] < K;
      TYPE a[SHAPE_A] = checka ? *pa : 0;
      TYPE b[SHAPE_B] = checkb ? *pb : 0;

      // reduction loop
      float acc[TM, TN] = 0;
      for(int k = K; k > 0; k -= TK){
        bool checka[SHAPE_A] = k > TK;
        bool checkb[SHAPE_B] = k > TK;
        pa += TK * STRIDE_AK;
        pb += TK * STRIDE_BK;
        acc += USEA @ USEB;
        a = *?(checka)pa;
        b = *?(checkb)pb;
      }
      acc = acc * alpha;
      TYPE c[TM, TN] = acc;

      // epilogue
      int rxm[TM] = pidm * TM + 0 ... TM;
      int rxn[TN] = pidn * TN + 0 ... TN;
      int offc[TM, TN] = rxm[:, newaxis] * ldc + rxn[newaxis, :];
      TYPE* pc[TM, TN] = C + offc;
      bool checkc[TM, TN] = (rxm[:, newaxis] < M) && (rxn[newaxis, :] < N);

#if (TZ==1)
      *?(checkc) pc = c;
#else
      // accumulate partial result using spin-locks
      int *plock  = locks + pid;
      int *pcount = plock + get_num_programs(0) * get_num_programs(1);
      for(int repeat = 1; repeat == 1; repeat = atomic_cas(plock, 0, 1));
      int count = *pcount;
      if(count == 0)
        *?(checkc) pc = c;
      else
        *?(checkc) pc = c + *?(checkc)pc;
      atomic_xchg(pcount, (count + 1) % TZ);
      atomic_xchg(plock, 0);
#endif
}
)";

}
