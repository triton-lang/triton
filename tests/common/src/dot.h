namespace src {

    const char *dot =
R"(
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
      K           = K / TZ;
      int rk[TK]  = ridz * K + 0 ... TK;

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
        acc += USEA @ USEB;
        bool checka[SHAPE_A] = k > TK;
        bool checkb[SHAPE_B] = k > TK;
        pa += TK * STRIDE_AK;
        pb += TK * STRIDE_BK;
        a = *?(checka)pa;
        b = *?(checkb)pb;
      }
      acc = acc * alpha;
      TYPE c[TM, TN] = acc;

      // epilogue
      int rxm[TM] = get_program_id(0) * TM + 0 ... TM;
      int rxn[TN] = get_program_id(1) * TN + 0 ... TN;
      int offc[TM, TN] = rxm[:, newaxis] * ldc + rxn[newaxis, :];
      TYPE* pc[TM, TN] = C + offc;
      bool checkc[TM, TN] = (rxm[:, newaxis] < M) && (rxn[newaxis, :] < N);

#if (TZ==1)
      *?(checkc) pc = c;
#else
      // accumulate partial result using spin-locks
      int *plock  = locks + rid;
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
