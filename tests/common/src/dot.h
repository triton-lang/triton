namespace src {

    const char *dot =
R"(
__global__ void dot(TYPE * A __noalias __readonly __aligned(16),
                    TYPE * B __noalias __readonly __aligned(16),
                    TYPE * C __noalias __aligned(16),
                    float alpha,
                    int M, int N, int K,
                    int lda __multipleof(8),
                    int ldb __multipleof(8),
                    int ldc __multipleof(8)) {
      // prologue
      int ridx = get_program_id(0);
      int ridy = get_program_id(1);
      int gridx = M / TM;
      int gridy = N / TN;
      int rid = ridx + ridy * gridx;
      ridx = rid / gridy;
      ridy = rid % gridy;
      int rm[TM] = ridx * TM + 0 ... TM;
      int rn[TN] = ridy * TN + 0 ... TN;
      int rk[TK] = 0 ... TK;

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
      float c[TM, TN] = 0;
      for(int k = K; k > 0; k -= TK){
        c += USEA @ USEB;
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
      int offc[TM, TN] = rxm[:, newaxis] * ldc + rxn[newaxis, :];
      TYPE* pc[TM, TN] = C + offc;
      bool checkc[TM, TN] = (rxm[:, newaxis] < M) && (rxn[newaxis, :] < N);
      *?(checkc)pc = (TYPE[TM, TN])c;
}
)";

}
