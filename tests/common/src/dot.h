namespace src {

    const char *dot =
R"(
void dot(TYPE * A, TYPE * B, TYPE * C,
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
  float c[TM, TN] = 0;
  // pointers to operands
  TYPE* pa[SHAPE_A] = A + rk[BROADCAST_AK] * STRIDE_AK + rm[BROADCAST_AM] * STRIDE_AM;
  TYPE* pb[SHAPE_B] = B + rk[BROADCAST_BK] * STRIDE_BK + rn[BROADCAST_BN] * STRIDE_BN;
  // prefetches operands
  TYPE a[SHAPE_A] = *pa;
  TYPE b[SHAPE_B] = *pb;
  // reduction loop
  for(int k = K; k > 0; k-= TK){
    c += USEA @ USEB;
    pa = pa + TK * STRIDE_AK;
    pb = pb + TK * STRIDE_BK;
    bool checka[SHAPE_A] = k > TK;
    bool checkb[SHAPE_B] = k > TK;
    a = checka ? *pa : 0;
    b = checkb ? *pb : 0;
  }
  // epilogue
  TYPE* pc[TM, TN] = C + rm[:, newaxis] + rn[newaxis, :] * ldc;
  *pc = c;
}
)";

}
