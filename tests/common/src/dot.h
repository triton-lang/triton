namespace src {

    const char *dot =
R"(
#ifdef AT
#define USEA ^a
#else
#define USEA a
#endif

#ifdef BT
#define USEB ^b
#else
#define USEB b
#endif

void dot(TYPE * A __noalias __readonly __aligned(16),
         TYPE * B __noalias __readonly __aligned(16),
         TYPE * C __noalias __readonly __aligned(16),
         int M, int N, int K,
         int lda __multipleof(8),
         int ldb __multipleof(8),
         int ldc) {
  int ridx = get_program_id(0);
  int ridy = get_program_id(1);
  int rxa[TM] = ridx * TM + 0 ... TM;
  int ryb[TN] = ridy * TN + 0 ... TN;
  int rka[TK] = 0 ... TK;
  int rkb[TK] = 0 ... TK;
  float xc[TM, TN] = 0;
#ifdef AT
  TYPE* pa[TK, TM] = A + rka[:, newaxis] + rxa[newaxis, :]*lda;
  bool checka[TK, TM] = rka[:, newaxis] < TK;
  TYPE a[TK, TM] = checka ? *pa : 0;
#else
  TYPE* pa[TM, TK] = A + rka[newaxis, :]*lda + rxa[:, newaxis];
  bool checka[TM, TK] = rka[newaxis, :] < TK;
  TYPE a[TM, TK] = checka ? *pa : 0;
#endif
#ifdef BT
  TYPE* pb[TN, TK] = B + rkb[newaxis, :]*ldb + ryb[:, newaxis];
  bool checkb[TN, TK] = rkb[newaxis, :] < TK;
  TYPE b[TN, TK] = checkb ? *pb : 0;
#else
  TYPE* pb[TK, TN] = B + rkb[:, newaxis] + ryb[newaxis, :]*ldb;
  bool checkb[TK, TN] = rkb[:, newaxis] < TK;
  TYPE b[TK, TN] = checkb ? *pb : 0;
#endif
  for(int k = K; k > 0; k = k - TK){
    xc = USEA @ USEB + xc;
#ifdef AT
    pa = pa + TK;
#else
    pa = pa + TK*lda;
#endif
#ifdef BT
    pb = pb + TK*ldb;
#else
    pb = pb + TK;
#endif
    checka = k > TK;
    checkb = k > TK;
    a = checka ? *pa : 0;
    b = checkb ? *pb : 0;
  }
  int rxc[TM] =  ridx * TM + (0 ... TM);
  int ryc[TN] =  ridy * TN + (0 ... TN);
  TYPE* pc[TM, TN] = C + ryc[newaxis, :]*ldc + rxc[:, newaxis];
  TYPE c[TM, TN] = xc;
  bool checkc0[TM] = rxc < M;
  bool checkc1[TN] = ryc < N;
  bool checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];
  *?(checkc) pc = c;
}
)";

}
