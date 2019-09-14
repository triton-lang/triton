namespace src {

    const char *dot =
R"(
#if AT == 1
#define USEA ^a
#define STRIDE_AK 1
#define STRIDE_AM lda
#define BROADCAST_AK :, newaxis
#define BROADCAST_AM newaxis, :
#define SHAPE_A TK, TM
#else
#define USEA a
#define STRIDE_AK lda
#define STRIDE_AM 1
#define BROADCAST_AK newaxis, :
#define BROADCAST_AM :, newaxis
#define SHAPE_A TM, TK
#endif

#if BT == 1
#define USEB ^b
#define STRIDE_BK ldb
#define STRIDE_BN 1
#define BROADCAST_BK newaxis, :
#define BROADCAST_BN :, newaxis
#define SHAPE_B TN, TK
#else
#define USEB b
#define STRIDE_BK 1
#define STRIDE_BN ldb
#define BROADCAST_BK :, newaxis
#define BROADCAST_BN newaxis, :
#define SHAPE_B TK, TN
#endif

void dot(TYPE * A, TYPE * B, TYPE * C,
         int M, int N, int K,
         int lda __multipleof(8),
         int ldb __multipleof(8),
         int ldc) {
  // prologue
  int ridx = get_program_id(0);
  int ridy = get_program_id(1);
  int rxa[TM] = ridx * TM + 0 ... TM;
  int ryb[TN] = ridy * TN + 0 ... TN;
  int rka[TK] = 0 ... TK;
  int rkb[TK] = 0 ... TK;
  float c[TM, TN] = 0;
  // pointers to operands
  TYPE* pa[SHAPE_A] = A + rka[BROADCAST_AK] * STRIDE_AK + rxa[BROADCAST_AM] * STRIDE_AM;
  TYPE* pb[SHAPE_B] = B + rkb[BROADCAST_BK] * STRIDE_BK + ryb[BROADCAST_BN] * STRIDE_BN;
  // prefetches operands
  TYPE a[SHAPE_A] = *pa;
  TYPE b[SHAPE_B] = *pb;
  // reduction loop
  for(int k = K; k > 0; k-= TK){
    c += USEA @ USEB;
    pa = pa + TK * STRIDE_AK;
    pb = pb + TK * STRIDE_BK;
    a = ((bool[SHAPE_A])(k > TK)) ? *pa : 0;
    b = ((bool[SHAPE_B])(k > TK)) ? *pb : 0;
  }
  // epilogue
  int rxc[TM] =  ridx * TM + 0 ... TM;
  int ryc[TN] =  ridy * TN + 0 ... TN;
  TYPE* pc[TM, TN] = C + rxc[:, newaxis] * ldc + ryc[newaxis, :];
  *pc = c;
}
)";

}
