namespace src {

    const char *copy1d =
R"(
void copy1d(TYPE * X __noalias __readonly __aligned(16),
            TYPE * Y __noalias __readonly __aligned(16),
            int N) {
  int ridm = get_program_id(0);
  int rm[TN] = ridm * TN + 0 ... TN;
  TYPE* px[TN] = X + rm;
  TYPE* py[TN] = Y + rm;
  *py = *px;
}
)";


    const char *copy2d =
R"(
#if ORDER == ROWMAJOR
#define STRIDE_XM ldx
#define STRIDE_XN 1
#define STRIDE_YM ldy
#define STRIDE_YN 1
#else
#define STRIDE_XM 1
#define STRIDE_XN ldx
#define STRIDE_YM 1
#define STRIDE_YN ldy
#endif

void copy2d(TYPE * X __noalias __readonly __aligned(16),
            TYPE * Y __noalias __writeonly __aligned(16),
            int M, int N,
            int ldx __multipleof(8),
            int ldy __multipleof(8)) {
  int ridm = get_program_id(0);
  int ridn = get_program_id(1);
  int rm[TM] = ridm * TM + 0 ... TM;
  int rn[TN] = ridn * TN + 0 ... TN;
  TYPE* px[TM, TN] = X + rm[:, newaxis] * ldx + rn[newaxis, :] ;
  TYPE* py[TM, TN] = Y + rm[:, newaxis]  + rn[newaxis, :] * ldy;
  *py = *px;
}
)";

}
