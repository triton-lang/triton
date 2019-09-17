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
void copy2d(TYPE * X __noalias __readonly __aligned(16),
            TYPE * Y __noalias __writeonly __aligned(16),
            int M __multipleof(8),
            int N __multipleof(8)) {
  int ridm = get_program_id(0);
  int ridn = get_program_id(1);
  int rm[TM] = ridm * TM + 0 ... TM;
  int rn[TN] = ridn * TN + 0 ... TN;
  TYPE* px[TM, TN] = X + rm[:, newaxis] * STRIDE_XM + rn[newaxis, :] * STRIDE_XN;
  TYPE* py[TM, TN] = Y + rm[:, newaxis] * STRIDE_YM + rn[newaxis, :] * STRIDE_YN;
  *py = *px;
}
)";

}
