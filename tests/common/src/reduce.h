namespace src {

    const char *reduce1d =
R"(
void reduce1d(TYPE * X __noalias __readonly __aligned(16),
              TYPE * Y __noalias __readonly __aligned(16),
              int N) {
}
)";


    const char *reduce2d =
R"(
void reduce2d(TYPE * X __noalias __readonly __aligned(16),
            TYPE * Y __noalias __writeonly __aligned(16),
            int M, int N, int ldx) {
  int ridm = get_program_id(0);
  int ridn = get_program_id(1);
  int rm[TM] = ridm * TM + 0 ... TM;
  int rn[TN] = ridn * TN + 0 ... TN;
  TYPE* px[TM, TN] = X + rm[:, newaxis] + rn[newaxis, :] * ldx;
  TYPE* py[TM, TN] = Y + rm[:, newaxis];
  *py = (*px)[:, +];
}
)";

}
