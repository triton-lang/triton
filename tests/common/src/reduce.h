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
  int rm[TS0] = ridm * TS0 + 0 ... TS0;
  int rn[TS1] = ridn * TS1 + 0 ... TS1;
  TYPE* px[TS0, TS1] = X + rm[:, newaxis] + rn[newaxis, :] * ldx;
  TYPE* py[TY] = Y + RY;
  *py = (*px)[RED];
}
)";

}
