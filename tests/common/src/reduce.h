namespace src {

    const char *reduce1d =
R"(
void reduce1d(TYPE * X __noalias __readonly __aligned(16),
              TYPE * Y __noalias __readonly __aligned(16),
              int S0) {
  int pid0 = get_program_id(0);
  int rs0[TS0] = pid0 * TS0 + 0 ... TS0;
  TYPE* px[TS0] = X + rs0;
  *Y = (*px)[RED];
}
)";


    const char *reduce2d =
R"(
void reduce2d(TYPE * X __noalias __readonly __aligned(16),
            TYPE * Y __noalias __writeonly __aligned(16),
            int S0, int S1) {
  int pid0 = get_program_id(0);
  int pid1 = get_program_id(1);
  int rs0[TS0] = pid0 * TS0 + 0 ... TS0;
  int rs1[TS1] = pid1 * TS1 + 0 ... TS1;
  TYPE* px[TS0, TS1] = X + rs0[:, newaxis] * STRIDE_XS0
                         + rs1[newaxis, :] * STRIDE_XS1;
  TYPE* py[TY] = Y + RY * STRIDE_YS0;
  *py = (*px)[RED];
}
)";


  const char* reduce_nd[] = {reduce1d, reduce2d};


}
