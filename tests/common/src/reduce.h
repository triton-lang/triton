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
  TYPE* py[TY0] = Y + RY0 * STRIDE_YS0;
  *py = (*px)[RED];
}
)";

    const char *reduce3d =
R"(
void reduce2d(TYPE * X __noalias __readonly __aligned(16),
              TYPE * Y __noalias __writeonly __aligned(16),
              int S0, int S1, int S2) {
  int pid0 = get_program_id(0);
  int pid1 = get_program_id(1);
  int pid2 = get_program_id(2);
  int rs0[TS0] = pid0 * TS0 + 0 ... TS0;
  int rs1[TS1] = pid1 * TS1 + 0 ... TS1;
  int rs2[TS2] = pid2 * TS2 + 0 ... TS2;
  // input pointers
  TYPE* px[TS0, TS1, TS2] = X + rs0[:, newaxis, newaxis] * STRIDE_XS0
                              + rs1[newaxis, :, newaxis] * STRIDE_XS1
                              + rs2[newaxis, newaxis, :] * STRIDE_XS2;
  // output pointers
  TYPE* py[TY0, TY1] = Y + RY0[:, newaxis] * STRIDE_YS0
                         + RY1[newaxis, :] * STRIDE_YS1;
  // write-back
  *py = (*px)[RED];
}
)";

  const char* reduce_nd[] = {reduce1d, reduce2d, reduce3d};


}
