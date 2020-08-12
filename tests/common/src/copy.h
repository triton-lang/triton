#ifndef _TRITON_TEST_SRC_COPY_H_
#define _TRITON_TEST_SRC_COPY_H_

namespace src {

    const char *copy1d =
R"(
void copy1d(TYPE * X __noalias __readonly __aligned(16),
            TYPE * Y __noalias __readonly __aligned(16),
            int S0 __retune) {
  int pid0 = get_program_id(0);
  int rs0[TS0] = pid0 * TS0 + 0 ... TS0;
  TYPE* px[TS0] = X + rs0;
  TYPE* py[TS0] = Y + rs0;
  *py = *px;
}
)";

    const char *copy2d =
R"(
void copy2d(TYPE * X __noalias __readonly __aligned(16),
            TYPE * Y __noalias __writeonly __aligned(16),
            int S0 __multipleof(8) __retune,
            int S1 __multipleof(8) __retune) {
  int pid0 = get_program_id(0);
  int pid1 = get_program_id(1);
  int rs0[TS0] = pid0 * TS0 + 0 ... TS0;
  int rs1[TS1] = pid1 * TS1 + 0 ... TS1;
  bool in_bounds[TS0, TS1] = rs0[:, newaxis] < S0 && rs1[newaxis, :] < S1;
  TYPE* px[TS0, TS1] = X + rs0[:, newaxis] * STRIDE_XS0 + rs1[newaxis, :] * STRIDE_XS1;
  TYPE* py[TS0, TS1] = Y + rs0[:, newaxis] * STRIDE_YS0 + rs1[newaxis, :] * STRIDE_YS1;
  *?(in_bounds)py = *?(in_bounds)px;
}
)";

    const char *copy3d =
R"(
void copy3d(TYPE * X __noalias __readonly __aligned(16),
            TYPE * Y __noalias __writeonly __aligned(16),
            int S0 __multipleof(8) __retune,
            int S1 __multipleof(8) __retune,
            int S2 __multipleof(8) __retune) {
  // program id
  int pid0 = get_program_id(0);
  int pid1 = get_program_id(1);
  int pid2 = get_program_id(2);
  // ranges
  int rs0[TS0] = pid0 * TS0 + 0 ... TS0;
  int rs1[TS1] = pid1 * TS1 + 0 ... TS1;
  int rs2[TS2] = pid2 * TS2 + 0 ... TS2;
  // X pointers
  TYPE* px[TS0, TS1, TS2] = X + rs0[:, newaxis, newaxis] * STRIDE_XS0
                              + rs1[newaxis, :, newaxis] * STRIDE_XS1
                              + rs2[newaxis, newaxis, :] * STRIDE_XS2;
  // Y pointers
  TYPE* py[TS0, TS1, TS2] = Y + rs0[:, newaxis, newaxis] * STRIDE_YS0
                              + rs1[newaxis, :, newaxis] * STRIDE_YS1
                              + rs2[newaxis, newaxis, :] * STRIDE_YS2;
  *py = *px;
}
)";

  const char* copy_nd[] = {copy1d, copy2d, copy3d};

}

#endif
