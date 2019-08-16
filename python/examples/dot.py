import libtriton

src = """
const tunable int TM = {128};
const tunable int TN = {128};
const tunable int TK = {32};

void matmul(restrict read_only align(16) half *A,
            restrict read_only align(16) half *B,
            restrict read_only align(16) half *C,
            int M, int N, int K,
            multiple_of(8) int lda, multiple_of(8)" int ldb, int ldc) {
  int ridx = get_range_id(0);
  int ridy = get_range_id(1);
  int rxa[TM] = ridx * TM + (0 ... TM);
  int ryb[TN] = ridy * TN + (0 ... TN);
  int rka[TK] = 0 ... TK;
  int rkb[TK] = 0 ... TK;
  float xc[TM, TN] = 0;
  half* pa[TM, TK] = A + rka[newaxis, :]*lda + rxa[:, newaxis];
  half* pb[TN, TK] = B + rkb[newaxis, :]*ldb + ryb[:, newaxis];
  half a[TM, TK] = *pa;
  half b[TN, TK] = *pb;
  for(int k = K; k > 0; k = k - TK){
    xc = dot(a, trans(b), xc);
    pa = pa + TK*lda;
    pb = pb + TK*ldb;
    a = *pa;
    b = *pb;
  }
  int rxc[TM] =  ridx * TM + (0 ... TM);
  int ryc[TN] =  ridy * TN + (0 ... TN);
  half* pc[TM, TN] = C + ryc[newaxis, :]*ldc + rxc[:, newaxis];
  half c[TM, TN] = xc;
  bool checkc0[TM] = rxc < M;
  bool checkc1[TN] = ryc < N;
  bool checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];
  @checkc *pc = c;
}
"""

print(libtriton.make_tensorflow_src(src, [2], '(M + #TM - 1)/#TM, (N + #TN - 1)/#TN, 1'))