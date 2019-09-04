import tensorflow as tf
import triton
import numpy as np

src = '''
  #if AT == 1
  #define USE_A         ^a
  #define STRIDE_AK     lda
  #define STRIDE_AM     1
  #define BROADCAST_AK  :, newaxis
  #define BROADCAST_AM  newaxis, :
  #define SHAPE_A       TK, TM
  #else
  #define USE_A         a
  #define STRIDE_AK     1
  #define STRIDE_AM     lda
  #define BROADCAST_AK  newaxis, :
  #define BROADCAST_AM  :, newaxis
  #define SHAPE_A       TM, TK
  #endif

  #if BT == 1
  #define USE_B         ^b
  #define STRIDE_BK     1
  #define STRIDE_BM     ldb
  #define BROADCAST_BN  newaxis, :
  #define BROADCAST_BK  :, newaxis
  #define SHAPE_B       TN, TK
  #else
  #define USE_B         b
  #define STRIDE_BK     ldb
  #define STRIDE_BM     1
  #define BROADCAST_BN  :, newaxis
  #define BROADCAST_BK  newaxis, :
  #define SHAPE_B       TK, TN
  #endif

  void dot (TYPE* A __readonly  __noalias __align(16),
            TYPE* B __readonly  __noalias __align(16),
            TYPE* C __writeonly __noalias __align(16),
            int lda, int ldb, int ldc,
            int N, int* lut,
            int* locks, int nlocks) {
    int ridx = get_program_id(0);
    float c[TM, TN] = 0;
    int rka[TK] = 0 ... TK;
    int rkb[TK] = 0 ... TK;
    // load LUT header
    int *header = lut + get_program_id(1) * 4;
    int offset = *(header + 0);
    int K      = *(header + 1);
    int column = *(header + 2);
    int lockid = *(header + 3);
    int *plut   = lut + offset * 2;
    int offx = ridx;
    int offy = 0;
    // compute x, y offsets
    int rxa[TM] = offx * TM + (0 ... TM);
    int ryb[TN] = offy * TN + (0 ... TN);
    // bounds checking
    bool checka[SHAPE_A] = (rxa < N)[:, newaxis];
    bool checkb[SHAPE_B] = 1;
    // base offset
    int offa[SHAPE_A] = rxa[BROADCAST_AM] * STRIDE_AM + rka[BROADCAST_AK] * STRIDE_AK;
    int offb[SHAPE_B] = ryb[BROADCAST_BN] * STRIDE_BN + rkb[BROADCAST_BK] * STRIDE_BK;
    for(int k = K; k > 0; k -= 1) {
       // fetch block indices
       int ak = *(plut + 0);
       int bk = *(plut + 1);
       lut += 2;
       // compute pointers to blocks
       TYPE* pa[SHAPE_A] = A + offa + ak * TK * lda;
       TYPE* pb[SHAPE_B] = B + offb + bk * TK * TN;
       // load blocks
       TYPE   a[SHAPE_A] = checka ? *pa : 0;
       TYPE   b[SHAPE_B] = *pb;
       // multiply blocks
       c += USE_A @ USE_B;
    }
    int   rxc[TM]    = ridx   * TM + (0 ... TM);
    int   ryc[TN]    = column * TN + (0 ... TN);
    TYPE* pc[TM, TN] = C + rxc[:, newaxis] + ryc[newaxis, :]*ldc;
    bool  checkc[TM, TN] = (rxc < N)[:, newaxis];
    if(lockid == 0) {
      *?(checkc) pc = c;
    }
    else {
      int *plock = locks + ridx*nlocks + lockid - 1;
      int *pcount = plock + get_num_program(0)*nlocks;
      while(atomic_cas(plock, 0, 1));
      int count = *pcount;
      if(count == 0)
        *?(checkc) pc = c;
      else
        *?(checkc) pc = c + *pc;
      atomic_exch(pcount, 1);
      atomic_exch(plock, 0);
    }
  }
'''

# std::string dot::triton_c_src_dw() const {
#   bool AT = (op_ == WGRAD);
#   bool BT = (op_ == FPROP);
#   std::string usea = AT ? "trans(a)" : "a";
#   std::string useb = BT ? "trans(b)" : "b";
#   std::string sizea = AT ? "TK, TM" : "TM, TK";
#   std::string sizeb = BT ? "TN, TK" : "TK, TN";
#   std::string bca0 = AT ? "newaxis, :" : ":, newaxis";
#   std::string bca1 = AT ? ":, newaxis" : "newaxis, :";
#   std::string bcb0 = BT ? ":, newaxis" : "newaxis, :";
#   std::string bcb1 = BT ? "newaxis, :" : ":, newaxis";
#   std::string lda0 = AT ? "*lda" : "";
#   std::string lda1 = AT ? "" : "*lda";
#   std::string ldb0 = BT ? ""  : "*ldb";
#   std::string ldb1 = BT ? "*ldb" : "" ;
#   std::string result =
#   R"(
#   const tunable int TM = {)" + std::to_string(BS_) + R"(};
#   const tunable int TN = {)" + std::to_string(BS_) + R"(};
#   const tunable int TK = {32};
#   void bsdot(restrict read_only align(16) )" + ab_ty_ + R"( *A,
#              restrict read_only align(16) )" + ab_ty_ + R"( *B,
#              )" + c_ty_ + R"(* C,
#              int lda, int ldb, int ldc,
#              int N, int* lut,
#              int* locks, int nlocks) {
#     int ridx = get_range_id(0);
#     float acc[TM, TN] = 0;
#     int rka[TK] = 0 ... TK;
#     int rkb[TK] = 0 ... TK;
#     int *header = lut + ridx * 2;
#     int offx = *(header + 0);
#     int offy = *(header + 1);
#     int rxa[TM] = offx*TM + (0 ... TM);
#     int ryb[TN] = offy*TN + (0 ... TN);
#     bool checka[TK, TM] = (rka < N)[:, newaxis];
#     bool checkb[TK, TN] = (rkb < N)[:, newaxis];
#     int offa[)" + sizea + "] = rxa[" + bca0 + "]" + lda0 + " + rka[" + bca1 + "]" + lda1 + R"(;
#     int offb[)" + sizeb + "] = ryb[" + bcb0 + "]" + ldb0 + " + rkb[" + bcb1 + "]" + ldb1 + R"(;
#     )" + ab_ty_ + " * pa[" + sizea + R"(] = A + offa;
#     )" + ab_ty_ + " * pb[" + sizeb + R"(] = B + offb;
#     )" + ab_ty_ + "   a[" + sizea + R"(] = checka ? *pa : 0;
#     )" + ab_ty_ + "   b[" + sizeb + R"(] = checkb ? *pb : 0;
#     for(int k = N; k > 0; k = k - TK) {
#        acc = dot()" + usea + ", " + useb + R"(, acc);
#        pa = pa + TK)" + lda1 + R"(;
#        pb = pb + TK)" + ldb1 + R"(;
#        a = checka ? *pa : 0;
#        b = checkb ? *pb : 0;
#     }
#     int rxc[TM] = (0 ... TM);
#     int ryc[TN] = (0 ... TN);
#     )" + c_ty_ + R"( c[TM, TN] = acc;
#     )" + c_ty_ + R"(* pc[TM, TN] = C + rxc[:, newaxis]*TM + ryc[newaxis, :] + ridx*TM*TN;
#     *pc = c;
#   })";