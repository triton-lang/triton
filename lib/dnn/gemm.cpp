#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"
#include "triton/dnn/gemm.h"
#include <string>

namespace triton{
namespace dnn{


void gemm::init(driver::stream* stream, driver::buffer* locks) {
  std::vector<int32_t> hlocks(2048, 0);
  stream->write(locks, false, 0, hlocks);
}

void gemm::set_arg(driver::kernel *kernel,
                      driver::buffer *a, driver::buffer *b, driver::buffer *c,
                      int32_t M, int32_t N, int32_t K,
                      driver::buffer *locks, int32_t grid_0, int32_t grid_1) {
  kernel->setArg(0, a);
  kernel->setArg(1, b);
  kernel->setArg(2, c);
  kernel->setArg(3, M);
  kernel->setArg(4, N);
  kernel->setArg(5, K);
  kernel->setArg(6, M);
  kernel->setArg(7, N);
  kernel->setArg(8, M);
  kernel->setArg(9, locks);
  kernel->setArg(10, grid_0);
  kernel->setArg(11, grid_1);
}

std::vector<unsigned> gemm::default_params(bool AT, bool BT) {
  if(AT && BT)
    return {32, 64, 32, 64, 16, 8, 2, 2, 4, 2, 8, 4, 2, 1};
  else if(AT && !BT)
    return {32, 64, 32, 64, 16, 8, 2, 2, 4, 2, 8, 4, 2, 1};
  else if(!AT && BT)
    return {16, 2, 64, 16, 2, 64, 16, 8, 2, 2, 8, 8, 8, 1};
  else
    return {16, 2, 128, 32, 32, 32, 4, 2, 2, 8, 8, 4, 2, 1};
}

std::string gemm::src(bool AT, bool BT) {
  std::string AS0 = "TM", AS1 = "TK";
  std::string BS0 = "TK", BS1 = "TN";
  std::string bca0 = "[newaxis, :]", bca1 = "[:, newaxis]";
  std::string bcb0 = "[:, newaxis]", bcb1 = "[newaxis, :]";
  std::string lda0 = "*lda", lda1 = "";
  std::string ldb0 = "", ldb1 = "*ldb";
  std::string usea = AT ? "trans(a)" : "a";
  std::string useb = BT ? "trans(b)" : "b";
  if(AT){
    std::swap(AS0, AS1);
    std::swap(bca0, bca1);
    std::swap(lda0, lda1);
  }
  if(BT){
    std::swap(BS0, BS1);
    std::swap(bcb0, bcb1);
    std::swap(ldb0, ldb1);
  }
  std::string res =
R"(
const tunable int32 TM = {16, 32, 64, 128};
const tunable int32 TN = {16, 32, 64, 128};
const tunable int32 TK = {8};
const tunable int32 GZ = {1};

void matmul(restrict read_only fp32 *A, restrict read_only fp32 *B, fp32 *C,
           int32 M, int32 N, int32 K,
           int32 lda, int32 ldb, int32 ldc,
           int32 *locks, int32 grid0, int32 grid1) {
  int32 rxa[TM] = get_global_range[TM](0);
  int32 ryb[TN] = get_global_range[TN](1);
  int32 rz = get_global_range[1](2);
  int32 rka[TK] = 0 ... TK;
  int32 rkb[TK] = 0 ... TK;
  fp32 c[TM, TN] = 0;
  int32 div = K / GZ;
  int32 rem = K % GZ;
  K = select(rz < rem, div - 1, div);
  int32 offk = select(rz < rem, rz*(div + 1), rz*div + rem);
  fp32* pa[)" + AS0 + ", " + AS1 + "] = A + (offk + rka" + bca0 + ")" + lda0 + " + rxa" + bca1 + lda1 + R"(;
  fp32* pb[)" + BS0 + ", " + BS1 + "] = B + (offk + rkb" + bcb0 + ")" + ldb0 + " + ryb" + bcb1 + ldb1 + R"(;
  fp32 a[)" + AS0 + ", " + AS1 + R"(] = *pa;
  fp32 b[)" + BS0 + ", " + BS1 + R"(] = *pb;
  int32 last_a = ((M*K - 1) - (TM*TK + 1)) / lda;
  int32 last_b = ((K*N - 1) - (TN*TK + 1)) / ldb;
  last_a = last_a / TK * TK;
  last_b = last_b / TK * TK;
  int32 bound = K - max(last_a, last_b);
  for(int32 k = K; k > bound; k = k - TK){
    c = dot()" + usea + ", " + useb + R"(, c);
    pa = pa + TK)" + lda0 + R"(;
    pb = pb + TK)" + ldb0 + R"(;
    a = *pa;
    b = *pb;
  }
  int32 rxc[TM] = get_global_range[TM](0);
  int32 ryc[TN] = get_global_range[TN](1);
  for(int32 k = bound; k > 0; k = k - 1){
    int1 checka[TM, 1] = rxc[:, newaxis] < M;
    int1 checkb[TN, 1] = ryc[:, newaxis] < N;
    fp32* pa[TM, 1] = A + (offk + K - k))" + lda0 + " + rxc[:, newaxis]" + lda1 + R"(;
    fp32* pb[TN, 1] = B + (offk + K - k))" + ldb0 + " + ryc[:, newaxis]" + ldb1 + R"(;
    fp32 a[TM, 1] = checka ? *pa : 0;
    fp32 b[TN, 1] = checkb ? *pb : 0;
    c = dot(a, trans(b), c);
  }
  int32 ridx = get_range_id(0);
  int32 ridy = get_range_id(1);
  int1 checkc0[TM] = rxc < M;
  int1 checkc1[TN] = ryc < N;
  int1 checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];
  fp32* pc[TM, TN] = C + ryc[newaxis, :]*ldc + rxc[:, newaxis];
  int32 *plock = locks + ridx + ridy*grid0;
  while(__atomic_cas(plock, 0, 1));
  int32 *pcount = plock + grid0*grid1;
  int32 count = *pcount;
  int32 countp1 = select(count == GZ - 1, 0, count + 1);
  if(count == 0) {
    @checkc *pc = c;
    *pcount = countp1;
  }
  else {
    @checkc *pc = c + *pc;
    *pcount = countp1;
  }
  __atomic_cas(plock, 1, 0);
}
)";
  return res;
}

}
}
