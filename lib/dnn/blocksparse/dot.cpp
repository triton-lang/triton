#include "triton/dnn/blocksparse/dot.h"

namespace triton{
namespace dnn{
namespace blocksparse{


size_t dot::num_flops() const {

}

bool dot::operator <(const base& other) const {
  auto *y = dynamic_cast<const dot*>(&other);
  if(!y)
    return true;
  return  std::tie(M_, N_, K_)
        < std::tie(y->M_, y->N_, y->K_);
}

std::vector<params_t> dot::search_space() const {

}

params_t dot::heuristics() const {

}

base * dot::clone() const {
  return new dot(*this);
}

dot::dot(int32_t M, int32_t N, int32_t K):
    base("bsdot"), M_(M), N_(N), K_(K) {
  ab_ty_ = "fp32";
  c_ty_ = "fp32";
}

void dot::enqueue_impl(driver::stream *stream, driver::kernel *kernel,
                       std::vector<driver::buffer *> args, runtime::launch_information info) {
  driver::buffer *a = args[0];
  driver::buffer *b = args[1];
  driver::buffer *c = args[2];
  driver::buffer *lut = args[3];
  int32_t lda = M_;
  int32_t ldc = M_;
  kernel->setArg(0, a);
  kernel->setArg(1, b);
  kernel->setArg(2, c);
  kernel->setArg(3, lda);
  kernel->setArg(4, ldc);
  kernel->setArg(5, lut);
  int32_t TM = info.globals["TM"];
  int32_t TN = info.globals["TN"];
  size_t grid_0 = (M_ + TM - 1) / TM;
  size_t grid_1 = (N_ + TN - 1) / TN;
  stream->enqueue(kernel, {grid_0, grid_1, 1}, {info.num_threads, 1, 1});
  stream->synchronize();
}

void dot::triton_c_src(std::ostream &os) const {
  std::string result =

  R"(
  const tunable int32 TM = {64, 128};
  const tunable int32 TN = {32};
  const tunable int32 TK = {32};

  void bsdot(restrict read_only align(16) )" + ab_ty_ + R"( *A,
              restrict read_only align(16) )" + ab_ty_ + R"( *B,
              fp32* C,
              int32 lda, int32 ldc,
              int32* lut_base){
    int32 ridx = get_range_id(0);
    int32 ridy = get_range_id(1);
    fp32 c[TM, TN] = 0;
    int32 rka[TK] = 0 ... TK;
    int32 rkb[TK] = 0 ... TK;
    int32 rxa[TM] = ridx * TM + (0 ... TM);
    int32 ryb[TN] = 0 ... TN;
    int32 offa[TM, TK] = rxa[:, newaxis] + rka[newaxis, :]*lda;
    int32 offb[TK, TN] = ryb[newaxis, :] + rkb[:, newaxis]*TK;
    int32 *header = lut_base + ridy * 4;
    int32 offset = *(header + 0);
    int32 K      = *(header + 1);
    int32 h2     = *(header + 2);
    int32 h3     = *(header + 3);
    int32 *lut   = lut_base + offset*2;
    for(int32 k = K; k > 0; k = k - 1){
       int32 ak = *(lut + 0);
       int32 bk = *(lut + 1);
       fp32* pa[TM, TK] = A + offa + ak * TK * lda;
       fp32* pb[TK, TN] = B + offb + bk * TK * TN;
       fp32 a[TM, TK] = *pa;
       fp32 b[TK, TN] = *pb;;
       c = dot(a, b, c);
       lut = lut + 2;
    }
    int32 rxc[TM] = ridx * TM + (0 ... TM);
    int32 ryc[TN] = ridy * TN + (0 ... TN);
    fp32* pc[TM, TN] = C + rxc[:, newaxis] + ryc[newaxis, :]*ldc;
    *pc = c;
  })";

  os << result;
}

}
}
}
