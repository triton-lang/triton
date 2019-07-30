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
  return  std::tie(N_, S_, C_, BS_, nlocks_, ab_ty_, c_ty_)
        < std::tie(y->N_, y->S_, y->C_, y->BS_, y->nlocks_, y->ab_ty_, y->c_ty_);
}

std::vector<params_t> dot::search_space() const {
  throw std::runtime_error("not implemented");
}

params_t dot::heuristics() const {
  throw std::runtime_error("not implemented");
}

base * dot::clone() const {
  return new dot(*this);
}

dot::dot(int32_t N, int32_t K, int32_t S, int32_t C,
         const std::string& ty, int32_t BS, int32_t nlocks):
    base("bsdot"),
    N_(N), K_(K), S_(S), C_(C),
    ab_ty_(ty), c_ty_(ty),
    BS_(BS), nlocks_(nlocks) {
}

void dot::init_impl(driver::stream *stream, driver::cu_module *module) {
//  int32_t TM = info.globals["TM"];
//  size_t grid_0 = (N_ + TM - 1) / TM;
//  if(nlocks_){
//    locks_ = triton::driver::buffer::create(stream->context(), grid_0 * nlocks_ * 2 * 4);
//    ((driver::cu_buffer*)locks_)->set_zero(stream, grid_0 * nlocks_ * 2 * 4);
//  }
}

void dot::deinit_impl() {
//  if(locks_)
//    delete locks_;
}

void dot::enqueue_impl(driver::stream *stream, driver::kernel *kernel,
                       std::vector<driver::buffer *> args, runtime::launch_information info) {
  driver::buffer *a = args[0];
  driver::buffer *b = args[1];
  driver::buffer *c = args[2];
  driver::buffer *lut = args[3];
  driver::buffer *locks = args[4];
  int32_t lda = N_;
  int32_t ldc = N_;
  kernel->setArg(0, a);
  kernel->setArg(1, b);
  kernel->setArg(2, c);
  kernel->setArg(3, lda);
  kernel->setArg(4, ldc);
  kernel->setArg(5, N_);
  kernel->setArg(6, lut);
  kernel->setArg(7, locks);
  kernel->setArg(8, nlocks_);
  int32_t TM = info.globals["TM"];
  size_t grid_0 = (N_ + TM - 1) / TM;
  size_t grid_1 = S_;
  std::cout << N_ << " " << grid_0 << std::endl;
  if(nlocks_){
//    locks_ = triton::driver::buffer::create(stream->context(), grid_0 * nlocks_ * 2 * 4);
    ((driver::cu_buffer*)locks)->set_zero(stream, grid_0 * nlocks_ * 2 * 4);
  }
  stream->enqueue(kernel, {grid_0, grid_1, 1}, {info.num_threads, 1, 1});
}

void dot::triton_c_src(std::ostream &os) const {
  std::string result =

  R"(
  const tunable int32 TM = {64};
  const tunable int32 TN = {)" + std::to_string(BS_) + R"(};
  const tunable int32 TK = {)" + std::to_string(BS_) + R"(};

  void bsdot(restrict read_only align(16) )" + ab_ty_ + R"( *A,
              restrict read_only align(16) )" + ab_ty_ + R"( *B,
              )" + c_ty_ + R"(* C,
              int32 lda, int32 ldc, int32 N,
              int32* lut, int32* locks, int32 nlocks){
    int32 ridx = get_range_id(0);
    int32 ridy = get_range_id(1);
    fp32 acc[TM, TN] = 0;
    int32 rxa[TM] = ridx * TM + (0 ... TM);
    int32 ryb[TN] = 0 ... TN;
    int32 rka[TK] = 0 ... TK;
    int32 rkb[TK] = 0 ... TK;
    int32 offa[TM, TK] = rxa[:, newaxis] + rka[newaxis, :]*lda;
    int32 offb[TK, TN] = ryb[newaxis, :] + rkb[:, newaxis]*TK;
    int32 *header = lut + ridy * 4;
    int32 offset = *(header + 0);
    int32 K      = *(header + 1);
    int32 column = *(header + 2);
    int32 lockid   = *(header + 3);
    int32 *plut   = lut + offset * 2;
    for(int32 k = K; k > 0; k = k - 1){
       int32 ak = *(plut + 0);
       int32 bk = *(plut + 1);
       )" + ab_ty_ + R"(* pa[TM, TK] = A + offa + ak * TK * lda;
       )" + ab_ty_ + R"(* pb[TK, TN] = B + offb + bk * TK * TN;
       )" + ab_ty_ + R"( a[TM, TK] = *pa;
       )" + ab_ty_ + R"( b[TK, TN] = *pb;
       acc = dot(a, b, acc);
       plut = plut + 2;
    }
    int32 rxc[TM] = ridx * TM + (0 ... TM);
    int32 ryc[TN] = column * TN + (0 ... TN);
    )" + c_ty_ + R"(" c[TM, TN] = acc;
    )" + c_ty_ + R"(* pc[TM, TN] = C + rxc[:, newaxis] + ryc[newaxis, :]*ldc;
    int1 checkc[TM, TN] = (rxc < N)[:, newaxis];
    if(lockid == 0){
      @checkc *pc = c;
    }
    else{
      int32 *plock = locks + ridx*nlocks + lockid - 1;
      int32 *pcount = plock + get_num_program(0)*nlocks;
      while(__atomic_cas(plock, 0, 1));
      int32 count = *pcount;
      if(count == 0) {
        @checkc *pc = c;
      }
      else {
        @checkc *pc = c + *pc;
      }
      *pcount = 1;
      __atomic_exch(plock, 0);
    }
  })";

  os << result;
}

}
}
}
