#include "triton/dnn/heuristics.h"
#include "triton/dnn/blocksparse/dot.h"

namespace triton{
namespace dnn{
namespace blocksparse{


size_t dot::num_flops() const {
  return 2.*nblocks_*BS_*BS_*N_;
}

std::vector<int64_t> dot::retune_params() const{
  return {N_, S_, C_, BS_, nlocks_, op_};
}

std::vector<params_t> dot::search_space() const {
  return bsdot_search_space(op_ == FPROP, BS_);
}

params_t dot::heuristics() const {
  return bsdot_heuristics(op_ == FPROP, BS_, N_, S_);
}

base * dot::clone() const {
  return new dot(*this);
}

dot::dot(int32_t N, int32_t K, int32_t S, int32_t C,
         const std::string& ty, int32_t BS, int32_t nlocks, int32_t nblocks, op_t op):
    base("bsdot"),
    N_(N), K_(K), S_(S), C_(C),
    ab_ty_(ty), c_ty_(ty),
    BS_(BS), nlocks_(nlocks), nblocks_(nblocks), op_(op){
}

void dot::init_impl(driver::stream *stream, driver::cu_module *module, triton::runtime::launch_information info) {
  int32_t TM = info.globals["TM"];
  size_t grid_0 = (N_ + TM - 1) / TM;
  if(nlocks_ && !locks_){
    locks_.reset(triton::driver::buffer::create(stream->context(), grid_0 * nlocks_ * 2 * 4));
    ((driver::cu_buffer*)locks_.get())->set_zero(stream, grid_0 * nlocks_ * 2 * 4);
  }
}

void dot::deinit_impl() {
}

void dot::enqueue_impl(driver::stream *stream, driver::kernel *kernel,
                       std::vector<driver::buffer *> args, runtime::launch_information info) {
  driver::buffer *a = args[0];
  driver::buffer *b = args[1];
  driver::buffer *c = args[2];
  driver::buffer *lut = args[3];
  int32_t lda = N_;
  int32_t ldc = N_;
  kernel->setArg(0, a);
  kernel->setArg(1, b);
  kernel->setArg(2, c);
  kernel->setArg(3, lda);
  kernel->setArg(4, ldc);
  kernel->setArg(5, N_);
  kernel->setArg(6, lut);
  kernel->setArg(7, locks_.get());
  kernel->setArg(8, nlocks_);
  int32_t TM = info.globals["TM"];
  size_t grid_0 = (N_ + TM - 1) / TM;
  size_t grid_1 = S_;
  if(nlocks_)
    ((driver::cu_buffer*)locks_.get())->set_zero(stream, grid_0 * nlocks_ * 2 * 4);
  stream->enqueue(kernel, {grid_0, grid_1, 1}, {info.num_threads, 1, 1});
}

driver::buffer* dot::get_locks() const {
  return locks_.get();
}

void dot::triton_c_src(std::ostream &os) const {
  std::string usea = (op_ == WGRAD) ? "trans(a)" : "a";
  std::string useb = (op_ == FPROP) ? "trans(b)" : "b";
  std::string sizea = "TM, TK";
  std::string sizeb = (op_ == FPROP) ? "TN, TK" : "TK, TN";
  std::string bca0 = ":, newaxis";
  std::string bca1 = "newaxis, :";
  std::string bcb0 = (op_ == FPROP) ? ":, newaxis" : "newaxis, :";
  std::string bcb1 = (op_ == FPROP) ? "newaxis, :" : ":, newaxis";
  std::string ldb0 = (op_ == FPROP) ? ""  : "*TK";
  std::string ldb1 = (op_ == FPROP) ? "*TK" : "" ;
  std::string result =
  R"(
  const tunable int TM = {16, 32, 64, 128};
  const tunable int TN = {)" + std::to_string(BS_) + R"(};
  const tunable int TK = {)" + std::to_string(BS_) + R"(};

  void bsdot(restrict read_only align(16) )" + ab_ty_ + R"( *A,
             restrict read_only align(16) )" + ab_ty_ + R"( *B,
             )" + c_ty_ + R"(* C,
             int lda, int ldc, int N,
             int* lut, int* locks, int nlocks){
    int ridx = get_range_id(0);
    int ridy = get_range_id(1);
    float acc[TM, TN] = 0;
    int rxa[TM] = ridx * TM + (0 ... TM);
    int ryb[TN] = 0 ... TN;
    int rka[TK] = 0 ... TK;
    int rkb[TK] = 0 ... TK;
    bool checka[TM, TK] = (rxa < N)[:, newaxis];
    int offa[)" + sizea + "] = rxa[" + bca0 + "] + rka[" + bca1 + R"(]*lda;
    int offb[)" + sizeb + "] = ryb[" + bcb0 + "]" + ldb0 + " + rkb[" + bcb1 + "]" + ldb1 + R"(;
    int *header = lut + ridy * 4;
    int offset = *(header + 0);
    int K      = *(header + 1);
    int column = *(header + 2);
    int lockid = *(header + 3);
    int *plut   = lut + offset * 2;
    for(int k = K; k > 0; k = k - 1)
    {
       int ak = *(plut + 0);
       int bk = *(plut + 1);
       )" + ab_ty_ + "* pa[" + sizea + R"(] = A + offa + ak * TK * lda;
       )" + ab_ty_ + "* pb[" + sizeb + R"(] = B + offb + bk * TK * TN;
       )" + ab_ty_ + "   a[" + sizea + R"(] = checka ? *pa : 0;
       )" + ab_ty_ + "   b[" + sizeb + R"(] = *pb;
       acc = dot()" + usea + ", " + useb + R"(, acc);
       plut = plut + 2;
    }
    int rxc[TM] = ridx * TM + (0 ... TM);
    int ryc[TN] = column * TN + (0 ... TN);
    )" + c_ty_ + R"(" c[TM, TN] = acc;
    )" + c_ty_ + R"(* pc[TM, TN] = C + rxc[:, newaxis] + ryc[newaxis, :]*ldc;
    bool checkc[TM, TN] = (rxc < N)[:, newaxis];
    if(lockid == 0)
      @checkc *pc = c;
    else
    {
      int *plock = locks + ridx*nlocks + lockid - 1;
      int *pcount = plock + get_num_program(0)*nlocks;
      while(__atomic_cas(plock, 0, 1));
      int count = *pcount;
      if(count == 0){
        @checkc *pc = c;
      }
      else{
        @checkc *pc = c + *pc;
      }
      __atomic_exch(pcount, 1);
      __atomic_exch(plock, 0);
    }
  })";

  os << result;
}

}
}
}
