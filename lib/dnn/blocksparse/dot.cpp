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
  kernel->setArg(0, a);
  kernel->setArg(1, b);
  kernel->setArg(2, c);
  if(op_ == FPROP || op_ == BPROP){
    kernel->setArg(3, N_);
    kernel->setArg(4, BS_);
    kernel->setArg(5, N_);
  }
  else{
    kernel->setArg(3, N_);
    kernel->setArg(4, N_);
    kernel->setArg(5, BS_);
  }
  kernel->setArg(6, N_);
  kernel->setArg(7, lut);
  kernel->setArg(8, locks_.get());
  kernel->setArg(9, nlocks_);
  if(op_ == FPROP || op_ == BPROP){
    int32_t TM = info.globals["TM"];
    size_t grid_0 = (N_ + TM - 1) / TM;
    size_t grid_1 = S_;
    if(nlocks_)
      ((driver::cu_buffer*)locks_.get())->set_zero(stream, grid_0 * nlocks_ * 2 * 4);
    stream->enqueue(kernel, {grid_0, grid_1, 1}, {info.num_threads, 1, 1});
  }
  else{
    size_t grid_0 = nblocks_;
    stream->enqueue(kernel, {grid_0, 1, 1}, {info.num_threads, 1, 1});
  }
}

driver::buffer* dot::get_locks() const {
  return locks_.get();
}

std::string dot::triton_c_src_ydx() const {
  bool AT = (op_ == WGRAD);
  bool BT = (op_ == FPROP);
  std::string usea = AT ? "trans(a)" : "a";
  std::string useb = BT ? "trans(b)" : "b";
  std::string sizea = "TM, TK";
  std::string sizeb = BT ? "TN, TK" : "TK, TN";
  std::string bca0 = ":, newaxis";
  std::string bca1 = "newaxis, :";
  std::string bcb0 = BT ? ":, newaxis" : "newaxis, :";
  std::string bcb1 = BT ? "newaxis, :" : ":, newaxis";
  std::string lda0 = AT ? "*lda" : "";
  std::string lda1 = AT ? "" : "*lda";
  std::string ldb0 = BT ? ""  : "*ldb";
  std::string ldb1 = BT ? "*ldb" : "" ;
  std::string result =
  R"(
  const tunable int TM = {16, 32, 64, 128};
  const tunable int TN = {)" + std::to_string(BS_) + R"(};
  const tunable int TK = {)" + std::to_string(BS_) + R"(};

  void bsdot(restrict read_only align(16) )" + ab_ty_ + R"( *A,
             restrict read_only align(16) )" + ab_ty_ + R"( *B,
             )" + c_ty_ + R"(* C,
             int lda, int ldb, int ldc,
             int N, int* lut,
             int* locks, int nlocks) {
    int ridx = get_range_id(0);
    float acc[TM, TN] = 0;
    int rka[TK] = 0 ... TK;
    int rkb[TK] = 0 ... TK;
    int *header = lut + get_range_id(1) * 4;
    int offset = *(header + 0);
    int K      = *(header + 1);
    int column = *(header + 2);
    int lockid = *(header + 3);
    int rxa[TM] = ridx * TM + (0 ... TM);
    int ryb[TN] = 0 ... TN;
    int *plut   = lut + offset * 2;
    int offa[)" + sizea + "] = rxa[" + bca0 + "]" + lda0 + " + rka[" + bca1 + "]" + lda1 + R"(;
    int offb[)" + sizeb + "] = ryb[" + bcb0 + "]" + ldb0 + " + rkb[" + bcb1 + "]" + ldb1 + R"(;
    bool checka[TM, TK] = (rxa < N)[:, newaxis];
    for(int k = K; k > 0; k = k - 1) {
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
    if(lockid == 0) {
      @checkc *pc = c;
    }
    else {
      int *plock = locks + ridx*nlocks + lockid - 1;
      int *pcount = plock + get_num_program(0)*nlocks;
      while(__atomic_cas(plock, 0, 1));
      int count = *pcount;
      if(count == 0)
        @checkc *pc = c;
      else
        @checkc *pc = c + *pc;
      __atomic_exch(pcount, 1);
      __atomic_exch(plock, 0);
    }
  })";

  return result;
}

std::string dot::triton_c_src_dw() const {
  bool AT = (op_ == WGRAD);
  bool BT = (op_ == FPROP);
  std::string usea = AT ? "trans(a)" : "a";
  std::string useb = BT ? "trans(b)" : "b";
  std::string sizea = AT ? "TK, TM" : "TM, TK";
  std::string sizeb = BT ? "TN, TK" : "TK, TN";
  std::string bca0 = AT ? "newaxis, :" : ":, newaxis";
  std::string bca1 = AT ? ":, newaxis" : "newaxis, :";
  std::string bcb0 = BT ? ":, newaxis" : "newaxis, :";
  std::string bcb1 = BT ? "newaxis, :" : ":, newaxis";
  std::string lda0 = AT ? "*lda" : "";
  std::string lda1 = AT ? "" : "*lda";
  std::string ldb0 = BT ? ""  : "*ldb";
  std::string ldb1 = BT ? "*ldb" : "" ;
  std::string result =
  R"(
  const tunable int TM = {)" + std::to_string(BS_) + R"(};
  const tunable int TN = {)" + std::to_string(BS_) + R"(};
  const tunable int TK = {32};

  void bsdot(restrict read_only align(16) )" + ab_ty_ + R"( *A,
             restrict read_only align(16) )" + ab_ty_ + R"( *B,
             )" + c_ty_ + R"(* C,
             int lda, int ldb, int ldc,
             int N, int* lut,
             int* locks, int nlocks) {
    int ridx = get_range_id(0);
    float acc[TM, TN] = 0;
    int rka[TK] = 0 ... TK;
    int rkb[TK] = 0 ... TK;
    int *header = lut + ridx * 2;
    int offx = *(header + 0);
    int offy = *(header + 1);
    int rxa[TM] = offx*TM + (0 ... TM);
    int ryb[TN] = offy*TN + (0 ... TN);
    bool checka[TK, TM] = (rka < N)[:, newaxis];
    bool checkb[TK, TN] = (rkb < N)[:, newaxis];
    int offa[)" + sizea + "] = rxa[" + bca0 + "]" + lda0 + " + rka[" + bca1 + "]" + lda1 + R"(;
    int offb[)" + sizeb + "] = ryb[" + bcb0 + "]" + ldb0 + " + rkb[" + bcb1 + "]" + ldb1 + R"(;
    )" + ab_ty_ + " * pa[" + sizea + R"(] = A + offa;
    )" + ab_ty_ + " * pb[" + sizeb + R"(] = B + offb;
    )" + ab_ty_ + "   a[" + sizea + R"(] = checka ? *pa : 0;
    )" + ab_ty_ + "   b[" + sizeb + R"(] = checkb ? *pb : 0;
    for(int k = N; k > 0; k = k - TK) {
       acc = dot()" + usea + ", " + useb + R"(, acc);
       pa = pa + TK)" + lda1 + R"(;
       pb = pb + TK)" + ldb1 + R"(;
       a = checka ? *pa : 0;
       b = checkb ? *pb : 0;
    }
    int rxc[TM] = (0 ... TM);
    int ryc[TN] = (0 ... TN);
    )" + c_ty_ + R"( c[TM, TN] = acc;
    )" + c_ty_ + R"(* pc[TM, TN] = C + rxc[:, newaxis]*TM + ryc[newaxis, :] + ridx*TM*TN;
    *pc = c;
  })";

  return result;
}
void dot::triton_c_src(std::ostream &os) const {
  if(op_ == FPROP || op_ == BPROP)
    os << triton_c_src_ydx();
  else
    os << triton_c_src_dw();
}



}
}
}
