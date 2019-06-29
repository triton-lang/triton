#include "triton/dnn/shift.h"


namespace triton{
namespace dnn{

void shift::set_ld(const std::vector<int32_t>& shapes,
                  std::vector<int32_t>& ld) {
  size_t size = shapes.size();
  ld.resize(size);
  ld[3] = 1;
  ld[2] = shapes[3]*ld[3];
  ld[1] = shapes[2]*ld[2];
  ld[0] = shapes[1]*ld[1];
}

shift::shift(int B, int NC,
             int D, int H, int W,
             int T, int R, int S,
             int NF,
             const std::vector<int32_t>& shift_h, const std::vector<int32_t>& shift_w,
             std::string a_ty, std::string b_ty,
             type ty, bool bias)
  : NB_(B), NC_(NC),
    AD_(D), AH_(H), AW_(W),
    BD_(T), BH_(R), BW_(S),
    NF_(NF),
    shift_h_(shift_h), shift_w_(shift_w),
    a_ty_(a_ty), b_ty_(b_ty),
    ty_(ty), bias_(bias) {
  // max number of channels
  TK_ = 16;
  MAX_C_ = 8192 + TK_;
  // equivalent matmul
  M_ = NB_*AH_*AW_;
  N_ = NF_;
  K_ = NC_;
  // shapes
  // input layout: C, H, W, BS
  // filter layout: C, K
  // output layout: K, H, W, BS
  shapes_a_ = {NC, H, W, B};
  shapes_b_ = {NC, NF};
  shapes_c_ = {NF, H, W, B};
  // memory strides
  set_ld(shapes_a_, ld_a_);
  // build LUTs
  build_deltas();
}

void shift::build_deltas() {
  // compute offset
  auto offset = [&](unsigned c) {
    return c*ld_a_[0] + shift_h_[c]*ld_a_[1] + shift_w_[c]*ld_a_[2];
  };
  h_deltas_.resize(MAX_C_);
  // populate look-up table
  for(unsigned c = 0; c < TK_; c++)
    h_deltas_[c] =  offset(c);
  for(unsigned c = 0; c < NC_; c++)
    h_deltas_[TK_ + c] = offset(c + TK_) - offset(c);
}

size_t shift::a_size(){
  return std::accumulate(shapes_a_.begin(), shapes_a_.end(),
                         1, std::multiplies<int>());
}

size_t shift::b_size(){
  return std::accumulate(shapes_b_.begin(), shapes_b_.end(),
                         1, std::multiplies<int>());
}

size_t shift::c_size(){
  return std::accumulate(shapes_c_.begin(), shapes_c_.end(),
                         1, std::multiplies<int>());
}

std::vector<int32_t> shift::c_shapes(){
  return shapes_c_;
}

size_t shift::get_nflops() {
  return 2. * M_ * N_ * K_;
}


void shift::init(driver::stream *stream, driver::cu_module *module) {
  triton::driver::buffer* delta = ((triton::driver::cu_module*)module)->symbol("delta");
  stream->write(delta, false, 0, h_deltas_.size()*4, h_deltas_.data());
}

void shift::enqueue(driver::stream *stream, driver::kernel *kernel,
                    driver::buffer *a, driver::buffer *b, driver::buffer *c,
                    size_t TM, size_t TN, size_t nthreads) {
  kernel->setArg(0, a);
  kernel->setArg(1, b);
  kernel->setArg(2, c);
  kernel->setArg(3, M_);
  kernel->setArg(4, N_);
  kernel->setArg(5, K_);
  kernel->setArg(6, NB_*AH_*AW_);
  kernel->setArg(7, NB_);
  kernel->setArg(8, AH_);
  kernel->setArg(9, AW_);
  kernel->setArg(10, BH_);
  kernel->setArg(11, BW_);
  // dry run
  std::array<size_t, 3> grid = {(M_ + TM - 1)/TM, (N_ + TN - 1)/TN, 1};
  stream->enqueue(kernel, grid, {nthreads, 1, 1});
}

void shift::src(std::ostream &os) {
  os <<
R"(
const tunable int32 TM = {16, 32, 64, 128};
const tunable int32 TN = {16, 32, 64, 128};
const tunable int32 TK = {)" << TK_ << R"(};

__constant__ int32* delta = alloc_const int32[)" << MAX_C_ << R"(];

void shift(restrict read_only align(16) )" << a_ty_ << R"( *a,
           restrict read_only align(16) )" << b_ty_ << R"( *b,
           fp32 *c,
           multiple_of(4) int32 M, multiple_of(4) int32 N, multiple_of(4) int32 K,
           multiple_of(4) int32 lda,
           int32 ABS, int32 AH, int32 AW, int32 AR, int32 AS) {
  int32 rxa[TM] = get_global_range[TM](0);
  int32 ryb[TN] = get_global_range[TN](1);
  int32 rka[TK] = 0 ... TK;
  int32 rkb[TK] = 0 ... TK;
  fp32 C[TM, TN] = 0;
  int32 pad_h = AR / 2;
  int32 pad_w = AS / 2;
  int32 rawhc[TM] = rxa / ABS;
  int32 raw[TM] = rawhc % AW;
  int32 rahc[TM] = rawhc / AW;
  int32 rah[TM] = rahc % AH;
  int1 maskh[TM] = (rah >= pad_h) && (rah < (AH - pad_h));
  int1 maskw[TM] = (raw >= pad_w) && (raw < (AW - pad_w));
  int1 mask[TM, TK] = maskh[:, newaxis] && maskw[:, newaxis];
  __constant__ int32* pd[TK] = delta + rka;
  multiple_of(4) int32 d[TK];
  d = *pd;
  int32 offa1[TK] = rka*lda;
  int32 inc[TM, TK] = mask ? d[newaxis, :] : offa1[newaxis, :];
  )" << a_ty_ << R"(* pa[TM, TK] = a + rxa[:, newaxis] + inc;
  )" << b_ty_ << R"(* pb[TN, TK] = b + rkb[newaxis, :]*N + ryb[:, newaxis];
  )" << a_ty_ << R"( a[TM, TK] = *pa;
  )" << b_ty_ << R"( b[TN, TK] = *pb;
  for(int32 k = K; k > 0; k = k - TK){
    C = dot(a, trans(b), C);
    pb = pb + TK*N;
    pd = pd + TK;
    d = *pd;
    pa = pa + (mask ? d[newaxis, :] : TK*lda);
    int1 checka[TM, TK] = k > TK;
    int1 checkb[TN, TK] = k > TK;
    @checka a = *pa;
    @checkb b = *pb;
  }
  int32 rxc[TM] = get_global_range[TM](0);
  int32 ryc[TN] = get_global_range[TN](1);
  fp32* pc[TM, TN] = c + ryc[newaxis, :]*M + rxc[:, newaxis];
  int1 checkc0[TM] = rxc < M;
  int1 checkc1[TN] = ryc < N;
  int1 checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];
  @checkc *pc = C;
}
)";
}

}
}
