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
  build_masks();
}

void shift::build_deltas() {
  h_deltas_ = std::vector<int32_t>(512, 0);
  for(unsigned c = 0; c < NC_; c++){
    h_deltas_[c] = c*ld_a_[0];
    h_deltas_[c] += shift_h_[c]*ld_a_[1];
    h_deltas_[c] += shift_w_[c]*ld_a_[2];
  }
}

void shift::build_masks() {
  size_t S0 = NC_;
  size_t S1 = BH_;
  size_t S2 = BW_;
  h_masks_.resize(S0*S1*S2);
  for(size_t ph = 0; ph < S1; ++ph)
  for(size_t pw = 0; pw < S2; ++pw){
    int32_t* ptr = &h_masks_[ph*S0 + pw*S0*S1];
    for(size_t i = 0; i < S0; ++i){
      bool in_bounds_h = shift_h_[i] + ph >= 0 && shift_h_[i] + ph < BH_;
      bool in_bounds_w = shift_w_[i] + pw >= 0 && shift_w_[i] + pw < BW_;
      ptr[i] = in_bounds_h && in_bounds_w;
    }
  }
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
  triton::driver::buffer* masks = ((triton::driver::cu_module*)module)->symbol("masks");
  stream->write(delta, false, 0, h_deltas_.size()*4, h_deltas_.data());
  stream->write(masks, false, 0, h_masks_.size()*4, h_masks_.data());
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
  kernel->setArg(6, NB_);
  kernel->setArg(7, AH_);
  kernel->setArg(8, AW_);
  kernel->setArg(9, BH_);
  kernel->setArg(10, BW_);
  // dry run
  std::array<size_t, 3> grid = {(M_ + TM - 1)/TM, (N_ + TN - 1)/TN, 1};
  stream->enqueue(kernel, grid, {nthreads, 1, 1});
}

void shift::src(std::ostream &os) {
  os <<
R"(
const tunable int32 TM = {16, 32, 64, 128};
const tunable int32 TN = {16, 32, 64, 128};
const tunable int32 TK = {8};

__constant__ int32* delta = alloc_const int32[512];
__constant__ int32* masks = alloc_const int32[8192];

void shift(restrict read_only fp32 *a, restrict read_only fp32 *b, fp32 *c,
           int32 M, int32 N, int32 K,
           int32 ABS, int32 AH, int32 AW, int32 AR, int32 AS) {
  int32 rxa[TM] = get_global_range[TM](0);
  int32 ryb[TN] = get_global_range[TN](1);
  int32 rka[TK] = 0 ... TK;
  int32 rkb[TK] = 0 ... TK;
  fp32 C[TM, TN] = 0;
  fp32* pxa[TM, TK] = a + rxa[:, newaxis];
  fp32* pb[TN, TK] = b + rkb[newaxis, :]*N + ryb[:, newaxis];
  int32 pad_h = AR/2;
  int32 pad_w = AS/2;
  int32 rawhc[TM] = rxa / ABS;
  int32 raw[TM] = rawhc % AW;
  int32 rahc[TM] = rawhc / AW;
  int32 rah[TM] = rahc % AH;
  int1 maskh[TM] = (rah >= pad_h) && (rah < (AH - pad_h));
  int1 maskw[TM] = (raw >= pad_w) && (raw < (AW - pad_w));
  int32 offd[TM] = (maskh && maskw) ? 0 : 256;
  __constant__ int32* pd[TM, TK] = delta + rka[newaxis, :] + offd[:, newaxis];
  for(int32 k = K; k > 0; k = k - TK){
    int32 delta[TM, TK] = *pd;
    fp32 *pa[TM, TK] = pxa + delta;
    fp32 a[TM, TK] = *pa;
    fp32 b[TN, TK] = *pb;
    C = dot(a, trans(b), C);
    pb = pb + TK*N;
    pd = pd + TK;
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
