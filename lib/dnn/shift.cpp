#include <sstream>
#include "triton/dnn/shift.h"
#include "triton/tools/bench.hpp"

namespace triton{
namespace dnn{


shift::shift(int B, int C,
             int D, int H, int W,
             int T, int R, int S,
             int F,
             int stride_h, int stride_w,
             const int32_t *shift_h, const int32_t *shift_w,
             std::string a_ty, std::string b_ty,
             type ty, bool bias)
  : base("shift"),
    B_(B), C_(C),
    AD_(D), AH_(H), AW_(W),
    BD_(T), BH_(R), BW_(S),
    F_(F),
    stride_d_(1), stride_h_(stride_h), stride_w_(stride_w),
    shift_h_(shift_h), shift_w_(shift_w),
    a_ty_(a_ty), b_ty_(b_ty),
    ty_(ty), bias_(bias) {
  // max number of channels
  TK_ = 16;
  MAX_C_ = 8192 + TK_;
  // activation sizes
  CD_ = AD_ / stride_d_;
  CH_ = AH_ / stride_h_;
  CW_ = AW_ / stride_w_;
  // A memory strides: [C, H, W, B]
  lda_n_ = 1;
  lda_w_ = B_;
  lda_h_ = B_*AW_;
  lda_c_ = B_*AW_*AH_;
  // B memory strides: [C, F]
  ldb_n_ = 1;
  ldb_h_ = 1;
  ldb_w_ = 1;
  ldb_c_ = F_;
  // C memory strides: [F, H, W, B]
  ldc_n_ = 1;
  ldc_w_ = B_;
  ldc_h_ = B_*CW_;
  ldc_f_ = B_*CW_*CH_;
  // C shapes
  shapes_c_ = {F, CH_, CW_, B};
  // Equivalent matmul
  M_ = B_*CH_*CW_;
  N_ = F_;
  K_ = C_;
  // transpose
  AT_ = false;
  BT_ = true;
  // Weight gradient
  if(ty_ == WGRAD){
    std::swap(ldb_n_, ldc_n_);
    std::swap(ldb_w_, ldc_w_);
    std::swap(ldb_h_, ldc_h_);
    std::swap(ldb_c_, ldc_f_);
    std::swap(lda_n_, ldb_n_);
    std::swap(lda_w_, ldb_w_);
    std::swap(lda_h_, ldb_h_);
    std::swap(lda_c_, ldb_c_);
    std::swap(M_, K_);
    std::swap(M_, N_);
    AT_ = true;
    BT_ = false;
    shapes_c_ = {C, F};
  }
  // Input gradient
  if(ty_ == BPROP){
    std::swap(lda_n_, ldc_n_);
    std::swap(lda_w_, ldc_w_);
    std::swap(lda_h_, ldc_h_);
    std::swap(lda_c_, ldc_f_);
    std::swap(K_, N_);
    AT_ = false;
    BT_ = false;
    shapes_c_ = {C, AH_, AW_, B};
  }
}

base* shift::clone() const {
  return new shift(*this);
}

void shift::build_delta_a() {
  h_delta_a.resize(MAX_C_);
  if(ty_ == FPROP){
    // compute offset
    auto offset = [&](unsigned c) {
      return c*lda_c_ + shift_h_[c]*lda_h_ + shift_w_[c]*lda_w_;
    };
    // populate look-up table
    for(unsigned c = 0; c < TK_; c++)
      h_delta_a[c] =  offset(c);
    for(unsigned c = 0; c < C_; c++)
      h_delta_a[TK_ + c] = offset(c + TK_) - offset(c);
  }
  if(ty_ == BPROP){
    for(unsigned c = 0; c < C_; c++){
      h_delta_a[c] = shift_h_[c]*ldc_h_ + shift_w_[c]*ldc_w_;
    }
  }
  if(ty_ == WGRAD){
    for(unsigned c = 0; c < C_; c++)
      h_delta_a[c] = shift_h_[c]*ldb_h_ + shift_w_[c]*ldb_w_;
  }
}

size_t shift::c_size() {
  return std::accumulate(shapes_c_.begin(), shapes_c_.end(),
                         1, std::multiplies<int>());
}

std::vector<int32_t> shift::c_shapes(){
  return shapes_c_;
}

size_t shift::num_flops() const {
  return 2.*M_*N_*K_;
}

bool shift::operator <(const base& other) const{
  auto *y = dynamic_cast<const shift*>(&other);
  if(!y)
    return true;
  return std::tie(B_, C_, AD_, AH_, AW_, BD_, BH_, BW_, F_,
                  shift_h_, shift_w_, ty_, bias_)
       < std::tie(y->B_, y->C_, y->AD_, y->AH_, y->AW_, y->BD_, y->BH_, y->BW_, y->F_,
                  y->shift_h_, y->shift_w_, y->ty_, y->bias_);
}

void shift::init_impl(driver::stream *stream, driver::cu_module *module) {
  build_delta_a();
  triton::driver::buffer* delta_a = ((triton::driver::cu_module*)module)->symbol("delta_a");
  stream->write(delta_a, false, 0, h_delta_a.size()*4, h_delta_a.data());
}

void shift::enqueue_impl(driver::stream *stream, driver::kernel *kernel,
                    std::vector<driver::buffer *> args,
                    const std::vector<unsigned> &ranges, size_t nthreads) {
  driver::buffer *a = args[0], *b = args[1], *c = args[2];
  kernel->setArg(0, a);
  kernel->setArg(1, b);
  kernel->setArg(2, c);
  kernel->setArg(3, M_);
  kernel->setArg(4, N_);
  kernel->setArg(5, K_);
  kernel->setArg(6, stride_h_);
  kernel->setArg(7, stride_w_);
  kernel->setArg(8, lda_n_);
  kernel->setArg(9,  lda_w_);
  kernel->setArg(10, lda_h_);
  kernel->setArg(11, lda_c_);
  kernel->setArg(12, ldb_n_);
  kernel->setArg(13, ldb_w_);
  kernel->setArg(14, ldb_h_);
  kernel->setArg(15, ldb_c_);
  kernel->setArg(16, ldc_n_);
  kernel->setArg(17, ldc_w_);
  kernel->setArg(18, ldc_h_);
  kernel->setArg(19, ldc_f_);
  kernel->setArg(20, B_);
  kernel->setArg(21, AH_);
  kernel->setArg(22, AW_);
  kernel->setArg(23, BH_);
  kernel->setArg(24, BW_);
  kernel->setArg(25, CH_);
  kernel->setArg(26, CW_);
  unsigned TM = ranges[0], TN = ranges[1];
  std::array<size_t, 3> grid = {(M_ + TM - 1)/TM, (N_ + TN - 1)/TN, 1};
  if(ty_ == BPROP)
    ((driver::cu_buffer*)c)->set_zero(stream, AH_*AW_*B_*C_*4);
  stream->enqueue(kernel, grid, {nthreads, 1, 1});
}

void shift::triton_c_src(std::ostream &os) const {
  std::string AS0 = "TM", AS1 = "TK";
  std::string BS0 = "TK", BS1 = "TN";
  std::string bcb0 = "[:, newaxis]", bcb1 = "[newaxis, :]";
  std::string usea = AT_ ? "trans(a)" : "a";
  std::string useb = BT_ ? "trans(b)" : "b";
  std::string bca0 = "[newaxis, :]", bca1 = "[:, newaxis]";
  if(AT_){
    std::swap(AS0, AS1);
    std::swap(bca0, bca1);
  }
  if(BT_){
    std::swap(BS0, BS1);
    std::swap(bcb0, bcb1);
  }
  std::string AS = AS0 + ", " + AS1;
  std::string BS = BS0 + ", " + BS1;

  os <<
R"(
const tunable int32 TM = {16, 32, 64, 128};
const tunable int32 TN = {16, 32, 64, 128};
const tunable int32 TK = {)" << TK_ << R"(};

__constant__ int32* delta_a = alloc_const int32[)" << MAX_C_ << R"(];

void shift(restrict read_only align(16) )" << a_ty_ << R"( *A,
           restrict read_only align(16) )" << b_ty_ << R"( *B,
           fp32 *C,
           int32 M, int32 N, int32 K,
           int32 stride_h, int32 stride_w,
           int32 lda_b, int32 lda_w, int32 lda_h, int32 lda_c,
           int32 ldb_b, int32 ldb_w, int32 ldb_h, int32 ldb_c,
           int32 ldc_b, int32 ldc_w, int32 ldc_h, int32 ldc_c,
           int32 NB, int32 AH, int32 AW,
           int32 BH, int32 BW,
           int32 CH, int32 CW) {
  int32 rxa[TM] = get_global_range[TM](0);
  int32 ryb[TN] = get_global_range[TN](1);
  int32 rka[TK] = 0 ... TK;
  int32 rkb[TK] = 0 ... TK;
  fp32 c[TM, TN] = 0;
  int32 pad_h = BH / 2;
  int32 pad_w = BW / 2;)";

/* A offsets */
if(ty_ == FPROP){
  os << R"(
  int32  rawh[TM] =  rxa / NB;
  int32   rab[TM] =  rxa % NB;
  int32   raw[TM] = (rawh % CW) * stride_w;
  int32   rah[TM] = (rawh / CW) * stride_h;
  int32 offxa[TM] =  rab*lda_b + raw*lda_w + rah*lda_h;
  int32 offa0[TM, TK] = offxa[:, newaxis];
  __constant__ int32* pd[TK] = delta_a + rka;
  multiple_of(4) int32 d[TK] = *pd;
  int32 offa_interior[TM, TK] = d[newaxis, :];
  int32 offa_exterior[TM, TK] = rka[newaxis, :] * lda_c;
  int1 interiorh[TM] = (rah >= pad_h) && (rah < (AH - pad_h));
  int1 interiorw[TM] = (raw >= pad_w) && (raw < (AW - pad_w));
  int1 interior[TM, TK] = interiorh[:, newaxis] && interiorw[:, newaxis];
  int32 offa1[TM, TK] = interior ? offa_interior : offa_exterior;)";
}
if(ty_ == BPROP){
  os << R"(
  int32  rawh[TM] =  rxa / NB;
  int32   rab[TM] =  rxa % NB;
  int32   raw[TM] = (rawh % CW);
  int32   rah[TM] = (rawh / CW);
  int32 offxa[TM] =  rab*lda_b + raw*lda_w + rah*lda_h;
  int32 offa0[TM, TK] = offxa[:, newaxis];
  int32 offa1[TM, TK] = rka[newaxis, :] * lda_c;)";
}
if(ty_ == WGRAD){
  os << R"(
  int32 offa0[TK, TM] = rxa[newaxis, :] * lda_c;
  int32 offa1[TK, TM] = rka[:, newaxis];)";
}

/* B offsets */
if(ty_ == FPROP){
  os << R"(
  int32 offb0[TN, TK] = ryb[:, newaxis];
  int32 offb1[TN, TK] = rkb[newaxis, :] * ldb_c;)";
}
if(ty_ == BPROP){
  os << R"(
  int32 offb0[TK, TN] = ryb[newaxis, :] * ldb_c;
  int32 offb1[TK, TN] = rkb[:, newaxis];)";
}
if(ty_ == WGRAD){
  os << R"(
  __constant__ int32* pd[TN] = delta_a + ryb;
  int32 d[TN] = *pd;
  int32 shift[TK, TN] = d[newaxis, :];
  int32 rbwh[TK] = rkb / NB;
  int32 rbb[TK] = rkb % NB;
  int32 rbw[TK] = (rbwh % CW)*stride_w;
  int32 rbh[TK] = (rbwh / CW)*stride_h;
  int32 offkb[TK] = rbb*ldb_b + rbw*ldb_w + rbh*ldb_h;
  int1 interiorh[TK] = (rbh >= pad_h) && (rbh < (AH - pad_h));
  int1 interiorw[TK] = (rbw >= pad_w) && (rbw < (AW - pad_w));
  int1 interior[TK, TN] = interiorh[:, newaxis] && interiorw[:, newaxis];
  int32 incb[TK, TN] = interior ? shift : 0;
  int32 offb0[TK, TN] = ryb[newaxis, :] * ldb_c;
  int32 offb1[TK, TN] = offkb[:, newaxis] + incb;)";
}

/* Main loop */
  os << R"(
  )" << a_ty_ << "* pa[" << AS << R"(] = A + offa0 + offa1;
  )" << b_ty_ << "* pb[" << BS << R"(] = B + offb0 + offb1;
  int1 checka[)" << AS << "] = (rka < K)" << bca0  << R"(;
  int1 checkb[)" << BS << "] = (rkb < K)" << bcb0  << R"(;
  )" << a_ty_ << "   a[" << AS << R"(] = checka ? *pa : 0;
  )" << b_ty_ << "   b[" << BS << R"(] = checkb ? *pb : 0;
  for(int32 k = K; k > 0; k = k - TK){
    c = dot()" << usea << "," << useb << R"(, c);
    int1 checka[)" << AS << R"(] = k > TK;
    int1 checkb[)" << BS << R"(] = k > TK;)";

/* Increment A pointers */
if(ty_ == FPROP){
    os << R"(
    pd = pd + TK;
    d = *pd;
    offa_interior = d[newaxis, :];
    offa_exterior = TK * lda_c;
    int32 offa[TM, TK] = interior ? offa_interior : offa_exterior;
    pa = pa + offa;)";
}
if(ty_ == BPROP){
  os << R"(
  pa = pa + TK * lda_c;)";
}
if(ty_ == WGRAD){
    os << R"(
    pa = pa + TK;)";
}
  os << R"(
  @checka a = *pa;)";

/* Increment B pointers */
if(ty_ == WGRAD){
  os << R"(
    rkb   = rkb + TK;
    rbwh  = rkb / NB;
    rbb   = rkb % NB;
    rbw   = (rbwh % CW)*stride_w;
    rbh   = (rbwh / CW)*stride_h;
    offkb = rbb*ldb_b + rbw*ldb_w + rbh*ldb_h;
    interiorh = (rbh >= pad_h) && (rbh < (AH - pad_h));
    interiorw = (rbw >= pad_w) && (rbw < (AW - pad_w));
    interior  = interiorh[:, newaxis] && interiorw[:, newaxis];
    incb   = interior ? shift : 0;
    pb = B + offb0 + offkb[:, newaxis] + incb;)";
}
if(ty_ == FPROP){
  os << R"(
    pb = pb + TK * ldb_c;)";
}
if(ty_ == BPROP){
  os << R"(
    pb = pb + TK;)";
}
  os << R"(
    @checkb b = *pb;
  }
  int32 rxc[TM] = get_global_range[TM](0);
  int32 ryc[TN] = get_global_range[TN](1);)";

/* C offsets */
if(ty_ == BPROP){
  os << R"(
  int32 rcwh[TM] = rxc / NB;
  int32 rcb[TM] = rxc % NB;
  int32 rcw[TM] = (rcwh % CW) * stride_w;
  int32 rch[TM] = (rcwh / CW) * stride_h;
  int32 offxc[TM] = rcb*ldc_b + rcw*ldc_w + rch*ldc_h;
  )";
  }
if(ty_ == FPROP){
  os << R"(
  int32 rcwh[TM] = rxc / NB;
  int32 rcb[TM] = rxc % NB;
  int32 rcw[TM] = (rcwh % CW);
  int32 rch[TM] = (rcwh / CW);
  int32 offxc[TM] = rcb*ldc_b + rcw*ldc_w + rch*ldc_h;
  )";
}
if(ty_ == WGRAD){
  os << R"(
  int32 offxc[TM] = rxc;
  )";
}
  os << R"("
  fp32* pc[TM, TN] = C + offxc[:, newaxis] + ryc[newaxis, :]*ldc_c;
  int1 checkc0[TM] = rxc < M;
  int1 checkc1[TN] = ryc < N;
  int1 checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];)";
if(ty_ == BPROP){
  os << R"(
  int1 interiorh[TM] = (rch >= pad_h) && (rch < (AH - pad_h));
  int1 interiorw[TM] = (rcw >= pad_w) && (rcw < (AW - pad_w));
  int1 interior[TM, TN] = interiorh[:, newaxis] && interiorw[:, newaxis];
  __constant__ int32* pd[TN] = delta_a + ryc;
  fp32* shift_pc[TM, TN] = pc + (*pd)[newaxis, :];
  pc = interior ? shift_pc : pc;
  @checkc __atomic_add(pc, c);
  )";
}
else{
  os << R"(
  @checkc *pc = c;)";
}
  os << R"(
})";
}

}
}
