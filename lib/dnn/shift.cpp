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
  // transpose
  AT_ = false;
  BT_ = true;
  // activation sizes
  CD_ = AD_ / stride_d_;
  CH_ = AH_ / stride_h_;
  CW_ = AW_ / stride_w_;
  // equivalent matmul
  M_ = B_*CH_*CW_;
  N_ = F_;
  K_ = C_;
  // shapes
  // input layout: C, H, W, B
  // filter layout: C, F
  // output layout: F, H, W, B
  shapes_a_ = {C, AH_, AW_, B};
  shapes_b_ = {C, F};
  shapes_c_ = {F, CH_, CW_, B};
  if(ty_ == WGRAD){
    shapes_b_.swap(shapes_c_);
    shapes_a_.swap(shapes_b_);
    AT_ = true;
    BT_ = false;
    M_ = F_;
    N_ = C_;
    K_ = B_*CH_*CW_;
  }
  if(ty_ == BPROP){
    shapes_a_.swap(shapes_c_);
    AT_ = false;
    BT_ = false;
    K_ = F_;
    M_ = B_*CH_*CW_;
    N_ = C_;
  }
  // memory strides
  set_ld(shapes_a_, ld_a_);
  set_ld(shapes_b_, ld_b_);
  set_ld(shapes_c_, ld_c_);
}

base* shift::clone() const {
  return new shift(*this);
}

void shift::build_deltas() {
  h_deltas_.resize(MAX_C_);
  if(ty_ == FPROP){
    // compute offset
    auto offset = [&](unsigned c) {
      return c*ld_a_[0] + shift_h_[c]*ld_a_[1] + shift_w_[c]*ld_a_[2];
    };
    // populate look-up table
    for(unsigned c = 0; c < TK_; c++)
      h_deltas_[c] =  offset(c);
    for(unsigned c = 0; c < C_; c++)
      h_deltas_[TK_ + c] = offset(c + TK_) - offset(c);
  }
  if(ty_ == BPROP){
    for(unsigned c = 0; c < C_; c++){
      h_deltas_[c] = shift_h_[c]*ld_c_[1] + shift_w_[c]*ld_c_[2];
    }
  }
  if(ty_ == WGRAD){
    for(unsigned c = 0; c < C_; c++)
      h_deltas_[c] = shift_h_[c]*ld_b_[1] + shift_w_[c]*ld_b_[2];
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
  build_deltas();
  triton::driver::buffer* delta = ((triton::driver::cu_module*)module)->symbol("delta");
  stream->write(delta, false, 0, h_deltas_.size()*4, h_deltas_.data());
}

void shift::enqueue_impl(driver::stream *stream, driver::kernel *kernel,
                    std::vector<driver::buffer *> args,
                    const std::vector<unsigned> &ranges, size_t nthreads) {
  int32_t lda = AT_ ? K_ : M_;
  int32_t ldb = BT_ ? N_ : K_;
  int32_t ldc = M_;
  if(ty_ == FPROP)
    lda *= stride_h_*stride_w_;
  if(ty_ == WGRAD)
    ldb *= stride_h_*stride_w_;
  if(ty_ == BPROP)
    ldc *= stride_h_*stride_w_;
  driver::buffer *a = args[0], *b = args[1], *c = args[2];
  kernel->setArg(0, a);
  kernel->setArg(1, b);
  kernel->setArg(2, c);
  kernel->setArg(3, M_);
  kernel->setArg(4, N_);
  kernel->setArg(5, K_);
  kernel->setArg(6, stride_h_);
  kernel->setArg(7, stride_w_);
  kernel->setArg(8, lda);
  kernel->setArg(9, ldb);
  kernel->setArg(10, ldc);
  kernel->setArg(11, B_);
  kernel->setArg(12, AH_);
  kernel->setArg(13, AW_);
  kernel->setArg(14, BH_);
  kernel->setArg(15, BW_);
  kernel->setArg(16, CH_);
  kernel->setArg(17, CW_);
  unsigned TM = ranges[0], TN = ranges[1];
  std::array<size_t, 3> grid = {(M_ + TM - 1)/TM, (N_ + TN - 1)/TN, 1};
  if(ty_ == BPROP)
    ((driver::cu_buffer*)c)->set_zero(stream, ldc*N_*4);
  stream->enqueue(kernel, grid, {nthreads, 1, 1});
}

void shift::triton_c_src(std::ostream &os) const {
  std::string AS0 = "TM", AS1 = "TK";
  std::string BS0 = "TK", BS1 = "TN";
  std::string bcb0 = "[:, newaxis]", bcb1 = "[newaxis, :]";
  std::string ldb0 = "", ldb1 = "*ldb";
  std::string usea = AT_ ? "trans(a)" : "a";
  std::string useb = BT_ ? "trans(b)" : "b";
  std::string rkb = "rkb";
  std::string rka = "rka";
  std::string bca0 = "[newaxis, :]", bca1 = "[:, newaxis]";
  std::string lda0 = "*lda", lda1 = "";
  if(ty_ == FPROP){
    rka = "inc";
    bca0 = "";
    lda0 = "";
  }

  if(AT_){
    std::swap(AS0, AS1);
    std::swap(bca0, bca1);
    std::swap(lda0, lda1);
  }
  if(BT_){
    std::swap(BS0, BS1);
    std::swap(bcb0, bcb1);
    std::swap(ldb0, ldb1);
  }
  std::string AS = AS0 + ", " + AS1;
  std::string BS = BS0 + ", " + BS1;

  os <<
R"(
const tunable int32 TM = {16, 32, 64, 128};
const tunable int32 TN = {16, 32, 64, 128};
const tunable int32 TK = {)" << TK_ << R"(};

__constant__ int32* delta = alloc_const int32[)" << MAX_C_ << R"(];

void shift(restrict read_only align(16) )" << a_ty_ << R"( *a,
           restrict read_only align(16) )" << b_ty_ << R"( *b,
           fp32 *c,
           int32 M, int32 N, int32 K,
           int32 stride_h, int32 stride_w,
           int32 lda, int32 ldb, int32 ldc,
           int32 NB, int32 AH, int32 AW, int32 BH, int32 BW, int32 CH, int32 CW) {
  int32 rxa[TM] = get_global_range[TM](0);
  int32 ryb[TN] = get_global_range[TN](1);
  int32 rka[TK] = 0 ... TK;
  int32 rkb[TK] = 0 ... TK;
  fp32 C[TM, TN] = 0;
  int32 pad_h = BH / 2;
  int32 pad_w = BW / 2;)";
if(ty_ == FPROP){
  os << R"(
  int32 rawh[TM] = rxa / NB;
  int32 rab[TM] = rxa % NB;
  int32 raw[TM] = (rawh % CW)*stride_w;
  int32 rah[TM] = (rawh / CW)*stride_h;
  __constant__ int32* pd[TK] = delta + rka;
  multiple_of(4) int32 d[TK] = *pd;
  int1 interiorh[TM] = (rah >= pad_h) && (rah < (AH - pad_h));
  int1 interiorw[TM] = (raw >= pad_w) && (raw < (AW - pad_w));
  int1 interior[TM, TK] = interiorh[:, newaxis] && interiorw[:, newaxis];
  int32 inc_true[TM, TK] = d[newaxis, :];
  int32 inc_false[TM, TK] = rka[newaxis, :] * lda;
  int32 inc[TM, TK] = interior ? inc_true : inc_false;
  int32 offxa[TM] = rab + raw*NB + rah*NB*AW;)";
}
else{
  os << R"(
  int32 offxa[TM] = rxa;)";
}
if(ty_ == WGRAD){
  os << R"(
  __constant__ int32* pd[TN] = delta + ryb;
  int32 d[TN] = *pd;
  int32 shift[TK, TN] = d[newaxis, :];
  int32 rbwh[TK] = rkb / NB;
  int32 rbb[TK] = rkb % NB;
  int32 rbw[TK] = (rbwh % CW)*stride_w;
  int32 rbh[TK] = (rbwh / CW)*stride_h;
  int32 offkb[TK] = rbb + rbw*NB + rbh*NB*AW;
  int1 interiorh[TK] = (rbh >= pad_h) && (rbh < (AH - pad_h));
  int1 interiorw[TK] = (rbw >= pad_w) && (rbw < (AW - pad_w));
  int1 interior[TK, TN] = interiorh[:, newaxis] && interiorw[:, newaxis];
  int32 inc[TK, TN] = interior ? shift : 0;
  )" << b_ty_ << "* pb_base[" << BS << "] = b + ryb" << bcb1 << ldb1 << R"(;
  )" << b_ty_ << "* pb[" << BS << "] = pb_base + offkb[:, newaxis] + inc;";
}
else{
  os << R"(
  int32 offkb[TK] = rkb;
  )" << b_ty_ << "* pb[" << BS << "] = b + ryb" << bcb1 << ldb1 << " + " << "offkb" << bcb0 << ldb0 << R"(;
  )";
}
  os << R"(
  )" << a_ty_ << "* pa[" << AS << "] = a + offxa" << bca1 << lda1 << " + " << rka << bca0 << lda0 << R"(;
  int1 checka[)" << AS << "] = (rka < K)" << bca0  << R"(;
  int1 checkb[)" << BS << "] = (rkb < K)" << bcb0  << R"(;
  )" << a_ty_ << "   a[" << AS << R"(] = checka ? *pa : 0;
  )" << b_ty_ << "   b[" << BS << R"(] = checkb ? *pb : 0;
  for(int32 k = K; k > 0; k = k - TK){
    C = dot()" << usea << "," << useb << R"(, C);
    int1 checka[)" << AS << R"(] = k > TK;
    int1 checkb[)" << BS << R"(] = k > TK;)";
if(ty_ == FPROP){
    os << R"(
    pd = pd + TK;
    d = *pd;
    inc_true = d[newaxis, :];
    inc_false = TK * lda;
    inc = interior ? inc_true : inc_false;
    pa = pa + inc;
    @checka a = *pa;)";
}
else{
    os << R"(
    pa = pa + TK)" << lda0 << R"(;
    @checka a = *pa;)";
}
if(ty_ == WGRAD){
  os << R"(
    rkb   = rkb + TK;
    rbwh  = rkb / NB;
    rbb   = rkb % NB;
    rbw   = (rbwh % CW)*stride_w;
    rbh   = (rbwh / CW)*stride_h;
    offkb = rbb + rbw*NB + rbh*NB*AW;
    interiorh = (rbh >= pad_h) && (rbh < (AH - pad_h));
    interiorw = (rbw >= pad_w) && (rbw < (AW - pad_w));
    interior  = interiorh[:, newaxis] && interiorw[:, newaxis];
    inc   = interior ? shift : 0;
    pb = pb_base + offkb[:, newaxis] + inc;
    @checkb b = *pb;)";
}
else{
  os << R"(
    pb = pb + TK)" << ldb0 << R"(;
    @checkb b = *pb;)";
}
  os << R"(
  }
  int32 rxc[TM] = get_global_range[TM](0);
  int32 ryc[TN] = get_global_range[TN](1);)";
  if(ty_ == BPROP){
  os << R"(
  int32 rcwh[TM] = rxc / NB;
  int32 rcb[TM] = rxc % NB;
  int32 rcw[TM] = (rcwh % CW) * stride_w;
  int32 rch[TM] = (rcwh / CW) * stride_h;
  int32 offxc[TM] = rcb + rcw*NB + rch*NB*AW;
  )";
  }
  else{
  os << R"(
  int32 offxc[TM] = rxc;
  )";
  }
  os << R"("
  fp32* pc[TM, TN] = c + ryc[newaxis, :]*ldc + offxc[:, newaxis];
  int1 checkc0[TM] = rxc < M;
  int1 checkc1[TN] = ryc < N;
  int1 checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];)";
if(ty_ == BPROP){
  os << R"(
  int1 interiorh[TM] = (rch >= pad_h) && (rch < (AH - pad_h));
  int1 interiorw[TM] = (rcw >= pad_w) && (rcw < (AW - pad_w));
  int1 interior[TM, TN] = interiorh[:, newaxis] && interiorw[:, newaxis];
  __constant__ int32* pd[TN] = delta + ryc;
  fp32* shift_pc[TM, TN] = pc + (*pd)[newaxis, :];
  pc = interior ? shift_pc : pc;
  @checkc __atomic_add(pc, C);
  )";
}
else{
  os << R"(
  @checkc *pc = C;)";
}
  os << R"(
})";
}

}
}
