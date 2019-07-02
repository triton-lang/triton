#include "triton/dnn/shift.h"


namespace triton{
namespace dnn{

void shift::set_ld(const std::vector<int32_t>& shapes,
                  std::vector<int32_t>& ld) {
  size_t size = shapes.size();
  ld.resize(size);
  ld[size - 1] = 1;
  for(int i = size - 1; i >= 1; i--)
    ld[i - 1] = shapes[i] * ld[i];
}

shift::shift(int B, int C,
             int D, int H, int W,
             int T, int R, int S,
             int F,
             const std::vector<int32_t>& shift_h, const std::vector<int32_t>& shift_w,
             std::string a_ty, std::string b_ty,
             type ty, bool bias)
  : B_(B), C_(C),
    AD_(D), AH_(H), AW_(W),
    BD_(T), BH_(R), BW_(S),
    F_(F),
    shift_h_(shift_h), shift_w_(shift_w),
    a_ty_(a_ty), b_ty_(b_ty),
    ty_(ty), bias_(bias) {
  // max number of channels
  TK_ = 16;
  MAX_C_ = 8192 + TK_;
  // transpose
  AT_ = false;
  BT_ = true;
  // equivalent matmul
  M_ = B_*AH_*AW_;
  N_ = F_;
  K_ = C_;
  // shapes
  // input layout: C, H, W, B
  // filter layout: C, F
  // output layout: F, H, W, B
  shapes_a_ = {C, H, W, B};
  shapes_b_ = {C, F};
  shapes_c_ = {F, H, W, B};
  if(ty_ == WGRAD){
    shapes_b_.swap(shapes_c_);
    shapes_a_.swap(shapes_b_);
    AT_ = true;
    BT_ = false;
    M_ = K_;
    N_ = C_;
    K_ = B_*AH_*AW_;
  }
  if(ty_ == BPROP){
    shapes_a_.swap(shapes_c_);
    AT_ = false;
    BT_ = false;
    K_ = F_;
    M_ = B_*AH_*AW_;
    N_ = C_;
  }
  // memory strides
  set_ld(shapes_a_, ld_a_);
  set_ld(shapes_b_, ld_b_);
  set_ld(shapes_c_, ld_c_);
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
  for(unsigned c = 0; c < C_; c++)
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
  if(ty_ == WGRAD)
    std::swap(a, b);
  kernel->setArg(0, a);
  kernel->setArg(1, b);
  kernel->setArg(2, c);
  kernel->setArg(3, M_);
  kernel->setArg(4, N_);
  kernel->setArg(5, K_);
  kernel->setArg(6, B_*AH_*AW_);
  kernel->setArg(7, N_);
  kernel->setArg(8, B_);
  kernel->setArg(9, AH_);
  kernel->setArg(10, AW_);
  kernel->setArg(11, BH_);
  kernel->setArg(12, BW_);
  std::array<size_t, 3> grid = {(M_ + TM - 1)/TM, (N_ + TN - 1)/TN, 1};
  if(ty_ == BPROP)
    ((driver::cu_buffer*)c)->set_zero(stream, M_*N_*4);
  stream->enqueue(kernel, grid, {nthreads, 1, 1});
}

void shift::src(std::ostream &os) {
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
           multiple_of(4) int32 lda, multiple_of(4) int32 ldb,
           int32 ABS, int32 AH, int32 AW, int32 AR, int32 AS) {
  int32 rxa[TM] = get_global_range[TM](0);
  int32 ryb[TN] = get_global_range[TN](1);
  int32 rka[TK] = 0 ... TK;
  int32 rkb[TK] = 0 ... TK;
  fp32 C[TM, TN] = 0;
  int32 pad_h = AR / 2;
  int32 pad_w = AS / 2;)";
if(ty_ == FPROP){
  os << R"(
  int32 rawhc[TM] = rxa / ABS;
  int32 raw[TM] = rawhc % AW;
  int32 rahc[TM] = rawhc / AW;
  int32 rah[TM] = rahc % AH;
  __constant__ int32* pd[TK] = delta + rka;
  multiple_of(4) int32 d[TK] = *pd;
  int1 maskh[TM] = (rah >= pad_h) && (rah < (AH - pad_h));
  int1 maskw[TM] = (raw >= pad_w) && (raw < (AW - pad_w));
  int1 mask[TM, TK] = maskh[:, newaxis] && maskw[:, newaxis];
  int32 inc_true[TM, TK] = d[newaxis, :];
  int32 inc_false[TM, TK] = rka[newaxis, :] * lda;
  int32 inc[TM, TK] = mask ? inc_true : inc_false;)";
}
if(ty_ == WGRAD){
  os << R"(
  int32 shift[TK, TN] = 0;)";
}
  os << R"(
  )" << a_ty_ << "* pa[" << AS << "] = a + rxa" << bca1 << " + " << rka << bca0 << lda0 << R"(;
  )" << b_ty_ << "* pb[" << BS << "] = b + ryb" << bcb1 << " + " << rkb << bcb0 << ldb0 << R"(;
  )" << a_ty_ << "   a[" << AS << R"(] = *pa;
  )" << b_ty_ << "   b[" << BS << R"(] = *pb;
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
    inc = mask ? inc_true : inc_false;
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
    int32 rbwhc[TK] = rkb / ABS;
    int32 rbw[TK] = rbwhc % AW;
    int32 rbhc[TK] = rbwhc / AW;
    int32 rbh[TK] = rbhc % AH;
    int1 maskh[TK] = (rbh >= pad_h) && (rbh < (AH - pad_h));
    int1 maskw[TK] = (rbw >= pad_w) && (rbw < (AW - pad_w));
    int1 mask[TK, TN] = maskh[:, newaxis] && maskw[:, newaxis];
    int32 inc[TK, TN] = mask ? 0 : shift;
    pb = pb +  TK;
    )" << b_ty_ << R"(* pbb[TK, TN] = pb + inc;
    @checkb b = *pbb;)";
}
else{
  os << R"(
    pb = pb + TK)" << ldb0 << R"(;
    @checkb b = *pb;)";
}
  os << R"(
  }
  int32 rxc[TM] = get_global_range[TM](0);
  int32 ryc[TN] = get_global_range[TN](1);
  fp32* pc[TM, TN] = c + ryc[newaxis, :]*M + rxc[:, newaxis];
  int1 checkc0[TM] = rxc < M;
  int1 checkc1[TN] = ryc < N;
  int1 checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];)";
if(ty_ == BPROP){
  os << R"(
  int32 rcwhc[TM] = rxc / ABS;
  int32 rcw[TM] = rcwhc % AW;
  int32 rchc[TM] = rcwhc / AW;
  int32 rch[TM] = rchc % AH;
  int1 maskh[TM] = (rch >= pad_h) && (rch < (AH - pad_h));
  int1 maskw[TM] = (rcw >= pad_w) && (rcw < (AW - pad_w));
  int1 interior[TM, TN] = maskh[:, newaxis] && maskw[:, newaxis];
  fp32* shiftpc[TM, TN] = pc + 0;
  pc = interior ? shiftpc : pc;
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
