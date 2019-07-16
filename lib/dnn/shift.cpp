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
             type ty, bool bias,
             layout_t layout)
  : base("shift"),
    B_(B), C_(C),
    AD_(D), AH_(H), AW_(W),
    BD_(T), BH_(R), BW_(S),
    F_(F),
    stride_d_(1), stride_h_(stride_h), stride_w_(stride_w),
    shift_h_(shift_h), shift_w_(shift_w),
    a_ty_(a_ty), b_ty_(b_ty), c_ty_(b_ty),
    op_(ty), bias_(bias),
    layout_(layout){
//  std::cout << B_ << " " << C_ << " " << F_ << " " << stride_h_ << " " << stride_w_ << " " << a_ty_ << " " << b_ty_ << " " << ty_ << " " << layout_ << std::endl;
  // max number of channels
  TK_ = (ty == FPROP && a_ty_ == "fp32") ? 8 : 16;
  MAX_C_ = 8192 + TK_;
  // activation sizes
  CD_ = AD_ / stride_d_;
  CH_ = AH_ / stride_h_;
  CW_ = AW_ / stride_w_;
  // A memory strides: [C, H, W, B]
  switch(layout_){
  case CHWN: {
    lda_n_ = 1;
    lda_w_ = B_;
    lda_h_ = B_*AW_;
    lda_c_ = B_*AW_*AH_;
    break;
  }
  case NCHW: {
    lda_w_ = 1;
    lda_h_ = AW_;
    lda_c_ = AW_*AH_;
    lda_n_ = AW_*AH_*C_;
    break;
  }
  default:
    throw std::runtime_error("unsupported input layout");
  }
  // Shift edge
  shift_edge_h_ = (AH_ == stride_h_ && stride_h_ > 1);
  shift_edge_w_ = (AW_ == stride_w_ && stride_w_ > 1);
  // B memory strides: [C, F]
  ldb_n_ = 1;
  ldb_h_ = 1;
  ldb_w_ = 1;
  ldb_c_ = F_;
  // C memory strides: [F, H, W, B]
  switch(layout_){
  case CHWN: {
    ldc_n_ = 1;
    ldc_w_ = B_;
    ldc_h_ = B_*CW_;
    ldc_f_ = B_*CW_*CH_;
    break;
  }
  case NCHW: {
    ldc_w_ = 1;
    ldc_h_ = CW_;
    ldc_f_ = CW_*CH_;
    ldc_n_ = CW_*CH_*F_;
    break;
  }
  default:
    throw std::runtime_error("unsupported input layout");
  }
  // Equivalent matmul
  M_ = B_*CH_*CW_;
  N_ = F_;
  K_ = C_;
  // transpose
  AT_ = false;
  BT_ = true;
  // C shapes
  if(layout_ == CHWN)
    shapes_c_ = {F, CH_, CW_, B};
  if(layout_ == NCHW)
    shapes_c_ = {B, F, CH_, CW_};
  // Weight gradient
  if(op_ == WGRAD){
    // b <-> c
    // b <-> a
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
  if(op_ == BPROP){
    // a <-> c
    std::swap(lda_n_, ldc_n_);
    std::swap(lda_w_, ldc_w_);
    std::swap(lda_h_, ldc_h_);
    std::swap(lda_c_, ldc_f_);
    std::swap(K_, N_);
    AT_ = false;
    BT_ = false;
    if(layout_ == CHWN)
      shapes_c_ = {C, AH_, AW_, B};
    if(layout_ == NCHW)
      shapes_c_ = {B, C, AH_, AW_};
  }
  // locks
  max_locks_ = (op_ == WGRAD) ? 8192 : 0;
  locks_ = nullptr;
}

base* shift::clone() const {
  return new shift(*this);
}

void shift::build_delta_a() {
  h_delta_a.resize(MAX_C_);
  auto shift_h = [&](int c) { return shift_edge_h_ ? (c / AH_) % AH_ : shift_h_[c]; };
  auto shift_w = [&](int c) { return shift_edge_w_ ? c % AW_ : shift_w_[c]; };
  if(op_ == FPROP){
    // compute offset
    auto offset = [&](unsigned c) {
      return c*lda_c_ + shift_h(c)*lda_h_ + shift_w(c)*lda_w_;
    };
    // populate look-up table
    for(unsigned c = 0; c < TK_; c++)
      h_delta_a[c] =  offset(c);
    for(unsigned c = 0; c < C_; c++)
      h_delta_a[TK_ + c] = offset(c + TK_) - offset(c);
  }
  if(op_ == BPROP){
    for(unsigned c = 0; c < C_; c++){
      h_delta_a[c] = shift_h(c)*ldc_h_ + shift_w(c)*ldc_w_;
    }
  }
  if(op_ == WGRAD){
    for(unsigned c = 0; c < C_; c++)
      h_delta_a[c] = shift_h(c)*ldb_h_ + shift_w(c)*ldb_w_;
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
  return std::tie(B_, C_, F_,
                  AD_, AH_, AW_,
                  BD_, BH_, BW_,
                  CD_, CH_, CW_,
                  shift_h_, shift_w_,
                  stride_h_, stride_w_,
                  layout_, op_,
                  bias_)
       < std::tie(y->B_, y->C_, y->F_,
                  y->AD_, y->AH_, y->AW_,
                  y->BD_, y->BH_, y->BW_,
                  y->CD_, y->CH_, y->CW_,
                  y->shift_h_, y->shift_w_,
                  y->stride_h_, y->stride_w_,
                  y->layout_, y->op_,
                  y->bias_);
}

void shift::init_impl(driver::stream *stream, driver::cu_module *module) {
  build_delta_a();
  triton::driver::buffer* delta_a = ((triton::driver::cu_module*)module)->symbol("delta_a");
  stream->write(delta_a, false, 0, h_delta_a.size()*4, h_delta_a.data());
  // locks
  if(locks_ == nullptr && max_locks_ > 0){
    std::vector<int32_t> hlocks(2*max_locks_, 0);
    locks_ = triton::driver::buffer::create(stream->context(), 2*max_locks_*4);
    stream->write(locks_, false, 0, hlocks);
  }
}

void shift::deinit_impl() {
  if(locks_ != nullptr){
    delete locks_;
    locks_ = nullptr;
  }
}

void shift::enqueue_impl(driver::stream *stream, driver::kernel *kernel,
                    std::vector<driver::buffer *> args,
                    runtime::launch_information info) {
  unsigned TM = info.global_range_size[0], TN = info.global_range_size[1];
  unsigned grid_0 = (M_ + TM - 1)/TM;
  unsigned grid_1 = (N_ + TN - 1)/TN;
  unsigned num_locks = grid_0 * grid_1;
  unsigned grid_2 = num_locks < max_locks_ ? info.globals.at("GZ") : 1;
  std::array<size_t, 3> grid = {grid_0, grid_1, grid_2};
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
  kernel->setArg(27, (num_locks > max_locks_) ? nullptr : locks_);
  kernel->setArg(28, (int32_t)grid[0]);
  kernel->setArg(29, (int32_t)grid[1]);
  if(op_ == BPROP){
    size_t c_nbytes = (c_ty_ == "fp16") ? 2 : 4;
    ((driver::cu_buffer*)c)->set_zero(stream, AH_*AW_*B_*C_*c_nbytes);
  }
  stream->enqueue(kernel, grid, {info.num_threads, 1, 1});
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
  bool is_chwn = layout_ == CHWN;

  auto compute_bhw = [&](std::string rx, std::string sz, std::string rkx){
    if(is_chwn) {
      return R"(
  int32 )" + rx + "wh[" + sz + "] =  "  + rkx + R"( / NB;
  int32 )" + rx + "b[" + sz + "]  =  "  + rkx + R"( % NB;
  int32 )" + rx + "w[" + sz + "]  =  "  + rx  + R"(wh % CW;
  int32 )" + rx + "h[" + sz + "]  =  "  + rx  + R"(wh / CW;)";
    }
    else {
      return R"(
  int32 )" + rx + "bh[" + sz + "] = " + rkx + R"( / CW;
  int32 )" + rx + "w[" + sz + "]  = " + rkx + R"( % CW;
  int32 )" + rx + "h[" + sz + "]  = " + rx  + R"(bh % CH;
  int32 )" + rx + "b[" + sz + "]  = " + rx  + R"(bh / CH;)";
    }
  };

  auto compute_interior = [&](std::string rx, std::string sz0, std::string sz1) {
    std::string result;
    if(shift_edge_h_)
      result +=  "int1 interiorh[" + sz0 + "] = 1;\n  ";
    else
      result +=  "int1 interiorh[" + sz0 + "] = (" + rx + "h >= pad_h) && (" + rx + "h < (AH - pad_h));\n  ";
    if(shift_edge_w_)
      result +=  "int1 interiorw[" + sz0 + "] = 1;";
    else
      result +=  "int1 interiorw[" + sz0 + "] = (" + rx + "w >= pad_w) && (" + rx + "w < (AW - pad_w));";
    result +=  R"(
    int1 interior[)" + sz0 + ", " + sz1 + "] = interiorh[:, newaxis] && interiorw[:, newaxis];";
    return result;
  };

  std::string result =
R"(
const tunable int32 TM = {16, 32, 64, 128};
const tunable int32 TN = {16, 32, 64, 128};
const tunable int32 TK = {)" + std::to_string(TK_) + "};";
if(op_ == WGRAD)
  result += "const tunable int32 GZ = {1, 4, 16};";
else
  result += "const tunable int32 GZ = {1};";

result += R"(
__constant__ int32* delta_a = alloc_const int32[)" + std::to_string(MAX_C_) + R"(];

void shift(restrict read_only align(16) )" + a_ty_ + R"( *A,
           restrict read_only align(16) )" + b_ty_ + R"( *B,
           )" + c_ty_ + R"( *C,
           int32 M, int32 N, int32 K,
           int32 stride_h, int32 stride_w,
           int32 lda_b, int32 lda_w, int32 lda_h, int32 lda_c,
           int32 ldb_b, int32 ldb_w, int32 ldb_h, int32 ldb_c,
           int32 ldc_b, int32 ldc_w, int32 ldc_h, int32 ldc_c,
           int32 NB,
           int32 AH, int32 AW,
           int32 BH, int32 BW,
           int32 CH, int32 CW,
           int32* locks, int32 grid0, int32 grid1) {
  int32 rxa[TM] = get_global_range[TM](0);
  int32 ryb[TN] = get_global_range[TN](1);
  int32 rz = get_global_range[1](2);
  int32 rka[TK] = 0 ... TK;
  int32 rkb[TK] = 0 ... TK;
  fp32 acc[TM, TN] = 0;
  int32 pad_h = BH / 2;
  int32 pad_w = BW / 2;
  int32 split = select(locks == 0, 1, GZ);
  int32 div = K / split;
  int32 rem = K % split;
  K = select(rz < rem, div - 1, div);
  int32 offk = select(rz < rem, rz*(div + 1), rz*div + rem);)";
if(op_ == WGRAD){
  result += R"(
  rka = rka + offk;
  rkb = rkb + offk;
  )";
}

/* A offsets */
if(op_ == FPROP){
  result +=
  compute_bhw("ra", "TM", "rxa") + R"(
  raw = raw * stride_w;
  rah = rah * stride_h;
  int32 offxa[TM] =  rab*lda_b + raw*lda_w + rah*lda_h;
  int32 offa0[TM, TK] = offxa[:, newaxis];
  __constant__ int32* pd[TK] = delta_a + rka;
  int32 d[TK] = *pd;
  int32 offa_interior[TM, TK] = d[newaxis, :];
  int32 offa_exterior[TM, TK] = rka[newaxis, :] * lda_c;
  )" + compute_interior("ra", "TM", "TK") + R"(
  int32 offa1[TM, TK] = interior ? offa_interior : offa_exterior;)";
}
if(op_ == BPROP){
  result +=
  compute_bhw("ra", "TM", "rxa") + R"(
  int32 offxa[TM] =  rab*lda_b + raw*lda_w + rah*lda_h;
  int32 offa0[TM, TK] = offxa[:, newaxis];
  int32 offa1[TM, TK] = rka[newaxis, :] * lda_c;)";
}
if(op_ == WGRAD && layout_ == CHWN){
  result +=  R"(
  int32 offa0[TK, TM] = rxa[newaxis, :] * lda_c;
  int32 offa1[TK, TM] = rka[:, newaxis];)";
}
if(op_ == WGRAD && layout_ == NCHW){
  result +=
  compute_bhw("ra", "TK", "rka") + R"(
  int32 offa0[TK, TM] = rxa[newaxis, :] * lda_c;
  int32 offxa[TK] =  rab*lda_b + raw*lda_w + rah*lda_h;
  int32 offa1[TK, TM] = offxa[:, newaxis];)";
}

/* B offsets */
if(op_ == FPROP){
  result +=  R"(
  int32 offb0[TN, TK] = ryb[:, newaxis];
  int32 offb1[TN, TK] = rkb[newaxis, :] * ldb_c;)";
}
if(op_ == BPROP){
  result +=  R"(
  int32 offb0[TK, TN] = ryb[newaxis, :] * ldb_c;
  int32 offb1[TK, TN] = rkb[:, newaxis];)";
}
if(op_ == WGRAD){
  result +=
  compute_bhw("rb", "TK", "rkb") + R"(
  __constant__ int32* pd[TN] = delta_a + ryb;
  int32 d[TN] = *pd;
  int32 shift[TK, TN] = d[newaxis, :];
  rbw = rbw * stride_w;
  rbh = rbh * stride_h;
  int32 offkb[TK] = rbb*ldb_b + rbw*ldb_w + rbh*ldb_h;
  )" + compute_interior("rb", "TK", "TN") + R"(
  int32 incb[TK, TN] = interior ? shift : 0;
  int32 offb0[TK, TN] = ryb[newaxis, :] * ldb_c;
  int32 offb1[TK, TN] = offkb[:, newaxis] + incb;)";
}

/* Main loop */
  result +=  R"(
  )" + a_ty_ + "* pa[" + AS + R"(] = A + offa0 + offa1;
  )" + b_ty_ + "* pb[" + BS + R"(] = B + offb0 + offb1;
  int1 checka[)" + AS + "] = (rka < K + offk)" + bca0  + R"(;
  int1 checkb[)" + BS + "] = (rkb < K + offk)" + bcb0  + R"(;
  )" + a_ty_ + "   a[" + AS + R"(] = checka ? *pa : 0;
  )" + b_ty_ + "   b[" + BS + R"(] = checkb ? *pb : 0;
  for(int32 k = K; k > 0; k = k - TK){
    acc = dot()" + usea + "," + useb + R"(, acc);
    int1 checka[)" + AS + R"(] = k > TK;
    int1 checkb[)" + BS + R"(] = k > TK;)";

/* Increment A pointers */
if(op_ == FPROP){
  result +=  R"(
    pd = pd + TK;
    d = *pd;
    offa_interior = d[newaxis, :];
    offa_exterior = TK * lda_c;
    int32 offa[TM, TK] = interior ? offa_interior : offa_exterior;
    pa = pa + offa;)";
}
if(op_ == BPROP){
  result +=  R"(
    pa = pa + TK * lda_c;)";
}
if(op_ == WGRAD && layout_ == CHWN){
  result +=  R"(
    pa = pa + TK;)";
}
if(op_ == WGRAD && layout_ == NCHW){
  result += R"(
    rka = rka + TK;)"
    + compute_bhw("ra", "TK", "rka") + R"(
    offxa =  rab*lda_b + raw*lda_w + rah*lda_h;
    pa = A + offa0 + offxa[:, newaxis];)";
}
  result +=  R"(
    @checka a = *pa;)";

/* Increment B pointers */
if(op_ == WGRAD){
  result += R"(
    rkb = rkb + TK;)"
    + compute_bhw("rb", "TK", "rkb") + R"(
    rbw = rbw * stride_w;
    rbh = rbh * stride_h;
    offkb = rbb*ldb_b + rbw*ldb_w + rbh*ldb_h;
    )" + compute_interior("rb", "TK", "TN") + R"(
    incb   = interior ? shift : 0;
    pb = B + offb0 + offkb[:, newaxis] + incb;)";
}
if(op_ == FPROP){
  result +=  R"(
    pb = pb + TK * ldb_c;)";
}
if(op_ == BPROP){
  result +=  R"(
    pb = pb + TK;)";
}
  result +=  R"(
    @checkb b = *pb;
  }
  int32 rxc[TM] = get_global_range[TM](0);
  int32 ryc[TN] = get_global_range[TN](1);)";

/* C offsets */
if(op_ == BPROP){
  result +=
  compute_bhw("rc", "TM", "rxc") + R"(
  rcw = rcw * stride_w;
  rch = rch * stride_h;
  int32 offxc[TM] = rcb*ldc_b + rcw*ldc_w + rch*ldc_h;)";
  }
if(op_ == FPROP){
  result +=
  compute_bhw("rc", "TM", "rxc") + R"(
  int32 offxc[TM] = rcb*ldc_b + rcw*ldc_w + rch*ldc_h;)";
}
if(op_ == WGRAD){
  result +=  R"(
  int32 offxc[TM] = rxc;)";
}
  result +=  R"("
  )" + c_ty_ + R"( c[TM, TN] = acc;
  )" + c_ty_ + R"(* pc[TM, TN] = C + offxc[:, newaxis] + ryc[newaxis, :]*ldc_c;
  int1 checkc0[TM] = rxc < M;
  int1 checkc1[TN] = ryc < N;
  int1 checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];)";
if(op_ == BPROP){
  result += R"(
  )" + compute_interior("rc", "TM", "TN") + R"(
  __constant__ int32* pd[TN] = delta_a + ryc;
  )" + c_ty_ + R"(* shift_pc[TM, TN] = pc + (*pd)[newaxis, :];
  pc = interior ? shift_pc : pc;
  @checkc __atomic_add(pc, c);
  )";
}
else{
  result +=  R"(
  int1 has_lock = (GZ > 1) && (locks != 0);
  if(has_lock){
    int32 ridx = get_range_id(0);
    int32 ridy = get_range_id(1);
    int32 *plock = locks + ridx + ridy*grid0;
    while(__atomic_cas(plock, 0, 1));
    int32 *pcount = plock + grid0*grid1;
    int32 count = *pcount;
    int32 countp1 = select(count == split - 1, 0, count + 1);
    if(count == 0) {
      @checkc *pc = c;
      *pcount = countp1;
    }
    else {
      @checkc *pc = c + *pc;
      *pcount = countp1;
    }
    *plock = 0;
  }
  else{
    @checkc *pc = c;
  })";
}
  result +=  R"(
})";

  os << result;
}

}
}
