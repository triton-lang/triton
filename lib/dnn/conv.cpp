#include "triton/dnn/conv.h"

namespace triton{
namespace dnn{

conv::conv(int B, int NC,
     int D, int H, int W,
     int T, int R, int S, int NF,
     int stride_d, int stride_h, int stride_w,
     int pad_d, int pad_h, int pad_w,
     int upsample_d, int upsample_h, int upsample_w,
     type ty, bool bias)
  : NB_(B), NC_(NC), AD_(D), AH_(H), AW_(W), BD_(T), BH_(R), BW_(S), NF_(NF),
    stride_d_(stride_d), stride_h_(stride_h), stride_w_(stride_w),
    pad_d_(pad_d), pad_h_(pad_h), pad_w_(pad_w),
    upsample_d_(upsample_d), upsample_h_(upsample_h), upsample_w_(upsample_w),
    ty_(ty), bias_(bias)
{
  CD_ = (AD_*upsample_d_ - BD_ + 1 + 2*pad_d_ + stride_d_ - 1)/stride_d_;
  CH_ = (AH_*upsample_h_ - BH_ + 1 + 2*pad_h_ + stride_h_ - 1)/stride_h_;
  CW_ = (AW_*upsample_w_ - BW_ + 1 + 2*pad_w_ + stride_w_ - 1)/stride_w_;
  // shapes
  shapes_a_ = {NB_, NC_, AD_, AH_, AW_};
  shapes_b_ = {NC_, BD_, BH_, BW_, NF_};
  shapes_c_ = {NB_, NF_, CD_, CH_, CW_};
  // a layout - NCHW
  a_outer_idx_ = 0;
  a_inner_idx_ = 1;
  a_pix_idx_ = 2;
  // b layout - CRSK
  b_inner_idx_ = 0;
  b_pix_idx_ = 1;
  b_outer_idx_ = 4;
  // c layout - NKPQ
  c_outer_0_idx_ = 0;
  c_outer_1_idx_ = 1;
  c_pix_idx = 2;
  // swap a and c for bprop
  if(ty_ == BPROP){
    std::swap(AD_, CD_);
    std::swap(AH_, CH_);
    std::swap(AW_, CW_);
    shapes_a_.swap(shapes_c_);
    pad_d_ = (CD_*stride_d_ - AD_*upsample_d_ + BD_ - 1 - stride_d_ + 1)/2;
    pad_h_ = (CH_*stride_h_ - AH_*upsample_h_ + BH_ - 1 - stride_h_ + 1)/2;
    pad_w_ = (CW_*stride_w_ - AW_*upsample_w_ + BW_ - 1 - stride_w_ + 1)/2;
  }
  // swap b and c for wgrad
  if(ty_ == WGRAD){
    shapes_b_.swap(shapes_c_);
    std::swap(BD_, CD_);
    std::swap(BH_, CH_);
    std::swap(BW_, CW_);
    std::swap(a_outer_idx_, a_inner_idx_);
    std::swap(b_inner_idx_, c_outer_0_idx_);
    std::swap(b_outer_idx_, c_outer_1_idx_);
    std::swap(b_pix_idx_, c_pix_idx);
  }
  // leading dimensions
  auto set_ld = [](const std::vector<int32_t>& shapes,
               std::vector<int32_t>& ld) {
    size_t size = shapes.size();
    ld.resize(size);
    ld[4] = 1;
    ld[3] = shapes[4]*ld[4];
    ld[2] = shapes[3]*ld[3];
    ld[1] = shapes[2]*ld[2];
    ld[0] = shapes[1]*ld[1];
  };
  set_ld(shapes_a_, ld_a_);
  set_ld(shapes_b_, ld_b_);
  set_ld(shapes_c_, ld_c_);
  // equivalent matmul
  b_trans_ = ty_ != BPROP;
  b_lut_ = ty_ == WGRAD;
  if(ty_ == WGRAD) {
    M_ = shapes_c_[0]*shapes_c_[1]*shapes_c_[2]*shapes_c_[3];
    N_ = shapes_c_[4];
    K_ = shapes_b_[0]*shapes_b_[2]*shapes_b_[3]*shapes_b_[4];
  }
  else {
    M_ = shapes_c_[0]*shapes_c_[2]*shapes_c_[3]*shapes_c_[4];
    N_ = shapes_c_[1];
    K_ = shapes_b_[0]*shapes_b_[1]*shapes_b_[2]*shapes_b_[3];
  }
  // look-up table info
  if(ty_ == FPROP)
    Fs_ = shapes_b_[1]*shapes_b_[2]*shapes_b_[3];
  else
    Fs_ = K_;
  TK_ = 8;
  Luts_ = (TK_ + Fs_ - 1) / Fs_ * Fs_;
  build_deltas();
  build_masks();
  size_t cst_size = h_b_deltas_.size()*4;
  is_b_deltas_cst_  = cst_size < 65536;
  cst_size += h_a_deltas_.size()*4;
  is_a_deltas_cst = cst_size < 65536;
  cst_size += h_masks_.size()*4;
  is_mask_cst_ = cst_size < 65536;
}

size_t conv::a_size()
{ return std::accumulate(shapes_a_.begin(), shapes_a_.end(),
                         1, std::multiplies<int>()); }

size_t conv::b_size()
{ return std::accumulate(shapes_b_.begin(), shapes_b_.end(),
                         1, std::multiplies<int>()); }

size_t conv::c_size()
{ return std::accumulate(shapes_c_.begin(), shapes_c_.end(),
                         1, std::multiplies<int>()); }

std::vector<int32_t> conv::c_shapes()
{ return shapes_c_; }

void conv::build_deltas(){
  h_a_deltas_.resize(Luts_ + upsample_d_*upsample_h_*upsample_w_*Luts_);
  if(b_lut_)
    h_b_deltas_.resize(Luts_);

  auto unpack = [&](int32_t ltrs) {
    int32_t l = (!b_trans_) ? ltrs % NF_ : ltrs / (BD_*BH_*BW_);
    int32_t trs = (!b_trans_) ? ltrs / NF_ : ltrs % (BD_*BH_*BW_);
    int32_t tr = trs / BW_;
    int32_t s = trs % BW_;
    int32_t t = tr / BH_;
    int32_t r = tr % BH_;
    if(!b_trans_){
      r = BH_ - 1 - r;
      s = BW_ - 1 - s;
    }
    return std::make_tuple(l, t, r, s);
  };

  for(size_t i = 0; i < Luts_; ++i)
    h_a_deltas_[i] = (((i + TK_) % Luts_) - i);

  size_t Ds0 = Luts_;
  size_t Ds1 = upsample_w_;
  size_t Ds2 = upsample_h_;
  size_t Ds3 = upsample_d_;
  for(size_t pd = 0; pd < Ds3; ++pd)
  for(size_t ph = 0; ph < Ds2; ++ph)
  for(size_t pw = 0; pw < Ds1; ++pw) {
    int32_t* deltas_ptr = &h_a_deltas_[Luts_ + pw*Ds0 + ph*Ds0*Ds1 + pd*Ds0*Ds1*Ds2];
    // cumulative increments
    for(size_t i = 0; i < Ds0; ++i) {
      // unpack
      int32_t ctrs = i;
      int32_t c, t, r, s;
      std::tie(c, t, r, s) = unpack(ctrs);
      // next indices
      int32_t nextctrs = ctrs + TK_;
      int32_t nextc, nextt, nextr, nexts;
      std::tie(nextc, nextt, nextr, nexts) = unpack(nextctrs);
      // diffs
      int32_t cdiff = nextc - c;
      int32_t tdiff = (nextt + pd)/upsample_d_ - (t + pd)/upsample_d_;
      int32_t rdiff = (nextr + ph)/upsample_h_ - (r + ph)/upsample_h_;
      int32_t sdiff = (nexts + pw)/upsample_w_ - (s + pw)/upsample_w_;
      // delta pointers
      deltas_ptr[i] = cdiff*ld_a_[a_inner_idx_] + tdiff*ld_a_[a_pix_idx_] + rdiff*ld_a_[a_pix_idx_ + 1] + sdiff*ld_a_[a_pix_idx_ + 2];
    }
  }

  if(b_lut_) {
    for(size_t i = 0; i < Ds0; ++i) {
      int32_t c, t, r, s;
      int32_t nextc, nextt, nextr, nexts;
      std::tie(c, t, r, s) = unpack(i);
      std::tie(nextc, nextt, nextr, nexts) = unpack(i + TK_);
      int32_t cdiff = nextc - c;
      int32_t tdiff = nextt - t;
      int32_t rdiff = nextr - r;
      int32_t sdiff = nexts - s;
      h_b_deltas_[i] = cdiff*ld_b_[b_inner_idx_] + tdiff*ld_b_[b_pix_idx_] + rdiff*ld_b_[b_pix_idx_ + 1] + sdiff*ld_b_[b_pix_idx_ + 2];
    }
  }
}

void conv::build_masks(){
  h_masks_.resize(Luts_ + (2*pad_h_+1)*(2*pad_w_+1)*(2*pad_d_+1)*Luts_);

  auto unpack = [&](int32_t ltrs){
    int32_t l = (!b_trans_) ? ltrs % NF_ : ltrs / (BD_*BH_*BW_);
    int32_t trs = (!b_trans_) ? ltrs / NF_ : ltrs % (BD_*BH_*BW_);
    int32_t tr = trs / BW_;
    int32_t s = trs % BW_;
    int32_t t = tr / BH_;
    int32_t r = tr % BH_;
    if(!b_trans_){
      r = BH_ - 1 - r;
      s = BW_ - 1 - s;
    }
    return std::make_tuple(l, t, r, s);
  };
  size_t Ms0 = Luts_;
  size_t Ms1 = 2*pad_w_ + 1;
  size_t Ms2 = 2*pad_h_ + 1;
  size_t Ms3 = 2*pad_d_ + 1;
  for(size_t pd = 0; pd < Ms3; ++pd)
  for(size_t ph = 0; ph < Ms2; ++ph)
  for(size_t pw = 0; pw < Ms1; ++pw){
    int32_t* masks_ptr = &h_masks_[Luts_ + pw*Ms0 + ph*Ms0*Ms1 + pd*Ms0*Ms1*Ms2];
    for(size_t i = 0; i < Ms0; ++i){
       int32_t l, t, r, s;
       int32_t mask = 0x0;
       for(size_t j = 0; j < TK_; ++j){
         std::tie(l, t, r, s) = unpack(i + j);
         bool in_bounds_d = (t + pd) >= pad_d_ && (t + pd) < (BD_ + pad_d_);
         bool in_bounds_h = (r + ph) >= pad_h_ && (r + ph) < (BH_ + pad_h_);
         bool in_bounds_w = (s + pw) >= pad_w_ && (s + pw) < (BW_ + pad_w_);
         mask |= (in_bounds_d && in_bounds_h && in_bounds_w) << j;
       }
       masks_ptr[i] = mask;
    }
  }
  for(size_t i = 0; i < Luts_; ++i)
    h_masks_[i] = 0x0;
}

std::array<size_t, 3> conv::get_grid(size_t TM, size_t TN)
{ return {(M_ + TM - 1)/TM, (N_ + TN - 1)/TN, 1}; }

size_t conv::get_nflops()
{ return 2.*M_*N_*K_; }

void conv::init(driver::stream *stream, triton::driver::cu_module* module) {
  auto init_lut = [&](bool is_cst, const char *name, std::vector<int32_t> host) -> triton::driver::buffer*{
    if(host.empty())
      return nullptr;
    size_t nbytes = host.size()*4;
    // get buffer
    triton::driver::buffer* buffer;
    if(is_cst)
      buffer = module->symbol(name);
    else
      buffer = triton::driver::buffer::create(stream->context(), nbytes);
    // copy
    stream->write(buffer, false, 0, nbytes, host.data());
    return buffer;
  };

  d_a_deltas_ = init_lut(is_a_deltas_cst, "delta", h_a_deltas_);
  d_b_deltas_ = init_lut(is_b_deltas_cst_, "b_delta", h_b_deltas_);
  d_masks_ = init_lut(is_mask_cst_, "masks", h_masks_);
}

void conv::set_arg(driver::kernel *kernel,
                    driver::buffer *a, driver::buffer *b, driver::buffer *c, driver::buffer *bias)
{
  kernel->setArg(0, a);
  kernel->setArg(1, b);
  kernel->setArg(2, c);
  kernel->setArg(3, bias);
  kernel->setArg(4, M_);
  kernel->setArg(5, N_);
  kernel->setArg(6, K_);
  kernel->setArg(7, AH_);
  kernel->setArg(8, AW_);
  kernel->setArg(9, BH_);
  kernel->setArg(10, BW_);
  kernel->setArg(11, CH_);
  kernel->setArg(12, CW_);
  // A arguments
  kernel->setArg(13, ld_a_[a_outer_idx_]);
  kernel->setArg(14, ld_a_[a_inner_idx_]);
  kernel->setArg(15, ld_a_[2]);
  kernel->setArg(16, ld_a_[3]);
  kernel->setArg(17, ld_a_[4]);
  // B arguments
  kernel->setArg(18, ld_b_[b_inner_idx_]);
  kernel->setArg(19, ld_b_[b_pix_idx_]);
  kernel->setArg(20, ld_b_[b_pix_idx_+1]);
  kernel->setArg(21, ld_b_[b_pix_idx_+2]);
  kernel->setArg(22, ld_b_[b_outer_idx_]);
  // C arguments
  kernel->setArg(23, ld_c_[c_outer_0_idx_]);
  kernel->setArg(24, ld_c_[c_outer_1_idx_]);
  kernel->setArg(25, ld_c_[c_pix_idx]);
  kernel->setArg(26, ld_c_[c_pix_idx+1]);
  kernel->setArg(27, ld_c_[c_pix_idx+2]);
  // pad
  kernel->setArg(28, pad_h_);
  kernel->setArg(29, pad_w_);
  // stride
  kernel->setArg(30, stride_h_);
  kernel->setArg(31, stride_w_);
  // dilate
  kernel->setArg(32, upsample_h_);
  kernel->setArg(33, upsample_w_);
  size_t idx = 34;
  if(!is_a_deltas_cst)
    kernel->setArg(idx++, d_a_deltas_);
  if(!is_b_deltas_cst_)
    kernel->setArg(idx++, d_b_deltas_);
  if(!is_mask_cst_)
    kernel->setArg(idx++, d_masks_);
}

std::vector<unsigned> conv::default_params() {
  if(ty_==FPROP)
    return {16, 2, 64, 32, 2, 64, 16, 8, 2, 2, 8, 1, 8, 4};
  else if(ty_ == BPROP)
    return {32, 2, 64, 32, 64, 32, 4, 2, 2, 4, 2, 8, 4, 2};
  else if(ty_ == WGRAD)
    return {32, 2, 64, 32, 2, 64, 16, 8, 2, 2, 4, 2, 8};
}


template<class IN_DTYPE, class OUT_DTYPE>
void conv::cpu_xprop(OUT_DTYPE* C,  IN_DTYPE* A, IN_DTYPE* B)
{
  IN_DTYPE acc;
  for(int32_t n = 0; n < shapes_c_[0]; ++n)
  for(int32_t cf = 0; cf < shapes_c_[1] ; ++cf)
  for(int32_t cd = 0 ; cd < shapes_c_[2]; ++cd)
  for(int32_t ch = 0 ; ch < shapes_c_[3]; ++ch)
  for(int32_t cw = 0; cw < shapes_c_[4]; ++cw)
  {
    acc = 0;
    int32_t d = cd*stride_d_ - pad_d_;
    int32_t h = ch*stride_h_ - pad_h_;
    int32_t w = cw*stride_w_ - pad_w_;
    for(int32_t ac = 0; ac < shapes_a_[1]; ++ac)
    for(int32_t bd = 0; bd < shapes_b_[1]; ++bd)
    for(int32_t bh = 0; bh < shapes_b_[2]; ++bh)
    for(int32_t bw = 0; bw < shapes_b_[3]; ++bw){
      int32_t ad = d + bd;
      int32_t ah = h + bh;
      int32_t aw = w + bw;
      bool in_bounds = (ad >= 0 && ad < shapes_a_[2] &&
                        ah >= 0 && ah < shapes_a_[3] &&
                        aw >= 0 && aw < shapes_a_[4]);
      IN_DTYPE a = 0;
      if(in_bounds)
        a = A[n*ld_a_[0] + ac*ld_a_[1] + ad*ld_a_[2] + ah*ld_a_[3] + aw*ld_a_[4]];
      IN_DTYPE b;
      if(b_trans_)
        b = B[ac*ld_b_[0] + bd*ld_b_[1] + bh*ld_b_[2] + bw*ld_b_[3] + cf*ld_b_[4]];
      else{
        int32_t bdd = shapes_b_[1] - 1 - bd;
        int32_t bhh = shapes_b_[2] - 1 - bh;
        int32_t bww = shapes_b_[3] - 1 - bw;
        b = B[cf*ld_b_[0] + bdd*ld_b_[1] + bhh*ld_b_[2] + bww*ld_b_[3] + ac*ld_b_[4]];
      }
      acc = std::fma(a, b, acc);
    }
    C[n*ld_c_[0] + cf*ld_c_[1] + cd*ld_c_[2] + ch*ld_c_[3] + cw*ld_c_[4]] = acc;
  }
}

template<class IN_DTYPE, class OUT_DTYPE>
void conv::cpu_wgrad(OUT_DTYPE* C,  IN_DTYPE* A, IN_DTYPE* B)
{
  IN_DTYPE acc;
  for(int32_t c = 0 ; c  < shapes_c_[0]; ++c)
  for(int32_t cd = 0; cd < shapes_c_[1]; ++cd)
  for(int32_t ch = 0; ch < shapes_c_[2]; ++ch)
  for(int32_t cw = 0; cw < shapes_c_[3]; ++cw)
  for(int32_t k = 0 ; k  < shapes_c_[4]; ++k)
  {
    acc = 0;
    int32_t d = cd*stride_d_ - pad_d_;
    int32_t h = ch*stride_h_ - pad_h_;
    int32_t w = cw*stride_w_ - pad_w_;
    for(int32_t n = 0; n   < shapes_b_[0]; ++n)
    for(int32_t bd = 0; bd < shapes_b_[2]; ++bd)
    for(int32_t bh = 0; bh < shapes_b_[3]; ++bh)
    for(int32_t bw = 0; bw < shapes_b_[4]; ++bw){
      int32_t ad = d + bd;
      int32_t ah = h + bh;
      int32_t aw = w + bw;
      bool in_bounds = (ad >= 0 && ad < shapes_a_[2] &&
                        ah >= 0 && ah < shapes_a_[3] &&
                        aw >= 0 && aw < shapes_a_[4]);
      IN_DTYPE a = 0;
      if(in_bounds)
        a = A[n*ld_a_[0] + c*ld_a_[1] + ad*ld_a_[2] + ah*ld_a_[3] + aw*ld_a_[4]];
      IN_DTYPE b = B[n*ld_b_[0] + k*ld_b_[1] + bd*ld_b_[2] + bh*ld_b_[3] + bw*ld_b_[4]];
      acc = std::fma(a, b, acc);
    }
    C[c*ld_c_[0] + cd*ld_c_[1] + ch*ld_c_[2] + cw*ld_c_[3] + k*ld_c_[4]] = acc;
  }
}

template<class IN_DTYPE, class OUT_DTYPE>
void conv::cpu_ref(OUT_DTYPE* C,  IN_DTYPE* A, IN_DTYPE* B)
{
  if(ty_ == FPROP || ty_ == BPROP)
    cpu_xprop(C, A, B);
  else
    cpu_wgrad(C, A, B);
}

template void conv::cpu_ref<float,float>(float*, float*, float*);
template void conv::cpu_xprop<float,float>(float*, float*, float*);
template void conv::cpu_wgrad<float,float>(float*, float*, float*);

}
}
