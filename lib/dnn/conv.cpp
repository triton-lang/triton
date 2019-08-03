#include <cmath>
#include "triton/dnn/conv.h"

namespace triton{
namespace dnn{

conv::conv(int B, int NC,
     int D, int H, int W,
     int T, int R, int S, int NF,
     int stride_d, int stride_h, int stride_w,
     int pad_d, int pad_h, int pad_w,
     int upsample_d, int upsample_h, int upsample_w,
     std::string a_ty, std::string b_ty,
     type ty, bool bias)
  : base("conv"),
    NB_(B), NC_(NC), AD_(D), AH_(H), AW_(W), BD_(T), BH_(R), BW_(S), NF_(NF),
    stride_d_(stride_d), stride_h_(stride_h), stride_w_(stride_w),
    pad_d_(pad_d), pad_h_(pad_h), pad_w_(pad_w),
    upsample_d_(upsample_d), upsample_h_(upsample_h), upsample_w_(upsample_w),
    a_ty_(a_ty), b_ty_(b_ty),
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
    std::swap(stride_d_, upsample_d_);
    std::swap(stride_h_, upsample_h_);
    std::swap(stride_w_, upsample_w_);
    pad_d_ = (CD_*stride_d_ - AD_*upsample_d_ + BD_ - 1 - stride_d_ + 1)/2;
    pad_h_ = (CH_*stride_h_ - AH_*upsample_h_ + BH_ - 1 - stride_h_ + 1)/2;
    pad_w_ = (CW_*stride_w_ - AW_*upsample_w_ + BW_ - 1 - stride_w_ + 1)/2;
    std::swap(b_inner_idx_, b_outer_idx_);
    std::swap(NC_, NF_);
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
  set_ld(shapes_a_, ld_a_);
  set_ld(shapes_b_, ld_b_);
  set_ld(shapes_c_, ld_c_);
  // equivalent matmul
  bool upsampled_b = (ty_ == BPROP) && (upsample_d_ > 1 || upsample_h_ > 1 || upsample_w_ > 1);
  b_trans_ = ty_ != BPROP;
  b_lut_ = ty_ == WGRAD || upsampled_b;
  M_ = shapes_c_[c_outer_0_idx_]*shapes_c_[c_pix_idx]*shapes_c_[c_pix_idx+1]*shapes_c_[c_pix_idx+2];
  N_ = shapes_c_[c_outer_1_idx_];
  K_ = shapes_b_[b_inner_idx_]*BD_*BH_*BW_;
  // look-up table info
  if(ty_ == FPROP)
    Fs_ = shapes_b_[1]*shapes_b_[2]*shapes_b_[3];
  else
    Fs_ = K_;
  TK_ = 8;
  Luts_ = (TK_ + Fs_ - 1) / Fs_ * Fs_;
  build_a_deltas();
  if(b_lut_)
    build_b_deltas();
  build_masks();
  size_t cst_size = h_b_deltas_.size()*4;
  is_b_deltas_cst_  = cst_size < 65536;
  cst_size += h_a_deltas_.size()*4;
  is_a_deltas_cst = cst_size < 65536;
  cst_size += h_masks_.size()*4;
  is_mask_cst_ = cst_size < 65536;
  max_grid_0_ = 256;
  max_grid_1_ = 256;
}

// comparison for maps
std::vector<int64_t> conv::retune_params() const {
  return {NB_, NC_, AD_, AH_, AW_,
          NF_, BD_, BH_, BW_,
          pad_d_, pad_h_, pad_w_,
          stride_d_, stride_h_, stride_w_,
          ty_, bias_};
}

// clone
base* conv::clone() const {
  return new conv(*this);
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


std::tuple<int32_t, int32_t, int32_t, int32_t> conv::unpack(int32_t ltrs, bool flip, int32_t EBD, int32_t EBH, int32_t EBW) {
  int32_t l, t, r, s;
  if(b_trans_){
    l = ltrs / (EBD*EBH*EBW);
    int32_t trs = ltrs % (EBD*EBH*EBW);
    int32_t tr = trs / EBW;
    s = trs % EBW;
    t = tr / EBH;
    r = tr % EBH;
  }
  else{
    int32_t rs = ltrs / NC_;
    l = ltrs % NC_;
    r = rs / EBW;
    s = rs % EBW;
  }
  if(flip){
    r = EBH - 1 - r;
    s = EBW - 1 - s;
  }
  return std::make_tuple(l, t, r, s);
}

void conv::build_b_deltas(){
  h_b_deltas_.resize(Luts_*upsample_d_*upsample_h_*upsample_w_);

  size_t Ds0 = Luts_;
  size_t Ds1 = upsample_w_;
  size_t Ds2 = upsample_h_;
  size_t Ds3 = upsample_d_;
  for(size_t ud = 0; ud < Ds3; ++ud)
  for(size_t uh = 0; uh < Ds2; ++uh)
  for(size_t uw = 0; uw < Ds1; ++uw) {
    int32_t* deltas_ptr = &h_b_deltas_[uw*Ds0 + uh*Ds0*Ds1 + ud*Ds0*Ds1*Ds2];
    for(size_t i = 0; i < Luts_; ++i) {
      int32_t EBD = 1;
      int32_t EBH = ((upsample_h_ - uh - 1) + BH_) / upsample_h_;
      int32_t EBW = ((upsample_w_ - uw - 1) + BW_) / upsample_w_;
      if(EBD == 0 || EBH == 0 || EBW == 0)
        continue;
      int32_t c, t, r, s;
      int32_t nextc, nextt, nextr, nexts;
      std::tie(c, t, r, s) = unpack(i, false, EBD, EBH, EBW);
      std::tie(nextc, nextt, nextr, nexts) = unpack(i + TK_, false, EBD, EBH, EBW);
      int32_t cdiff = nextc - c;
      int32_t tdiff = (nextt - t)*upsample_d_;
      int32_t rdiff = (nextr - r)*upsample_h_;
      int32_t sdiff = (nexts - s)*upsample_w_;
      deltas_ptr[i] = cdiff*ld_b_[b_inner_idx_] + tdiff*ld_b_[b_pix_idx_] + rdiff*ld_b_[b_pix_idx_ + 1] + sdiff*ld_b_[b_pix_idx_ + 2];
    }
  }
}

void conv::build_a_deltas(){
  h_a_deltas_.resize(Luts_ + upsample_d_*upsample_h_*upsample_w_*Luts_);
  for(size_t i = 0; i < Luts_; ++i)
    h_a_deltas_[i] = (((i + TK_) % Luts_) - i);
  size_t Ds0 = Luts_;
  size_t Ds1 = upsample_w_;
  size_t Ds2 = upsample_h_;
  size_t Ds3 = upsample_d_;
  for(size_t ud = 0; ud < Ds3; ++ud)
  for(size_t uh = 0; uh < Ds2; ++uh)
  for(size_t uw = 0; uw < Ds1; ++uw) {
    int32_t* deltas_ptr = &h_a_deltas_[Luts_ + uw*Ds0 + uh*Ds0*Ds1 + ud*Ds0*Ds1*Ds2];
    // cumulative increments
    for(size_t i = 0; i < Ds0; ++i) {
      int32_t EBD = 1;
      int32_t EBH = ((upsample_h_ - uh - 1) + BH_) / upsample_h_;
      int32_t EBW = ((upsample_w_ - uw - 1) + BW_) / upsample_w_;
      if(EBD == 0 || EBH == 0 || EBW == 0)
        continue;
      // unpack
      int32_t ctrs = i;
      int32_t c, t, r, s;
      std::tie(c, t, r, s) = unpack(ctrs, !b_trans_, EBD, EBH, EBW);
      // next indices
      int32_t nextctrs = ctrs + TK_;
      int32_t nextc, nextt, nextr, nexts;
      std::tie(nextc, nextt, nextr, nexts) = unpack(nextctrs, !b_trans_, EBD, EBH, EBW);
      // diffs
      int32_t cdiff = nextc - c;
      int32_t tdiff = nextt - t;
      int32_t rdiff = nextr - r;
      int32_t sdiff = nexts - s;
      if(ty_ == WGRAD){
        tdiff = tdiff * stride_d_;
        rdiff = rdiff * stride_h_;
        sdiff = sdiff * stride_w_;
      }
      // delta pointers
      deltas_ptr[i] = cdiff*ld_a_[a_inner_idx_] + tdiff*ld_a_[a_pix_idx_] + rdiff*ld_a_[a_pix_idx_ + 1] + sdiff*ld_a_[a_pix_idx_ + 2];
    }
  }
}

void conv::build_masks(){
  h_masks_.resize(Luts_ + upsample_d_*upsample_h_*upsample_w_*(2*pad_h_+1)*(2*pad_w_+1)*(2*pad_d_+1)*Luts_);

  size_t Ms0 = Luts_;
  size_t Ms1 = 2*pad_w_ + 1;
  size_t Ms2 = 2*pad_h_ + 1;
  size_t Ms3 = 2*pad_d_ + 1;
  size_t Ms4 = upsample_w_;
  size_t Ms5 = upsample_h_;
  size_t Ms6 = upsample_d_;
  for(size_t ud = 0; ud < Ms6; ++ud)
  for(size_t uh = 0; uh < Ms5; ++uh)
  for(size_t uw = 0; uw < Ms4; ++uw)
  for(size_t pd = 0; pd < Ms3; ++pd)
  for(size_t ph = 0; ph < Ms2; ++ph)
  for(size_t pw = 0; pw < Ms1; ++pw){
    int32_t* masks_ptr = &h_masks_[Luts_ + pw*Ms0 + ph*Ms0*Ms1 + pd*Ms0*Ms1*Ms2 + uw*Ms0*Ms1*Ms2*Ms3 + uh*Ms0*Ms1*Ms2*Ms3*Ms4 + ud*Ms0*Ms1*Ms2*Ms3*Ms4*Ms5];
    for(size_t i = 0; i < Ms0; ++i){
       int32_t l, t, r, s;
       int32_t mask = 0x0;
       for(size_t j = 0; j < TK_; ++j){
         int32_t EBD = 1;
         int32_t EBH = ((upsample_h_ - uh - 1) + BH_) / upsample_h_;
         int32_t EBW = ((upsample_w_ - uw - 1) + BW_) / upsample_w_;
         if(EBD == 0 || EBH == 0 || EBW == 0)
           continue;
         std::tie(l, t, r, s) = unpack(i + j, !b_trans_, EBD, EBH, EBW);
         bool in_bounds_d = (t + pd) >= pad_d_ && (t + pd) < (EBD + pad_d_);
         bool in_bounds_h = (r + ph) >= pad_h_ && (r + ph) < (EBH + pad_h_);
         bool in_bounds_w = (s + pw) >= pad_w_ && (s + pw) < (EBW + pad_w_);
         mask |= (in_bounds_d && in_bounds_h && in_bounds_w) << j;
       }
       masks_ptr[i] = mask;
    }
  }
  for(size_t i = 0; i < Luts_; ++i)
    h_masks_[i] = 0x0;
}

std::array<size_t, 3> conv::get_grid(size_t TM, size_t TN){
  return {(M_ + TM - 1)/TM, (N_ + TN - 1)/TN, 1};
}

size_t conv::num_flops() const{
  return 2.*M_*N_*K_;
}

void conv::init_impl(driver::stream *stream, triton::driver::cu_module* module, triton::runtime::launch_information info) {
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
  if(d_a_deltas_ == nullptr)
    d_a_deltas_ = init_lut(is_a_deltas_cst, "delta", h_a_deltas_);
  if(d_b_deltas_ == nullptr)
    d_b_deltas_ = init_lut(is_b_deltas_cst_, "b_delta", h_b_deltas_);
  if(d_masks_ == nullptr)
    d_masks_ = init_lut(is_mask_cst_, "masks", h_masks_);
  if(d_locks_ == nullptr){
    d_locks_ = triton::driver::buffer::create(stream->context(), max_grid_0_*max_grid_1_*4*2);
    ((triton::driver::cu_buffer*)d_locks_)->set_zero(stream, max_grid_0_*max_grid_1_*4*2);
  }
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
  kernel->setArg(13, NC_);
  // A arguments
  kernel->setArg(14, ld_a_[a_outer_idx_]);
  kernel->setArg(15, ld_a_[a_inner_idx_]);
  kernel->setArg(16, ld_a_[2]);
  kernel->setArg(17, ld_a_[3]);
  kernel->setArg(18, ld_a_[4]);
  // B arguments
  kernel->setArg(19, ld_b_[b_inner_idx_]);
  kernel->setArg(20, ld_b_[b_pix_idx_]);
  kernel->setArg(21, ld_b_[b_pix_idx_+1]);
  kernel->setArg(22, ld_b_[b_pix_idx_+2]);
  kernel->setArg(23, ld_b_[b_outer_idx_]);
  // C arguments
  kernel->setArg(24, ld_c_[c_outer_0_idx_]);
  kernel->setArg(25, ld_c_[c_outer_1_idx_]);
  kernel->setArg(26, ld_c_[c_pix_idx]);
  kernel->setArg(27, ld_c_[c_pix_idx+1]);
  kernel->setArg(28, ld_c_[c_pix_idx+2]);
  // pad
  kernel->setArg(29, pad_h_);
  kernel->setArg(30, pad_w_);
  // stride
  kernel->setArg(31, stride_h_);
  kernel->setArg(32, stride_w_);
  // dilate
  kernel->setArg(33, upsample_h_);
  kernel->setArg(34, upsample_w_);
  kernel->setArg(35, (int32_t)0);
  kernel->setArg(36, (int32_t)0);
  kernel->setArg(37, pad_h_);
  kernel->setArg(38, pad_w_);
  kernel->setArg(39, (int32_t)0);
  kernel->setArg(40, (int32_t)0);
  kernel->setArg(41, d_locks_);
  kernel->setArg(42, max_grid_0_);
  kernel->setArg(43, max_grid_1_);
  size_t idx = 44;
  if(!is_a_deltas_cst)
    kernel->setArg(idx++, d_a_deltas_);
  if(!is_b_deltas_cst_)
    kernel->setArg(idx++, d_b_deltas_);
  if(!is_mask_cst_)
    kernel->setArg(idx++, d_masks_);
}

void conv::enqueue_impl(driver::stream *stream, driver::kernel *kernel,
                        std::vector<driver::buffer*> args,
                        runtime::launch_information info) {
  driver::buffer *a = args[0], *b = args[1], *c = args[2], *bias = args[3];
  unsigned TM = info.global_range_size[0], TN = info.global_range_size[1];
  unsigned GZ = 1;
  set_arg(kernel, a, b, c, bias);
  std::array<size_t, 3> grid = {1};
  grid[0] = (M_ + TM - 1)/TM;
  grid[1] = (N_ + TN - 1)/TN;
  grid[2] = GZ;
  grid[0] /= upsample_h_*upsample_w_;
  kernel->setArg(11, CH_/upsample_h_);
  kernel->setArg(12, CW_/upsample_w_);

  // initialize to zero if necessary
  bool init_zero = false;
  for(int32_t off_uh = 0; off_uh < upsample_h_; off_uh++)
  for(int32_t off_uw = 0; off_uw < upsample_w_; off_uw++) {
    int32_t EBD = 1;
    int32_t EBH = ((upsample_h_ - off_uh - 1) + BH_) / upsample_h_;
    int32_t EBW = ((upsample_w_ - off_uw - 1) + BW_) / upsample_w_;
    if(EBD == 0 || EBH == 0 || EBW == 0)
      init_zero = true;
  }
  if(init_zero)
    ((driver::cu_buffer*)c)->set_zero(stream, c_size()*4);

  for(int32_t off_uh = 0; off_uh < upsample_h_; off_uh++)
  for(int32_t off_uw = 0; off_uw < upsample_w_; off_uw++) {
    int32_t EBD = 1;
    int32_t EBH = ((upsample_h_ - off_uh - 1) + BH_) / upsample_h_;
    int32_t EBW = ((upsample_w_ - off_uw - 1) + BW_) / upsample_w_;
    if(EBD == 0 || EBH == 0 || EBW == 0)
      continue;
    int32_t K = shapes_b_[b_inner_idx_]*EBD*EBH*EBW;
    kernel->setArg(6, K);
    kernel->setArg(9, EBH);
    kernel->setArg(10, EBW);
    kernel->setArg(29, pad_h_);
    kernel->setArg(30, pad_w_);
    kernel->setArg(35, off_uh);
    kernel->setArg(36, off_uw);
    kernel->setArg(37, (pad_h_ + (1 - upsample_h_)*off_uh)/upsample_h_);
    kernel->setArg(38, (pad_w_ + (1 - upsample_w_)*off_uw)/upsample_w_);
    kernel->setArg(39, (off_uh + pad_h_) % upsample_h_);
    kernel->setArg(40, (off_uw + pad_w_) % upsample_w_);
    stream->enqueue(kernel, grid, {info.num_threads, 1, 1});
  }
}

std::vector<unsigned> conv::default_params() {
  if(b_lut_){
    if(!b_trans_)
      return {16, 2, 32, 16, 16, 8, 8, 2, 2, 4, 2, 8, 4, 2, 1};
    else
      return {32, 2, 64, 32, 2, 64, 16, 8, 2, 2, 4, 2, 8, 1};
  }
  else if(ty_ == FPROP)
    return {16, 2, 64, 32, 2, 64, 16, 8, 2, 2, 8, 1, 8, 4, 1};
  else
    return {16, 2, 64, 16, 16, 16, 4, 2, 2, 4, 2, 8, 4, 2, 1};
}


/* CPU reference implementation */

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

/* Triton-C source code */

void conv::triton_c_src(std::ostream &os) const {
  std::string BS    = b_trans_ ? "[TN,TK]"      : "[TK, TN]";
  std::string bcb0  = b_trans_ ? "[:, newaxis]" : "[newaxis, :]";
  std::string bcb1  = b_trans_ ? "[newaxis, :]" : "[:, newaxis]";
  std::string ldb0  = b_trans_ ? "*ldb_s"       : "";
  std::string useb  = b_trans_ ? "trans(b)"     : "b";
  std::string flipr = b_trans_ ? ""             : "BH - 1 -";
  std::string flips = b_trans_ ? ""             : "BW - 1 -";
  std::string upar = ty_ == WGRAD ? "stride_h * ": "";
  std::string upas = ty_ == WGRAD ? "stride_w * ": "";
  std::string upah = ty_ == WGRAD ? "": "*stride_h";
  std::string upaw = ty_ == WGRAD ? "": "*stride_w";
  std::vector<std::string> crs = {"c", "r", "s"};
  std::vector<std::string> rsc = {"r", "s", "c"};
  std::vector<std::string> ax  = b_trans_ ? crs : rsc;
  std::vector<std::string> redax;
  if(b_trans_)
    redax = {"NC", "BH", "BW"};
  else
    redax = {"BH", "BW", "NC"};
  std::string inc_pb = b_lut_ ? "db" + bcb1 : "TK" + ldb0;
  std::string inc_pdb = b_trans_ ? "incd" : "TK";
  std::string a_delta_mem = is_a_deltas_cst ? "__constant__" : "";
  std::string b_delta_mem = is_b_deltas_cst_? "__constant__" : "";
  std::string masks_mem = is_mask_cst_? "__constant__" : "";

  os <<
      R"(
const tunable int TM = {16, 32, 64};
const tunable int TN = {16, 32, 64};
const tunable int TK = {)" << TK_ << R"(};
const tunable int GZ = {1};
)";
if(is_a_deltas_cst)
  os << "__constant__ int* delta = alloc_const int[" + std::to_string(h_a_deltas_.size()) + "];\n";
if(b_lut_ && is_b_deltas_cst_)
  os << "__constant__ int* b_delta = alloc_const int[" + std::to_string(h_b_deltas_.size()) + "];\n";
if(is_mask_cst_)
  os << "__constant__ int* masks = alloc_const int[" + std::to_string(h_masks_.size()) + "];\n";
os << R"(

 void conv(read_only restrict )" << a_ty_ << R"( *a,
           read_only restrict )" << b_ty_ << R"( *b,
           float *c,
           float *bias,
           int M, int N, int K,
           int AH, int AW,
           int BH, int BW,
           int CH, int CW,
           int NC,
           int lda_n, int lda_c, int lda_d, int lda_h, int lda_w,
           int ldb_c, int ldb_t, int ldb_r, int ldb_s, int ldb_k,
           int ldc_n, int ldc_k, int ldc_m, int ldc_p, int ldc_q,
           int pad_h, int pad_w,
           int stride_h, int stride_w,
           int upsample_h, int upsample_w,
           int off_uh, int off_uw,
           int off_uah, int off_uaw,
           int off_uch, int off_ucw,
           int *locks, int grid0, int grid1)";
if(!is_a_deltas_cst)
  os << ", int* delta";
if(b_lut_ && !is_b_deltas_cst_)
  os << ", int* b_delta";
if(!is_mask_cst_)
  os << ", int* masks";
 os << R"(){
  int rxa[TM] = get_global_range[TM](0);
  int rb0[TN] = get_global_range[TN](1);
  int rz = get_global_range[1](2);
  int rka[TK] = 0 ... TK;
  int rkb[TK] = 0 ... TK;
  float C[TM, TN] = 0;
  int ldlut = )" + std::to_string(Luts_) + R"(;
  int div = K / GZ;
  int rem = K % GZ;
  K = select(rz < rem, div, div + rem);
  int offk = rz*div;
  rka = rka + offk;
  rkb = rkb + offk;
  int rabh[TM] = rxa / CW;
  int raw[TM] = rxa % CW;
  int rab[TM] = rabh / CH;
  int rah[TM] = rabh % CH;
  rah = rah)" + upaw + R"( - off_uah;
  raw = raw)" + upah + R"( - off_uaw;
  int ra0[TM] = rab*lda_n + rah*lda_h + raw*lda_w;
  int ra)" + ax[0] + ax[1] + "[TK] = rka / " + redax[2] + R"(;
  int ra)" + ax[2] + "[TK] = rka %  " + redax[2] + R"(;
  int ra)" + ax[0] + "[TK] = ra" + ax[0] + ax[1] + " / " + redax[1] + R"(;
  int ra)" + ax[1] + "[TK] = ra" + ax[0] + ax[1] + " % " + redax[1] + R"(;
  rar = )" + flipr + R"( rar;
  ras = )" + flips + R"( ras;
  rar = )" + upar + R"( rar;
  ras = )" + upas + R"( ras;
  int ra1[TK] = rac*lda_c + rar*lda_h + ras*lda_w;
  )" << a_ty_ << R"(* pa[TM, TK] = a + ra1[newaxis, :] + ra0[:, newaxis];)";
if(b_lut_){
 os << R"(
  int rb)" + ax[0] + ax[1] + "[TK] = rkb / " + redax[2] + R"(;
  int rb)" + ax[2] + "[TK] = rkb %  " + redax[2] + R"(;
  int rb)" + ax[0] + "[TK] = rb" + ax[0] + ax[1] + " / " + redax[1] + R"(;
  int rb)" + ax[1] + "[TK] = rb" + ax[0] + ax[1] + " % " + redax[1] + R"(;
  rbr = rbr*upsample_h + off_uh;
  rbs = rbs*upsample_w + off_uw;
  int offdb[TK] = rkb % ldlut;
  int rb1[TK] = rbc*ldb_c + rbr*ldb_r + rbs*ldb_s;
  )" + b_delta_mem + R"( int* pdb[TK] = b_delta + offdb + off_uw*ldlut + off_uh*ldlut*upsample_w;
  int db[TK] = *pdb;)";
}
else{
os << R"(
  int rb1[TK] = rkb)" + ldb0 + ";";
}
os << R"(
  )" << b_ty_ << R"(* pb)" + BS + " = b + rb1" + bcb1 + " + rb0" + bcb0 + R"(*ldb_k;
  int offda[TK] = rka % ldlut;
  )" + a_delta_mem + R"( int* pincd[TK] = delta + offda;
  )" + a_delta_mem + R"( int* pda[TK]  = delta + ldlut + offda + off_uw*ldlut + off_uh*ldlut*upsample_w;
  int da[TK] = *pda;
  int incd[TK] = *pincd;
  int maskh[TM] = pad_h + min(rah, 0) + max(rah + BH - AH, 0);
  int maskw[TM] = pad_w + min(raw, 0) + max(raw + BW - AW, 0);
  int offma = offk % ldlut;
  )" + masks_mem + R"( int* pm[TM] = masks + ldlut + offma + maskw*ldlut + maskh*ldlut*(2*pad_w + 1) + off_uw*ldlut*(2*pad_w+1)*(2*pad_h+1) + off_uh*ldlut*(2*pad_w+1)*(2*pad_h+1)*upsample_w;
  )" + a_delta_mem + R"( int* pincm[TM] = delta + offma;
  int incm[TM] = *pincm;
  int maska0[TM] = *pm;
  int maska1[TK] = 1 << (0 ... TK);
  bool checka[TM, TK] = (maska0[:, newaxis] & maska1[newaxis, :]) > 0;
  bool checkb0[TN] = rb0 < N;
  bool checkb)" + BS + " = checkb0" + bcb0 + R"(;
  )" << a_ty_ << R"( a[TM, TK] = checka ? *pa : 0;
  )" << b_ty_ << R"( b)" + BS + R"( = checkb ? *pb : 0;
  int rkamin[TK] = rka - offk + TK;
  for(int k = K; k > 0; k = k - TK){
    C = dot(a, )" + useb + R"(, C);
    pa = pa + da[newaxis, :];
    pb = pb + )" + inc_pb + R"(;
    pda = pda + incd;)";
if(b_lut_){
  os << R"(
    pdb = pdb + )" + inc_pdb + R"(;
    db = *pdb;)";
}
  os << R"(
    pincd = pincd + incd;
    da = *pda;
    incd = *pincd;
    pm = pm + incm;
    pincm = pincm + incm;
    incm = *pincm;
    bool checka1[TK] = (rkamin < k);
    maska0 = *pm;
    checka = (maska0[:, newaxis] & maska1[newaxis, :]) > 0;
    checka = checka && checka1[newaxis,:];
    a = checka ? *pa : 0;
    checkb = checkb && (k > TK);
    @checkb b = *pb;
  }
  int rxc[TM] = get_global_range[TM](0);
  int rc1[TN] = get_global_range[TN](1);
  int rcn[TM] = rxc / (CH*CW);
  int rcpq[TM] = rxc % (CH*CW);
  int rcp[TM] = rcpq / CW;
  int rcq[TM] = rcpq % CW;
  rcp = rcp * upsample_h + off_uch;
  rcq = rcq * upsample_w + off_ucw;
  bool checkc1[TN] = rc1 < N;
  int rc0[TM] = rcn * ldc_n + rcp * ldc_p + rcq * ldc_q;
  float* pc[TM, TN]  = c + rc1[newaxis, :]*ldc_k + rc0[:, newaxis];
  bool checkc0[TM] = rxc < M;
  bool checkc[TM, TN]  = checkc0[:, newaxis] && checkc1[newaxis, :];
  int ridx = get_range_id(0);
  int ridy = get_range_id(1);
  int *plock = locks + ridx + ridy*grid0;
  while(__atomic_cas(plock, 0, 1) == 1);
  int *pcount = plock + grid0*grid1;
  int count = *pcount;
  int countp1 = select(count == GZ - 1, 0, count + 1);
  if(count == 0) {)";
 if(bias_ && ty_==FPROP){
   os << R"(
   float* pbias[TN] = bias + rc1;
   float bias[TN] = checkc1 ? *pbias : 0;
   C = C + bias[newaxis, :];)";
 }
   os << R"(
    @checkc *pc = C;
    *pcount = countp1;
  }
  else {
    @checkc *pc = C + *pc;
    *pcount = countp1;
  }
  *plock = 0;
})";
}

template void conv::cpu_ref<float,float>(float*, float*, float*);
template void conv::cpu_xprop<float,float>(float*, float*, float*);
template void conv::cpu_wgrad<float,float>(float*, float*, float*);

}
}
