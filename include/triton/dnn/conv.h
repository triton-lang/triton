#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"
#include "triton/jit.h"

namespace triton{
namespace dnn{

class conv {
public:
  enum type {
    FPROP,
    BPROP,
    WGRAD
  };


  conv(int B, int NC,
       int D, int H, int W,
       int T, int R, int S, int NF,
       int stride_d, int stride_h, int stride_w,
       int pad_d, int pad_h, int pad_w,
       type ty = FPROP)
    : NB_(B), NC_(NC), AD_(D), AH_(H), AW_(W), BD_(T), BH_(R), BW_(S), NF_(NF),
      stride_d_(stride_d), stride_h_(stride_h), stride_w_(stride_w),
      upsample_d_(1), upsample_h_(1), upsample_w_(1),
      pad_d_(pad_d), pad_h_(pad_h), pad_w_(pad_w),
      ty_(ty)
  {
    CD_ = (AD_*upsample_d_ - BD_ + 1 + 2*pad_d_ + stride_d_ - 1)/stride_d_;
    CH_ = (AH_*upsample_h_ - BH_ + 1 + 2*pad_h_ + stride_h_ - 1)/stride_h_;
    CW_ = (AW_*upsample_w_ - BW_ + 1 + 2*pad_w_ + stride_w_ - 1)/stride_w_;
    // shapes
    shapes_a_ = {NB_, NC_, AD_, AH_, AW_};
    shapes_b_ = {NC_, BD_, BH_, BW_, NF_};
    shapes_c_ = {NB_, NF_, CD_, CH_, CW_};
    // swap a and c for bprop
    if(ty_ == BPROP){
      pad_d_ = (CD_ - AD_ + BD_ - 1) / 2;
      pad_h_ = (CH_ - AH_ + BH_ - 1) / 2;
      pad_w_ = (CW_ - AW_ + BW_ - 1) / 2;
      shapes_a_.swap(shapes_c_);
    }
    // swap b and c for wgrad
    if(ty_ == WGRAD){
      shapes_b_.swap(shapes_c_);
      std::swap(BD_, CD_);
      std::swap(BH_, CH_);
      std::swap(BW_, CW_);
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
    if(ty_ == WGRAD){
      M_ = shapes_c_[0]*shapes_c_[1]*shapes_c_[2]*shapes_c_[3];
      N_ = shapes_c_[4];
      K_ = shapes_b_[0]*shapes_b_[2]*shapes_b_[3]*shapes_b_[4];
    }
    else{
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

  size_t a_size() {
    return std::accumulate(shapes_a_.begin(), shapes_a_.end(),
                           1, std::multiplies<int>());
  }

  size_t b_size() {
    return std::accumulate(shapes_b_.begin(), shapes_b_.end(),
                           1, std::multiplies<int>());
  }

  size_t c_size() {
    return std::accumulate(shapes_c_.begin(), shapes_c_.end(),
                           1, std::multiplies<int>());
  }

  std::vector<int32_t> c_shapes() {
    return shapes_c_;
  }

  void build_deltas(){
    h_a_deltas_.resize(Luts_ + upsample_d_*upsample_h_*upsample_w_*Luts_);
    if(b_lut_)
      h_b_deltas_.resize(Luts_);

    auto unpack = [&](int32_t ltrs){
      int32_t l = (ty_ == BPROP) ? ltrs % NF_ : ltrs / (BD_*BH_*BW_);
      int32_t trs = (ty_ == BPROP) ? ltrs / NF_ : ltrs % (BD_*BH_*BW_);
      int32_t tr = trs / BW_;
      int32_t s = trs % BW_;
      int32_t t = tr / BH_;
      int32_t r = tr % BH_;
      if(ty_ == BPROP){
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
    for(size_t pw = 0; pw < Ds1; ++pw){
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
        if(ty_ == WGRAD)
          deltas_ptr[i] = cdiff*ld_a_[0] + tdiff*ld_a_[2] + rdiff*ld_a_[3] + sdiff*ld_a_[4];
        else
          deltas_ptr[i] = cdiff*ld_a_[1] + tdiff*ld_a_[2] + rdiff*ld_a_[3] + sdiff*ld_a_[4];
      }
    }

    if(ty_ == WGRAD){
      for(size_t i = 0; i < Ds0; ++i) {
        int32_t c, t, r, s;
        int32_t nextc, nextt, nextr, nexts;
        std::tie(c, t, r, s) = unpack(i);
        std::tie(nextc, nextt, nextr, nexts) = unpack(i + TK_);
        int32_t cdiff = nextc - c, tdiff = nextt - t, rdiff = nextr - r, sdiff = nexts - s;
        h_b_deltas_[i] = cdiff*ld_b_[0] + tdiff*ld_b_[2] + rdiff*ld_b_[3] + sdiff*ld_b_[4];
      }
    }
  }

  void build_masks(){
    h_masks_.resize(Luts_ + (2*pad_h_+1)*(2*pad_w_+1)*(2*pad_d_+1)*Luts_);

    auto unpack = [&](int32_t ltrs){
      int32_t l = (ty_ == BPROP) ? ltrs % NF_ : ltrs / (BD_*BH_*BW_);
      int32_t trs = (ty_ == BPROP) ? ltrs / NF_ : ltrs % (BD_*BH_*BW_);
      int32_t tr = trs / BW_;
      int32_t s = trs % BW_;
      int32_t t = tr / BH_;
      int32_t r = tr % BH_;
      if(ty_ == BPROP){
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

  std::array<size_t, 3> get_grid(size_t TM, size_t TN){
    return {(M_ + TM - 1)/TM, (N_ + TN - 1)/TN, 1};
  }

  size_t get_nflops(){
    return 2.*M_*N_*K_;
  }

  void init(driver::stream *stream, triton::jit &jit) {
    auto init_lut = [&](bool is_cst, const char *name, std::vector<int32_t> host) -> triton::driver::buffer*{
      if(host.empty())
        return nullptr;
      size_t nbytes = host.size()*4;
      // get buffer
      triton::driver::buffer* buffer;
      if(is_cst)
        buffer = jit.get_buffer(name);
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

  void set_arg(driver::kernel *kernel,
                      driver::buffer *a, driver::buffer *b, driver::buffer *c)
  {
    kernel->setArg(0, a);
    kernel->setArg(1, b);
    kernel->setArg(2, c);
    kernel->setArg(3, M_);
    kernel->setArg(4, N_);
    kernel->setArg(5, K_);
    kernel->setArg(6, AH_);
    kernel->setArg(7, AW_);
    kernel->setArg(8, BH_);
    kernel->setArg(9, BW_);
    kernel->setArg(10, CH_);
    kernel->setArg(11, CW_);
    // A arguments
    if(ty_ == WGRAD){
      kernel->setArg(12, ld_a_[1]);
      kernel->setArg(13, ld_a_[0]);
    }
    else{
      kernel->setArg(12, ld_a_[0]);
      kernel->setArg(13, ld_a_[1]);
    }
    kernel->setArg(14, ld_a_[2]);
    kernel->setArg(15, ld_a_[3]);
    kernel->setArg(16, ld_a_[4]);
    // B arguments
    if(ty_ == WGRAD){
      kernel->setArg(17, ld_b_[0]);
      kernel->setArg(18, ld_b_[2]);
      kernel->setArg(19, ld_b_[3]);
      kernel->setArg(20, ld_b_[4]);
      kernel->setArg(21, ld_b_[1]);
    }
    else{
      kernel->setArg(17, ld_b_[0]);
      kernel->setArg(18, ld_b_[1]);
      kernel->setArg(19, ld_b_[2]);
      kernel->setArg(20, ld_b_[3]);
      kernel->setArg(21, ld_b_[4]);
    }
    // C arguments
    if(ty_ == WGRAD){
      kernel->setArg(22, ld_c_[0]);
      kernel->setArg(23, ld_c_[4]);
      kernel->setArg(24, ld_c_[1]);
      kernel->setArg(25, ld_c_[2]);
      kernel->setArg(26, ld_c_[3]);
    }
    else{
      kernel->setArg(22, ld_c_[0]);
      kernel->setArg(23, ld_c_[1]);
      kernel->setArg(24, ld_c_[2]);
      kernel->setArg(25, ld_c_[3]);
      kernel->setArg(26, ld_c_[4]);
    }
    kernel->setArg(27, pad_h_);
    kernel->setArg(28, pad_w_);
    size_t idx = 29;
    if(!is_a_deltas_cst)
      kernel->setArg(idx++, d_a_deltas_);
    if(!is_b_deltas_cst_)
      kernel->setArg(idx++, d_b_deltas_);
    if(!is_mask_cst_)
      kernel->setArg(idx++, d_masks_);
  }

  std::vector<unsigned> default_params() {
    if(ty_==FPROP)
      return {16, 2, 64, 32, 2, 64, 16, 8, 2, 2, 8, 1, 8, 4};
    else if(ty_ == BPROP)
      return {32, 2, 64, 32, 64, 32, 4, 2, 2, 4, 2, 8, 4, 2};
    else if(ty_ == WGRAD)
      return {32, 2, 64, 32, 2, 64, 16, 8, 2, 2, 4, 2, 8};
  }


  std::string src() {
    bool is_wgrad = ty_ == WGRAD;
    std::string BS    = b_trans_ ? "[TN,TK]"      : "[TK, TN]";
    std::string bcb0  = b_trans_ ? "[:, newaxis]" : "[newaxis, :]";
    std::string bcb1  = b_trans_ ? "[newaxis, :]" : "[:, newaxis]";
    std::string ldb0  = b_trans_ ? "*ldb_s"       : "";
    std::string ldb1  = b_trans_ ? "*ldb_k"       : "*ldb_c";
    std::string useb  = b_trans_ ? "trans(b)"     : "b";
    std::string flipr = b_trans_ ? ""             : "BH - 1 -";
    std::string flips = b_trans_ ? ""             : "BW - 1 -";
    std::string ax    = b_trans_ ? "crs"          : "rsc";
    std::vector<std::string> redax;
    if(b_trans_)
      redax = {"C", "BH", "BW"};
    else
      redax = {"BH", "BW", "N"};
    std::string inc_pb = is_wgrad ? "db[newaxis, :]" : "TK" + ldb0;
    std::string a_delta_mem = is_a_deltas_cst ? "__constant__" : "";
    std::string b_delta_mem = is_b_deltas_cst_? "__constant__" : "";
    std::string masks_mem = is_mask_cst_? "__constant__" : "";

    std::string res =
        R"(
        const tunable int32 TM = {16, 32, 64};
        const tunable int32 TN = {16, 32, 64};
        const tunable int32 TK = {8};
        )";
    if(is_a_deltas_cst)
        res += "__constant__ int32* delta = alloc_const int32[" + std::to_string(h_a_deltas_.size()) + "];\n";
    if(is_wgrad && is_b_deltas_cst_)
        res += "__constant__ int32* b_delta = alloc_const int32[" + std::to_string(h_b_deltas_.size()) + "];\n";
    if(is_mask_cst_)
        res += "__constant__ int32* masks = alloc_const int32[" + std::to_string(h_masks_.size()) + "];\n";
    res += R"(

        void conv(read_only restrict fp32 *a,
                  read_only restrict fp32 *b,
                  fp32 *c,
                  int32 M, int32 N, int32 K,
                  int32 AH, int32 AW,
                  int32 BH, int32 BW,
                  int32 CH, int32 CW,
                  int32 lda_n, int32 lda_c, int32 lda_d, int32 lda_h, int32 lda_w,
                  int32 ldb_c, int32 ldb_t, int32 ldb_r, int32 ldb_s, int32 ldb_k,
                  int32 ldc_n, int32 ldc_k, int32 ldc_m, int32 ldc_p, int32 ldc_q,
                  int32 pad_h, int32 pad_w)";
    if(!is_a_deltas_cst)
      res += ", int32* delta\n";
    if(is_wgrad && !is_b_deltas_cst_)
      res += ", int32* b_delta\n";
    if(!is_mask_cst_)
      res += ", int32* masks\n";
     res += R"(){
            int32 rxa[TM] = get_global_range[TM](0);
            int32 rb0[TN] = get_global_range[TN](1);
            int32 rka[TK] = 0 ... TK;
            int32 rkb[TK] = 0 ... TK;
            fp32 C[TM, TN] = 0;
            int32 ldlut = )" + std::to_string(Fs_) + R"(;
            int32 rabh[TM] = rxa / CW;
            int32 raw[TM] = rxa % CW - pad_w;
            int32 rab[TM] = rabh / CH;
            int32 rah[TM] = rabh % CH - pad_h;
            int32 ra0[TM] = rab*lda_n + rah*lda_h + raw*lda_w;
            int32 ra)" + ax[0] + ax[1] + "[TK] = rka / " + redax[2] + R"(;
            int32 ra)" + ax[2] + "[TK] = rka %  " + redax[2] + R"(;
            int32 ra)" + ax[0] + "[TK] = ra" + ax[0] + ax[1] + " / " + redax[1] + R"(;
            int32 ra)" + ax[1] + "[TK] = ra" + ax[0] + ax[1] + " % " + redax[1] + R"(;
            rar = )" + flipr + R"( rar;
            ras = )" + flips + R"( ras;
            int32 ra1[TK] = rac*lda_c + rar*lda_h + ras*lda_w;
            fp32* pa[TM, TK] = a + ra1[newaxis, :] + ra0[:, newaxis];)";
            if(ty_ == WGRAD){
             res += R"(
                int32 rbcr[TK] = rkb / BW;
                int32 rbs[TK] = rkb %  BW;
                int32 rbc[TK] = rbcr / BH;
                int32 rbr[TK] = rbcr % BH;
                int32 rb1[TK] = rbc*ldb_c + rbr*ldb_r + ras*ldb_s;
                )" + b_delta_mem + R"( int32* pdb[TK] = b_delta + rkb;
                int32 db[TK] = *pdb;)";
            }
            else{
             res += R"(
                int32 rb1[TK] = rkb;)";
            }
            res += R"(
            fp32* pb)" + BS + " = b + rb1" + bcb1 + ldb0 + " + rb0" + bcb0 + ldb1 + R"(;
            )" + a_delta_mem + R"( int32* pincd[TK] = delta + rka;
            )" + a_delta_mem + R"( int32* pd[TK]  = delta + ldlut + rka;
            int32 d[TK] = *pd;
            int32 incd[TK] = *pincd;
            int32 maskh[TM] = pad_h + min(rah, 0) + max(rah + BH - AH, 0);
            int32 maskw[TM] = pad_w + min(raw, 0) + max(raw + BW - AW, 0);
            )" + masks_mem + R"( int32* pm[TM] = masks + ldlut + maskw*ldlut + maskh*ldlut*(2*pad_w + 1);
            )" + a_delta_mem + R"( int32* pincm[TM] = delta;
            int32 incm[TM] = *pincm;
            int32 checka0[TM] = *pm;
            int32 checka1[TK] = 1 << rka;
            int1 checka[TM, TK] = (checka0[:, newaxis] & checka1[newaxis, :]) > 0;
            fp32 a[TM, TK] = checka ? *pa : 0;
            fp32 b)" + BS + R"( = *pb;
            for(int32 k = K; k > 0; k = k - TK){
              C = dot(a, )" + useb + R"(, C);
              pa = pa + d[newaxis, :];
              pb = pb + )" + inc_pb + R"(;
              b = *pb;
              pd = pd + incd;)";
            if(ty_ == WGRAD){
              res += R"(
                pdb = pdb + incd;
                db = *pdb;)";
            }
            res += R"(
              pincd = pincd + incd;
              d = *pd;
              incd = *pincd;
              pm = pm + incm;
              pincm = pincm + incm;
              incm = *pincm;
              checka0 = *pm;
              checka = (checka0[:, newaxis] & checka1[newaxis, :]) > 0;
              checka = checka && (k > TK);
              a = checka ? *pa : 0;
            }
            int32 rxc[TM] = get_global_range[TM](0);
            int32 rc1[TN] = get_global_range[TN](1);
            int32 rcn[TM] = rxc / (CH*CW);
            int32 rcpq[TM] = rxc % (CH*CW);
            int32 rc0[TM] = rcn * ldc_n + rcpq * ldc_q;
            fp32* pc[TM, TN]  = c + rc1[newaxis, :]*ldc_k + rc0[:, newaxis];
            int1 checkc0[TM] = rxc < M;
            int1 checkc1[TN] = rc1 < N;
            int1 checkc[TM, TN]  = checkc0[:, newaxis] && checkc1[newaxis, :];
            @checkc *pc = C;
        })";

    return res;
  }

  template<class IN_DTYPE, class OUT_DTYPE>
  void cpu_xprop(OUT_DTYPE* C,  IN_DTYPE* A, IN_DTYPE* B)
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
        if(ty_==FPROP)
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
  void cpu_wgrad(OUT_DTYPE* C,  IN_DTYPE* A, IN_DTYPE* B)
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
  void cpu_ref(OUT_DTYPE* C,  IN_DTYPE* A, IN_DTYPE* B)
  {
    if(ty_ == FPROP || ty_ == BPROP)
      cpu_xprop(C, A, B);
    else
      cpu_wgrad(C, A, B);
  }

private:
  // image size
  int32_t NB_;
  int32_t NC_;
  int32_t AD_;
  int32_t AH_;
  int32_t AW_;
  // filter size
  int32_t BD_;
  int32_t BH_;
  int32_t BW_;
  int32_t NF_;
  // activation size
  int32_t CD_;
  int32_t CH_;
  int32_t CW_;
  // upsampling
  int32_t upsample_d_;
  int32_t upsample_h_;
  int32_t upsample_w_;
  // padding
  int32_t pad_d_;
  int32_t pad_h_;
  int32_t pad_w_;
  // striding
  int32_t stride_d_;
  int32_t stride_h_;
  int32_t stride_w_;
  // equivalent matmul
  int32_t M_;
  int32_t N_;
  int32_t K_;
  // helpers
  int32_t Fs_;
  int32_t TK_;
  int32_t Luts_;
  // memory strides for A
  std::vector<int32_t> shapes_a_;
  std::vector<int32_t> ld_a_;
  // memory strides for B
  std::vector<int32_t> shapes_b_;
  std::vector<int32_t> ld_b_;
  // memory stride for C
  std::vector<int32_t> shapes_c_;
  std::vector<int32_t> ld_c_;
  // constant memory
  std::vector<int32_t> h_a_deltas_;
  std::vector<int32_t> h_b_deltas_;
  std::vector<int32_t> h_masks_;
  driver::buffer* d_a_deltas_;
  driver::buffer* d_b_deltas_;
  driver::buffer* d_masks_;
  bool is_a_deltas_cst;
  bool is_b_deltas_cst_;
  bool is_mask_cst_;
  // type
  type ty_;
  bool b_trans_;
  bool b_lut_;
};

}
}
