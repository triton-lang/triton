#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"

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
       int upsample_d, int upsample_h, int upsample_w,
       int pad_d, int pad_h, int pad_w,
       type ty = FPROP)
    : NB_(B), NC_(NC), AD_(D), AH_(H), AW_(W), BD_(T), BH_(R), BW_(S), NF_(NF),
      upsample_d_(upsample_d), upsample_h_(upsample_h), upsample_w_(upsample_w),
      stride_d_(1), stride_h_(1), stride_w_(1),
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
    Fs_ = shapes_b_[1]*shapes_b_[2]*shapes_b_[3];
    TK_ = 8;
    Luts_ = (TK_ + Fs_ - 1) / Fs_ * Fs_;
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

  void build_deltas(std::vector<int>& deltas){
    if(ty_ == WGRAD)
      throw std::runtime_error("no look-up table necessary for wgrad");
    deltas.resize(Luts_ + upsample_d_*upsample_h_*upsample_w_*Luts_);
    auto unpack = [&](int32_t trs){
      int32_t tr = trs / BW_;
      int32_t s = trs - tr*BW_;
      int32_t t = tr / BH_;
      int32_t r = tr - t*BH_;
      return std::make_tuple(t, r, s);
    };
    for(size_t i = 0; i < Luts_; ++i)
      deltas[i] = (((i + TK_) % Luts_) - i);
    size_t Ds0 = Luts_;
    size_t Ds1 = upsample_w_;
    size_t Ds2 = upsample_h_;
    size_t Ds3 = upsample_d_;
    for(size_t pd = 0; pd < Ds3; ++pd)
    for(size_t ph = 0; ph < Ds2; ++ph)
    for(size_t pw = 0; pw < Ds1; ++pw){
      int32_t* deltas_ptr = &deltas[Luts_ + pw*Ds0 + ph*Ds0*Ds1 + pd*Ds0*Ds1*Ds2];
      // cumulative increments
      for(size_t i = 0; i < Ds0; ++i){
        int32_t ctrs = i;
        int32_t c = ctrs / Fs_;
        int32_t t, r, s;
        std::tie(t, r, s) = unpack(ctrs % Fs_);
        // next indices
        int32_t nextctrs = ctrs + TK_;
        int32_t nextc = nextctrs / Fs_;
        int32_t nextt, nextr, nexts;
        std::tie(nextt, nextr, nexts) = unpack(nextctrs % Fs_);
        // diffs
        int32_t cdiff = nextc - c;
        int32_t tdiff = (nextt + pd)/upsample_d_ - (t + pd)/upsample_d_;
        int32_t rdiff = (nextr + ph)/upsample_h_ - (r + ph)/upsample_h_;
        int32_t sdiff = (nexts + pw)/upsample_w_ - (s + pw)/upsample_w_;
        // delta pointers
        deltas_ptr[i] = cdiff*ld_a_[1] + tdiff*ld_a_[2] + rdiff*ld_a_[3] + sdiff*ld_a_[4];
      }
    }
  }

  void build_masks(std::vector<int>& masks){
    if(ty_ == WGRAD)
      throw std::runtime_error("no look-up table necessary for wgrad");
    masks.resize(Luts_ + (2*pad_h_+1)*(2*pad_w_+1)*(2*pad_d_+1)*Luts_);
    auto unpack = [&](int32_t trs){
      int32_t tr = trs / BW_;
      int32_t s = trs - tr*BW_;
      int32_t t = tr / BH_;
      int32_t r = tr - t*BH_;
      return std::make_tuple(t, r, s);
    };
    size_t Ms0 = Luts_;
    size_t Ms1 = 2*pad_w_ + 1;
    size_t Ms2 = 2*pad_h_ + 1;
    size_t Ms3 = 2*pad_d_ + 1;
    for(size_t pd = 0; pd < Ms3; ++pd)
    for(size_t ph = 0; ph < Ms2; ++ph)
    for(size_t pw = 0; pw < Ms1; ++pw){
      int32_t* masks_ptr = &masks[Luts_ + pw*Ms0 + ph*Ms0*Ms1 + pd*Ms0*Ms1*Ms2];
      for(size_t i = 0; i < Ms0; ++i){
         int32_t t, r, s;
         int32_t mask = 0x0;
         for(size_t j = 0; j < TK_; ++j){
           std::tie(t, r, s) = unpack((i + j) % Fs_);
           bool in_bounds_d = (t + pd) >= pad_d_ && (t + pd) < (BD_ + pad_d_);
           bool in_bounds_h = (r + ph) >= pad_h_ && (r + ph) < (BH_ + pad_h_);
           bool in_bounds_w = (s + pw) >= pad_w_ && (s + pw) < (BW_ + pad_w_);
           mask |= (in_bounds_d && in_bounds_h && in_bounds_w) << j;
         }
         masks_ptr[i] = mask;
      }
    }
    for(size_t i = 0; i < Luts_; ++i)
      masks[i] = 0x0;
  }

  std::array<size_t, 3> get_grid(size_t TM, size_t TN){
    return {(M_ + TM - 1)/TM, (N_ + TN - 1)/TN, 1};
  }

  size_t get_nflops(){
    return 2.*M_*N_*K_;
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
    kernel->setArg(12, ld_a_[0]);
    kernel->setArg(13, ld_a_[1]);
    kernel->setArg(14, ld_a_[2]);
    kernel->setArg(15, ld_a_[3]);
    kernel->setArg(16, ld_a_[4]);
    kernel->setArg(17, ld_b_[0]);
    kernel->setArg(18, ld_b_[1]);
    kernel->setArg(19, ld_b_[2]);
    kernel->setArg(20, ld_b_[3]);
    kernel->setArg(21, ld_b_[4]);
    kernel->setArg(22, ld_c_[0]);
    kernel->setArg(23, ld_c_[1]);
    kernel->setArg(24, ld_c_[2]);
    kernel->setArg(25, ld_c_[3]);
    kernel->setArg(26, ld_c_[4]);
    kernel->setArg(27, pad_h_);
    kernel->setArg(28, pad_w_);
  }

  std::vector<unsigned> default_params() {
    if(ty_ == FPROP || ty_ == BPROP)
      return {16, 2, 64, 32, 2, 64, 16, 8, 2, 2, 8, 1, 8, 4};
    else
      return {8, 2, 16, 8, 2, 16, 8, 2, 8, 8};
  }


  std::string xprop() {
    std::string res =
        R"(
        const tunable int32 TM = {16, 32, 64};
        const tunable int32 TN = {16, 32, 64};
        const tunable int32 TK = {8};

        __constant__ int32* delta = alloc_const int32[18];
        __constant__ int32* masks = alloc_const int32[1024];

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
                  int32 pad_h, int32 pad_w){
            int32 rxa[TM] = get_global_range[TM](0);
            int32 rb0[TN] = get_global_range[TN](1);
            int32 rka[TK] = 0 ... TK;
            int32 rb1[TK] = 0 ... TK;
            fp32 C[TM, TN] = 0;
            int32 rabh[TM] = rxa / CW;
            int32 raw[TM] = rxa % CW - pad_w;
            int32 rab[TM] = rabh / CH;
            int32 rah[TM] = rabh % CH - pad_h;
            int32 ra0[TM] = rab*lda_n + rah*lda_h + raw*lda_w;
            int32 racr[TK] = rka / BW;
            int32 ras[TK] = rka % BW;
            int32 rac[TK] = racr / BH;
            int32 rar[TK] = racr % BH;
            int32 ra1[TK] = rac*lda_c + rar*lda_h + ras*lda_w;
            fp32* pa[TM, TK] = a + ra1[newaxis, :] + ra0[:, newaxis];
            fp32* pb[TN, TK] = b + rb1[newaxis, :]*ldb_s + rb0[:, newaxis];
            __constant__ int32* pincd[TK] = delta + rka;
            __constant__ int32* pd[TK] = delta + BH*BW + rka;
            int32 d[TK] = *pd;
            int32 incd[TK] = *pincd;
            int32 maskh[TM] = pad_h + min(rah, 0) + max(rah + BH - AH, 0);
            int32 maskw[TM] = pad_w + min(raw, 0) + max(raw + BW - AW, 0);
            __constant__ int32* pm[TM] = masks + BH*BW + maskw*BH*BW + maskh*BH*BW*(2*pad_w + 1);
            __constant__ int32* pincm[TM] = delta;
            int32 incm[TM] = *pincm;
            int32 checka0[TM] = *pm;
            int32 checka1[TK] = 1 << rka;
            int1 checka[TM, TK] = (checka0[:, newaxis] & checka1[newaxis, :]) > 0;
            fp32 a[TM, TK] = checka ? *pa : 0;
            fp32 b[TN, TK] = *pb;
            for(int32 k = K; k > 0; k = k - TK){
              C = dot(a, trans(b), C);
              pb = pb + TK*ldb_s;
              pa = pa + d[newaxis, :];
              b = *pb;
              pd = pd + incd;
              pincd = pincd + incd;
              d = *pd;
              incd = *pincd;
              pm = pm + incm;
              pincm = pincm + incm;
              incm = *pincm;
              checka0 = *pm;
              checka = (checka0[:, newaxis] & checka1[newaxis, :]) > 0;
              a = checka ? *pa : 0;
            }
            int32 rxc[TM] = get_global_range[TM](0);
            int32 rc1[TN] = get_global_range[TN](1);
            int32 rcn[TM] = rxc / (CH*CW);
            int32 rcpq[TM] = rxc % (CH*CW);
            int32 rc0[TM] = rcn * ldc_n + rcpq;
            fp32* pc[TM, TN]  = c + rc1[newaxis, :]*ldc_k + rc0[:, newaxis];
            int1 checkc0[TM] = rxc < M;
            int1 checkc1[TN] = rc1 < N;
            int1 checkc[TM, TN]  = checkc0[:, newaxis] && checkc1[newaxis, :];
            @checkc *pc = C;
        })";
    return res;
  }

  // C = A * B
  // where A is N,C,AH,AW
  //       B is N,K,BH,BW
  //       C is C,CH,CW,K
  std::string wgrad() {
    std::string res =
        R"(
        const tunable int32 TM = {16, 32, 64};
        const tunable int32 TN = {16, 32, 64};
        const tunable int32 TK = {8};

        void conv(read_only restrict fp32 *a,
                  read_only restrict fp32 *b,
                  fp32 *c,
                  int32 M, int32 N, int32 K,
                  int32 AH, int32 AW,
                  int32 CH, int32 CW,
                  int32 BH, int32 BW,
                  int32 lda_n, int32 lda_c, int32 lda_d, int32 lda_h, int32 lda_w,
                  int32 ldb_n, int32 ldb_k, int32 ldb_m, int32 ldb_p, int32 ldb_q,
                  int32 ldc_c, int32 ldc_t, int32 ldc_r, int32 ldc_s, int32 ldc_k,
                  int32 pad_h, int32 pad_w){
            int32 rxa[TM] = get_global_range[TM](0);
            int32 ryb[TN] = get_global_range[TN](1);
            int32 rk[TK] = 0 ... TK;
            fp32 C[TM, TN] = 0;
            int32 racr[TM] = rxa / CW;
            int32 raw_base[TM] = rxa % CW - pad_w;
            int32 rac[TM] = racr / CH;
            int32 rah_base[TM] = racr % CH - pad_h;
            fp32* pa_base[TM, TK] = a + rac[:, newaxis]*lda_c;
            fp32* pb_base[TN, TK] = b + ryb[:, newaxis]*ldb_k;
            for(int32 k = K; k > 0; k = k - TK){
              int32 rknp[TK] = rk / BW;
              int32 rkq[TK] = rk % BW;
              int32 rkn[TK] = rknp / BH;
              int32 rkp[TK] = rknp % BH;
              int32 rah[TM, TK] = rah_base[:, newaxis] + rkp[newaxis, :];
              int32 raw[TM, TK] = raw_base[:, newaxis] + rkq[newaxis, :];
              int1 checka[TM, TK] = (rah >= 0) && (rah < AH) && (raw >= 0) && (raw < AW);
              fp32* pa[TM, TK] = pa_base + rah*lda_h + raw*lda_w + rkn*lda_n;
              fp32* pb[TN, TK] = pb_base + rkp*ldb_p + rkq*ldb_q + rkn*ldb_n;
              fp32 A[TM, TK] = checka ? *pa : 0;
              fp32 B[TN, TK] = *pb;
              C = dot(A, trans(B), C);
              rk = rk + TK;
            }
            int32 rxc[TM] = get_global_range[TM](0);
            int32 ryc[TN] = get_global_range[TN](1);
            int32 rccr[TM] = rxc / CW;
            int32 rcs[TM] = rxa % CW;
            int32 rcc[TM] = racr / CH;
            int32 rcr[TM] = racr % CH;
            int32 rc0[TM] = rcc*ldc_c + rcr*ldc_r + rcs*ldc_s;
            fp32* pc[TM, TN]  = c + rc0[:, newaxis] + ryc[newaxis, :]*ldc_k;
            int1 checkc0[TM] = rxc < M;
            int1 checkc1[TN] = ryc < N;
            int1 checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];
            @checkc *pc = C;
        })";
    return res;
  }

  std::string src() {
    if(ty_ == FPROP || ty_ == BPROP)
      return xprop();
    else
      return wgrad();
  }

  template<class IN_DTYPE, class OUT_DTYPE>
  void cpu_xprop(OUT_DTYPE* C,  IN_DTYPE* A, IN_DTYPE* B)
  {
    IN_DTYPE acc;
    for(int32_t n = 0; n < shapes_c_[0]; ++n)
    for(int32_t k = 0; k < shapes_c_[1] ; ++k)
    for(int32_t cd = 0 ; cd < shapes_c_[2]; ++cd)
    for(int32_t ch = 0 ; ch < shapes_c_[3]; ++ch)
    for(int32_t cw = 0; cw < shapes_c_[4]; ++cw)
    {
      acc = 0;
      int32_t d = cd*stride_d_ - pad_d_;
      int32_t h = ch*stride_h_ - pad_h_;
      int32_t w = cw*stride_w_ - pad_w_;
      for(int32_t c = 0; c < shapes_b_[0]; ++c)
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
          a = A[n*ld_a_[0] + c*ld_a_[1] + ad*ld_a_[2] + ah*ld_a_[3] + aw*ld_a_[4]];
        IN_DTYPE b = B[c*ld_b_[0] + bd*ld_b_[1] + bh*ld_b_[2] + bw*ld_b_[3] + k*ld_b_[4]];
        acc = std::fma(a, b, acc);
      }
      C[n*ld_c_[0] + k*ld_c_[1] + cd*ld_c_[2] + ch*ld_c_[3] + cw*ld_c_[4]] = acc;
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
  // type
  type ty_;
  bool is_bprop_;
};

}
}
