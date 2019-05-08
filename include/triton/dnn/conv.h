#include <string>
#include <vector>
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


  conv(int B, int NC, int H, int W, int R, int S, int NF,
       int upsample_h, int upsample_w,
       int pad_h, int pad_w,
       type ty = FPROP)
    : B_(B), NC_(NC), D_(1), H_(H), W_(W), T_(1), R_(R), S_(S), NF_(NF),
      upsample_d_(1), upsample_h_(upsample_h), upsample_w_(upsample_w),
      stride_d_(1), stride_h_(1), stride_w_(1),
      pad_d_(0), pad_h_(pad_h), pad_w_(pad_w),
      ty_(ty)
  {
    RD_ = (D_*upsample_d_ - T_ + 1 + 2*pad_d_ + stride_d_ - 1)/stride_d_;
    RH_ = (H_*upsample_h_ - R_ + 1 + 2*pad_h_ + stride_h_ - 1)/stride_h_;
    RW_ = (W_*upsample_w_ - S_ + 1 + 2*pad_w_ + stride_w_ - 1)/stride_w_;
    M_ = B*RD_*RH_*RW_;
    N_ = NF;
    K_ = NC*T_*R_*S_;
    // memory strides for data
    stride_a_w_ = 1;
    stride_a_h_ = W_*stride_a_w_;
    stride_a_d_ = H_*stride_a_h_;
    stride_a_c_ = D_*stride_a_d_;
    stride_a_n_ = NC_*stride_a_c_;
    // memory stride for activations
    stride_c_q_ = 1;
    stride_c_p_ = RW_*stride_c_q_;
    stride_c_m_ = RH_*stride_c_p_;
    stride_c_k_ = RD_*stride_c_m_;
    stride_c_n_ = NF_*stride_c_k_;
    // swap a and c for bprop
    if(ty_ == BPROP){
      std::swap(stride_a_n_, stride_c_n_);
      std::swap(stride_a_c_, stride_c_k_);
      std::swap(stride_a_h_, stride_c_p_);
      std::swap(stride_a_w_, stride_c_q_);
      std::swap(D_, RD_);
      std::swap(H_, RH_);
      std::swap(W_, RW_);
      pad_d_ = (RD_ - D_ + T_ - 1) / 2;
      pad_h_ = (RH_ - H_ + R_ - 1) / 2;
      pad_w_ = (RW_ - W_ + S_ - 1) / 2;
    }
    // look-up table info
    Fs_ = T_*R_*S_;
    TK_ = 8;
    Luts_ = (TK_ + Fs_ - 1) / Fs_ * Fs_;
  }

  void build_deltas(std::vector<int>& deltas){
    deltas.resize(Luts_ + upsample_d_*upsample_h_*upsample_w_*Luts_);
    auto unpack = [&](int32_t trs){
      int32_t tr = trs / S_;
      int32_t s = trs - tr*S_;
      int32_t t = tr / R_;
      int32_t r = tr - t*R_;
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
        deltas_ptr[i] = cdiff*stride_a_c_ + sdiff*stride_a_w_ + rdiff*stride_a_h_ + tdiff*stride_a_d_;
      }
    }
  }

  void build_masks(std::vector<int>& masks){
    masks.resize(Luts_ + (2*pad_h_+1)*(2*pad_w_+1)*(2*pad_d_+1)*Luts_);
    auto unpack = [&](int32_t trs){
      int32_t tr = trs / S_;
      int32_t s = trs - tr*S_;
      int32_t t = tr / R_;
      int32_t r = tr - t*R_;
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
           bool in_bounds_d = (t + pd) >= pad_d_ && (t + pd) < (T_ + pad_d_);
           bool in_bounds_h = (r + ph) >= pad_h_ && (r + ph) < (R_ + pad_h_);
           bool in_bounds_w = (s + pw) >= pad_w_ && (s + pw) < (S_ + pad_w_);
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

    if(ty_ == BPROP)
      std::swap(a, c);
    kernel->setArg(0, a);
    kernel->setArg(1, b);
    kernel->setArg(2, c);
    kernel->setArg(3, M_);
    kernel->setArg(4, N_);
    kernel->setArg(5, K_);
    kernel->setArg(6, B_);
    kernel->setArg(7, H_);
    kernel->setArg(8, W_);
    kernel->setArg(9, NF_);
    kernel->setArg(10, RH_);
    kernel->setArg(11, RW_);
    kernel->setArg(12, NC_);
    kernel->setArg(13, R_);
    kernel->setArg(14, S_);
    kernel->setArg(15, stride_a_n_);
    kernel->setArg(16, stride_a_c_);
    kernel->setArg(17, stride_a_h_);
    kernel->setArg(18, stride_a_w_);
    kernel->setArg(19, stride_c_n_);
    kernel->setArg(20, stride_c_k_);
    kernel->setArg(21, stride_c_p_);
    kernel->setArg(22, stride_c_q_);
    kernel->setArg(23, pad_h_);
    kernel->setArg(24, pad_w_);
  }

  std::vector<unsigned> default_params() {
    if(ty_ == FPROP)
      return {16, 2, 64, 32, 2, 64, 16, 8, 2, 2, 8, 1, 8, 4};
    else
      return {16, 2, 64, 16, 32, 16, 4, 2, 2, 4, 2, 8, 4, 2};
  }


  std::string src() {
    std::string bs0 = "TN", bs1 = "TK";
    std::string ldb0 = "*NF", ldb1 = "";
    std::string bcb0 = "[:, newaxis]", bcb1 = "[newaxis, :]";
    std::string b = "b";
    if(ty_ == BPROP){
      std::swap(bs0, bs1);
      std::swap(ldb0, ldb1);
      std::swap(bcb0, bcb1);
      b = "trans(b)";
    }
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
                  int32 B, int32 H, int32 W,
                  int32 NF, int32 RH, int32 RW,
                  int32 NC, int32 R, int32 S,
                  int32 lda_n, int32 lda_c, int32 lda_h, int32 lda_w,
                  int32 ldc_n, int32 ldc_k, int32 ldc_p, int32 ldc_q,
                  int32 pad_h, int32 pad_w){
            int32 rxa[TM] = get_global_range[TM](0);
            int32 rb0[TN] = get_global_range[TN](1);
            int32 rka[TK] = 0 ... TK;
            int32 rb1[TK] = 0 ... TK;
            fp32 C[TM, TN] = 0;
            int32 rabh[TM] = rxa / RW;
            int32 raw[TM] = rxa % RW - pad_w;
            int32 rab[TM] = rabh / RH;
            int32 rah[TM] = rabh % RH - pad_h;
            int32 ra0[TM] = rab*lda_n + rah*lda_h + raw*lda_w;
            int32 racr[TK] = rka / S;
            int32 ras[TK] = rka % S;
            int32 rac[TK] = racr / R;
            int32 rar[TK] = racr % R;
            int32 ra1[TK] = rac*lda_c + rar*lda_h + ras*lda_w;
            fp32* pa[TM, TK] = a + ra1[newaxis, :] + ra0[:, newaxis];
            fp32* pb[)" + bs0 + ", " + bs1 + R"(] = b + rb1)" + bcb1 + ldb0 + " + rb0" + bcb0 + ldb1 + R"(;
            __constant__ int32* pincd[TK] = delta + rka;
            __constant__ int32* pd[TK] = delta + R*S + rka;
            int32 d[TK] = *pd;
            int32 incd[TK] = *pincd;
            int32 maskh[TM] = pad_h + min(rah, 0) + max(rah + R - H, 0);
            int32 maskw[TM] = pad_w + min(raw, 0) + max(raw + S - W, 0);
            __constant__ int32* pm[TM] = masks + R*S + maskw*R*S + maskh*R*S*(2*pad_w + 1);
            __constant__ int32* pincm[TM] = delta;
            int32 incm[TM] = *pincm;
            int32 checka0[TM] = *pm;
            int32 checka1[TK] = 1 << rka;
            int1 checka[TM, TK] = (checka0[:, newaxis] & checka1[newaxis, :]) > 0;
            fp32 a[TM, TK] = checka ? *pa : 0;
            fp32 b[)" + bs0 + ", " + bs1 + R"(] = *pb;
            for(int32 k = K; k > 0; k = k - TK){
              C = dot(a, trans(b), C);
              pb = pb + TK)" + ldb0 + R"(;
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
            int32 rcn[TM] = rxc / (RH*RW);
            int32 rcpq[TM] = rxc % (RH*RW);
            int32 rc0[TM] = rcn * ldc_n + rcpq;
            fp32* pc[TM, TN]  = c + rc1[newaxis, :]*ldc_k + rc0[:, newaxis];
            int1 checkc0[TM] = rxc < M;
            int1 checkc1[TN] = rc1 < N;
            int1 checkc[TM, TN]  = checkc0[:, newaxis] && checkc1[newaxis, :];
            @checkc *pc = C;
        })";
    return res;
  }

  template<class IN_DTYPE, class OUT_DTYPE>
  void cpu_ref(OUT_DTYPE* C,  IN_DTYPE* A, IN_DTYPE* B)
  {
    auto idx = [&](int32_t x, int32_t y, int32_t z, int32_t w, int32_t u,
                   int32_t /*s0*/, int32_t s1, int32_t s2, int32_t s3, int32_t s4)
    { return u + w*s4 + z*s4*s3 + y*s4*s3*s2 + x*s4*s3*s2*s1; };

    if(ty_==BPROP){
      std::swap(A, C);
    }
    std::cout << A[0] << std::endl;
    IN_DTYPE accs[1];
    float tmp[1];
    for(int32_t m = 0 ; m < RD_; ++m)
    for(int32_t p = 0 ; p < RH_; ++p)
    for(int32_t q = 0; q < RW_; ++q)
    for(int32_t n = 0; n < B_; ++n)
    for(int32_t k = 0; k < NF_ ; ++k)
    {
      for(int32_t i = 0; i < 1; ++i)
        accs[i] = 0;
      int32_t mm = m*stride_d_ - pad_d_;
      int32_t pp = p*stride_h_ - pad_h_;
      int32_t qq = q*stride_w_ - pad_w_;
      for(int32_t kk = 0; kk < 1; ++kk)
      for(int32_t c = 0; c < NC_; ++c)
      for(int32_t t = 0; t < T_; ++t)
      for(int32_t r = 0; r < R_; ++r)
      for(int32_t s = 0; s < S_; ++s){
        int32_t d = mm + t;
        int32_t h = pp + r;
        int32_t w = qq + s;
        bool in_bounds = (d >= 0 && h >= 0 && w >= 0 && d < D_ && h < H_ && w < W_);
        IN_DTYPE a = in_bounds?A[idx(n, c, d, h, w, B_, NC_, D_, H_, W_)]:0;
        IN_DTYPE b;
        if(ty_==FPROP)
          b = B[idx(c, t, r, s, k*1 + kk, NC_, T_, R_, S_, NF_*1)];
        else
          b = B[idx(c, t, s, r, k*1 + kk, NC_, T_, R_, S_, NF_*1)];
        accs[kk] = std::fma(a, b, accs[kk]);
      }
      for(int32_t kk = 0; kk < 1; ++kk){
        tmp[kk] = accs[kk];
      }
      C[idx(n, k, m, p, q, B_, NF_, RD_, RH_, RW_)] = tmp[0];
    }
  }

private:
  // image size
  int32_t B_;
  int32_t NC_;
  int32_t D_;
  int32_t H_;
  int32_t W_;
  // filter size
  int32_t T_;
  int32_t R_;
  int32_t S_;
  int32_t NF_;
  // activation size
  int32_t RD_;
  int32_t RH_;
  int32_t RW_;
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
  // memory strides for data
  int32_t stride_a_w_;
  int32_t stride_a_h_;
  int32_t stride_a_d_;
  int32_t stride_a_c_;
  int32_t stride_a_n_;
  // memory stride for activations
  int32_t stride_c_q_;
  int32_t stride_c_p_;
  int32_t stride_c_m_;
  int32_t stride_c_k_;
  int32_t stride_c_n_;
  // type
  type ty_;
  bool is_bprop_;
};

}
}
