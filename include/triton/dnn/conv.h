#include <string>
#include <vector>

namespace triton{
namespace dnn{

class conv {
public:
  enum type {
    FPROP,
    BPROP,
    WGRAD
  };

  static void build_lut(int TK,
                        int stride_d, int stride_h, int stride_w, int stride_c,
                        int pad_d, int pad_h, int pad_w,
                        int T, int R, int S,
                        std::vector<int>& res, std::vector<int>& masks) {
      /* convolution parameters */
      int F = T * R * S;
      int Nlut = (TK + F - 1) / F * F;
      int upsample_w = 1;
      int upsample_h = 1;
      int upsample_d = 1;
      /* unpack index wrt filters */
      auto unpack = [&](int32_t trs){
        int32_t tr = trs / S;
        int32_t s = trs - tr*S;
        int32_t t = tr / R;
        int32_t r = tr - t*R;
        return std::make_tuple(t, r, s);
      };
      /* increments */
      for(size_t i = 0; i < Nlut; ++i)
        res[i] = (((i + TK) % Nlut) - i);
      /* deltas */
      size_t Ds0 = Nlut;
      size_t Ds1 = upsample_w;
      size_t Ds2 = upsample_h;
      size_t Ds3 = upsample_d;
      for(size_t pd = 0; pd < Ds3; ++pd)
      for(size_t ph = 0; ph < Ds2; ++ph)
      for(size_t pw = 0; pw < Ds1; ++pw){
        int32_t* deltas_ptr = &res[Nlut + pw*Ds0 + ph*Ds0*Ds1 + pd*Ds0*Ds1*Ds2];
        // cumulative increments
        for(size_t i = 0; i < Ds0; ++i){
          int32_t ctrs = i;
          int32_t c = ctrs / F;
          int32_t t, r, s;
          std::tie(t, r, s) = unpack(ctrs % F);
          // next indices
          int32_t nextctrs = ctrs + TK;
          int32_t nextc = nextctrs / F;
          int32_t nextt, nextr, nexts;
          std::tie(nextt, nextr, nexts) = unpack(nextctrs % F);
          // diffs
          int32_t cdiff = nextc - c;
          int32_t tdiff = (nextt + pd)/upsample_d - (t + pd)/upsample_d;
          int32_t rdiff = (nextr + ph)/upsample_h - (r + ph)/upsample_h;
          int32_t sdiff = (nexts + pw)/upsample_w - (s + pw)/upsample_w;
          // delta pointers
          deltas_ptr[i] = cdiff*stride_c + sdiff*stride_w + rdiff*stride_h + tdiff*stride_d;
        }
      }

      /* Masks */
      size_t Ms0 = Nlut;
      size_t Ms1 = 2*pad_w + 1;
      size_t Ms2 = 2*pad_h + 1;
      size_t Ms3 = 2*pad_d + 1;
      for(size_t pd = 0; pd < Ms3; ++pd)
      for(size_t ph = 0; ph < Ms2; ++ph)
      for(size_t pw = 0; pw < Ms1; ++pw){
        int32_t* masks_ptr = &masks[Nlut + pw*Ms0 + ph*Ms0*Ms1 + pd*Ms0*Ms1*Ms2];
        for(size_t i = 0; i < Ms0; ++i){
           int32_t t, r, s;
           int32_t mask = 0x0;
           for(size_t j = 0; j < TK; ++j){
             std::tie(t, r, s) = unpack((i + j) % F);
             bool in_bounds_d = (t + pd) >= pad_d && (t + pd) < (T + pad_d);
             bool in_bounds_h = (r + ph) >= pad_h && (r + ph) < (R + pad_h);
             bool in_bounds_w = (s + pw) >= pad_w && (s + pw) < (S + pad_w);
             mask |= (in_bounds_d && in_bounds_h && in_bounds_w) << j;
           }
           masks_ptr[i] = mask;
        }
      }
      for(size_t i = 0; i < Nlut; ++i)
        masks[i] = 0x0;

  }

  static std::vector<unsigned> default_params() {
    return {16, 2, 64, 32, 2, 64, 16, 8, 2, 2, 8, 1, 8, 4 };
  }

  static void init_cst(int stride_d, int stride_h, int stride_w, int stride_c,
                       int pad_d, int pad_h, int pad_w,
                       int T, int R, int S,
                       std::vector<int> &h_delta, std::vector<int> &h_masks) {
    int upsample_d = 1;
    int upsample_h = 1;
    int upsample_w = 1;
    int TK = 8;
    int F = T * R * S;
    int nlut = (TK + F - 1) / F * F;
    h_delta.resize(nlut + upsample_d*upsample_h*upsample_w*nlut);
    h_masks.resize(nlut + (2*pad_h+1)*(2*pad_w+1)*(2*pad_d+1)*nlut);
    build_lut(TK, stride_d, stride_h, stride_w, stride_c, pad_d, pad_h, pad_w, T, R, S, h_delta, h_masks);
  }

  static std::string src(type ty = FPROP) {

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
            fp32* pb[TN, TK] = b + rb1[newaxis, :]*NF + rb0[:, newaxis];
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
            fp32 b[TN, TK] = *pb;
            for(int32 k = K; k > 0; k = k - TK){
              C = dot(a, trans(b), C);
              pb = pb + TK*NF;
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
};

}
}
