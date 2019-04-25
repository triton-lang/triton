#include <vector>
#include <chrono>
#include "triton/driver/device.h"
#include <algorithm>

template<class T, bool AT, bool BT>
void simple_gemm(std::vector<T> &c, const std::vector<T> &a, const std::vector<T> &b, size_t M, size_t N, size_t K){
  for(size_t m = 0; m < M; m++)
  for(size_t n = 0; n < N; n++){
    T acc = 0;
    for(size_t k = 0; k < K; k++)
      acc += (AT?a[k + m*K]:a[m + k*M]) * (BT?b[n + k*N]:b[k + n*K]);
    c[m + n*M] = acc;
  }
}


class timer{
    typedef std::chrono::high_resolution_clock high_resolution_clock;
    typedef std::chrono::nanoseconds nanoseconds;

public:
    explicit timer(bool run = false)
    { if (run) start(); }

    void start()
    { _start = high_resolution_clock::now(); }

    nanoseconds get() const
    { return std::chrono::duration_cast<nanoseconds>(high_resolution_clock::now() - _start); }

private:
    high_resolution_clock::time_point _start;
};

template<class T>
T min(std::vector<T> x)
{ return *std::min_element(x.begin(), x.end()); }


template<class OP, class SYNC>
double bench(OP const & op, SYNC const & sync, triton::driver::device const & device)
{
  timer tmr;
  std::vector<size_t> times;
  double total_time = 0;
  op();
  sync();
  while(total_time*1e-9 < 1e-3){
    float norm = 1;
    // normalize clock if possible to get roughly constant result
    if(auto cu_device = dynamic_cast<const triton::driver::cu_device*>(&device))
      norm = (float)cu_device->current_sm_clock()/cu_device->max_sm_clock();
    tmr.start();
    op();
    sync();
    times.push_back(norm*tmr.get().count());
    total_time+=times.back();
  }
  return min(times);
}

//

void build_conv_lut(int TK,
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


// Index computation
inline int32_t idx(int32_t x, int32_t y, int32_t z, int32_t w, int32_t u,
                   int32_t /*s0*/, int32_t s1, int32_t s2, int32_t s3, int32_t s4)
{ return u + w*s4 + z*s4*s3 + y*s4*s3*s2 + x*s4*s3*s2*s1; }


// Pack

template <class T> T clamp(T x, T lo, T hi){
  return std::max<T>(lo, std::min<T>(x, hi));
}


template<class T, class U>
T pack(U* tmp, U scale);

template<>
double pack<double, double>(double* tmp, double scale)
{ return tmp[0]*scale; }

template<>
float pack<float, float>(float* tmp, float scale)
{ return tmp[0]*scale; }

template<>
int pack<int, float>(float* tmp, float scale)
{
  int res = 0;
  for(int i = 0; i < 4; i++){
    int8_t clamped = std::round(clamp(tmp[i]*scale, (float)-128, (float)127));
    res |= (clamped & 0xFF) << (8*i);
  }
  return res;
}

template<class T> struct pack_increment
{ enum{ VALUE = 1}; };

template<> struct pack_increment<int>
{ enum{ VALUE = 4}; };

// Dot
template<class T>
inline T dot(T x, T y, T z)
{
  return std::fma(x, y, z);
}

inline int dot(int x, int y, int z){
  int res = 0;
  for(int i = 0; i < 4; i++){
    int32_t a = ((x >> (8*i)) & 0x000000FF);
    int32_t b = ((y >> (8*i)) & 0x000000FF);
    res +=  (*(int8_t*)(&a)) * (*(int8_t*)(&b));
  }
  return res + z;
}



template<class IN_DTYPE, class OUT_DTYPE>
void cpp_conv_nchw(int32_t C, int32_t N, int32_t K,
              int32_t D, int32_t H, int32_t W,
              int32_t T, int32_t R, int32_t S,
              int32_t pad_d, int32_t pad_h, int32_t pad_w,
              int32_t stride_d, int32_t stride_h, int32_t stride_w,
              int32_t M, int32_t P, int32_t Q,
              std::vector<OUT_DTYPE>& O,
              const std::vector<IN_DTYPE>& I,
              const std::vector<IN_DTYPE>& F)
{
  static const int PACK_IN = pack_increment<IN_DTYPE>::VALUE;
  static const int PACK_OUT = pack_increment<OUT_DTYPE>::VALUE;
  if(C % PACK_IN != 0) throw std::runtime_error("Number of input channels must be a multiple of 4");
  if(K % PACK_OUT != 0) throw std::runtime_error("Number of output channels must be a multiple of 4");
  C /= PACK_IN;
  K /= PACK_OUT;
  int32_t Kout = K;
  IN_DTYPE accs[PACK_OUT];
  float tmp[PACK_OUT];
  for(int32_t m = 0 ; m < M; ++m)
  for(int32_t p = 0 ; p < P; ++p)
  for(int32_t q = 0; q < Q; ++q)
  for(int32_t n = 0; n < N; ++n)
  for(int32_t k = 0; k < Kout ; ++k)
  {
    for(int32_t i = 0; i < PACK_OUT; ++i)
      accs[i] = 0;
    int32_t mm = m*stride_d - pad_d;
    int32_t pp = p*stride_h - pad_h;
    int32_t qq = q*stride_w - pad_w;
    for(int32_t kk = 0; kk < PACK_OUT; ++kk)
    for(int32_t c = 0; c < C; ++c)
    for(int32_t t = 0; t < T; ++t)
    for(int32_t r = 0; r < R; ++r)
    for(int32_t s = 0; s < S; ++s){
      int32_t d = mm + t;
      int32_t h = pp + r;
      int32_t w = qq + s;
      bool in_bounds = (d >= 0 && h >= 0 && w >= 0 && d < D && h < H && w < W);
      IN_DTYPE i = in_bounds?I[idx(n, c, d, h, w, N, C, D, H, W)]:0;
      IN_DTYPE f = F[idx(c, t, r, s, k*PACK_OUT + kk, C, T, R, S, K*PACK_OUT)];
      accs[kk] = dot(i, f, accs[kk]);
    }
    for(int32_t kk = 0; kk < PACK_OUT; ++kk){
      tmp[kk] = accs[kk];
    }
    O[idx(n, k, m, p, q, N, K, M, P, Q)] = tmp[0];
  }
}


// input layout: C, H, W, BS
// filter layout: C, K
// output layout: K, H, W, BS
template<class IN_DTYPE, class OUT_DTYPE>
void shift_conv(int32_t C, int32_t H, int32_t W, int32_t BS,
                int32_t K,
                std::vector<OUT_DTYPE>& O,
                const std::vector<IN_DTYPE>& I,
                const std::vector<IN_DTYPE>& F,
                const std::vector<int32_t> shift_h,
                const std::vector<int32_t> shift_w)
{
  OUT_DTYPE acc;
  for(int32_t p = 0; p < H; ++p)
  for(int32_t q = 0; q < W; ++q)
  for(int32_t bs = 0; bs < BS; ++bs)
  for(int32_t k = 0; k < K; ++k)
  {
    acc = 0;
    for(int32_t c = 0; c < C; ++c){
      int32_t h = p + shift_h[c];
      int32_t w = q + shift_w[c];
      bool in_bounds = (h >= 0 && w >= 0 && h < H && w < W);
      IN_DTYPE a = in_bounds?I[bs + w*BS + h*BS*W + c*BS*H*W]:0;
      IN_DTYPE b = F[k + c*K];
      acc = dot(a, b, acc);
    }
    O[bs + q*BS + p*BS*W + k*BS*H*W] = acc;
  }
}
