#include <vector>
#include <chrono>
#include <cmath>
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

template<class T>
void simple_gemm(bool AT, bool BT, std::vector<T> &c, const std::vector<T> &a, const std::vector<T> &b, size_t M, size_t N, size_t K) {
  if(AT && BT)
    simple_gemm<T, true, true>(c, a, b, M, N, K);
  else if(AT && !BT)
    simple_gemm<T, true, false>(c, a, b, M, N, K);
  else if(!AT && BT)
    simple_gemm<T, false, true>(c, a, b, M, N, K);
  else
    simple_gemm<T, false, false>(c, a, b, M, N, K);
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
      acc = std::fma(a, b, acc);
    }
    O[bs + q*BS + p*BS*W + k*BS*H*W] = acc;
  }
}
