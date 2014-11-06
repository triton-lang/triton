#ifndef ATIDLAS_BENCH_COMMON_HPP_
#define ATIDLAS_BENCH_COMMON_HPP_

#include "vector"
#include "atidlas/tools/misc.hpp"

int ceil(int N, int pad)
{
    return (N%pad==0)?N:(N+pad-1)/pad*pad;
}

std::vector<int> create_log_range(int min, int max, int N, int pad)
{
  std::vector<int> res(N);
  for(int i = 0 ; i < N ; ++i)
  {
    res[i] = std::exp(std::log(min) + (float)(std::log(max) - std::log(min))*i/N);
    res[i] = ceil(res[i], pad);
  }
  return res;
}

std::vector<int> create_full_range(int min, int max, int pad)
{
    std::vector<int> N;
    for(int i = ceil(min, pad) ; i < ceil(max, pad) ; i+=pad)
        N.push_back(i);
    return N;
}

// BLAS1 Sizes
static const std::vector<int> BLAS1_N = create_log_range(1e3, 2e7, 50, 64);

// BLAS2 Sizes
static const std::vector<int> BLAS2_M = atidlas::tools::make_vector<int>() << 256;
static const std::vector<int> BLAS2_N = create_full_range(128, 5000, 64);

// BLAS3 Sizes
static const std::vector<int> BLAS3_N = create_full_range(128, 5000, 64);


template<class T>
float bandwidth(std::size_t N, float t)
{
  return N * sizeof(T) * 1e-9 / t;
}

template<class T>
T median(std::vector<T> x)
{
  size_t size = x.size();
  std::sort(x.begin(), x.end());
  if (size  % 2 == 0)
      return (x[size / 2 - 1] + x[size / 2]) / 2;
  else
      return x[size / 2];
}

#endif
