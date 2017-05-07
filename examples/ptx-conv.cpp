#include <sstream>
#include <chrono>
#include <exception>
#include <fstream>
#include <iomanip>
#include "isaac/driver/backend.h"
#include "isaac/driver/module.h"
#include "isaac/driver/error.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/cublas.h"
#include "isaac/driver/stream.h"
#include "isaac/driver/buffer.h"
#include "isaac/templates/error.hpp"

#include <string>
#include <iostream>
#include <cassert>
#include <cstdlib>

#include "isaac/tools/bench.hpp"
#include "isaac/templates/conv.h"

namespace sc = isaac;
namespace drv = isaac::driver;


inline int32_t idx(int32_t x, int32_t y, int32_t z, int32_t w,
                   int32_t /*s0*/, int32_t s1, int32_t s2, int32_t s3)
{ return w + z*s3 + y*s3*s2 + x*s3*s2*s1; }

void cpp_conv_nchw(int32_t C, int32_t N, int32_t K,
              int32_t H, int32_t W,
              int32_t R, int32_t S,
              int32_t pad_h, int32_t pad_w,
              int32_t stride_h, int32_t stride_w,
              int32_t P, int32_t Q,
              float* O, float* I, float* F)
{
  for(int32_t k = 0; k < K; ++k)
    for(int32_t p = 0 ; p < P; ++p)
      for(int32_t q = 0; q < Q; ++q)
        for(int32_t n = 0; n < N; ++n)
        {
          int32_t pp = p*stride_h - pad_h;
          int32_t qq = q*stride_w - pad_w;
          float acc = 0;
          for(int32_t c = 0; c < C; ++c)
            for(int32_t r = 0; r < R; ++r)
              for(int32_t s = 0; s < S; ++s)
              {
                int32_t h = pp + r;
                int32_t w = qq + s;
                if(h >= 0 && h < H && w >= 0 && w < W)
                  acc += F[idx(k, c, r, s, K, C, R, S)]*I[idx(n, c, h, w, N, C, H, W)];
              }
          O[idx(n, k, p, q, N, K, P, Q)] = acc;
        }
}

void cpp_conv_chwn(int32_t C, int32_t N, int32_t K,
              int32_t H, int32_t W,
              int32_t R, int32_t S,
              int32_t pad_h, int32_t pad_w,
              int32_t stride_h, int32_t stride_w,
              int32_t P, int32_t Q,
              float* O, float* I, float* F)
{
  for(int32_t k = 0; k < K ; ++k)
    for(int32_t p = 0 ; p < P; ++p)
      for(int32_t q = 0; q < Q; ++q)
        for(int32_t n = 0; n < N; ++n)
        {
          int32_t pp = p*stride_h - pad_h;
          int32_t qq = q*stride_w - pad_w;
          float acc = 0;
          for(int32_t c = 0; c < C; ++c)
            for(int32_t r = 0; r < R; ++r)
              for(int32_t s = 0; s < S; ++s)
              {
                int32_t h = pp + r;
                int32_t w = qq + s;
                if(h >= 0 && h < H && w >= 0 && w < W)
                  acc += F[idx(c, r, s, k, C, R, S, K)]*I[idx(c, h, w, n, C, H, W, N)];
              }
          O[idx(k, p, q, n, K, P, Q, N)] = acc;
        }
}

double get_tflops(uint64_t P, uint64_t Q, uint64_t K, uint64_t N, uint64_t C, uint64_t R, uint64_t S, double time){
  return 2*P*Q*K*N*C*R*S/(time*1e3);
}

bool test = false;

int main(){
  auto ctx = drv::backend::contexts::get_default();
  int32_t dtsize = 4;

  //Arguments

  int32_t C = 1, N = 4, K = 32;
  int32_t H = 68, W = 260;
  int32_t R = 5, S = 5;
  int32_t pad_h = 0, pad_w = 0;
  int32_t stride_h = 1, stride_w = 1;
  int32_t P = (H - R + 1 + 2*pad_h)/stride_h, Q = (W - S + 1 + 2*pad_w)/stride_w;
  std::vector<float> iO(K*P*Q*N);
  std::vector<float> iI(C*H*W*N);
  std::vector<float> iF(C*R*S*K);
  drv::Buffer O(ctx, iO.size()*dtsize);
  drv::Buffer I(ctx, iI.size()*dtsize);
  for(size_t i = 0; i < iI.size(); ++i) iI[i] = (float)rand()/RAND_MAX;
  drv::Buffer F(ctx, iF.size()*dtsize);
  for(size_t i = 0; i < iF.size(); ++i) iF[i] = (float)rand()/RAND_MAX;
  drv::Stream queue(ctx);
  queue.write(O, true, 0, iO.size()*dtsize, iO.data());
  queue.write(I, true, 0, iI.size()*dtsize, iI.data());
  queue.write(F, true, 0, iF.size()*dtsize, iF.data());
  sc::scalar alpha(1., sc::FLOAT_TYPE);
  sc::scalar beta(1., sc::FLOAT_TYPE);

  if(test)
    cpp_conv_chwn(C, N, K, H, W, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, iO.data(), iI.data(), iF.data());
  std::vector<float> rO(iO.size());


  std::vector<int> rv = {2,4};
  std::vector<int> rl = {1,2,4};
  std::vector<int> rs = {1,2,4,8};
  float best = 0;
  for(size_t vec: rv)
  for(size_t bp: std::vector<int>{})
  for(size_t bq: std::vector<int>{1,2,4})
  for(size_t bn: rl)
  for(size_t bk: rl)
  for(size_t bf_n: rl)
  for(size_t ps: std::vector<int>{1,2,4})
  for(size_t qs: std::vector<int>{1,2,4})
  for(size_t ns: rs)
  for(size_t ks: rs)
  for(size_t crs_l: rl)
  for(size_t crs_s: std::vector<int>{1})
  for(size_t cs: std::vector<int>{1})
  for(size_t bc: std::vector<int>{1})
  for(size_t gridc: std::vector<int>{1})
  {
  // Compile
  isaac::templates::Conv conv(sc::FLOAT_TYPE, C, H, W, N, K, P, Q, R, S, pad_h, pad_w, stride_h, stride_w, vec, bp, bq, bn, bk, bf_n, ps, qs, ns, ks, crs_l, crs_s, cs, bc, gridc);
  std::string src;
  try{
    src = conv.dump(ctx.device(), "fconv");
  }catch(isaac::templates::invalid_parameters){
    continue;
  }
  drv::Module program(ctx, src, true);
  drv::Kernel kernel(program, "fconv");

  //Launch
  float time;
  try{
    time = bench([&](){ conv.enqueue(kernel, queue, alpha, I, F, beta, O); },
      [&](){ queue.synchronize(); }, ctx.device());
  }catch(drv::exception::cuda::launch_out_of_resources){
    continue;
  }

  //Report
  float tflops = get_tflops(P,Q,K,N,C,R,S,time);
  best = std::max(tflops, best);
  std::cout << "//" << vec << " " << bp << " " << bq << " " << bn << " " << bk << " " << bf_n << " " << ps << " " << qs << " " << ns << " " << ks << " " << crs_l << " " << crs_s << " " << cs << " " << bc << " " << gridc << ": " << std::setprecision(3) << tflops << "  [ " << best << " ] " << std::endl;

  //Test
  if(test){
    queue.read(O, true, 0, rO.size()*dtsize, rO.data());
    for(size_t i = 0 ; i < rO.size(); ++i)
      if(fabs((iO[i] - rO[i])/rO[i]) > 1e-4 || std::isnan(rO[i])) {  std::cout << "// Failure at idx " << i << ": " << iO[i] << " != " << rO[i] << std::endl; exit(1); }
  }

  }

  //cuDNN
  float time = bench([&](){sc::driver::cudnnConv(sc::FLOAT_TYPE, ctx, queue, H, W, N, K, P, Q, C, R, S, pad_h, pad_w, stride_h, stride_w, alpha, I, F, beta, O); },
                      [&](){ queue.synchronize();  }, ctx.device());
  float tflops = get_tflops(P,Q,K,N,C,R,S,time);
  std::cout << "TFLOPs: " << tflops << std::endl;
}
