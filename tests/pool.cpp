#include <sstream>
#include <chrono>
#include <exception>
#include <iomanip>
#include <string>
#include <iostream>
#include <iterator>
#include <cassert>
#include <cmath>
#include <cfenv>


#include "isaac/driver/backend.h"
#include "isaac/driver/error.h"
#include "isaac/driver/module.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/stream.h"
#include "isaac/driver/buffer.h"

#include "isaac/templates/conv.h"
#include "isaac/templates/error.hpp"

#include "isaac/driver/cublas.h"

#include "isaac/tools/collections.hpp"
#include "isaac/api.h"

#include "test_utils.hpp"

namespace sc = isaac;
namespace drv = isaac::driver;


template<class IN_DTYPE, class OUT_DTYPE>
void cpp_pool(sc::PoolType pool_type,
              int32_t N, int32_t C,
              int32_t D, int32_t H, int32_t W,
              int32_t T, int32_t R, int32_t S,
              int32_t pad_d, int32_t pad_h, int32_t pad_w,
              int32_t stride_d, int32_t stride_h, int32_t stride_w,
              int32_t M, int32_t P, int32_t Q,
              OUT_DTYPE* O, IN_DTYPE* I,
              float i_scale, float o_scale)
{
  static const int PACK_IN = pack_increment<IN_DTYPE>::VALUE;
  static const int PACK_OUT = pack_increment<OUT_DTYPE>::VALUE;

  float scale = (pool_type==sc::AvgPool)?1./(float)(T*R*S):1.;
  if(C % PACK_IN != 0) throw std::runtime_error("Number of input channels must be a multiple of 4");
  int32_t Cin = C/PACK_IN;
  int32_t Cout = C/PACK_OUT;

  std::function<IN_DTYPE(IN_DTYPE, IN_DTYPE)> accumulate = (pool_type == sc::MaxPool)?&max<IN_DTYPE>:&plus<IN_DTYPE>;

  float acc[PACK_IN];
  float unpacked[PACK_IN];

  for(int32_t m = 0 ; m < M; ++m)
  for(int32_t p = 0 ; p < P; ++p)
  for(int32_t q = 0; q < Q; ++q)
  for(int32_t n = 0; n < N; ++n)
  for(int32_t c = 0; c < Cin ; ++c)
  {
    int32_t mm = m*stride_d - pad_d;
    int32_t pp = p*stride_h - pad_h;
    int32_t qq = q*stride_w - pad_w;

    // Initialize accumulators
    for(size_t j = 0; j < PACK_IN ; j++)
      acc[j] = (sc::MaxPool)?-INFINITY:0;

    // Accumulate
    for(int32_t t = 0; t < T; ++t)
    for(int32_t r = 0; r < R; ++r)
    for(int32_t s = 0; s < S; ++s){
      int32_t d = mm + t;
      int32_t h = pp + r;
      int32_t w = qq + s;
      bool in_bounds = (d >= 0 && h >= 0 && w >= 0 && d < D && h < H && w < W);
      IN_DTYPE i = in_bounds?I[idx(n, c, d, h, w, N, Cin, D, H, W)]:0;
      unpack(unpacked, i, 1.);
      for(size_t j = 0; j < PACK_IN ; j++)
        acc[j] = accumulate(acc[j], unpacked[j]);
    }

    // Write back
    for(int32_t cc = 0; cc < PACK_IN; cc+=PACK_OUT)
      O[idx(n, (c*PACK_IN/PACK_OUT + cc), m, p, q, N, Cout, M, P, Q)] = pack<OUT_DTYPE>(&acc[cc], (o_scale/i_scale)*scale);
  }
}


template<class T>
bool abs_cmp(T a, T b)
{ return std::abs(a) < std::abs(b);}

template<class IN_DTYPE, class OUT_DTYPE>
void do_test_impl(sc::driver::Context const & ctx, isaac::PoolType pool_type, size_t N, size_t K, size_t D, size_t H, size_t W, size_t T, size_t R, size_t S, size_t pad_d, size_t pad_h, size_t pad_w, size_t stride_d, size_t stride_h, size_t stride_w){
  static const int PACK_IN = pack_increment<IN_DTYPE>::VALUE;
  static const int PACK_OUT = pack_increment<OUT_DTYPE>::VALUE;
  sc::DType in_dtype = sc::to_DType<IN_DTYPE>::value;
  sc::DType out_dtype = sc::to_DType<OUT_DTYPE>::value;
  size_t dtsize = sc::size_of(in_dtype);

  //Initialize input/output buffers
  sc::param_t M, P, Q;
  sc::templates::Conv::output_shapes(D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, 1, 1, 1, M, P, Q);

  std::vector<OUT_DTYPE> iO(N*K*P*Q*M/PACK_OUT);
  std::vector<OUT_DTYPE> rO(iO.size());
  std::vector<IN_DTYPE> iI(N*K*H*W*D/PACK_IN);
  drv::Buffer O(ctx, iO.size()*dtsize);
  drv::Buffer I(ctx, iI.size()*dtsize);
  srand(0);
  for(size_t i = 0; i < iI.size(); ++i)
    iI[i] = (in_dtype==sc::INT8X4_TYPE)?rand():(float)rand()/RAND_MAX;
  float i_scale = (in_dtype==sc::INT8X4_TYPE)?1.513:1;
  float o_scale = (out_dtype==sc::INT8X4_TYPE)?1.513:1;

  //Ground result (cuDNN)
  cpp_pool(pool_type, N, K, D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, M, P, Q, rO.data(), iI.data(), i_scale, o_scale);

  //Test ISAAC
  drv::Stream stream(ctx);
  stream.write(I, true, 0, iI.size()*dtsize, iI.data());
  sc::POOL(ctx.device(), stream, in_dtype, out_dtype, pool_type, K, M, P, Q, N, T, R, S, D, H, W, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, I, O, i_scale, o_scale);
  stream.read(O, true, 0, iO.size()*dtsize, (void*)iO.data());
  if(!is_correct(iO, rO, 1e-3))
    exit(EXIT_FAILURE);

  std::vector<int> rv = {1, 2, 4};
  std::vector<int> rl = {32, 64, 128, 256};
  std::vector<int> rs = {4, 8};
  for(auto x: sc::cpp::cartesian({rv, rl, rs, std::vector<int>{4}})){
    isaac::templates::Pool pool(in_dtype, out_dtype, pool_type, K, D, H, W, N, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w,
                                x[0], x[1], x[2], x[3]);
    //Compile
    std::string src;
    try{
      src = pool.dump(ctx.device(), "pool");
    }catch(isaac::templates::invalid_parameters){
      continue;
    }
    //Compile
    drv::Module program(ctx, src);
    drv::Kernel kernel(program, "pool");
    //Launch
    try{
      pool.enqueue(kernel, stream, I, O, i_scale, o_scale);
    }catch(isaac::driver::exception::cuda::launch_out_of_resources){
      continue;
    }
    stream.synchronize();
    //Test
    stream.read(O, true, 0, iO.size()*dtsize, (void*)iO.data());
    if(!is_correct(iO, rO, 1e-3))
      exit(EXIT_FAILURE);
  }
}

template<class IN_DTYPE, class OUT_DTYPE>
int do_test(sc::driver::Context const & ctx, std::string const & prefix, sc::PoolType pool_type, size_t N, size_t K, size_t D, size_t H, size_t W, size_t T, size_t R, size_t S, size_t pad_d, size_t pad_h, size_t pad_w, size_t stride_d, size_t stride_h, size_t stride_w){
  auto params = {N, K, D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w};
  std::cout << "(";
  std::copy(params.begin(), params.end(), std::ostream_iterator<size_t>(std::cout, ", "));
  std::cout << "\b\b) [" << prefix << "]" << std::endl;
  do_test_impl<IN_DTYPE, OUT_DTYPE>(ctx, pool_type, N, K, D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w);
  return EXIT_SUCCESS;
}

int main(){
  auto ctx = drv::backend::contexts::get_default();
  std::cout << "===============" << std::endl;
  std::cout << "POOL" << std::endl;
  std::cout << "===============" << std::endl;
  std::cout << "---------------" << std::endl;
  do_test<float, float>(ctx, "max + core", sc::MaxPool, 5, 41, 31, 7, 13, 3, 3, 3, 0, 0, 0, 1, 1, 1);
  do_test<int32_t, int32_t>(ctx, "max + int8x4", sc::MaxPool, 5, 40, 31, 7, 13, 3, 3, 3, 0, 0, 0, 1, 1, 1);
  do_test<int32_t, float>(ctx, "max + dequantize", sc::MaxPool, 5, 40, 31, 7, 13, 3, 3, 3, 0, 0, 0, 1, 1, 1);
  do_test<float, float>(ctx, "avg + core", sc::AvgPool, 5, 41, 31, 7, 13, 3, 3, 3, 0, 0, 0, 1, 1, 1);
  do_test<int32_t, int32_t>(ctx, "avg + int8x4", sc::AvgPool, 5, 40, 31, 7, 13, 3, 3, 3, 0, 0, 0, 1, 1, 1);
  do_test<float, float>(ctx, "max + stride", sc::MaxPool, 5, 41, 31, 7, 13, 3, 3, 3, 0, 0, 0, 6, 3, 4);
  do_test<float, float>(ctx, "max + pad", sc::MaxPool, 5, 41, 31, 7, 13, 3, 3, 3, 1, 2, 1, 1, 1, 1);
  do_test<float, float>(ctx, "max + pad + stride", sc::MaxPool, 5, 41, 31, 7, 13, 3, 3, 3, 1, 2, 1, 6, 3, 4);
  do_test<int32_t, int32_t>(ctx, "max + int8x4 + stride", sc::MaxPool, 5, 40, 31, 7, 13, 3, 3, 3, 0, 0, 0, 6, 3, 4);
  do_test<int32_t, int32_t>(ctx, "max + int8x4 + pad", sc::MaxPool, 5, 40, 31, 7, 13, 3, 3, 3, 1, 2, 1, 1, 1, 1);
  do_test<int32_t, int32_t>(ctx, "max + int8x4 + pad + stride", sc::MaxPool, 5, 40, 31, 7, 13, 3, 3, 3, 1, 2, 1, 6, 3, 4);
  std::cout << "---------------" << std::endl;
}
