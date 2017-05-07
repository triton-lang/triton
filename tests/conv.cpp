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

inline int32_t idx(int32_t x, int32_t y, int32_t z, int32_t w,
                   int32_t /*s0*/, int32_t s1, int32_t s2, int32_t s3)
{ return w + z*s3 + y*s3*s2 + x*s3*s2*s1; }

template<class DTYPE>
inline void to_cudnn(std::vector<DTYPE> const & in, std::vector<DTYPE>& out,
                     size_t C, size_t H, size_t W, size_t N){
  for(size_t c = 0; c < C ; ++c)
    for(size_t h = 0 ; h < H; ++h)
      for(size_t w = 0; w < W; ++w)
        for(size_t n = 0; n < N; ++n)
          out[idx(n, c, h, w, N, C, H, W)] = in[idx(c, h, w, n, C, H, W, N)];
}

template<class DTYPE>
inline void from_cudnn(std::vector<DTYPE> const & in, std::vector<DTYPE>& out,
                     size_t N, size_t K, size_t P, size_t Q){
    for(size_t k = 0; k < K ; ++k)
      for(size_t p = 0 ; p < P; ++p)
        for(size_t q = 0; q < Q; ++q)
          for(size_t n = 0; n < N; ++n)
            out[idx(k, p, q, n, K, P, Q, N)] = in[idx(n, k, p, q, N, K, P, Q)];
}


template<class DTYPE>
void do_test_impl(sc::driver::Context const & ctx, size_t N, size_t K, size_t H, size_t W, size_t C, size_t R, size_t S, size_t pad_h, size_t pad_w, size_t stride_h, size_t stride_w){
  sc::DType dtype = sc::to_DType<DTYPE>::value;
  size_t dtsize = sc::size_of(dtype);

  //alpha, beta are not half-precision
  sc::DType ab_dtype = dtype;
  if(ab_dtype != sc::DOUBLE_TYPE)
    ab_dtype = sc::FLOAT_TYPE;
  sc::scalar alpha(1., ab_dtype), beta(0., ab_dtype);

  //Initialize input/output buffers
  size_t P = (H - R + 1 + 2*pad_h)/stride_h;
  size_t Q = (W - S + 1 + 2*pad_w)/stride_w;
  std::vector<DTYPE> iO(N*K*P*Q);
  std::vector<DTYPE> iI(N*C*H*W);
  std::vector<DTYPE> iF(K*C*R*S);
  drv::Buffer O(ctx, iO.size()*dtsize);
  drv::Buffer I(ctx, iI.size()*dtsize);
  drv::Buffer F(ctx, iF.size()*dtsize);
  srand(0);  
  for(size_t i = 0; i < iI.size(); ++i) iI[i] = (float)rand()/RAND_MAX;
  for(size_t i = 0; i < iF.size(); ++i) iF[i] = (float)rand()/RAND_MAX;
  std::vector<DTYPE> iI_cudnn(iI.size());
  std::vector<DTYPE> iF_cudnn(iF.size());
  to_cudnn(iI, iI_cudnn, C, H, W, N);
  to_cudnn(iF, iF_cudnn, C, R, S, K);

  //Ground result (cuDNN)
  drv::Stream stream(ctx);
  stream.write(O, true, 0, iO.size()*dtsize, iO.data());
  stream.write(I, true, 0, iI.size()*dtsize, iI_cudnn.data());
  stream.write(F, true, 0, iF.size()*dtsize, iF_cudnn.data());
  sc::driver::cudnnConv(dtype, ctx, stream, H, W, N, K, P, Q, C, R, S, pad_h, pad_w, stride_h, stride_w, alpha, I, F, beta, O);
  std::vector<DTYPE> rO_cudnn(iO.size());
  std::vector<DTYPE> rO(iO.size());
  stream.read(O, true, 0, rO_cudnn.size()*dtsize, (void*)rO_cudnn.data());
  stream.write(O, true, 0, iO.size()*dtsize, iO.data());
  stream.write(I, true, 0, iI.size()*dtsize, iI.data());
  stream.write(F, true, 0, iF.size()*dtsize, iF.data());
  from_cudnn(rO_cudnn, rO, N, K, P, Q);

  //Test ISAAC
  sc::CONV(ctx.device(), stream, dtype, N, K, P, Q, C, R, S, H, W, pad_h, pad_w, stride_h, stride_w, alpha, I, F, beta, O);
  stream.read(O, true, 0, iO.size()*dtsize, (void*)iO.data());
  if(!is_correct(iO, rO, max_rounding_error(DTYPE(C))))
    exit(EXIT_FAILURE);

  std::vector<int> rv = {1,2,4};
  std::vector<int> rl = {1,8};
  std::vector<int> rs = {1,4};
  std::vector<int> rgrid = {1,8};
  std::vector<int> _1 = {1};

  for(auto x: sc::cpp::cartesian({rv, rl, rl, rl, rl, std::vector<int>{8}, rs, rs, rs, rs, rl, _1, rs, rl, rgrid})){
    isaac::templates::Conv conv(dtype, C, H, W, N, K, P, Q, R, S, pad_h, pad_w, stride_h, stride_w,
                                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14]);
    //Compile
    std::string src;
    try{
      src = conv.dump(ctx.device(), "fprop");
    }catch(isaac::templates::invalid_parameters){
      continue;
    }
    //Compile
    drv::Module program(ctx, src, true);
    drv::Kernel kernel(program, "fprop");
    //Launch
    try{
      conv.enqueue(kernel, stream, alpha, I, F, beta, O);
    }catch(isaac::driver::exception::cuda::launch_out_of_resources){
      continue;
    }
    stream.synchronize();
    //Test
    stream.read(O, true, 0, iO.size()*dtsize, (void*)iO.data());
    size_t depth = x[12]*x[13]*x[14];
    double eps = max_rounding_error(DTYPE(C/depth))*depth;
    if(!is_correct(iO, rO, eps))
      exit(EXIT_FAILURE);
  }
}

template<class DTYPE>
int do_test(sc::driver::Context const & ctx, size_t N, size_t K, size_t H, size_t W, size_t C, size_t R, size_t S, size_t pad_h, size_t pad_w, size_t stride_h, size_t stride_w){
  std::cout << "CONV: FPROP" << std::endl;
  std::cout << "-----------" << std::endl;
  std::cout << "(" << N << ", " << K << ", " << H << ", " << W << ", " <<  C << ", " << R << ", " << S << ")..." << std::endl;
  do_test_impl<DTYPE>(ctx, N, K, H, W, C, R, S, pad_h, pad_w, stride_h, stride_w);
  std::cout << "-----------" << std::endl;
  std::cout << std::endl;
  return EXIT_SUCCESS;
}

int main(){
  auto ctx = drv::backend::contexts::get_default();
  if(ctx.device().compute_capability().first>=6){
    std::cout << "===============" << std::endl;
    std::cout << "HALF:" << std::endl;
    std::cout << "===============" << std::endl;
    do_test<half_float::half>(ctx, 31, 41, 23, 23, 64, 5, 5, 3, 0, 1, 1);
  }
  std::cout << "===============" << std::endl;
  std::cout << "FLOAT:" << std::endl;
  std::cout << "===============" << std::endl;
  do_test<float>(ctx, 31, 43, 23, 23, 64, 5, 5, 3, 0, 1, 1);
}
