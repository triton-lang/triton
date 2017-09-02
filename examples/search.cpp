#include <sstream>
#include <chrono>
#include <exception>
#include <iomanip>
#include <string>
#include <iostream>
#include <cassert>

#include "opts.hpp"

#include "isaac/driver/backend.h"
#include "isaac/driver/error.h"
#include "isaac/driver/module.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/stream.h"
#include "isaac/driver/buffer.h"

#include "isaac/driver/cublas.h"
#include "isaac/external/half.hpp"

#include "isaac/tools/bench.hpp"
#include "isaac/tools/collections.hpp"
#include "isaac/templates/gemm.h"
#include "isaac/templates/conv.h"
#include "isaac/templates/error.hpp"

namespace sc = isaac;
namespace drv = isaac::driver;

void search_conv(int32_t W, int32_t H, int32_t C, int32_t N, int32_t K, int32_t R, int32_t S, int32_t pad_h, int32_t pad_w, int32_t stride_h, int32_t stride_w, sc::DType dtype){
  auto ctx = drv::backend::contexts::get_default();
  size_t dtsize = sc::size_of(dtype);
  int32_t P = (H - R + 1 + 2*pad_h)/stride_h, Q = (W - S + 1 + 2*pad_w)/stride_w;

  //Setup
  std::vector<float> iO();
  std::vector<float> iI(C*H*W*N);
  std::vector<float> iF(C*R*S*K);
  drv::Buffer O(ctx, K*P*Q*N*dtsize);
  drv::Buffer I(ctx, C*H*W*N*dtsize);
  drv::Buffer F(ctx, C*R*S*K*dtsize);
  drv::Stream stream(ctx);
  sc::scalar alpha(1., dtype),  beta(1., dtype);

  //Exhaustive search
  std::vector<int> rv = {4};
  std::vector<int> rl = {1, 2, 4, 8};
  std::vector<int> rs = {1, 2, 4, 8};
  std::vector<int> r1 = {1};
  double best;
  for(sc::param_t x0: rv)
  for(sc::param_t x1: rl)
  for(sc::param_t x2: rl)
  for(sc::param_t x3: rl)
  for(sc::param_t x4: rl)
  for(sc::param_t x5: rl)
  for(sc::param_t x6: rl)
  for(sc::param_t x7: rl)
  for(sc::param_t x8: rs)
  for(sc::param_t x9: rs)
  for(sc::param_t x10: rl)
  for(sc::param_t x11: r1)
  for(sc::param_t x12: r1)
  for(sc::param_t x13: rl)
  for(sc::param_t x14: rl)
  {
    std::vector<sc::param_t> x{x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14};
    sc::templates::Conv generator(dtype, C, H, W, N, K, P, Q, R, S, pad_h, pad_w, stride_h, stride_w, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14]);
    //Compile
    try{
      std::string src = generator.dump(ctx.device(), "conv");
      drv::Module program(ctx, src);
      drv::Kernel kernel(program, "conv");
      double tsec = bench([&](){ generator.enqueue(kernel, stream, alpha, I, F, beta, O); }, [&](){ stream.synchronize(); }, ctx.device());
      double tflops = sc::templates::Conv::tflops(P,Q,K,N,C,R,S,tsec);
      best = std::max(tflops, best);
      std::cout << "//";
      std::copy(x.begin(), x.end(), std::ostream_iterator<int>(std::cout, " "));
      std::cout << ": " << tflops << " TFLOPS [BEST: " << best << "]" << std::endl;
    }catch(isaac::templates::invalid_parameters const &){
      continue;
    }catch(drv::exception::cuda::launch_out_of_resources const &){
      continue;
    }
  }
  std::cout << "ISAAC: " << best << std::endl;
}

void search_gemm(int32_t M, int32_t N, int32_t K, sc::IsaacOperation_t AT, sc::IsaacOperation_t BT, sc::DType dtype){
  auto ctx = drv::backend::contexts::get_default();
  size_t dtsize = sc::size_of(dtype);

  // Setup
  size_t ldc = M;
  size_t lda = (AT==sc::ISAAC_OP_N)?M:K;
  size_t ldb = (BT==sc::ISAAC_OP_N)?K:N;
  int32_t offc = 0, offa = 0, offb = 0;
  drv::Buffer C(ctx, M*N*dtsize);
  drv::Buffer A(ctx, M*K*dtsize);
  drv::Buffer B(ctx, K*N*dtsize);
  drv::Stream stream(ctx);
  sc::scalar alpha(1., dtype), beta(0., dtype);

  // Exhaustive search
  std::vector<int> r1 = {1};
  std::vector<int> rv = {4};
  std::vector<int> rr = {1, 2, 4, 8};
  std::vector<int> rl = {4, 8, 16, 32};
  std::vector<int> rs = {4, 8, 16};
  double best = 0;
  for(sc::param_t x0: rv)
  for(sc::param_t x1: rl)
  for(sc::param_t x2: rl)
  for(sc::param_t x3: rl)
  for(sc::param_t x4: rs)
  for(sc::param_t x5: r1)
  for(sc::param_t x6: rs)
  for(sc::param_t x7: rl)
  for(sc::param_t x8: rl)
  for(sc::param_t x9: rl)
  for(sc::param_t x10: rl)
  for(sc::param_t x11: r1)
  for(sc::param_t x12: rr)
  for(sc::param_t x13: r1)
  {
    std::vector<sc::param_t> x{x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13};
    isaac::templates::GEMM generator(dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]);
    // Compile
    try{
      std::string src = generator.dump(ctx.device(), "gemm");
      drv::Module program(ctx, src);
      drv::Kernel kernel(program, "gemm");
      double time = bench([&](){ generator.enqueue(kernel, stream, alpha, A, B, beta, C); }, [&](){ stream.synchronize(); }, ctx.device());
      double tflops = 2*1e-3*M*N*K/time;
      best = std::max(tflops, best);
      std::cout << "//";
      std::copy(x.begin(), x.end(), std::ostream_iterator<int>(std::cout, " "));
      std::cout << ": " << tflops << " TFLOPS [BEST: " << best << "]" << std::endl;
    }catch(isaac::templates::invalid_parameters const &){
      continue;
    }catch(drv::exception::cuda::launch_out_of_resources const &){
      continue;
    }
  }
  std::cout << "ISAAC: " << best << std::endl;
  // cuBlas
  if(sc::driver::dispatch::cublasinit()){
    char cuAT = (AT==sc::ISAAC_OP_T)?'T':'N';
    char cuBT = (BT==sc::ISAAC_OP_T)?'T':'N';
    double cutime = bench([&](){ sc::driver::cublasGemm(dtype, stream, cuAT, cuBT, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, sc::driver::CUBLAS_PREFER_FASTEST);}
                 , [&](){ stream.synchronize(); }, ctx.device());
    double cutflops = 2*1e-3*M*N*K/cutime;
    std::cout << "cuBLAS: " << cutflops << std::endl;
  }
}

int main(int argc, char *argv[]){
  std::cout << std::fixed << std::setprecision(2);
  opts::Application program("search", "exhaustive auto-tuning for ISAAC");
  program.add<sc::DType>("dtype", "data-type", "float32", {{"float16", sc::HALF_TYPE}, {"float32", sc::FLOAT_TYPE}, {"float64", sc::DOUBLE_TYPE}});
  program.add<std::tuple<size_t, size_t, size_t>>("shape", "tensor shapes to generate the kernel for", std::make_tuple(2048, 2048, 2048));
  program.add<std::string>("layout", "Transposition layout for A and B", "NT", {{"NN", "NN"}, {"NT", "NT"}, {"TN", "TN"}, {"TT", "TT"}});

  program.parse(argc, argv);

  // Data-Type
  sc::DType dtype = program.get<sc::DType>("dtype");
  // Shapes
  size_t M, N, K;
  std::tie(M, N, K) = program.get<std::tuple<size_t, size_t, size_t>>("shape");

  //GEMM
//  std::string layout = program.get<std::string>("layout");
//  sc::IsaacOperation_t AT = layout[0]=='T'?sc::ISAAC_OP_T:sc::ISAAC_OP_N;
//  sc::IsaacOperation_t BT = layout[1]=='T'?sc::ISAAC_OP_T:sc::ISAAC_OP_N;
//  search_gemm(M, N, K, AT, BT, dtype);

  //Search
  search_conv(700, 161, 1, 4, 32, 5, 20, 0, 0, 2, 2, sc::FLOAT_TYPE);
}
