#include <sstream>
#include <chrono>
#include <exception>
#include <iomanip>
#include <string>
#include <iostream>
#include <cassert>

#include "isaac/driver/backend.h"
#include "isaac/driver/error.h"
#include "isaac/driver/module.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/stream.h"
#include "isaac/driver/buffer.h"

#include "isaac/driver/cublas.h"
#include "isaac/half.hpp"

#include "isaac/tools/bench.hpp"
#include "isaac/tools/collections.hpp"
#include "isaac/templates/gemm.h"
#include "isaac/templates/error.hpp"

namespace sc = isaac;
namespace drv = isaac::driver;

void do_bench(int32_t M, int32_t N, int32_t K, sc::IsaacOperation_t AT, sc::IsaacOperation_t BT, sc::DType dtype){
  auto ctx = drv::backend::contexts::get_default();
  size_t dtsize = sc::size_of(dtype);

  //Buffers
  int32_t AS0 = M, AS1 = K;
  int32_t BS0 = K, BS1 = N;
  if(AT=='T') std::swap(AS0, AS1);
  if(BT=='T') std::swap(BS0, BS1);
  int32_t ldc = M, lda = AS0, ldb = BS0;
  int32_t offc = 0, offa = 0, offb = 0;
  drv::Buffer C(ctx, M*N*dtsize);
  drv::Buffer A(ctx, M*K*dtsize);
  drv::Buffer B(ctx, K*N*dtsize);
  drv::Stream queue(ctx);
  sc::scalar alpha(1., dtype), beta(0., dtype);

  // cuBlas
  double time = bench([&](){ sc::driver::cublasGemm(dtype, ctx, queue, AT, BT, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);}
               , [&](){ queue.synchronize(); }, ctx.device());
  std::cout <<  2*1e-3*M*N*K/time << std::endl;

  //Exhaustive search
  std::vector<int> r1 = {1};
  std::vector<int> rv = {4};
  std::vector<int> rr = {1, 2, 4};
  std::vector<int> rl = {2, 4, 8, 16, 32};
  std::vector<int> rs = {1, 2, 4, 8, 16};
  double best = 0;
  for(auto x: sc::cpp::cartesian({rv, rl, rl, rl, rs, r1, rs, rl, rl, rl, rl, r1, r1, r1}))
  {
    isaac::templates::GEMM gemm(dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]);
    //Compile
    std::string src;
    try{
      src = gemm.dump(ctx.device(), "gemm");
    }catch(isaac::templates::invalid_parameters){
      continue;
    }
    drv::Module program(ctx, src, true);
    drv::Kernel kernel(program, "gemm");
    //Launch
    double time;
    try{
      time = bench([&](){ gemm.enqueue(kernel, queue, alpha, A, B, beta, C); }, [&](){ queue.synchronize(); }, ctx.device());
    }catch(drv::exception::cuda::launch_out_of_resources){
      continue;
    }
    //Report
    double tflops = 2*1e-3*M*N*K/time;
    best = std::max(tflops, best);
    std::cout << "//" << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << " " << x[4] << " " << x[5] << " " << x[6] << " " << x[7] << " " << x[8] << " " << x[9] << " " << x[10] << " " << x[11] << " " << x[12] << " " << x[13] << " " << std::setprecision(3) << tflops << "  [ " << best << " ] " << std::endl;
  }
}

int main(){
  do_bench(2048, 2048, 2048, sc::ISAAC_OP_N, sc::ISAAC_OP_T, sc::FLOAT_TYPE);
}
