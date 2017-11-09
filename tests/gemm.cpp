#include <sstream>
#include <chrono>
#include <exception>
#include <iomanip>
#include <string>
#include <iostream>
#include <cassert>
#include <cmath>
#include <cfenv>

#include "isaac/driver/backend.h"
#include "isaac/driver/cublas.h"
#include "isaac/driver/error.h"
#include "isaac/driver/module.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/stream.h"
#include "isaac/driver/buffer.h"

#include "isaac/templates/gemm.h"
#include "isaac/templates/error.hpp"
#include "isaac/tools/collections.hpp"

#include "isaac/api.h"

#include "test_utils.hpp"

namespace sc = isaac;
namespace drv = isaac::driver;

template<class DTYPE>
void do_test(sc::driver::Context const & ctx, sc::IsaacOperation_t AT, sc::IsaacOperation_t BT, int32_t M, int32_t N, int32_t K){
  sc::DType dtype = sc::to_DType<DTYPE>::value;
  size_t dtsize = sc::size_of(dtype);

  //Shapes
  int32_t AS0 = M, AS1 = K;
  int32_t BS0 = K, BS1 = N;
  if(AT==sc::ISAAC_OP_T) std::swap(AS0, AS1);
  if(BT==sc::ISAAC_OP_T) std::swap(BS0, BS1);
  int32_t ldc = M, lda = AS0, ldb = BS0;
  int32_t offc = 0, offa = 0, offb = 0;
  sc::scalar alpha(1., dtype), beta(3.2, dtype);

  //Initialize Buffers
  drv::Buffer C(ctx, M*N*dtsize);
  drv::Buffer A(ctx, M*K*dtsize);
  drv::Buffer B(ctx, K*N*dtsize);
  std::vector<DTYPE> iC(M*N);
  std::vector<DTYPE> iA(M*K);
  std::vector<DTYPE> iB(K*N);
  srand(0);
  for(size_t i = 0; i < iA.size(); ++i) iA[i] = (float)rand()/RAND_MAX;
  for(size_t i = 0; i < iB.size(); ++i) iB[i] = (float)rand()/RAND_MAX;

  drv::Stream stream(ctx);
  stream.write(C, true, 0, M*N*dtsize, iC.data());
  stream.write(A, true, 0, M*K*dtsize, iA.data());
  stream.write(B, true, 0, K*N*dtsize, iB.data());

  //Ground result (cuBLAS)
  char cuAT = (AT==sc::ISAAC_OP_T)?'T':'N';
  char cuBT = (BT==sc::ISAAC_OP_T)?'T':'N';
  sc::driver::cublasGemm(dtype, stream, cuAT, cuBT, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  std::vector<DTYPE> rC(M*N);
  stream.read(C, true, 0, M*N*dtsize, (void*)rC.data());
  stream.write(C, true, 0, M*N*dtsize, iC.data());

  //ISAAC result
  std::vector<DTYPE> hC(M*N);

//  //Test selected profile
//  sc::GEMM(ctx.device(), stream, dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc, alpha, A, B, beta, C);
//  stream.read(C, true, 0, M*N*dtsize, (void*)hC.data());
//  if(!is_correct(hC, rC, max_rounding_error(DTYPE(K))))
//    exit(EXIT_FAILURE);
//  stream.write(C, true, 0, M*N*dtsize, iC.data());

  std::vector<int> rv = {1, 2, 4};
  std::vector<int> rl = {1, 8};
  std::vector<int> rs = {1, 4};
  std::vector<int> rgrid = {1, 8};
  std::vector<int> r1 = {1};
  for(auto x: sc::cpp::cartesian({rv, rl, rl, rl, rs, r1, rs, rl, rl, rl, rl, rs, rl, rgrid})){
    isaac::templates::GEMM gemm(dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc,
                                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]);
    //Compile
    std::string src;
    try{
      src = gemm.dump(ctx.device(), "gemm");
    }catch(isaac::templates::invalid_parameters){
      continue;
    }
    drv::Module program(ctx, src);
    drv::Kernel kernel(program, "gemm");

    //Launch
    gemm.enqueue(kernel, stream, alpha, A, B, beta, C);
    stream.synchronize();

    //Test
    stream.read(C, true, 0, M*N*dtsize, (void*)hC.data());
    stream.write(C, true, 0, M*N*dtsize, iC.data());
    size_t depth = x[11]*x[12]*x[13];
    double eps = max_rounding_error(DTYPE(K/depth))*depth;
    if(!is_correct(hC, rC, eps))
      exit(EXIT_FAILURE);
  }
}

template<class DTYPE>
int do_test(sc::driver::Context const & ctx, size_t M, size_t N, size_t K){
  auto _N = sc::ISAAC_OP_N;
  auto _T = sc::ISAAC_OP_T;
  std::cout << "Testing (" << M << ", " << N << ", " << K << ") ..." << std::endl;
  std::cout << "Layout : NN ..." << std::endl;
  do_test<DTYPE>(ctx, _N, _N, M, N, K);
  std::cout << "Layout : NT ..." << std::endl;
  do_test<DTYPE>(ctx, _N, _T, M, N, K);
  std::cout << "Layout : TN ..." << std::endl;
  do_test<DTYPE>(ctx, _T, _N, M, N, K);
  std::cout << "Layout : TT ..." << std::endl;
  do_test<DTYPE>(ctx, _T, _T, M, N, K);
  std::cout << "---------------" << std::endl;

  return EXIT_SUCCESS;
}

int main(){
  auto ctx = drv::backend::contexts::get_default();
//  if(ctx.device().compute_capability().first>=6)
//  {
//    std::cout << "===============" << std::endl;
//    std::cout << "HALF:" << std::endl;
//    std::cout << "===============" << std::endl;
//    do_test<half_float::half>(ctx, 67, 83, 673);
//    do_test<half_float::half>(ctx, 1,83,673);
//    do_test<half_float::half>(ctx, 67,1,673);
//    do_test<half_float::half>(ctx, 67, 83, 1);
//    do_test<half_float::half>(ctx, 64, 96, 640);
//  }
  std::cout << "===============" << std::endl;
  std::cout << "FLOAT:" << std::endl;
  std::cout << "===============" << std::endl;
  do_test<float>(ctx, 67, 83, 673);
  do_test<float>(ctx, 1, 83, 673);
  do_test<float>(ctx, 67, 1, 673);
  do_test<float>(ctx, 67, 83, 1);
  do_test<float>(ctx, 64, 96, 640);
  std::cout << "===============" << std::endl;
  std::cout << "DOUBLE:" << std::endl;
  std::cout << "===============" << std::endl;
  do_test<double>(ctx, 67, 83, 673);
  do_test<double>(ctx, 1, 83, 673);
  do_test<double>(ctx, 67, 1, 673);
  do_test<double>(ctx, 67, 83, 1);
  do_test<double>(ctx, 64, 96, 640);

}
