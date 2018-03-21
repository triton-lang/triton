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


template<class IN_DTYPE, class OUT_DTYPE, class UNPACK_DTYPE, char AT, char BT>
void cpp_gemm_impl(int M, int N, int K, OUT_DTYPE* C, int ldc, UNPACK_DTYPE alpha, IN_DTYPE* A, int lda, IN_DTYPE* B, int ldb, UNPACK_DTYPE beta, UNPACK_DTYPE a_scale, UNPACK_DTYPE b_scale, UNPACK_DTYPE c_scale)
{
  static const int PACK_IN = pack_increment<IN_DTYPE>::VALUE;
  static const int PACK_OUT = pack_increment<OUT_DTYPE>::VALUE;
  UNPACK_DTYPE accs[PACK_OUT];
  K /= PACK_IN;
  for(int i = 0; i < M; ++i)
  for(int j = 0; j < N; ++j){
    // Initialize accumulators
    for(int32_t jj = 0; jj < PACK_OUT; ++jj)
      accs[jj] = 0;
    // Accumulate
    for(int32_t jj = 0; jj < PACK_OUT; ++jj)
    for(int k = 0; k < K; ++k){
      IN_DTYPE a = A[matidx<AT>(i, k, lda)];
      IN_DTYPE b = B[matidx<BT>(k, j*PACK_OUT + jj, ldb)];
      accs[jj] = dot(a, b, accs[jj]);
    }
    // Rescale
    for(int32_t jj = 0; jj < PACK_OUT; ++jj){
      accs[jj] /= (a_scale*b_scale);
      accs[jj] =  alpha*accs[jj] + ((beta!=0)?beta*C[i + j*ldc]:0);
    }
    // Write Back
    C[i + j*ldc] = pack<OUT_DTYPE>(accs, c_scale);
  }
}

template<class IN_DTYPE, class OUT_DTYPE, class UNPACK_DTYPE>
void cpp_gemm(int M, int N, int K, OUT_DTYPE* C, int ldc, UNPACK_DTYPE alpha, IN_DTYPE* A, int lda, IN_DTYPE* B, int ldb, UNPACK_DTYPE beta, char AT, char BT, UNPACK_DTYPE a_scale, UNPACK_DTYPE b_scale, UNPACK_DTYPE c_scale)
{
  if(AT=='N' && BT=='N') cpp_gemm_impl<IN_DTYPE, OUT_DTYPE, UNPACK_DTYPE, 'N','N'>(M, N, K, C, ldc, alpha, A, lda, B, ldb, beta, a_scale, b_scale, c_scale);
  if(AT=='T' && BT=='N') cpp_gemm_impl<IN_DTYPE, OUT_DTYPE, UNPACK_DTYPE, 'T','N'>(M, N, K, C, ldc, alpha, A, lda, B, ldb, beta, a_scale, b_scale, c_scale);
  if(AT=='N' && BT=='T') cpp_gemm_impl<IN_DTYPE, OUT_DTYPE, UNPACK_DTYPE, 'N','T'>(M, N, K, C, ldc, alpha, A, lda, B, ldb, beta, a_scale, b_scale, c_scale);
  if(AT=='T' && BT=='T') cpp_gemm_impl<IN_DTYPE, OUT_DTYPE, UNPACK_DTYPE, 'T','T'>(M, N, K, C, ldc, alpha, A, lda, B, ldb, beta, a_scale, b_scale, c_scale);
}

template<class T>
bool abs_cmp(T a, T b)
{ return std::abs(a) < std::abs(b);}

template<class IN_DTYPE, class OUT_DTYPE>
void do_test(sc::driver::Context const & ctx, sc::IsaacOperation_t AT, sc::IsaacOperation_t BT, int32_t M, int32_t N, int32_t K){
  typedef typename unpack_type<OUT_DTYPE>::Type UNPACK_DTYPE;

  sc::DType in_dtype = sc::to_DType<IN_DTYPE>::value;
  sc::DType out_dtype = sc::to_DType<OUT_DTYPE>::value;
  sc::DType ab_dtype = (out_dtype==sc::INT8X4_TYPE)?sc::FLOAT_TYPE:out_dtype;

  static const int PACK_IN = pack_increment<IN_DTYPE>::VALUE;
  static const int PACK_OUT = pack_increment<OUT_DTYPE>::VALUE;

  size_t in_dtsize = sc::size_of(in_dtype);
  size_t out_dtsize = sc::size_of(out_dtype);

  // Strides
  int32_t ldc = M;
  int32_t lda = (AT==sc::ISAAC_OP_N)?M : K / PACK_IN;
  int32_t ldb = (BT==sc::ISAAC_OP_N)?K / PACK_IN : N;
  int32_t offc = 0, offa = 0, offb = 0;

  // Initialize Buffers
  std::vector<OUT_DTYPE> iC(M * N / PACK_OUT);
  std::vector<IN_DTYPE> iA(M * K / PACK_IN);
  std::vector<IN_DTYPE> iB(K * N / PACK_IN);
  std::vector<OUT_DTYPE> rC(M * N / PACK_OUT);

  drv::Buffer C(ctx, iC.size()*out_dtsize);
  drv::Buffer A(ctx, iA.size()*in_dtsize);
  drv::Buffer B(ctx, iB.size()*in_dtsize);

  // Scales
  srand(0);
  for(size_t i = 0; i < iA.size(); ++i) iA[i] = (in_dtype==sc::INT8X4_TYPE)?rand():(float)rand()/RAND_MAX;
  for(size_t i = 0; i < iB.size(); ++i) iB[i] = (in_dtype==sc::INT8X4_TYPE)?rand():(float)rand()/RAND_MAX;
  UNPACK_DTYPE a_scale = (in_dtype==sc::INT8X4_TYPE)?((float)127 / *std::max_element(iA.begin(), iA.end(), abs_cmp<IN_DTYPE>)):1;
  UNPACK_DTYPE b_scale = (in_dtype==sc::INT8X4_TYPE)?((float)127 / *std::max_element(iB.begin(), iB.end(), abs_cmp<IN_DTYPE>)):1;
  UNPACK_DTYPE c_scale = (out_dtype==sc::INT8X4_TYPE)?((float)127 / (0.25*K)):1;
  UNPACK_DTYPE alpha = 1., beta = 0.;
  sc::scalar sc_alpha(alpha, ab_dtype), sc_beta(beta, ab_dtype);


  // Initialize buffers
  drv::Stream stream(ctx);
  stream.write(C, true, 0, iC);
  stream.write(A, true, 0, iA);
  stream.write(B, true, 0, iB);

  //Ground truth
  char cuAT = (AT==sc::ISAAC_OP_T)?'T':'N';
  char cuBT = (BT==sc::ISAAC_OP_T)?'T':'N';
  cpp_gemm<IN_DTYPE, OUT_DTYPE>(M, N, K, rC.data(), ldc, alpha, iA.data(), lda, iB.data(), ldb, beta, cuAT, cuBT, a_scale, b_scale, c_scale);

  //ISAAC results
  std::vector<OUT_DTYPE> hC(M*N);
  sc::GEMM(ctx.device(), stream, in_dtype, out_dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc, sc_alpha, A, B, sc_beta, C, a_scale, b_scale, c_scale);
  stream.read(C, true, 0, hC);
  if(!is_correct(hC, rC, 1e-4))
    exit(EXIT_FAILURE);
  stream.write(C, true, 0, iC);

  std::vector<int> rv = {1, 2, 4};
  std::vector<int> rl = {1, 8};
  std::vector<int> rs = {1, 4};
  std::vector<int> rgrid = {1, 8};
  std::vector<int> r1 = {1};
  for(auto x: sc::cpp::cartesian({rv, rl, rl, rl, rs, r1, rs, rl, rl, rl, rl, rs, rl, rgrid})){
    isaac::templates::GEMM gemm(in_dtype, out_dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc,
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
    gemm.enqueue(kernel, stream, sc_alpha, A, B, sc_beta, C, a_scale, b_scale, c_scale);
    stream.synchronize();

    //Test
    stream.read(C, true, 0, hC);
    stream.write(C, true, 0, iC);
    if(!is_correct(hC, rC, 1e-4))
      exit(EXIT_FAILURE);
  }
}

template<class IN_DTYPE, class OUT_DTYPE>
int do_test(sc::driver::Context const & ctx, std::string const & prefix,
            size_t M, size_t N, size_t K,
            sc::IsaacOperation_t AT, sc::IsaacOperation_t BT)
{

  std::cout << "(" << M << ", " << N << ", " << K << ", " << AT << ", " << BT << ") [" << prefix << "]" << std::endl;
  do_test<IN_DTYPE, OUT_DTYPE>(ctx, AT, BT, M, N, K);
  return EXIT_SUCCESS;
}

int main(){
  auto _N = sc::ISAAC_OP_N;
  auto _T = sc::ISAAC_OP_T;
  auto ctx = drv::backend::contexts::get_default();
  std::cout << "===============" << std::endl;
  std::cout << "GEMM:" << std::endl;
  std::cout << "===============" << std::endl;
  std::cout << "---------------" << std::endl;
  do_test<int, float>(ctx, "int8x4 + dequantize", 67, 83, 640, _N, _N);
  do_test<float, float>(ctx, "core, float", 67, 83, 673, _N, _N);
  do_test<float, float>(ctx, "core, float", 67, 83, 673, _N, _T);
  do_test<float, float>(ctx, "core, float", 67, 83, 673, _T, _N);
  do_test<float, float>(ctx, "core, float", 67, 83, 673, _T, _T);
  do_test<float, float>(ctx, "core, float", 1, 83, 673, _N, _N);
  do_test<float, float>(ctx, "core, float", 1, 83, 673, _N, _T);
  do_test<float, float>(ctx, "core, float", 1, 83, 673, _T, _N);
  do_test<float, float>(ctx, "core, float", 1, 83, 673, _T, _T);
  do_test<float, float>(ctx, "core, float", 67, 1, 673, _N, _N);
  do_test<float, float>(ctx, "core, float", 67, 1, 673, _N, _T);
  do_test<float, float>(ctx, "core, float", 67, 1, 673, _T, _N);
  do_test<float, float>(ctx, "core, float", 67, 1, 673, _T, _T);
  do_test<float, float>(ctx, "core, float", 67, 83, 1, _N, _N);
  do_test<float, float>(ctx, "core, float", 67, 83, 1, _N, _T);
  do_test<float, float>(ctx, "core, float", 67, 83, 1, _T, _N);
  do_test<float, float>(ctx, "core, float", 67, 83, 1, _T, _T);
  do_test<double, double>(ctx, "core, double", 67, 83, 673, _N, _N);
  do_test<double, double>(ctx, "core, double", 67, 83, 673, _N, _T);
  do_test<double, double>(ctx, "core, double", 67, 83, 673, _T, _N);
  do_test<double, double>(ctx, "core, double", 67, 83, 673, _T, _T);
  do_test<float, float>(ctx, "core + vectorized, float", 64, 96, 640, _N, _N);
  do_test<double, double>(ctx, "core + vectorized, double", 64, 96, 640, _N, _N);
  std::cout << "---------------" << std::endl;
}
