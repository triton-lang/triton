#include <cmath>
#include "api.hpp"
#include "isaac/array.h"
#include "clBLAS.h"
#include "cublas.h"

namespace sc = isaac;

template<typename T>
void test_impl(std::string const & ST, simple_matrix_base<T> & cC, simple_matrix_base<T> const & cA, simple_matrix_base<T> const & cB, sc::array_base & C,
          sc::array_base const & A, sc::array_base const & AT,  sc::array_base const & B, sc::array_base const & BT, int& nfail, int& npass)
{
  std::string DT = std::is_same<T, float>::value?"S":"D";
  sc::int_t M = C.shape()[0], N = C.shape()[1], K = A.shape()[1];
  T alpha = 1.43, beta = 0;

  for(int i = 0 ; i < M ; ++i){
    for(int j = 0 ; j < N ; ++j){
      T cij = 0;
      for(int k = 0 ; k < K ; ++k)
        cij += cA(i,k)*cB(k,j);
      cC(i,j) = alpha*cij + beta*cC(i, j);
    }
  }
  std::vector<T> cCbuffer(M*N);
  for(int i = 0 ; i < M ; ++i)
    for(int j = 0 ; j < N ; ++j)
      cCbuffer[i + j*M] = cC(i,j);
  std::vector<T> buffer(M*N);


  sc::driver::CommandQueue queue = sc::driver::backend::queues::get(C.context(),0);
  if(C.context().backend()==sc::driver::OPENCL)
  {
      cl_command_queue clqueue = queue.handle().cl();
     //Row-major
      ADD_TEST_MATMUL(DT+"GEMM-ROW-NN"+ST, BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasRowMajor, clblasNoTrans, clblasNoTrans, N, M, K, alpha, cl(B), off(B), ld(B),
                                                                 cl(A), off(A), ld(A), beta, cl(C), off(C), ld(C), 1, &clqueue, 0, NULL, NULL));
      ADD_TEST_MATMUL(DT+"GEMM-ROW-NT"+ST, BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasRowMajor, clblasTrans, clblasNoTrans, N, M, K, alpha, cl(BT), off(BT), ld(BT),
                                                                 cl(A), off(A), ld(A), beta, cl(C), off(C), ld(C), 1, &clqueue, 0, NULL, NULL));
      ADD_TEST_MATMUL(DT+"GEMM-ROW-TN"+ST, BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasRowMajor, clblasNoTrans, clblasTrans, N, M, K, alpha, cl(B), off(B), ld(B),
                                                                 cl(AT), off(AT), ld(AT), beta, cl(C), off(C), ld(C), 1, &clqueue, 0, NULL, NULL));
      ADD_TEST_MATMUL(DT+"GEMM-ROW-TT"+ST, BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasRowMajor, clblasTrans, clblasTrans, N, M, K, alpha, cl(BT), off(BT), ld(BT),
                                                                 cl(AT), off(AT), ld(AT), beta, cl(C), off(C), ld(C), 1, &clqueue, 0, NULL, NULL));
      //Column-major
      ADD_TEST_MATMUL(DT+"GEMM-COL-NN"+ST, BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasColumnMajor, clblasNoTrans, clblasNoTrans, M, N, K, alpha, cl(A), off(A), ld(A),
                                                                 cl(B), off(B), ld(B), beta, cl(C), off(C), ld(C), 1, &clqueue, 0, NULL, NULL));
      ADD_TEST_MATMUL(DT+"GEMM-COL-NT"+ST, BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasColumnMajor, clblasNoTrans, clblasTrans, M, N, K, alpha, cl(A), off(A), ld(A),
                                                                 cl(BT), off(BT), ld(BT), beta, cl(C), off(C), ld(C), 1, &clqueue, 0, NULL, NULL));
      ADD_TEST_MATMUL(DT+"GEMM-COL-TN"+ST, BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasColumnMajor, clblasTrans, clblasNoTrans, M, N, K, alpha, cl(AT), off(AT), ld(AT),
                                                                 cl(B), off(B), ld(B), beta, cl(C), off(C), ld(C), 1, &clqueue, 0, NULL, NULL));
      ADD_TEST_MATMUL(DT+"GEMM-COL-TT"+ST, BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasColumnMajor, clblasTrans, clblasTrans, M, N, K, alpha, cl(AT), off(AT), ld(AT),
                                                                 cl(BT), off(BT), ld(BT), beta, cl(C), off(C), ld(C), 1, &clqueue, 0, NULL, NULL));

  }

  if(C.context().backend()==sc::driver::CUDA)
  {
      ADD_TEST_MATMUL(DT+"GEMM-NN"+ST, BLAS<T>::F(cublasSgemm,cublasDgemm)('N', 'N', M, N, K, alpha, (T*)cu(A) + off(A), ld(A),
                                                                 (T*)cu(B) + off(B), ld(B), beta, (T*)cu(C) + off(C), ld(C)));
      ADD_TEST_MATMUL(DT+"GEMM-NT"+ST, BLAS<T>::F(cublasSgemm,cublasDgemm)('N', 'T', M, N, K, alpha, (T*)cu(A) + off(A), ld(A),
                                                                 (T*)cu(BT) + off(BT), ld(BT), beta, (T*)cu(C) + off(C), ld(C)));
      ADD_TEST_MATMUL(DT+"GEMM-TN"+ST, BLAS<T>::F(cublasSgemm,cublasDgemm)('T', 'N', M, N, K, alpha, (T*)cu(AT) + off(AT), ld(AT),
                                                                 (T*)cu(B) + off(B), ld(B), beta, (T*)cu(C) + off(C), ld(C)));
      ADD_TEST_MATMUL(DT+"GEMM-TT"+ST, BLAS<T>::F(cublasSgemm,cublasDgemm)('T', 'T', M, N, K, alpha, (T*)cu(AT) + off(AT), ld(AT),
                                                                 (T*)cu(BT) + off(BT), ld(BT), beta, (T*)cu(C) + off(C), ld(C)));
  }
}

template<typename T>
void test(sc::driver::Context const & ctx, int& nfail, int& npass)
{
    sc::int_t M = 173, N = 241, K = 293;
    sc::int_t SUBM = 7, SUBN = 11, SUBK = 29;

    INIT_MATRIX(M, SUBM, 5, 1, N, SUBN, 7, 1, cC, C, ctx);
    INIT_MATRIX(M, SUBM, 8, 1, K, SUBK, 4, 1, cA, A, ctx);
    INIT_MATRIX(K, SUBK, 9, 1, N, SUBN, 6, 1, cB, B, ctx);
    test_impl("FULL", cC, cA, cB, C, A, AT, B, BT, nfail, npass);
    test_impl("SUB", cC_s, cA_s, cB_s, C_s, A_s, AT_s, B_s, BT_s, nfail, npass);
}

int main()
{
  clblasSetup();
  int err = run_test(test<float>, test<double>);
  clblasTeardown();
  return err;
}
