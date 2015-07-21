#include <cmath>
#include "common.hpp"
#include "isaac/array.h"
#include "isaac/model/model.h"
#include "isaac/wrap/clBLAS.h"

namespace isc = isaac;

template<typename T>
void test_impl(T epsilon, simple_matrix_base<T> & cC, simple_matrix_base<T> const & cA, simple_matrix_base<T> const & cB,
                          isc::array & C, isc::array const & A, isc::array const & AT,  isc::array const & B, isc::array const & BT,
                          interface_t interf, const char * prefix)
{
  int failure_count = 0;

  isc::int_t M = C.shape()[0];
  isc::int_t N = C.shape()[1];
  isc::int_t K = A.shape()[1];

  T alpha = 1;
  T beta = 0;

  isc::driver::CommandQueue queue = isc::driver::queues[C.context()][0];

  for(int i = 0 ; i < M ; ++i)
  {
    for(int j = 0 ; j < N ; ++j)
    {
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

#define RUN_TEST(NAME, GPU_OP)\
  std::cout << "[" << prefix << "] \t" << NAME << "..." << std::flush;\
  GPU_OP;\
  queue.synchronize();\
  isc::copy(C, buffer);\
  if(diff(buffer, cCbuffer, epsilon))\
  {\
    failure_count++;\
    std::cout << " [Failure!]" << std::endl;\
  }\
  else\
    std::cout << std::endl;

  if(interf==clBLAS)
  {
      cl_command_queue clqueue = (*queue.handle().cl)();

      //Row-major
      RUN_TEST("GEMM(ROW, N, N)", BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasRowMajor, clblasNoTrans, clblasNoTrans, N, M, K, alpha, CHANDLE(B), OFF(B), LD(B),
                                                                 CHANDLE(A), OFF(A), LD(A), beta, CHANDLE(C), OFF(C), LD(C), 1, &clqueue, 0, NULL, NULL));
      RUN_TEST("GEMM(ROW, N, T)", BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasRowMajor, clblasTrans, clblasNoTrans, N, M, K, alpha, CHANDLE(BT), OFF(BT), LD(BT),
                                                                 CHANDLE(A), OFF(A), LD(A), beta, CHANDLE(C), OFF(C), LD(C), 1, &clqueue, 0, NULL, NULL));
      RUN_TEST("GEMM(ROW, T, N)", BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasRowMajor, clblasNoTrans, clblasTrans, N, M, K, alpha, CHANDLE(B), OFF(B), LD(B),
                                                                 CHANDLE(AT), OFF(AT), LD(AT), beta, CHANDLE(C), OFF(C), LD(C), 1, &clqueue, 0, NULL, NULL));
      RUN_TEST("GEMM(ROW, T, T)", BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasRowMajor, clblasTrans, clblasTrans, N, M, K, alpha, CHANDLE(BT), OFF(BT), LD(BT),
                                                                 CHANDLE(AT), OFF(AT), LD(AT), beta, CHANDLE(C), OFF(C), LD(C), 1, &clqueue, 0, NULL, NULL));

      //Column-major
      RUN_TEST("GEMM(COL, N, N)", BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasColumnMajor, clblasNoTrans, clblasNoTrans, M, N, K, alpha, CHANDLE(A), OFF(A), LD(A),
                                                                 CHANDLE(B), OFF(B), LD(B), beta, CHANDLE(C), OFF(C), LD(C), 1, &clqueue, 0, NULL, NULL));

      RUN_TEST("GEMM(COL, N, T)", BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasColumnMajor, clblasNoTrans, clblasTrans, M, N, K, alpha, CHANDLE(A), OFF(A), LD(A),
                                                                 CHANDLE(BT), OFF(BT), LD(BT), beta, CHANDLE(C), OFF(C), LD(C), 1, &clqueue, 0, NULL, NULL));

      RUN_TEST("GEMM(COL, T, N)", BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasColumnMajor, clblasTrans, clblasNoTrans, M, N, K, alpha, CHANDLE(AT), OFF(AT), LD(AT),
                                                                 CHANDLE(B), OFF(B), LD(B), beta, CHANDLE(C), OFF(C), LD(C), 1, &clqueue, 0, NULL, NULL));
      RUN_TEST("GEMM(COL, T, T)", BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasColumnMajor, clblasTrans, clblasTrans, M, N, K, alpha, CHANDLE(AT), OFF(AT), LD(AT),
                                                                 CHANDLE(BT), OFF(BT), LD(BT), beta, CHANDLE(C), OFF(C), LD(C), 1, &clqueue, 0, NULL, NULL));



  }
  else
  {
      RUN_TEST("C = A * B", C = dot(A,B))
      RUN_TEST("C = A' * B", C = dot(trans(AT),B))
      RUN_TEST("C = A * B'", C = dot(A,trans(BT)))
      RUN_TEST("C = A' * B'", C = dot(trans(AT),trans(BT)))
  }

  if(failure_count>0)
    exit(EXIT_FAILURE);
}

template<typename T>
void test_impl(T epsilon, isc::driver::Context const & ctx)
{
    int_t M = 173;
    int_t N = 241;
    int_t K = 293;

    int_t SUBM = 7;
    int_t SUBN = 13;
    int_t SUBK = 41;

    {
        INIT_MATRIX(M, SUBM, 5, 1, N, SUBN, 7, 1, cC, C, ctx);
        INIT_MATRIX(M, SUBM, 8, 1, K, SUBK, 4, 1, cA, A, ctx);
        INIT_MATRIX(K, SUBK, 9, 1, N, SUBN, 6, 1, cB, B, ctx);
        test_impl(epsilon, cC_full, cA_full, cB_full, C_full, A_full, AT_full, B_full, BT_full, clBLAS, "BLAS, FULL");
        test_impl(epsilon, cC_slice, cA_slice, cB_slice, C_slice, A_slice, AT_slice, B_slice, BT_slice, clBLAS, "BLAS, SUB");
    }

    {
        INIT_MATRIX(M, SUBM, 5, 2, N, SUBN, 7, 3, cC, C, ctx);
        INIT_MATRIX(M, SUBM, 8, 2, K, SUBK, 4, 3, cA, A, ctx);
        INIT_MATRIX(K, SUBK, 9, 4, N, SUBN, 6, 2, cB, B, ctx);
        test_impl(epsilon, cC_full, cA_full, cB_full, C_full, A_full, AT_full, B_full, BT_full, CPP, "C++, FULL");
        test_impl(epsilon, cC_slice, cA_slice, cB_slice, C_slice, A_slice, AT_slice, B_slice, BT_slice, CPP, "C++, SUB");
    }

}

int main()
{
  clblasSetup();
  auto data = isc::driver::queues.contexts();
  for(const auto & elem : data)
  {
    isc::driver::Device device = elem.second[0].device();
    std::cout << "Device: " << device.name() << " on " << device.platform().name() << " " << device.platform().version() << std::endl;
    std::cout << "---" << std::endl;
    std::cout << ">> float" << std::endl;
    test_impl<float>(1e-4, elem.first);
    std::cout << ">> double" << std::endl;
    test_impl<double>(1e-9, elem.first);
    std::cout << "---" << std::endl;
  }
  clblasTeardown();
  return EXIT_SUCCESS;
}
