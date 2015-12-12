#include <cmath>
#include "common.hpp"
#include "isaac/array.h"
#include "clBLAS.h"
#include "cublas.h"

namespace sc = isaac;

template<typename T>
void test(T epsilon, simple_matrix_base<T> & cC, simple_matrix_base<T> const & cA, simple_matrix_base<T> const & cB,
                          sc::array_base & C, sc::array_base const & A, sc::array_base const & AT,  sc::array_base const & B, sc::array_base const & BT,
                          interface_t interf, const char * prefix)
{
  int failure_count = 0;

  sc::int_t M = C.shape()[0];
  sc::int_t N = C.shape()[1];
  sc::int_t K = A.shape()[1];

  T alpha = 1.43;
  T beta = 0;

  sc::driver::CommandQueue queue = sc::driver::backend::queues::get(C.context(),0);

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
  sc::copy(C, buffer);\
  if(diff(buffer, cCbuffer, epsilon))\
  {\
    failure_count++;\
    std::cout << " [Failure!]" << std::endl;\
  }\
  else\
    std::cout << std::endl;\

  if(C.context().backend()==sc::driver::OPENCL && interf==clBLAS)
  {
      cl_command_queue clqueue = queue.handle().cl();

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

  if(C.context().backend()==sc::driver::CUDA && interf==cuBLAS)
  {
      RUN_TEST("GEMM-NN", BLAS<T>::F(cublasSgemm,cublasDgemm)('N', 'N', M, N, K, alpha, (T*)CUHANDLE(A) + OFF(A), LD(A),
                                                                 (T*)CUHANDLE(B) + OFF(B), LD(B), beta, (T*)CUHANDLE(C) + OFF(C), LD(C)));
      RUN_TEST("GEMM-NT", BLAS<T>::F(cublasSgemm,cublasDgemm)('N', 'T', M, N, K, alpha, (T*)CUHANDLE(A) + OFF(A), LD(A),
                                                                 (T*)CUHANDLE(BT) + OFF(BT), LD(BT), beta, (T*)CUHANDLE(C) + OFF(C), LD(C)));
      RUN_TEST("GEMM-TN", BLAS<T>::F(cublasSgemm,cublasDgemm)('T', 'N', M, N, K, alpha, (T*)CUHANDLE(AT) + OFF(AT), LD(AT),
                                                                 (T*)CUHANDLE(B) + OFF(B), LD(B), beta, (T*)CUHANDLE(C) + OFF(C), LD(C)));
      RUN_TEST("GEMM-TT", BLAS<T>::F(cublasSgemm,cublasDgemm)('T', 'T', M, N, K, alpha, (T*)CUHANDLE(AT) + OFF(AT), LD(AT),
                                                                 (T*)CUHANDLE(BT) + OFF(BT), LD(BT), beta, (T*)CUHANDLE(C) + OFF(C), LD(C)));
  }

  if(interf==CPP)
  {
      RUN_TEST("C = A * B", C = alpha*dot(A,B) + beta*C)
      RUN_TEST("C = A' * B", C = alpha*dot(AT.T,B) + beta*C)
      RUN_TEST("C = A * B'", C = alpha*dot(A,BT.T) + beta*C)
      RUN_TEST("C = A' * B'", C = alpha*dot(AT.T,BT.T) + beta*C)
  }

  if(failure_count>0)
    exit(EXIT_FAILURE);
}

template<typename T>
void test(T epsilon, sc::driver::Context const & ctx)
{
    int_t M = 173;
    int_t N = 241;
    int_t K = 293;

    int_t SUBM = 7;
    int_t SUBN = 11;
    int_t SUBK = 29;

    {
        INIT_MATRIX(M, SUBM, 5, 1, N, SUBN, 7, 1, cC, C, ctx);
        INIT_MATRIX(M, SUBM, 8, 1, K, SUBK, 4, 1, cA, A, ctx);
        INIT_MATRIX(K, SUBK, 9, 1, N, SUBN, 6, 1, cB, B, ctx);
        test(epsilon, cC, cA, cB, C, A, AT, B, BT, clBLAS, "clBLAS, FULL");
        test(epsilon, cC, cA, cB, C, A, AT, B, BT, cuBLAS, "cuBLAS, FULL");
        test(epsilon, cC_s, cA_s, cB_s, C_s, A_s, AT_s, B_s, BT_s, clBLAS, "clBLAS, SUB");
        test(epsilon, cC_s, cA_s, cB_s, C_s, A_s, AT_s, B_s, BT_s, cuBLAS, "cuBLAS, SUB");
    }

    {
        INIT_MATRIX(M, SUBM, 5, 2, N, SUBN, 7, 3, cC, C, ctx);
        INIT_MATRIX(M, SUBM, 8, 2, K, SUBK, 4, 3, cA, A, ctx);
        INIT_MATRIX(K, SUBK, 9, 4, N, SUBN, 6, 2, cB, B, ctx);
        test(epsilon, cC, cA, cB, C, A, AT, B, BT, CPP, "C++, FULL");
        test(epsilon, cC_s, cA_s, cB_s, C_s, A_s, AT_s, B_s, BT_s, CPP, "C++, SUB");
    }

}

int main()
{
  clblasSetup();
  std::list<isaac::driver::Context const *> data;
  sc::driver::backend::contexts::get(data);
  for(isaac::driver::Context const * context : data)
  {
    sc::driver::Device device = sc::driver::backend::queues::get(*context,0).device();
    if(device.type() != sc::driver::Device::Type::GPU)
        continue;
    std::cout << "Device: " << device.name() << " on " << device.platform().name() << " " << device.platform().version() << std::endl;
    std::cout << "---" << std::endl;
    std::cout << ">> float" << std::endl;
    test<float>(eps_float, *context);
    if(device.fp64_support())
    {
        std::cout << ">> double" << std::endl;
        test<double>(eps_double, *context);
    }
    std::cout << "---" << std::endl;
  }
  clblasTeardown();
  return EXIT_SUCCESS;
}
