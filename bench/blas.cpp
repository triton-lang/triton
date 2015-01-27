#include "atidlas/array.h"
#include "atidlas/tools/timer.hpp"
#include "common.hpp"
#ifdef BENCH_CLAMDBLAS
  #include "clAmdBlas.h"
#endif
#ifdef BENCH_CBLAS
  #include "cblas.h"
#endif
#ifdef BENCH_CUBLAS
  #include <cublas.h>
#endif
#include <iomanip>
#include <stdlib.h>
#include <cmath>


namespace ad = atidlas;
typedef atidlas::int_t int_t;

template<class T>
void bench(ad::numeric_type dtype)
{
  unsigned int dtsize = ad::size_of(dtype);
  float total_time = 0;
  std::vector<double> times;
  ad::tools::timer timer;

#define BENCHMARK(OP, PERF) \
  {\
  times.clear();\
  total_time = 0;\
  OP;\
  ad::cl::synchronize(ad::cl::default_context());\
  while(total_time < 1e-2){\
    timer.start(); \
    OP;\
    ad::cl::synchronize(ad::cl::default_context());\
    times.push_back(timer.get());\
    total_time += times.back();\
  }\
  float tres = median(times);\
  std::cout << " " << PERF << std::flush;\
  }

  /*---------*/
  /*--BLAS1--*/
  /*---------*/
  std::cout << "#AXPY" << std::endl;
  for(std::vector<int_t>::const_iterator it = BLAS1_N.begin() ; it != BLAS1_N.end() ; ++it)
  {
    int_t N = *it;
    std::cout << N;
    /* ATIDLAS */
    atidlas::array x(N, dtype), y(N, dtype);
    BENCHMARK(y = x + y, bandwidth(3*N, tres, dtsize));
    /* clAmdBlas */
#ifdef BENCH_CLAMDBLAS
    BENCHMARK(clAmdBlasSaxpy(N, 1, x.data()(), 0, 1, y.data()(), 0, 1, 1, &atidlas::cl::get_queue(x.context(), 0)(), 0, NULL, NULL), bandwidth(3*N, tres, dtsize))
#endif
    /* BLAS */
#ifdef BENCH_CBLAS
    std::vector<float> cx(N), cy(N);
    atidlas::copy(x, cx);
    atidlas::copy(y, cy);
    BENCHMARK(cblas_saxpy(N, 1, cx.data(), 1, cy.data(), 1), bandwidth(3*N, tres, dtsize));
#endif
    /* CuBLAS */
#ifdef BENCH_CUBLAS
    T *cux, *cuy;
    cudaMalloc((void**) &cux, N * sizeof(T));
    cudaMalloc((void**) &cuy, N * sizeof(T));
    BENCHMARK(cublasSaxpy(N, 2, x, 1, y, 1), bandwidth(3*N, tres, dtsize))
    cudaFree(cux);
    cudaFree(cuy);
#endif
    std::cout << std::endl;
  }
  std::cout << "\n\n" << std::flush;

  std::cout << "#DOT" << std::endl;
  for(std::vector<int_t>::const_iterator it = BLAS1_N.begin() ; it != BLAS1_N.end() ; ++it)
  {
    int_t N = *it;
    std::cout << N;
    /* ATIDLAS */
    atidlas::array x(N, dtype), y(N, dtype);
    atidlas::array scratch(N, dtype);
    atidlas::scalar s(dtype);
    BENCHMARK(s = dot(x,y), bandwidth(2*N, tres, dtsize));
    /* clAmdBlas */
#ifdef BENCH_CLAMDBLAS
    BENCHMARK(clAmdBlasSdot(N, s.data()(), 0, x.data()(), 0, 1, y.data()(), 0, 1, scratch.data()(), 1, &atidlas::cl::get_queue(x.context(), 0)(), 0, NULL, NULL), bandwidth(2*N, tres, dtsize))
#endif
    /* BLAS */
#ifdef BENCH_CBLAS
    std::vector<float> cx(N), cy(N);
    atidlas::copy(x, cx);
    atidlas::copy(y, cy);
    BENCHMARK(cblas_sdot(N, cx.data(), 1, cy.data(), 1), bandwidth(2*N, tres, dtsize));
#endif
    std::cout << std::endl;
  }
  std::cout << "\n\n" << std::flush;

  /*---------*/
  /*--BLAS2--*/
  /*---------*/
  //T-layout
  std::cout << "#GEMV-T" << std::endl;
  for(std::vector<int>::const_iterator Mit = BLAS2_M.begin() ; Mit != BLAS2_M.end() ; ++Mit)
    for(std::vector<int_t>::const_iterator Nit = BLAS2_N.begin() ; Nit != BLAS2_N.end() ; ++Nit)
    {
      int_t M = *Mit;
      int_t N = *Nit;
      std::cout << M << "," << N;
      /* ATIDLAS */
      atidlas::array A(N, M, dtype), y(M, dtype), x(N, dtype);
      BENCHMARK(y = dot(trans(A),x), bandwidth(M*N + M + N, tres, dtsize));
      /* clAmdBlas */
  #ifdef BENCH_CLAMDBLAS
      BENCHMARK(clAmdBlasSgemv(clAmdBlasColumnMajor, clAmdBlasTrans, N, M, 1, A.data()(), A.ld(), x.data()(), 0, 1, 0, y.data()(), 0, 1, 1, &atidlas::cl::get_queue(x.context(), 0)(),0, NULL, NULL), bandwidth(M*N + M + N, tres, dtsize))
  #endif
      /* BLAS */
  #ifdef BENCH_CBLAS
      std::vector<float> cA(N*M), cx(N), cy(M);
      atidlas::copy(x, cx);
      atidlas::copy(y, cy);
      atidlas::copy(A, cA);
      BENCHMARK(cblas_sgemv(CblasColMajor, CblasTrans, N, M, 1, cA.data(), N, cx.data(), 1, 0, cy.data(), 1), bandwidth(M*N + M + N, tres, dtsize));
  #endif
      std::cout << std::endl;
    }
    std::cout << "\n\n" << std::flush;

//  /*---------*/
//  /*--BLAS3--*/
//  /*---------*/
    std::cout << "#GEMM-NT" << std::endl;
    for(std::vector<int_t>::const_iterator Mit = BLAS3_M.begin() ; Mit != BLAS3_M.end() ; ++Mit)
    for(std::vector<int_t>::const_iterator Nit = BLAS3_N.begin() ; Nit != BLAS3_N.end() ; ++Nit)
    for(std::vector<int_t>::const_iterator Kit = BLAS3_K.begin() ; Kit != BLAS3_K.end() ; ++Kit)
    {
      int_t M = *Mit, N = *Nit, K = *Kit;
      std::cout << M << "," << N << "," << K;
      /* ATIDLAS */
      atidlas::array C(M, N, dtype), A(M, K, dtype), B(N, K, dtype);
      BENCHMARK(C = dot(A,trans(B)), gflops((double)2*M*N*K, tres));
      /* clAmdBlas */
  #ifdef BENCH_CLAMDBLAS
      BENCHMARK(clAmdBlasSgemm(clAmdBlasColumnMajor, clAmdBlasNoTrans, clAmdBlasTrans, M, N, K, 1, A.data()(), A.ld(), B.data()(), B.ld(),
                               0, C.data()(), C.ld(), 1, &atidlas::cl::get_queue(C.context(), 0)(),0, NULL, NULL), gflops((double)2*M*N*K, tres))
  #endif
      /* BLAS */
  #ifdef BENCH_CBLAS
      std::vector<float> cC(M*N), cA(M*K), cB(N*K);
      atidlas::copy(C, cC);
      atidlas::copy(A, cA);
      atidlas::copy(B, cB);
      BENCHMARK(cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, N, K, 1, cA.data(), M, cB.data(), N, 1, cC.data(), M), gflops((double)2*M*N*K, tres));
  #endif
      std::cout << std::endl;
    }

}

int main(int argc, char* argv[])
{
#ifdef BENCH_CLAMDBLAS
  clAmdBlasSetup();
#endif

  int device_idx = 0;
  if(atidlas::cl::queues.size()>1){
    atidlas::cl::queues_t & queues = atidlas::cl::queues;
    if(argc!=2)
    {
      std::cerr << "usage : blas-bench [DEVICE_IDX]" << std::endl;
      std::cout << "Devices available: " << std::endl;
      unsigned int current=0;
      for(atidlas::cl::queues_t::const_iterator it = queues.begin() ; it != queues.end() ; ++it){
        atidlas::cl::Device device = it->first.getInfo<CL_CONTEXT_DEVICES>()[0];
        std::cout << current++ << ": " << device.getInfo<CL_DEVICE_NAME>() << "(" << atidlas::cl::Platform(device.getInfo<CL_DEVICE_PLATFORM>()).getInfo<CL_PLATFORM_NAME>() << ")" << std::endl;
      }
      exit(EXIT_FAILURE);
    }
    else if(argc==2)
      device_idx = atoi(argv[1]);
  }

  atidlas::cl::default_context_idx = device_idx;
  std::cout << "#Benchmark : BLAS" << std::endl;
  std::cout << "#----------------" << std::endl;
  bench<float>(ad::FLOAT_TYPE);

#ifdef BENCH_CLAMDBLAS
  clAmdBlasTeardown();
#endif
}
