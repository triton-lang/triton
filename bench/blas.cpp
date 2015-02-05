#include "atidlas/array.h"
#include "atidlas/symbolic/execute.h"
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
typedef ad::int_t int_t;

template<class T>
void bench(ad::numeric_type dtype)
{
  float total_time = 0;
  std::vector<double> times;
  ad::tools::timer timer;
  unsigned int dtsize = ad::size_of(dtype);

#define BENCHMARK(OP, PERF) \
  {\
  times.clear();\
  total_time = 0;\
  OP;\
  while(total_time < 1e-1){\
    timer.start(); \
    OP;\
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
  for(auto N : BLAS1_N)
  {
    
    std::cout << N;
    /* ATIDLAS */
    ad::array x(N, dtype), y(N, dtype);
    cl::CommandQueue & queue = ad::cl_ext::queues[x.context()][0];
    ad::model & model = ad::get_model(queue, ad::VECTOR_AXPY_TYPE, dtype);
    ad::array_expression E = ad::detail::assign(y, x + y);
    model.tune(E);
    ad::operation_cache cache;
    model.execute(E, &cache);
    queue.flush();
    queue.finish();
    BENCHMARK(cache.enqueue(); queue.flush(); queue.finish();, bandwidth(3*N, tres, dtsize));
    /* clAmdBlas */
#ifdef BENCH_CLAMDBLAS
    BENCHMARK(clAmdBlasSaxpy(N, 1, x.data()(), 0, 1, y.data()(), 0, 1, 1, &queue(), 0, NULL, NULL); queue.flush(); queue.finish();, bandwidth(3*N, tres, dtsize))
#endif
    /* BLAS */
#ifdef BENCH_CBLAS
    std::vector<float> cx(N), cy(N);
    ad::copy(x, cx);
    ad::copy(y, cy);
    BENCHMARK(cblas_saxpy(N, 1, cx.data(), 1, cy.data(), 1), bandwidth(3*N, tres, dtsize));
#endif
    /* CuBLAS */
#ifdef BENCH_CUBLAS
    T *cux, *cuy;
    cudaMalloc((void**) &cux, N * sizeof(T));
    cudaMalloc((void**) &cuy, N * sizeof(T));
    BENCHMARK(cublasSaxpy(N, 2, cux, 1, cuy, 1); cudaThreadSynchronize();, bandwidth(3*N, tres, dtsize))
    cudaFree(cux);
    cudaFree(cuy);
#endif
    std::cout << std::endl;
  }
  std::cout << "\n\n" << std::flush;

//  std::cout << "#DOT" << std::endl;
//  for(std::vector<int_t>::const_iterator it = BLAS1_N.begin() ; it != BLAS1_N.end() ; ++it)
//  {
//    int_t N = *it;
//    std::cout << N;
//    /* ATIDLAS */
//    ad::array x(N, dtype), y(N, dtype);
//    ad::array scratch(N, dtype);
//    ad::scalar s(dtype);
//    CL_BENCHMARK(s = dot(x,y), bandwidth(2*N, tres, dtsize));
//    /* clAmdBlas */
//#ifdef BENCH_CLAMDBLAS
//    CL_BENCHMARK(clAmdBlasSdot(N, s.data()(), 0, x.data()(), 0, 1, y.data()(), 0, 1, scratch.data()(), 1, &ad::cl_ext::get_queue(x.context(), 0)(), 0, NULL, NULL), bandwidth(2*N, tres, dtsize))
//#endif
//    /* BLAS */
//#ifdef BENCH_CBLAS
//    std::vector<float> cx(N), cy(N);
//    ad::copy(x, cx);
//    ad::copy(y, cy);
//    CPU_BENCHMARK(cblas_sdot(N, cx.data(), 1, cy.data(), 1), bandwidth(2*N, tres, dtsize));
//#endif
//    std::cout << std::endl;
//  }
//  std::cout << "\n\n" << std::flush;

//  /*---------*/
//  /*--BLAS2--*/
//  /*---------*/
//  //T-layout
//  std::cout << "#GEMV-T" << std::endl;
//  for(std::vector<int>::const_iterator Mit = BLAS2_M.begin() ; Mit != BLAS2_M.end() ; ++Mit)
//    for(std::vector<int_t>::const_iterator Nit = BLAS2_N.begin() ; Nit != BLAS2_N.end() ; ++Nit)
//    {
//      int_t M = *Mit;
//      int_t N = *Nit;
//      std::cout << M << "," << N;
//      /* ATIDLAS */
//      ad::array A(N, M, dtype), y(M, dtype), x(N, dtype);
//      CL_BENCHMARK(y = dot(trans(A),x), bandwidth(M*N + M + N, tres, dtsize));
//      /* clAmdBlas */
//  #ifdef BENCH_CLAMDBLAS
//      CL_BENCHMARK(clAmdBlasSgemv(clAmdBlasColumnMajor, clAmdBlasTrans, N, M, 1, A.data()(), A.ld(), x.data()(), 0, 1, 0, y.data()(), 0, 1, 1, &ad::cl_ext::get_queue(x.context(), 0)(),0, NULL, NULL), bandwidth(M*N + M + N, tres, dtsize))
//  #endif
//      /* BLAS */
//  #ifdef BENCH_CBLAS
//      std::vector<float> cA(N*M), cx(N), cy(M);
//      ad::copy(x, cx);
//      ad::copy(y, cy);
//      ad::copy(A, cA);
//      CPU_BENCHMARK(cblas_sgemv(CblasColMajor, CblasTrans, N, M, 1, cA.data(), N, cx.data(), 1, 0, cy.data(), 1), bandwidth(M*N + M + N, tres, dtsize));
//  #endif
//      std::cout << std::endl;
//    }
//    std::cout << "\n\n" << std::flush;

////  /*---------*/
////  /*--BLAS3--*/
////  /*---------*/
//    std::cout << "#GEMM-NT" << std::endl;
//    for(std::vector<int_t>::const_iterator Mit = BLAS3_M.begin() ; Mit != BLAS3_M.end() ; ++Mit)
//    for(std::vector<int_t>::const_iterator Nit = BLAS3_N.begin() ; Nit != BLAS3_N.end() ; ++Nit)
//    for(std::vector<int_t>::const_iterator Kit = BLAS3_K.begin() ; Kit != BLAS3_K.end() ; ++Kit)
//    {
//      int_t M = *Kit, N = *Kit, K = *Kit;
//      std::cout << M << "," << N << "," << K;
//      /* ATIDLAS */
//      ad::array C(M, N, dtype), A(M, K, dtype), B(N, K, dtype);
//      CL_BENCHMARK(C = dot(A,trans(B)), gflops((double)2*M*N*K, tres));
//      /* clAmdBlas */
//  #ifdef BENCH_CLAMDBLAS
//      CL_BENCHMARK(clAmdBlasSgemm(clAmdBlasColumnMajor, clAmdBlasNoTrans, clAmdBlasTrans, M, N, K, 1, A.data()(), A.ld(), B.data()(), B.ld(),
//                               0, C.data()(), C.ld(), 1, &ad::cl_ext::get_queue(C.context(), 0)(),0, NULL, NULL), gflops((double)2*M*N*K, tres))
//  #endif
//      /* BLAS */
//  #ifdef BENCH_CBLAS
//      std::vector<float> cC(M*N), cA(M*K), cB(N*K);
//      ad::copy(C, cC);
//      ad::copy(A, cA);
//      ad::copy(B, cB);
//      CPU_BENCHMARK(cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, N, K, 1, cA.data(), M, cB.data(), N, 1, cC.data(), M), gflops((double)2*M*N*K, tres));
//  #endif
//      std::cout << std::endl;
//    }

}

int main(int argc, char* argv[])
{
#ifdef BENCH_CLAMDBLAS
  clAmdBlasSetup();
#endif

  int device_idx = 0;
  ad::cl_ext::queues_type::data_type const & queues = ad::cl_ext::queues.data();

  if(queues.size()>1){
    if(argc!=2)
    {
      std::cerr << "usage : blas-bench [DEVICE_IDX]" << std::endl;
      std::cout << "Devices available: " << std::endl;
      unsigned int current=0;
      for(const auto & queue : queues){
        cl::Device device = queue.first.getInfo<CL_CONTEXT_DEVICES>()[0];
        std::cout << current++ << ": " << device.getInfo<CL_DEVICE_NAME>() << "(" << cl::Platform(device.getInfo<CL_DEVICE_PLATFORM>()).getInfo<CL_PLATFORM_NAME>() << ")" << std::endl;
      }
      exit(EXIT_FAILURE);
    }
    else if(argc==2)
      device_idx = atoi(argv[1]);
  }

  ad::cl_ext::default_context_idx = device_idx;
  std::cout << "#Benchmark : BLAS" << std::endl;
  std::cout << "#----------------" << std::endl;
  bench<float>(ad::FLOAT_TYPE);

#ifdef BENCH_CLAMDBLAS
  clAmdBlasTeardown();
#endif
}
