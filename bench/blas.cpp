#include "isaac/array.h"
#include "isaac/symbolic/execute.h"
#include "isaac/tools/timer.hpp"
#ifdef BENCH_CLBLAS
  #include "clBLAS.h"
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
#include <numeric>
#include <regex>

namespace ad = isaac;
typedef ad::int_t int_t;

int ceil(int N, int pad)
{
    return (N%pad==0)?N:(N+pad-1)/pad*pad;
}

std::vector<int> create_log_range(int min, int max, int N, int pad)
{
  std::vector<int> res(N);
  for(int i = 0 ; i < N ; ++i)
  {
    res[i] = std::exp(std::log(min) + (float)(std::log(max) - std::log(min))*i/N);
    res[i] = ceil(res[i], pad);
  }
  return res;
}

std::vector<int> create_full_range(int min, int max, int pad)
{
    std::vector<int> N;
    for(int i = ceil(min, pad) ; i < ceil(max, pad) ; i+=pad)
        N.push_back(i);
    return N;
}


template<class T>
T median(std::vector<T> x)
{
  size_t size = x.size();
  std::sort(x.begin(), x.end());
  if (size  % 2 == 0)
      return (x[size / 2 - 1] + x[size / 2]) / 2;
  else
      return x[size / 2];
}

template<class T>
T mean(std::vector<T> x)
{
  T res = 0;
  int N = x.size();
  for(int i = 0 ; i < N ; ++i)
    res += x[i];
  return res/N;
}

static double time_event(unsigned long sum, ad::driver::Event const & e)
{ return sum + e.elapsed_time();}

template<class T>
void bench(ad::numeric_type dtype, std::string operation)
{

//
// MACROS FOR BENCHMARKING
//
#define CL_HANDLE(X) (*X.handle().cl)()
#define BENCHMARK_ISAAC(OP, PERF) \
  {\
  std::vector<long> times;\
  double total_time = 0;\
  while(total_time*1e-9 < 1e-3){\
    std::list<ad::driver::Event> events;\
    flush = ad::zeros(1e6, 1, dtype);\
    OP;\
    queue.synchronize();\
    times.push_back(std::accumulate(events.begin(), events.end(), 0, &time_event));\
    total_time+=times.back();\
  }\
  double t = median(times);\
  std::cout << " " << PERF << std::flush;\
  }

#define BENCHMARK_CLBLAS(OP, PERF) \
  {\
  std::vector<long> times;\
  double total_time = 0;\
  while(total_time*1e-9 < 1e-3){\
    cl::Event event;\
    OP;\
    queue.synchronize();\
    times.push_back(event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());\
    total_time+=times.back();\
  }\
  double t = median(times);\
  std::cout << " " << PERF << std::flush;\
  }

#define BENCHMARK_HOST(OP, PERF) \
  {\
  ad::tools::timer tmr;\
  std::vector<int> cache_flusher(10000000, 0);\
  tmr.start();\
  OP;\
  double t = 1e9*tmr.get();\
  std::cout << " " << PERF << std::flush;\
  }

#define BENCHMARK_CUDA(OP, PERF) \
  {\
  std::vector<long> times;\
  double total_time = 0;\
  float time;\
  cudaEvent_t start, stop;\
  cudaEventCreate(&start);\
  cudaEventCreate(&stop);\
  while(total_time*1e-3 < 1e-3){\
    flush = ad::zeros(1e6, 1, dtype);\
    cudaEventRecord(start,0);\
    OP;\
    cudaEventRecord(stop,0);\
    cudaEventSynchronize(stop);\
    cudaEventElapsedTime(&time, start, stop);\
    times.push_back(time*1e6);\
    total_time+=time;\
  }\
  double t = median(times);\
  std::cout << "\t" << PERF << std::flush;\
  }

  unsigned int dtsize = ad::size_of(dtype);
  ad::driver::CommandQueue & queue = ad::driver::queues.default_queues()[0];
  std::map<std::string, std::string> metric{ {"axpy", "GB/s"}, {"dot", "GB/s"}, {"gemv", "GB/s"}, {"gemm", "GFLOPS"}};
  ad::array flush(1e6, dtype);
  std::cout << "#" << operation << " (" << metric[operation] << ")" << std::endl;
  std::cout << "N";
  std::cout << "\tISAAC (predictive)";
  std::cout << "\tISAAC (optimal)";
#ifdef BENCH_CLBLAS
  std::cout << "\tclBLAS";
#endif
#ifdef BENCH_CBLAS
  std::cout << "\tBLAS";
#endif
#ifdef BENCH_CUBLAS
  std::cout << "\tcuBLAS";
#endif
  std::cout << std::endl;
  //
  // RUN BENCHMARKS
  //

//  /*---------*/
//  /*--BLAS1--*/
//  /*---------*/

  if(operation=="axpy")
  {
    for(int_t N: create_log_range(1e3, 2e7, 50, 64))
    {
      std::cout << N;
      ad::array x(N, dtype), y(N, dtype);
      /* ISAAC */
      std::list<ad::driver::Event> events;\
      BENCHMARK_ISAAC(y = ad::control(x + y, ad::execution_options_type(0, &events), ad::dispatcher_options_type(false)), 3*N*dtsize/t)
      BENCHMARK_ISAAC(y = ad::control(x + y, ad::execution_options_type(0, &events), ad::dispatcher_options_type(true)), 3*N*dtsize/t)
      /* clblas */
  #ifdef BENCH_CLBLAS
      BENCHMARK_CLBLAS(clblasSaxpy(N, 1, CL_HANDLE(x.data()), 0, 1, CL_HANDLE(y.data()), 0, 1, 1, &CL_HANDLE(queue), 0, NULL, &event()), 3*N*dtsize/t)
  #endif
      /* BLAS */
  #ifdef BENCH_CBLAS
      std::vector<float> cx(N), cy(N);
      ad::copy(x, cx);
      ad::copy(y, cy);
      BENCHMARK_HOST(cblas_saxpy(N, 1, cx.data(), 1, cy.data(), 1), 3*N*dtsize/t);
  #endif
      /* CuBLAS */
  #ifdef BENCH_CUBLAS
      T *cux, *cuy;
      cudaMalloc((void**) &cux, N * sizeof(T));
      cudaMalloc((void**) &cuy, N * sizeof(T));
      BENCHMARK_CUDA(cublasSaxpy(N, 2, cux, 1, cuy, 1), 3*N*dtsize/t)
      cudaFree(cux);
      cudaFree(cuy);
  #endif
      std::cout << std::endl;
    }
  }

  if(operation=="dot")
  {
    for(int_t N: create_log_range(1e3, 2e7, 50, 64))
    {
      std::cout << N;
      /* ISAAC */
      ad::array x(N, dtype), y(N, dtype);
      ad::array scratch(N, dtype);
      ad::scalar s(dtype);
      s = dot(x,y); queue.synchronize();
      BENCHMARK_ISAAC(s = ad::control(dot(x,y), ad::execution_options_type(0, &events), ad::dispatcher_options_type(true)), 2*N*dtsize/t)
      /* clblas */
  #ifdef BENCH_CLBLAS
      BENCHMARK_CLBLAS(clblasSdot(N, CL_HANDLE(s.data()), 0, CL_HANDLE(x.data()), 0, 1, CL_HANDLE(y.data()), 0, 1, CL_HANDLE(scratch.data()), 1, &CL_HANDLE(queue), 0, NULL, &event()), 2*N*dtsize/t)
  #endif
      /* BLAS */
  #ifdef BENCH_CBLAS
      std::vector<float> cx(N), cy(N);
      ad::copy(x, cx);
      ad::copy(y, cy);
      BENCHMARK_HOST(cblas_sdot(N, cx.data(), 1, cy.data(), 1), 2*N*dtsize/t);
  #endif
  #ifdef BENCH_CUBLAS
      T *cux, *cuy;
      T result;
      cudaMalloc((void**) &cux, N * sizeof(T));
      cudaMalloc((void**) &cuy, N * sizeof(T));
      BENCHMARK_CUDA(cublasSdot(N, cux, 1, cuy, 1), 2*N*dtsize/t)
      cudaFree(cux);
      cudaFree(cuy);
  #endif
      std::cout << std::endl;
    }
    std::cout << "\n\n" << std::flush;
  }

  if(operation.substr(0, 4)=="gemv")
  {
    std::vector<std::tuple<int_t, int_t> > MNs;
    MNs.push_back(std::make_tuple(896,896));
    MNs.push_back(std::make_tuple(3072,3072));
    MNs.push_back(std::make_tuple(64,32000));
    MNs.push_back(std::make_tuple(896,32000));
    MNs.push_back(std::make_tuple(32000, 64));
    MNs.push_back(std::make_tuple(32000, 896));

    /*---------*/
    /*--BLAS2--*/
    /*---------*/
    //T-layout
    for(std::tuple<int_t, int_t> MN: MNs)
    {
        int_t M = std::get<0>(MN);
        int_t N = std::get<1>(MN);
        std::cout << M << "," << N;
        /* ISAAC */
        ad::array A(N, M, dtype), y(M, dtype), x(N, dtype);
        int_t lda = A.ld();
        y = dot(trans(A),x); queue.synchronize();
        BENCHMARK_ISAAC(y = ad::control(dot(trans(A),x), ad::execution_options_type(0, &events), ad::dispatcher_options_type(false)),(M*N + M + N)*dtsize/t);
        BENCHMARK_ISAAC(y = ad::control(dot(trans(A),x), ad::execution_options_type(0, &events), ad::dispatcher_options_type(true)),(M*N + M + N)*dtsize/t);
    #ifdef BENCH_CLBLAS
        BENCHMARK_CLBLAS(clblasSgemv(clblasColumnMajor, clblasTrans, N, M, 1, CL_HANDLE(A.data()), 0, lda, CL_HANDLE(x.data()), 0, 1, 0, CL_HANDLE(y.data()), 0, 1, 1, &CL_HANDLE(queue),0, NULL, &event()), (M*N + M + N)*dtsize/t)
    #endif
    #ifdef BENCH_CBLAS
        std::vector<float> cA(N*M), cx(N), cy(M);
        ad::copy(x, cx);
        ad::copy(y, cy);
        ad::copy(A, cA);
        BENCHMARK_HOST(cblas_sgemv(CblasColMajor, CblasTrans, N, M, 1, cA.data(), lda, cx.data(), 1, 0, cy.data(), 1), (M*N + M + N)*dtsize/t);
    #endif
    #ifdef BENCH_CUBLAS
        T *cuA, *cux, *cuy;
        cudaMalloc((void**) &cuA, N * M * sizeof(T));
        cudaMalloc((void**) &cux, N * sizeof(T));
        cudaMalloc((void**) &cuy, M * sizeof(T));
        BENCHMARK_CUDA(cublasSgemv('t', N, M, 1, cuA, lda, cux, 1, 0, cuy, 1), (M*N + M + N)*dtsize/t)
        cudaFree(cuA);
        cudaFree(cux);
        cudaFree(cuy);
    #endif
        std::cout << std::endl;
      }
      std::cout << "\n\n" << std::flush;
  }

  if(operation.substr(0,4)=="gemm")
  {
    std::vector<std::tuple<int_t, int_t, int_t> > MNKs;
    MNKs.push_back(std::make_tuple(896,896,896));
    MNKs.push_back(std::make_tuple(3072,3072,3072));
    MNKs.push_back(std::make_tuple(1024,64,768));
    MNKs.push_back(std::make_tuple(768,64,128));
    MNKs.push_back(std::make_tuple(64,64,32000));
    MNKs.push_back(std::make_tuple(1024,1024,32000));

    /*---------*/
    /*--BLAS3--*/
    /*---------*/
    for(std::tuple<int_t, int_t, int_t> MNK: MNKs)
    {
        int_t M = std::get<0>(MNK);
        int_t N = std::get<1>(MNK);
        int_t K = std::get<2>(MNK);
        std::cout << M << "," << N << "," << K;
        /* ISAAC */
        ad::array C(M, N, dtype), A(M, K, dtype), B(N, K, dtype);
        int_t lda = A.ld(), ldb = B.ld(), ldc = C.ld();
        BENCHMARK_ISAAC(C = ad::control(dot(A,trans(B)), ad::execution_options_type(0, &events), ad::dispatcher_options_type(false)), (double)2*M*N*K/t);
        //BENCHMARK_ISAAC(C = ad::control(dot(A,trans(B)), ad::execution_options_type(0, &events), ad::dispatcher_options_type(true)), (double)2*M*N*K/t);
        /* clblas */
    #ifdef BENCH_CLBLAS
        BENCHMARK_CLBLAS(clblasSgemm(clblasColumnMajor, clblasNoTrans, clblasTrans, M, N, K, 1, CL_HANDLE(A.data()), 0, lda, CL_HANDLE(B.data()), 0, ldb,
                                            0, CL_HANDLE(C.data()), 0, ldc, 1, &CL_HANDLE(queue),0, NULL, &event()), (double)2*M*N*K/t)
    #endif
        /* BLAS */
    #ifdef BENCH_CBLAS
        std::vector<float> cC(M*N), cA(M*K), cB(N*K);
        ad::copy(C, cC);
        ad::copy(A, cA);
        ad::copy(B, cB);
        BENCHMARK_HOST(cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, N, K, 1, cA.data(), lda, cB.data(), ldb, 1, cC.data(), ldc), (double)2*M*N*K/t);
    #endif
    #ifdef BENCH_CUBLAS
        T *cuA, *cuB, *cuC;
        cudaMalloc((void**) &cuA, M * K * sizeof(T));
        cudaMalloc((void**) &cuB, K * N * sizeof(T));
        cudaMalloc((void**) &cuC, M * N * sizeof(T));
        BENCHMARK_CUDA(cublasSgemm('n', 't', M, N, K, 1, cuA, lda, cuB, ldb, 1, cuC, ldc), (double)2*M*N*K/t)
        cudaFree(cuA);
        cudaFree(cuB);
        cudaFree(cuC);
    #endif
        std::cout << std::endl;
      }
  }

}

int main(int argc, char* argv[])
{
  std::vector<std::string> args(argv, argv + argc);
#ifdef BENCH_CLBLAS
  clblasSetup();
#endif
  ad::driver::queues.queue_properties = CL_QUEUE_PROFILING_ENABLE;

  int device_idx = 0;
  ad::driver::queues_type::container_type queues = ad::driver::queues.contexts();

  std::string operation;
  if(queues.size() > 1)
  {
    if(args.size() != 3)
    {
      std::cerr << "usage : blas-bench DEVICE_IDX OPERATION" << std::endl;
      std::cout << "Devices available: " << std::endl;
      unsigned int current=0;
      for(ad::driver::queues_type::container_type::const_iterator it = queues.begin() ; it != queues.end() ; ++it)
      {
        ad::driver::Device device = it->first.device();
        std::cout << current++ << ": " << device.name() << " on " << device.platform().name() << " " << device.platform().version() << std::endl;
      }
      exit(EXIT_FAILURE);
    }
    device_idx = atoi(argv[1]);
    operation = args[2];
  }
  else
  {
    if(args.size() != 2)
    {
      std::cerr << "usage : blas-bench OPERATION" << std::endl;
      exit(EXIT_FAILURE);
    }
    operation = args[1];
  }

  ad::driver::queues.default_device = device_idx;
  std::cout << "#Benchmark : BLAS" << std::endl;
  std::cout << "#----------------" << std::endl;
  bench<float>(ad::FLOAT_TYPE, operation);

#ifdef BENCH_CLBLAS
  clblasTeardown();
#endif
}
