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
#include <numeric>

namespace ad = atidlas;
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

template <typename T>
class make_vector {
public:
  typedef make_vector<T> my_type;
  my_type& operator<< (const T& val) {
    data_.push_back(val);
    return *this;
  }
  operator std::vector<T>() const {
    return data_;
  }
private:
  std::vector<T> data_;
};




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

static double time_event(unsigned long sum, cl::Event const & e)
{ return sum + e.getProfilingInfo<CL_PROFILING_COMMAND_END>() -  e.getProfilingInfo<CL_PROFILING_COMMAND_START>();}

template<class T>
void bench(ad::numeric_type dtype){

#define BENCHMARK_ATIDLAS(OP, PERF) \
  {\
  std::vector<long> times;\
  double total_time = 0;\
  while(total_time*1e-9 < 1e-2){\
    std::list<cl::Event> events;\
    OP;\
    queue.finish();\
    times.push_back(std::accumulate(events.begin(), events.end(), 0, &time_event));\
    total_time+=times.back();\
  }\
  double t = median(times);\
  std::cout << " " << PERF << std::flush;\
  }

#define BENCHMARK_CLAMDBLAS(OP, PERF) \
  {\
  std::vector<long> times;\
  double total_time = 0;\
  while(total_time*1e-9 < 1e-2){\
    cl::Event event;\
    OP;\
    queue.finish();\
    times.push_back(event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -  event.getProfilingInfo<CL_PROFILING_COMMAND_START>());\
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
  while(total_time*1e-3 < 1e-1){\
    cudaEventRecord(start,0);\
    OP;\
    cudaEventRecord(stop,0);\
    cudaEventSynchronize(stop);\
    cudaEventElapsedTime(&time, start, stop);\
    times.push_back(time*1e6);\
    total_time+=time;\
  }\
  double t = median(times);\
  std::cout << " " << PERF << std::flush;\
  }

unsigned int dtsize = ad::size_of(dtype);
cl::CommandQueue & queue = ad::cl_ext::queues[ad::cl_ext::default_context()][0];

  // BLAS1 Sizes
  static const std::vector<int> BLAS1_N = create_log_range(1e3, 2e7, 50, 64);

  // BLAS2 Sizes
  static const std::vector<int> BLAS2_N = make_vector<int>() << 64;
  static const std::vector<int> BLAS2_M = create_full_range(128, 10000, 64);

  // BLAS3 Sizes
  static const std::vector<int> BLAS3_M = make_vector<int>() << 1024;
  static const std::vector<int> BLAS3_N = make_vector<int>() << 128;
  static const std::vector<int> BLAS3_K = create_full_range(128, 5000, 64);

//  /*---------*/
//  /*--BLAS1--*/
//  /*---------*/
//  std::cout << "#AXPY" << std::endl;
//  for(int_t N : create_log_range(1e3, 2e7, 50, 64))
//  {
//    std::cout << N;
//    ad::array x(N, dtype), y(N, dtype);
//    /* ATIDLAS */
//    y = x + y; queue.finish();
//    BENCHMARK_ATIDLAS(y = ad::control(x + y, ad::execution_options_type(0, &events), ad::dispatcher_options_type(true)), 3*N*dtsize/t)
//    /* clAmdBlas */
//#ifdef BENCH_CLAMDBLAS
//    BENCHMARK_CLAMDBLAS(clAmdBlasSaxpy(N, 1, x.data()(), 0, 1, y.data()(), 0, 1, 1, &queue(), 0, NULL, &event()), 3*N*dtsize/t)
//#endif
//    /* BLAS */
//#ifdef BENCH_CBLAS
//    std::vector<float> cx(N), cy(N);
//    ad::copy(x, cx);
//    ad::copy(y, cy);
//    BENCHMARK_HOST(cblas_saxpy(N, 1, cx.data(), 1, cy.data(), 1), 3*N*dtsize/t);
//#endif
//    /* CuBLAS */
//#ifdef BENCH_CUBLAS
//    T *cux, *cuy;
//    cudaMalloc((void**) &cux, N * sizeof(T));
//    cudaMalloc((void**) &cuy, N * sizeof(T));
//    BENCHMARK_CUDA(cublasSaxpy(N, 2, cux, 1, cuy, 1), 3*N*dtsize/t)
//    cudaFree(cux);
//    cudaFree(cuy);
//#endif
//    std::cout << std::endl;
//  }
//  std::cout << "\n\n" << std::flush;

  std::cout << "#DOT" << std::endl;
  for(int_t i = 0 ; i < BLAS1_N.size() ; ++i)
  {
    int_t N = BLAS1_N[i];
    std::cout << N;
    /* ATIDLAS */
    ad::array x(N, dtype), y(N, dtype);
    ad::array scratch(N, dtype);
    ad::scalar s(dtype);
    s = dot(x,y); queue.finish();
    BENCHMARK_ATIDLAS(s = ad::control(dot(x,y), ad::execution_options_type(0, &events), ad::dispatcher_options_type(true)), 2*N*dtsize/t)
    /* clAmdBlas */
#ifdef BENCH_CLAMDBLAS
    BENCHMARK_CLAMDBLAS(clAmdBlasSdot(N, s.data()(), 0, x.data()(), 0, 1, y.data()(), 0, 1, scratch.data()(), 1, &queue(), 0, NULL, &event()), 2*N*dtsize/t)
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
    BENCHMARK_CUDA(cublasSdot(N, cux, 1, cuy, 1, &result), 2*N*dtsize/t)
    cudaFree(cux);
    cudaFree(cuy);
#endif
    std::cout << std::endl;
  }
  std::cout << "\n\n" << std::flush;

//  /*---------*/
//  /*--BLAS2--*/
//  /*---------*/
//  //T-layout
//  std::cout << "#GEMV-T" << std::endl;
//  for(int_t N: std::vector<int>{128})
//    for(int_t M: create_full_range(128, 10000, 64))
//    {
//      std::cout << M << "," << N;
//      /* ATIDLAS */
//      ad::array A(N, M, dtype), y(M, dtype), x(N, dtype);
//      y = dot(trans(A),x); queue.finish();
//      BENCHMARK_ATIDLAS(y = ad::control(dot(trans(A),x), ad::execution_options_type(0, &events), ad::dispatcher_options_type(true)),(M*N + M + N)*dtsize/t);
//  #ifdef BENCH_CLAMDBLAS
//      BENCHMARK_CLAMDBLAS(clAmdBlasSgemv(clAmdBlasColumnMajor, clAmdBlasTrans, N, M, 1, A.data()(), A.ld(), x.data()(), 0, 1, 0, y.data()(), 0, 1, 1, &queue(),0, NULL, &event()), (M*N + M + N)*dtsize/t)
//  #endif
//  #ifdef BENCH_CBLAS
//      std::vector<float> cA(N*M), cx(N), cy(M);
//      ad::copy(x, cx);
//      ad::copy(y, cy);
//      ad::copy(A, cA);
//      BENCHMARK_HOST(cblas_sgemv(CblasColMajor, CblasTrans, N, M, 1, cA.data(), N, cx.data(), 1, 0, cy.data(), 1), (M*N + M + N)*dtsize/t);
//  #endif
//  #ifdef BENCH_CUBLAS
//      T *cuA, *cux, *cuy;
//      cudaMalloc((void**) &cuA, N * M * sizeof(T));
//      cudaMalloc((void**) &cux, N * sizeof(T));
//      cudaMalloc((void**) &cuy, M * sizeof(T));
//      BENCHMARK_CUDA(cublasSgemv(cublasTrans, N, M, 1, cuA, N, cux, 1, 0, cuy, 1), (M*N + M + N)*dtsize/t)
//      cudaFree(cuA);
//      cudaFree(cux);
//      cudaFree(cuy);
//  #endif
//      std::cout << std::endl;
//    }
//    std::cout << "\n\n" << std::flush;

//  /*---------*/
//  /*--BLAS3--*/
//  /*---------*/
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

  ad::cl_ext::queue_properties = CL_QUEUE_PROFILING_ENABLE;

  int device_idx = 0;
  ad::cl_ext::queues_type::data_type const & queues = ad::cl_ext::queues.data();

  if(queues.size()>1){
    if(argc!=2)
    {
      std::cerr << "usage : blas-bench [DEVICE_IDX]" << std::endl;
      std::cout << "Devices available: " << std::endl;
      unsigned int current=0;
      for(ad::cl_ext::queues_type::data_type::const_iterator it = queues.begin() ; it != queues.end() ; ++it){
        cl::Device device = it->first.getInfo<CL_CONTEXT_DEVICES>()[0];
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
