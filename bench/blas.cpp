#include "isaac/array.h"
#include "isaac/symbolic/execute.h"
#include "isaac/tools/timer.hpp"
#ifdef BENCH_CLBLAS
  #include "isaac/wrap/clBLAS.h"
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

#define HAS_A_BLAS defined(BENCH_CBLAS) or defined(BENCH_CLBLAS) or defined(BENCH_CUBLAS)
namespace isc = isaac;
typedef isc::int_t int_t;

template<std::size_t> struct int_{};

template <class Tuple, size_t Pos>
std::ostream& print_tuple(std::ostream& out, const Tuple& t, int_<Pos> ) {
out << std::get< std::tuple_size<Tuple>::value-Pos >(t) << ',';
return print_tuple(out, t, int_<Pos-1>());
}

template <class Tuple>
std::ostream& print_tuple(std::ostream& out, const Tuple& t, int_<1> ) {
return out << std::get<std::tuple_size<Tuple>::value-1>(t);
}

template <class... Args>
std::ostream& operator<<(std::ostream& out, const std::tuple<Args...>& t) {
print_tuple(out, t, int_<sizeof...(Args)>());
return out;
}

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

static double time_event(unsigned long sum, isc::driver::Event const & e)
{ return sum + e.elapsed_time();}

template<class T>
void bench(isc::numeric_type dtype, std::string operation)
{

//
// MACROS FOR BENCHMARKING
//
#define CL_HANDLE(X) X.handle().cl()

#define BENCHMARK_ISAAC(OP, PERF) \
  {\
  std::vector<long> times;\
  double total_time = 0;\
  while(total_time*1e-9 < 1e-3){\
    std::list<isc::driver::Event> events;\
    flush = isc::zeros(1e6, 1, dtype);\
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
    cl_event event;\
    flush = isc::zeros(1e6, 1, dtype);\
    OP;\
    queue.synchronize();\
    times.push_back(isc::driver::Event(event).elapsed_time());\
    total_time+=times.back();\
  }\
  double t = median(times);\
  std::cout << " " << PERF << std::flush;\
  }

#define BENCHMARK_HOST(OP, PERF) \
  {\
  isc::tools::timer tmr;\
  double total_time = 0;\
  std::vector<double> times;\
  while(total_time < 1e-2){\
    std::vector<int> cache_flusher(10000000, 0);\
    tmr.start();\
    OP;\
    double time = tmr.get();\
    times.push_back(time);\
    total_time += time;\
  }\
  double t = 1e9*median(times);\
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
  OP;\
  cudaThreadSynchronize();\
  while(total_time*1e-3 < 1e-3){\
    flush = isc::zeros(1e6, 1, dtype);\
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

  unsigned int dtsize = isc::size_of(dtype);
  isc::driver::CommandQueue & queue = isc::driver::backend::queue(isc::driver::backend::default_context(),0);
  std::map<std::string, std::string> metric{ {"axpy", "GB/s"}, {"dot", "GB/s"}, {"gemv", "GB/s"}, {"gemm", "GFLOPS"}};
  isc::array flush(1e6, dtype);
  std::cout << "#" << operation << " (" << metric[operation] << ")" << std::endl;
  std::cout << "N";
  std::cout << "\tISAAC";
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

  /*---------*/
  /*--BLAS1--*/
  /*---------*/

  if(operation=="axpy")
  {
    float alpha = 1;
    for(int_t N: create_log_range(1e3, 2e7, 50, 64))
    {
      std::cout << N;
      isc::array x(N, dtype), y(N, dtype);
      /* ISAAC */
      std::list<isc::driver::Event> events;\
      BENCHMARK_ISAAC(y = isc::control(x + alpha*y, isc::execution_options_type(0, &events)), 3*N*dtsize/t)
      /* clblas */
  #ifdef BENCH_CLBLAS
      BENCHMARK_CLBLAS(clblasSaxpy(N, alpha, CL_HANDLE(x.data()), 0, 1, CL_HANDLE(y.data()), 0, 1, 1, &CL_HANDLE(queue), 0, NULL, &event), 3*N*dtsize/t);
  #endif
      /* BLAS */
  #ifdef BENCH_CBLAS
      std::vector<float> cx(N), cy(N);
      isc::copy(x, cx);
      isc::copy(y, cy);
      BENCHMARK_HOST(cblas_saxpy(N, alpha, cx.data(), 1, cy.data(), 1), 3*N*dtsize/t);
  #endif
      /* CuBLAS */
  #ifdef BENCH_CUBLAS
      T *cux, *cuy;
      cudaMalloc((void**) &cux, N * sizeof(T));
      cudaMalloc((void**) &cuy, N * sizeof(T));
      BENCHMARK_CUDA(cublasSaxpy(N, alpha, cux, 1, cuy, 1), 3*N*dtsize/t)
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
      isc::array x(N, dtype), y(N, dtype);
      isc::array scratch(N, dtype);
      isc::scalar s(dtype);
      s = dot(x,y); queue.synchronize();
      BENCHMARK_ISAAC(s = isc::control(dot(x,y), isc::execution_options_type(0, &events)), 2*N*dtsize/t)
      /* clblas */
  #ifdef BENCH_CLBLAS
      BENCHMARK_CLBLAS(clblasSdot(N, CL_HANDLE(s.data()), 0, CL_HANDLE(x.data()), 0, 1, CL_HANDLE(y.data()), 0, 1, CL_HANDLE(scratch.data()), 1, &CL_HANDLE(queue), 0, NULL, &event), 2*N*dtsize/t)
  #endif
      /* BLAS */
  #ifdef BENCH_CBLAS
      std::vector<float> cx(N), cy(N);
      isc::copy(x, cx);
      isc::copy(y, cy);
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
        isc::array A(N, M, dtype), y(M, dtype), x(N, dtype);
    #if HAS_A_BLAS
        int_t lda = A.ld();
    #endif
        y = dot(trans(A),x); queue.synchronize();
        BENCHMARK_ISAAC(y = isc::control(dot(trans(A),x), isc::execution_options_type(0, &events)),(M*N + M + N)*dtsize/t);
    #ifdef BENCH_CLBLAS
        BENCHMARK_CLBLAS(clblasSgemv(clblasColumnMajor, clblasTrans, N, M, 1, CL_HANDLE(A.data()), 0, lda, CL_HANDLE(x.data()), 0, 1, 0, CL_HANDLE(y.data()), 0, 1, 1, &CL_HANDLE(queue),0, NULL, &event), (M*N + M + N)*dtsize/t)
    #endif
    #ifdef BENCH_CBLAS
        std::vector<float> cA(N*M), cx(N), cy(M);
        isc::copy(x, cx);
        isc::copy(y, cy);
        isc::copy(A, cA);
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
    std::vector<std::tuple<char, char, int_t, int_t, int_t> > MNKs;
    MNKs.push_back(std::make_tuple('N','T',1536,1536,1536));
    //AlexNet (Forward)
    MNKs.push_back(std::make_tuple('N','N',3025,96,363));
    MNKs.push_back(std::make_tuple('N','N',729,128,1200));
    MNKs.push_back(std::make_tuple('N','N',169,384,2304));
    MNKs.push_back(std::make_tuple('N','N',169,192,1728));
    MNKs.push_back(std::make_tuple('N','N',169,128,1728));
    //AlexNet (Backward)
    MNKs.push_back(std::make_tuple('T','N',1728,128,169));
    MNKs.push_back(std::make_tuple('T','N',1728,192,169));
    MNKs.push_back(std::make_tuple('T','N',2304,384,169));
    MNKs.push_back(std::make_tuple('T','N',1200,128,729));
    MNKs.push_back(std::make_tuple('T','N',363,96,3025));

    MNKs.push_back(std::make_tuple('N','T',169,1728,128));
    MNKs.push_back(std::make_tuple('N','T',169,1728,192));
    MNKs.push_back(std::make_tuple('N','T',169,2304,384));
    MNKs.push_back(std::make_tuple('N','T',729,1200,128));

    //Covariance (e.g., ICA)
    MNKs.push_back(std::make_tuple('N','N',64,64,32000));
    MNKs.push_back(std::make_tuple('N','N',1024,1024,32000));

    /*---------*/
    /*--BLAS3--*/
    /*---------*/
    for(std::tuple<char, char, int_t, int_t, int_t> MNK: MNKs)
    {
        bool AT = std::get<0>(MNK)=='T';
        bool BT = std::get<1>(MNK)=='T';
        int_t M = std::get<2>(MNK);
        int_t N = std::get<3>(MNK);
        int_t K = std::get<4>(MNK);
        std::cout << MNK;
        std::cout << std::flush;
        /* ISAAC */
        int_t As1 = M, As2 = K;
        if(AT) std::swap(As1, As2);
        int_t Bs1 = K, Bs2 = N;
        if(BT) std::swap(Bs1, Bs2);

        isc::array C(M, N, dtype), A(As1, As2, dtype), B(Bs1, Bs2, dtype);
    #if HAS_A_BLAS
        int_t lda = A.ld(), ldb = B.ld(), ldc = C.ld();
    #endif
//        BENCHMARK_ISAAC(C = isc::control(AT?(BT?dot(A.T(),B.T()):dot(A.T(),B)):(BT?dot(A,B.T()):dot(A,B)), isc::execution_options_type(0, &events)), (double)2*M*N*K/t);
        /* clblas */
    #ifdef BENCH_CLBLAS
        BENCHMARK_CLBLAS(clblasSgemm(clblasColumnMajor, AT?clblasTrans:clblasNoTrans, BT?clblasTrans:clblasNoTrans, M, N, K, 1, CL_HANDLE(A.data()), 0, lda, CL_HANDLE(B.data()), 0, ldb,
                                            0, CL_HANDLE(C.data()), 0, ldc, 1, &CL_HANDLE(queue),0, NULL, &event), (double)2*M*N*K/t)
    #endif
        /* BLAS */
    #ifdef BENCH_CBLAS
        std::vector<float> cC(M*N), cA(M*K), cB(N*K);
        isc::copy(C, cC);
        isc::copy(A, cA);
        isc::copy(B, cB);
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
  isc::driver::backend::queue_properties = CL_QUEUE_PROFILING_ENABLE;

  int device_idx = 0;
  std::list<isc::driver::Context const *> const & contexts = isc::driver::backend::contexts();

  std::string operation;
  if(contexts.size() > 1)
  {
    if(args.size() != 3)
    {
      std::cerr << "usage : blas-bench DEVICE_IDX OPERATION" << std::endl;
      std::cout << "Devices available: " << std::endl;
      unsigned int current=0;
      for(isc::driver::Context const * context: contexts)
      {
          isc::driver::Device device = isc::driver::backend::queue(*context,0).device();
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

  isc::driver::backend::default_device = device_idx;
  std::cout << "#Benchmark : BLAS" << std::endl;
  std::cout << "#----------------" << std::endl;
  bench<float>(isc::FLOAT_TYPE, operation);

#ifdef BENCH_CLBLAS
  clblasTeardown();
#endif
}
