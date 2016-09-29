#include "isaac/array.h"
#include "isaac/runtime/execute.h"
#ifdef BENCH_CLBLAS
  #include "clBLAS.h"
#endif
#ifdef BENCH_MKL
  #include "mkl_cblas.h"
#elif defined(BENCH_CBLAS)
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

#include "common.hpp"


namespace sc = isaac;
typedef sc::int_t int_t;

template<class T>
void bench(sc::numeric_type dtype, std::string operation)
{

  Timer tmr;

//
// MACROS FOR BENCHMARKING
//
#define CL_HANDLE(X) X.handle().cl()
#define CU_HANDLE(X) X.handle().cu()

#define BENCHMARK_ISAAC(OP, PERF) \
  {\
  std::vector<long> times;\
  double total_time = 0;\
  OP;\
  queue.synchronize();\
  while(total_time*1e-9 < 1e-1){\
    tmr.start();\
    OP;\
    queue.synchronize();\
    times.push_back(tmr.get().count());\
    total_time+=times.back();\
  }\
  double t = min(times);\
  std::cout << "\t" << (int)PERF << std::flush;\
  }

#define BENCHMARK_CLBLAS(OP, PERF) \
  {\
  std::vector<long> times;\
  double total_time = 0;\
  OP;\
  queue.synchronize();\
  while(total_time*1e-9 < 1e-1){\
    tmr.start();\
    OP;\
    queue.synchronize();\
    times.push_back(tmr.get().count());\
    total_time+=times.back();\
  }\
  double t = min(times);\
  std::cout << "\t" << PERF << std::flush;\
  }

#define BENCHMARK_HOST(OP, PERF) \
  {\
  long total_time = 0;\
  std::vector<long> times;\
  OP;\
  while(total_time*1e-9 < 1e-1){\
    tmr.start();\
    OP;\
    long time = tmr.get().count();\
    times.push_back(time);\
    total_time += time;\
  }\
  double t = min(times);\
  std::cout << "\t" << PERF << std::flush;\
  }

#define BENCHMARK_CUDA(OP, PERF) \
  {\
    std::vector<long> times;\
    double total_time = 0;\
    OP;\
    cudaDeviceSynchronize();\
    while(total_time*1e-9 < 1e-1){\
      tmr.start();\
      OP;\
      cudaDeviceSynchronize();\
      times.push_back(tmr.get().count());\
      total_time+=times.back();\
    }\
    double t = min(times);\
    std::cout << "\t" << PERF << std::flush;\
  }

  unsigned int dtsize = sc::size_of(dtype);
  sc::driver::CommandQueue & queue = sc::driver::backend::queues::get(sc::driver::backend::contexts::get_default(),0);
  std::map<std::string, std::string> metric{ {"axpy", "GB/s"}, {"dot", "GB/s"}, {"gemv", "GB/s"}, {"gemm", "GFLOPS"}};
  sc::array flush((int)1e6, sc::FLOAT_TYPE);

  //
  // RUN BENCHMARKS
  //

  /*---------*/
  /*--BLAS1--*/
  /*---------*/

  if(operation=="axpy")
  {
    float alpha = 1;
    for(int_t N: create_log_range((int)1e3, (int)1e8, 50, 64))
    {
      std::cout << N;
      sc::array x(N, dtype), y(N, dtype);
      /* ISAAC */
      BENCHMARK_ISAAC(y = x + alpha*y, 3*N*dtsize/t)
//      BENCHMARK_ISAAC(y = sc::execution_handler(x + alpha*y, sc::execution_options_type(), sc::dispatcher_options_type(true)), 3*N*dtsize/t)
      /* clblas */
  #ifdef BENCH_CLBLAS
      if(x.context().backend()==sc::driver::OPENCL)
          BENCHMARK_CLBLAS(clblasSaxpy(N, alpha, CL_HANDLE(x.data()), 0, 1, CL_HANDLE(y.data()), 0, 1, 1, &CL_HANDLE(queue), 0, NULL, NULL), 3*N*dtsize/t);
  #endif
      /* BLAS */
  #ifdef BENCH_CBLAS
      std::vector<float> cx(N), cy(N);
      sc::copy(x, cx);
      sc::copy(y, cy);
      BENCHMARK_HOST(cblas_saxpy(N, alpha, cx.data(), 1, cy.data(), 1), 3*N*dtsize/t);
  #endif
      /* CuBLAS */
  #ifdef BENCH_CUBLAS
      BENCHMARK_CUDA(cublasSaxpy(N, alpha, (T*)CU_HANDLE(x.data()), 1, (T*)CU_HANDLE(y.data()), 1), 3*N*dtsize/t)
  #endif
      std::cout << std::endl;
    }
  }

  if(operation=="dot")
  {
    for(int_t N: create_log_range((int)1e3, (int)1e8, 50, 64))
    {
      std::cout << N;
      /* ISAAC */
      sc::array x(N, dtype), y(N, dtype);
      sc::array scratch(N, dtype);
      sc::scalar s(dtype);
      s = dot(x,y); queue.synchronize();
      BENCHMARK_ISAAC(s = dot(x,y), 2*N*dtsize/t)
      /* clblas */
  #ifdef BENCH_CLBLAS
      if(x.context().backend()==sc::driver::OPENCL)
        BENCHMARK_CLBLAS(clblasSdot(N, CL_HANDLE(s.data()), 0, CL_HANDLE(x.data()), 0, 1, CL_HANDLE(y.data()), 0, 1, CL_HANDLE(scratch.data()), 1, &CL_HANDLE(queue), 0, NULL, NULL), 2*N*dtsize/t)
  #endif
      /* BLAS */
  #ifdef BENCH_CBLAS
      std::vector<float> cx(N), cy(N);
      sc::copy(x, cx);
      sc::copy(y, cy);
      BENCHMARK_HOST(cblas_sdot(N, cx.data(), 1, cy.data(), 1), 2*N*dtsize/t);
  #endif
  #ifdef BENCH_CUBLAS
      BENCHMARK_CUDA(cublasSdot(N, (T*)CU_HANDLE(x.data()), 1, (T*)CU_HANDLE(y.data()), 1), 2*N*dtsize/t)
  #endif
      std::cout << std::endl;
    }
  }

  if(operation.substr(0, 4)=="gemv")
  {
    std::vector<std::tuple<std::string, char,int_t, int_t> > MNs;
    //Linear System
    MNs.push_back(std::make_tuple("square153[N]", 'N',153,153));
    MNs.push_back(std::make_tuple("square153[T]", 'T',153,153));
    MNs.push_back(std::make_tuple("square1024[T]", 'T',1024,1024));
    MNs.push_back(std::make_tuple("square2867[N]", 'N',2867,2867));
    MNs.push_back(std::make_tuple("square2867[T]", 'T',2867,2867));
    //Normalization
    MNs.push_back(std::make_tuple("norm64[N]", 'N', 64, 60000));
    MNs.push_back(std::make_tuple("norm64[T]", 'T', 64, 60000));
    MNs.push_back(std::make_tuple("norm256[N]", 'N', 256, 60000));
    MNs.push_back(std::make_tuple("norm256[T]", 'T', 256, 60000));
    MNs.push_back(std::make_tuple("norm1024[N]", 'N', 1024, 60000));
    MNs.push_back(std::make_tuple("norm1024[T]", 'T', 1024, 60000));

    //Householder
    MNs.push_back(std::make_tuple("tallskinny-1[N]", 'N', 10, 60000));
    MNs.push_back(std::make_tuple("tallskinny-1[T]", 'T', 10, 60000));
    MNs.push_back(std::make_tuple("tallskinny-2[N]", 'N', 30, 60000));
    MNs.push_back(std::make_tuple("tallskinny-2[T]", 'T', 30, 60000));

    /*---------*/
    /*--BLAS2--*/
    /*---------*/
    for(std::tuple<std::string, char, int_t, int_t> MN: MNs)
    {
        bool AT = std::get<1>(MN) == 'T';
        int_t M = std::get<2>(MN);
        int_t N = std::get<3>(MN);
        std::cout << '"' << std::get<0>(MN) << '"';
        int_t As1 = M, As2 = N;
        if(AT) std::swap(As1, As2);

        /* ISAAC */
        sc::array A(As1, As2, dtype), y(M, dtype), x(N, dtype);
    #ifdef HAS_A_BLAS
        int_t lda = A.stride()[1];
    #endif
        BENCHMARK_ISAAC(y = AT?dot(A.T,x):dot(A,x),(M*N + M + N)*dtsize/t);
//        BENCHMARK_ISAAC(y = sc::execution_handler(AT?dot(A.T,x):dot(A,x), sc::execution_options_type(), sc::dispatcher_options_type(true)),(M*N + M + N)*dtsize/t);
    #ifdef BENCH_CLBLAS
        if(y.context().backend()==sc::driver::OPENCL)
            BENCHMARK_CLBLAS(clblasSgemv(clblasColumnMajor, AT?clblasTrans:clblasNoTrans, As1, As2, 1, CL_HANDLE(A.data()), 0, lda, CL_HANDLE(x.data()), 0, 1, 0, CL_HANDLE(y.data()), 0, 1, 1, &CL_HANDLE(queue),0, NULL, NULL), (M*N + M + N)*dtsize/t)
    #endif
    #ifdef BENCH_CBLAS
        std::vector<float> cA(M*N), cx(N), cy(M);
        sc::copy(x, cx);
        sc::copy(y, cy);
        sc::copy(A, cA);
        BENCHMARK_HOST(cblas_sgemv(CblasColMajor, AT?CblasTrans:CblasNoTrans, As1, As2, 1, cA.data(), lda, cx.data(), 1, 0, cy.data(), 1), (M*N + M + N)*dtsize/t);
    #endif
    #ifdef BENCH_CUBLAS
        BENCHMARK_CUDA(cublasSgemv(AT?'t':'n', As1, As2, 1, (T*)CU_HANDLE(A.data()), lda, (T*)CU_HANDLE(x.data()), 1, 0, (T*)CU_HANDLE(y.data()), 1), (M*N + M + N)*dtsize/t)
    #endif
        std::cout << std::endl;
      }
  }

  if(operation.substr(0,4)=="gemm")
  {
    std::vector<std::tuple<std::string, int_t, int_t, int_t, char, char> > MNKs;
    //Square
    MNKs.push_back(std::make_tuple("Square",896,896,896,'N','T'));
    MNKs.push_back(std::make_tuple("Square",2560,2560,2560,'N','T'));
    //DeepBench
    for(size_t MK: std::vector<size_t>{1760, 2048, 2560})
      for(size_t N: std::vector<size_t>{16, 32, 64, 128, MK})
        MNKs.push_back(std::make_tuple("Deep", MK, N, MK, 'N', 'N'));
    for(size_t MK: std::vector<size_t>{1760, 2048, 2560})
      for(size_t N: std::vector<size_t>{16, 32, 64, 128, MK})
        MNKs.push_back(std::make_tuple("Deep", MK, N, MK, 'T', 'N'));
    for(size_t MK: std::vector<size_t>{1760})
      MNKs.push_back(std::make_tuple("Deep", MK, 1733, MK, 'N', 'T'));
    //Covariance (e.g., ICA, 10minutes/100Hz)
    MNKs.push_back(std::make_tuple("Cov",32,32,60000,'N','T'));
    MNKs.push_back(std::make_tuple("Cov",256,256,60000,'N','T'));
    //Bi-diagonalization
    MNKs.push_back(std::make_tuple("Lapack",4096,4096,32,'N','T'));
    MNKs.push_back(std::make_tuple("Lapack",3456,3456,32,'N','T'));
    MNKs.push_back(std::make_tuple("Lapack",896,896,32,'N','T'));

    std::cout << "BENCH\tM\tN\tK\ta_t\tb_t\tISAAC";;
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

    /*---------*/
    /*--BLAS3--*/
    /*---------*/
    for(auto MNK: MNKs)
    {
        bool AT = std::get<4>(MNK)=='T';
        bool BT = std::get<5>(MNK)=='T';
        int_t M = std::get<1>(MNK);
        int_t N = std::get<2>(MNK);
        int_t K = std::get<3>(MNK);
        std::cout << std::get<0>(MNK) << "\t" << M << "\t" << N
                  << "\t" << K << "\t" << std::get<4>(MNK) << "\t" << std::get<5>(MNK);
        /* ISAAC */
        int_t As1 = M, As2 = K;
        if(AT) std::swap(As1, As2);
        int_t Bs1 = K, Bs2 = N;
        if(BT) std::swap(Bs1, Bs2);

        sc::array C(M, N, dtype), A(As1, As2, dtype), B(Bs1, Bs2, dtype);
    #ifdef HAS_A_BLAS
        int_t lda = A.stride()[1], ldb = B.stride()[1], ldc = C.stride()[1];
    #endif
        BENCHMARK_ISAAC(C = AT?(BT?dot(A.T,B.T):dot(A.T,B)):(BT?dot(A,B.T):dot(A,B)), (double)2*M*N*K/t*1e-3);
//        BENCHMARK_ISAAC(C = sc::execution_handler(AT?(BT?dot(A.T,B.T):dot(A.T,B)):(BT?dot(A,B.T):dot(A,B)), sc::execution_options_type(0), sc::dispatcher_options_type(true)), (double)2*M*N*K/t);
        /* clblas */
    #ifdef BENCH_CLBLAS
        if(C.context().backend()==sc::driver::OPENCL)
            BENCHMARK_CLBLAS(clblasSgemm(clblasColumnMajor, AT?clblasTrans:clblasNoTrans, BT?clblasTrans:clblasNoTrans, M, N, K, 1, CL_HANDLE(A.data()), 0, lda, CL_HANDLE(B.data()), 0, ldb,
                                                0, CL_HANDLE(C.data()), 0, ldc, 1, &CL_HANDLE(queue),0, NULL, NULL), (double)2*M*N*K/t*1e-3)
    #endif
        /* BLAS */
    #ifdef BENCH_CBLAS
        std::vector<float> cC(M*N), cA(M*K), cB(N*K);
        sc::copy(C, cC);
        sc::copy(A, cA);
        sc::copy(B, cB);
        BENCHMARK_HOST(cblas_sgemm(CblasColMajor, AT?CblasTrans:CblasNoTrans, BT?CblasTrans:CblasNoTrans, M, N, K, 1, cA.data(), lda, cB.data(), ldb, 1, cC.data(), ldc), (double)2*M*N*K/t*1e-3);
    #endif
    #ifdef BENCH_CUBLAS
        BENCHMARK_CUDA(cublasSgemm(AT?'t':'n', BT?'t':'n', M, N, K, 1, (T*)CU_HANDLE(A.data()), lda, (T*)CU_HANDLE(B.data()), ldb, 1, (T*)CU_HANDLE(C.data()), ldc), (double)2*M*N*K/t*1e-3)
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
  sc::driver::backend::default_queue_properties = CL_QUEUE_PROFILING_ENABLE;

  int device_idx = 0;
  std::list<sc::driver::Context const *> contexts;
  sc::driver::backend::contexts::get(contexts);

  std::string operation;
  if(contexts.size() > 1)
  {
    if(args.size() != 3)
    {
      std::cerr << "usage : blas-bench DEVICE_IDX OPERATION" << std::endl;
      std::cout << "Devices available: " << std::endl;
      unsigned int current=0;
      for(sc::driver::Context const * context: contexts)
      {
          sc::driver::Device device = sc::driver::backend::queues::get(*context,0).device();
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

  sc::driver::backend::default_device = device_idx;
  std::cout << std::fixed << std::setprecision(2);
  bench<float>(sc::FLOAT_TYPE, operation);

#ifdef BENCH_CLBLAS
  clblasTeardown();
#endif
}
