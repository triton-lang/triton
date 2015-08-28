#include "isaac/array.h"
#include "isaac/symbolic/execute.h"
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

#include "common.hpp"


namespace sc = isaac;
typedef sc::int_t int_t;

static long time_event(long sum, sc::driver::Event const & e)
{
    return sum + e.elapsed_time();
}


template<class T>
void bench(sc::numeric_type dtype, std::string operation)
{

//
// MACROS FOR BENCHMARKING
//
#define CL_HANDLE(X) X.handle().cl()
#define CU_HANDLE(X) X.handle().cu()

#define BENCHMARK_ISAAC(OP, PERF) \
  {\
  std::vector<double> times;\
  double total_time = 0;\
  while(total_time*1e-9 < 1e-2){\
    std::list<sc::driver::Event> events;\
    OP;\
    queue.synchronize();\
    times.push_back((double)std::accumulate(events.begin(), events.end(), 0, &time_event));\
    total_time+=times.back();\
  }\
  double t = mean(times);\
  std::cout << " " << (int)(PERF) << std::flush;\
  }

#define BENCHMARK_CLBLAS(OP, PERF) \
  {\
  std::vector<long> times;\
  double total_time = 0;\
  while(total_time*1e-9 < 1e-2){\
    cl_event event;\
    OP;\
    queue.synchronize();\
    times.push_back(sc::driver::Event(event).elapsed_time());\
    total_time+=times.back();\
  }\
  double t = mean(times);\
  std::cout << " " << (int)(PERF) << std::flush;\
  }

#define BENCHMARK_HOST(OP, PERF) \
  {\
  Timer tmr;\
  long total_time = 0;\
  std::vector<long> times;\
  while(total_time*1e-9 < 1e-2){\
    tmr.start();\
    OP;\
    long time = tmr.get().count();\
    times.push_back(time);\
    total_time += time;\
  }\
  double t = mean(times);\
  std::cout << " " << (int)(PERF) << std::flush;\
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
  while(total_time*1e-3 < 1e-2){\
    cudaEventRecord(start,0);\
    OP;\
    cudaEventRecord(stop,0);\
    cudaEventSynchronize(stop);\
    cudaEventElapsedTime(&time, start, stop);\
    times.push_back(time*1e6);\
    total_time+=time;\
  }\
  double t = mean(times);\
  std::cout << "\t" << (int)(PERF) << std::flush;\
  }

  unsigned int dtsize = sc::size_of(dtype);
  sc::driver::CommandQueue & queue = sc::driver::backend::queues::get(sc::driver::backend::contexts::get_default(),0);
  std::map<std::string, std::string> metric{ {"axpy", "GB/s"}, {"dot", "GB/s"}, {"gemv", "GB/s"}, {"gemm", "GFLOPS"}};
  sc::array flush((int)1e6, sc::FLOAT_TYPE);
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
    for(int_t N: create_log_range((int)1e3, (int)1e8, 50, 64))
    {
      std::cout << N;
      sc::array x(N, dtype), y(N, dtype);
      /* ISAAC */
      std::list<sc::driver::Event> events;
      BENCHMARK_ISAAC(y = sc::control(x + alpha*y, sc::execution_options_type(0, &events)), 3*N*dtsize/t)
      /* clblas */
  #ifdef BENCH_CLBLAS
      if(x.context().backend()==sc::driver::OPENCL)
            BENCHMARK_CLBLAS(clblasSaxpy(N, alpha, CL_HANDLE(x.data()), 0, 1, CL_HANDLE(y.data()), 0, 1, 1, &CL_HANDLE(queue), 0, NULL, &event), 3*N*dtsize/t);
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
      BENCHMARK_ISAAC(s = sc::control(dot(x,y), sc::execution_options_type(0, &events)), 2*N*dtsize/t)
      /* clblas */
  #ifdef BENCH_CLBLAS
      if(x.context().backend()==sc::driver::OPENCL)
        BENCHMARK_CLBLAS(clblasSdot(N, CL_HANDLE(s.data()), 0, CL_HANDLE(x.data()), 0, 1, CL_HANDLE(y.data()), 0, 1, CL_HANDLE(scratch.data()), 1, &CL_HANDLE(queue), 0, NULL, &event), 2*N*dtsize/t)
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
    std::vector<std::tuple<char,int_t, int_t> > MNs;
    MNs.push_back(std::make_tuple('N',896,896));
    MNs.push_back(std::make_tuple('N',3072,3072));
    //AlexNet
    MNs.push_back(std::make_tuple('N',1000,256));
    MNs.push_back(std::make_tuple('N',4096,256));

    MNs.push_back(std::make_tuple('T',169,256));
    MNs.push_back(std::make_tuple('T',169,384));
    MNs.push_back(std::make_tuple('T',729,256));
    MNs.push_back(std::make_tuple('T',3025,96));

    /*---------*/
    /*--BLAS2--*/
    /*---------*/
    //T-layout
    for(std::tuple<char, int_t, int_t> MN: MNs)
    {
        bool AT = std::get<0>(MN) == 'T';
        int_t M = std::get<1>(MN);
        int_t N = std::get<2>(MN);
        std::cout << MN << ",";
        int_t As1 = M, As2 = N;
        if(AT) std::swap(As1, As2);

        /* ISAAC */
        sc::array A(As1, As2, dtype), y(M, dtype), x(N, dtype);
    #ifdef HAS_A_BLAS
        int_t lda = A.ld();
    #endif
        BENCHMARK_ISAAC(y = sc::control(AT?dot(A.T(),x):dot(A,x), sc::execution_options_type(0, &events)),(M*N + M + N)*dtsize/t);
    #ifdef BENCH_CLBLAS
        if(y.context().backend()==sc::driver::OPENCL)
            BENCHMARK_CLBLAS(clblasSgemv(clblasColumnMajor, AT?clblasTrans:clblasNoTrans, As1, As2, 1, CL_HANDLE(A.data()), 0, lda, CL_HANDLE(x.data()), 0, 1, 0, CL_HANDLE(y.data()), 0, 1, 1, &CL_HANDLE(queue),0, NULL, &event), (M*N + M + N)*dtsize/t)
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
    std::vector<std::tuple<std::string, char, char, int_t, int_t, int_t> > MNKs;
    //Square
    MNKs.push_back(std::make_tuple("Square [N=896]",'N','T',896,896,896));
    MNKs.push_back(std::make_tuple("Square [N=2560]",'N','T',2560,2560,2560));

    //Convolution
    MNKs.push_back(std::make_tuple("Convolution [AlexNet-1]",'N','N',3025,96,363));
    MNKs.push_back(std::make_tuple("Convolution [AlexNet-2]",'N','N',729,128,1200));
    MNKs.push_back(std::make_tuple("Convolution [AlexNet-3]",'N','N',169,384,2304));
    MNKs.push_back(std::make_tuple("Convolution [AlexNet-4]",'N','N',169,192,1728));
    MNKs.push_back(std::make_tuple("Convolution [AlexNet-5]",'N','N',169,128,1728));
//    MNKs.push_back(std::make_tuple("Convolution [LeNet-1],'N','N',576,20,25));
//    MNKs.push_back(std::make_tuple("Convolution [LeNet-2]",'N','N',64,50,500));

    //Convolution Gradient-1
//    MNKs.push_back(std::make_tuple("Convolution Gradient-1 [AlexNet-5]",'T','N',1728,128,169));
//    MNKs.push_back(std::make_tuple("Convolution Gradient-1 [AlexNet-4]",'T','N',1728,192,169));
//    MNKs.push_back(std::make_tuple("Convolution Gradient-1 [AlexNet-3]",'T','N',2304,384,169));
//    MNKs.push_back(std::make_tuple("Convolution Gradient-1 [AlexNet-2]",'T','N',1200,128,729));
//    MNKs.push_back(std::make_tuple("Convolution Gradient-1 [AlexNet-1]",'T','N',363,96,3025));
//    MNKs.push_back(std::make_tuple("Conv. Gradient-1 [LeNet-2]",'T','N',500,50,64));
//    MNKs.push_back(std::make_tuple("Conv. Gradient-1 [LeNet-1]",'T','N',25,20,576));

    //Convolution Gradient-2
//    MNKs.push_back(std::make_tuple("Convolution Gradient-2 [AlexNet-5]",'N','T',169,1728,128));
//    MNKs.push_back(std::make_tuple("Convolution Gradient-2 [AlexNet-4]",'N','T',169,1728,192));
//    MNKs.push_back(std::make_tuple("Convolution Gradient-2 [AlexNet-3]",'N','T',169,2304,384));
//    MNKs.push_back(std::make_tuple("Convolution Gradient-2 [AlexNet-2]",'N','T',729,1200,128));
//    MNKs.push_back(std::make_tuple("Conv. Gradient-2 [LeNet-2]",'N','T',64,500,50));

    //Covariance (e.g., ICA, 10minutes/1khz)
    MNKs.push_back(std::make_tuple("ICA [32 channels]",'N','T',32,32,600000));
    MNKs.push_back(std::make_tuple("ICA [256 channels]",'N','T',256,256,600000));

//    //Bi-diagonalization
    MNKs.push_back(std::make_tuple("Bidiagonalization [Iteration 1]",'N','T',4096,4096,32));
    MNKs.push_back(std::make_tuple("Bidiagonalization [Iteration 10]",'N','T',3456,3456,32));
    MNKs.push_back(std::make_tuple("Bidiagonalization [Iteration 50]",'N','T',896,896,32));

    /*---------*/
    /*--BLAS3--*/
    /*---------*/
    for(std::tuple<std::string, char, char, int_t, int_t, int_t> MNK: MNKs)
    {
        bool AT = std::get<1>(MNK)=='T';
        bool BT = std::get<2>(MNK)=='T';
        int_t M = std::get<3>(MNK);
        int_t N = std::get<4>(MNK);
        int_t K = std::get<5>(MNK);
        std::cout << "\"" << std::get<0>(MNK) << "\"";
        std::cout << std::flush;
        /* ISAAC */
        int_t As1 = M, As2 = K;
        if(AT) std::swap(As1, As2);
        int_t Bs1 = K, Bs2 = N;
        if(BT) std::swap(Bs1, Bs2);

        sc::array C(M, N, dtype), A(As1, As2, dtype), B(Bs1, Bs2, dtype);
    #ifdef HAS_A_BLAS
        int_t lda = A.ld(), ldb = B.ld(), ldc = C.ld();
    #endif
        BENCHMARK_ISAAC(C = sc::control(AT?(BT?dot(A.T(),B.T()):dot(A.T(),B)):(BT?dot(A,B.T()):dot(A,B)), sc::execution_options_type(0, &events), sc::dispatcher_options_type(true)), (double)2*M*N*K/t);
        /* clblas */
    #ifdef BENCH_CLBLAS
        if(C.context().backend()==sc::driver::OPENCL)
            BENCHMARK_CLBLAS(clblasSgemm(clblasColumnMajor, AT?clblasTrans:clblasNoTrans, BT?clblasTrans:clblasNoTrans, M, N, K, 1, CL_HANDLE(A.data()), 0, lda, CL_HANDLE(B.data()), 0, ldb,
                                                0, CL_HANDLE(C.data()), 0, ldc, 1, &CL_HANDLE(queue),0, NULL, &event), (double)2*M*N*K/t)
    #endif
        /* BLAS */
    #ifdef BENCH_CBLAS
        std::vector<float> cC(M*N), cA(M*K), cB(N*K);
        sc::copy(C, cC);
        sc::copy(A, cA);
        sc::copy(B, cB);
        BENCHMARK_HOST(cblas_sgemm(CblasColMajor, AT?CblasTrans:CblasNoTrans, BT?CblasTrans:CblasNoTrans, M, N, K, 1, cA.data(), lda, cB.data(), ldb, 1, cC.data(), ldc), (double)2*M*N*K/t);
    #endif
    #ifdef BENCH_CUBLAS
        BENCHMARK_CUDA(cublasSgemm(AT?'t':'n', BT?'t':'n', M, N, K, 1, (T*)CU_HANDLE(A.data()), lda, (T*)CU_HANDLE(B.data()), ldb, 1, (T*)CU_HANDLE(C.data()), ldc), (double)2*M*N*K/t)
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
  std::cout << "#Benchmark : BLAS" << std::endl;
  std::cout << "#----------------" << std::endl;
  bench<float>(sc::FLOAT_TYPE, operation);

#ifdef BENCH_CLBLAS
  clblasTeardown();
#endif
}
