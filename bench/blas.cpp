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

typedef sc::int_t int_t;

Timer tmr;

template<class OP, class SYNC>
double bench(OP const & op, SYNC const & sync)
{
  std::vector<long> times;
  double total_time = 0;
  op();
  sync();
  while(total_time*1e-9 < 2e-1){
    tmr.start();
    op();
    sync();
    times.push_back(tmr.get().count());
    total_time+=times.back();
  }
  return min(times);
}

template<class T>
void bench(sc::numeric_type dtype, std::string operation)
{
  using std::get;
  using std::make_tuple;

  //unsigned int dtsize = sc::size_of(dtype);
  sc::driver::CommandQueue & queue = sc::driver::backend::queues::get(sc::driver::backend::contexts::get_default(),0);
  auto sync = [&](){ queue.synchronize(); };
#ifdef BENCH_CUBLAS
  auto cusync = [&](){ cudaDeviceSynchronize(); };
#endif
  /*---------*/
  /*--BLAS1--*/
  /*---------*/

  if(operation=="axpy")
  {
    float alpha = 1;
    for(int_t N: create_log_range((int)1e3, (int)1e8, 50, 64))
    {
      std::vector<double> times;
      sc::array x(N, dtype), y(N, dtype);
      //Bench
      times.push_back(bench([&](){y = x + alpha*y;}, sync));
#ifdef BENCH_CLBLAS
      if(x.context().backend()==sc::driver::OPENCL)
        times.push_back(bench([&]() {clblasSaxpy(N, alpha, cl(x), 0, 1, cl(y), 0, 1, 1, &cl(queue), 0, NULL, NULL);}, sync));
#endif
#ifdef BENCH_CBLAS
      std::vector<float> cx(N), cy(N);
      sc::copy(x, cx);
      sc::copy(y, cy);
      times.push_back(bench([&](){cblas_saxpy(N, alpha, cx.data(), 1, cy.data(), 1);}, sync));
#endif
#ifdef BENCH_CUBLAS
      times.push_back(bench([&](){cublasSaxpy(N, alpha, (T*)cu(x), 1, (T*)cu(y), 1);}, cusync));
#endif
    }
  }

  if(operation=="dot")
  {
    for(int_t N: create_log_range((int)1e3, (int)1e8, 50, 64))
    {
      std::vector<double> times;
      sc::array x(N, dtype), y(N, dtype);
      sc::array scratch(N, dtype);
      sc::scalar s(dtype);
      //Bench
      times.push_back(bench([&](){s = dot(x,y);}, sync));
#ifdef BENCH_CLBLAS
      if(x.context().backend()==sc::driver::OPENCL)
        times.push_back(bench([&]() {clblasSdot(N, cl(s), 0, cl(x), 0, 1, cl(y), 0, 1, cl(scratch), 1, &cl(queue), 0, NULL, NULL);}, sync));
#endif
#ifdef BENCH_CBLAS
      std::vector<float> cx(N), cy(N);
      sc::copy(x, cx);
      sc::copy(y, cy);
      times.push_back(bench([&](){cblas_sdot(N, cx.data(), 1, cy.data(), 1);}, sync));
#endif
#ifdef BENCH_CUBLAS
      times.push_back(bench([&](){cublasSdot(N, (T*)cu(x), 1, (T*)cu(y), 1);}, cusync));
#endif
    }
  }

  if(operation.substr(0, 4)=="gemv")
  {
    std::vector<std::tuple<std::string, char,int_t, int_t> > MNs;
    //Linear System
    MNs.push_back(make_tuple("square153[N]", 'N',153,153));
    MNs.push_back(make_tuple("square153[T]", 'T',153,153));
    MNs.push_back(make_tuple("square1024[T]", 'T',1024,1024));
    MNs.push_back(make_tuple("square2867[N]", 'N',2867,2867));
    MNs.push_back(make_tuple("square2867[T]", 'T',2867,2867));
    //Normalization
    MNs.push_back(make_tuple("norm64[N]", 'N', 64, 60000));
    MNs.push_back(make_tuple("norm64[T]", 'T', 64, 60000));
    MNs.push_back(make_tuple("norm256[N]", 'N', 256, 60000));
    MNs.push_back(make_tuple("norm256[T]", 'T', 256, 60000));
    MNs.push_back(make_tuple("norm1024[N]", 'N', 1024, 60000));
    MNs.push_back(make_tuple("norm1024[T]", 'T', 1024, 60000));
    //Householder
    MNs.push_back(make_tuple("tallskinny-1[N]", 'N', 10, 60000));
    MNs.push_back(make_tuple("tallskinny-1[T]", 'T', 10, 60000));
    MNs.push_back(make_tuple("tallskinny-2[N]", 'N', 30, 60000));
    MNs.push_back(make_tuple("tallskinny-2[T]", 'T', 30, 60000));

    /*---------*/
    /*--BLAS2--*/
    /*---------*/
    for(std::tuple<std::string, char, int_t, int_t> MN: MNs)
    {
      std::vector<double> times;
      bool AT = get<1>(MN) == 'T';
      int_t M = get<2>(MN);
      int_t N = get<3>(MN);
      int_t As1 = M, As2 = N;
      if(AT) std::swap(As1, As2);
      sc::array A(As1, As2, dtype), y(M, dtype), x(N, dtype);
#ifdef HAS_A_BLAS
      int_t lda = A.stride()[1];
#endif
      //Bench
      times.push_back(bench([&](){y = AT?dot(A.T,x):dot(A,x);}, sync));
#ifdef BENCH_CLBLAS
      if(x.context().backend()==sc::driver::OPENCL)
        times.push_back(bench([&]() {clblasSgemv(clblasColumnMajor, AT?clblasTrans:clblasNoTrans, As1, As2, 1, cl(A), 0, lda, cl(x), 0, 1, 0, cl(y), 0, 1, 1, &cl(queue),0, NULL, NULL);}, sync));
#endif
#ifdef BENCH_CBLAS
      std::vector<float> cA(M*N), cx(N), cy(M);
      sc::copy(x, cx);
      sc::copy(y, cy);
      sc::copy(A, cA);
      times.push_back(bench([&](){cblas_sgemv(CblasColMajor, AT?CblasTrans:CblasNoTrans, As1, As2, 1, cA.data(), lda, cx.data(), 1, 0, cy.data(), 1);}, sync));
#endif
#ifdef BENCH_CUBLAS
      times.push_back(bench([&](){cublasSgemv(AT?'t':'n', As1, As2, 1, (T*)cu(A), lda, (T*)cu(x), 1, 0, (T*)cu(y), 1);}, cusync));
#endif
    }
  }

  if(operation.substr(0,4)=="gemm")
  {
    std::vector<std::tuple<std::string, int_t, int_t, int_t, char, char> > MNKs;
    //DeepBench
    for(size_t MK: std::vector<size_t>{1760, 2048, 2560})
      for(size_t N: std::vector<size_t>{16, 32, 64, 128, 7000})
        MNKs.push_back(make_tuple("Deep", MK, N, MK, 'N', 'N'));
    for(size_t MK: std::vector<size_t>{1760, 2048, 2560})
      for(size_t N: std::vector<size_t>{16, 32, 64, 128, 7000})
        MNKs.push_back(make_tuple("Deep", MK, N, MK, 'T', 'N'));
    for(size_t MK: std::vector<size_t>{1760, 4096})
      MNKs.push_back(make_tuple("Deep", MK, 7133, MK, 'N', 'T'));
    //Covariance (e.g., ICA, 10minutes/100Hz)
    MNKs.push_back(make_tuple("Cov",32,32,60000,'N','T'));
    MNKs.push_back(make_tuple("Cov",256,256,60000,'N','T'));
    //Bi-diagonalization
    MNKs.push_back(make_tuple("Lapack",4096,4096,32,'N','T'));
    MNKs.push_back(make_tuple("Lapack",3456,3456,32,'N','T'));
    MNKs.push_back(make_tuple("Lapack",896,896,32,'N','T'));

    std::cout << color_stream(ITALIC) << color_stream(BOLD) ;
    std::cout << "BENCH\tM\tN\tK\tAT\tBT\tISAAC";
#ifdef BENCH_CLBLAS
    std::cout << "\tclBLAS";
#endif
#ifdef BENCH_CBLAS
    std::cout << "\tBLAS";
#endif
#ifdef BENCH_CUBLAS
    std::cout << "\tcuBLAS";
#endif
    std::cout << color_stream(RESET) << std::endl;

    /*---------*/
    /*--BLAS3--*/
    /*---------*/
    for(auto MNK: MNKs)
    {
      std::vector<double> times;
      std::vector<double> tflops;
      std::string name = get<0>(MNK);
      int_t M = get<1>(MNK);
      int_t N = get<2>(MNK);
      int_t K = get<3>(MNK);
      char cAT = get<4>(MNK);
      char cBT = get<5>(MNK);
      bool AT = cAT=='T';
      bool BT = cBT=='T';
      int_t As1 = M, As2 = K;
      if(AT) std::swap(As1, As2);
      int_t Bs1 = K, Bs2 = N;
      if(BT) std::swap(Bs1, Bs2);
      sc::array C(M, N, dtype), A(As1, As2, dtype), B(Bs1, Bs2, dtype);
#ifdef HAS_A_BLAS
      int_t lda = A.stride()[1], ldb = B.stride()[1], ldc = C.stride()[1];
#endif
      //bench
      times.push_back(bench([&](){C = AT?(BT?dot(A.T,B.T)
                                            :dot(A.T,B))
                                        :(BT?dot(A,B.T)
                                            :dot(A,B));}, sync));
#ifdef BENCH_CLBLAS
      if(C.context().backend()==sc::driver::OPENCL)
        times.push_back(bench([&]() {clblasSgemm(clblasColumnMajor, AT?clblasTrans:clblasNoTrans, BT?clblasTrans:clblasNoTrans,
                                                 M, N, K, 1, cl(A), 0, lda, cl(B), 0, ldb,
                                                 0, cl(C), 0, ldc, 1, &cl(queue),0, NULL, NULL);}, sync));
#endif
#ifdef BENCH_CBLAS
      std::vector<float> cC(M*N), cA(M*K), cB(N*K);
      sc::copy(C, cC);
      sc::copy(A, cA);
      sc::copy(B, cB);
      times.push_back(bench([&](){cblas_sgemm(CblasColMajor, AT?CblasTrans:CblasNoTrans, BT?CblasTrans:CblasNoTrans, M, N, K, 1, cA.data(), lda, cB.data(), ldb, 1, cC.data(), ldc);}, sync));
#endif
#ifdef BENCH_CUBLAS
      times.push_back(bench([&](){cublasSgemm(AT?'t':'n', BT?'t':'n', M, N, K, 1, (T*)cu(A), lda, (T*)cu(B), ldb, 1, (T*)cu(C), ldc);}, cusync));
#endif
      std::transform(times.begin(), times.end(), std::back_inserter(tflops), [&](double t){ return 2*M*N*K/t*1e-3;});
      auto fastest = tflops;
      std::sort(fastest.begin(), fastest.end(), std::greater<double>());
      std::cout << name << "\t" << M << "\t" << N << "\t" << K << "\t" << cAT << "\t" << cBT;
     for(auto x: tflops){
        std::cout << "\t";
        if(x/fastest[1] >= 1.05)
          std::cout << color_stream(FG_LIGHT_BLUE) << x << color_stream(RESET);
        else
          std::cout << x;
      }
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
