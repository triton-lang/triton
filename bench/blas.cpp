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
#include <string>
#include "common.hpp"
#include "half.hpp"

typedef sc::int_t int_t;

Timer tmr;

/* C++ wrapper for BLAS */
#ifdef BENCH_CLBLAS
template<typename... Args> void clblasAxpy(float, Args... args){ clblasSaxpy(args...); }
template<typename... Args> void clblasAxpy(double, Args... args){ clblasDaxpy(args...); }
template<typename... Args> void clblasDot(float, Args... args){ clblasSdot(args...); }
template<typename... Args> void clblasDot(double, Args... args){ clblasDdot(args...); }
template<typename... Args> void clblasGemv(float, Args... args){ clblasSgemv(args...); }
template<typename... Args> void clblasGemv(double, Args... args){ clblasDgemv(args...); }
template<typename... Args> void clblasGemm(float, Args... args){ clblasSgemm(args...); }
template<typename... Args> void clblasGemm(double, Args... args){ clblasDgemm(args...); }
#endif

#ifdef BENCH_CBLAS
template<typename... Args> void cblasAxpy(float, Args... args){ cblas_saxpy(args...); }
template<typename... Args> void cblasAxpy(double, Args... args){ cblas_daxpy(args...); }
template<typename... Args> void cblasDot(float, Args... args){ cblas_sdot(args...); }
template<typename... Args> void cblasDot(double, Args... args){ cblas_ddot(args...); }
template<typename... Args> void cblasGemv(float, Args... args){ cblas_sgemv(args...); }
template<typename... Args> void cblasGemv(double, Args... args){ cblas_dgemv(args...); }
template<typename... Args> void cblasGemm(float, Args... args){ cblas_sgemm(args...); }
template<typename... Args> void cblasGemm(double, Args... args){ cblas_dgemm(args...); }
#endif

//cuBLAS
#ifdef BENCH_CUBLAS
template<typename... Args> void cublasAxpy(float, Args... args){ cublasSaxpy(args...); }
template<typename... Args> void cublasAxpy(double, Args... args){ cublasDaxpy(args...); }
template<typename... Args> void cublasDot(float, Args... args){ cublasSdot(args...); }
template<typename... Args> void cublasDot(double, Args... args){ cublasDdot(args...); }
template<typename... Args> void cublasGemv(float, Args... args){ cublasSgemv(args...); }
template<typename... Args> void cublasGemv(double, Args... args){ cublasDgemv(args...); }
template<typename... Args> void cublasGemm(float, Args... args){ cublasSgemm(args...); }
template<typename... Args> void cublasGemm(double, Args... args){ cublasDgemm(args...); }
#endif

//
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

void print_results_header(std::vector<std::string> sections, bool
                          #ifdef BENCH_CLBLAS
                          on_cl
                          #endif
                          , bool
                          #ifdef BENCH_CUBLAS
                          on_cu
                          #endif
                          ){
    std::cout << color_stream(ITALIC) << color_stream(BOLD) ;
    std::copy(sections.begin(), sections.end(), std::ostream_iterator<std::string>(std::cout, "\t"));
    std::cout << "ISAAC";
#ifdef BENCH_CLBLAS
    if(on_cl)
    std::cout << "\tclBLAS";
#endif
#ifdef BENCH_CBLAS
    std::cout << "\tBLAS";
#endif
#ifdef BENCH_CUBLAS
    if(on_cu)
    std::cout << "\tcuBLAS";
#endif
    std::cout << color_stream(RESET) << std::endl;
}

void print_results(std::vector<double> const & times, std::vector<std::string> const & prefix, std::function<double(double)> fn){
    std::copy(prefix.begin(), prefix.end(), std::ostream_iterator<std::string>(std::cout, "\t"));
    std::vector<double> perf;
    std::transform(times.begin(), times.end(), std::back_inserter(perf), fn);
    auto fastest = perf;
    std::sort(fastest.begin(), fastest.end(), std::greater<double>());
    for(auto x: perf){
      if(x/fastest[1] >= 1.05)
        std::cout << color_stream(FG_LIGHT_BLUE) << x << color_stream(RESET);
      else
        std::cout << x;
      std::cout << "\t";
    }
    std::cout << std::endl;
}

template<class T>
std::string str(T const & x){ return std::to_string(x); }

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

  bool on_cl = queue.backend()==sc::driver::OPENCL;
  bool on_cu = queue.backend()==sc::driver::CUDA;
  size_t dtsize = sc::size_of(dtype);
  /*---------*/
  /*--BLAS1--*/
  /*---------*/

  if(operation=="axpy")
  {
    float alpha = 1;
    print_results_header({"N"}, on_cl, on_cu);
    for(int_t MB: std::vector<int_t>{1, 10, 100, 1000})
    {
      int_t N = MB*1e6/dtsize/3;
      std::vector<double> times;
      sc::array x(N, dtype), y(N, dtype);
      //Bench
      times.push_back(bench([&](){y = x + alpha*y;}, sync));
#ifdef BENCH_CLBLAS
      if(on_cl)
        times.push_back(bench([&]() {clblasAxpy(T(), N, alpha, cl(x), 0, 1, cl(y), 0, 1, 1, &cl(queue), 0, nullptr, nullptr);}, sync));
#endif
#ifdef BENCH_CBLAS
      std::vector<float> cx(N), cy(N);
      sc::copy(x, cx);
      sc::copy(y, cy);
      times.push_back(bench([&](){cblasAxpy(T(), N, alpha, cx.data(), 1, cy.data(), 1);}, sync));
#endif
#ifdef BENCH_CUBLAS
      if(on_cu)
        times.push_back(bench([&](){cublasAxpy(T(), N, alpha, (T*)cu(x), 1, (T*)cu(y), 1);}, cusync));
#endif
      print_results(times, {str(MB)}, [&](double t){return MB*1e6/t;});
    }
  }

  if(operation=="dot")
  {
    print_results_header({"MB"}, on_cl, on_cu);
    for(int_t MB: std::vector<int_t>{1, 10, 100, 1000})
    {
      int_t N = MB*1e6/dtsize/2;
      std::vector<double> times;
      sc::array x(N, dtype), y(N, dtype);
      sc::array scratch(N, dtype);
      sc::scalar s(dtype);
      //Bench
      times.push_back(bench([&](){s = dot(x,y);}, sync));
#ifdef BENCH_CLBLAS
      if(on_cl)
        times.push_back(bench([&]() {clblasDot(T(), N, cl(s), 0, cl(x), 0, 1, cl(y), 0, 1, cl(scratch), 1, &cl(queue), 0, nullptr, nullptr);}, sync));
#endif
#ifdef BENCH_CBLAS
      std::vector<float> cx(N), cy(N);
      sc::copy(x, cx);
      sc::copy(y, cy);
      times.push_back(bench([&](){cblasDot(T(), N, cx.data(), 1, cy.data(), 1);}, sync));
#endif
#ifdef BENCH_CUBLAS
      if(on_cu)
        times.push_back(bench([&](){cublasDot(T(), N, (T*)cu(x), 1, (T*)cu(y), 1);}, cusync));
#endif
      print_results(times, {str(MB)}, [&](double t){return MB*1e6/t;});
    }
  }

  if(operation.substr(0, 4)=="gemv")
  {
    std::vector<std::tuple<std::string, std::string,int_t, int_t> > MNs;
    //Linear System
    MNs.push_back(make_tuple("Square", "N",153,153));
    MNs.push_back(make_tuple("Square", "N",1024, 1024));
    MNs.push_back(make_tuple("Square", "N",2867,2867));
    MNs.push_back(make_tuple("Square", "T",153,153));
    MNs.push_back(make_tuple("Square", "T",1024,1024));
    MNs.push_back(make_tuple("Square", "T",2867,2867));
    //Normalization
    MNs.push_back(make_tuple("Short", "N", 64, 60000));
    MNs.push_back(make_tuple("Short", "N", 256, 60000));
    MNs.push_back(make_tuple("Short", "N", 1024, 60000));
    MNs.push_back(make_tuple("Short", "T", 64, 60000));
    MNs.push_back(make_tuple("Short", "T", 256, 60000));
    MNs.push_back(make_tuple("Short", "T", 1024, 60000));
    //Householder
    MNs.push_back(make_tuple("Tall", "N", 10, 60000));
    MNs.push_back(make_tuple("Tall", "N", 30, 60000));
    MNs.push_back(make_tuple("Tall", "T", 10, 60000));
    MNs.push_back(make_tuple("Tall", "T", 30, 60000));

    /*---------*/
    /*--BLAS2--*/
    /*---------*/
    print_results_header({"BENCH", "M", "N", "AT"}, on_cl, on_cu);
    for(auto MN: MNs)
    {
      std::vector<double> times;
      std::string name = get<0>(MN);
      std::string cAT = get<1>(MN);
      int_t M = get<2>(MN);
      int_t N = get<3>(MN);
      int_t As1 = M, As2 = N;
      bool AT = (cAT == "T");
      if(AT) std::swap(As1, As2);
      sc::array A(As1, As2, dtype), y(M, dtype), x(N, dtype);
#ifdef HAS_A_BLAS
      int_t lda = A.stride()[1];
#endif
      //Bench
      times.push_back(bench([&](){y = AT?dot(A.T,x):dot(A,x);}, sync));
#ifdef BENCH_CLBLAS
      if(on_cl)
        times.push_back(bench([&]() {clblasGemv(T(), clblasColumnMajor, AT?clblasTrans:clblasNoTrans, As1, As2, 1, cl(A), 0, lda, cl(x), 0, 1, 0, cl(y), 0, 1, 1, &cl(queue),0, nullptr, nullptr);}, sync));
#endif
#ifdef BENCH_CBLAS
      std::vector<float> cA(M*N), cx(N), cy(M);
      sc::copy(x, cx);
      sc::copy(y, cy);
      sc::copy(A, cA);
      times.push_back(bench([&](){cblasGemv(T(), CblasColMajor, AT?CblasTrans:CblasNoTrans, As1, As2, 1, cA.data(), lda, cx.data(), 1, 0, cy.data(), 1);}, sync));
#endif
#ifdef BENCH_CUBLAS
      if(on_cu)
        times.push_back(bench([&](){cublasGemv(T(), AT?'t':'n', As1, As2, 1, (T*)cu(A), lda, (T*)cu(x), 1, 0, (T*)cu(y), 1);}, cusync));
#endif
      print_results(times, {name, str(M), str(N), cAT}, [&](double t){ return (M*N + M + N)*dtsize/t;});
    }
  }

  if(operation.substr(0,4)=="gemm")
  {
    std::vector<std::tuple<std::string, int_t, int_t, int_t, std::string, std::string> > MNKs;
    //DeepBench
    for(size_t MK: std::vector<size_t>{1760, 2048, 2560})
      for(size_t N: std::vector<size_t>{16, 32, 64, 128, 7000})
        MNKs.push_back(make_tuple("Deep", MK, N, MK, "N", "N"));
    for(size_t MK: std::vector<size_t>{1760, 2048, 2560})
      for(size_t N: std::vector<size_t>{16, 32, 64, 128, 7000})
        MNKs.push_back(make_tuple("Deep", MK, N, MK, "T", "N"));
    for(size_t MK: std::vector<size_t>{1760, 4096})
      MNKs.push_back(make_tuple("Deep", MK, 7133, MK, "N", "T"));
    //Covariance (e.g., ICA, 10minutes/100Hz)
    MNKs.push_back(make_tuple("Cov",32,32,60000,"N","T"));
    MNKs.push_back(make_tuple("Cov",256,256,60000,"N","T"));
    //Bi-diagonalization
    MNKs.push_back(make_tuple("Lapack",4096,4096,32,"N","T"));
    MNKs.push_back(make_tuple("Lapack",3456,3456,32,"N","T"));
    MNKs.push_back(make_tuple("Lapack",896,896,32,"N","T"));

    print_results_header({"BENCH", "M", "N", "K", "AT", "BT"}, on_cl, on_cu);
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
      std::string cAT = get<4>(MNK);
      std::string cBT = get<5>(MNK);
      bool AT = cAT=="T";
      bool BT = cBT=="T";
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
      if(on_cl)
        times.push_back(bench([&]() {clblasGemm(T(), clblasColumnMajor, AT?clblasTrans:clblasNoTrans, BT?clblasTrans:clblasNoTrans,
                                                 M, N, K, 1, cl(A), 0, lda, cl(B), 0, ldb,
                                                 0, cl(C), 0, ldc, 1, &cl(queue),0, nullptr, nullptr);}, sync));
#endif
#ifdef BENCH_CBLAS
      std::vector<float> cC(M*N), cA(M*K), cB(N*K);
      sc::copy(C, cC);
      sc::copy(A, cA);
      sc::copy(B, cB);
      times.push_back(bench([&](){cblasGemm(T(), CblasColMajor, AT?CblasTrans:CblasNoTrans, BT?CblasTrans:CblasNoTrans, M, N, K, 1, cA.data(), lda, cB.data(), ldb, 1, cC.data(), ldc);}, sync));
#endif
#ifdef BENCH_CUBLAS
      if(on_cu)
        times.push_back(bench([&](){cublasGemm(T(), AT?'t':'n', BT?'t':'n', M, N, K, 1, (T*)cu(A), lda, (T*)cu(B), ldb, 0, (T*)cu(C), ldc);}, cusync));
#endif
      print_results(times, {name, str(M), str(N), str(K), cAT, cBT}, [&](double t){ return 2*M*N*K/t*1e-3;});
    }
  }

}

void handle_misusage(){
  std::cerr << "Usage : blas-bench [--op {axpy, dot, gemv, gemm}] [--dtype {float32, float64}] [--device DEVICE_IDX] [--help]" << std::endl;
  std::cerr << "--op: operation to benchmark (default = gemm)" << std::endl;
  std::cerr << "--dtype: data-type to benchmark (default = float32)" << std::endl;
  std::cerr << "--device: index of isaac device in [0, ..., ndevices - 1] (default = 0)" << std::endl;
  std::cerr << "--help: display this message" << std::endl;
  exit(EXIT_FAILURE);
}

std::string getopt(std::vector<std::string> const & args,
            std::string const & key,
            std::vector<std::string> const & set = {},
            std::string dft = "")
{
  auto it = std::find(args.begin(), args.end(), key);
  if(it==args.end()){
    if(dft.empty())
      handle_misusage();
    return dft;
  }
  auto next = it + 1;
  if(next==args.end() || next->compare(0, 2, "--")==0)
    handle_misusage();
  if(set.size() && std::find(set.begin(), set.end(), *next)==set.end())
    handle_misusage();
  return *next;
}

int main(int argc, char* argv[])
{
  std::vector<std::string> args(argv, argv + argc);
#ifdef BENCH_CLBLAS
  clblasSetup();
#endif
  sc::driver::backend::default_queue_properties = CL_QUEUE_PROFILING_ENABLE;

  if(std::find(args.begin(), args.end(), "--help") != args.end())
    handle_misusage();

  std::string operation = getopt(args, "--op", {"axpy", "dot", "gemv", "gemm"}, "gemm");
  std::string dtype = getopt(args, "--dtype", {"float16", "float32", "float64"}, "float32");
  int device;
  try{
    device = std::stoi(getopt(args, "--device", {}, "0"));
  }catch(...){ handle_misusage(); }
  sc::driver::backend::default_device = device;

  /* List devices */
  std::cout << "Devices available:" << std::endl;
  std::cout << "------------------" << std::endl;
  size_t i = 0;
  std::vector<sc::driver::Platform> platforms;
  sc::driver::backend::platforms(platforms);
  for(sc::driver::Platform const & pf: platforms){
    std::vector<sc::driver::Device> devices;
    pf.devices(devices);
    for(sc::driver::Device const & device: devices)
      std::cout << "[" << (i++==sc::driver::backend::default_device?"x":" ") << "]"
                << " - " << device.name()
                << " on " << pf.name() << std::endl;
  }
  std::cout << "------------------" << std::endl;

  std::cout << std::fixed << std::setprecision(2);
  //if(dtype=="float16")
  //  bench<half_float::half>(sc::HALF_TYPE, operation);
  if(dtype=="float32")
    bench<float>(sc::FLOAT_TYPE, operation);
  if(dtype=="float64")
    bench<double>(sc::DOUBLE_TYPE, operation);

#ifdef BENCH_CLBLAS
  clblasTeardown();
#endif
}
