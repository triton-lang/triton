#include <tuple>
#include "opts.hpp"
#include "isaac/driver/backend.h"
#include "isaac/driver/cublas.h"
#include "isaac/driver/context.h"
#include "isaac/driver/buffer.h"
#include "isaac/driver/stream.h"
#include "isaac/tools/bench.hpp"
#include "isaac/api.h"

namespace sc = isaac;
namespace drv = sc::driver;
using sc::param_t;
using std::make_tuple;

double geometric_mean(std::vector<double> const&data){
  double logsum = std::accumulate(data.begin(), data.end(),
                                  (double)0, [](double acc, double x){ return acc + std::log(x);});
  return std::exp(logsum/data.size());
}

void print_results_header(std::vector<std::string> sections){
    std::cout << color_stream(ITALIC) << color_stream(BOLD) ;
    std::copy(sections.begin(), sections.end(), std::ostream_iterator<std::string>(std::cout, "\t"));
    std::cout << "ISAAC\tcuDNN";
    std::cout << color_stream(RESET) << std::endl;
}

void print_results(std::vector<double> const & times, std::vector<std::string> const & prefix, std::function<bool(double, double)> cmp, std::function<double(double)> fn){
    std::copy(prefix.begin(), prefix.end(), std::ostream_iterator<std::string>(std::cout, "\t"));
    std::vector<double> perf;
    std::transform(times.begin(), times.end(), std::back_inserter(perf), fn);
    auto fastest = perf;
    std::sort(fastest.begin(), fastest.end(), cmp);
    for(auto x: perf){
      if(std::max(x,fastest[1])/std::min(x, fastest[1]) >= 1.05)
        std::cout << color_stream(FG_LIGHT_BLUE) << x << color_stream(RESET);
      else
        std::cout << x;
      std::cout << "\t";
    }
    std::cout << std::endl;
}

struct Metric{
  virtual std::function<bool(double, double)> cmp() const = 0;
  virtual double conv(param_t P, param_t Q, param_t K, param_t N, param_t C, param_t R, param_t S, double tsec) const = 0;
  virtual double gemm(param_t M, param_t N, param_t K, double tsec) const = 0;
};

class FLOPS: public Metric{
public:
  FLOPS(double scale): scale_(scale){}
  std::function<bool(double, double)> cmp() const { return std::greater<double>(); }
  double conv(param_t P, param_t Q, param_t K, param_t N, param_t C, param_t R, param_t S, double tsec) const
  { return  sc::templates::Conv::tflops(P,Q,K,N,C,R,S,tsec) * 1e12 / scale_; }
  double gemm(param_t M, param_t N, param_t K, double tsec) const
  { return  sc::templates::GEMM::tflops(M, N, K, tsec) * 1e12 / scale_; }
private:
  double scale_;
};

class Time: public Metric{
public:
  Time(double scale): scale_(scale){}
  std::function<bool(double, double)> cmp() const { return std::less<double>(); }
  double conv(param_t, param_t, param_t, param_t, param_t, param_t, param_t, double tsec) const { return tsec*1e-9/scale_; }
  double gemm(param_t, param_t, param_t, double tsec) const { return tsec*1e-9/scale_; }
private:
  double scale_;
};


int main(int argc, char* argv[])
{
  std::cout << std::fixed << std::setprecision(2);

  opts::Application program("bench", "benchmarking suite for ISAAC");
  program.add<sc::DType>("dtype", "data-type", "float32", {{"float16", sc::HALF_TYPE}, {"float32", sc::FLOAT_TYPE}, {"float64", sc::DOUBLE_TYPE}});
  program.add("conv", "benchmark CONV", true);
  program.add("gemm", "benchmark GEMM", true);
  program.add<std::shared_ptr<Metric>>("metric", "performance metric for the results", "tflops", {{"tflops", std::make_shared<FLOPS>(1e12)}, {"ms", std::make_shared<Time>(1e-3)}});
  program.add<size_t>("device", "device to run benchmarks on", 0);
  program.parse(argc, argv);

  //Get dtype
  sc::DType dtype = program.get<sc::DType>("dtype");
  size_t dtsize = sc::size_of(dtype);

  //Get device
  drv::Device device = drv::backend::devices()[program.get<size_t>("device")];
  drv::Context ctx(device);
  drv::Stream stream(ctx);

  //Get operations to benchmark
  bool bench_conv = program.get<bool>("conv");
  bool bench_gemm = program.get<bool>("gemm");

  //Get metric
  auto metric = program.get<std::shared_ptr<Metric>>("metric");

  //Benchmark convolution
  if(bench_conv)
  {
    typedef std::tuple<param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t> conv_tuple;
    std::vector<conv_tuple> shapes;
    //Cluster 1
    for(size_t N: std::vector<size_t>{4, 8, 16, 32})
      shapes.push_back(std::make_tuple(700, 161, 1, N, 32, 5, 20, 0, 0, 2, 2));
    //Cluster 2
    for(size_t N: std::vector<size_t>{4, 8, 16, 32})
      shapes.push_back(std::make_tuple(341, 79, 32, N, 32, 5, 10, 0, 0, 2, 2));
    //Cluster 3
    shapes.push_back(std::make_tuple(480, 48, 1, 16, 16, 3, 3, 1, 1, 1, 1));
    shapes.push_back(std::make_tuple(240, 24, 16, 16, 32, 3, 3, 1, 1, 1, 1));
    shapes.push_back(std::make_tuple(120, 12, 32, 16, 64, 3, 3, 1, 1, 1, 1));
    shapes.push_back(std::make_tuple(60, 6, 64, 16, 128, 3, 3, 1, 1, 1, 1));
    //Cluster 4
    shapes.push_back(std::make_tuple(108, 108, 3, 8, 64, 3, 3, 1, 1, 2, 2));
    shapes.push_back(std::make_tuple(54, 54, 64, 8, 64, 3, 3, 1, 1, 1, 1));
    shapes.push_back(std::make_tuple(27, 27, 128, 8, 128, 3, 3, 1, 1, 1, 1));
    shapes.push_back(std::make_tuple(14, 14, 128, 8, 256, 3, 3, 1, 1, 1, 1));
    shapes.push_back(std::make_tuple(7, 7, 256, 8, 512, 3, 3, 1, 1, 1, 1));
    //Cluster 5-6
    for(size_t N: std::vector<size_t>{8, 16}){
      shapes.push_back(std::make_tuple(224, 224, 3, N, 64, 3, 3, 1, 1, 1, 1));
      shapes.push_back(std::make_tuple(112, 112, 64, N, 128, 3, 3, 1, 1, 1, 1));
      shapes.push_back(std::make_tuple(56, 56, 128, N, 256, 3, 3, 1, 1, 1, 1));
      shapes.push_back(std::make_tuple(28, 28, 256, N, 512, 3, 3, 1, 1, 1, 1));
      shapes.push_back(std::make_tuple(14, 14, 512, N, 512, 3, 3, 1, 1, 1, 1));
      shapes.push_back(std::make_tuple(7, 7, 512, N, 512, 3, 3, 1, 1, 1, 1));
    }
    //Cluster 7
    shapes.push_back(std::make_tuple(224, 224, 3, 16, 64, 7, 7, 3, 3, 2, 2));
    shapes.push_back(std::make_tuple(28, 28, 192, 16, 32, 5, 5, 2, 2, 1, 1));
    shapes.push_back(std::make_tuple(28, 28, 192, 16, 64, 1, 1, 0, 0, 1, 1));
    shapes.push_back(std::make_tuple(14, 14, 512, 16, 48, 5, 5, 2, 2, 1, 1));
    shapes.push_back(std::make_tuple(14, 14, 512, 16, 192, 1, 1, 0, 0, 1, 1));
    shapes.push_back(std::make_tuple(7, 7, 832, 16, 256, 1, 1, 0, 0, 1, 1));
    shapes.push_back(std::make_tuple(7, 7, 832, 16, 128, 5, 5, 2, 2, 1, 1));

    param_t W, H, P, Q, C, N, K, R, S, pad_h, pad_w, stride_h, stride_w;
    std::cout << "======================================================================" << std::endl;
    std::cout << "FCONV" << std::endl;
    std::cout << "======================================================================" << std::endl;
    print_results_header({"N", "K", "P", "Q", "C", "R", "S"});
    std::vector<double> speedup;
    for(auto shape: shapes){
      std::tie(W, H, C, N, K, R, S, pad_h, pad_w, stride_h, stride_w) = shape;
      P = (H - R + 1 + 2*pad_h + stride_h - 1)/stride_h;
      Q = (W - S + 1 + 2*pad_w + stride_w - 1)/stride_w;

      sc::scalar alpha(1., dtype);
      sc::scalar beta(0., dtype);

      drv::Buffer O(ctx, N*K*P*Q*dtsize);
      drv::Buffer I(ctx, C*H*W*N*dtsize);
      drv::Buffer F(ctx, K*C*R*S*dtsize);

      std::vector<double> times;
      times.push_back(bench([&](){ sc::CONV(device, stream, dtype, N, K, P, Q, C, R, S, H, W, pad_h, pad_w, stride_h, stride_w, alpha, I, F, beta, O); }, [&](){ stream.synchronize(); }, device));
      if(sc::driver::dispatch::cudnninit())
        times.push_back(bench([&](){ sc::driver::cudnnConv(dtype, stream, H, W, N, K, P, Q, C, R, S, pad_h, pad_w, stride_h, stride_w, alpha, I, F, beta, O); }, [&](){ stream.synchronize();  }, device));
      speedup.push_back(times[1]/times[0]);
      print_results(times, {str(N), str(K), str(P), str(Q), str(C), str(R), str(S)}, metric->cmp(), [&](double tsec){ return metric->conv(P, Q, K, N, C, R, S, tsec);});
    }
    std::cout << "======================================================================" << std::endl;
    std::cout << "Speedup: " << geometric_mean(speedup) << std::endl;
    std::cout << std::endl;
  }

  //Benchmark GEMM
  if(bench_gemm)
  {
    typedef std::tuple<sc::IsaacOperation_t, sc::IsaacOperation_t, param_t, param_t, param_t> gemm_tuple;
    std::vector<gemm_tuple> shapes;

    // LinPack
    for(param_t N: std::vector<param_t>{512, 1024, 2048})
      shapes.push_back(std::make_tuple(sc::ISAAC_OP_N, sc::ISAAC_OP_T, N, N, N));

    // DeepBench
    for(sc::IsaacOperation_t AT: std::vector<sc::IsaacOperation_t>{sc::ISAAC_OP_N, sc::ISAAC_OP_T})
      for(param_t M: std::vector<param_t>{2560})
        for(param_t N: std::vector<param_t>{16, 32, 64, 128})
          shapes.push_back(std::make_tuple(AT, sc::ISAAC_OP_N, M, N, M));

    // OpenNMT
    //shapes.push_back(std::make_tuple(sc::ISAAC_OP_N, sc::ISAAC_OP_N, 2000, 128, 500));
    //shapes.push_back(std::make_tuple(sc::ISAAC_OP_N, sc::ISAAC_OP_N, 2000, 640, 500));
    //shapes.push_back(std::make_tuple(sc::ISAAC_OP_N, sc::ISAAC_OP_N, 2000, 2048, 500));
    //shapes.push_back(std::make_tuple(sc::ISAAC_OP_N, sc::ISAAC_OP_N, 2000, 640, 1000));
    //shapes.push_back(std::make_tuple(sc::ISAAC_OP_N, sc::ISAAC_OP_N, 500, 640, 1000));
    //shapes.push_back(std::make_tuple(sc::ISAAC_OP_N, sc::ISAAC_OP_N, 500, 640, 500));
    //shapes.push_back(std::make_tuple(sc::ISAAC_OP_N, sc::ISAAC_OP_N, 50000, 640, 500));


    // PCA/ICA
    for(param_t N: std::vector<param_t>{16, 64, 256})
      for(param_t K: std::vector<param_t>{64000})
        shapes.push_back(std::make_tuple(sc::ISAAC_OP_N, sc::ISAAC_OP_T, N, N, K));

    // LaPACK
    for(param_t N: std::vector<param_t>{1024, 2048, 4096})
      for(param_t K: std::vector<param_t>{32})
        shapes.push_back(std::make_tuple(sc::ISAAC_OP_N, sc::ISAAC_OP_T, N, N, K));

    sc::IsaacOperation_t AT, BT;
    param_t M, N, K;
    std::cout << "======================================================================" << std::endl;
    std::cout << "GEMM:" << std::endl;
    std::cout << "======================================================================" << std::endl;
    print_results_header({"AT", "BT", "M", "N", "K"});
    std::vector<double> speedup;
    for(auto shape: shapes){
      std::tie(AT, BT, M, N, K) = shape;
      sc::scalar alpha(1., dtype);
      sc::scalar beta(0., dtype);

      size_t ldc = M;
      size_t lda = (AT==sc::ISAAC_OP_N)?M:K;
      size_t ldb = (BT==sc::ISAAC_OP_N)?K:N;

      char cuAT = (AT==sc::ISAAC_OP_T)?'T':'N';
      char cuBT = (BT==sc::ISAAC_OP_T)?'T':'N';

      drv::Buffer C(ctx, M*N*dtsize);
      drv::Buffer A(ctx, M*K*dtsize);
      drv::Buffer B(ctx, K*N*dtsize);

      std::vector<double> times;
      times.push_back(bench([&](){ sc::GEMM(device, stream, dtype, AT, BT, M, N, K, 0, lda, 0, ldb, 0, ldc, alpha, A, B, beta, C); }, [&](){ stream.synchronize(); }, device));
      if(sc::driver::dispatch::cublasinit())
        times.push_back(bench([&](){ sc::driver::cublasGemm(dtype, stream, cuAT, cuBT, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, sc::driver::CUBLAS_PREFER_FASTEST); }, [&](){ stream.synchronize();  }, device));
      speedup.push_back(times[1]/times[0]);
      print_results(times, {str(AT), str(BT), str(M), str(N), str(K)}, metric->cmp(), [&](double tsec){ return metric->gemm(M, N, K, tsec);});
    }
    std::cout << "======================================================================" << std::endl;
    std::cout << "Speedup: " << geometric_mean(speedup) << std::endl;
  }
}
