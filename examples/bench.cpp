#include <tuple>
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

void handle_misusage(){
  std::cerr << "Usage : blas-bench [--dtype {float16, float32, float64}]" << std::endl;
  std::cerr << "--dtype: data-type to benchmark (default = float32)" << std::endl;
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

void print_results_header(std::vector<std::string> sections){
    std::cout << color_stream(ITALIC) << color_stream(BOLD) ;
    std::copy(sections.begin(), sections.end(), std::ostream_iterator<std::string>(std::cout, "\t"));
    std::cout << "ISAAC\tcuDNN";
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


int main(int argc, char* argv[])
{
  std::vector<std::string> args(argv, argv + argc);
  std::cout << std::fixed << std::setprecision(2);

  //Get dtype
  static std::map<std::string, sc::DType> sc_dtype = {{"float16", sc::HALF_TYPE}, {"float32", sc::FLOAT_TYPE}, {"float64", sc::DOUBLE_TYPE}};
  sc::DType dtype = sc_dtype[getopt(args, "--dtype", {"float16", "float32", "float64"}, "float32")];
  int32_t dtsize = sc::size_of(dtype);

  //Get device
  auto ctx = drv::backend::contexts::get_default();
  drv::Device const & device = drv::backend::contexts::get_default().device();
  drv::Stream stream(ctx);

  //Benchmark convolution
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
      P = (H - R + 1 + 2*pad_h)/stride_h;
      Q = (W - S + 1 + 2*pad_w)/stride_w;

      sc::scalar alpha(1., dtype);
      sc::scalar beta(0., dtype);

      drv::Buffer O(ctx, N*K*P*Q*dtsize);
      drv::Buffer I(ctx, C*H*W*N*dtsize);
      drv::Buffer F(ctx, K*C*R*S*dtsize);

      std::vector<double> times;
//      times.push_back(bench([&](){ sc::CONV(device, stream, dtype, N, K, P, Q, C, R, S, H, W, pad_h, pad_w, stride_h, stride_w, alpha, I, F, beta, O); }, [&](){ stream.synchronize(); }, device));
      times.push_back(bench([&](){ sc::driver::cudnnConv(dtype, stream, H, W, N, K, P, Q, C, R, S, pad_h, pad_w, stride_h, stride_w, alpha, I, F, beta, O); }, [&](){ stream.synchronize();  }, device));
      speedup.push_back(times[1]/times[0]);
      print_results(times, {str(N), str(K), str(P), str(Q), str(C), str(R), str(S)}, [&](double tsec){ return sc::templates::Conv::tflops(P,Q,K,N,C,R,S,tsec);});
    }
    std::cout << "======================================================================" << std::endl;
    std::cout << "Speedup: " << geometric_mean(speedup) << std::endl;
    std::cout << std::endl;
  }

  //Benchmark GEMM
  {
    typedef std::tuple<sc::IsaacOperation_t, sc::IsaacOperation_t, param_t, param_t, param_t> gemm_tuple;
    std::vector<gemm_tuple> shapes;

    // LinPack
    for(param_t N: std::vector<param_t>{512, 1024, 2048})
      shapes.push_back(std::make_tuple(sc::ISAAC_OP_N, sc::ISAAC_OP_T, N, N, N));

    // DeepBench [Forward]
    for(param_t M: std::vector<param_t>{1760})
      for(param_t N: std::vector<param_t>{8, 16, 32, 64, 128})
        shapes.push_back(std::make_tuple(sc::ISAAC_OP_N, sc::ISAAC_OP_N, M, N, M));

    // DeepBench [Backward]
    for(param_t M: std::vector<param_t>{1760})
      for(param_t N: std::vector<param_t>{8, 16, 32, 64, 128})
        shapes.push_back(std::make_tuple(sc::ISAAC_OP_T, sc::ISAAC_OP_N, M, N, M));

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
      times.push_back(bench([&](){ sc::driver::cublasGemm(dtype, stream, cuAT, cuBT, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }, [&](){ stream.synchronize();  }, device));
      speedup.push_back(times[1]/times[0]);
      print_results(times, {str(AT), str(BT), str(M), str(N), str(K)}, [&](double tsec){ return sc::templates::GEMM::tflops(M, N, K, tsec);});
    }
    std::cout << "======================================================================" << std::endl;
    std::cout << "Speedup: " << geometric_mean(speedup) << std::endl;
  }
}
