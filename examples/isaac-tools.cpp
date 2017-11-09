#include "opts.hpp"
#include "isaac/scalar.h"
#include "isaac/api.h"
#include "isaac/driver/cublas.h"
#include "isaac/driver/backend.h"
#include "isaac/driver/context.h"
#include "isaac/driver/stream.h"
#include "isaac/runtime/predict.h"
#include "isaac/templates/gemm.h"
#include "isaac/templates/error.hpp"
#include "isaac/tools/bench.hpp"

namespace sc = isaac;
namespace drv = sc::driver;
using sc::param_t;

enum Code {
  RESET = 0,
  BOLD = 1,
  ITALIC = 3,
  FG_RED = 31,
  FG_GREEN = 32,
  FG_YELLOW = 33,
  FG_BLUE = 34,
  FG_MAGENTA = 35,
  FG_CYAN = 36,
  FG_LIGHT_GRAY = 37,
  FG_DARK_GRAY = 90,
  FG_LIGHT_RED = 91,
  FG_LIGHT_GREEN = 92,
  FG_LIGHT_YELLOW = 93,
  FG_LIGHT_BLUE = 94,
  FG_LIGHT_MAGENTA = 95,
  FG_LIGHT_CYAN = 96,
  FG_WHITE = 97
};

class color_stream {
    Code code;
public:
    color_stream(Code pCode) : code(pCode) {}
    friend std::ostream&
    operator<<(std::ostream& os, const color_stream& mod) {
        return os << "\033[" << mod.code << "m";
    }
};

/* Helpers for benchmarking */
typedef std::tuple<sc::DType, sc::IsaacOperation_t, sc::IsaacOperation_t, sc::param_t, sc::param_t, sc::param_t> gemm_params_t;
typedef std::tuple<sc::DType, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t> conv_params_t;
typedef std::tuple<sc::DType, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t> pool_params_t;

struct TestBench{
  // GEMM
  static std::vector<gemm_params_t> gemm(sc::DType dtype){
    std::vector<gemm_params_t> shapes;
    // LinPack
    for(param_t N: std::vector<param_t>{512, 1024, 2048})
      shapes.push_back(std::make_tuple(dtype, sc::ISAAC_OP_N, sc::ISAAC_OP_T, N, N, N));
    // DeepBench
    for(sc::IsaacOperation_t AT: std::vector<sc::IsaacOperation_t>{sc::ISAAC_OP_N, sc::ISAAC_OP_T})
      for(param_t M: std::vector<param_t>{1760})
        for(param_t N: std::vector<param_t>{16, 32, 64, 128})
          shapes.push_back(std::make_tuple(dtype, AT, sc::ISAAC_OP_N, M, N, M));
    // PCA/ICA
    for(param_t N: std::vector<param_t>{16, 64, 256})
      for(param_t K: std::vector<param_t>{64000})
        shapes.push_back(std::make_tuple(dtype, sc::ISAAC_OP_N, sc::ISAAC_OP_T, N, N, K));
    // LaPACK
    for(param_t N: std::vector<param_t>{1024, 2048, 4096})
      for(param_t K: std::vector<param_t>{32})
        shapes.push_back(std::make_tuple(dtype, sc::ISAAC_OP_N, sc::ISAAC_OP_T, N, N, K));
    return shapes;
  }

  // CONV
  static std::vector<conv_params_t> conv(sc::DType dtype){
    // Vector of (dtype, D, W, H, C, N, K, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w)

    std::vector<conv_params_t> shapes;
    // DeepSpeech
    for(size_t N: std::vector<size_t>{8})
      shapes.push_back(std::make_tuple(dtype, 1, 700, 161, 1, N, 32, 1, 5, 20, 0, 0, 0, 1, 1, 1));
    for(size_t N: std::vector<size_t>{8})
      shapes.push_back(std::make_tuple(dtype, 1, 341, 79, 32, N, 32, 1, 5, 10, 0, 0, 0, 1, 1, 1));

    // OCR
    shapes.push_back(std::make_tuple(dtype, 1, 480, 48, 1, 16, 16, 1, 3, 3, 0, 1, 1, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 1, 240, 24, 16, 16, 32, 1, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 1, 120, 12, 32, 16, 64, 1, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 1, 60, 6, 64, 16, 128, 1, 3, 3, 0, 1, 1, 1, 1, 1));

    // Face Recognition
    shapes.push_back(std::make_tuple(dtype, 1, 108, 108, 3, 8, 64, 1, 3, 3, 0, 1, 1, 1, 2, 2));
    shapes.push_back(std::make_tuple(dtype, 1, 54, 54, 64, 8, 64, 1, 3, 3, 0, 1, 1, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 1, 27, 27, 128, 8, 128, 1, 3, 3, 0, 1, 1, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 1, 14, 14, 128, 8, 256, 1, 3, 3, 0, 1, 1, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 1, 7, 7, 256, 8, 512, 1, 3, 3, 0, 1, 1, 1, 1, 1));

    // Vision
    for(size_t N: std::vector<size_t>{8}){
      shapes.push_back(std::make_tuple(dtype, 1, 224, 224, 3, N, 64, 1, 3, 3, 0, 1, 1, 1, 1, 1));
      shapes.push_back(std::make_tuple(dtype, 1, 112, 112, 64, N, 128, 1, 3, 3, 0, 1, 1, 1, 1, 1));
      shapes.push_back(std::make_tuple(dtype, 1, 56, 56, 128, N, 256, 1, 3, 3, 0, 1, 1, 1, 1, 1));
      shapes.push_back(std::make_tuple(dtype, 1, 28, 28, 256, N, 512, 1, 3, 3, 0, 1, 1, 1, 1, 1));
      shapes.push_back(std::make_tuple(dtype, 1, 14, 14, 512, N, 512, 1, 3, 3, 0, 1, 1, 1, 1, 1));
      shapes.push_back(std::make_tuple(dtype, 1, 7, 7, 512, N, 512, 1, 3, 3, 0, 1, 1, 1, 1, 1));
    }
    shapes.push_back(std::make_tuple(dtype, 1, 224, 224, 3, 16, 64, 1, 7, 7, 0, 3, 3, 1, 2, 2));
    shapes.push_back(std::make_tuple(dtype, 1, 28, 28, 192, 16, 32, 1, 5, 5, 0, 2, 2, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 1, 28, 28, 192, 16, 64, 1, 1, 1, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 1, 14, 14, 512, 16, 48, 1, 5, 5, 0, 2, 2, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 1, 14, 14, 512, 16, 192, 1, 1, 1, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 1, 7, 7, 832, 16, 256, 1, 1, 1, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 1, 7, 7, 832, 16, 128, 1, 5, 5, 0, 2, 2, 1, 1, 1));

    // Speaker ID
    shapes.push_back(std::make_tuple(dtype, 1, 350, 80, 64, 16, 128, 1, 5, 5, 0, 1, 1, 1, 2, 2));
    shapes.push_back(std::make_tuple(dtype, 1, 175, 40, 128, 16, 256, 1, 5, 5, 0, 1, 1, 1, 2, 2));

    // ResNET
    for(size_t N: std::vector<size_t>{8}){
      shapes.push_back(std::make_tuple(dtype, 1, 112, 112, 64, N, 64, 1, 1, 1, 0, 0, 0, 1, 1, 1));
      shapes.push_back(std::make_tuple(dtype, 1, 56, 56, 64, N, 256, 1, 1, 1, 0, 0, 0, 1, 1, 1));
      shapes.push_back(std::make_tuple(dtype, 1, 56, 56, 256, N, 64, 1, 1, 1, 0, 0, 0, 1, 1, 1));
      shapes.push_back(std::make_tuple(dtype, 1, 56, 56, 256, N, 128, 1, 1, 1, 0, 0, 0, 1, 2, 2));
      shapes.push_back(std::make_tuple(dtype, 1, 28, 28, 128, N, 512, 1, 1, 1, 0, 0, 0, 1, 1, 1));
      shapes.push_back(std::make_tuple(dtype, 1, 28, 28, 512, N, 128, 1, 1, 1, 0, 0, 0, 1, 1, 1));
      shapes.push_back(std::make_tuple(dtype, 1, 28, 28, 512, N, 256, 1, 1, 1, 0, 0, 0, 1, 2, 2));
      shapes.push_back(std::make_tuple(dtype, 1, 14, 14, 256, N, 1024, 1, 1, 1, 0, 0, 0, 1, 1, 1));
      shapes.push_back(std::make_tuple(dtype, 1, 28, 28, 512, N, 1024, 1, 1, 1, 0, 0, 0, 1, 2, 2));
      shapes.push_back(std::make_tuple(dtype, 1, 14, 14, 1024, N, 2048, 1, 1, 1, 0, 0, 0, 1, 2, 2));
      shapes.push_back(std::make_tuple(dtype, 1, 7, 7, 512, N, 512, 1, 3, 3, 0, 1, 1, 1, 1, 1));
      shapes.push_back(std::make_tuple(dtype, 1, 7, 7, 512, N, 2048, 1, 1, 1, 0, 1, 1, 1, 1, 1));
      shapes.push_back(std::make_tuple(dtype, 1, 14, 14, 1024, N, 2048, 1, 1, 1, 0, 1, 1, 1, 2, 2));
    }

    // 3D-Unet
    shapes.push_back(std::make_tuple(dtype, 31, 204, 204,   4, 1,  24, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 29, 202, 202,  24, 1,  24, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 27, 100, 100,  24, 1,  72, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 25,  98,  98,  72, 1,  72, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 23,  48,  48,  72, 1, 216, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 21,  46,  46, 216, 1, 216, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 19,  22,  22, 216, 1, 648, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 17,  20,  20, 648, 1, 648, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 15,  36,  36, 648, 1, 432, 1, 1, 1, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 13,  36,  36, 432, 1, 216, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 11,  34,  34, 216, 1, 216, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 11,  64,  64, 216, 1, 144, 1, 1, 1, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 11,  64,  64, 144, 1, 72,  3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 9 ,  62,  62,  72, 1, 72,  3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 7 , 120, 120,  72, 1, 48,  1, 1, 1, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 5 , 120, 120,  48, 1, 24,  3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 3 , 118, 118,  24, 1, 24,  3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 1 , 116, 116,  3 , 1, 24,  1, 1, 1, 0, 0, 0, 1, 1, 1));

    return shapes;
  }

  // POOL
  static std::vector<pool_params_t> pool(sc::DType dtype){
    std::vector<pool_params_t> shapes;

    // 3D-Unet
    shapes.push_back(std::make_tuple(dtype, 31, 204, 204, 4,  24, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 29, 202, 202, 4,  24, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 27, 100, 100, 4,  72, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 25,  98,  98, 4,  72, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 23,  48,  48, 4, 216, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 21,  46,  46, 4, 216, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 19,  22,  22, 4, 648, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 17,  20,  20, 4, 648, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 15,  36,  36, 4, 432, 1, 1, 1, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 13,  36,  36, 4, 216, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 11,  34,  34, 4, 216, 3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 11,  64,  64, 4, 144, 1, 1, 1, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 11,  64,  64, 4, 72,  3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 9 ,  62,  62, 4, 72,  3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 7 , 120, 120, 4, 48,  1, 1, 1, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 5 , 120, 120, 4, 24,  3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 3 , 118, 118, 4, 24,  3, 3, 3, 0, 0, 0, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 1 , 116, 116, 4, 24,  1, 1, 1, 0, 0, 0, 1, 1, 1));

    return shapes;
  }

};
/* Metrics for benchmarking */
struct Metric{
  virtual std::function<bool(double, double)> cmp() const = 0;
  virtual double conv(param_t P, param_t Q, param_t M, param_t K, param_t N, param_t C, param_t R, param_t S, param_t T, double tsec) const = 0;
  virtual double gemm(param_t M, param_t N, param_t K, double tsec) const = 0;
  virtual double pool(param_t P, param_t Q, param_t M, param_t K, param_t N, param_t, param_t, param_t, double tsec) const = 0;
};

class FLOPS: public Metric{
public:
  FLOPS(double scale): scale_(scale){}
  std::function<bool(double, double)> cmp() const { return std::greater<double>(); }
  double conv(param_t P, param_t Q, param_t M, param_t K, param_t N, param_t C, param_t R, param_t S, param_t T, double tsec) const
  { return  sc::templates::Conv::tflops(P,Q,M,K,N,C,R,S,T,tsec) * 1e12 / scale_; }
  double gemm(param_t M, param_t N, param_t K, double tsec) const
  { return  sc::templates::GEMM::tflops(M, N, K, tsec) * 1e12 / scale_; }
  double pool(param_t P, param_t Q, param_t M, param_t K, param_t N, param_t T, param_t R, param_t S, double tsec) const
  { return sc::templates::Pool::tflops(P, Q, M, K, N, T, R, S, tsec) * 1e12 / scale_;}

private:
  double scale_;
};

class Time: public Metric{
public:
  Time(double scale): scale_(scale){}
  std::function<bool(double, double)> cmp() const { return std::less<double>(); }
  double conv(param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, double tsec) const { return tsec*1e-9/scale_; }
  double gemm(param_t, param_t, param_t, double tsec) const { return tsec*1e-9/scale_; }
  double pool(param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, double tsec) const { return tsec*1e-9/scale_; }

private:
  double scale_;
};

void print_results_header(std::vector<std::string> sections){
    std::cout << color_stream(ITALIC) << color_stream(BOLD) ;
    std::copy(sections.begin(), sections.end(), std::ostream_iterator<std::string>(std::cout, "\t"));
    std::cout << color_stream(RESET) << std::endl;
}

void print_results(std::vector<double> const & times, std::vector<std::string> const & prefix, std::function<bool(double, double)> cmp, std::function<double(double)> fn){
    std::copy(prefix.begin(), prefix.end(), std::ostream_iterator<std::string>(std::cout, "\t"));
    std::vector<double> perf;
    std::transform(times.begin(), times.end(), std::back_inserter(perf), fn);
    auto fastest = perf;
    std::sort(fastest.begin(), fastest.end(), cmp);

    for(auto x: perf){
      if(x == fastest[0] && x / fastest[1] > 1.05)
        std::cout << color_stream(FG_LIGHT_BLUE) << x << color_stream(RESET);
      else
        std::cout << x;
      std::cout << "\t";
    }
    std::cout << std::endl;
}

void benchmark_gemm(Metric const & metric, sc::driver::Context& ctx, sc::driver::Device& device, sc::driver::Stream& stream,
                    sc::DType dtype, sc::IsaacOperation_t AT, sc::IsaacOperation_t BT, size_t M, size_t N, size_t K,
                    sc::templates::Generator* generator){
  size_t ldc = M;
  size_t lda = (AT==sc::ISAAC_OP_N)?M:K;
  size_t ldb = (BT==sc::ISAAC_OP_N)?K:N;

  size_t dtsize = sc::size_of(dtype);
  sc::scalar alpha(1., dtype);
  sc::scalar beta(0., dtype);
  char cuAT = (AT==sc::ISAAC_OP_T)?'T':'N';
  char cuBT = (BT==sc::ISAAC_OP_T)?'T':'N';

  sc::driver::Buffer C(ctx, M*N*dtsize);
  sc::driver::Buffer A(ctx, M*K*dtsize);
  sc::driver::Buffer B(ctx, K*N*dtsize);

  std::vector<double> times;
  times.push_back(bench([&](){ sc::GEMM(device, stream, dtype, AT, BT, M, N, K, 0, lda, 0, ldb, 0, ldc, alpha, A, B, beta, C, (sc::templates::GEMM*)generator); }, [&](){ stream.synchronize(); }, device));
  if(sc::driver::dispatch::cublasinit()){
    cublasGemmAlgo_t fastest;
    sc::driver::cublasGemm(dtype, stream, cuAT, cuBT, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, &fastest);
    times.push_back(bench([&](){ sc::driver::cublasGemm(dtype, stream, cuAT, cuBT, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, NULL, fastest); }, [&](){ stream.synchronize();  }, device));
    //times.push_back(bench([&](){ sc::driver::cublasGemm(dtype, stream, cuAT, cuBT, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }, [&](){ stream.synchronize();  }, device));
  }
  print_results(times, {str(AT), str(BT), str(M), str(N), str(K)}, metric.cmp(), [&](double tsec){ return metric.gemm(M, N, K, tsec);});
}

void benchmark_conv(Metric const & metric, sc::driver::Context& ctx, sc::driver::Device& device, sc::driver::Stream& stream,
                    sc::DType dtype, size_t D, size_t H, size_t W, size_t C, size_t N, size_t K, size_t T, size_t R, size_t S, size_t pad_d, size_t pad_h, size_t pad_w, size_t stride_d, size_t stride_h, size_t stride_w,
                    sc::templates::Generator* generator){

  param_t M, P, Q;
  sc::templates::Conv::output_shapes(D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, M, P, Q);

  size_t dtsize = sc::size_of(dtype);
  sc::scalar alpha(1., dtype);
  sc::scalar beta(0., dtype);

  sc::driver::Buffer O(ctx, N*K*M*P*Q*dtsize);
  sc::driver::Buffer I(ctx, C*D*H*W*N*dtsize);
  sc::driver::Buffer F(ctx, K*C*T*R*S*dtsize);

  std::vector<double> times;
  times.push_back(bench([&](){ sc::CONV(device, stream, dtype, N, K, M, P, Q, C, T, R, S, D, H, W, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, alpha, I, F, beta, O, (sc::templates::Conv*)generator); }, [&](){ stream.synchronize(); }, device));
  if(sc::driver::dispatch::cudnninit())
    times.push_back(bench([&](){ sc::driver::cudnnConv(dtype, stream, D, H, W, N, K, M, P, Q, C, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, alpha, I, F, beta, O); }, [&](){ stream.synchronize();  }, device));
  print_results(times, {str(N), str(K), str(M), str(P), str(Q), str(C), str(T), str(R), str(S)}, metric.cmp(), [&](double tsec){ return metric.conv(M, P, Q, K, N, C, T, R, S, tsec);});
}

void benchmark_pool(Metric const & metric, sc::driver::Context& ctx, sc::driver::Device& device, sc::driver::Stream& stream,
                    sc::DType dtype, size_t D, size_t H, size_t W, size_t N, size_t K, size_t T, size_t R, size_t S, size_t pad_d, size_t pad_h, size_t pad_w, size_t stride_d, size_t stride_h, size_t stride_w,
                    sc::templates::Generator* generator){

  param_t M, P, Q;
  sc::templates::Conv::output_shapes(D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, M, P, Q);

  size_t dtsize = sc::size_of(dtype);
  sc::scalar alpha(1., dtype);
  sc::scalar beta(0., dtype);

  sc::driver::Buffer O(ctx, N*K*M*P*Q*dtsize);
  sc::driver::Buffer I(ctx, K*D*H*W*N*dtsize);

  std::vector<double> times;
  times.push_back(bench([&](){ sc::POOL(device, stream, dtype, K, M, P, Q, N, T, R, S, D, H, W, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, I, O, (sc::templates::Pool*)generator); }, [&](){ stream.synchronize(); }, device));
  if(sc::driver::dispatch::cudnninit())
    times.push_back(bench([&](){ sc::driver::cudnnPool(dtype, stream, D, H, W, N, K, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, alpha, I, beta, O); }, [&](){ stream.synchronize();  }, device));
  print_results(times, {str(N), str(K), str(M), str(P), str(Q),  str(T), str(R), str(S)}, metric.cmp(), [&](double tsec){ return metric.pool(M, P, Q, K, N, T, R, S, tsec);});
}


/* ------------------------------- */

void loop_nest(std::vector<size_t> const & ranges, std::function<void(std::vector<size_t> const &)> const & f){
  size_t D = ranges.size();
  std::vector<size_t> values(D, 0);
  // Start with innermost loop
  size_t i = D - 1;
  while(true){
    //Execute function
    f(values);
    //Increment counters
    while(values[i]++ == ranges[i] - 1){
      if(i == 0)
        return;
      values[i--] = 0;
    }
    i = D - 1;
  }
}

template<class T>
void loop_nest(std::vector<std::vector<T>> const & iterates, std::function<void(std::vector<T>)> const & f){
  //Ranges to iterate over
  std::vector<size_t> ranges;
  for(auto const & x: iterates)
    ranges.push_back(x.size());
  //Proxy function
  auto proxy = [&](std::vector<size_t> const & idx){
    std::vector<T> x(iterates.size());
    for(size_t i = 0; i < x.size(); ++i)
    x[i] = iterates[i][idx[i]];
  f(x);
  };
  //Iterate
  loop_nest(ranges, proxy);
}


void search_conv(int32_t W, int32_t H, int32_t C, int32_t N, int32_t K, int32_t R, int32_t S, int32_t pad_h, int32_t pad_w, int32_t stride_h, int32_t stride_w, sc::DType dtype){
  auto ctx = drv::backend::contexts::get_default();
  size_t dtsize = sc::size_of(dtype);

  size_t D = 1, T = 1, stride_d = 1, pad_d = 0;
  size_t P = (H - R + 1 + 2*pad_h + stride_h - 1)/stride_h;
  size_t Q = (W - S + 1 + 2*pad_w + stride_w - 1)/stride_w;
  size_t M = (D - T + 1 + 2*pad_d + stride_d - 1)/stride_d;

  //Setup
  drv::Buffer O(ctx, K*P*Q*M*N*dtsize);
  drv::Buffer I(ctx, C*H*W*D*N*dtsize);
  drv::Buffer F(ctx, C*R*S*T*K*dtsize);
  drv::Stream stream(ctx);
  sc::scalar alpha(1., dtype),  beta(1., dtype);

  //Exhaustive search
  std::vector<sc::param_t> r1 = {1};
  std::vector<sc::param_t> rv = {4};
  std::vector<sc::param_t> rr = {1, 2, 4, 8};
  std::vector<sc::param_t> rl = {4, 8, 16, 32};
  std::vector<sc::param_t> rs = {4, 8, 16};
  double best;
  loop_nest<sc::param_t>({rv, rl, rl, rs, rs, rl, rl, r1, rr, rr}, [&](std::vector<sc::param_t> const & x){
    sc::templates::Conv generator(dtype, C, D, H, W, N, K, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]);
    //Compile
    try{
      std::string src = generator.dump(ctx.device(), "conv");
      drv::Module program(ctx, src);
      drv::Kernel kernel(program, "conv");
      double tsec = bench([&](){ generator.enqueue(kernel, stream, alpha, I, F, beta, O); }, [&](){ stream.synchronize(); }, ctx.device());
      double tflops = sc::templates::Conv::tflops(P,Q,M,K,N,C,R,S,T,tsec);
      best = std::max(tflops, best);
      std::cout << "//";
      std::copy(x.begin(), x.end(), std::ostream_iterator<int>(std::cout, " "));
      std::cout << ": " << tflops << " TFLOPS [BEST: " << best << "]" << std::endl;
    }catch(isaac::templates::invalid_parameters const &){
      return;
    }catch(drv::exception::cuda::launch_out_of_resources const &){
      return;
    }
  });
  std::cout << "ISAAC: " << best << std::endl;
}

void search_gemm(int32_t M, int32_t N, int32_t K, sc::IsaacOperation_t AT, sc::IsaacOperation_t BT, sc::DType dtype){
  auto ctx = drv::backend::contexts::get_default();
  size_t dtsize = sc::size_of(dtype);

  // Setup
  size_t ldc = M;
  size_t lda = (AT==sc::ISAAC_OP_N)?M:K;
  size_t ldb = (BT==sc::ISAAC_OP_N)?K:N;
  int32_t offc = 0, offa = 0, offb = 0;
  drv::Buffer C(ctx, M*N*dtsize);
  drv::Buffer A(ctx, M*K*dtsize);
  drv::Buffer B(ctx, K*N*dtsize);
  drv::Stream stream(ctx);
  sc::scalar alpha(1., dtype), beta(0., dtype);

  // Exhaustive search
  std::vector<sc::param_t> r1 = {1};
  std::vector<sc::param_t> rv = {4};
  std::vector<sc::param_t> rr = {1, 2, 4, 8};
  std::vector<sc::param_t> rl = {4, 8, 16, 32};
  std::vector<sc::param_t> rs = {4, 8, 16};
  double best = 0;

  loop_nest<sc::param_t>({rv, rl, rl, rl, rs, r1, rs, rl, rl, rl, rl, r1, rr, rr}, [&](std::vector<sc::param_t> const & x){
    isaac::templates::GEMM generator(dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]);
    // Compile
    try{
      std::string src = generator.dump(ctx.device(), "gemm");
      drv::Module program(ctx, src);
      drv::Kernel kernel(program, "gemm");
      double time = bench([&](){ generator.enqueue(kernel, stream, alpha, A, B, beta, C); }, [&](){ stream.synchronize(); }, ctx.device());
      double tflops = 2*1e-3*M*N*K/time;
      best = std::max(tflops, best);
      std::cout << "//";
      std::copy(x.begin(), x.end(), std::ostream_iterator<int>(std::cout, " "));
      std::cout << ": " << tflops << " TFLOPS [BEST: " << best << "]" << std::endl;
    }catch(isaac::templates::invalid_parameters const &){
      return;
    }catch(drv::exception::cuda::launch_out_of_resources const &){
      return;
    }
  });
  std::cout << "ISAAC: " << best << std::endl;
}

/* Helpers for dumping source code */
void dump_source(sc::driver::Device const & device, sc::templates::Generator& generator, opts::Options* options, std::string const & name){
  if(options->get<std::string>("format") == "ptx")
    std::cout << generator.dump(device, name) << std::endl;
  else{
    auto x = generator.tuning_params();
    std::cout << "Tuning parameters: " << std::flush;
    for(size_t i = 0; i < x.size(); ++i)
      std::cout << ((i>0)?", ":"") << x[i] << std::flush;
    std::cout << std::endl;
  }
}

/* Application code */
int main(int argc, char* argv[]){
  opts::Application program("isaac-tools", "Command-line interface for ISAAC");
  // Options
  opts::Options* options = program.options();
  options->add<size_t>("device", "Device to run on", 0);
  options->add<sc::DType>("dtype", "Data-type to use for computations", "float32", {{"float32", sc::FLOAT_TYPE}, {"float64", sc::DOUBLE_TYPE}});
  options->add<std::string>("name", "Name to give to the generated kernel", "kernel");
  options->add_group("search", "Exhaustively search for best tuning parameters");
  opts::Options* dump = options->add_group("dump", "Dump source-code generated by ISAAC");
  dump->add("format", "Format to generate", "ptx", {"ptx", "params"});
  dump->add("target", "Target GPU (sm_xx)", {"sm_50", "sm_52", "sm_60", "sm_61", "sm_70"});
  opts::Options* bench = options->add_group("bench", "Benchmark source code generated by ISAAC");
  bench->add("suite", "Benchmarking suite to run", "custom", {"custom", "deepbench"});
  bench->add<std::shared_ptr<Metric>>("metric", "performance metric for the results", "tflops", {{"tflops", std::make_shared<FLOPS>(1e12)}, {"ms", std::make_shared<Time>(1e-3)},  {"us", std::make_shared<Time>(1e-6)}});
  // Constraints
  options->add_constraint(opts::OneOf({"bench", "dump", "search"}));
  options->add_constraint(opts::OneOf({"gemm", "conv", "pool"}));
  // GEMM
  opts::Options* gemm = options->add_group("gemm", "Use matrix-multiplication");
  gemm->add("layout", "Transposition layout for A and B", "NT", {"NN", "NT", "TN", "TT"});
  gemm->add<std::vector<size_t>>("shape", "Matrix shapes (M,N,K)", {2048, 2048, 2048}, opts::SizeConstraint(3));
  gemm->add<std::vector<size_t>>("kernel", "Bypass predictive model to use given tuning parameters", opts::SizeConstraint(14));
  // CONV
  opts::Options* conv = options->add_group("conv", "Use convolutions");
  conv->add<std::vector<size_t>>("shape", "Tensor shapes (D, H, W, C, N, K, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w)", {1, 70, 14, 512, 128, 64, 1, 7, 7, 0, 0, 0, 1, 1, 1}, opts::SizeConstraint(15));
  conv->add<std::vector<size_t>>("kernel", "Bypass predictive model to use given tuning parameters", opts::SizeConstraint(9));
  // POOL
  opts::Options* pool = options->add_group("pool", "Use pooling");
  pool->add<std::vector<size_t>>("shape", "Tensor shapes (D, H, W, N, K, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w)", {1, 70, 14, 128, 64, 1, 7, 7, 0, 0, 0, 1, 1, 1}, opts::SizeConstraint(14));
  pool->add<std::vector<size_t>>("kernel", "Bypass predictive model to use given tuning parameters", opts::SizeConstraint(4));
  program.parse(argc, argv);

  if(options->has("bench"))
    std::cout << std::fixed << std::setprecision(2);
  //Device
  sc::driver::Device device = sc::driver::backend::devices()[options->get<size_t>("device")];
  if(options->has("dump") && dump->has("target")){
    std::string target = dump->get<std::string>("target");
    char major = target[3];
    char minor = target[4];
    device.interpret_as(std::make_pair((size_t)std::atoi(&major), (size_t)std::atoi(&minor)));
  }
  static sc::driver::Context context(device);
  sc::driver::Stream stream(context);
  // Data-Type
  sc::DType dtype = options->get<sc::DType>("dtype");
  // Kernel name
  std::string name = options->get<std::string>("name");

  /* Get optimized kernel generator */
  std::unique_ptr<sc::templates::Generator> generator;

  // GEMM
  if(options->has("gemm")){
    std::string layout = gemm->get<std::string>("layout");
    sc::IsaacOperation_t AT = layout[0]=='T'?sc::ISAAC_OP_T:sc::ISAAC_OP_N;
    sc::IsaacOperation_t BT = layout[1]=='T'?sc::ISAAC_OP_T:sc::ISAAC_OP_N;
    auto shape = gemm->get<std::vector<size_t>>("shape");
    size_t M = shape[0], N = shape[1], K = shape[2];
    //Get Source
    size_t ldc = M;
    size_t lda = (AT==sc::ISAAC_OP_N)?M:K;
    size_t ldb = (BT==sc::ISAAC_OP_N)?K:N;
    if(options->has("search")){
      search_gemm(M, N, K, AT, BT, dtype);
    }
    if(gemm->has("kernel")){
      auto x = gemm->get<std::vector<size_t>>("kernel");
      generator.reset(new sc::templates::GEMM(dtype, AT, BT, M, N, K, 0, lda, 0, ldb, 0, ldc, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]));
    }
    else{
      sc::runtime::GEMMProfile* profile = (sc::runtime::GEMMProfile*)sc::runtime::database.at({device.architecture(), sc::runtime::GEMM}).get();
      generator.reset(new sc::templates::GEMM(profile->predict(stream, dtype, AT, BT, M, N, K, 0, lda, 0, ldb, 0, ldc)));
    }
    if(options->has("dump"))
      dump_source(device, *generator, dump, name);
    if(options->has("bench")){
      auto metric = bench->get<std::shared_ptr<Metric>>("metric");
      print_results_header({"AT", "BT", "M", "N", "K", "ISAAC", "cuBLAS"});
      std::vector<gemm_params_t> shapes;
      //User provided shapes
      if(bench->get<std::string>("suite")=="custom")
        shapes = {std::make_tuple(dtype, AT, BT, M, N, K)};

      //SC17 paper shapes
      if(bench->get<std::string>("suite")=="deepbench")
        shapes = TestBench::gemm(dtype);

      //Print results
      for(auto x: shapes){
        std::tie(dtype, AT, BT, M, N, K) = x;
        benchmark_gemm(*metric, context, device, stream, dtype, AT, BT, M, N, K, gemm->has("kernel")?generator.get():NULL);
      }
    }
  }

  // CONV
  if(options->has("conv")){
    auto x = conv->get<std::vector<size_t>>("shape");
    param_t D = x[0], H = x[1], W = x[2], C = x[3], N = x[4], K = x[5], T = x[6], R = x[7], S = x[8], pad_d = x[9], pad_h = x[10], pad_w = x[11], stride_d = x[12], stride_h = x[13], stride_w = x[14];
    param_t M, P, Q;
    sc::templates::Conv::output_shapes(D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, M, P, Q);

    if(options->has("search"))
      search_conv(W, H, C, N, K, R, S, pad_h, pad_w, stride_h, stride_w, dtype);
    if(conv->has("kernel")){
      auto x = conv->get<std::vector<size_t>>("kernel");
      generator.reset(new sc::templates::Conv(dtype, C, D, H, W, N, K, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]));
    }
    else{
      sc::runtime::ConvProfile* profile = (sc::runtime::ConvProfile*)sc::runtime::database.at({device.architecture(), sc::runtime::CONV}).get();
      generator.reset(new sc::templates::Conv(profile->predict(stream, dtype, C, D, H, W, N, K, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w)));
    }
    if(options->has("dump"))
      dump_source(device, *generator, dump, name);
    if(options->has("bench")){
      auto metric = bench->get<std::shared_ptr<Metric>>("metric");
      print_results_header({"N", "K", "M", "P", "Q", "C", "T", "R", "S", "ISAAC", "cuDNN"});
      std::vector<conv_params_t> shapes;
      //User provided shapes
      if(bench->get<std::string>("suite")=="custom")
        shapes = {std::make_tuple(dtype, D, W, H, C, N, K, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w)};
      //SuperComputing17 shapes
      if(bench->get<std::string>("suite")=="deepbench")
        shapes = TestBench::conv(dtype);
      //Print results
      for(auto x: shapes){
        std::tie(dtype, D, W, H, C, N, K, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w) = x;
        benchmark_conv(*metric, context, device, stream, dtype, D, H, W, C, N, K, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, conv->has("kernel")?generator.get():NULL);
      }
    }
  }

  // POOL
  if(options->has("pool")){
    auto x = pool->get<std::vector<size_t>>("shape");
    param_t D = x[0], W = x[1], H = x[2], N = x[3], K = x[4], T = x[5], R = x[6], S = x[7], pad_d = x[8], pad_h = x[9], pad_w = x[10], stride_d = x[11], stride_h = x[12], stride_w = x[13];
    param_t M, P, Q;
    sc::templates::Conv::output_shapes(D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, M, P, Q);

    if(pool->has("kernel")){
      auto x = pool->get<std::vector<size_t>>("kernel");
      generator.reset(new sc::templates::Pool(dtype, K, D, H, W, N, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, x[0], x[1], x[2], x[3]));
    }
    else{
      generator.reset(new sc::templates::Pool(dtype, K, D, H, W, N, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w));
    }
    if(options->has("dump"))
      dump_source(device, *generator, dump, name);
    if(options->has("bench")){
      auto metric = bench->get<std::shared_ptr<Metric>>("metric");
      print_results_header({"N", "K", "M", "P", "Q", "T", "R", "S", "ISAAC", "cuDNN"});
      std::vector<pool_params_t> shapes;
      //User provided shapes
      if(bench->get<std::string>("suite")=="custom")
        shapes = {std::make_tuple(dtype, D, W, H, N, K, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w)};
      //SuperComputing17 shapes
      if(bench->get<std::string>("suite")=="deepbench")
        shapes = TestBench::pool(dtype);
      //Print results
      for(auto x: shapes){
        std::tie(dtype, D, W, H, N, K, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w) = x;
        benchmark_pool(*metric, context, device, stream, dtype, D, H, W, N, K, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, pool->has("kernel")?generator.get():NULL);
      }
    }
  }
}
