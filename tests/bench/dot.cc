#include <cstring>
#include <sstream>
#include <cstdio>
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/tools/bench.hpp"
#include "triton/external/half.hpp"
#include "triton/runtime/function.h"
#include "src/dot.h"
#include "cuda/cublas.h"


struct perf_t {
  double triton;
  double cublas;
};

namespace drv = triton::driver;
namespace rt = triton::runtime;

inline size_t ceil(size_t x, size_t y) {
  return (x + y - 1) / y;
};


std::vector<double> do_bench(drv::stream* stream, bool AT, bool BT, int32_t M, int32_t N, int32_t K){
  typedef half_float::half NumericT;
  std::string ty = "half";
  size_t dt_nbytes = sizeof(NumericT);
  drv::context* context = stream->context();
  // leading dimensions
  int32_t lda = AT ? K : M;
  int32_t ldb = BT ? N : K;
  int32_t ldc = M;
  // create inputs
  auto dc = std::unique_ptr<drv::buffer>(drv::buffer::create(context, M*N*dt_nbytes));
  auto da = std::unique_ptr<drv::buffer>(drv::buffer::create(context, M*K*dt_nbytes));
  auto db = std::unique_ptr<drv::buffer>(drv::buffer::create(context, K*N*dt_nbytes));
  // create options
  rt::function::options_space_t opt;
  opt.defines.push_back({"TYPE", {ty}});
  if(AT)
    opt.defines.push_back({"AT", {""}});
  if(BT)
    opt.defines.push_back({"BT", {""}});
  opt.defines.push_back({"TM", {"16", "32", "64", "128"}});
  opt.defines.push_back({"TN", {"16", "32", "64", "128"}});
  opt.defines.push_back({"TK", {"32"}});
  opt.num_warps = {1, 2, 4, 8};
  // create grid
  auto grid = [&](const rt::function::options_t& x) {
                    return rt::grid_t{ceil(M, x.D<int>("TM")),
                                      ceil(N, x.D<int>("TN"))};
                  };
  // create function
  rt::function function(src::dot, opt);
  // benchmark available libraries
  std::vector<double> result;
  auto tflops = [&](double nanosec) { return 2.*M*N*K / nanosec * 1e-3; };
  // cublas
  if(cublas::cublasinit()){
    NumericT alpha(static_cast<double>(1));
    NumericT beta(static_cast<double>(0));
    cublasGemmAlgo_t fastest;
    cublasGemm(CUDA_R_16F, stream, AT, BT, M, N, K, &alpha, &*da, lda, &*db, ldb, &beta, &*dc, ldc, &fastest);
    double cublas_ms = triton::tools::bench([&]() { cublasGemm(CUDA_R_16F, stream, AT, BT, M, N, K,
                                                    &alpha, &*da, lda, &*db, ldb, &beta, &*dc, ldc, nullptr, fastest); }, stream);
    result.push_back(tflops(cublas_ms));
  }
  // triton
  double triton_ms = triton::tools::bench([&]() { function({&*da, &*db, &*dc, M, N, K, lda, ldb, ldc}, grid, stream);}, stream);
  result.push_back(tflops(triton_ms));
  // done
  return result;
}

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context);
  // shapes to benchmark
  typedef std::tuple<bool, bool, int, int, int> config_t;
  std::vector<config_t> configs = {
    config_t{false, true, 512, 512, 512},
    config_t{false, true, 2048, 2048, 2048},
    config_t{false, true, 8192, 8192, 8192}
  };
  // does the work
  bool AT, BT;
  int32_t M, N, K;
  for(const auto& c: configs){
    std::tie(AT, BT, M, N, K) = c;
    std::cout << "// " << AT << " " << BT << " " << M << " " << N << " " << K << std::flush;
    for(auto perf: do_bench(stream, AT, BT, M, N, K))
      std::cout << ", " << perf << std::flush;
    std::cout << std::endl;
  }
}
