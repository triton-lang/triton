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


namespace drv = triton::driver;
namespace rt = triton::runtime;

inline size_t ceil(size_t x, size_t y) {
  return (x + y - 1) / y;
};

inline rt::function::grid_fn_ty grid2d(size_t M, size_t N) {
  return [M, N](const rt::function::options_t& x) {
    return rt::grid_t{ceil(M, x.D<int>("TM")),
                      ceil(N, x.D<int>("TN"))};
  };
}



std::vector<double> do_bench(drv::stream* stream, bool AT, bool BT, int32_t M, int32_t N, int32_t K){
  typedef float NumericT;
  std::string ty = "float";
  cublasDataType_t cuty = CUDA_R_32F;
  size_t dt_nbytes = sizeof(NumericT);
  drv::context* context = stream->context();
  // leading dimensions
  int32_t lda = AT ? K : M;
  int32_t ldb = BT ? N : K;
  int32_t ldc = M;
  // create inputs
  auto da = std::unique_ptr<drv::buffer>(drv::buffer::create(context, M*K*dt_nbytes));
  auto db = std::unique_ptr<drv::buffer>(drv::buffer::create(context, K*N*dt_nbytes));
  auto dc = std::unique_ptr<drv::buffer>(drv::buffer::create(context, M*N*dt_nbytes));
  // create options
  rt::function::options_space_t opt;
  opt.defines.push_back({"TYPE", {ty}});
  opt.defines.push_back({"AT", {AT?"1":"0"}});
  opt.defines.push_back({"BT", {BT?"1":"0"}});
  opt.defines.push_back({"TM", {"128"}});
  opt.defines.push_back({"TN", {"128"}});
  opt.defines.push_back({"TK", {"8"}});
  opt.num_warps = {8};
  // create function
  rt::function function(src::dot, opt);
  // benchmark available libraries
  std::vector<double> result;
  auto tflops = [&](double nanosec) { return 2.*M*N*K / nanosec * 1e-3; };
//  // cublas
//  if(cublas::cublasinit()){
//    NumericT alpha(static_cast<double>(1));
//    NumericT beta(static_cast<double>(0));
//    cublasGemmAlgo_t fastest;
//    cublasGemm(cuty, stream, AT, BT, M, N, K, &alpha, &*da, lda, &*db, ldb, &beta, &*dc, ldc, &fastest);
//    double cublas_ms = triton::tools::bench([&]() { cublasGemm(cuty, stream, AT, BT, M, N, K,
//                                                               &alpha, &*da, lda, &*db, ldb, &beta, &*dc,
//                                                               ldc, nullptr, fastest); }, stream);
//    result.push_back(tflops(cublas_ms));
//  }
  // triton
  double triton_ms = triton::tools::bench([&]() { function({&*da, &*db, &*dc, M, N, K, lda, ldb, ldc}, grid2d(M, N), stream);}, stream);
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
  std::vector<config_t> configs;
  for(auto x: std::vector<std::array<bool, 2>>{{false, true}}){
    std::vector<config_t> tmp = {
      config_t{x[0], x[1], 2048, 2048, 2048}
//      config_t{x[0], x[1], 16, 2048, 2048},
//      config_t{x[0], x[1], 32, 2048, 2048},
//      config_t{x[0], x[1], 64, 2048, 2048},
//      config_t{x[0], x[1], 128, 2048, 2048},
//      config_t{x[0], x[1], 7000, 2048, 2048},
//      config_t{x[0], x[1], 16, 4096, 4096},
//      config_t{x[0], x[1], 32, 4096, 4096},
//      config_t{x[0], x[1], 64, 4096, 4096},
//      config_t{x[0], x[1], 128, 4096, 4096},
//      config_t{x[0], x[1], 7000, 4096, 4096},
    };
    configs.insert(configs.end(), tmp.begin(), tmp.end());
  }
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
