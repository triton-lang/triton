#include <cstring>
#include <sstream>
#include <cstdio>
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/tools/bench.hpp"
#include "triton/external/half.hpp"
#include "triton/runtime/function.h"
#include "src/copy.h"
#include "util.h"
#include "cuda/cublas.h"


std::vector<double> do_bench(drv::stream* stream, int32_t M, int32_t N, order_t order){
  typedef float NumericT;
  std::string ty = "float";
  size_t dt_nbytes = sizeof(NumericT);
  drv::context* context = stream->context();
  int32_t ld = order == ROWMAJOR ? N : M;
  // create inputs
  auto dx = std::unique_ptr<drv::buffer>(drv::buffer::create(context, M*N*dt_nbytes));
  auto dy = std::unique_ptr<drv::buffer>(drv::buffer::create(context, M*N*dt_nbytes));
  // create options
  rt::function::options_space_t opt;
  opt.defines.push_back({"TYPE", {ty}});
  opt.defines.push_back({"ORDER", {order==ROWMAJOR?"ROWMAJOR":"COLMAJOR"}});
  opt.defines.push_back({"TM", {"32"}});
  opt.defines.push_back({"TN", {"32"}});
  opt.num_warps = {4};
  // create function
  rt::function function(src::copy2d, opt);
  // benchmark available libraries
  std::vector<double> result;
  auto gbps = [&](double ns) { return 2*M*N*dt_nbytes / (ns * 1e-9) * 1e-9; };
  // triton
  double triton_ns = triton::tools::bench([&]() { function({&*dx, &*dy, M, N, ld, ld}, grid2d(M, N), stream);}, stream);
  result.push_back(gbps(triton_ns));
  // done
  return result;
}

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context);
  // shapes to benchmark
  typedef std::tuple<int, int, order_t> config_t;
  std::vector<config_t> configs;
  for(auto x: std::vector<order_t>{COLMAJOR}){
    std::vector<config_t> tmp = {
      config_t{4096, 4096, x}
    };
    configs.insert(configs.end(), tmp.begin(), tmp.end());
  }
  // does the work
  int32_t M, N;
  order_t ord;
  for(const auto& c: configs){
    std::tie(M, N, ord) = c;
    std::cout << "// " << M << ", " << N << ", " << ord << std::flush;
    for(auto perf: do_bench(stream, M, N, ord))
      std::cout << ", " << perf << std::flush;
    std::cout << std::endl;
  }
}
