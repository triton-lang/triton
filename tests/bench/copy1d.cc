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


std::vector<double> do_bench(drv::stream* stream, int32_t N){
  typedef float NumericT;
  std::string ty = "float";
  size_t dt_nbytes = sizeof(NumericT);
  drv::context* context = stream->context();
  // create inputs
  auto dx = std::unique_ptr<drv::buffer>(drv::buffer::create(context, N*dt_nbytes));
  auto dy = std::unique_ptr<drv::buffer>(drv::buffer::create(context, N*dt_nbytes));
  // create options
  rt::function::options_space_t opt;
  opt.defines.push_back({"TYPE", {ty}});
  opt.defines.push_back({"TN", {"128"}});
  opt.num_warps = {1, 2, 4, 8};
  // create function
  rt::function function(src::copy1d, opt);
  // benchmark available libraries
  std::vector<double> result;
  auto gbps = [&](double ns) { return 2*N*dt_nbytes / (ns * 1e-9) * 1e-9; };
  // triton
  double triton_ns = triton::tools::bench([&]() { function({&*dx, &*dy, N}, grid1d(N), stream);}, stream);
  result.push_back(gbps(triton_ns));
  // done
  return result;
}

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context);
  // shapes to benchmark
  typedef std::tuple<int> config_t;
  std::vector<config_t> configs = { 1024*1024*32 };
  int N;
  for(const auto& c: configs){
    std::tie(N) = c;
    std::cout << "// " << c << std::flush;
    for(auto perf: do_bench(stream, N))
      std::cout << ", " << perf << std::flush;
    std::cout << std::endl;
  }
}
