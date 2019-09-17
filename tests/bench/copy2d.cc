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


std::vector<double> do_bench(drv::stream* stream, int32_t M, int32_t N, order_t order_x, order_t order_y){
  typedef float NumericT;
  std::string ty = "float";
  size_t dt_nbytes = sizeof(NumericT);
  drv::context* context = stream->context();
  // create inputs
  auto dx = std::unique_ptr<drv::buffer>(drv::buffer::create(context, M*N*dt_nbytes));
  auto dy = std::unique_ptr<drv::buffer>(drv::buffer::create(context, M*N*dt_nbytes));
  // create options
  rt::function::options_space_t opt;
  opt.defines.push_back({"TYPE", {ty}});
  opt.defines.push_back({"STRIDE_XM", {(order_x == ROWMAJOR)?"M":"1"}});
  opt.defines.push_back({"STRIDE_XN", {(order_x == ROWMAJOR)?"1":"N"}});
  opt.defines.push_back({"STRIDE_YM", {(order_y == ROWMAJOR)?"M":"1"}});
  opt.defines.push_back({"STRIDE_YN", {(order_y == ROWMAJOR)?"1":"N"}});
  opt.defines.push_back({"TM", {"32"}});
  opt.defines.push_back({"TN", {"32"}});
  opt.num_warps = {4};
  // create function
  rt::function function(src::copy2d, opt);
  // benchmark available libraries
  std::vector<double> result;
  auto gbps = [&](double ns) { return 2*M*N*dt_nbytes / (ns * 1e-9) * 1e-9; };
  // triton
  double triton_ns = triton::tools::bench([&]() { function({&*dx, &*dy, M, N}, grid2d(M, N), stream);}, stream);
  result.push_back(gbps(triton_ns));
  // done
  return result;
}

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context);
  // shapes to benchmark
  typedef std::tuple<int, int, order_t, order_t> config_t;
  std::vector<config_t> configs = {
    {4096, 4096, ROWMAJOR, ROWMAJOR},
    {4096, 4096, COLMAJOR, ROWMAJOR},
    {4096, 4096, ROWMAJOR, COLMAJOR},
    {4096, 4096, COLMAJOR, COLMAJOR},
  };
  // does the work
  int32_t M, N;
  order_t ord_x, ord_y;
  for(const auto& c: configs){
    std::tie(M, N, ord_x, ord_y) = c;
    std::cout << "// " << M << ", " << N << ", " << ord_x << ", " << ord_y << std::flush;
    for(auto perf: do_bench(stream, M, N, ord_x, ord_y))
      std::cout << ", " << perf << std::flush;
    std::cout << std::endl;
  }
}
