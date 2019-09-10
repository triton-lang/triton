#include <iomanip>
#include <cstring>
#include <sstream>
#include <cstdio>
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/tools/bench.hpp"
#include "triton/external/half.hpp"
#include "triton/runtime/function.h"
#include "src/reduce.h"
#include "cuda/cublas.h"
#include "util.h"

namespace drv = triton::driver;
namespace rt = triton::runtime;


bool do_test(drv::stream* stream, int M, int N, std::string op, int nwarp){
  typedef float NumericT;
  std::string ty = "float";
  size_t dt_nbytes = sizeof(NumericT);
  drv::context* context = stream->context();
  std::vector<NumericT> hy(M);
  std::vector<NumericT> hx(M*N);
  srand(0);
  init_rand(hy);
  init_rand(hx);
  auto dy = std::shared_ptr<drv::buffer>(drv::buffer::create(context, hy.size()*dt_nbytes));
  auto dx = std::shared_ptr<drv::buffer>(drv::buffer::create(context, hx.size()*dt_nbytes));
  stream->write(&*dy, true, 0, hy);
  stream->write(&*dx, true, 0, hx);
  rt::function::options_space_t opt;
  opt.defines.push_back({"TYPE", {ty}});
  opt.defines.push_back({"TM", {std::to_string(M)}});
  opt.defines.push_back({"TN", {std::to_string(N)}});
  opt.num_warps = {nwarp};
  rt::function function(src::reduce2d, opt);
  function({&*dy, &*dx, M, N, M}, grid2d(M, N), stream);
  stream->synchronize();
}

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context);
  // shapes to benchmark
  typedef std::tuple<int, int, std::string> config_t;
  std::vector<config_t> configs = {
    config_t{32, 32, "+"}
  };
  // does the work
  int M, N;
  std::string op;
  for(const auto& c: configs){
    std::tie(M, N, op) = c;
    std::cout << "Testing " << c << " ... " << std::flush;
    if(do_test(stream, M, N, op, 1))
      std::cout << " Pass! " << std::endl;
    else
      std::cout << " Fail! " << std::endl;
  }
}
