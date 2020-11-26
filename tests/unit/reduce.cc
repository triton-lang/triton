#include <iomanip>
#include <cstring>
#include <sstream>
#include <cstdio>
#include <functional>
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/tools/bench.hpp"
#include "triton/external/half.hpp"
#include "triton/runtime/function.h"
#include "cuda/cublas.h"
#include "reduce.h"
#include "util.h"


int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context->backend());
  // shapes to benchmark
  typedef std::tuple<std::vector<int>, int, reduce_op_t> config_t;
  std::vector<config_t> configs = {
    config_t{{8, 8, 4}, 2, ADD},
    config_t{{32}, 0, ADD},
    config_t{{32, 32}, 0, MAX},
    config_t{{32, 32}, 1, ADD},
    config_t{{32, 64}, 0, ADD},
    config_t{{64, 32}, 1, ADD}
  };
  // does the work
  int axis;
  std::vector<int> shape;
  reduce_op_t op;
  for(const auto& c: configs){
    std::tie(shape, axis, op) = c;
    std::cout << "Testing " << c << " ... " << std::flush;
    if(do_test(context, stream, shape, axis, op, 1))
      std::cout << " Pass! " << std::endl;
    else
      std::cout << " Fail! " << std::endl;
  }
}
